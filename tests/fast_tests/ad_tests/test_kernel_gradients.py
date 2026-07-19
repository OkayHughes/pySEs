"""Kernel-level gradient probes for the AD-hardening audit (strategy §5, layer 1).

Each probe seeds a hazard-bearing kernel from the audit table
(docs/ad_hardening_strategy.md §2.2) with deterministic synthetic states —
healthy interiors plus adversarial states at the documented hazard sites —
and checks JVP/VJP finiteness, the forward/reverse dot-product identity,
and agreement with central finite differences.

xfail(strict=True) entries are the executable form of the hazard
inventory: they pin today's failure and flip loudly when a later
increment fixes (or accidentally changes) the behavior.

Covered here: Zerroukat remap (Tier 1), tracer limiter (Tier 2),
min-reduction bound extraction (Tier 2), NH thermodynamics (Tier 3), DIRK
Jacobian + tridiagonal solve (Tier 4), deep-atmosphere phi_to_z (Tier 5),
and a spectral-operator + DSS control group. Heavier fixture-based
subsystems (hyperviscosity, full DIRK Newton, full step) arrive with
harness layers 2-3.
"""
import functools

import numpy as np
import pytest

from pyses._config import get_backend as _get_backend
from pyses.dynamical_cores.vertical_remap import zerroukat_remap
from pyses.dynamical_cores.homme.implicit_terms import (
    calc_dirk_jacobian, solve_strict_diag_dominant_tridiag)
from pyses.dynamical_cores.homme.thermodynamics import (
    eval_pressure_exner_nonhydrostatic)
from pyses.dynamical_cores.model_info import models
from pyses.dynamical_cores.utils_3d import phi_to_z
from pyses.mesh_generation.periodic_plane import init_uniform_grid
from pyses.operations_2d.limiters import clip_and_sum_limiter
from pyses.operations_2d.local_assembly import project_scalar
from pyses.operations_2d.operators import horizontal_gradient

from ...context import to_host
from .probe_utils import probe_fd_directional, probe_forward_reverse

_be = _get_backend()
jnp = _be.np
device_wrapper = _be.array

pytestmark = pytest.mark.skipif(_be.wrapper_type not in ("jax", "torch"),
                                reason="backend has no AD support")

# Small column geometry shared by the vertical kernels.
NELEM, NPT, NLEV, NTRACER = 2, 2, 6, 2

PHYS = {"p0": 1.0e5, "Rgas": 287.04, "cp": 1004.64,
        "gravity": 9.80616, "radius_earth": 6.371e6}


# ---------------------------------------------------------------------------
# Tier 1 — Zerroukat vertical remap (search + gather + PPM filter + divides)
# ---------------------------------------------------------------------------

def _remap_inputs(seed=7):
  rng = np.random.default_rng(seed)
  d_model = 40.0 + 60.0 * rng.random((NELEM, NPT, NPT, NLEV))
  d_ref = 40.0 + 60.0 * rng.random((NELEM, NPT, NPT, NLEV))
  # Same column mass so the remap is a pure redistribution.
  d_ref *= d_model.sum(-1, keepdims=True) / d_ref.sum(-1, keepdims=True)
  # Non-monotone mixing ratios so the monotonicity filter actually fires.
  q = 1.0 + rng.random((NELEM, NPT, NPT, NLEV, NTRACER))
  tracer_mass = q * d_model[..., None]
  return (device_wrapper(tracer_mass), device_wrapper(d_model),
          device_wrapper(d_ref))


def _remap_fn(tracer_mass, d_model, d_ref):
  # filter=True is the hot-path configuration (remap_dynamics/remap_tracers).
  return zerroukat_remap(tracer_mass, d_model, d_ref,
                         num_lev=NLEV, filter=True)


def test_zerroukat_remap_forward_reverse():
  probe_forward_reverse(_remap_fn, _remap_inputs(), what="zerroukat_remap")


def test_zerroukat_remap_fd():
  probe_fd_directional(_remap_fn, _remap_inputs(), what="zerroukat_remap")


def test_zerroukat_remap_thin_layer_finite():
  # Near-degenerate model layer: the unguarded 1/zhdp and zgam divides
  # (vertical_remap.py:78,85) amplify but stay finite until thickness
  # reaches ~1e-160; this pins how far the current kernel can be pushed.
  rng = np.random.default_rng(11)
  d_model = 40.0 + 60.0 * rng.random((NELEM, NPT, NPT, NLEV))
  d_model[..., 2] = 1e-12  # thin (but nonzero) interior layer
  d_ref = 40.0 + 60.0 * rng.random((NELEM, NPT, NPT, NLEV))
  d_ref *= d_model.sum(-1, keepdims=True) / d_ref.sum(-1, keepdims=True)
  q = 1.0 + rng.random((NELEM, NPT, NPT, NLEV, NTRACER))
  primals = (device_wrapper(q * d_model[..., None]), device_wrapper(d_model),
             device_wrapper(d_ref))
  probe_forward_reverse(_remap_fn, primals, what="zerroukat_remap thin-layer")


@pytest.mark.xfail(
    strict=True,
    reason="category C (strategy §3): frozen integer cell search, "
           "vertical_remap.py:70-76 — AD differentiates the frozen branch "
           "(measured +700) while FD across the switch sees -5252; "
           "remediation lands with increment 7")
def test_zerroukat_remap_interface_crossing():
  # Category C (strategy §3): the containing-cell search is an integer op
  # (vertical_remap.py:70-76). Place a reference interface 1e-9 Pa from a
  # model interface and take an FD step that crosses it: AD differentiates
  # the frozen branch while central FD straddles the switch.
  delta = 1e-9
  d_model = np.full((1, 1, 1, 4), 100.0)
  d_ref = np.array([100.0 + delta, 100.0 - delta, 100.0, 100.0])[None, None, None, :]
  # Strongly non-monotone column so the filtered reconstruction differs
  # sharply between the two candidate cells.
  q = np.array([1.0, 8.0, 1.0, 8.0])[None, None, None, :, None]
  primals = (device_wrapper(q * d_model[..., None]),
             device_wrapper(d_model), device_wrapper(d_ref))

  # Direction that moves only the first reference interface.
  direction = np.zeros_like(d_ref)
  direction[..., 0] = 1.0
  tangents = (device_wrapper(np.zeros(primals[0].shape)),
              device_wrapper(np.zeros(primals[1].shape)),
              device_wrapper(direction))

  def scalar_fn(*p):
    out = zerroukat_remap(*p, num_lev=4, filter=True)
    return jnp.sum(out * out)

  _, ad_dir = _be.jvp(scalar_fn, primals, tangents)
  h = 1e-6  # crosses the interface at distance delta = 1e-9
  hosts = [np.asarray(to_host(p)) for p in primals]
  tans = [np.asarray(to_host(t)) for t in tangents]
  f_plus = float(to_host(scalar_fn(*[device_wrapper(p + h * t)
                                     for p, t in zip(hosts, tans)])))
  f_minus = float(to_host(scalar_fn(*[device_wrapper(p - h * t)
                                      for p, t in zip(hosts, tans)])))
  fd_dir = (f_plus - f_minus) / (2.0 * h)
  np.testing.assert_allclose(float(to_host(ad_dir)), fd_dir, rtol=1e-3)


# ---------------------------------------------------------------------------
# Tier 2 — tracer limiter and min/max bound extraction
# ---------------------------------------------------------------------------

def _limiter_inputs(seed=13, saturate=False):
  rng = np.random.default_rng(seed)
  d_mass = 50.0 + 50.0 * rng.random((NELEM, NPT, NPT, NLEV))
  mass_matrix = 0.5 + rng.random((NELEM, NPT, NPT))
  if saturate:
    q = np.full((NELEM, NPT, NPT, NLEV), 0.9)
  else:
    q = rng.random((NELEM, NPT, NPT, NLEV))
  tracer_min = np.full((NELEM, NLEV), 0.2)
  tracer_max = np.full((NELEM, NLEV), 0.8)
  return tuple(device_wrapper(a) for a in
               (q * d_mass, mass_matrix, tracer_min, tracer_max, d_mass))


def test_limiter_forward_reverse():
  probe_forward_reverse(clip_and_sum_limiter, _limiter_inputs(),
                        what="clip_and_sum_limiter")


def test_limiter_fd():
  probe_fd_directional(clip_and_sum_limiter, _limiter_inputs(),
                       what="clip_and_sum_limiter")


def test_limiter_fully_saturated_finite():
  # Every DOF overshoots: the redistribution denominator is exactly zero.
  # Regression for the documented safe-divide fix (limiters.py:72-79) that
  # keeps the masked branch from poisoning the cotangent.
  probe_forward_reverse(clip_and_sum_limiter, _limiter_inputs(saturate=True),
                        what="clip_and_sum_limiter saturated")


def test_min_reduction_kink_at_tie():
  # Category B: element bound extraction (calc_minmax) is a min/max
  # reduction; at an exact tie AD returns a one-sided subgradient while
  # central FD returns the average of the one-sided slopes.
  x = device_wrapper(np.array([0.5, 0.5, 1.0]))
  grad = _be.grad(lambda x: jnp.min(x))(x)
  e0 = np.array([1.0, 0.0, 0.0])
  h = 1e-7
  fd = (float(to_host(jnp.min(device_wrapper(np.array([0.5 + h, 0.5, 1.0]))))) -
        float(to_host(jnp.min(device_wrapper(np.array([0.5 - h, 0.5, 1.0])))))) / (2 * h)
  ad = float(np.vdot(np.asarray(to_host(grad)), e0))
  np.testing.assert_allclose(ad, fd, rtol=1e-6)


# ---------------------------------------------------------------------------
# Tier 3 — non-hydrostatic thermodynamics (divides + fractional powers)
# ---------------------------------------------------------------------------

def _thermo_inputs(seed=17):
  rng = np.random.default_rng(seed)
  theta_v_d_mass = 300.0 * (50.0 + 50.0 * rng.random((NELEM, NPT, NPT, NLEV)))
  d_phi = -(100.0 + 100.0 * rng.random((NELEM, NPT, NPT, NLEV)))
  return device_wrapper(theta_v_d_mass), device_wrapper(d_phi)


def _thermo_fn(theta_v_d_mass, d_phi):
  return eval_pressure_exner_nonhydrostatic(theta_v_d_mass, d_phi, 1.0, PHYS)


def test_nh_pressure_forward_reverse():
  probe_forward_reverse(_thermo_fn, _thermo_inputs(),
                        what="eval_pressure_exner_nonhydrostatic")


def test_nh_pressure_fd():
  probe_fd_directional(_thermo_fn, _thermo_inputs(),
                       what="eval_pressure_exner_nonhydrostatic")


def test_nh_pressure_zero_thickness_finite():
  # Regression for the category-A guard (increment 4): a collapsed
  # geopotential layer used to hit an unguarded divide by d_phi in
  # eval_pressure_exner_nonhydrostatic, giving inf pressure / NaN exner in
  # the primal and both AD modes.  The safe-divide guard keeps degenerate
  # columns finite; the healthy-state probes above pin that physical
  # states are unchanged.
  theta_v_d_mass, d_phi = (np.array(to_host(a)) for a in _thermo_inputs())
  d_phi[..., 2] = 0.0
  probe_forward_reverse(
      _thermo_fn, (device_wrapper(theta_v_d_mass), device_wrapper(d_phi)),
      what="eval_pressure_exner_nonhydrostatic zero-thickness")


# ---------------------------------------------------------------------------
# Tier 4 — DIRK Jacobian and tridiagonal Thomas solve
# ---------------------------------------------------------------------------

def _tridiag_inputs(seed=19):
  rng = np.random.default_rng(seed)
  jacL = 0.2 * rng.standard_normal((NELEM, NPT, NPT, NLEV - 1))
  jacU = 0.2 * rng.standard_normal((NELEM, NPT, NPT, NLEV - 1))
  # Strict diagonal dominance, as the solver's contract requires.
  jacD = np.ones((NELEM, NPT, NPT, NLEV))
  jacD[..., :-1] += np.abs(jacU)
  jacD[..., 1:] += np.abs(jacL)
  rhs = rng.standard_normal((NELEM, NPT, NPT, NLEV))
  return tuple(device_wrapper(a) for a in (jacL, jacD, jacU, rhs))


def test_tridiag_solve_forward_reverse():
  probe_forward_reverse(solve_strict_diag_dominant_tridiag, _tridiag_inputs(),
                        what="solve_strict_diag_dominant_tridiag")


def test_tridiag_solve_fd():
  probe_fd_directional(solve_strict_diag_dominant_tridiag, _tridiag_inputs(),
                       what="solve_strict_diag_dominant_tridiag")


def _dirk_jacobian_inputs(seed=23):
  rng = np.random.default_rng(seed)
  d_mass = 50.0 + 50.0 * rng.random((NELEM, NPT, NPT, NLEV))
  d_phi = -(100.0 + 100.0 * rng.random((NELEM, NPT, NPT, NLEV)))
  pnh = 1.0e4 + 9.0e4 * rng.random((NELEM, NPT, NPT, NLEV))
  return tuple(device_wrapper(a) for a in (d_mass, d_phi, pnh))


def _dirk_jacobian_fn(d_mass, d_phi, pnh):
  return calc_dirk_jacobian(100.0, d_mass, d_phi, pnh, PHYS)


def test_dirk_jacobian_forward_reverse():
  probe_forward_reverse(_dirk_jacobian_fn, _dirk_jacobian_inputs(),
                        what="calc_dirk_jacobian")


def test_dirk_jacobian_fd():
  probe_fd_directional(_dirk_jacobian_fn, _dirk_jacobian_inputs(),
                       what="calc_dirk_jacobian")


# ---------------------------------------------------------------------------
# Tier 5 — deep-atmosphere geometry
# ---------------------------------------------------------------------------

def _phi_fn(phi):
  return phi_to_z(phi, PHYS, models.homme_nonhydrostatic_deep)


def test_phi_to_z_deep_forward_reverse():
  rng = np.random.default_rng(29)
  phi = device_wrapper(PHYS["gravity"] * 3.0e4 * rng.random((NELEM, NPT, NPT, NLEV)))
  probe_forward_reverse(_phi_fn, (phi,), what="phi_to_z deep")
  probe_fd_directional(_phi_fn, (phi,), what="phi_to_z deep")


# ---------------------------------------------------------------------------
# Control group — spectral operators + DSS projection (expected clean)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _plane_grid():
  return init_uniform_grid(3, 3, 3)


def test_gradient_projection_forward_reverse():
  grid, dims = _plane_grid()
  rng = np.random.default_rng(31)
  field_shape = np.shape(to_host(grid["physical_coords"][:, :, :, 0]))
  f = device_wrapper(rng.standard_normal(field_shape))

  def op_fn(f):
    grad = horizontal_gradient(f, grid)
    return project_scalar(grad[:, :, :, 0], grid, dims)

  probe_forward_reverse(op_fn, (f,), what="horizontal_gradient + DSS")
  probe_fd_directional(op_fn, (f,), what="horizontal_gradient + DSS")
