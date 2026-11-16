"""Straight-through surrogate derivative for the tracer limiters.

Kernel-level contract of ``clip_and_sum_limiter(..., smooth_tau=...)`` and
``full_limiter(..., smooth_tau=...)`` (docs/ad_hardening_strategy.md,
increment 8): the value is the exact limiter's, the derivative is that of a
smooth-mask surrogate.

1. Primal unchanged: ``smooth_tau=None`` is bitwise the exact limiter; with
   ``smooth_tau`` set the output agrees to ~1e-15 relative (one rounding).
2. Tangent mass conservation: per element-level, the mass-matrix-weighted
   sum of J v equals that of the tracer-mass tangent, flag on and off.
3. Off the switches (every cell >= 100 tau from both bounds, |A| >= 100
   tau_A), surrogate jvp and vjp equal the exact ones. (At 10 tau the
   logistic tails are still ~5e-5, too large for a 1e-12 comparison.)
5. The forward/reverse dot-product identity holds to roundoff, flag on.

Bounds (spec section 6): ``soft_reduce_*`` and ``calc_minmax(...,
smooth_tau=...)`` keep exact values and split the derivative over ties with
weights a 1-ulp change does not reshuffle (the exact split jumps).

The step-level effect (1-ulp stability of J v, the scanned-step identity)
is measured in test_step_gradients.py.
"""
import functools

import numpy as np
import pytest

from pyses._config import get_backend as _get_backend
from pyses.mesh_generation.periodic_plane import init_uniform_grid
from pyses.operations_2d.limiters import (_clip_and_sum_exact, _default_tau_A,
                                          clip_and_sum_limiter, full_limiter)
from pyses.operations_2d.local_assembly import project_scalar
from pyses.smoothing import soft_reduce_max
from pyses.tracer_transport.eulerian_spectral import calc_minmax

from ...context import to_host
from .probe_utils import probe_forward_reverse, scaled_tangents_like

_be = _get_backend()
jnp = _be.np
device_wrapper = _be.array

HAS_AD = _be.wrapper_type in ("jax", "torch")
needs_ad = pytest.mark.skipif(not HAS_AD, reason="backend has no AD support")

NELEM, NPT, NLEV = 3, 4, 5
QMIN, QMAX = 0.2, 0.8
TAU = 1e-8

LIMITERS = {"clip_and_sum": clip_and_sum_limiter, "full": full_limiter}


def _host(x):
  return np.asarray(to_host(x), dtype=float)


def _inputs(seed=3, margin=None):
  """An active-limiter state: q uniform in [0, 1] against bounds [0.2, 0.8].

  With ``margin`` set, cells within ``margin`` of either bound are pushed
  out of that band, so every clip decision is at least ``margin`` from its
  switch. Element means stay inside the bounds (no bound relaxation)."""
  rng = np.random.default_rng(seed)
  d_mass = 50.0 + 50.0 * rng.random((NELEM, NPT, NPT, NLEV))
  mass_matrix = 0.5 + rng.random((NELEM, NPT, NPT))
  q = rng.random((NELEM, NPT, NPT, NLEV))
  if margin is not None:
    for bound in (QMIN, QMAX):
      near = np.abs(q - bound) < margin
      q = np.where(near, bound + np.sign(q - bound + 1e-300) * margin, q)
  c = mass_matrix[..., None] * d_mass
  mean = (q * c).sum((1, 2)) / c.sum((1, 2))
  assert np.all((mean > QMIN) & (mean < QMAX))
  tracer_min = np.full((NELEM, NLEV), QMIN)
  tracer_max = np.full((NELEM, NLEV), QMAX)
  return tuple(device_wrapper(a) for a in (q * d_mass, mass_matrix, tracer_min, tracer_max, d_mass))


def _redistributed_mass(inputs):
  tm, mm, lo, hi, dm = (_host(a) for a in inputs)
  c = mm[..., None] * dm
  q = tm / dm
  add = (np.where(q > hi[:, None, None, :], q - hi[:, None, None, :], 0.0) +
         np.where(q < lo[:, None, None, :], q - lo[:, None, None, :], 0.0)) * c
  return add.sum((1, 2))


# --- 1. Primal unchanged ----------------------------------------------------

def test_flag_off_is_exact_bitwise():
  args = _inputs()
  np.testing.assert_array_equal(_host(clip_and_sum_limiter(*args)),
                                _host(_clip_and_sum_exact(*args)))


@pytest.mark.parametrize("name", list(LIMITERS))
def test_flag_on_primal_matches_exact(name):
  limiter = LIMITERS[name]
  args = _inputs()
  exact = _host(limiter(*args))
  smooth = _host(limiter(*args, smooth_tau=TAU))
  assert np.max(np.abs(smooth - exact)) <= 1e-15 * np.max(np.abs(exact))


# --- 2. Tangent mass conservation ---------------------------------------------

@needs_ad
@pytest.mark.parametrize("smooth_tau", [None, TAU])
@pytest.mark.parametrize("name", list(LIMITERS))
def test_tangent_conserves_mass(name, smooth_tau):
  limiter = LIMITERS[name]
  args = _inputs()
  assert np.all(np.abs(_redistributed_mass(args)) > 0.0), "fixture must activate the limiter"
  # Perturb tracer mass, bounds and layer mass; hold the quadrature weights
  # fixed (sum(mm * out) = sum(mm * tracer_mass) is the conserved identity, so a
  # mass-matrix tangent would add dmm * (out - tracer_mass) terms).
  tangents = list(scaled_tangents_like(args, seed=5))
  tangents[1] = jnp.zeros_like(args[1])
  tangents = tuple(tangents)
  _, Jv = _be.jvp(lambda *a: limiter(*a, smooth_tau=smooth_tau), args, tangents)
  mm = _host(args[1])[..., None]
  lhs = (mm * _host(Jv)).sum((1, 2))
  rhs = (mm * _host(tangents[0])).sum((1, 2))
  np.testing.assert_allclose(lhs, rhs, rtol=1e-12, atol=1e-12 * np.max(np.abs(rhs)))


# --- 3. Equals the exact derivative off the switches --------------------------

@needs_ad
def test_surrogate_equals_exact_off_switches():
  args = _inputs(margin=100 * TAU)
  tau_A = float(_host(_default_tau_A(TAU, args[1], args[4])))
  assert np.all(np.abs(_redistributed_mass(args)) >= 100 * tau_A)
  tangents = scaled_tangents_like(args, seed=7)
  _, Jv_exact = _be.jvp(clip_and_sum_limiter, args, tangents)
  _, Jv_surr = _be.jvp(lambda *a: clip_and_sum_limiter(*a, smooth_tau=TAU), args, tangents)
  np.testing.assert_allclose(_host(Jv_surr), _host(Jv_exact), rtol=1e-12,
                             atol=1e-12 * np.max(np.abs(_host(Jv_exact))))

  w = device_wrapper(np.random.default_rng(9).standard_normal(_host(args[0]).shape))
  _, pull_exact = _be.vjp(clip_and_sum_limiter, *args)
  _, pull_surr = _be.vjp(lambda *a: clip_and_sum_limiter(*a, smooth_tau=TAU), *args)
  for g_e, g_s in zip(pull_exact(w), pull_surr(w)):
    g_e, g_s = _host(g_e), _host(g_s)
    np.testing.assert_allclose(g_s, g_e, rtol=1e-12, atol=1e-12 * max(np.max(np.abs(g_e)), 1e-300))


# --- 5. Dot-product identity ----------------------------------------------------

@needs_ad
@pytest.mark.parametrize("name", list(LIMITERS))
def test_surrogate_forward_reverse_identity(name):
  limiter = LIMITERS[name]
  probe_forward_reverse(lambda *a: limiter(*a, smooth_tau=TAU), _inputs(),
                        what=f"{name} limiter surrogate", rtol=1e-12)


# --- Bounds: soft tie split ------------------------------------------------------

def test_soft_reduce_max_value_exact():
  x = device_wrapper(np.random.default_rng(1).random((4, 3, 5)))
  np.testing.assert_array_equal(_host(soft_reduce_max(x, (0, 2), TAU)), _host(jnp.max(x, axis=(0, 2))))


@needs_ad
def test_soft_reduce_max_tie_split_is_ulp_stable():
  tied = np.array([0.5, 0.5, 0.3, 0.5])
  nudged = tied.copy()
  nudged[0] = np.nextafter(0.5, 1.0)          # breaks the tie by one ulp
  soft = [_host(_be.grad(lambda x: soft_reduce_max(x, 0, TAU))(device_wrapper(a)))
          for a in (tied, nudged)]
  np.testing.assert_allclose(soft[0], [1 / 3, 1 / 3, 0.0, 1 / 3], atol=1e-6)
  np.testing.assert_allclose(soft[1], soft[0], atol=1e-6)
  exact = _host(_be.grad(lambda x: jnp.max(x, axis=0))(device_wrapper(nudged)))
  np.testing.assert_allclose(exact, [1.0, 0.0, 0.0, 0.0])


@functools.lru_cache(maxsize=1)
def _plane_grid():
  return init_uniform_grid(3, 3, 3)


def _bounds_inputs():
  """One tracer, DSS-continuous (shared nodes equal, as in the model), with a
  random level and a plateau level (all values 0.3)."""
  grid, dims = _plane_grid()
  shape = np.shape(to_host(grid["physical_coords"][:, :, :, 0]))
  rng = np.random.default_rng(21)
  levels = [_host(project_scalar(device_wrapper(rng.random(shape)), grid, dims)),
            np.full(shape, 0.3)]
  return device_wrapper(np.stack(levels, axis=-1)[np.newaxis]), grid, dims


def _ulp_perturbed(a, seed):
  h = _host(a)
  up = np.random.default_rng(seed).random(h.shape) < 0.5
  return device_wrapper(np.where(up, np.nextafter(h, np.inf), np.nextafter(h, -np.inf)))


@needs_ad
def test_calc_minmax_soft_bounds_value_and_ulp_stability():
  tracers, grid, dims = _bounds_inputs()
  for exact_b, soft_b in zip(calc_minmax(tracers, grid, dims),
                             calc_minmax(tracers, grid, dims, smooth_tau=TAU)):
    np.testing.assert_array_equal(_host(soft_b), _host(exact_b))
  v = (device_wrapper(np.random.default_rng(23).standard_normal(_host(tracers).shape)),)

  def jv(tau, x):
    _, out = _be.jvp(lambda t: calc_minmax(t, grid, dims, smooth_tau=tau), (x,), v)
    return np.concatenate([_host(o).ravel() for o in out])

  def rel_change(tau):
    base = jv(tau, tracers)
    return max(np.linalg.norm(jv(tau, _ulp_perturbed(tracers, s)) - base) / np.linalg.norm(base)
               for s in range(3))
  assert rel_change(TAU) < 1e-6
  assert rel_change(None) > 1e-3      # the exact split jumps at the plateau's ties


@needs_ad
def test_calc_minmax_soft_bounds_forward_reverse_identity():
  tracers, grid, dims = _bounds_inputs()
  probe_forward_reverse(lambda t: calc_minmax(t, grid, dims, smooth_tau=TAU), (tracers,),
                        what="calc_minmax soft bounds", rtol=1e-12)
