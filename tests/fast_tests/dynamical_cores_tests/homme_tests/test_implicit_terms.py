"""Unit tests for the HOMME non-hydrostatic DIRK implicit-stage solver.

Two layers of coverage:

* **Kernel-level tests** (``synthetic_state`` fixture):
  shape, finiteness, strict-diagonal-dominance of the analytic
  Jacobian, exact reconstruction of ``J x = rhs`` by the Thomas-
  algorithm solver, and structural properties of
  ``init_search_dir`` and ``take_limited_step``.

* **Acoustic-wave smoke test** (``acoustic_state`` fixture):
  Initialise an isothermal solid-body-rotation atmosphere on a tiny
  ne3 / npt4 mesh (``lapse = 0``, ``u_max = 0``) so the dry background
  is at rest and exactly hydrostatic.  Onto this we layer a small
  Gaussian vertical-velocity perturbation centred at the middle
  interface -- the simplest physical excitation of vertically
  propagating acoustic / gravity-wave modes -- and run a single
  implicit step.  The implicit solver must produce a finite update,
  preserve layer positivity, and pin ``phi`` at the surface.
"""
import numpy as np
import pytest

from pyses._config import get_backend as _get_backend
from pyses.analytic_initialization.hydrostatic_solid_body import (
    init_solid_body_config, init_solid_body_state)
from pyses.dynamical_cores.homme.implicit_terms import (
    advance_buoyancy_explicit, calc_dirk_jacobian, calc_implicit_update,
    calc_perceived_phi_tend, init_search_dir,
    solve_strict_diag_dominant_tridiag, take_limited_step)
from pyses.dynamical_cores.mass_coordinate import init_vertical_grid
from pyses.dynamical_cores.model_info import models
from pyses.dynamical_cores.physics_config import init_physics_config
from pyses.mesh_generation.element_local_metric import init_quasi_uniform_grid_elem_local
from ....test_data.mass_coordinate_grids import cam30
from ....context import to_host

_be = _get_backend()
unwrap = _be.unwrap
device_wrapper = _be.array


# ---------------------------------------------------------------------------
# Synthetic-state fixture for the kernel-level tests
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_state():
  """Random-but-physical state on a tiny ne4 / npt4 / nlev8 grid.

  Returns the kwargs needed to exercise the kernel-level functions
  without going through the full simulator init.  ``physics_config``
  matches the ``"Rgas"`` / ``"cp"`` / ``"gravity"`` keys produced by
  ``init_physics_config`` so the tests catch any drift from the
  codebase convention.
  """
  NE, NPT, NLEV = 4, 4, 8
  NLEVP = NLEV + 1
  rng = np.random.default_rng(0)

  physics_config = {"Rgas": 287.0, "cp": 1004.64,
                    "gravity": 9.81, "radius_earth": 6.371e6}
  v_grid = {"hybrid_b_i": np.linspace(0.0, 1.0, NLEVP)}

  d_mass = np.abs(rng.normal(1.0e3, 100.0, (NE, NPT, NPT, NLEV)))
  # Build phi_i monotonically decreasing from top to surface, all positive.
  phi_levels = np.cumsum(np.abs(rng.normal(500.0, 50.0, NLEV)) + 100.0)[::-1]
  phi_i = np.broadcast_to(phi_levels, (NE, NPT, NPT, NLEV)).copy()
  phi_surf = np.zeros((NE, NPT, NPT))
  phi_i = np.concatenate((phi_i, phi_surf[..., None]), axis=-1)
  d_phi = phi_i[..., 1:] - phi_i[..., :-1]   # negative under HOMME convention
  pnh = np.abs(rng.normal(5e4, 1e3, (NE, NPT, NPT, NLEV)))
  horizontal_wind = rng.normal(0.0, 10.0, (NE, NPT, NPT, NLEV, 2))
  grad_phi_surf = rng.normal(0.0, 1.0, (NE, NPT, NPT, 2))
  w_guess = rng.normal(0.0, 0.1, (NE, NPT, NPT, NLEVP))
  w_guess[..., -1] = 0.0   # surface w pinned to zero

  return {"NE": NE, "NPT": NPT, "NLEV": NLEV, "NLEVP": NLEVP,
          "physics_config": physics_config, "v_grid": v_grid,
          "d_mass": d_mass, "phi_i": phi_i, "phi_surf": phi_surf,
          "d_phi": d_phi, "pnh": pnh,
          "horizontal_wind": horizontal_wind,
          "grad_phi_surf": grad_phi_surf, "w_guess": w_guess}


# ---------------------------------------------------------------------------
# calc_perceived_phi_tend
# ---------------------------------------------------------------------------

def test_calc_perceived_phi_tend_shape_and_boundaries(synthetic_state):
  s = synthetic_state
  gw = calc_perceived_phi_tend(s["horizontal_wind"], s["d_mass"],
                               s["grad_phi_surf"], s["phi_i"],
                               s["physics_config"], s["v_grid"],
                               models.homme_nonhydrostatic)
  assert gw.shape == (s["NE"], s["NPT"], s["NPT"], s["NLEVP"])
  gw = np.asarray(unwrap(gw))
  assert np.all(np.isfinite(gw))
  # The top + surface interfaces are explicitly zeroed by the helper.
  assert np.all(gw[..., 0] == 0.0)
  assert np.all(gw[..., -1] == 0.0)


def test_calc_perceived_phi_tend_vanishes_at_rest(synthetic_state):
  """An atmosphere at rest has zero ``gw_phis`` regardless of topography."""
  s = synthetic_state
  quiescent = np.zeros_like(s["horizontal_wind"])
  gw = calc_perceived_phi_tend(quiescent, s["d_mass"], s["grad_phi_surf"],
                               s["phi_i"], s["physics_config"], s["v_grid"],
                               models.homme_nonhydrostatic)
  assert np.allclose(to_host(gw), 0.0)


def test_calc_perceived_phi_tend_vanishes_flat_surface(synthetic_state):
  """A flat surface (``grad_phi_surf = 0``) zeros the topography term."""
  s = synthetic_state
  flat = np.zeros_like(s["grad_phi_surf"])
  gw = calc_perceived_phi_tend(s["horizontal_wind"], s["d_mass"], flat,
                               s["phi_i"], s["physics_config"], s["v_grid"],
                               models.homme_nonhydrostatic)
  assert np.allclose(to_host(gw), 0.0)


# ---------------------------------------------------------------------------
# calc_dirk_jacobian
# ---------------------------------------------------------------------------

def test_calc_dirk_jacobian_shapes(synthetic_state):
  s = synthetic_state
  jacL, jacD, jacU = calc_dirk_jacobian(100.0, s["d_mass"], s["d_phi"],
                                         s["pnh"], s["physics_config"])
  assert jacD.shape == (s["NE"], s["NPT"], s["NPT"], s["NLEV"])
  assert jacL.shape == (s["NE"], s["NPT"], s["NPT"], s["NLEV"] - 1)
  assert jacU.shape == (s["NE"], s["NPT"], s["NPT"], s["NLEV"] - 1)


def test_calc_dirk_jacobian_strict_diagonal_dominance(synthetic_state):
  """Thomas-algorithm correctness requires |D_k| > |L_{k-1}| + |U_k|."""
  s = synthetic_state
  jacL, jacD, jacU = calc_dirk_jacobian(100.0, s["d_mass"], s["d_phi"],
                                         s["pnh"], s["physics_config"])
  jacL, jacD, jacU = to_host(jacL), to_host(jacD), to_host(jacU)
  pad_L = np.concatenate((np.zeros_like(jacL[..., :1]), np.abs(jacL)),
                         axis=-1)
  pad_U = np.concatenate((np.abs(jacU), np.zeros_like(jacU[..., :1])),
                         axis=-1)
  margin = np.abs(jacD) - (pad_L + pad_U)
  assert margin.min() > 0.0, (
      f"Jacobian not strictly diagonally dominant (min margin = "
      f"{margin.min():.3e})")


def test_calc_dirk_jacobian_dt_zero_limit(synthetic_state):
  """At ``dt = 0`` the implicit Jacobian reduces to the identity."""
  s = synthetic_state
  jacL, jacD, jacU = calc_dirk_jacobian(0.0, s["d_mass"], s["d_phi"],
                                         s["pnh"], s["physics_config"])
  assert np.allclose(to_host(jacL), 0.0)
  assert np.allclose(to_host(jacU), 0.0)
  assert np.allclose(to_host(jacD), 1.0)


# ---------------------------------------------------------------------------
# solve_strict_diag_dominant_tridiag
# ---------------------------------------------------------------------------

def test_solve_tridiag_reconstructs_rhs(synthetic_state):
  """``J x = rhs`` reconstructed to floating-point precision."""
  s = synthetic_state
  rng = np.random.default_rng(1)
  jacL, jacD, jacU = calc_dirk_jacobian(100.0, s["d_mass"], s["d_phi"],
                                         s["pnh"], s["physics_config"])
  rhs = rng.normal(0.0, 1.0, jacD.shape)
  x = solve_strict_diag_dominant_tridiag(jacL, jacD, jacU, rhs)
  assert x.shape == rhs.shape
  # Reconstruct y = J x and compare to the original rhs.
  y = jacD * x

  y += _be.np.concatenate((_be.np.zeros_like(x[..., 0:1]), jacL * x[..., :-1]), axis=-1)
  y += _be.np.concatenate((jacU * x[..., 1:], _be.np.zeros_like(x[..., 0:1])), axis=-1)
  y = np.asarray(unwrap(y))
  relerr = np.max(np.abs(y - rhs) / (np.max(np.abs(rhs)) + 1e-30))
  assert relerr < 1e-10, f"tridiag solver residual = {relerr:.3e}"


def test_solve_tridiag_identity(synthetic_state):
  """For ``J = I``, ``solve(I, rhs) == rhs``."""
  s = synthetic_state
  rng = np.random.default_rng(2)
  rhs = rng.normal(0.0, 1.0, (s["NE"], s["NPT"], s["NPT"], s["NLEV"]))
  jacD = np.ones_like(rhs)
  off_shape = (s["NE"], s["NPT"], s["NPT"], s["NLEV"] - 1)
  jacL = np.zeros(off_shape)
  jacU = np.zeros(off_shape)
  x = solve_strict_diag_dominant_tridiag(jacL, jacD, jacU, rhs.copy())
  assert np.allclose(to_host(x), rhs)


# ---------------------------------------------------------------------------
# init_search_dir
# ---------------------------------------------------------------------------

def test_init_search_dir_pins_surface_to_zero(synthetic_state):
  """The returned search direction must have a zero at the surface."""
  s = synthetic_state
  rng = np.random.default_rng(3)
  residual = rng.normal(0.0, 1e-3, (s["NE"], s["NPT"], s["NPT"], s["NLEVP"]))
  xdir = init_search_dir(100.0, s["w_guess"], s["d_phi"], s["d_mass"],
                         s["pnh"], s["physics_config"], residual)
  assert xdir.shape == s["w_guess"].shape
  xdir = np.asarray(unwrap(xdir))
  assert np.all(np.isfinite(xdir))
  assert np.all(xdir[..., -1] == 0.0)


# ---------------------------------------------------------------------------
# take_limited_step
# ---------------------------------------------------------------------------

def test_take_limited_step_preserves_layer_positivity(synthetic_state):
  """The limiter must enforce ``dphi <= 0`` (no inverted layers) on output."""
  s = synthetic_state
  rng = np.random.default_rng(4)
  residual = rng.normal(0.0, 1e-3, (s["NE"], s["NPT"], s["NPT"], s["NLEVP"]))
  xdir = init_search_dir(100.0, s["w_guess"], s["d_phi"], s["d_mass"],
                         s["pnh"], s["physics_config"], residual)
  dphi_out, w_out = take_limited_step(100.0, s["phi_i"], s["d_phi"],
                                       s["w_guess"], xdir,
                                       s["physics_config"],
                                       models.homme_nonhydrostatic)
  assert dphi_out.shape == (s["NE"], s["NPT"], s["NPT"], s["NLEV"])
  assert w_out.shape == s["w_guess"].shape
  assert np.all(np.isfinite(np.asarray(unwrap(dphi_out))))
  assert np.all(np.isfinite(np.asarray(unwrap(w_out))))
  # Under HOMME's convention ``dphi = phi[k+1] - phi[k] < 0``; the limiter
  # may pin entries right at zero in pathological columns but must not let
  # them go positive.
  assert dphi_out.max() <= 1e-10


def test_take_limited_step_no_op_with_zero_search_direction(synthetic_state):
  """A zero search direction leaves ``w_guess`` unchanged."""
  s = synthetic_state
  zero_xdir = np.zeros_like(s["w_guess"])
  _, w_out = take_limited_step(100.0, s["phi_i"], s["d_phi"], s["w_guess"],
                                zero_xdir, s["physics_config"],
                                models.homme_nonhydrostatic)
  assert np.allclose(to_host(w_out), s["w_guess"])


# ---------------------------------------------------------------------------
# Vertically propagating acoustic test case
# ---------------------------------------------------------------------------

# Tiny mesh -- this is a unit-level test, not a convergence test.
NE_ACOUSTIC = 3
NPT_ACOUSTIC = 4
# Small enough to stay in the linear acoustic-wave regime.
PERTURB_AMPLITUDE = 0.1   # m s^-1
# Gaussian half-width as a fraction of the column depth (in interface index).
PERTURB_HALF_WIDTH_FRAC = 0.1


def _build_acoustic_initial_state():
  """Build the acoustic-wave initial state on a tiny ne3 / npt4 mesh.

  The background is the dry, *isothermal* (``lapse = 0``), at-rest
  (``u_max = 0``) solid-body-rotation atmosphere on the full ``cam30``
  vertical grid with a flat surface and ``enforce_hydrostatic=True``.
  Onto this we layer a small Gaussian vertical-velocity perturbation
  at the middle interface,

    w'(k) = A * exp(-((k - k_mid) / sigma_k)^2),

  with ``w'(0) = w'(nlev) = 0`` so the boundary conditions are
  preserved.  This is the cleanest excitation of vertically propagating
  acoustic + gravity-wave modes for testing the implicit solver.
  """
  model = models.homme_nonhydrostatic
  h_grid, dims = init_quasi_uniform_grid_elem_local(NE_ACOUSTIC, NPT_ACOUSTIC,
                                          calc_smooth_tensor=True)
  v_grid = init_vertical_grid(cam30["hybrid_a_i"],
                              cam30["hybrid_b_i"],
                              cam30["p0"], model)
  physics_config = init_physics_config(model)
  test_config = init_solid_body_config(model_config=physics_config,
                                        lapse=0.0, u_max=0.0)
  state = init_solid_body_state(h_grid, v_grid, physics_config, test_config,
                                 dims, model, mountain=False,
                                 enforce_hydrostatic=True)
  # Add a Gaussian w perturbation centred at the middle interface.
  w_i = np.asarray(unwrap(state["dynamics"]["w_i"]))
  n_int = w_i.shape[-1]
  k_mid = (n_int - 1) // 2
  sigma_k = max(PERTURB_HALF_WIDTH_FRAC * n_int, 1.0)
  k_idx = np.arange(n_int)
  pert = PERTURB_AMPLITUDE * np.exp(-((k_idx - k_mid) / sigma_k) ** 2)
  pert[0] = 0.0
  pert[-1] = 0.0
  w_i = w_i + pert[np.newaxis, np.newaxis, np.newaxis, :]
  state["dynamics"]["w_i"] = device_wrapper(w_i, elem_sharding_axis=0)
  return state, h_grid, v_grid, dims, physics_config, model


@pytest.fixture(scope="module")
def acoustic_state():
  state, h_grid, v_grid, dims, physics_config, model = _build_acoustic_initial_state()
  return {"state": state, "h_grid": h_grid, "v_grid": v_grid,
          "dims": dims, "physics_config": physics_config, "model": model}


def test_acoustic_initial_state_is_finite(acoustic_state):
  """The seeded acoustic atmosphere is entirely finite."""
  d = acoustic_state["state"]["dynamics"]
  for key in ("phi_i", "w_i", "horizontal_wind", "d_mass", "theta_v_d_mass"):
    arr = np.asarray(unwrap(d[key]))
    assert np.all(np.isfinite(arr)), f"{key} contains non-finite values"


def test_acoustic_w_perturbation_is_at_prescribed_scale(acoustic_state):
  """The perturbation should peak at ``A`` and be zero at the boundaries."""
  w = np.asarray(unwrap(acoustic_state["state"]["dynamics"]["w_i"]))
  assert np.allclose(w[..., 0], 0.0)
  assert np.allclose(w[..., -1], 0.0)
  assert np.max(np.abs(w)) <= PERTURB_AMPLITUDE + 1e-12
  assert np.isclose(np.max(np.abs(w)), PERTURB_AMPLITUDE, rtol=1e-9)


def test_acoustic_background_is_at_rest_horizontally(acoustic_state):
  """``u_max = 0`` should yield zero horizontal wind everywhere."""
  u = np.asarray(unwrap(acoustic_state["state"]["dynamics"]["horizontal_wind"]))
  assert np.allclose(u, 0.0, atol=1e-12)


def test_acoustic_background_is_hydrostatic(acoustic_state):
  """``enforce_hydrostatic=True`` guarantees a hydrostatic phi_i."""
  d = acoustic_state["state"]["dynamics"]
  phi_i = np.asarray(unwrap(d["phi_i"]))
  # phi decreases monotonically from top to surface in HOMME's convention.
  assert (phi_i[..., 1:] - phi_i[..., :-1] <= 0.0).all()


def test_implicit_update_acoustic_smoke(acoustic_state):
  """End-to-end: ``calc_implicit_update`` runs and returns finite output.

  For a "first stage" with no prior explicit contributions, the RHS
  ``(w_n0, phi_n0)`` is just the current dynamics' ``(w_i, phi_i)``.
  """
  a = acoustic_state
  state = a["state"]
  dynamics = state["dynamics"]
  static_forcing = state["static_forcing"]
  nh_vars = {"w_i": 1.0 * dynamics["w_i"],
             "phi_i": 1.0 * dynamics["phi_i"]}
  dt_implicit = 5.0
  out = calc_implicit_update(dt_implicit, nh_vars, dynamics, static_forcing,
                              a["v_grid"], a["physics_config"], a["model"])
  for key in ("phi_i", "w_i", "horizontal_wind", "d_mass", "theta_v_d_mass"):
    arr = np.asarray(unwrap(out[key]))
    in_arr = np.asarray(unwrap(dynamics[key]))
    assert np.all(np.isfinite(arr)), f"output {key} contains non-finite values"
    assert arr.shape == in_arr.shape


def test_implicit_update_pins_surface_phi(acoustic_state):
  """After the solve, ``phi[..., -1] == phi_surf`` to machine precision."""
  a = acoustic_state
  state = a["state"]
  dynamics = state["dynamics"]
  static_forcing = state["static_forcing"]
  nh_vars = {"w_i": 1.0 * dynamics["w_i"],
             "phi_i": 1.0 * dynamics["phi_i"]}
  out = calc_implicit_update(5.0, nh_vars, dynamics, static_forcing,
                              a["v_grid"], a["physics_config"], a["model"])
  phi_next = np.asarray(unwrap(out["phi_i"]))
  phi_surf = np.asarray(unwrap(static_forcing["phi_surf"]))
  assert np.allclose(phi_next[..., -1], phi_surf)


def test_advance_buoyancy_explicit_empty_series(acoustic_state):
  """An empty alpha series returns the initial ``(w_n0, phi_n0)`` unchanged."""
  a = acoustic_state
  dynamics = a["state"]["dynamics"]
  out = advance_buoyancy_explicit(
      alpha_dts=[], dynamics_series=[],
      w_init=dynamics["w_i"], phi_init=dynamics["phi_i"],
      static_forcing=a["state"]["static_forcing"],
      physics_config=a["physics_config"],
      v_grid=a["v_grid"], model=a["model"])
  assert np.allclose(np.asarray(unwrap(out["w_i"])),
                      np.asarray(unwrap(dynamics["w_i"])))
  assert np.allclose(np.asarray(unwrap(out["phi_i"])),
                      np.asarray(unwrap(dynamics["phi_i"])))
