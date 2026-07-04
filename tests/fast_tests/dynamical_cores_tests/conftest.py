"""Session-amortised builders for the dynamical-cores fast tests.

The cubed-sphere mesh + Jacobian metric computation in
``init_quasi_uniform_grid_elem_local`` and
the analytic baroclinic-wave initialisation in
``init_baroclinic_wave_state`` dominate per-test setup cost.  Both
results are pure functions of their inputs, so we memoise them in
module-level dicts keyed by the call arguments.  The module-level
caches persist for the lifetime of the pytest process, which is enough
to amortise across the whole ``fast_tests/`` run.

Tests that mutate their state (e.g.
``state["dynamics"]["horizontal_wind"] += ...``) get a fresh deep copy
of the model state on each fixture invocation, so the cached arrays
stay pristine.  Grid / config dicts are returned by reference (they're
read-only by convention).

The original ``quasi_uniform_test_states`` signature and the 14
``@fixture`` factories are preserved; the caching is transparent.
Tests that build grids / states directly (outside the fixtures) can
opt in by importing the ``cached_*`` builders from this module.
"""
from pytest import fixture

from pyses.analytic_initialization.moist_baroclinic_wave import (
    init_baroclinic_wave_config, init_baroclinic_wave_state)
from pyses.dynamical_cores.mass_coordinate import init_vertical_grid
from pyses.dynamical_cores.model_info import models
from pyses.dynamical_cores.physics_config import init_physics_config
from pyses.mesh_generation.element_local_metric import (
    init_quasi_uniform_grid_elem_local)
from ...test_data.mass_coordinate_grids import cam30


# ---------------------------------------------------------------------------
# Module-level caches.
# ---------------------------------------------------------------------------

_GRID_CACHE = {}              # ("elem_local", nx, npt, calc_smooth_tensor) -> (h_grid, dims)
_V_GRID_CACHE = {}            # model -> v_grid
_PHYSICS_CACHE = {}           # model -> physics_config
_BAROCLINIC_CFG_CACHE = {}    # model -> baroclinic-wave test_config
_BAROCLINIC_STATE_CACHE = {}  # state_key -> model_state


# ---------------------------------------------------------------------------
# Cached builders -- public.
# ---------------------------------------------------------------------------

def cached_quasi_uniform_grid_elem_local(nx, npt, *, calc_smooth_tensor=False):
  """Return ``(h_grid, dims)`` from the element-local cubed-sphere builder,
  cached by ``(nx, npt, calc_smooth_tensor)``."""
  key = ("elem_local", int(nx), int(npt), bool(calc_smooth_tensor))
  if key not in _GRID_CACHE:
    _GRID_CACHE[key] = init_quasi_uniform_grid_elem_local(
        nx, npt, calc_smooth_tensor=calc_smooth_tensor)
  return _GRID_CACHE[key]


def cached_v_grid(model):
  """Cached ``init_vertical_grid`` on the full ``cam30`` levels, keyed by
  ``model``."""
  if model not in _V_GRID_CACHE:
    _V_GRID_CACHE[model] = init_vertical_grid(cam30["hybrid_a_i"],
                                              cam30["hybrid_b_i"],
                                              cam30["p0"], model)
  return _V_GRID_CACHE[model]


def cached_physics_config(model):
  """Cached ``init_physics_config(model)``."""
  if model not in _PHYSICS_CACHE:
    _PHYSICS_CACHE[model] = init_physics_config(model)
  return _PHYSICS_CACHE[model]


def cached_baroclinic_test_config(model):
  """Cached ``init_baroclinic_wave_config(model_config=physics_config)``."""
  if model not in _BAROCLINIC_CFG_CACHE:
    _BAROCLINIC_CFG_CACHE[model] = init_baroclinic_wave_config(
        model_config=cached_physics_config(model))
  return _BAROCLINIC_CFG_CACHE[model]


def _deep_copy_state(obj):
  """Recursively copy a model-state-like nested dict.

  Backend array leaves are duplicated via ``.copy()`` (numpy / torch) or
  ``1.0 *`` (JAX returns a fresh array regardless), so callers are free
  to mutate the returned arrays (``+=`` etc.) without corrupting the
  cache.
  """
  if isinstance(obj, dict):
    return {k: _deep_copy_state(v) for k, v in obj.items()}
  if isinstance(obj, tuple):
    return tuple(_deep_copy_state(v) for v in obj)
  if isinstance(obj, list):
    return [_deep_copy_state(v) for v in obj]
  if hasattr(obj, "copy") and callable(obj.copy):
    return obj.copy()
  if hasattr(obj, "shape") and hasattr(obj, "dtype"):
    # Immutable backend arrays (JAX, torch sometimes); ``1.0 *`` reliably
    # returns a fresh array of the same shape / dtype.
    return 1.0 * obj
  return obj


def cached_baroclinic_state(nx, npt, model, *, mountain=False, moist=False,
                            calc_smooth_tensor=False,
                            eps=1e-5, enforce_hydrostatic=True):
  """Memoised analytic baroclinic-wave state on the element-local cubed-sphere
  grid.  Returns a *deep-copied* state so the caller can mutate without
  corrupting the cache."""
  key = (int(nx), int(npt), bool(calc_smooth_tensor),
         model, bool(mountain), bool(moist),
         float(eps), bool(enforce_hydrostatic))
  if key not in _BAROCLINIC_STATE_CACHE:
    h_grid, dims = cached_quasi_uniform_grid_elem_local(
        nx, npt, calc_smooth_tensor=calc_smooth_tensor)
    _BAROCLINIC_STATE_CACHE[key] = init_baroclinic_wave_state(
        h_grid, cached_v_grid(model), cached_physics_config(model),
        cached_baroclinic_test_config(model), dims, model,
        mountain=mountain, moist=moist,
        enforce_hydrostatic=enforce_hydrostatic, eps=eps)
  return _deep_copy_state(_BAROCLINIC_STATE_CACHE[key])


# ---------------------------------------------------------------------------
# Backwards-compatible API: ``quasi_uniform_test_states`` keeps its original
# signature; the fixture bodies below are unchanged.
# ---------------------------------------------------------------------------

def quasi_uniform_test_states(nx, npt, model, mountain=False, moist=False,
                              *, calc_smooth_tensor=False, eps=1e-5,
                              enforce_hydrostatic=True):
  """Build (or retrieve from cache) the grid + analytic baroclinic-wave state."""
  h_grid, dims = cached_quasi_uniform_grid_elem_local(
      nx, npt, calc_smooth_tensor=calc_smooth_tensor)
  return {"h_grid": h_grid,
          "v_grid": cached_v_grid(model),
          "dims": dims,
          "physics_config": cached_physics_config(model),
          "model_state": cached_baroclinic_state(
              nx, npt, model, mountain=mountain, moist=moist,
              calc_smooth_tensor=calc_smooth_tensor,
              eps=eps, enforce_hydrostatic=enforce_hydrostatic)}


# ---------------------------------------------------------------------------
# The original 14 fixtures.  Bodies are unchanged: they call through to
# ``quasi_uniform_test_states`` which is now backed by the module-level
# cache.
# ---------------------------------------------------------------------------

@fixture
def nx15_np4_dry_se():
  return quasi_uniform_test_states(15, 4, models.cam_se)


@fixture
def nx15_np4_dry_se_whole():
  return quasi_uniform_test_states(15, 4, models.cam_se_whole_atmosphere)


@fixture
def nx15_np4_dry_homme_hydro():
  return quasi_uniform_test_states(15, 4, models.homme_hydrostatic)


@fixture
def nx15_np4_dry_homme_nonhydro():
  return quasi_uniform_test_states(15, 4, models.homme_nonhydrostatic)


@fixture
def nx7_np4_dry_se():
  return quasi_uniform_test_states(7, 4, models.cam_se)


@fixture
def nx7_np4_dry_se_whole():
  return quasi_uniform_test_states(7, 4, models.cam_se_whole_atmosphere)


@fixture
def nx7_np4_dry_homme_hydro():
  return quasi_uniform_test_states(7, 4, models.homme_hydrostatic)


@fixture
def nx7_np4_dry_homme_nonhydro():
  return quasi_uniform_test_states(7, 4, models.homme_nonhydrostatic)


@fixture
def nx15_np4_moist_se():
  return quasi_uniform_test_states(15, 4, models.cam_se, moist=True)


@fixture
def nx15_np4_moist_se_whole():
  return quasi_uniform_test_states(15, 4, models.cam_se_whole_atmosphere,
                                   moist=True)


@fixture
def nx15_np4_moist_homme_hydro():
  return quasi_uniform_test_states(15, 4, models.homme_hydrostatic, moist=True)


@fixture
def nx15_np4_moist_homme_nonhydro():
  return quasi_uniform_test_states(15, 4, models.homme_nonhydrostatic,
                                   moist=True)


@fixture
def nx15_np4_mountain_se():
  return quasi_uniform_test_states(15, 4, models.cam_se, mountain=True)


@fixture
def nx15_np4_mountain_se_whole():
  return quasi_uniform_test_states(15, 4, models.cam_se_whole_atmosphere,
                                   mountain=True)


@fixture
def nx15_np4_mountain_homme_hydro():
  return quasi_uniform_test_states(15, 4, models.homme_hydrostatic,
                                   mountain=True)


@fixture
def nx15_np4_mountain_homme_nonhydro():
  return quasi_uniform_test_states(15, 4, models.homme_nonhydrostatic,
                                   mountain=True)
