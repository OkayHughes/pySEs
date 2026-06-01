"""Physics-forcing application is exact when every other sub-step is disabled.

Uses the instrumented coupling step (tests/instrumented_coupling.py) with the
adiabatic dynamics advance, sponge, hyperviscosity, tracer advection and both
vertical remaps switched off, so the coupling step reduces to a single
operator-split forward-Euler add of the physics tendency.

In that regime the *only* correct result, regardless of coupling strategy, is

    state_out = state_in + physics_dt * physics_tendency

applied field-by-field. This test pins exactly that for both ``lump_all`` and
``dribble_all``, checking each dynamical field (wind, temperature, layer mass)
and each tracer species (moisture, dry air, passive) independently — so a
strategy that mis-scales the increment, or leaks a forcing into the wrong
field, fails loudly.
"""

import numpy as np
import pytest

from pyses._config import get_backend as _get_backend
from pyses.analytic_initialization.moist_baroclinic_wave import (
    init_baroclinic_wave_config, init_baroclinic_wave_state)
from pyses.dynamical_cores.mass_coordinate import init_vertical_grid
from pyses.dynamical_cores.model_config import init_default_config
from pyses.dynamical_cores.model_info import models
from pyses.dynamical_cores.model_state import (
    wrap_dynamics, wrap_tracers, sum_dynamics_series, sum_tracers_series)
from pyses.dynamical_cores.physics_dynamics_coupling import coupling_types
from pyses.dynamical_cores.time_step import time_step_options
from pyses.dynamical_cores.time_stepping import init_timestep_config

from .conftest import cached_quasi_uniform_grid_elem_local
from ...instrumented_coupling import advance_coupling_step_instrumented
from ...test_data.mass_coordinate_grids import cam30

_be = _get_backend()
bnp = _be.np

# Everything except the physics-forcing add is turned off, so the coupling step
# is a pure forward-Euler application of the tendency.
_ALL_PROCESSES_OFF = dict(
    enable_adiabatic_dynamics=False,
    enable_sponge_layer=False,
    enable_hyperviscosity=False,
    enable_tracer_transport=False,
    enable_remap_dynamics=False,
    enable_remap_tracers=False,
)

# Distinct, finite per-field tendencies so a cross-field leak would be visible.
_WIND_TEND = 1.0e-4     # m / s^2
_T_TEND = 1.0e-3        # K / s
_D_MASS_TEND = 1.0e-2   # Pa / s
_VAPOR_TEND = 1.0e-8    # (dry mixing ratio) / s


def _build(coupling):
  model = models.cam_se
  nx, npt = 3, 4
  h_grid, dims = cached_quasi_uniform_grid_elem_local(nx, npt, calc_smooth_tensor=True)
  v_grid = init_vertical_grid(cam30["hybrid_a_i"], cam30["hybrid_b_i"], cam30["p0"], model)
  physics_config, diffusion_config, base_tc = init_default_config(nx, h_grid, v_grid, dims, model)
  timestep_config = init_timestep_config(
      base_tc["physics_dt"], h_grid, physics_config, diffusion_config, dims, model,
      dynamics_tstep_type=time_step_options.RK3_5STAGE,
      physics_dynamics_coupling=coupling)
  test_config = init_baroclinic_wave_config(model_config=physics_config)
  state = init_baroclinic_wave_state(
      h_grid, v_grid, physics_config, test_config, dims, model, mountain=False, moist=True)
  return dict(model=model, h_grid=h_grid, v_grid=v_grid, physics_config=physics_config,
              diffusion_config=diffusion_config, timestep_config=timestep_config,
              dims=dims, state=state)


def _physics_forcing(state, model):
  """Force wind, temperature, layer mass and water vapour; nothing else."""
  dyn = state["dynamics"]
  ts = state["tracers"]
  dyn_forcing = wrap_dynamics(_WIND_TEND * bnp.ones_like(dyn["horizontal_wind"]),
                              _T_TEND * bnp.ones_like(dyn["T"]),
                              _D_MASS_TEND * bnp.ones_like(dyn["d_mass"]),
                              model)
  moisture = {k: (_VAPOR_TEND * bnp.ones_like(v) if k == "water_vapor"
                  else bnp.zeros(v.shape))
              for k, v in ts["moisture_species"].items()}
  zpass = {k: bnp.zeros(v.shape) for k, v in ts["tracers"].items()}
  zdry = {k: bnp.zeros(v.shape) for k, v in ts["dry_air_species"].items()}
  trac_forcing = wrap_tracers(moisture, zpass, model, dry_air_species=zdry)
  return {"dynamics": dyn_forcing, "tracers": trac_forcing}


def _close(a, b):
  return np.allclose(np.asarray(_be.unwrap(a)), np.asarray(_be.unwrap(b)),
                     rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("coupling", [coupling_types.lump_all, coupling_types.dribble_all])
def test_pure_forcing_matches_forward_euler_per_field(coupling):
  """With all other sub-steps off, each field/tracer == in + physics_dt * tendency."""
  case = _build(coupling)
  state = case["state"]
  forcing = _physics_forcing(state, case["model"])
  dt = case["timestep_config"]["physics_dt"]

  out = advance_coupling_step_instrumented(
      state, case["h_grid"], case["v_grid"], case["physics_config"],
      case["diffusion_config"], case["timestep_config"], case["dims"], case["model"],
      physics_forcing=forcing, **_ALL_PROCESSES_OFF)

  # Reference = one forward-Euler step over the full physics timestep.
  expected_dyn = sum_dynamics_series([state["dynamics"], forcing["dynamics"]],
                                     [1.0, dt], case["model"])
  expected_trac = sum_tracers_series([state["tracers"], forcing["tracers"]],
                                     [1.0, dt], case["model"])

  # Each dynamical field independently.
  for field in ("horizontal_wind", "T", "d_mass"):
    assert _close(out["dynamics"][field], expected_dyn[field]), (
        f"{coupling.name}: dynamics field {field!r} not applied as in + dt*tendency")

  # Each tracer species independently (forced water vapour changes; the
  # unforced dry-air and passive tracers must be left exactly alone).
  for group in ("moisture_species", "dry_air_species", "tracers"):
    for name in out["tracers"][group]:
      assert _close(out["tracers"][group][name], expected_trac[group][name]), (
          f"{coupling.name}: tracer {group}/{name!r} not applied as in + dt*tendency")
