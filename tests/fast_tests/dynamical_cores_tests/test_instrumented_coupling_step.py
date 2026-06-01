"""The instrumented coupling step must reproduce the production one.

``advance_coupling_step_instrumented`` (tests/instrumented_coupling.py) is a
copy of :func:`pyses.dynamical_cores.run_dycore.advance_coupling_step` with each
of its eight sub-steps behind an independent ``enable_*`` flag. This test pins
the contract that, with every flag enabled, the two routines return identical
state for the same model configuration — so the instrumented copy can be
trusted as a faithful baseline when ablating individual sub-steps.

A second test flips flags off to confirm the toggles are actually wired (a
disabled sub-step changes the result), so the equivalence above is meaningful
rather than an artefact of dead code.
"""

import numpy as np

from pyses._config import get_backend as _get_backend
from pyses.analytic_initialization.moist_baroclinic_wave import (
    init_baroclinic_wave_config, init_baroclinic_wave_state)
from pyses.dynamical_cores.mass_coordinate import init_vertical_grid
from pyses.dynamical_cores.model_config import init_default_config
from pyses.dynamical_cores.model_info import models
from pyses.dynamical_cores.model_state import wrap_dynamics, wrap_tracers
from pyses.dynamical_cores.physics_dynamics_coupling import coupling_types
from pyses.dynamical_cores.run_dycore import advance_coupling_step
from pyses.dynamical_cores.time_step import time_step_options
from pyses.dynamical_cores.time_stepping import init_timestep_config

from .conftest import cached_quasi_uniform_grid_elem_local
from ...instrumented_coupling import advance_coupling_step_instrumented
from ...test_data.mass_coordinate_grids import cam30

_be = _get_backend()
bnp = _be.np


def _build_case():
  """A cam_se, multi-level, lump_all configuration that exercises all 8 sub-steps.

  cam_se gives moisture/dry-air species (tracer transport + theta conversion);
  the 30-level cam30 grid makes ``do_remap`` true (dynamics + tracer remap);
  the default tensor diffusion config carries hyperviscosity and a sponge layer;
  and a lump_all coupling with a non-zero ``physics_forcing`` activates the two
  physics-forcing sub-steps.
  """
  model = models.cam_se
  nx, npt = 3, 4
  h_grid, dims = cached_quasi_uniform_grid_elem_local(nx, npt, calc_smooth_tensor=True)
  v_grid = init_vertical_grid(cam30["hybrid_a_i"], cam30["hybrid_b_i"], cam30["p0"], model)

  physics_config, diffusion_config, base_tc = init_default_config(nx, h_grid, v_grid, dims, model)
  # lump_all so the physics tendency is actually applied (exercises forcing).
  timestep_config = init_timestep_config(
      base_tc["physics_dt"], h_grid, physics_config, diffusion_config, dims, model,
      dynamics_tstep_type=time_step_options.RK3_5STAGE,
      physics_dynamics_coupling=coupling_types.lump_all,
  )

  test_config = init_baroclinic_wave_config(model_config=physics_config)
  model_state = init_baroclinic_wave_state(
      h_grid, v_grid, physics_config, test_config, dims, model, mountain=False, moist=True)

  # Small, finite physics tendency touching winds, temperature and moisture.
  dyn = model_state["dynamics"]
  ts = model_state["tracers"]
  dyn_forcing = wrap_dynamics(1e-4 * bnp.ones_like(dyn["horizontal_wind"]),
                              1e-3 * bnp.ones_like(dyn["T"]),
                              bnp.zeros(dyn["d_mass"].shape),
                              model)
  moist_forcing = {k: 1e-9 * bnp.ones_like(v) for k, v in ts["moisture_species"].items()}
  zpass = {k: bnp.zeros(v.shape) for k, v in ts["tracers"].items()}
  zdry = {k: bnp.zeros(v.shape) for k, v in ts["dry_air_species"].items()}
  trac_forcing = wrap_tracers(moist_forcing, zpass, model, dry_air_species=zdry)
  physics_forcing = {"dynamics": dyn_forcing, "tracers": trac_forcing}

  return dict(state_in=model_state, h_grid=h_grid, v_grid=v_grid,
              physics_config=physics_config, diffusion_config=diffusion_config,
              timestep_config=timestep_config, dims=dims, model=model,
              physics_forcing=physics_forcing)


def _assert_states_equal(reference, candidate, path="state"):
  """Recursively assert two model-state pytrees are bit-identical."""
  if isinstance(reference, dict):
    assert isinstance(candidate, dict), f"{path}: type mismatch"
    assert reference.keys() == candidate.keys(), (
        f"{path}: key mismatch {set(reference) ^ set(candidate)}")
    for key in reference:
      _assert_states_equal(reference[key], candidate[key], f"{path}.{key}")
    return
  ref = np.asarray(_be.unwrap(reference))
  cand = np.asarray(_be.unwrap(candidate))
  assert ref.shape == cand.shape, f"{path}: shape {ref.shape} vs {cand.shape}"
  assert np.array_equal(ref, cand), (
      f"{path}: arrays differ (max |diff| = {np.max(np.abs(ref - cand))})")


def test_instrumented_matches_reference_when_all_enabled():
  """All flags on -> identical to the production advance_coupling_step."""
  case = _build_case()
  reference = advance_coupling_step(**case)
  candidate = advance_coupling_step_instrumented(**case)  # flags default to True
  _assert_states_equal(reference, candidate)


def test_disabling_substeps_changes_result():
  """Each toggle is wired: turning one off perturbs the output state."""
  case = _build_case()

  # Dynamics-side ablations must change the winds. We probe these with tracer
  # transport disabled: for the default tensor-hyperviscosity config the tracer
  # solver's mass-consistency correction is coupled to the hyperviscosity term
  # (it reads ``d_mass_tracer``), so disabling hyperviscosity alone is not a
  # standalone ablation. The winds are independent of tracer transport, so
  # holding it off in both the reference and the probe still isolates each
  # dynamics flag.
  dyn_ref = advance_coupling_step_instrumented(**case, enable_tracer_transport=False)
  dyn_ref_wind = np.asarray(_be.unwrap(dyn_ref["dynamics"]["horizontal_wind"]))
  for flag in ("enable_physics_dynamics_forcing", "enable_adiabatic_dynamics",
               "enable_sponge_layer", "enable_hyperviscosity", "enable_remap_dynamics"):
    out = advance_coupling_step_instrumented(
        **case, enable_tracer_transport=False, **{flag: False})
    wind = np.asarray(_be.unwrap(out["dynamics"]["horizontal_wind"]))
    assert not np.array_equal(wind, dyn_ref_wind), f"{flag}=False left the winds unchanged"

  # ...and a moisture/tracer-side ablation must change the water vapour.
  reference = advance_coupling_step_instrumented(**case)
  ref_vapor = np.asarray(_be.unwrap(reference["tracers"]["moisture_species"]["water_vapor"]))
  for flag in ("enable_physics_moisture_forcing", "enable_tracer_transport",
               "enable_remap_tracers"):
    out = advance_coupling_step_instrumented(**case, **{flag: False})
    vapor = np.asarray(_be.unwrap(out["tracers"]["moisture_species"]["water_vapor"]))
    assert not np.array_equal(vapor, ref_vapor), f"{flag}=False left the vapour unchanged"
