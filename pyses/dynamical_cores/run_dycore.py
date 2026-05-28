from .time_stepping import (advance_dynamics_euler,
                            advance_hypervis_euler,
                            advance_dynamics_ullrich_5stage,
                            advance_sponge_euler)
from .._config import get_backend as _get_backend, runtime_assert
from .model_state import remap_dynamics, remap_tracers
from .time_step import time_step_options
from .model_state import (sum_dynamics_series,
                          sum_tracers_series,
                          wrap_model_state,
                          check_dynamics_nan,
                          check_tracers_nan,
                          sum_consistency_struct,
                          se_T_to_theta_d_d_mass,
                          se_theta_d_d_mass_to_T,
                          renormalize_dry_air_species)
from .homme.thermodynamics import eval_balanced_geopotential
from .mass_coordinate import d_mass_to_surface_mass, surface_mass_to_midlevel_mass
from .physics_dynamics_coupling import coupling_types
from .tracer_advection.eulerian_spectral import advance_tracers
from .model_info import cam_se_models, cam_se_stable_models
from functools import partial
_be = _get_backend()
jit = _be.jit
DEBUG = _be.debug



@partial(jit, static_argnames=["model", "dims", "timestep_config"])
def advance_coupling_step(state_in,
                          h_grid,
                          v_grid,
                          physics_config,
                          diffusion_config,
                          timestep_config,
                          dims,
                          model,
                          physics_forcing=None):
  """
  Advance the model by one physics timestep.

  Performs the full physics–dynamics coupling sequence, including
  subcycled dynamics (and optional hyperviscosity), tracer subcycling
  with consistency corrections, sponge layer damping, and vertical
  remapping.

  Parameters
  ----------
  state_in : model state dict
      Current model state containing ``"dynamics"``, ``"tracers"``,
      and ``"static_forcing"`` sub-dicts.
  h_grid : `SpectralElementGrid`
      Horizontal spectral element grid struct.
  v_grid : `dict`
      Vertical grid struct containing hybrid coordinate coefficients.
  physics_config : `dict`
      Model physics configuration dict.
  diffusion_config : `dict`
      Hyperviscosity and sponge-layer configuration dict.
  timestep_config : `dict`
      Timestep configuration dict.  Must contain
      ``"physics_dynamics_coupling"``, ``"tracer_subcycle"``,
      ``"dynamics_subcycle"``, ``"physics_dt"``, ``"dynamics"``,
      and ``"hyperviscosity"`` keys.
  dims : frozendict[str, int]
      Grid dimension metadata.
  model : model_info.models
      Dynamical core identifier (from ``model_info.models``).
  physics_forcing : dict, optional
      Physics tendencies to be applied during the coupling step.
      If ``None``, no physics forcing is applied.

  Returns
  -------
  state_out : model state dict
      Updated model state after advancing one physics timestep.
  """
  physics_dynamics_coupling = timestep_config["physics_dynamics_coupling"]
  do_remap = v_grid["hybrid_a_m"].shape[0] > 1

  dynamics_state = state_in["dynamics"]
  tracer_state = state_in["tracers"]
  static_forcing = state_in["static_forcing"]
  dribble_dynamics = (physics_dynamics_coupling == coupling_types.dribble_all or
                      physics_dynamics_coupling == coupling_types.lump_tracers_dribble_dynamics)

  if (physics_dynamics_coupling == coupling_types.lump_tracers_dribble_dynamics) and physics_forcing is not None:
    tracer_state = sum_tracers_series([tracer_state, physics_forcing["tracers"]],
                                      [1.0, timestep_config["physics_dt"]],
                                      model)

  if physics_dynamics_coupling == coupling_types.lump_all and physics_forcing is not None:
    dynamics_state = sum_dynamics_series([dynamics_state, physics_forcing["dynamics"]],
                                         [1.0, timestep_config["physics_dt"]],
                                         model)
    tracer_state = sum_tracers_series([tracer_state, physics_forcing["tracers"]],
                                      [1.0, timestep_config["physics_dt"]],
                                      model)
  for q_split in range(timestep_config["tracer_subcycle"]):
    if do_remap:
      dynamics_state = remap_dynamics(dynamics_state,
                                      state_in["static_forcing"],
                                      v_grid,
                                      physics_config,
                                      len(v_grid["hybrid_b_m"]),
                                      model)
    tracer_consist_init = {"d_mass_init": 1.0 * dynamics_state["d_mass"]}
    if dribble_dynamics and physics_forcing is not None:
      dynamics_state = sum_dynamics_series([dynamics_state, physics_forcing["dynamics"]],
                                           [1.0, timestep_config["tracer_advection"]["dt"]],
                                           model)
    if physics_dynamics_coupling == coupling_types.dribble_all and physics_forcing is not None:
      tracer_state = sum_tracers_series([tracer_state, physics_forcing["tracers"]],
                                        [1.0, timestep_config["physics_dt"]],
                                        model)

    for n_split in range(timestep_config["dynamics_subcycle"]):
      if model in cam_se_models:
        moisture_species = tracer_state["moisture_species"]
        dry_air_species = tracer_state["dry_air_species"]
      else:
        moisture_species = None
        dry_air_species = None
      # Run the adiabatic dynamics step in the _stable (theta_d) variant so the
      # explicit terms use the skew-symmetric / Exner-form formulation; the
      # T-based interface is restored after the advance, so hyperviscosity,
      # sponge, tracer transport, and the dynamics_next/dynamics_state swap
      # below all continue to see ``model`` unchanged.  Skip the conversion
      # when ``model`` is already a _stable variant (otherwise we'd try to read
      # the T key that doesn't exist on those states).
      # if model in cam_se_models and model not in cam_se_stable_models:
      #   dynamics_state, step_model = se_T_to_theta_d_d_mass(
      #       dynamics_state, v_grid, physics_config, model)
      # else:
      step_model = model
      if timestep_config["dynamics"]["step_type"] == time_step_options.Euler:
        dynamics_next, tracer_consist_dyn = advance_dynamics_euler(dynamics_state,
                                                                   static_forcing,
                                                                   h_grid,
                                                                   v_grid,
                                                                   physics_config,
                                                                   timestep_config,
                                                                   dims,
                                                                   step_model,
                                                                   moisture_species=moisture_species,
                                                                   dry_air_species=dry_air_species)
      elif timestep_config["dynamics"]["step_type"] == time_step_options.RK3_5STAGE:
        dynamics_next, tracer_consist_dyn = advance_dynamics_ullrich_5stage(dynamics_state,
                                                                            static_forcing,
                                                                            h_grid,
                                                                            v_grid,
                                                                            physics_config,
                                                                            timestep_config,
                                                                            dims,
                                                                            step_model,
                                                                            moisture_species=moisture_species,
                                                                            dry_air_species=dry_air_species)
      else:
        raise ValueError("Unknown dynamics timestep type")
      # Restore the external T-prognostic interface for the rest of the
      # iteration; also restore dynamics_state so the swap on the bottom of
      # the loop preserves a T-form allocation for the next iteration.  Only
      # convert back if we converted in (i.e., model was a non-stable variant).
      # if model in cam_se_models and model not in cam_se_stable_models:
      #   dynamics_next, _ = se_theta_d_d_mass_to_T(
      #       dynamics_next, v_grid, physics_config, step_model)
      #   dynamics_state, _ = se_theta_d_d_mass_to_T(
      #       dynamics_state, v_grid, physics_config, step_model)
      if "disable_diffusion" not in diffusion_config.keys():
        if timestep_config["hyperviscosity"]["step_type"] == time_step_options.Euler:
          dynamics_next, tracer_consist_visc = advance_hypervis_euler(dynamics_next,
                                                                      static_forcing,
                                                                      h_grid,
                                                                      v_grid,
                                                                      physics_config,
                                                                      diffusion_config,
                                                                      timestep_config,
                                                                      dims,
                                                                      model)
          if n_split > 0:
            tracer_consist_visc_total = sum_consistency_struct(tracer_consist_visc_total,
                                                               tracer_consist_visc,
                                                               1.0,
                                                               1.0 / timestep_config["dynamics_subcycle"])
          else:
            tracer_consist_visc_total = sum_consistency_struct(tracer_consist_visc,
                                                               tracer_consist_visc,
                                                               1.0 / timestep_config["dynamics_subcycle"],
                                                               0.0)
        if "sponge_layer" in diffusion_config.keys():
          dynamics_next = advance_sponge_euler(dynamics_next,
                                               h_grid,
                                               physics_config,
                                               diffusion_config,
                                               timestep_config,
                                               dims,
                                               model)
      if "d_mass_tracer" in diffusion_config.keys() or "disable_diffusion" in diffusion_config.keys():
        tracer_consist_visc_total = None

      if n_split > 0:
        tracer_consist_dyn_total = sum_consistency_struct(tracer_consist_dyn_total,
                                                          tracer_consist_dyn,
                                                          1.0,
                                                          1.0 / timestep_config["dynamics_subcycle"])
      else:
        tracer_consist_dyn_total = sum_consistency_struct(tracer_consist_dyn,
                                                          tracer_consist_dyn,
                                                          1.0 / timestep_config["dynamics_subcycle"],
                                                          0.0)

      dynamics_state, dynamics_next = dynamics_next, dynamics_state
    tracer_consist_init["d_mass_end"] = 1.0 * dynamics_state["d_mass"]
    tracer_state = advance_tracers(tracer_state,
                                   tracer_consist_dyn_total,
                                   tracer_consist_init,
                                   h_grid,
                                   dims,
                                   physics_config,
                                   diffusion_config,
                                   timestep_config,
                                   model,
                                   tracer_consist_hypervis=tracer_consist_visc_total)
    if do_remap:
      tracer_state = remap_tracers(dynamics_state,
                                   tracer_state,
                                   v_grid,
                                   len(v_grid["hybrid_b_m"]),
                                   model)
    # PPM remap (and any horizontal tracer transport) is conservative on
    # tracer mass but not bit-exact on mixing ratio, so the sum of dry-air
    # species drifts from 1 by tiny element-scale residuals each cycle.
    # Those residuals corrupt R_dry / cp_dry and the splitform PGF chain,
    # seeding a top-layer d_mass oscillation that compounds across cycles.
    # Renormalise to enforce sum(dry_air_species) == 1 pointwise; for
    # non-cam_se models this is a no-op.  Placed here at the coupling-step
    # boundary so any future tracer-transport scheme inherits the fix.
    tracer_state = renormalize_dry_air_species(tracer_state, model)
  return wrap_model_state(dynamics_state,
                          static_forcing,
                          tracer_state)


def validate_custom_configuration(state_in,
                                  h_grid, v_grid,
                                  physics_config,
                                  diffusion_config,
                                  timestep_config,
                                  dims,
                                  model):
  """
  Validate a user-supplied model configuration.

  Intended as an extension point where configuration-specific
  sanity checks can be added.  Currently unimplemented.

  Parameters
  ----------
  state_in : model state dict
      Current model state.
  h_grid : `SpectralElementGrid`
      Horizontal spectral element grid struct.
  v_grid : `dict`
      Vertical grid struct.
  physics_config : `dict`
      Model physics configuration dict.
  diffusion_config : `dict`
      Hyperviscosity and sponge-layer configuration dict.
  timestep_config : `dict`
      Timestep configuration dict.
  dims : frozendict[str, int]
      Grid dimension metadata.
  model : model_info.models
      Dynamical core identifier (from ``model_info.models``).
  """
  pass


def init_simulator(h_grid,
                   v_grid,
                   physics_config,
                   diffusion_config,
                   timestep_config,
                   dims,
                   model):
  """
  Create a generator-based simulator that advances the model forward
  in time indefinitely.

  The returned generator accepts optional physics forcings via
  ``send`` and yields ``(t, state)`` after each physics timestep.

  Parameters
  ----------
  h_grid : `SpectralElementGrid`
      Horizontal spectral element grid struct.
  v_grid : `dict`
      Vertical grid struct containing hybrid coordinate coefficients.
  physics_config : `dict`
      Model physics configuration dict.
  diffusion_config : `dict`
      Hyperviscosity and sponge-layer configuration dict.
  timestep_config : `dict`
      Timestep configuration dict including ``"physics_dt"``.
  dims : frozendict[str, int]
      Grid dimension metadata.
  model : model_info.models
      Dynamical core identifier (from ``model_info.models``).

  Returns
  -------
  simulator : generator
      A Python generator.  Call ``next(sim)`` or
      ``sim.send(physics_forcing)`` to advance by one physics timestep.
      Each iteration yields ``(t, state_n)`` where ``t`` is the
      elapsed simulation time (s) and ``state_n`` is the updated
      model state dict.

  Examples
  --------
  ::

      sim = init_simulator(h_grid, v_grid, physics_config,
                           diffusion_config, timestep_config, dims, model)
      next(sim)  # prime the generator
      t, state = sim.send(None)
  """
  def simulator(state_in, physics_forcing=None):
    state_n = state_in
    t = 0.0
    while True:
      state_n = advance_coupling_step(state_n,
                                      h_grid,
                                      v_grid,
                                      physics_config,
                                      diffusion_config,
                                      timestep_config,
                                      dims,
                                      model,
                                      physics_forcing=physics_forcing)
      assert not check_dynamics_nan(state_n["dynamics"], h_grid, model)
      assert not check_tracers_nan(state_n["tracers"], h_grid, model)
      t += timestep_config["physics_dt"]
      physics_forcing = yield t, state_n
  return simulator
