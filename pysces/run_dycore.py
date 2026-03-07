from .dynamical_cores.time_stepping import (advance_dynamics_euler,
                                            advance_hypervis_euler,
                                            advance_dynamics_ullrich_5stage,
                                            advance_sponge_euler)
from .dynamical_cores.model_state import remap_dynamics
from .time_step import time_step_options
from .dynamical_cores.model_state import (sum_dynamics_series,
                                          advance_tracers,
                                          wrap_model_state,
                                          check_dynamics_nan,
                                          check_tracers_nan,
                                          sum_consistency_struct)
from .dynamical_cores.physics_dynamics_coupling import coupling_types
from .dynamical_cores.tracer_advection.eulerian_spectral import advance_tracers
from .model_info import cam_se_models



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
  dims : `frozendict`
      Grid dimension metadata.
  model : `models`
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

  dynamics_state = state_in["dynamics"]
  tracer_state = state_in["tracers"]
  static_forcing = state_in["static_forcing"]
  dribble_dynamics = (physics_dynamics_coupling == coupling_types.dribble_all or
                      physics_dynamics_coupling == coupling_types.lump_tracers_dribble_dynamics)

  if (physics_dynamics_coupling == coupling_types.lump_tracers_dribble_dynamics) and physics_forcing is not None:
    tracer_state = advance_tracers([tracer_state, physics_forcing["tracers"]],
                                   [1.0, timestep_config["physics_dt"]],
                                   model)

  if physics_dynamics_coupling == coupling_types.lump_all  and physics_forcing is not None:
    dynamics_state = sum_dynamics_series([dynamics_state, physics_forcing["dynamics"]],
                                         [1.0, timestep_config["physics_dt"]],
                                         model)
    tracer_state = advance_tracers([tracer_state, physics_forcing["tracers"]],
                                   [1.0, timestep_config["physics_dt"]],
                                   model)
  for q_split in range(timestep_config["tracer_subcycle"]):
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
    if physics_dynamics_coupling == coupling_types.dribble_all  and physics_forcing is not None:
      tracer_state = advance_tracers([tracer_state, physics_forcing["tracers"]],
                                     [1.0, timestep_config["physics_dt"]],
                                     model)

    for n_split in range(timestep_config["dynamics_subcycle"]):
      if model in cam_se_models:
        moisture_species = tracer_state["moisture_species"]
        dry_air_species = tracer_state["dry_air_species"]
      else:
        moisture_species = None
        dry_air_species = None
      if timestep_config["dynamics"]["step_type"] == time_step_options.Euler:
        dynamics_next, tracer_consist_dyn = advance_dynamics_euler(dynamics_state,
                                                                   static_forcing,
                                                                   h_grid,
                                                                   v_grid,
                                                                   physics_config,
                                                                   timestep_config,
                                                                   dims,
                                                                   model,
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
                                                                            model,
                                                                            moisture_species=moisture_species,
                                                                            dry_air_species=dry_air_species)
      else:
        raise ValueError("Unknown dynamics timestep type")
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
        if "enable_sponge_layer" in diffusion_config.keys():
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
      assert not check_dynamics_nan(dynamics_next, h_grid, model)
      assert not check_tracers_nan(tracer_state, h_grid, model)

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
  dims : `frozendict`
      Grid dimension metadata.
  model : `models`
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
  dims : `frozendict`
      Grid dimension metadata.
  model : `models`
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
      t += timestep_config["physics_dt"]
      physics_forcing = yield t, state_n
  return simulator
