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
jnp = _be.np
DEBUG = _be.debug



def _advance_coupling_step(state_in,
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
      # Dribble the tracer forcing per tracer sub-step using the sub-step dt
      # (as the dribble-dynamics branch above does), so the increments sum to
      # one physics_dt over the tracer subcycle rather than
      # tracer_subcycle * physics_dt.
      tracer_state = sum_tracers_series([tracer_state, physics_forcing["tracers"]],
                                        [1.0, timestep_config["tracer_advection"]["dt"]],
                                        model)

    # One adiabatic dynamics sub-step: convert to the _stable (theta_d) variant
    # for the explicit step (skew-symmetric / Exner form), advance, restore the
    # external T interface, then apply hyperviscosity and the sponge.  Returns
    # the per-sub-step consistency structs; ``tracer_consist_visc`` is None for
    # configs without an Euler-stepped hyperviscosity contribution (including a
    # non-Euler hypervis stepper, d_mass_tracer, or disabled diffusion), so a
    # bound value always reaches advance_tracers.  The old dynamics_next/state
    # swap is subsumed by the scan carry.
    dyn_scale = 1.0 / timestep_config["dynamics_subcycle"]

    def _one_dyn_subcycle(dynamics_state):
      if model in cam_se_models:
        moisture_species = tracer_state["moisture_species"]
        dry_air_species = tracer_state["dry_air_species"]
      else:
        moisture_species = None
        dry_air_species = None
      if model in cam_se_models and model not in cam_se_stable_models:
        dynamics_step, step_model = se_T_to_theta_d_d_mass(
            dynamics_state, v_grid, physics_config, model)
      else:
        dynamics_step = dynamics_state
        step_model = model
      if timestep_config["dynamics"]["step_type"] == time_step_options.Euler:
        dynamics_next, tracer_consist_dyn = advance_dynamics_euler(dynamics_step,
                                                                   static_forcing,
                                                                   h_grid,
                                                                   v_grid,
                                                                   physics_config,
                                                                   timestep_config,
                                                                   dims,
                                                                   step_model,
                                                                   moisture_species=moisture_species,
                                                                   dry_air_species=dry_air_species)
      elif timestep_config["dynamics"]["step_type"] in (time_step_options.RK3_5STAGE, time_step_options.RK3_5STAGE_HEVI):
        dynamics_next, tracer_consist_dyn = advance_dynamics_ullrich_5stage(dynamics_step,
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
      if model in cam_se_models and model not in cam_se_stable_models:
        dynamics_next, _ = se_theta_d_d_mass_to_T(
            dynamics_next, v_grid, physics_config, step_model)
      tracer_consist_visc = None
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
        if "sponge_layer" in diffusion_config.keys():
          dynamics_next = advance_sponge_euler(dynamics_next,
                                               h_grid,
                                               physics_config,
                                               diffusion_config,
                                               timestep_config,
                                               dims,
                                               model)
      if "d_mass_tracer" in diffusion_config.keys() or "disable_diffusion" in diffusion_config.keys():
        tracer_consist_visc = None
      return dynamics_next, tracer_consist_dyn, tracer_consist_visc

    # First sub-step seeds the accumulators (net effect: each total is dyn_scale
    # times the sum over sub-steps); the rest run through ``scan`` so the RK/HEVI
    # body compiles once instead of unrolling ``dynamics_subcycle`` copies.
    dynamics_state, tracer_consist_dyn, tracer_consist_visc = _one_dyn_subcycle(dynamics_state)
    tracer_consist_dyn_total = sum_consistency_struct(tracer_consist_dyn, tracer_consist_dyn,
                                                      dyn_scale, 0.0)
    if tracer_consist_visc is None:
      tracer_consist_visc_total = None
    else:
      tracer_consist_visc_total = sum_consistency_struct(tracer_consist_visc, tracer_consist_visc,
                                                         dyn_scale, 0.0)

    if timestep_config["dynamics_subcycle"] > 1:
      def _dyn_step(carry, _):
        dynamics_state, tracer_consist_dyn_total, tracer_consist_visc_total = carry
        dynamics_state, tracer_consist_dyn, tracer_consist_visc = _one_dyn_subcycle(dynamics_state)
        tracer_consist_dyn_total = sum_consistency_struct(tracer_consist_dyn_total,
                                                          tracer_consist_dyn, 1.0, dyn_scale)
        if tracer_consist_visc_total is not None:
          tracer_consist_visc_total = sum_consistency_struct(tracer_consist_visc_total,
                                                             tracer_consist_visc, 1.0, dyn_scale)
        return (dynamics_state, tracer_consist_dyn_total, tracer_consist_visc_total), None

      (dynamics_state, tracer_consist_dyn_total, tracer_consist_visc_total), _ = _be.scan(
          _dyn_step,
          (dynamics_state, tracer_consist_dyn_total, tracer_consist_visc_total),
          jnp.arange(timestep_config["dynamics_subcycle"] - 1))
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
    # Vertical remap (and any horizontal tracer transport) may not conserve total sum
    # of dry air species to be one, so we renormalize.
    # Error is relegated to N2 species if necessary.
    # This matches behavior in CAM-SE (to my understanding).
    tracer_state = renormalize_dry_air_species(tracer_state, model)
  return wrap_model_state(dynamics_state,
                          static_forcing,
                          tracer_state)


# Public jitted entry point.  Intentionally *not* buffer-donating: several
# callers (e.g. the instrumented-coupling ablation tests) invoke it more than
# once on the same input state, which donation would invalidate.  The
# production time-stepping loop opts into donation explicitly via
# ``init_simulator(..., donate_state=True)``.
_COUPLING_STATIC_ARGNAMES = ["model", "dims", "timestep_config"]
advance_coupling_step = partial(jit, static_argnames=_COUPLING_STATIC_ARGNAMES)(_advance_coupling_step)


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
                   model,
                   nan_check_interval=1,
                   donate_state=False):
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
  # Opt-in buffer donation for the production loop: the loop reassigns
  # ``state_n`` every step and never reuses the previous state, so XLA may reuse
  # the input allocation for the output (halving peak state memory).  Off by
  # default because a consumer that *retains* a yielded state across a step, or
  # reuses the initial ``state_in`` afterwards, would hit a donated (deleted)
  # buffer.  NumPy/torch backends ignore ``donate_argnames``.
  if donate_state:
    step_fn = _be.jit(_advance_coupling_step,
                      static_argnames=_COUPLING_STATIC_ARGNAMES,
                      donate_argnames=["state_in"])
  else:
    step_fn = advance_coupling_step

  def simulator(state_in, physics_forcing=None):
    state_n = state_in
    t = 0.0
    step_idx = 0
    while True:
      state_n = step_fn(state_n,
                        h_grid,
                        v_grid,
                        physics_config,
                        diffusion_config,
                        timestep_config,
                        dims,
                        model,
                        physics_forcing=physics_forcing)
      # The NaN scan materializes a device scalar (host sync).  Amortize it over
      # ``nan_check_interval`` steps for GPU throughput; the default (1) keeps
      # the every-step guard callers rely on to detect blow-ups.
      if nan_check_interval <= 1 or step_idx % nan_check_interval == 0:
        assert not check_dynamics_nan(state_n["dynamics"], h_grid, model)
        assert not check_tracers_nan(state_n["tracers"], h_grid, model)
      t += timestep_config["physics_dt"]
      step_idx += 1
      physics_forcing = yield t, state_n
  return simulator
