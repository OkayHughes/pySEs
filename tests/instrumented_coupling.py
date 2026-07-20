"""Instrumented copy of :func:`advance_coupling_step` for ablation testing.

:func:`advance_coupling_step_instrumented` reproduces the production coupling
step from :mod:`pyses.dynamical_cores.run_dycore` line-for-line, but wraps each
of the eight physically-distinct sub-steps in an independent ``enable_*`` flag:

1. ``enable_physics_dynamics_forcing`` — apply the physics tendency to the
   dynamical variables (wind / temperature / mass).
2. ``enable_physics_moisture_forcing`` — apply the physics tendency to the
   tracer (moisture) variables.
3. ``enable_adiabatic_dynamics`` — advance the adiabatic dynamical core.
4. ``enable_sponge_layer`` — apply the sponge-layer damping.
5. ``enable_hyperviscosity`` — apply hyperviscosity.
6. ``enable_remap_dynamics`` — vertically remap the dynamics back to the
   reference levels.
7. ``enable_tracer_transport`` — advect the tracers.
8. ``enable_remap_tracers`` — vertically remap the tracers.

All flags default to ``True``; with every flag enabled the routine is a
faithful reproduction of :func:`advance_coupling_step` and returns identical
results for the same model configuration (this is the property exercised by
``test_instrumented_coupling_step``). The flags exist so a single sub-step can
be ablated in isolation when diagnosing a misbehaving configuration.

Disabling a sub-step makes it the identity on its target state. Two coupling
caveats follow from the way the tracer solver consumes dynamics-derived
consistency fluxes:

* With ``enable_adiabatic_dynamics=False`` the dynamics state is frozen and the
  tracer-consistency mass flux handed to the tracer solver is zero, so tracers
  are not transported by (absent) dynamics.
* The hyperviscosity and sponge sub-steps act on the freshly-advanced state but
  are gated independently, so they remain togglable whether or not the
  adiabatic advance ran.
* With ``enable_hyperviscosity=False`` on a config that carries neither
  ``d_mass_tracer`` nor ``disable_diffusion``, the tracer solver is handed a
  config copy marked ``disable_diffusion`` so its fallback branch (which the
  missing hypervis-consistency struct forces) takes the neutral ones-scale
  path instead of KeyError-ing — making the hyperviscosity toggle a standalone
  ablation even with tracer transport enabled.
"""

from functools import partial

from pyses._config import get_backend as _get_backend
from pyses.dynamical_cores.time_stepping import (advance_dynamics_euler,
                                                 advance_hypervis_euler,
                                                 advance_dynamics_ullrich_5stage,
                                                 advance_sponge_euler)
from pyses.dynamical_cores.model_state import (remap_dynamics,
                                               remap_tracers,
                                               sum_dynamics_series,
                                               sum_tracers_series,
                                               wrap_model_state,
                                               sum_consistency_struct,
                                               se_T_to_theta_d_d_mass,
                                               se_theta_d_d_mass_to_T,
                                               renormalize_dry_air_species,
                                               wrap_tracer_consist_dynamics)
from pyses.dynamical_cores.time_step import time_step_options
from pyses.dynamical_cores.physics_dynamics_coupling import coupling_types
from pyses.dynamical_cores.tracer_advection.eulerian_spectral import advance_tracers
from pyses.dynamical_cores.model_info import cam_se_models, cam_se_stable_models

_be = _get_backend()
jit = _be.jit
bnp = _be.np

_ENABLE_FLAGS = (
    "enable_physics_dynamics_forcing",
    "enable_physics_moisture_forcing",
    "enable_adiabatic_dynamics",
    "enable_sponge_layer",
    "enable_hyperviscosity",
    "enable_remap_dynamics",
    "enable_tracer_transport",
    "enable_remap_tracers",
)


@partial(jit, static_argnames=["model", "dims", "timestep_config", *_ENABLE_FLAGS])
def advance_coupling_step_instrumented(state_in,
                                       h_grid,
                                       v_grid,
                                       physics_config,
                                       diffusion_config,
                                       timestep_config,
                                       dims,
                                       model,
                                       physics_forcing=None,
                                       *,
                                       enable_physics_dynamics_forcing=True,
                                       enable_physics_moisture_forcing=True,
                                       enable_adiabatic_dynamics=True,
                                       enable_sponge_layer=True,
                                       enable_hyperviscosity=True,
                                       enable_remap_dynamics=True,
                                       enable_tracer_transport=True,
                                       enable_remap_tracers=True):
  """Advance one physics timestep with independently togglable sub-steps.

  See the module docstring for the meaning of each ``enable_*`` flag. With all
  flags ``True`` this matches :func:`pyses.dynamical_cores.run_dycore.\
advance_coupling_step` exactly. Arguments otherwise mirror that function.
  """
  physics_dynamics_coupling = timestep_config["physics_dynamics_coupling"]
  do_remap = v_grid["hybrid_a_m"].shape[0] > 1

  dynamics_state = state_in["dynamics"]
  tracer_state = state_in["tracers"]
  static_forcing = state_in["static_forcing"]
  dribble_dynamics = (physics_dynamics_coupling == coupling_types.dribble_all or
                      physics_dynamics_coupling == coupling_types.lump_tracers_dribble_dynamics)

  # (2) physics forcing of moisture variables (lump_tracers_dribble_dynamics).
  if ((physics_dynamics_coupling == coupling_types.lump_tracers_dribble_dynamics)
      and physics_forcing is not None and enable_physics_moisture_forcing):
    tracer_state = sum_tracers_series([tracer_state, physics_forcing["tracers"]],
                                      [1.0, timestep_config["physics_dt"]],
                                      model)

  # (1)/(2) physics forcing of dynamics and moisture variables (lump_all).
  if physics_dynamics_coupling == coupling_types.lump_all and physics_forcing is not None:
    if enable_physics_dynamics_forcing:
      dynamics_state = sum_dynamics_series([dynamics_state, physics_forcing["dynamics"]],
                                           [1.0, timestep_config["physics_dt"]],
                                           model)
    if enable_physics_moisture_forcing:
      tracer_state = sum_tracers_series([tracer_state, physics_forcing["tracers"]],
                                        [1.0, timestep_config["physics_dt"]],
                                        model)

  for q_split in range(timestep_config["tracer_subcycle"]):
    # (6) vertical remap of dynamics.
    if do_remap and enable_remap_dynamics:
      dynamics_state = remap_dynamics(dynamics_state,
                                      state_in["static_forcing"],
                                      v_grid,
                                      physics_config,
                                      len(v_grid["hybrid_b_m"]),
                                      model)
    tracer_consist_init = {"d_mass_init": 1.0 * dynamics_state["d_mass"]}
    # (1) physics forcing of dynamics variables (dribble).
    if dribble_dynamics and physics_forcing is not None and enable_physics_dynamics_forcing:
      dynamics_state = sum_dynamics_series([dynamics_state, physics_forcing["dynamics"]],
                                           [1.0, timestep_config["tracer_advection"]["dt"]],
                                           model)
    # (2) physics forcing of moisture variables (dribble_all). Dribbled per
    # tracer sub-step using the sub-step dt (matching the dribble-dynamics line
    # above), so the increments sum to ``physics_dt`` over the tracer subcycle
    # rather than ``tracer_subcycle * physics_dt``.
    if (physics_dynamics_coupling == coupling_types.dribble_all
        and physics_forcing is not None and enable_physics_moisture_forcing):
      tracer_state = sum_tracers_series([tracer_state, physics_forcing["tracers"]],
                                        [1.0, timestep_config["tracer_advection"]["dt"]],
                                        model)

    tracer_consist_dyn_total = None
    tracer_consist_visc_total = None
    for n_split in range(timestep_config["dynamics_subcycle"]):
      # (3) adiabatic dynamics advance.
      if enable_adiabatic_dynamics:
        if model in cam_se_models:
          moisture_species = tracer_state["moisture_species"]
          dry_air_species = tracer_state["dry_air_species"]
        else:
          moisture_species = None
          dry_air_species = None
        # Advance in the _stable (theta_d) variant, then restore the T-based
        # interface — identical to the production step.
        if model in cam_se_models and model not in cam_se_stable_models:
          dynamics_state, step_model = se_T_to_theta_d_d_mass(
              dynamics_state, v_grid, physics_config, model)
        else:
          step_model = model
        if timestep_config["dynamics"]["step_type"] == time_step_options.Euler:
          dynamics_next, tracer_consist_dyn = advance_dynamics_euler(
              dynamics_state, static_forcing, h_grid, v_grid, physics_config,
              timestep_config, dims, step_model,
              moisture_species=moisture_species, dry_air_species=dry_air_species)
        elif timestep_config["dynamics"]["step_type"] in (time_step_options.RK3_5STAGE,
                                                          time_step_options.RK3_5STAGE_HEVI):
          dynamics_next, tracer_consist_dyn = advance_dynamics_ullrich_5stage(
              dynamics_state, static_forcing, h_grid, v_grid, physics_config,
              timestep_config, dims, step_model,
              moisture_species=moisture_species, dry_air_species=dry_air_species)
        else:
          raise ValueError("Unknown dynamics timestep type")
        if model in cam_se_models and model not in cam_se_stable_models:
          dynamics_next, _ = se_theta_d_d_mass_to_T(
              dynamics_next, v_grid, physics_config, step_model)
          dynamics_state, _ = se_theta_d_d_mass_to_T(
              dynamics_state, v_grid, physics_config, step_model)
      else:
        # No advance: freeze dynamics and hand the tracer solver a zero mass
        # flux so absent dynamics transports no tracer.
        dynamics_next = dynamics_state
        tracer_consist_dyn = wrap_tracer_consist_dynamics(
            bnp.zeros(dynamics_state["horizontal_wind"].shape))

      # (5) hyperviscosity (acts on the advanced state).
      if enable_hyperviscosity and "disable_diffusion" not in diffusion_config.keys():
        if timestep_config["hyperviscosity"]["step_type"] == time_step_options.Euler:
          dynamics_next, tracer_consist_visc = advance_hypervis_euler(
              dynamics_next, static_forcing, h_grid, v_grid, physics_config,
              diffusion_config, timestep_config, dims, model)
          if tracer_consist_visc_total is None:
            tracer_consist_visc_total = sum_consistency_struct(
                tracer_consist_visc, tracer_consist_visc,
                1.0 / timestep_config["dynamics_subcycle"], 0.0)
          else:
            tracer_consist_visc_total = sum_consistency_struct(
                tracer_consist_visc_total, tracer_consist_visc,
                1.0, 1.0 / timestep_config["dynamics_subcycle"])

      # (4) sponge layer (acts on the advanced state).
      if enable_sponge_layer and "sponge_layer" in diffusion_config.keys():
        dynamics_next = advance_sponge_euler(dynamics_next,
                                             h_grid,
                                             physics_config,
                                             diffusion_config,
                                             timestep_config,
                                             dims,
                                             model)

      # Dynamics tracer-consistency accumulation (independent of hypervis).
      if tracer_consist_dyn_total is None:
        tracer_consist_dyn_total = sum_consistency_struct(
            tracer_consist_dyn, tracer_consist_dyn,
            1.0 / timestep_config["dynamics_subcycle"], 0.0)
      else:
        tracer_consist_dyn_total = sum_consistency_struct(
            tracer_consist_dyn_total, tracer_consist_dyn,
            1.0, 1.0 / timestep_config["dynamics_subcycle"])

      dynamics_state, dynamics_next = dynamics_next, dynamics_state

    # Mirror the production step: these configs carry no hypervis consistency.
    if "d_mass_tracer" in diffusion_config.keys() or "disable_diffusion" in diffusion_config.keys():
      tracer_consist_visc_total = None

    tracer_consist_init["d_mass_end"] = 1.0 * dynamics_state["d_mass"]
    # (7) tracer transport.
    if enable_tracer_transport:
      tracer_diffusion_config = diffusion_config
      if (not enable_hyperviscosity
          and "d_mass_tracer" not in diffusion_config
          and "disable_diffusion" not in diffusion_config):
        # Production's invariant: the tracer solver receives a None
        # hypervis-consistency struct only alongside a config carrying
        # ``d_mass_tracer`` or ``disable_diffusion`` (its fallback branch
        # reads one of them). Ablating hyperviscosity breaks that
        # invariant, so hand the tracer solver a copy marked
        # ``disable_diffusion``: it then takes the neutral ones-scale
        # path, and tracer hyperviscosity is already skipped by the
        # struct-is-None guard — exactly "hyperviscosity is the
        # identity" semantics.
        tracer_diffusion_config = dict(diffusion_config)
        tracer_diffusion_config["disable_diffusion"] = True
      tracer_state = advance_tracers(tracer_state,
                                     tracer_consist_dyn_total,
                                     tracer_consist_init,
                                     h_grid,
                                     dims,
                                     physics_config,
                                     tracer_diffusion_config,
                                     timestep_config,
                                     model,
                                     tracer_consist_hypervis=tracer_consist_visc_total)
    # (8) vertical remap of tracers.
    if do_remap and enable_remap_tracers:
      tracer_state = remap_tracers(dynamics_state,
                                   tracer_state,
                                   v_grid,
                                   len(v_grid["hybrid_b_m"]),
                                   model)
    tracer_state = renormalize_dry_air_species(tracer_state, model)
  return wrap_model_state(dynamics_state,
                          static_forcing,
                          tracer_state)
