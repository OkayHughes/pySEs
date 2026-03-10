from .._config import get_backend as _get_backend
from .time_stepping import advance_step_euler, advance_step_ssprk3, advance_hypervis_euler
from .tracers import advance_tracers_shallow_water
from ..dynamical_cores.time_step import time_step_options
from sys import stdout
_be = _get_backend()
jnp = _be.np
versatile_assert = _be.assert_true
is_main_proc = _be.is_main_proc


def simulate_shallow_water(end_time,
                           state_in,
                           grid,
                           physics_config,
                           diffusion_config,
                           timestep_config,
                           dims,
                           diffusion=True,
                           tracers_in=None):
  """
  Run the shallow-water model forward in time from ``0`` to ``end_time``.

  Parameters
  ----------
  end_time : float
      Total simulation duration in seconds.
  state_in : dict[str, Array]
      Initial shallow-water model state, as returned by ``wrap_model_state``.
  grid : SpectralElementGrid
      Horizontal spectral element grid.
  physics_config : dict[str, Any]
      Physical constants (e.g. ``radius_earth``, ``gravity``).
  diffusion_config : dict[str, Any]
      Hyperviscosity configuration.
  timestep_config : frozendict
      Time-step configuration from ``init_timestep_config``.
  dims : dict[str, int]
      Grid dimension parameters.
  diffusion : bool, optional
      Whether to apply hyperviscosity after each dynamics step (default: True).
  tracers_in : dict[str, Array] or None, optional
      Initial named passive tracer fields. If ``None``, tracers are not
      advanced.

  Returns
  -------
  result : dict[str, Any]
      Dict with key ``"dynamics"`` containing the final model state, and
      optionally ``"tracers"`` if ``tracers_in`` was provided.
  """
  state_n = state_in
  if tracers_in is not None:
    tracers_n = tracers_in
  t = 0.0
  times = jnp.arange(0.0, end_time, timestep_config["dt_coupling"])
  k = 0
  for t in times:
    if is_main_proc:
      print(f"{k/len(times-1)*100}%")
      stdout.flush()
    for dyn_subcycle_idx in range(timestep_config["dynamics_subcycle"]):
      step_type = timestep_config["dynamics"]["step_type"]
      tracer_init_struct = {"d_mass_init": state_n["h"]}
      if step_type == time_step_options.SSPRK3:
        state_tmp, tracer_consist_dyn = advance_step_ssprk3(state_n, grid, physics_config, timestep_config, dims)
      elif step_type == time_step_options.Euler:
        state_tmp, tracer_consist_dyn = advance_step_euler(state_n, grid, physics_config, timestep_config, dims)
      if diffusion:
        state_np1, tracer_consist_hypervis = advance_hypervis_euler(state_tmp, grid,
                                                                    physics_config,
                                                                    diffusion_config,
                                                                    timestep_config,
                                                                    dims)
      else:
        state_np1 = state_tmp
        tracer_consist_hypervis = None
      tracer_init_struct["d_mass_end"] = state_n["h"]
      if tracers_in is not None:
        tracers_n = advance_tracers_shallow_water(tracers_n,
                                                  tracer_consist_dyn,
                                                  tracer_init_struct,
                                                  grid,
                                                  dims,
                                                  physics_config,
                                                  diffusion_config,
                                                  timestep_config,
                                                  tracer_consist_hypervis=tracer_consist_hypervis)

      state_n, state_np1 = state_np1, state_n

      versatile_assert(jnp.logical_not(jnp.any(jnp.isnan(state_n["horizontal_wind"]))))
      versatile_assert(jnp.logical_not(jnp.any(jnp.isnan(state_n["h"]))))
    k += 1
  ret = {"dynamics": state_n}
  if tracers_in is not None:
    ret["tracers"] = tracers_n
  return ret
