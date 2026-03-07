from ..config import jit, DEBUG, jnp
from .explicit_terms import eval_explicit_terms
from .model_state import project_model_state, sum_state_series, extract_average_dyn, extract_average_hypervis, sum_avg_struct
from .hyperviscosity import eval_hypervis_quasi_uniform, eval_hypervis_variable_resolution
from ..time_step import time_step_options, stability_info
from ..operations_2d.operators import horizontal_divergence, horizontal_gradient
from ..operations_2d.local_assembly import project_scalar, minmax_scalar
from ..horizontal_grid import eval_cfl
from frozendict import frozendict
from functools import partial


@partial(jit, static_argnames=["dims", "timestep_config"])
def advance_step_euler(state_in,
                       grid,
                       physics_config,
                       timestep_config,
                       dims):
  """
  Advance the shallow-water model by one dynamics timestep using forward Euler.

  Parameters
  ----------
  state_in : dict[str, Array]
      Shallow-water model state as returned by ``wrap_model_state``.
  grid : SpectralElementGrid
      Horizontal spectral element grid.
  physics_config : dict[str, Any]
      Physical constants (e.g. ``radius_earth``, ``gravity``).
  timestep_config : frozendict
      Time-step configuration from ``init_timestep_config``.
  dims : dict[str, int]
      Grid dimension parameters.

  Returns
  -------
  state_out : dict[str, Array]
      Updated shallow-water model state after one Euler step.
  tracer_consist : dict[str, Array]
      Tracer-consistency struct with key ``"u_d_mass_avg"``.
  """

  state_tend = eval_explicit_terms(state_in,
                                   grid,
                                   physics_config)
  state_tend_c0 = project_model_state(state_tend, grid, dims)
  return sum_state_series([state_in, state_tend_c0], [1.0, timestep_config["dynamics"]["dt"]]), extract_average(state_tend_c0)


@partial(jit, static_argnames=["dims", "timestep_config"])
def advance_step_ssprk3(state0,
                        grid,
                        physics_config,
                        timestep_config,
                        dims):
  """
  Advance the shallow-water model by one dynamics timestep using strong-stability-preserving
  Runge-Kutta 3 (SSPRK3).

  Parameters
  ----------
  state0 : dict[str, Array]
      Shallow-water model state at the beginning of the timestep.
  grid : SpectralElementGrid
      Horizontal spectral element grid.
  physics_config : dict[str, Any]
      Physical constants (e.g. ``radius_earth``, ``gravity``).
  timestep_config : frozendict
      Time-step configuration from ``init_timestep_config``.
  dims : dict[str, int]
      Grid dimension parameters.

  Returns
  -------
  state_out : dict[str, Array]
      Updated shallow-water model state after one SSPRK3 step.
  avg : dict[str, Array]
      Tracer-consistency struct with the time-averaged
      thickness-weighted wind ``"u_d_mass_avg"``.
  """
  dt = timestep_config["dynamics"]["dt"]

  tend = eval_explicit_terms(state0, grid, physics_config)
  tend_c0 = project_model_state(tend, grid, dims)
  avg = extract_average_dyn(tend_c0)
  state1 = sum_state_series([state0, tend_c0], [1.0, dt])
  tend = eval_explicit_terms(state1, grid, physics_config)
  tend_c0 = project_model_state(tend, grid, dims)
  avg = sum_avg_struct(avg, extract_average_dyn(tend_c0), 1.0 / 6.0, 1.0 / 6.0)
  state2 = sum_state_series([state0, state1, tend_c0],
                            [3.0 / 4.0,
                             1.0 / 4.0,
                             1.0 / 4.0 * dt])
  tend = eval_explicit_terms(state2, grid, physics_config)
  tend_c0 = project_model_state(tend, grid, dims)
  avg = sum_avg_struct(avg, extract_average_dyn(tend_c0), 1.0, 2.0 / 3.0)
  return sum_state_series([state0, state2, tend_c0],
                          [1.0 / 3.0,
                           2.0 / 3.0,
                           2.0 / 3.0 * dt]), avg


@partial(jit, static_argnames=["dims", "timestep_config"])
def advance_hypervis_euler(state_in,
                           grid,
                           physics_config,
                           diffusion_config,
                           timestep_config,
                           dims):
  """
  Apply hyperviscosity to the shallow-water model state using forward Euler subcycling.

  Parameters
  ----------
  state_in : dict[str, Array]
      Shallow-water model state to be diffused.
  grid : SpectralElementGrid
      Horizontal spectral element grid.
  physics_config : dict[str, Any]
      Physical constants (e.g. ``radius_earth``).
  diffusion_config : dict[str, Any]
      Hyperviscosity configuration; must contain ``"nu"`` and ``"nu_d_mass"``.
  timestep_config : frozendict
      Time-step configuration; must contain ``"hypervis_subcycle"`` and
      ``"hyperviscosity"`` (with ``"dt"``).
  dims : dict[str, int]
      Grid dimension parameters.

  Returns
  -------
  state_out : dict[str, Array]
      Diffused shallow-water model state.
  avg : dict[str, Array]
      Tracer-consistency struct accumulated over hyperviscosity subcycles,
      with keys ``"d_mass_hypervis_avg"`` and ``"d_mass_hypervis_tend"``.
  """
  next_step = state_in
  for k in range(timestep_config["hypervis_subcycle"]):
    if "tensor_hypervis" in diffusion_config.keys():
      hvis_tend = eval_hypervis_variable_resolution(next_step, grid, physics_config, diffusion_config, dims)
    else:
      hvis_tend = eval_hypervis_quasi_uniform(next_step, grid, physics_config, diffusion_config, dims)
    if k == 0:
      avg = extract_average_hypervis(next_step, hvis_tend, diffusion_config)
      avg = sum_avg_struct(avg, avg, 0.0, 1.0 / timestep_config["hypervis_subcycle"])
    else:
      avg = sum_avg_struct(avg, extract_average_hypervis(next_step, hvis_tend, diffusion_config), 1.0, 1.0 / timestep_config["hypervis_subcycle"])
    next_step = sum_state_series([next_step, hvis_tend], [1.0, -timestep_config["hyperviscosity"]["dt"]])
  return next_step, avg


def init_timestep_config(dt_coupling,
                         h_grid,
                         dims,
                         physics_config,
                         diffusion_config,
                         hypervis_tstep_type=time_step_options.Euler,
                         dynamics_tstep_type=time_step_options.SSPRK3,
                         dyn_steps_per_coupling=-1,
                         hypervis_steps_per_dyn=3,
                         sphere=True):
  """
  Compute the shallow-water timestep configuration from CFL stability estimates.

  Parameters
  ----------
  dt_coupling : float
      The physics coupling timestep in seconds.
  h_grid : SpectralElementGrid
      Horizontal spectral element grid.
  dims : dict[str, int]
      Grid dimension parameters.
  physics_config : dict[str, Any]
      Physical constants including ``radius_earth``.
  diffusion_config : dict[str, Any]
      Hyperviscosity configuration.
  hypervis_tstep_type : time_step_options, optional
      Time integration scheme for hyperviscosity (default: Euler).
  dynamics_tstep_type : time_step_options, optional
      Time integration scheme for dynamics (default: SSPRK3).
  dyn_steps_per_coupling : int, optional
      Override for the number of dynamics subcycles per coupling step.
      Negative values use the CFL-derived estimate.
  hypervis_steps_per_dyn : int, optional
      Minimum number of hyperviscosity subcycles per dynamics step (default: 3).
  sphere : bool, optional
      Whether to use spherical CFL estimates (default: True).

  Returns
  -------
  timestep_config : frozendict
      Nested frozen dict containing ``"dynamics"``, ``"hyperviscosity"``,
      ``"tracer_advection"``, ``"dt_coupling"``,
      ``"dynamics_subcycle"``, and ``"hypervis_subcycle"`` entries.
  """
  cfl_info, _ = eval_cfl(h_grid, physics_config["radius_earth"], diffusion_config, dims, sphere=sphere)
  hypervisc_S = stability_info[hypervis_tstep_type]
  dynamics_S = stability_info[dynamics_tstep_type]
  dt_rkssp_stability = cfl_info["dt_rkssp_euler"]
  dt_hypervis_scalar = cfl_info["dt_hypervis_scalar"]
  dt_hypervis_vort = cfl_info["dt_hypervis_vort"]
  dt_hypervis_div = cfl_info["dt_hypervis_div"]

  # determine n_split
  max_dt_dynamics = dynamics_S * dt_rkssp_stability
  dynamics_subcycle = max(int(dt_coupling / max_dt_dynamics) + 1, dyn_steps_per_coupling)
  dt_dynamics = dt_coupling / dynamics_subcycle

  # determine hv_split
  max_dt_hypervis_scalar = hypervisc_S * dt_hypervis_scalar
  max_dt_hypervis_vort = hypervisc_S * dt_hypervis_vort
  max_dt_hypervis_div = hypervisc_S * dt_hypervis_div
  # calculate tightest constraint on HV stability
  max_dt_hypervis = min([max_dt_hypervis_scalar,
                         max_dt_hypervis_vort,
                         max_dt_hypervis_div])
  hypervisc_subcycle = max(int(dt_dynamics / max_dt_hypervis) + 1, hypervis_steps_per_dyn)
  dt_hypervis = dt_dynamics / hypervisc_subcycle

  if DEBUG:
    print("CFL estimates:")
    # print(f"SSP preservation (120m/s) RKSSP euler step dt  < S * {rkssp_euler_stability}s")
    print(f"Stability: RKSSP preservation (120m/s) dt_dyn = {dt_dynamics}s  < {max_dt_dynamics}s")
    #  dt < S  1 / nu * norm_jac_inv_hypervis
    print(f"Stability: nu_dpi  hyperviscosity dt = {dt_hypervis}s < {max_dt_hypervis_scalar}s")
    print(f"Stability: nu_vor hyperviscosity dt = {dt_hypervis}s < {max_dt_hypervis_vort}s")
    print(f"Stability: nu_div hyperviscosity dt = {dt_hypervis}s < {max_dt_hypervis_div}s")

  return frozendict(dynamics=frozendict(step_type=dynamics_tstep_type,
                                        dt=dt_dynamics),
                    hyperviscosity=frozendict(step_type=hypervis_tstep_type,
                                              dt=dt_hypervis),
                    tracer_advection=frozendict(step_type=time_step_options.SSPRK3,
                                                dt=dt_dynamics),
                    dt_coupling=dt_coupling,
                    dynamics_subcycle=dynamics_subcycle,
                    hypervis_subcycle=hypervisc_subcycle)
