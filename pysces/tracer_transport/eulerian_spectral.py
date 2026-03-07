from ..config import jnp, jit, vmap_1d_apply
from ..operations_2d.local_assembly import minmax_scalar, project_scalar
from ..dynamical_cores.operators_3d import horizontal_divergence_3d
from ..dynamical_cores.hyperviscosity import scalar_harmonic_3d
from ..operations_2d.limiters import full_limiter

from functools import partial


@partial(jit, static_argnames=["dims", "max"])
def minmax_scalar_3d(scalar,
                     h_grid,
                     dims,
                     max=True):
  """
  Compute the element-local min or max of a 3-D scalar field via DSS.

  Applies ``minmax_scalar`` level-by-level, gathering the global minimum
  (or maximum) at each GLL node over all elements sharing that node.

  Parameters
  ----------
  scalar : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      3-D scalar field on the GLL grid.
  h_grid : SpectralElementGrid
      Horizontal spectral element grid.
  dims : dict[str, int]
      Grid dimension parameters.
  max : bool, optional
      If True compute the maximum; if False compute the minimum
      (default: True).

  Returns
  -------
  result : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Field with each node replaced by the DSS-global min or max.
  """
  sph_op = partial(minmax_scalar, grid=h_grid, max=max, dims=dims)
  return vmap_1d_apply(sph_op, scalar, -1, -1)


@partial(jit, static_argnames=["dims"])
def project_tracer_3d(scalar,
                      h_grid,
                      dims):
  """
  Project a 3-D scalar tracer onto the continuous spectral element space (DSS).

  Applies ``project_scalar`` level-by-level via ``vmap_1d_apply``.

  Parameters
  ----------
  scalar : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      3-D discontinuous tracer field.
  h_grid : SpectralElementGrid
      Horizontal spectral element grid.
  dims : dict[str, int]
      Grid dimension parameters.

  Returns
  -------
  scalar_cont : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Globally continuous (C0) tracer field.
  """
  sph_op = partial(project_scalar, grid=h_grid, dims=dims)
  return vmap_1d_apply(sph_op, scalar, -1, -1)


@partial(jit, static_argnames=["dims"])
def calc_minmax(tracers, grid, dims):
  """
  Compute DSS-global element-wise min and max for a batch of 3-D tracers.

  Parameters
  ----------
  tracers : Array[tuple[n_tracers, elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Stacked tracer fields (mixing ratios).
  grid : SpectralElementGrid
      Horizontal spectral element grid.
  dims : dict[str, int]
      Grid dimension parameters.

  Returns
  -------
  tracer_elem_lev_mins : Array[tuple[n_tracers, elem_idx, lev_idx], Float]
      Element-level global minimum for each tracer.
  tracer_elem_lev_maxs : Array[tuple[n_tracers, elem_idx, lev_idx], Float]
      Element-level global maximum for each tracer.
  """
  minvals = jnp.min(tracers, axis=(2, 3))
  maxvals = jnp.max(tracers, axis=(2, 3))
  tracer_elem_lev_mins = []
  tracer_elem_lev_maxs = []
  for tracer_idx in range(tracers.shape[0]):
    minvals_global = minmax_scalar_3d(minvals[tracer_idx, :, jnp.newaxis, jnp.newaxis, :] * jnp.ones_like(tracers[0, :, :, :, :]),
                                      grid, dims, max=False)
    tracer_elem_lev_mins.append(jnp.min(minvals_global, axis=(1, 2)))
    maxvals_global = minmax_scalar_3d(maxvals[tracer_idx, :, jnp.newaxis, jnp.newaxis] * jnp.ones_like(tracers[0, :, :, :, :]),
                                      grid, dims, max=True)
    tracer_elem_lev_maxs.append(jnp.max(maxvals_global, axis=(1, 2)))
  return jnp.stack(tracer_elem_lev_mins, axis=0), jnp.stack(tracer_elem_lev_maxs, axis=0)


@partial(jit, static_argnames=["dims"])
def tracer_euler_step(tracer_mass_stacked,
                      dt,
                      u_d_mass_avg,
                      interim_d_mass,
                      d_mass_for_limiter,
                      hypervis_tracer_tend,
                      physics_config,
                      grid,
                      dims):
  """
  Advance a batch of tracer-mass fields by one Euler substep with flux-form advection and limiting.

  Parameters
  ----------
  tracer_mass_stacked : Array[tuple[n_tracers, elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Stacked tracer-mass fields (mixing ratio * layer thickness).
  dt : float
      Substep size in seconds.
  u_d_mass_avg : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx, 2], Float]
      Time-averaged thickness-weighted wind ``h * u``.
  interim_d_mass : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Estimated layer thickness at the interim time level (used to compute velocity).
  d_mass_for_limiter : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Layer thickness used as the denominator in the flux limiter.
  hypervis_tracer_tend : Array[tuple[n_tracers, elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Hyperviscosity tendency to add (set to zeros if not used).
  physics_config : dict[str, Any]
      Physical constants (e.g. ``radius_earth``).
  grid : SpectralElementGrid
      Horizontal spectral element grid.
  dims : dict[str, int]
      Grid dimension parameters.

  Returns
  -------
  tracer_mass_out : Array[tuple[n_tracers, elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Updated stacked tracer-mass fields after one Euler substep.
  """
  interim_velocity = u_d_mass_avg / interim_d_mass[:, :, :, :, jnp.newaxis]
  tracer_mass_out = []
  tracer_maxs, tracer_mins = calc_minmax(tracer_mass_stacked/interim_d_mass, grid, dims)
  for tracer_idx in range(tracer_mass_stacked.shape[0]):
    tracer_tend = -horizontal_divergence_3d(tracer_mass_stacked[tracer_idx, :, :, :, :, jnp.newaxis] * interim_velocity, grid, physics_config)
    tracer_out = tracer_mass_stacked[tracer_idx, :, :, :, :] + dt * tracer_tend + hypervis_tracer_tend[tracer_idx, :, :, :, :]
    tracer_out = full_limiter(tracer_out, grid["mass_matrix"],
                              tracer_mins[tracer_idx, :, :],
                              tracer_maxs[tracer_idx, :, :],
                              d_mass_for_limiter)
    tracer_mass_out.append(project_tracer_3d(tracer_out, grid, dims))
    # Note: this is not communication efficient.
  return jnp.stack(tracer_mass_out, axis=0)

@partial(jit, static_argnames=["dims"])
def calc_hypervis_tend_tracer(tracer_mass, d_mass_scale, grid, dims, dt, physics_config, diffusion_config):
  """
  Compute the fourth-order hyperviscosity tendency for a batch of tracer-mass fields.

  Parameters
  ----------
  tracer_mass : Array[tuple[n_tracers, elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Stacked tracer-mass fields.
  d_mass_scale : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Layer-thickness scaling applied to the first Laplacian pass.
  grid : SpectralElementGrid
      Horizontal spectral element grid.
  dims : dict[str, int]
      Grid dimension parameters.
  dt : float
      Timestep (seconds) used to scale the tendency.
  physics_config : dict[str, Any]
      Physical constants (e.g. ``radius_earth``).
  diffusion_config : dict[str, Any]
      Hyperviscosity config; must contain ``"nu_tracer"``.

  Returns
  -------
  tracer_mass_tend : Array[tuple[n_tracers, elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Hyperviscosity tendency for each tracer-mass field.
  """
  tracer_mass_tend = []
  for tracer_idx in range(tracer_mass.shape[0]):
    harmonic = scalar_harmonic_3d(d_mass_scale * tracer_mass[tracer_idx, :, :, :, :], grid, physics_config)
    harmonic = project_tracer_3d(harmonic, grid, dims)
    apply_tensor = "tensor_hypervis" in diffusion_config.keys()
    biharmonic = scalar_harmonic_3d(tracer_mass[tracer_idx, :, :, :, :], grid, physics_config, apply_tensor=apply_tensor)
    tracer_mass_tend.append(-diffusion_config["nu_tracer"] * dt * biharmonic)
  return jnp.stack(tracer_mass_tend, axis=0)

@jit
def intermediate_d_mass_dynamics(d_mass_init, d_mass_tend_avg_cont, dt_total, step):
  """
  Estimate the layer thickness at an intermediate RK2 stage.

  Parameters
  ----------
  d_mass_init : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Layer thickness at the start of the timestep.
  d_mass_tend_avg_cont : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Projected (C0) layer-thickness tendency from the dynamics step.
  dt_total : float
      Full dynamics timestep in seconds.
  step : int
      RK substep index (0, 1, or 2).

  Returns
  -------
  d_mass_intermediate : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Estimated layer thickness at stage ``step``.
  """
  return d_mass_init + step * dt_total / 2.0 * d_mass_tend_avg_cont


@jit
def limiter_d_mass(d_mass_init, d_mass_tend_avg, d_mass_tend_avg_cont, dt_total, step):
  """
  Compute the layer thickness used as the denominator in the flux limiter.

  Parameters
  ----------
  d_mass_init : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Layer thickness at the start of the timestep.
  d_mass_tend_avg : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Discontinuous layer-thickness tendency from the dynamics step.
  d_mass_tend_avg_cont : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Projected (C0) layer-thickness tendency.
  dt_total : float
      Full dynamics timestep in seconds.
  step : int
      RK substep index (0, 1, or 2).

  Returns
  -------
  d_mass_limiter : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Layer thickness for use in the limiter denominator at stage ``step``.
  """
  return d_mass_init + dt_total / 2.0 * (step * d_mass_tend_avg_cont + d_mass_tend_avg)


@partial(jit, static_argnames=["dims", "timestep_config"])
def advance_tracers_rk2(tracer_mass_in,
                        d_mass_init,
                        u_d_mass_avg,
                        d_mass_tend_dyn,
                        grid,
                        physics_config,
                        diffusion_config,
                        timestep_config,
                        dims,
                        d_mass_hypervis_tend=None,
                        d_mass_hypervis_avg=None):
  """
  Advance a batch of tracer-mass fields by one tracer timestep using a 3-stage RK2 scheme.

  The scheme uses three equal substeps of size ``dt / 2``, with a final
  averaging step to give second-order accuracy.  Hyperviscosity is applied
  only in the last substep.

  Parameters
  ----------
  tracer_mass_in : Array[tuple[n_tracers, elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Stacked tracer-mass fields at the start of the tracer timestep.
  d_mass_init : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Layer thickness at the start of the tracer timestep.
  u_d_mass_avg : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx, 2], Float]
      Time-averaged thickness-weighted wind from the dynamics step.
  d_mass_tend_dyn : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Layer-thickness tendency from the dynamics step.
  grid : SpectralElementGrid
      Horizontal spectral element grid.
  physics_config : dict[str, Any]
      Physical constants (e.g. ``radius_earth``).
  diffusion_config : dict[str, Any]
      Hyperviscosity configuration.
  timestep_config : frozendict
      Time-step config; must contain ``"tracer_advection"`` with ``"dt"``.
  dims : dict[str, int]
      Grid dimension parameters.
  d_mass_hypervis_tend : Array or None, optional
      Hyperviscous layer-thickness tendency for tracer consistency.
  d_mass_hypervis_avg : Array or None, optional
      Average layer thickness from the hyperviscosity step, used to scale
      tracer hyperviscosity.

  Returns
  -------
  tracer_mass_out : Array[tuple[n_tracers, elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Updated stacked tracer-mass fields after one tracer timestep.
  """
  d_mass_tend_dyn_cont = project_tracer_3d(d_mass_tend_dyn, grid, dims)
  dt = timestep_config["tracer_advection"]["dt"]
  num_rk_stages = 3
  if d_mass_hypervis_avg is not None:
    hypervis_d_mass_scale = d_mass_hypervis_avg 
  else:
    hypervis_d_mass_scale = diffusion_config["d_mass_tracer"][jnp.newaxis, jnp.newaxis, jnp.newaxis, :] * jnp.ones_like(tracer_mass_in[0, :, :, :, :])
  intermediate_d_mass = intermediate_d_mass_dynamics(d_mass_init, d_mass_tend_dyn_cont, dt, 0)
  d_mass_limiter = limiter_d_mass(d_mass_init, d_mass_tend_dyn, d_mass_tend_dyn_cont, dt, 0)
  tracer_mass_out = tracer_euler_step(tracer_mass_in,
                                      dt / 2.0,
                                      u_d_mass_avg,
                                      intermediate_d_mass,
                                      d_mass_limiter,
                                      jnp.zeros_like(tracer_mass_in),
                                      physics_config,
                                      grid,
                                      dims)
  intermediate_d_mass = intermediate_d_mass_dynamics(d_mass_init, d_mass_tend_dyn_cont, dt, 1)
  d_mass_limiter = limiter_d_mass(d_mass_init, d_mass_tend_dyn, d_mass_tend_dyn_cont, dt, 1)

  tracer_mass_out = tracer_euler_step(tracer_mass_out,
                                      dt / 2.0,
                                      u_d_mass_avg,
                                      intermediate_d_mass,
                                      d_mass_limiter,
                                      jnp.zeros_like(tracer_mass_in),
                                      physics_config,
                                      grid,
                                      dims)
  intermediate_d_mass = intermediate_d_mass_dynamics(d_mass_init, d_mass_tend_dyn_cont, dt, 2)
  d_mass_limiter = limiter_d_mass(d_mass_init, d_mass_tend_dyn, d_mass_tend_dyn_cont, dt, 2)

  if d_mass_hypervis_tend is not None and "disable_diffusion" not in diffusion_config.keys():
    nu_tracer = diffusion_config["nu_tracer"]
    d_mass_limiter += 3.0 * dt / 2.0 * nu_tracer * d_mass_hypervis_tend
    hypervis_tend = calc_hypervis_tend_tracer(tracer_mass_out, hypervis_d_mass_scale, grid, dims, 3.0 * dt / 2.0, physics_config, diffusion_config)
  else:
    hypervis_tend = jnp.zeros_like(tracer_mass_in)
  tracer_mass_out = tracer_euler_step(tracer_mass_out,
                                      dt / 2.0,
                                      u_d_mass_avg,
                                      intermediate_d_mass,
                                      d_mass_limiter,
                                      hypervis_tend,
                                      physics_config,
                                      grid,
                                      dims)
  tracer_mass_out = (tracer_mass_in + (num_rk_stages - 1.0) * tracer_mass_out) / num_rk_stages
  return tracer_mass_out
