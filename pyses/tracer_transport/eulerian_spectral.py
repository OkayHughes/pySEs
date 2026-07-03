from .._config import get_backend as _get_backend
import numpy as np
from ..operations_2d.local_assembly import minmax_scalar, project_scalar
from ..dynamical_cores.operators_3d import horizontal_divergence_3d
from ..dynamical_cores.hyperviscosity import scalar_harmonic_3d
from ..operations_2d.limiters import full_limiter, clip_and_sum_limiter
from functools import partial
_be = _get_backend()
jnp = _be.np
jit = _be.jit


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
  dims : frozendict[str, int]
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
  return _be.vmap(sph_op, in_axes=-1, out_axes=-1)(scalar)


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
  dims : frozendict[str, int]
      Grid dimension parameters.

  Returns
  -------
  scalar_cont : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Globally continuous (C0) tracer field.
  """
  sph_op = partial(project_scalar, grid=h_grid, dims=dims)
  return _be.vmap(sph_op, in_axes=-1, out_axes=-1)(scalar)


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
  dims : frozendict[str, int]
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
  # vmap over the leading tracer axis instead of a Python loop: one batched DSS
  # min/max exchange rather than n_tracers skinny ones (the DSS indices are
  # unbatched, only the payload is batched).  node_shape broadcasts the
  # per-(elem, lev) extremum back over the (npt, npt) block minmax_scalar_3d
  # expects.
  node_shape = tracers.shape[1:]

  def per_tracer_min(minv):
    field = jnp.broadcast_to(minv[:, np.newaxis, np.newaxis, :], node_shape)
    return jnp.min(minmax_scalar_3d(field, grid, dims, max=False), axis=(1, 2))

  def per_tracer_max(maxv):
    field = jnp.broadcast_to(maxv[:, np.newaxis, np.newaxis, :], node_shape)
    return jnp.max(minmax_scalar_3d(field, grid, dims, max=True), axis=(1, 2))

  tracer_elem_lev_mins = _be.vmap(per_tracer_min)(minvals)
  tracer_elem_lev_maxs = _be.vmap(per_tracer_max)(maxvals)
  return tracer_elem_lev_mins, tracer_elem_lev_maxs


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
  dims : frozendict[str, int]
      Grid dimension parameters.

  Returns
  -------
  tracer_mass_out : Array[tuple[n_tracers, elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Updated stacked tracer-mass fields after one Euler substep.
  """
  interim_velocity = u_d_mass_avg / interim_d_mass[:, :, :, :, np.newaxis]
  tracer_mins, tracer_maxs = calc_minmax(tracer_mass_stacked / interim_d_mass, grid, dims)

  # vmap the per-tracer advect/limit/project over the leading tracer axis
  # (tracer_mass, hypervis tendency, and the two limiter bounds are all batched
  # at axis 0; the wind, grid and limiter thickness are shared).
  def per_tracer(tracer_mass, hypervis_tend, tracer_min, tracer_max):
    tracer_tend = -horizontal_divergence_3d(tracer_mass[:, :, :, :, np.newaxis] * interim_velocity,
                                            grid,
                                            physics_config)
    tracer_out = tracer_mass + dt * tracer_tend + hypervis_tend
    tracer_out = clip_and_sum_limiter(tracer_out, grid["mass_matrix"],
                                      tracer_min, tracer_max, d_mass_for_limiter)
    return project_tracer_3d(tracer_out, grid, dims)

  return _be.vmap(per_tracer)(tracer_mass_stacked, hypervis_tracer_tend, tracer_mins, tracer_maxs)


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
  dims : frozendict[str, int]
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
  apply_tensor = "tensor_hypervis" in diffusion_config.keys()

  # vmap the biharmonic over the leading tracer axis (d_mass_scale, grid and
  # coefficients are shared) instead of a Python loop over tracers.
  def per_tracer(tracer_mass_single):
    harmonic = scalar_harmonic_3d(d_mass_scale * tracer_mass_single, grid, physics_config)
    harmonic = project_tracer_3d(harmonic, grid, dims)
    biharmonic = scalar_harmonic_3d(harmonic,
                                    grid,
                                    physics_config,
                                    apply_tensor=apply_tensor)
    return -diffusion_config["nu_tracer"] * dt * biharmonic

  return _be.vmap(per_tracer)(tracer_mass)


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
  dims : frozendict[str, int]
      Grid dimension parameters.
  d_mass_hypervis_tend : Array or None, optional
      Hyperviscosity layer-thickness tendency for tracer consistency.
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
    hypervis_d_mass_scale = jnp.ones_like(tracer_mass_in[0, :, :, :, :])
    if "disable_diffusion" not in diffusion_config.keys():
      hypervis_d_mass_scale *= diffusion_config["d_mass_tracer"][np.newaxis, np.newaxis, np.newaxis, :]
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
    hypervis_tend = calc_hypervis_tend_tracer(tracer_mass_out,
                                              hypervis_d_mass_scale,
                                              grid,
                                              dims,
                                              3.0 * dt / 2.0,
                                              physics_config,
                                              diffusion_config)
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
