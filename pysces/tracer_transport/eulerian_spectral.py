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
  [Description]

  Parameters
  ----------
  [first] : array_like
      the 1st param name `first`
  second :
      the 2nd param
  third : {'value', 'other'}, optional
      the 3rd param, by default 'value'

  Returns
  -------
  string
      a value in a string

  Raises
  ------
  KeyError
      when a key error
  """
  sph_op = partial(minmax_scalar, grid=h_grid, max=max, dims=dims)
  return vmap_1d_apply(sph_op, scalar, -1, -1)


@partial(jit, static_argnames=["dims"])
def project_tracer_3d(scalar,
                      h_grid,
                      dims):
  """
  [Description]

  Parameters
  ----------
  [first] : array_like
      the 1st param name `first`
  second :
      the 2nd param
  third : {'value', 'other'}, optional
      the 3rd param, by default 'value'

  Returns
  -------
  string
      a value in a string

  Raises
  ------
  KeyError
      when a key error
  """
  sph_op = partial(project_scalar, grid=h_grid, dims=dims)
  return vmap_1d_apply(sph_op, scalar, -1, -1)


@partial(jit, static_argnames=["dims"])
def calc_minmax(tracers, grid, dims):
  minvals = jnp.min(tracers, axis=(2, 3))
  maxvals = jnp.min(tracers, axis=(2, 3))
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
  return d_mass_init + step * dt_total / 2.0 * d_mass_tend_avg_cont


@jit
def limiter_d_mass(d_mass_init, d_mass_tend_avg, d_mass_tend_avg_cont, dt_total, step):
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
