from .._config import get_backend as _get_backend
_be = _get_backend()
jit = _be.jit
jnp = _be.np
device_wrapper = _be.array
do_mpi_communication = _be.do_mpi_communication
from ..operations_2d.operators import horizontal_weak_vector_laplacian, horizontal_weak_laplacian
from ..operations_2d.tensor_hyperviscosity import (eval_quasi_uniform_hypervisc_coeff,
                                                   eval_variable_resolution_hypervisc_coeff)
from ..operations_2d.horizontal_grid import eval_global_grid_deformation_metrics
from .model_state import project_model_state, wrap_model_state
from ..mpi.global_assembly import project_scalar_global
from ..operations_2d.local_assembly import project_scalar
from functools import partial


def diffusion_config_for_tracer_consist(diffusion_config,
                                        d_mass_tracer=1.0e3):
  """
  Modify a diffusion config to disable mass diffusion while preserving tracer consistency.

  Sets ``"nu_d_mass"`` to zero and adds a ``"d_mass_tracer"`` scaling array
  used by the tracer hyperviscosity routine.

  Parameters
  ----------
  diffusion_config : dict[str, Any]
      Existing hyperviscosity configuration (modified in-place).
  d_mass_tracer : float, optional
      Reference layer thickness used as a scaling for tracer hyperviscosity
      (default: 1000.0).

  Returns
  -------
  diffusion_config : dict[str, Any]
      The modified configuration dict (same object as input).
  """
  diffusion_config["nu_d_mass"] = 0.0
  diffusion_config["d_mass_tracer"] = jnp.array([d_mass_tracer])
  return diffusion_config


def init_hypervis_config_const(ne,
                               config,
                               nu_base=-1.0,
                               nu_d_mass=-1.0,
                               nu_tracer=-1.0,
                               nu_div_factor=2.5):
  """
  Initialize a quasi-uniform hyperviscosity configuration for the shallow-water model.

  Parameters
  ----------
  ne : int
      Number of elements along one edge of the cubed-sphere face.
  config : dict[str, Any]
      Physics configuration containing ``radius_earth``.
  nu_base : float, optional
      Biharmonic viscosity coefficient for the wind (m^4 s^-1).
      Negative values auto-compute from grid resolution.
  nu_d_mass : float, optional
      Viscosity for the layer-thickness field.
      Negative values match ``nu_base``.
  nu_tracer : float, optional
      Viscosity for passive tracers.
      Negative values match ``nu_base``.
  nu_div_factor : float, optional
      Additional factor applied to the divergence component of the wind
      Laplacian (default: 2.5).

  Returns
  -------
  diffusion_config : dict[str, Any]
      Hyperviscosity configuration dict for the shallow-water model.
  """
  nu = eval_quasi_uniform_hypervisc_coeff(ne, config["radius_earth"]) if nu_base <= 0 else nu_base
  nu_d_mass = nu if nu_d_mass < 0 else nu_d_mass
  nu_tracer = nu if nu_tracer < 0 else nu_tracer
  diffusion_config = {"constant_hypervis": 1.0,
                      "nu": device_wrapper(nu),
                      "nu_d_mass": device_wrapper(nu_d_mass),
                      "nu_tracer": device_wrapper(nu_tracer),
                      "nu_div_factor": device_wrapper(nu_div_factor)}
  return diffusion_config


def init_hypervis_config_tensor(h_grid,
                                dims,
                                config,
                                ad_hoc_scale=0.5):
  """
  Initialize a variable-resolution tensor hyperviscosity configuration for the shallow-water model.

  Parameters
  ----------
  h_grid : SpectralElementGrid
      Horizontal spectral element grid (must contain ``"hypervis_scaling"``).
  dims : frozendict[str, int]
      Grid dimension parameters.
  config : dict[str, Any]
      Physics configuration containing ``radius_earth``.
  ad_hoc_scale : float, optional
      Empirical rescaling factor applied to the computed viscosity coefficient
      (default: 0.5).

  Returns
  -------
  diffusion_config : dict[str, Any]
      Hyperviscosity configuration dict for the shallow-water model.
  """
  radius_earth = config["radius_earth"]
  _, max_min_dx, min_max_dx = eval_global_grid_deformation_metrics(h_grid, dims)
  nu_tens = eval_variable_resolution_hypervisc_coeff(min_max_dx,
                                                     h_grid["hypervis_scaling"],
                                                     dims["npt"],
                                                     radius_earth=radius_earth)
  nu = device_wrapper(ad_hoc_scale * nu_tens)
  diffusion_config = {"tensor_hypervis": 1.0,
                      "nu": nu,
                      "nu_d_mass": nu,
                      "nu_tracer": nu}
  return diffusion_config


@partial(jit, static_argnames=["dims"])
def eval_hypervis_quasi_uniform(state_in,
                                grid,
                                physics_config,
                                diffusion_config,
                                dims):
  """
  Evaluate quasi-uniform fourth-order hyperviscosity tendency for the shallow-water model.

  Parameters
  ----------
  state_in : dict[str, Array]
      Shallow-water model state with keys ``"horizontal_wind"``, ``"h"``,
      and ``"hs"``.
  grid : SpectralElementGrid
      Horizontal spectral element grid.
  physics_config : dict[str, Any]
      Physical constants; must contain ``"radius_earth"``.
  diffusion_config : dict[str, Any]
      Hyperviscosity configuration containing ``"nu"``, ``"nu_d_mass"``,
      and ``"nu_div_factor"``.
  dims : frozendict[str, int]
      Grid dimension parameters.

  Returns
  -------
  state_tend : dict[str, Array]
      Hyperviscosity tendency in the same format as ``wrap_model_state``.
  """
  a = physics_config["radius_earth"]
  u_tmp = horizontal_weak_vector_laplacian(state_in["horizontal_wind"], grid, a=a, damp=True)
  h_tmp = horizontal_weak_laplacian(state_in["h"][:, :, :], grid, a=a)
  lap1 = project_model_state(wrap_model_state(u_tmp, h_tmp, state_in["hs"]), grid, dims)
  u_tmp = diffusion_config["nu"] * horizontal_weak_vector_laplacian(lap1["horizontal_wind"],
                                                                    grid,
                                                                    a=a,
                                                                    damp=True,
                                                                    nu_div_fact=diffusion_config["nu_div_factor"])
  h_tmp = diffusion_config["nu_d_mass"] * horizontal_weak_laplacian(lap1["h"], grid, a=a)
  return project_model_state(wrap_model_state(u_tmp, h_tmp, state_in["hs"]), grid, dims)


@partial(jit, static_argnames=["dims"])
def eval_hypervis_variable_resolution(state_in,
                                      grid,
                                      physics_config,
                                      diffusion_config,
                                      dims):
  """
  Evaluate variable-resolution tensor fourth-order hyperviscosity tendency for the shallow-water model.

  Parameters
  ----------
  state_in : dict[str, Array]
      Shallow-water model state with keys ``"horizontal_wind"``, ``"h"``,
      and ``"hs"``.
  grid : SpectralElementGrid
      Horizontal spectral element grid (must contain ``"physical_to_cartesian"``).
  physics_config : dict[str, Any]
      Physical constants; must contain ``"radius_earth"``.
  diffusion_config : dict[str, Any]
      Tensor hyperviscosity configuration containing ``"nu"`` and ``"nu_d_mass"``.
  dims : frozendict[str, int]
      Grid dimension parameters.

  Returns
  -------
  state_tend : dict[str, Array]
      Hyperviscosity tendency in the same format as ``wrap_model_state``.
  """
  a = physics_config["radius_earth"]
  u_cart = jnp.einsum("fijs,fijcs->fijc", jnp.flip(state_in["horizontal_wind"], axis=-1), grid["physical_to_cartesian"])
  components_laplace = []
  for cart_idx in range(3):
    components_laplace.append(horizontal_weak_laplacian(u_cart[:, :, :, cart_idx], grid, a=a, apply_tensor=False))
  h_laplace = horizontal_weak_laplacian(state_in["h"], grid, a=a, apply_tensor=False)
  if do_mpi_communication:
    state_laplace_cont = project_scalar_global([*components_laplace, h_laplace], grid, dims, two_d=True)
  else:
    state_laplace_cont = []
    for comp in components_laplace:
      state_laplace_cont.append(project_scalar(comp, grid, dims))
    state_laplace_cont.append(project_scalar(h_laplace, grid, dims))
  components_biharm = []
  for cart_idx in range(3):
    components_biharm.append(horizontal_weak_laplacian(state_laplace_cont[cart_idx], grid, a=a, apply_tensor=True))
  h_biharm = horizontal_weak_laplacian(state_laplace_cont[3], grid, a=a, apply_tensor=True)
  if do_mpi_communication:
    state_biharm_cont = project_scalar_global([*components_biharm, h_biharm], grid, dims, two_d=True)
  else:
    state_biharm_cont = []
    for comp in components_biharm:
      state_biharm_cont.append(project_scalar(comp, grid, dims))
    state_biharm_cont.append(project_scalar(h_biharm, grid, dims))

  h_biharm_cont = diffusion_config["nu_d_mass"] * state_biharm_cont[3]
  u_cart = jnp.stack(state_biharm_cont[:3], axis=-1)
  u_sph = jnp.einsum("fijc,fijcs->fijs", u_cart, grid["physical_to_cartesian"])
  u_sph = jnp.flip(diffusion_config["nu"] * u_sph, axis=-1)
  return wrap_model_state(u_sph, h_biharm_cont, state_in["hs"])
