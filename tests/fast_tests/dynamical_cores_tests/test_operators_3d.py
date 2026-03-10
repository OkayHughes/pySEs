from src._config import get_backend as _get_backend
import numpy as np
from src.mesh_generation.equiangular_metric import init_quasi_uniform_grid
from src.operations_2d.operators import inner_product
from src.dynamical_cores.operators_3d import (horizontal_divergence_3d,
                                              horizontal_gradient_3d,
                                              horizontal_vorticity_3d,
                                              horizontal_weak_vector_laplacian_3d)
from src.dynamical_cores.model_state import project_scalar_3d
_be = _get_backend()
jnp = _be.np
eps = _be.eps
device_wrapper = _be.array
device_unwrapper = _be.unwrap
get_global_array = _be.get_global_array


def threedify(field, nlev, axis=-1):
  if axis < 0:
    shape_out = list(field.shape)
    shape_out.reverse()
    shape_out.insert(-axis - 1, 1)
    shape_out.reverse()
  else:
    shape_out = field.shape
    shape_out.insert(axis, 1)
  ones_shape = [1 for _ in range(len(field.shape) + 1)]
  ones_shape[axis] = nlev
  return field.reshape(shape_out) * jnp.ones(ones_shape)


def test_vector_identites():
  npt = 4
  nx = 31
  nlev = 3
  grid, dims = init_quasi_uniform_grid(nx, npt)
  config = {"radius_earth": 1.0}
  fn = jnp.cos(grid["physical_coords"][:, :, :, 1]) * jnp.cos(grid["physical_coords"][:, :, :, 0])
  fn_3d = device_wrapper(threedify(fn, nlev) * jnp.arange(nlev).reshape((1, 1, 1, -1)))
  v = np.stack((jnp.cos(grid["physical_coords"][:, :, :, 0]),
                jnp.cos(grid["physical_coords"][:, :, :, 0])), axis=-1)
  v_3d = threedify(device_wrapper(v), nlev, axis=-2)

  grad = horizontal_gradient_3d(fn_3d, grid, config)
  vort = horizontal_vorticity_3d(grad, grid, config)
  div = horizontal_divergence_3d(v_3d, grid, config)
  for k_idx in range(nlev):
    iprod_vort = inner_product(vort[:, :, :, k_idx], vort[:, :, :, k_idx], grid)
    assert (np.allclose(device_unwrapper(iprod_vort), 0.0, atol=eps))

    discrete_divergence_thm = (inner_product(v_3d[:, :, :, k_idx, 0], grad[:, :, :, k_idx, 0], grid) +
                               inner_product(v_3d[:, :, :, k_idx, 1], grad[:, :, :, k_idx, 1], grid) -
                               inner_product(fn_3d[:, :, :, k_idx], div[:, :, :, k_idx], grid))
    assert (jnp.allclose(discrete_divergence_thm, jnp.zeros_like(discrete_divergence_thm), atol=eps))


def test_divergence():
  npt = 4
  nx = 31
  nlev = 3
  grid, dims = init_quasi_uniform_grid(nx, npt)
  config = {"radius_earth": 1.0}
  vec = np.zeros_like(grid["physical_coords"])
  lat = grid["physical_coords"][:, :, :, 0]
  lon = grid["physical_coords"][:, :, :, 1]
  vec[:, :, :, 0] = np.cos(lat)**2 * np.cos(lon)**3
  vec[:, :, :, 1] = np.cos(lat)**2 * np.cos(lon)**3

  vec_3d = threedify(device_wrapper(vec), nlev, axis=-2)

  vort_analytic = (-3.0 * np.cos(lon)**2 * np.sin(lon) * np.cos(lat) +
                   3.0 * np.cos(lat) * np.sin(lat) * np.cos(lon)**3)

  div_analytic = (-3.0 * np.cos(lon)**2 * np.sin(lon) * np.cos(lat) -
                  3.0 * np.cos(lat) * np.sin(lat) * np.cos(lon)**3)
  div = project_scalar_3d(horizontal_divergence_3d(vec_3d, grid, config), grid, dims)
  vort = project_scalar_3d(horizontal_vorticity_3d(vec_3d, grid, config), grid, dims)
  for lev_idx in range(nlev):
    assert (inner_product(div_analytic - div[:, :, :, lev_idx],
                          div_analytic - div[:, :, :, lev_idx], grid) < 1e-5)
    assert (inner_product(vort_analytic - vort[:, :, :, lev_idx],
                          vort_analytic - vort[:, :, :, lev_idx], grid) < 1e-5)


def test_analytic_soln():
  npt = 4
  nx = 31
  nlev = 3
  grid, dims = init_quasi_uniform_grid(nx, npt)
  config = {"radius_earth": 1.0}

  fn = jnp.cos(grid["physical_coords"][:, :, :, 1]) * jnp.cos(grid["physical_coords"][:, :, :, 0])
  fn_3d = threedify(device_wrapper(fn), nlev)
  grad_f_numerical = horizontal_gradient_3d(fn_3d, grid, config)

  sph_grad_lat = -jnp.cos(grid["physical_coords"][:, :, :, 1]) * jnp.sin(grid["physical_coords"][:, :, :, 0])
  sph_grad_lon = -jnp.sin(grid["physical_coords"][:, :, :, 1])
  for lev_idx in range(nlev):
    assert (np.max(np.abs(get_global_array(sph_grad_lat - grad_f_numerical[:, :, :, lev_idx, 1], dims))) < 1e-4)
    assert (np.max(np.abs(get_global_array(sph_grad_lon - grad_f_numerical[:, :, :, lev_idx, 0], dims))) < 1e-4)


def test_vector_laplacian():
  npt = 4
  nx = 31
  nlev = 3
  grid, dims = init_quasi_uniform_grid(nx, npt)
  vec = jnp.stack((jnp.cos(grid["physical_coords"][:, :, :, 0]),
                   jnp.cos(grid["physical_coords"][:, :, :, 0])), axis=-1)
  config = {"radius_earth": 1.0}
  vec_3d = threedify(vec, nlev, axis=-2)
  laplace_v_wk = horizontal_weak_vector_laplacian_3d(vec_3d, grid, config)
  laplace_v_wk = jnp.stack((project_scalar_3d(laplace_v_wk[:, :, :, :, 0], grid, dims),
                            project_scalar_3d(laplace_v_wk[:, :, :, :, 1], grid, dims)), axis=-1)

  lap_diff = laplace_v_wk + 2 * vec_3d
  for lev_idx in range(nlev):
    assert ((inner_product(lap_diff[:, :, :, lev_idx, 0], lap_diff[:, :, :, lev_idx, 0], grid) +
             inner_product(lap_diff[:, :, :, lev_idx, 1], lap_diff[:, :, :, lev_idx, 1], grid)) < 1e-5)
  vec = jnp.stack((jnp.cos(grid["physical_coords"][:, :, :, 0])**2,
                   jnp.cos(grid["physical_coords"][:, :, :, 0])**2), axis=-1)
  vec_3d = threedify(vec, nlev, axis=-2)
  laplace_v_wk = horizontal_weak_vector_laplacian_3d(vec_3d, grid, config)
  laplace_v_wk = jnp.stack((project_scalar_3d(laplace_v_wk[:, :, :, :, 0], grid, dims),
                            project_scalar_3d(laplace_v_wk[:, :, :, :, 1], grid, dims)), axis=-1)
  lap_diff = laplace_v_wk + 3 * (np.cos(2 * grid["physical_coords"][:, :, :, 0]))[:, :, :, np.newaxis, np.newaxis]
  # hack to negate pole point
  lap_diff *= np.cos(grid["physical_coords"][:, :, :, 0])[:, :, :, np.newaxis, np.newaxis]**2
  for lev_idx in range(nlev):
    assert ((inner_product(lap_diff[:, :, :, lev_idx, 0], lap_diff[:, :, :, lev_idx, 0], grid) +
             inner_product(lap_diff[:, :, :, lev_idx, 1], lap_diff[:, :, :, lev_idx, 1], grid)) < 1e-5)
