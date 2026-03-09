from src._config import get_backend as _get_backend
_be = _get_backend()
jnp = _be.np
eps = _be.eps
device_wrapper = _be.array
device_unwrapper = _be.unwrap
import numpy as np
from src.mesh_generation.equiangular_metric import init_quasi_uniform_grid
from src.operations_2d.local_assembly import project_scalar
from src.operations_2d.operators import (horizontal_gradient,
                                            horizontal_divergence,
                                            horizontal_vorticity,
                                            inner_product)
from src.operations_2d.operators import (horizontal_weak_divergence,
                                            horizontal_weak_gradient_covariant,
                                            horizontal_weak_vector_laplacian)
from src.mesh_generation.periodic_plane import init_uniform_grid
from ...context import test_npts, seed


def test_vector_identites_sphere():
  for npt in test_npts:
    for nx in [30, 31]:
      grid, dims = init_quasi_uniform_grid(nx, npt)

      fn = jnp.cos(grid["physical_coords"][:, :, :, 1]) * jnp.cos(grid["physical_coords"][:, :, :, 0])
      grad = horizontal_gradient(fn, grid)
      vort = horizontal_vorticity(grad, grid)
      iprod_vort = inner_product(vort, vort, grid)
      assert (np.allclose(iprod_vort, 0.0, atol=eps))
      v = jnp.stack((jnp.cos(grid["physical_coords"][:, :, :, 0]),
                    jnp.cos(grid["physical_coords"][:, :, :, 0])), axis=-1)

      grad = horizontal_gradient(fn, grid)
      discrete_divergence_thm = (inner_product(v[:, :, :, 0], grad[:, :, :, 0], grid) +
                                 inner_product(v[:, :, :, 1], grad[:, :, :, 1], grid) +
                                 inner_product(fn, horizontal_divergence(v, grid), grid))
      assert (jnp.allclose(discrete_divergence_thm, jnp.zeros_like(discrete_divergence_thm), atol=eps))


def test_vector_identities_plane():
  for npt in test_npts:
    nx, ny = (31, 33)
    grid, dims = init_uniform_grid(nx, ny, npt, length_x=jnp.pi, length_y=jnp.pi)
    fn = jnp.cos(grid["physical_coords"][:, :, :, 1]) * jnp.cos(grid["physical_coords"][:, :, :, 0])
    grad = horizontal_gradient(fn, grid)
    vort = horizontal_vorticity(grad, grid)

    iprod_vort = inner_product(vort, vort, grid)
    assert (np.allclose(device_unwrapper(iprod_vort), 0.0, atol=eps))
    v = jnp.stack((jnp.cos(grid["physical_coords"][:, :, :, 0]),
                  jnp.cos(grid["physical_coords"][:, :, :, 0])), axis=-1)

    grad = horizontal_gradient(fn, grid)
    discrete_divergence_thm = (inner_product(v[:, :, :, 0], grad[:, :, :, 0], grid) +
                               inner_product(v[:, :, :, 1], grad[:, :, :, 1], grid) +
                               inner_product(fn, horizontal_divergence(v, grid), grid))
    assert (jnp.allclose(discrete_divergence_thm, jnp.zeros_like(discrete_divergence_thm), atol=eps))


def test_vector_identites_rand_sphere():
  for npt in test_npts:
    np.random.seed(seed)
    for nx in [30, 31]:
      grid, dims = init_quasi_uniform_grid(nx, npt)
      for _ in range(10):
        rand = np.random.normal(scale=10, size=grid["physical_coords"][:, :, :, 0].shape)
        fn = rand * jnp.ones_like(grid["physical_coords"][:, :, :, 0])
        _ = project_scalar(fn, grid, dims)
        fn = project_scalar(fn, grid, dims)

        grad = horizontal_gradient(fn, grid)
        vort = horizontal_vorticity(grad, grid)
        grad = jnp.stack((project_scalar(grad[:, :, :, 0], grid, dims),
                          project_scalar(grad[:, :, :, 1], grid, dims)), axis=-1)
        vort = project_scalar(vort, grid, dims)
        iprod_vort = inner_product(vort, vort, grid)
        assert (np.allclose(device_unwrapper(iprod_vort), 0.0, atol=eps))
        v = np.random.normal(scale=1, size=grid["physical_coords"].shape)
        v *= jnp.ones_like(grid["physical_coords"])
        v = jnp.stack((project_scalar(v[:, :, :, 0], grid, dims),
                      project_scalar(v[:, :, :, 1], grid, dims)), axis=-1)
        div = horizontal_divergence(v, grid)
        div = project_scalar(horizontal_divergence(v, grid), grid, dims)
        discrete_divergence_thm = (inner_product(v[:, :, :, 0], grad[:, :, :, 0], grid) +
                                   inner_product(v[:, :, :, 1], grad[:, :, :, 1], grid) +
                                   inner_product(fn, div, grid))
        assert (jnp.allclose(discrete_divergence_thm, jnp.zeros_like(discrete_divergence_thm), atol=eps))


def test_vector_identites_rand_plane():
  for npt in test_npts:
    np.random.seed(seed)
    nx, ny = (31, 33)
    grid, dims = init_uniform_grid(nx, ny, npt)
    for _ in range(10):
      fn = device_wrapper(np.random.normal(scale=10, size=grid["physical_coords"][:, :, :, 0].shape))
      fn = project_scalar(fn, grid, dims)
      grad = horizontal_gradient(fn, grid)
      vort = horizontal_vorticity(grad, grid)
      grad = jnp.stack((project_scalar(grad[:, :, :, 0], grid, dims),
                        project_scalar(grad[:, :, :, 1], grid, dims)), axis=-1)
      vort = project_scalar(vort, grid, dims)

      iprod_vort = inner_product(vort, vort, grid)
      assert (np.allclose(device_unwrapper(iprod_vort), 0.0, atol=eps))
      v = device_wrapper(np.random.normal(scale=1, size=grid["physical_coords"].shape))
      v = jnp.stack((project_scalar(v[:, :, :, 0], grid, dims),
                    project_scalar(v[:, :, :, 1], grid, dims)), axis=-1)
      div = horizontal_divergence(v, grid)
      div = project_scalar(horizontal_divergence(v, grid), grid, dims)
      discrete_divergence_thm = (inner_product(v[:, :, :, 0], grad[:, :, :, 0], grid) +
                                 inner_product(v[:, :, :, 1], grad[:, :, :, 1], grid) +
                                 inner_product(fn, div, grid))
      assert (jnp.allclose(discrete_divergence_thm, jnp.zeros_like(discrete_divergence_thm), atol=eps))


def test_divergence():
  for npt in test_npts:
    for nx in [60, 61]:
      grid, dims = init_quasi_uniform_grid(nx, npt)
      vec = np.zeros_like(grid["physical_coords"])
      lat = grid["physical_coords"][:, :, :, 0]
      lon = grid["physical_coords"][:, :, :, 1]
      vec[:, :, :, 0] = np.cos(lat)**2 * np.cos(lon)**3
      vec[:, :, :, 1] = np.cos(lat)**2 * np.cos(lon)**3
      vec = device_wrapper(vec)

      vort_analytic = device_wrapper((-3.0 * np.cos(lon)**2 * np.sin(lon) * np.cos(lat) +
                                      3.0 * np.cos(lat) * np.sin(lat) * np.cos(lon)**3))

      div_analytic = device_wrapper((-3.0 * np.cos(lon)**2 * np.sin(lon) * np.cos(lat) -
                                    3.0 * np.cos(lat) * np.sin(lat) * np.cos(lon)**3))
      div = project_scalar(horizontal_divergence(vec, grid), grid, dims)
      div_wk = project_scalar(horizontal_weak_divergence(vec, grid) / grid["mass_matrix"], grid, dims)
      vort = project_scalar(horizontal_vorticity(vec, grid), grid, dims)
      assert (inner_product(div_wk - div, div_wk - div, grid) < 1e-5)
      assert (inner_product(div_analytic - div, div_analytic - div, grid) < 1e-5)
      assert (inner_product(vort_analytic - vort, vort_analytic - vort, grid) < 1e-5)


def test_analytic_soln():
  for npt, tol in zip([3, 4], [1e-3, 1e-5]):
    for nx in [60, 61]:
      grid, dims = init_quasi_uniform_grid(nx, npt)
      fn = jnp.cos(grid["physical_coords"][:, :, :, 1]) * jnp.cos(grid["physical_coords"][:, :, :, 0])
      grad_f_numerical = horizontal_gradient(fn, grid)
      sph_grad_wk = horizontal_weak_gradient_covariant(fn, grid)
      sph_grad_wk = jnp.stack((project_scalar(sph_grad_wk[:, :, :, 0] / grid["mass_matrix"], grid, dims),
                              project_scalar(sph_grad_wk[:, :, :, 1] / grid["mass_matrix"], grid, dims)), axis=-1)
      grad_diff = sph_grad_wk - grad_f_numerical

      sph_grad_lat = -jnp.cos(grid["physical_coords"][:, :, :, 1]) * jnp.sin(grid["physical_coords"][:, :, :, 0])
      sph_grad_lon = -jnp.sin(grid["physical_coords"][:, :, :, 1])
      print("Approximation: disregarding pole points in gradient test")
      mask = np.logical_not(jnp.logical_or(grid["physical_coords"][:, :, :, 0] > (jnp.pi / 2.0 - 1e-3),
                                           grid["physical_coords"][:, :, :, 0] < -(jnp.pi / 2.0 - 1e-3)))
      assert ((inner_product(grad_diff[:, :, :, 0], grad_diff[:, :, :, 0], grid) +
              inner_product(grad_diff[:, :, :, 1], grad_diff[:, :, :, 1], grid)) < tol)
      assert (jnp.max(jnp.abs(mask * (sph_grad_lat - grad_f_numerical[:, :, :, 1]))) < tol)
      assert (jnp.max(jnp.abs(mask * (sph_grad_lon - grad_f_numerical[:, :, :, 0]))) < tol)


def test_vector_laplacian():
  for npt in test_npts:
    for nx in [60, 61]:
      grid, dims = init_quasi_uniform_grid(nx, npt)
      v = jnp.stack((jnp.cos(grid["physical_coords"][:, :, :, 0]),
                     jnp.cos(grid["physical_coords"][:, :, :, 0])), axis=-1)
      laplace_v_wk = horizontal_weak_vector_laplacian(v, grid)
      laplace_v_wk = jnp.stack((project_scalar(laplace_v_wk[:, :, :, 0], grid, dims),
                                project_scalar(laplace_v_wk[:, :, :, 1], grid, dims)), axis=-1)

      lap_diff = laplace_v_wk + 2 * v
      print("Approximation: disregarding pole points in vector laplacian test")
      mask = np.logical_not(jnp.logical_or(grid["physical_coords"][:, :, :, 0] > (jnp.pi / 2.0 - 1e-3),
                                           grid["physical_coords"][:, :, :, 0] < -(jnp.pi / 2.0 - 1e-3)))
      assert ((inner_product(mask * lap_diff[:, :, :, 0], mask * lap_diff[:, :, :, 0], grid) +
               inner_product(mask * lap_diff[:, :, :, 1], mask * lap_diff[:, :, :, 1], grid)) < 1e-2)
      v = jnp.stack((np.cos(grid["physical_coords"][:, :, :, 0])**2,
                     np.cos(grid["physical_coords"][:, :, :, 0])**2), axis=-1)
      laplace_v_wk = horizontal_weak_vector_laplacian(v, grid)
      laplace_v_wk = jnp.stack((project_scalar(laplace_v_wk[:, :, :, 0], grid, dims),
                                project_scalar(laplace_v_wk[:, :, :, 1], grid, dims)), axis=-1)
      lap_diff = laplace_v_wk + 3.0 * (np.cos(2 * grid["physical_coords"][:, :, :, 0]))[:, :, :, np.newaxis]
      assert ((inner_product(mask * lap_diff[:, :, :, 0], mask * lap_diff[:, :, :, 0], grid) +
              inner_product(mask * lap_diff[:, :, :, 1], mask * lap_diff[:, :, :, 1], grid)) < 1e-2)


def test_pole_points():
  # TODO: determine correct behavior for grids with points directly at the poles.
  # How should gradient behave?
  pass
