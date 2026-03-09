from src._config import get_backend as _get_backend
_be = _get_backend()
jnp = _be.np
eps = _be.eps
device_wrapper = _be.array
device_unwrapper = _be.unwrap
mpi_rank = _be.mpi_rank
mpi_size = _be.mpi_size
import numpy as np
from src.mesh_generation.equiangular_metric import init_quasi_uniform_grid
from src.operations_2d.local_assembly import project_scalar
from src.operations_2d.horizontal_grid import extract_subset_parallel_dim, make_grid_mpi_ready
from src.operations_2d.operators import (horizontal_gradient,
                                         horizontal_divergence,
                                         horizontal_vorticity,
                                         inner_product)
from src.mesh_generation.periodic_plane import init_uniform_grid
from src.mpi.global_assembly import project_scalar_global
from src.mpi.global_communication import global_sum
from src.mpi.processor_decomposition import init_decomp
from ...context import test_npts, seed


def test_vector_identites_sphere():

  for npt in test_npts:
    for nx in [30, 31]:
      grid, dims = init_quasi_uniform_grid(nx, npt)
      grid, dims = make_grid_mpi_ready(grid, dims, mpi_rank)
      fn = jnp.cos(grid["physical_coords"][:, :, :, 1]) * jnp.cos(grid["physical_coords"][:, :, :, 0])
      grad = horizontal_gradient(fn, grid)
      grad = jnp.stack(project_scalar_global([grad[:, :, :, 0],
                                              grad[:, :, :, 1]], grid, dims), axis=-1)
      vort = project_scalar_global([horizontal_vorticity(grad, grid)], grid, dims)[0]

      iprod_vort = inner_product(vort, vort, grid)
      assert (np.allclose(device_unwrapper(iprod_vort), 0.0, atol=eps))
      v = jnp.stack((jnp.cos(grid["physical_coords"][:, :, :, 0]),
                     jnp.cos(grid["physical_coords"][:, :, :, 0])), axis=-1)

      grad = horizontal_gradient(fn, grid)
      discrete_divergence_thm = (inner_product(v[:, :, :, 0], grad[:, :, :, 0], grid) +
                                 inner_product(v[:, :, :, 1], grad[:, :, :, 1], grid) +
                                 inner_product(fn, horizontal_divergence(v, grid), grid))
      assert (np.allclose(global_sum(discrete_divergence_thm), jnp.zeros((1,)), atol=eps))


def test_vector_identities_plane():
  for npt in test_npts:
    nx, ny = (31, 33)
    grid, dims = init_uniform_grid(nx, ny, npt, length_x=jnp.pi, length_y=jnp.pi)
    grid, dims = make_grid_mpi_ready(grid, dims, mpi_rank)
    fn = jnp.cos(grid["physical_coords"][:, :, :, 1]) * jnp.cos(grid["physical_coords"][:, :, :, 0])
    grad = horizontal_gradient(fn, grid)
    grad = jnp.stack(project_scalar_global([grad[:, :, :, 0],
                                            grad[:, :, :, 1]], grid, dims), axis=-1)
    vort = project_scalar_global([horizontal_vorticity(grad, grid)], grid, dims)[0]

    iprod_vort = inner_product(vort, vort, grid)
    assert (np.allclose(device_unwrapper(iprod_vort), 0.0, atol=eps))
    v = jnp.stack((jnp.cos(grid["physical_coords"][:, :, :, 0]),
                  jnp.cos(grid["physical_coords"][:, :, :, 0])), axis=-1)

    grad = horizontal_gradient(fn, grid)
    discrete_divergence_thm = (inner_product(v[:, :, :, 0], grad[:, :, :, 0], grid) +
                               inner_product(v[:, :, :, 1], grad[:, :, :, 1], grid) +
                               inner_product(fn, horizontal_divergence(v, grid), grid))
    assert (np.allclose(global_sum(discrete_divergence_thm), jnp.zeros((1,)), atol=eps))


def test_vector_identites_rand_sphere():
  for npt in test_npts:
    np.random.seed(seed)
    for nx in [30, 31]:
      grid, dims = init_quasi_uniform_grid(nx, npt)
      grid, dims = make_grid_mpi_ready(grid, dims, mpi_rank)
      for _ in range(10):
        fn = device_wrapper(np.random.normal(scale=10, size=grid["physical_coords"][:, :, :, 0].shape))
        fn = project_scalar_global([fn], grid, dims, (1))[0]
        grad = horizontal_gradient(fn, grid)
        grad = jnp.stack(project_scalar_global([grad[:, :, :, 0],
                                                grad[:, :, :, 1]], grid, dims), axis=-1)
        vort = project_scalar_global([horizontal_vorticity(grad, grid)], grid, dims)[0]
        iprod_vort = inner_product(vort, vort, grid)
        assert (np.allclose(device_unwrapper(iprod_vort), 0.0, atol=eps))
        v = device_wrapper(np.random.normal(scale=1, size=grid["physical_coords"].shape))
        v = jnp.stack(project_scalar_global([v[:, :, :, 0], v[:, :, :, 1]], grid, dims), axis=-1)
        div = horizontal_divergence(v, grid)
        div = project_scalar_global([horizontal_divergence(v, grid)], grid, dims, (1,))[0]
        discrete_divergence_thm = (inner_product(v[:, :, :, 0], grad[:, :, :, 0], grid) +
                                   inner_product(v[:, :, :, 1], grad[:, :, :, 1], grid) +
                                   inner_product(fn, div, grid))
        assert (np.allclose(global_sum(discrete_divergence_thm), jnp.zeros((1,)), atol=eps))


def test_vector_identites_rand_plane():
  for npt in test_npts:
    np.random.seed(seed)
    nx, ny = (31, 33)
    grid, dims = init_uniform_grid(nx, ny, npt)
    grid, dims = make_grid_mpi_ready(grid, dims, mpi_rank)
    for _ in range(10):
      fn = device_wrapper(np.random.normal(scale=10, size=grid["physical_coords"][:, :, :, 0].shape))
      fn = project_scalar_global([fn], grid, dims, (1,))[0]
      grad = horizontal_gradient(fn, grid)
      vort = horizontal_vorticity(grad, grid)
      grad = jnp.stack(project_scalar_global([grad[:, :, :, 0],
                                              grad[:, :, :, 1]], grid, dims), axis=-1)
      vort = project_scalar_global([horizontal_vorticity(grad, grid)], grid, dims)[0]
      iprod_vort = inner_product(vort, vort, grid)
      assert (np.allclose(device_unwrapper(iprod_vort), 0.0, atol=eps))
      v = device_wrapper(np.random.normal(scale=1, size=grid["physical_coords"].shape))
      v = jnp.stack(project_scalar_global([v[:, :, :, 0], v[:, :, :, 1]], grid, dims), axis=-1)
      div = horizontal_divergence(v, grid)
      div = project_scalar_global([horizontal_divergence(v, grid)], grid, dims, (1,))[0]
      discrete_divergence_thm = (inner_product(v[:, :, :, 0], grad[:, :, :, 0], grid) +
                                 inner_product(v[:, :, :, 1], grad[:, :, :, 1], grid) +
                                 inner_product(fn, div, grid))
      assert (np.allclose(global_sum(discrete_divergence_thm), jnp.zeros((1,)), atol=eps))


def test_equivalence_rand_sphere():
  for npt in test_npts:
    np.random.seed(seed)
    for nx in [6, 9]:
      grid_total, dims_total = init_quasi_uniform_grid(nx, npt)
      grid, dims = make_grid_mpi_ready(grid_total, dims_total, mpi_rank)
      decomp = init_decomp(dims_total["num_elem"], mpi_size)
      for _ in range(10):
        fn_total = device_wrapper(np.random.normal(scale=10, size=grid_total["physical_coords"][:, :, :, 0].shape))
        u_total = device_wrapper(np.random.normal(scale=10, size=grid_total["physical_coords"][:, :, :, 0].shape))
        v_total = device_wrapper(np.random.normal(scale=10, size=grid_total["physical_coords"][:, :, :, 0].shape))
        vec_total = jnp.stack((project_scalar(u_total, grid_total, dims_total),
                               project_scalar(v_total, grid_total, dims_total)), axis=-1)
        fn_total = project_scalar(fn_total, grid_total, dims_total)
        fn = extract_subset_parallel_dim(fn_total, mpi_rank, decomp)
        u = extract_subset_parallel_dim(u_total, mpi_rank, decomp)
        v = extract_subset_parallel_dim(v_total, mpi_rank, decomp)
        fn = project_scalar_global([fn], grid, dims)[0]

        vec = jnp.stack(project_scalar_global([u, v], grid, dims), axis=-1)
        grad = horizontal_gradient(fn, grid)
        vort = horizontal_vorticity(vec, grid)
        div = horizontal_divergence(vec, grid)
        grad_total = horizontal_gradient(fn_total, grid_total)
        vort_total = horizontal_vorticity(vec_total, grid_total)
        div_total = horizontal_divergence(vec_total, grid_total)
        assert np.allclose(extract_subset_parallel_dim(vec_total, mpi_rank, decomp), vec)

        assert np.allclose(extract_subset_parallel_dim(grad_total, mpi_rank, decomp), grad)

        assert np.allclose(extract_subset_parallel_dim(vort_total, mpi_rank, decomp), vort)

        assert np.allclose(extract_subset_parallel_dim(div_total, mpi_rank, decomp), div)
