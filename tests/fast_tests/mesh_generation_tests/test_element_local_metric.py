from src._config import get_backend as _get_backend
import numpy as np
from src.mesh_generation.element_local_metric import (init_quasi_uniform_grid_elem_local,
                                                      init_stretched_grid_elem_local)
from ...context import test_npts
_be = _get_backend()
get_global_array = _be.get_global_array


def test_gen_metric():
  for npt in test_npts:
    for nx in [6, 7]:
      grid, dims = init_quasi_uniform_grid_elem_local(nx, npt, wrapped=False)
      for ((elem_idx, i_idx, j_idx),
           (elem_idx_pair, i_idx_pair, j_idx_pair)) in grid["vertex_redundancy"]:
        assert (np.allclose(grid["physical_coords"][elem_idx, i_idx, j_idx, :],
                            grid["physical_coords"][elem_idx_pair, i_idx_pair, j_idx_pair, :]))


def test_gen_mass_mat():
  for npt in test_npts:
    for nx in [14, 15]:
      grid, dims = init_quasi_uniform_grid_elem_local(nx, npt)
      integrand = (grid["metric_determinant"] *
                   (grid["gll_weights"][np.newaxis, :, np.newaxis] *
                    grid["gll_weights"][np.newaxis, np.newaxis, :]))
      integrand = get_global_array(integrand, dims)
      assert (np.allclose(np.sum(integrand), 4 * np.pi))


def test_new_grid_tmp():
  np.random.seed(0)
  for npt in test_npts:
    for nx in [15, 16]:
      for _ in range(3):
        axis_dilation = np.random.uniform(high=1.5, low=1.0, size=(3,))
        offset = np.random.uniform(high=0.25, low=0.0, size=(3,))
        matrix = np.random.normal(size=(3, 3))
        Q, _ = np.linalg.qr(matrix)
        grid, dims = init_stretched_grid_elem_local(nx, npt,
                                                    axis_dilation=axis_dilation,
                                                    offset=offset,
                                                    orthogonal_transform=Q)
        integrand = (grid["metric_determinant"] *
                     (grid["gll_weights"][np.newaxis, :, np.newaxis] *
                      grid["gll_weights"][np.newaxis, np.newaxis, :]))
        integrand = get_global_array(integrand, dims)
        assert np.allclose(np.sum(integrand), 4 * np.pi)
        metdet = get_global_array(grid["metric_determinant"], dims)
        assert not np.any(np.isnan(metdet))
