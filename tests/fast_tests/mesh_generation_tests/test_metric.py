from pyses._config import get_backend as _get_backend
import numpy as np
from pyses.mesh_generation.cubed_sphere import init_cube_topo
from pyses.mesh_generation.mesh import init_element_corner_vert_redundancy
from pyses.mesh_generation.equiangular_metric import eval_metric_terms_equiangular, init_grid_from_topo
from pyses.mesh_generation.mesh import mesh_to_cart_bilinear, init_spectral_grid_redundancy
from ...context import test_npts
_be = _get_backend()


def test_gen_metric():
  for npt in test_npts:
    for nx in [6, 7]:
      face_connectivity, face_mask, face_position, face_position_2d = init_cube_topo(nx)
      vert_redundancy = init_element_corner_vert_redundancy(face_connectivity)
      gll_position, gll_jacobian = mesh_to_cart_bilinear(face_position_2d, npt)
      cube_redundancy = init_spectral_grid_redundancy(vert_redundancy, npt)
      gll_latlon, cube_to_sphere_jacobian = eval_metric_terms_equiangular(face_mask, gll_position, npt)
      for elem_idx in cube_redundancy.keys():
        for (i_idx, j_idx) in cube_redundancy[elem_idx].keys():
          for elem_idx_pair, i_idx_pair, j_idx_pair in cube_redundancy[elem_idx][(i_idx, j_idx)]:
              assert (np.max(np.abs(gll_latlon[elem_idx, i_idx, j_idx, :] -
                                    gll_latlon[elem_idx_pair, i_idx_pair, j_idx_pair, :])) < 1e-8)


def test_gen_mass_mat():
  for npt in test_npts:
    for nx in [14, 15]:
      face_connectivity, face_mask, face_position, face_position_2d = init_cube_topo(nx)
      vert_redundancy = init_element_corner_vert_redundancy(face_connectivity)
      grid, dims = init_grid_from_topo(face_connectivity,
                                       face_mask,
                                       face_position_2d,
                                       vert_redundancy,
                                       npt, wrapped=False)
      assert (np.allclose(np.sum(grid["metric_determinant"] *
                                 (grid["gll_weights"][np.newaxis, :, np.newaxis] *
                                  grid["gll_weights"][np.newaxis, np.newaxis, :])), 4 * np.pi))
