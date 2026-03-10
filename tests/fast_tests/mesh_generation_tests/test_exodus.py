from pyses._config import get_backend as _get_backend
import numpy as np
from pyses.mesh_generation.mesh_io import exodus_to_pyses_grid_corners
from pyses.mesh_generation.element_local_metric import init_unstructured_grid
from ...context import get_data_dir
from os.path import join
_be = _get_backend()


def test_grid():
  npt = 4
  arr = np.load(join(get_data_dir(), "conus.npz"))
  cart_coords = arr["cart_coords"]
  connect_map = arr["connect_map"]
  element_permuation = arr["element_permutation"]
  vert_pos, face_connectivity = exodus_to_pyses_grid_corners(cart_coords, connect_map, element_permuation)
  grid, dims = init_unstructured_grid(face_connectivity, vert_pos, npt)
  assert np.allclose(np.sum(grid["mass_matrix"]), 4 * np.pi)
