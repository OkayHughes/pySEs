from pysces.config import np
from pysces.mesh_generation.mesh_io import exodus_to_pysces_grid_corners
from pysces.mesh_generation.element_local_metric import init_unstructured_grid
from ...context import get_data_dir, get_figdir, plot_grid
from os.path import join

def test_grid():
  npt = 4
  arr = np.load(join(get_data_dir(), "conus.npz"))
  cart_coords = arr["cart_coords"]
  connect_map = arr["connect_map"]
  element_permuation = arr["element_permutation"]
  print("grid read")
  vert_pos, face_connectivity = exodus_to_pysces_grid_corners(cart_coords, connect_map, element_permuation)
  print("redundancy generated")
  print(face_connectivity.dtype)
  grid, dims = init_unstructured_grid(face_connectivity, vert_pos, npt)
  assert np.allclose(np.sum(grid["mass_matrix"]), 4*np.pi)
  