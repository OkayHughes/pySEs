from pyses._config import get_backend as _get_backend
import numpy as np
from pyses.mesh_generation.mesh_io import exodus_to_pyses_grid_corners, pyses_grid_to_obj
from pyses.mesh_generation.element_local_metric import init_unstructured_grid
from pyses.mesh_generation.element_local_metric import init_quasi_uniform_grid_elem_local
from pyses.mesh_generation.spherical_coord_utils import unit_sphere_to_cart_coords
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

def test_obj():
  arr = np.load(join(get_data_dir(), "conus.npz"))
  cart_coords = arr["cart_coords"]
  connect_map = arr["connect_map"]
  element_permuation = arr["element_permutation"]
  vert_pos, _ = exodus_to_pyses_grid_corners(cart_coords, connect_map, element_permuation)
  num_segments = 3
  obj_content, _ = pyses_grid_to_obj(vert_pos, num_segments=num_segments)
  with open(join(get_data_dir(), f"conus_nedge_{num_segments}.obj"), "w") as f:
    f.write(obj_content)

def test_obj_coarse():
  nx = 8
  npt = 2
  grid, _ = init_quasi_uniform_grid_elem_local(nx, npt, wrapped=False)
  cart_corners = unit_sphere_to_cart_coords(grid["physical_coords"])
  vert_pos = np.zeros((cart_corners.shape[0],
                       4,
                       3))
  # I find indexing conventions induce mistakes,
  # so do this explicitly
  idx = np.array([[0, 1], [2, 3]], dtype=np.int32)
  for i in range(2):
    for j in range(2):
      vert_pos[:, idx[i, j], :] = cart_corners[:, i, j, :]
  num_segments = 6
  obj_content, _ = pyses_grid_to_obj(vert_pos, num_segments=num_segments)
  with open(join(get_data_dir(), f"nx_{nx}_nedge_{num_segments}.obj"), "w") as f:
    f.write(obj_content)

def test_obj_periodic_plane():
  nx, ny = (8, 8)
  corner_pos_x = np.linspace(-1, 1, nx+1)
  corner_pos_y = np.linspace(-1, 1, ny+1)
  vert_pos = np.zeros(((nx * ny, 4, 3)))
  ct = 0
  for i in range(nx):
    for j in range(ny):
      vert_pos[ct, 0, :] = [corner_pos_x[i], corner_pos_y[j], 1.0]
      vert_pos[ct, 1, :] = [corner_pos_x[i], corner_pos_y[j+1], 1.0]
      vert_pos[ct, 2, :] = [corner_pos_x[i+1], corner_pos_y[j], 1.0]
      vert_pos[ct, 3, :] = [corner_pos_x[i+1], corner_pos_y[j+1], 1.0]
      ct += 1
  def pin_fn(vert_pos):
    x_cube = vert_pos[0]/vert_pos[2]
    y_cube = vert_pos[1]/vert_pos[2]
    return ((np.abs(np.abs(x_cube) - 1)) < 1e-8 and
            (np.abs(np.abs(y_cube) - 1)) < 1e-8)
  vert_pos /= np.linalg.norm(vert_pos, axis=-1)[:, :, np.newaxis]
  num_segments = 6
  obj_content, pin_mask = pyses_grid_to_obj(vert_pos, num_segments=num_segments, pin_fn=pin_fn, scale_pinned=False)
  pin_text = ""
  for vert_idx, pin_vert in enumerate(pin_mask):
    if pin_vert:
      pin_text += f"fix_vertex {vert_idx}\n"
  with open(join(get_data_dir(), f"pin_nx_{nx}_ny_{ny}_nedge_{num_segments}.txt"), "w") as f:
    f.write(pin_text)
  with open(join(get_data_dir(), f"nx_{nx}_ny_{ny}_nedge_{num_segments}.obj"), "w") as f:
    f.write(obj_content)
  
  