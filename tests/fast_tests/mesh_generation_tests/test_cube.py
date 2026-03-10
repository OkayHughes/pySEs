from src._config import get_backend as _get_backend
import numpy as np
from src.mesh_generation.cubed_sphere import face_topo, init_cube_topo
from src.mesh_generation.mesh import init_element_corner_vert_redundancy, edge_to_vert
from src.mesh_generation.mesh_definitions import TOP_FACE, BOTTOM_FACE, FRONT_FACE, BACK_FACE
from src.mesh_generation.mesh_definitions import LEFT_FACE, RIGHT_FACE
from src.mesh_generation.cubed_sphere import inv_elem_id_fn, elem_id_fn
from src.mesh_generation.mesh_definitions import TOP_EDGE, LEFT_EDGE, RIGHT_EDGE, BOTTOM_EDGE
_be = _get_backend()


def test_elem_id_fns():
  nx = 15
  for face_idx in [TOP_FACE, BOTTOM_FACE, FRONT_FACE, BACK_FACE, LEFT_FACE, RIGHT_FACE]:
      for x_idx in range(nx):
        for y_idx in range(nx):
          f, x, y = inv_elem_id_fn(nx, elem_id_fn(nx, face_idx, x_idx, y_idx))
          assert (f == face_idx and x == x_idx and y == y_idx)


def test_face_topo():
  # test if topology is consistent
  for face_id in face_topo:
    face_info = face_topo[face_id]
    for edge_id in face_info:
      edge_info = face_info[edge_id]
      face_id_pair1, edge_id_pair1, direction_match1 = edge_info
      face_id_pair2, edge_id_pair2, direction_match2 = face_topo[face_id_pair1][edge_id_pair1]
      assert (face_id_pair2 == face_id)
      assert (edge_id_pair2 == edge_id)
      assert (direction_match1 == direction_match2)


def test_gen_cube_topo():
  nx = 15
  face_connectivity, face_mask, face_position, face_position_2d = init_cube_topo(nx)
  for face_idx in [TOP_FACE, BOTTOM_FACE, FRONT_FACE, BACK_FACE, LEFT_FACE, RIGHT_FACE]:
    for x_idx in range(nx):
      for y_idx in range(nx):
        for edge_idx in [TOP_EDGE, LEFT_EDGE, RIGHT_EDGE, BOTTOM_EDGE]:
          elem_idx = elem_id_fn(nx, face_idx, x_idx, y_idx)
          idx_pair, edge_idx_pair, is_forwards = face_connectivity[elem_idx, edge_idx, :]
          face_idx_pair, x_idx_pair, y_idx_pair = inv_elem_id_fn(nx, idx_pair)
          v0_local, v1_local = edge_to_vert(edge_idx)
          v0_pair, v1_pair = edge_to_vert(edge_idx_pair, is_forwards=is_forwards)
          err = False
          for v_local, v_pair in [(v0_local, v0_pair), (v1_local, v1_pair)]:
            pos_local = face_position[elem_idx, v_local, :]
            pos_pair = face_position[idx_pair, v_pair, :]
            if np.max(np.abs(pos_local - pos_pair)) > 1e-10:
              err = True
              print(f"elem_idx: {elem_idx}, idx_pair: {idx_pair}")
              print(f"pos1: {pos_local}, pos2: {pos_pair}")
          assert (not err)


def test_vert_conn():
  nx = 15
  face_connectivity, face_mask, face_position, face_position_2d = init_cube_topo(nx)
  vert_redundancy = init_element_corner_vert_redundancy(face_connectivity)
  # test if all identified vertex pairings are correct
  for elem_idx in vert_redundancy.keys():
    for vert_idx in vert_redundancy[elem_idx].keys():
      for elem_idx_pair, vert_idx_pair in vert_redundancy[elem_idx][vert_idx]:
        assert (np.max(np.abs(face_position[elem_idx, vert_idx, :] -
                              face_position[elem_idx_pair, vert_idx_pair, :])) < 1e-10)
  # test if all vertex pairings were identified:
  # NOTE: only holds for regular grid.
  for face_idx in [TOP_FACE, BOTTOM_FACE, FRONT_FACE, BACK_FACE, LEFT_FACE, RIGHT_FACE]:
    for x_idx in range(nx):
      for y_idx in range(nx):
        elem_idx = elem_id_fn(nx, face_idx, x_idx, y_idx)
        for vert_idx in range(4):
          if ((x_idx == 0 and y_idx == 0 and vert_idx == 0) or
             (x_idx == nx - 1 and y_idx == 0 and vert_idx == 1) or
             (x_idx == 0 and y_idx == nx - 1 and vert_idx == 2) or
             (x_idx == nx - 1 and y_idx == nx - 1 and vert_idx == 3)):
            num_neighbors = 2
          else:
            num_neighbors = 3
          assert (len(vert_redundancy[elem_idx][vert_idx]) == num_neighbors)
