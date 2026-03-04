from ...context import test_npts
from pysces.mesh_generation.cubed_sphere import init_cube_topo
from pysces.mesh_generation.mesh import init_element_corner_vert_redundancy
from pysces.mesh_generation.mesh_definitions import TOP_FACE, BOTTOM_FACE, FRONT_FACE
from pysces.mesh_generation.mesh_definitions import BACK_FACE, LEFT_FACE, RIGHT_FACE
from pysces.mesh_generation.cubed_sphere import elem_id_fn
from pysces.mesh_generation.mesh import mesh_to_cart_bilinear, init_spectral_grid_redundancy


def test_gen_bilinear_grid_cs():
  nx = 7
  # note: test is only valid on quasi-uniform grid
  for npt in test_npts:
    face_connectivity, face_mask, face_position, face_position_2d = init_cube_topo(nx)
    vert_redundancy = init_element_corner_vert_redundancy(face_connectivity)
    gll_pos, gll_jacobian = mesh_to_cart_bilinear(face_position_2d, npt)
    vert_redundancy_gll = init_spectral_grid_redundancy(vert_redundancy, npt)
    for face_idx in [TOP_FACE, BOTTOM_FACE, FRONT_FACE, BACK_FACE, LEFT_FACE, RIGHT_FACE]:
      for x_idx in range(nx):
        for y_idx in range(nx):
          for i_idx in range(npt):
            for j_idx in range(npt):
              num_neighbors = 0
              if (((x_idx == 0 and y_idx == 0 and i_idx == 0 and j_idx == 0) or
                   (x_idx == 0 and y_idx == nx - 1 and i_idx == 0 and j_idx == npt - 1) or
                   (x_idx == nx - 1 and y_idx == nx - 1 and i_idx == npt - 1 and j_idx == npt - 1) or
                   (x_idx == nx - 1 and y_idx == 0 and i_idx == npt - 1 and j_idx == 0))):
                num_neighbors = 2
              elif ((i_idx == 0 and j_idx == 0) or
                    (i_idx == 0 and j_idx == npt - 1) or
                    (i_idx == npt - 1 and j_idx == 0) or
                    (i_idx == npt - 1 and j_idx == npt - 1)):
                num_neighbors = 3

              if j_idx != 0 and j_idx != npt - 1:
                if i_idx == 0 or i_idx == npt - 1:
                  num_neighbors = 1
              if i_idx != 0 and i_idx != npt - 1:
                if j_idx == 0 or j_idx == npt - 1:
                  num_neighbors = 1
              elem_idx = elem_id_fn(nx, face_idx, x_idx, y_idx)
              if (i_idx, j_idx) in vert_redundancy_gll[elem_idx].keys():
                assert (num_neighbors == len(vert_redundancy_gll[elem_idx][(i_idx, j_idx)]))
              else:
                assert (num_neighbors == 0)
