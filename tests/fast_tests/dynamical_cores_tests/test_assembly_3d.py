from pyses._config import get_backend as _get_backend
import numpy as np
from pyses.mesh_generation.cubed_sphere import init_cube_topo
from pyses.mesh_generation.mesh import init_element_corner_vert_redundancy
from pyses.mesh_generation.equiangular_metric import init_grid_from_topo
from pyses.dynamical_cores.model_state import project_scalar_3d
from pyses.mesh_generation.mesh import vert_red_flat_to_hierarchy
from ..operations_2d_tests.test_local_assembly import project_scalar_for
_be = _get_backend()
jnp = _be.np
device_wrapper = _be.array
use_wrapper = _be.use_wrapper
get_global_array = _be.get_global_array


def is_3d_field_c0(field_in, h_grid, dims):
  is_c0 = True
  field = get_global_array(field_in, dims)
  for lev_idx in range(field.shape[-1]):
    rows = h_grid["assembly_triple"][1]
    cols = h_grid["assembly_triple"][2]
    lev_slice = jnp.take(field, lev_idx, axis=-1)
    row_vals = lev_slice[rows[0], rows[1], rows[2]]
    col_vals = lev_slice[cols[0], cols[1], cols[2]]
    is_c0 = is_c0 and jnp.allclose(row_vals, col_vals)
  return is_c0


def _project_scalar_3d_for(variable, h_grid, dims):
  levs = []
  for lev_idx in range(variable.shape[-1]):
    levs.append(project_scalar_for(variable[:, :, :, lev_idx], h_grid))
  return np.stack(levs, axis=-1)


def test_project_3d():
  npt = 4
  nx = 3
  nlev = 3
  face_connectivity, face_mask, face_position, face_position_2d = init_cube_topo(nx)
  vert_redundancy = init_element_corner_vert_redundancy(face_connectivity)
  grid, dims = init_grid_from_topo(face_connectivity,
                                   face_mask,
                                   face_position_2d,
                                   vert_redundancy,
                                   npt,
                                   wrapped=use_wrapper)
  grid_nowrapper, _ = init_grid_from_topo(face_connectivity,
                                          face_mask,
                                          face_position_2d,
                                          vert_redundancy,
                                          npt,
                                          wrapped=False)
  vert_redundancy_gll = vert_red_flat_to_hierarchy(grid_nowrapper["vertex_redundancy"])
  fn = np.zeros((*grid["physical_coords"].shape[:-1], nlev))
  for lev_idx in range(nlev):
    for face_idx in range(grid["physical_coords"].shape[0]):
      for i_idx in range(npt):
        for j_idx in range(npt):
          fn[:] = 0.0
          fn[face_idx, i_idx, j_idx, lev_idx] = 1.0
          if face_idx in vert_redundancy_gll.keys():
            if (i_idx, j_idx) in vert_redundancy_gll[face_idx].keys():
              for remote_face_id, remote_i, remote_j in vert_redundancy_gll[face_idx][(i_idx, j_idx)]:
                fn[remote_face_id, remote_i, remote_j, lev_idx] = 1.0
            new_fn = device_wrapper(fn, elem_sharding_axis=0)
            assert jnp.allclose(jnp.nanmax((project_scalar_3d(new_fn, grid, dims) - new_fn)), 0.0)


def test_project_equiv_3d_rand():
  npt = 4
  nx = 15
  nlev = 5
  face_connectivity, face_mask, face_position, face_position_2d = init_cube_topo(nx)
  vert_redundancy = init_element_corner_vert_redundancy(face_connectivity)
  grid, dims = init_grid_from_topo(face_connectivity,
                                   face_mask,
                                   face_position_2d,
                                   vert_redundancy,
                                   npt,
                                   wrapped=False)
  grid_wrapped, dims_wrapped = init_grid_from_topo(face_connectivity,
                                                   face_mask,
                                                   face_position_2d,
                                                   vert_redundancy,
                                                   npt,
                                                   wrapped=use_wrapper)
  for _ in range(20):
    fn_rand = device_wrapper(np.random.uniform(size=(*grid_wrapped["physical_coords"][:, :, :, 1].shape, nlev)),
                             elem_sharding_axis=0)
    fn_loc = get_global_array(fn_rand, dims_wrapped)
    assert (np.allclose(get_global_array(project_scalar_3d(fn_rand, grid_wrapped, dims_wrapped), dims),
                        _project_scalar_3d_for(fn_loc, grid, dims)))


def test_project_c0():
  npt = 4
  nx = 15
  nlev = 5
  face_connectivity, face_mask, face_position, face_position_2d = init_cube_topo(nx)
  vert_redundancy = init_element_corner_vert_redundancy(face_connectivity)
  grid, dims = init_grid_from_topo(face_connectivity,
                                   face_mask,
                                   face_position_2d,
                                   vert_redundancy,
                                   npt,
                                   wrapped=use_wrapper)
  for _ in range(20):
    fn_rand = np.random.uniform(size=(*grid["physical_coords"][:, :, :, 1].shape, nlev))
    scalar_cont = project_scalar_3d(device_wrapper(fn_rand), grid, dims)
    assert is_3d_field_c0(scalar_cont, grid, dims)
