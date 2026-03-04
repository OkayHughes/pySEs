from pysces.config import np, device_wrapper, use_wrapper, jnp, get_global_array
from pysces.mesh_generation.cubed_sphere import init_cube_topo
from pysces.mesh_generation.mesh import init_element_corner_vert_redundancy
from pysces.horizontal_grid import shard_grid
from pysces.mesh_generation.equiangular_metric import init_grid_from_topo
from pysces.mesh_generation.mesh import vert_red_flat_to_hierarchy
from pysces.operations_2d.local_assembly import (project_scalar_wrapper,
                                                 project_scalar,
                                                 init_shard_extraction_map,
                                                 minmax_scalar)
from ...context import test_npts, pretty_print_scalar


def project_scalar_for(f,
                       grid,
                       *args):
  """
  Project a potentially discontinuous scalar onto the continuous
  subspace using a for loop, assuming all data is processor-local.

  *This is used for testing. Do not use in performance code*

  Parameters
  ----------
  f : `Array[tuple[elem_idx, gll_idx, gll_idx], Float]`
    Scalar field to project
  grid : `SpectralElementGrid`
    Spectral element grid struct that contains coordinate and metric data.

  Notes
  -----
  This is the most human-readable way to perform projection, and can be used to test
  if your grid is topologically malformed.

  Returns
  -------
  f_cont
      The globally continous scalar closest in norm to f.
  """
  # assumes that values from remote processors have already been accumulated
  metdet = grid["metric_determinant"]
  inv_mass_mat = grid["mass_matrix_denominator"]
  vert_redundancy_gll = grid["vertex_redundancy"]
  gll_weights = grid["gll_weights"]
  workspace = f.copy()
  workspace *= metdet * (gll_weights[np.newaxis, :, np.newaxis] * gll_weights[np.newaxis, np.newaxis, :])
  for ((local_face_idx, local_i, local_j),
       (remote_face_id, remote_i, remote_j)) in vert_redundancy_gll:
    workspace[remote_face_id, remote_i, remote_j] += (metdet[local_face_idx, local_i, local_j] *
                                                      f[local_face_idx, local_i, local_j] *
                                                      (gll_weights[local_i] * gll_weights[local_j]))
  # this line works even for multi-processor decompositions.
  workspace *= inv_mass_mat
  return workspace


def project_scalar_sharded_numpy(f, grid_nowrap, dims, extraction_struct, num_devices):
  # this should match the jax section of the `project_scalar_wrapper`
  # function in local_assembly.py, but operations are done with numpy
  shape = f.shape
  scaled_f = f * grid_nowrap["mass_matrix"]
  scaled_f = scaled_f.reshape((num_devices, -1, dims["npt"], dims["npt"]))
  relevant_data = (scaled_f)[extraction_struct["extract_from"]["shard_idx"],
                             extraction_struct["extract_from"]["elem_idx"],
                             extraction_struct["extract_from"]["i_idx"],
                             extraction_struct["extract_from"]["j_idx"]]
  i_test = np.arange(dims["npt"])[np.newaxis, np.newaxis, :, np.newaxis] * np.ones_like(scaled_f)
  i_maybe = i_test[extraction_struct["extract_from"]["shard_idx"],
                   extraction_struct["extract_from"]["elem_idx"],
                   extraction_struct["extract_from"]["i_idx"],
                   extraction_struct["extract_from"]["j_idx"]]
  assert jnp.allclose(extraction_struct["extract_from"]["i_idx"], i_maybe)

  relevant_data *= extraction_struct["mask"]
  np.add.at(scaled_f, (extraction_struct["sum_into"]["shard_idx"],
                       extraction_struct["sum_into"]["elem_idx"],
                       extraction_struct["sum_into"]["i_idx"],
                       extraction_struct["sum_into"]["j_idx"]), relevant_data)
  scaled_f = scaled_f.reshape(shape)
  return scaled_f * grid_nowrap["mass_matrix_denominator"]


def project_scalar_sharded_for(f, grid_nowrap, dims, extraction_struct, num_devices, oopsie=False):
  # this should match the jax section of the `project_scalar_wrapper`
  # function in local_assembly.py, but operations are done with numpy
  shape = f.shape
  scaled_f = f * grid_nowrap["mass_matrix"]
  sharded_shape = (num_devices, -1, dims["npt"], dims["npt"])
  scaled_f = scaled_f.reshape(sharded_shape)
  scaled_f_buffer = scaled_f.copy()
  max_dof = extraction_struct["extract_from"]["shard_idx"].shape[1]
  for shard_idx in range(num_devices):
    for pt_idx in range(max_dof):
      extract_from_shard_idx = extraction_struct["extract_from"]["shard_idx"][shard_idx, pt_idx]
      extract_from_elem_idx = extraction_struct["extract_from"]["elem_idx"][shard_idx, pt_idx]
      extract_from_i_idx = extraction_struct["extract_from"]["i_idx"][shard_idx, pt_idx]
      extract_from_j_idx = extraction_struct["extract_from"]["j_idx"][shard_idx, pt_idx]

      sum_into_shard_idx = extraction_struct["sum_into"]["shard_idx"][shard_idx, pt_idx]
      sum_into_elem_idx = extraction_struct["sum_into"]["elem_idx"][shard_idx, pt_idx]
      sum_into_i_idx = extraction_struct["sum_into"]["i_idx"][shard_idx, pt_idx]
      sum_into_j_idx = extraction_struct["sum_into"]["j_idx"][shard_idx, pt_idx]
      mask_val = extraction_struct["mask"][shard_idx, pt_idx]
      entry = scaled_f_buffer[extract_from_shard_idx,
                              extract_from_elem_idx,
                              extract_from_i_idx,
                              extract_from_j_idx]
      scaled_f[sum_into_shard_idx,
               sum_into_elem_idx,
               sum_into_i_idx,
               sum_into_j_idx] += mask_val * entry

  scaled_f = scaled_f.reshape(shape)
  return scaled_f * grid_nowrap["mass_matrix_denominator"]


def test_extraction_map():
  for num_devices in [2, 1]:
    for npt in test_npts:
      for nx in [1, 4]:
        face_connectivity, face_mask, face_position, face_position_2d = init_cube_topo(nx)
        vert_redundancy = init_element_corner_vert_redundancy(face_connectivity)
        grid_nowrap, dims_nowrap = init_grid_from_topo(face_connectivity,
                                                       face_mask,
                                                       face_position_2d,
                                                       vert_redundancy,
                                                       npt,
                                                       wrapped=False)

        extraction_struct, _ = init_shard_extraction_map(grid_nowrap["assembly_triple"],
                                                         num_devices,
                                                         grid_nowrap["metric_determinant"].shape[0],
                                                         dims_nowrap, wrapped=False)
        assert np.allclose(np.sum(extraction_struct["mask"]), grid_nowrap["assembly_triple"][0].shape[0])

        def add_one_point(field, f, i, j):
          axis_1 = np.eye(field.shape[0])[f, :].squeeze()
          axis_2 = np.eye(field.shape[1])[i, :].squeeze()
          axis_3 = np.eye(field.shape[2])[j, :].squeeze()
          return (axis_1[:, np.newaxis, np.newaxis] *
                  axis_2[np.newaxis, :, np.newaxis] *
                  axis_3[np.newaxis, np.newaxis, :])

        vert_redundancy_gll = vert_red_flat_to_hierarchy(grid_nowrap["vertex_redundancy"])
        for face_idx in range(grid_nowrap["physical_coords"].shape[0]):
          for i_idx in range(npt):
            for j_idx in range(npt):
              fn = np.zeros_like(grid_nowrap["physical_coords"][:, :, :, 0])
              fn += add_one_point(fn, face_idx, i_idx, j_idx)
              if face_idx in vert_redundancy_gll.keys():
                if (i_idx, j_idx) in vert_redundancy_gll[face_idx].keys():
                  for remote_face_id, remote_i, remote_j in vert_redundancy_gll[face_idx][(i_idx, j_idx)]:

                    fn += add_one_point(fn, remote_face_id, remote_i, remote_j)
              cont_fn = project_scalar_sharded_for(fn, grid_nowrap, dims_nowrap, extraction_struct, num_devices)
              assert np.allclose(cont_fn, fn)
              cont_fn = project_scalar_sharded_numpy(fn, grid_nowrap, dims_nowrap, extraction_struct, num_devices)
              assert np.allclose(cont_fn, fn)


def test_extraction_map_rand():
  num_devices = 2
  for npt in test_npts:
    for nx in [4, 8]:
      face_connectivity, face_mask, face_position, face_position_2d = init_cube_topo(nx)
      vert_redundancy = init_element_corner_vert_redundancy(face_connectivity)
      grid_nowrap, dims_nowrap = init_grid_from_topo(face_connectivity,
                                                     face_mask,
                                                     face_position_2d,
                                                     vert_redundancy,
                                                     npt, wrapped=False)
      extraction_struct, _ = init_shard_extraction_map(grid_nowrap["assembly_triple"],
                                                       num_devices,
                                                       grid_nowrap["metric_determinant"].shape[0],
                                                       dims_nowrap, wrapped=False)
      for _ in range(20):
        fn_rand = np.random.uniform(size=grid_nowrap["physical_coords"][:, :, :, 1].shape)
        assert np.allclose(project_scalar_sharded_numpy(fn_rand,
                                                        grid_nowrap,
                                                        dims_nowrap,
                                                        extraction_struct,
                                                        num_devices),
                           project_scalar_for(fn_rand, grid_nowrap))


def test_projection():
  for npt in test_npts:
    for nx in [3, 4]:
      face_connectivity, face_mask, face_position, face_position_2d = init_cube_topo(nx)
      vert_redundancy = init_element_corner_vert_redundancy(face_connectivity)
      raw_grid, dims = init_grid_from_topo(face_connectivity,
                                           face_mask,
                                           face_position_2d,
                                           vert_redundancy,
                                           npt,
                                           wrapped=use_wrapper)
      grid = shard_grid(raw_grid, dims)
      grid_nowrapper, _ = init_grid_from_topo(face_connectivity,
                                              face_mask,
                                              face_position_2d,
                                              vert_redundancy,
                                              npt,
                                              wrapped=False)

      def add_one_point(field, f, i, j):
        axis_1 = jnp.eye(field.shape[0])[f, :].squeeze()
        axis_2 = jnp.eye(field.shape[1])[i, :].squeeze()
        axis_3 = jnp.eye(field.shape[2])[j, :].squeeze()
        return (axis_1[:, jnp.newaxis, jnp.newaxis] *
                axis_2[jnp.newaxis, :, jnp.newaxis] *
                axis_3[jnp.newaxis, jnp.newaxis, :])

      vert_redundancy_gll = vert_red_flat_to_hierarchy(grid_nowrapper["vertex_redundancy"])
      for face_idx in range(grid["physical_coords"].shape[0]):
        for i_idx in range(npt):
          for j_idx in range(npt):
            fn = jnp.zeros_like(grid["physical_coords"][:, :, :, 0])
            fn += add_one_point(fn, face_idx, i_idx, j_idx)
            if face_idx in vert_redundancy_gll.keys():
              if (i_idx, j_idx) in vert_redundancy_gll[face_idx].keys():
                for remote_face_id, remote_i, remote_j in vert_redundancy_gll[face_idx][(i_idx, j_idx)]:
                  fn += add_one_point(fn, remote_face_id, remote_i, remote_j)
            cont_fn = project_scalar(fn, grid, dims)
            assert np.allclose(get_global_array(cont_fn, dims), get_global_array(fn, dims))


def test_minmax():
  for npt in test_npts:
    for nx in [3, 4]:
      face_connectivity, face_mask, face_position, face_position_2d = init_cube_topo(nx)
      vert_redundancy = init_element_corner_vert_redundancy(face_connectivity)
      raw_grid, dims = init_grid_from_topo(face_connectivity,
                                           face_mask,
                                           face_position_2d,
                                           vert_redundancy,
                                           npt,
                                           wrapped=use_wrapper)
      grid = shard_grid(raw_grid, dims)
      grid_nowrapper, _ = init_grid_from_topo(face_connectivity,
                                              face_mask,
                                              face_position_2d,
                                              vert_redundancy,
                                              npt,
                                              wrapped=False)
      def add_one_point(field, f, i, j):
        axis_1 = jnp.eye(field.shape[0])[f, :].squeeze()
        axis_2 = jnp.eye(field.shape[1])[i, :].squeeze()
        axis_3 = jnp.eye(field.shape[2])[j, :].squeeze()
        return (axis_1[:, jnp.newaxis, jnp.newaxis] *
                axis_2[jnp.newaxis, :, jnp.newaxis] *
                axis_3[jnp.newaxis, jnp.newaxis, :])
      vert_redundancy_gll = vert_red_flat_to_hierarchy(grid_nowrapper["vertex_redundancy"])
      for is_max, extremal_op in zip([True, False], [max, min]):
        for face_idx in range(grid["physical_coords"].shape[0]):
          for i_idx in range(npt):
            for j_idx in range(npt):
              fn = jnp.zeros_like(grid["physical_coords"][:, :, :, 0])
              extremal_face_idx = face_idx
              fn += add_one_point(fn, face_idx, i_idx, j_idx) * face_idx
              if face_idx in vert_redundancy_gll.keys():
                if (i_idx, j_idx) in vert_redundancy_gll[face_idx].keys():
                  for remote_face_id, remote_i, remote_j in vert_redundancy_gll[face_idx][(i_idx, j_idx)]:
                    extremal_face_idx = extremal_op(remote_face_id, extremal_face_idx)
                    fn += add_one_point(fn, remote_face_id, remote_i, remote_j) * remote_face_id
              fn_out = jnp.zeros_like(grid["physical_coords"][:, :, :, 0])
              fn_out += add_one_point(fn, face_idx, i_idx, j_idx) * extremal_face_idx 
              if face_idx in vert_redundancy_gll.keys():
                if (i_idx, j_idx) in vert_redundancy_gll[face_idx].keys():
                  for remote_face_id, remote_i, remote_j in vert_redundancy_gll[face_idx][(i_idx, j_idx)]:
                    fn_out += add_one_point(fn, remote_face_id, remote_i, remote_j) * extremal_face_idx
              max_fn = minmax_scalar(fn, grid, dims, max=is_max)
              assert np.allclose(get_global_array(max_fn, dims), get_global_array(fn_out, dims))


def test_projection_equiv():
  for npt in test_npts:
    for nx in [7, 8]:
      face_connectivity, face_mask, face_position, face_position_2d = init_cube_topo(nx)
      vert_redundancy = init_element_corner_vert_redundancy(face_connectivity)
      raw_grid, dims = init_grid_from_topo(face_connectivity,
                                           face_mask,
                                           face_position_2d,
                                           vert_redundancy,
                                           npt, wrapped=False)
      grid_wrapped, dims_wrapped = init_grid_from_topo(face_connectivity,
                                                       face_mask,
                                                       face_position_2d,
                                                       vert_redundancy,
                                                       npt, wrapped=use_wrapper)
      grid_wrapped = shard_grid(grid_wrapped, dims_wrapped)
      fn = device_wrapper(jnp.cos(grid_wrapped["physical_coords"][:, :, :, 1]) *
                          jnp.cos(grid_wrapped["physical_coords"][:, :, :, 0]))
      assert (np.allclose(get_global_array(project_scalar(fn, grid_wrapped, dims_wrapped), dims_wrapped),
                          get_global_array(fn, dims_wrapped)))
      ones = jnp.ones_like(grid_wrapped["metric_determinant"])
      ones_out = project_scalar(device_wrapper(ones), grid_wrapped, dims_wrapped)
      assert (np.allclose(get_global_array(ones_out, dims_wrapped), get_global_array(ones, dims_wrapped)))
      ones_out_for = project_scalar_for(get_global_array(ones, dims_wrapped), raw_grid)
      assert (np.allclose(ones_out_for, get_global_array(ones, dims_wrapped)))


def test_projection_equiv_rand():
  for npt in test_npts:
    for nx in [7, 8]:
      face_connectivity, face_mask, face_position, face_position_2d = init_cube_topo(nx)
      vert_redundancy = init_element_corner_vert_redundancy(face_connectivity)
      raw_grid, dims = init_grid_from_topo(face_connectivity,
                                           face_mask,
                                           face_position_2d,
                                           vert_redundancy,
                                           npt, wrapped=False)
      grid_wrapped, dims_wrapped = init_grid_from_topo(face_connectivity,
                                                       face_mask,
                                                       face_position_2d,
                                                       vert_redundancy,
                                                       npt,
                                                       wrapped=use_wrapper)
      grid_wrapped = shard_grid(grid_wrapped, dims_wrapped)
      for _ in range(20):
        fn_rand = np.random.uniform(size=grid_wrapped["physical_coords"][:, :, :, 1].shape)
        assert np.allclose(get_global_array(project_scalar_wrapper(device_wrapper(fn_rand, elem_sharding_axis=0),
                                                                   grid_wrapped,
                                                                   dims_wrapped),
                                            dims_wrapped),
                           project_scalar_for(fn_rand[:dims["num_elem"], :, :], raw_grid))
