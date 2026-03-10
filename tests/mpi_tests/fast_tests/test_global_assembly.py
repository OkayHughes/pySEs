from src.mesh_generation.equiangular_metric import init_quasi_uniform_grid
from ...reference_implementations.reference_global_assembly import (project_scalar_for_stub,
                                                                    assemble_scalar_for_pack,
                                                                    assemble_scalar_for_unpack,
                                                                    extract_fields_for,
                                                                    accumulate_fields_for,
                                                                    project_scalar_for_mpi)
from src.mpi.global_communication import (_exchange_buffers_stub)
from src.mpi.global_assembly import (extract_fields, accumulate_fields, _project_scalar_stub,
                                     project_scalar_global)
from src.mesh_generation.mesh import vert_red_flat_to_hierarchy
from src.operations_2d.horizontal_grid import make_grid_mpi_ready
from src.mesh_generation.periodic_plane import init_uniform_grid
from src.mpi.processor_decomposition import init_decomp, elem_idx_global_to_proc_idx, global_to_local
from src._config import get_backend as _get_backend
import numpy as np
from ...test_data.handmade_grids import init_test_grid, vert_redundancy_gll
from ...context import test_npts
from ...context import seed as global_seed
_be = _get_backend()
device_unwrapper = _be.unwrap
use_wrapper = _be.use_wrapper
device_wrapper = _be.array
mpi_size = _be.mpi_size
mpi_rank = _be.mpi_rank


def test_unordered_assembly_for_stub():
  for npt in test_npts:
    for nx in range(1, 3):
      for nproc in range(1, 6):
        print(f"dividing nx {nx} grid among {nproc} processors")
        grid_total, dim_total = init_quasi_uniform_grid(nx, npt, wrapped=False)
        vert_redundancy = vert_red_flat_to_hierarchy(grid_total["vertex_redundancy"])
        decomp = init_decomp(dim_total["num_elem"], nproc)
        grids = []
        dims = []
        fs = []
        total_elems = 0
        pairs = {}
        proc_ids = elem_idx_global_to_proc_idx(np.arange(dim_total["num_elem"]), decomp)
        for target_face_idx in vert_redundancy.keys():
          for (target_i, target_j) in vert_redundancy[target_face_idx].keys():
            pair_key = (proc_ids[target_face_idx],
                        global_to_local(target_face_idx, proc_ids[target_face_idx], decomp),
                        target_i,
                        target_j)
            pair_set = set()
            for (source_face_idx, source_i, source_j) in vert_redundancy[target_face_idx][(target_i, target_j)]:
              pair_set.add((proc_ids[source_face_idx],
                            global_to_local(source_face_idx, proc_ids[source_face_idx], decomp),
                            source_i,
                            source_j))
            pairs[pair_key] = pair_set.copy()
        for proc_idx in range(nproc):
          grid, dim = make_grid_mpi_ready(grid_total, dim_total, proc_idx, decomp=decomp)
          grids.append(grid)
          dims.append(dim)
          total_elems += dim["num_elem"]
        assert (dim_total["num_elem"] == total_elems)

        def zeros_f():
          fs = []
          for grid in grids:
            fs.append([np.zeros_like(grid["physical_coords"][:, :, :, 0])])
          return fs

        # test one scalar at a time
        for pair_key in pairs.keys():
          fs = zeros_f()
          target_proc_idx, target_local_face_idx, target_i, target_j = pair_key
          fs[target_proc_idx][0][target_local_face_idx, target_i, target_j] = 1.0
          for source_proc_idx, source_local_face_idx, source_i, source_j in pairs[pair_key]:
            fs[source_proc_idx][0][source_local_face_idx, source_i, source_j] = 1.0
          fs_ref = [[np.copy(f[0])] for f in fs]
          fs_out = project_scalar_for_stub(fs, grids)
          for (f, f_new) in zip(fs_ref, fs_out):
            assert (np.allclose(f[0], f_new[0]))

        # test all scalars at once
        fs_ref = [[] for _ in range(nproc)]
        fs = [[] for _ in range(nproc)]
        for pair_key in pairs.keys():
          fs_tmp = zeros_f()
          target_proc_idx, target_local_face_idx, target_i, target_j = pair_key
          fs_tmp[target_proc_idx][0][target_local_face_idx, target_i, target_j] = 1.0
          for source_proc_idx, source_local_face_idx, source_i, source_j in pairs[pair_key]:
            fs_tmp[source_proc_idx][0][source_local_face_idx, source_i, source_j] = 1.0
          for proc_idx in range(nproc):
            fs_ref[proc_idx].append(np.copy(fs_tmp[proc_idx][0]))
            fs[proc_idx].append(np.copy(fs_tmp[proc_idx][0]))
        fs_out = project_scalar_for_stub(fs, grids)
        for (f, f_new) in zip(fs_ref, fs_out):
          for (f_pair, f_new_pair) in zip(f, f_new):
            assert (np.allclose(f_pair, f_new_pair))


def test_unordered_assembly_triple_stub():
  for npt in test_npts:
    for nx in range(1, 3):
      for nproc in range(1, 6):
        print(f"dividing nx {nx} grid among {nproc} processors")
        grid_total, dim_total = init_quasi_uniform_grid(nx, npt, wrapped=False)
        decomp = init_decomp(dim_total["num_elem"], nproc)
        grids = []
        dims = []
        fs = []
        total_elems = 0
        pairs = {}
        proc_ids = elem_idx_global_to_proc_idx(np.arange(dim_total["num_elem"]), decomp)
        vert_redundancy = vert_red_flat_to_hierarchy(grid_total["vertex_redundancy"])
        for target_face_idx in vert_redundancy.keys():
          for (target_i, target_j) in vert_redundancy[target_face_idx].keys():
            pair_key = (proc_ids[target_face_idx],
                        global_to_local(target_face_idx, proc_ids[target_face_idx], decomp),
                        target_i,
                        target_j)
            pair_set = set()
            for (source_face_idx, source_i, source_j) in vert_redundancy[target_face_idx][(target_i, target_j)]:
              pair_set.add((proc_ids[source_face_idx],
                            global_to_local(source_face_idx, proc_ids[source_face_idx], decomp),
                            source_i,
                            source_j))
            pairs[pair_key] = pair_set.copy()
        for proc_idx in range(nproc):
          grid, dim = make_grid_mpi_ready(grid_total, dim_total, proc_idx, decomp=decomp)
          grids.append(grid)
          dims.append(dim)
          total_elems += dim["num_elem"]
        assert (dim_total["num_elem"] == total_elems)

        def zeros_f():
          fs = []
          for grid in grids:
            fs.append([np.zeros_like(grid["physical_coords"][:, :, :, 0])])
          return fs

        # test one scalar at a time
        for pair_key in pairs.keys():
          fs = zeros_f()
          target_proc_idx, target_local_face_idx, target_i, target_j = pair_key
          fs[target_proc_idx][0][target_local_face_idx, target_i, target_j] = 1.0
          for source_proc_idx, source_local_face_idx, source_i, source_j in pairs[pair_key]:
            fs[source_proc_idx][0][source_local_face_idx, source_i, source_j] = 1.0
          for source_proc_idx in range(len(fs)):
            fs[source_proc_idx][0] = device_wrapper(fs[source_proc_idx][0])
          fs_ref = [[np.copy(f[0])] for f in fs]
          fs_out = _project_scalar_stub(fs, grids, dims)
          for (f, f_new) in zip(fs_ref, fs_out):
            assert (np.allclose(f[0], device_unwrapper(f_new[0])))

        # test all scalars at once
        fs_ref = [[] for _ in range(nproc)]
        fs = [[] for _ in range(nproc)]
        for pair_key in pairs.keys():
          fs_tmp = zeros_f()
          target_proc_idx, target_local_face_idx, target_i, target_j = pair_key
          fs_tmp[target_proc_idx][0][target_local_face_idx, target_i, target_j] = 1.0
          for source_proc_idx, source_local_face_idx, source_i, source_j in pairs[pair_key]:
            fs_tmp[source_proc_idx][0][source_local_face_idx, source_i, source_j] = 1.0
          for proc_idx in range(nproc):
            fs_ref[proc_idx].append(device_wrapper(fs_tmp[proc_idx][0]))
            fs[proc_idx].append(device_wrapper(fs_tmp[proc_idx][0]))
        fs_out = _project_scalar_stub(fs, grids, dims)
        for (f, f_new) in zip(fs_ref, fs_out):
          for (f_pair, f_new_pair) in zip(f, f_new):
            assert (np.allclose(device_unwrapper(f_pair), device_unwrapper(f_new_pair)))


def test_stub_exchange():
  NELEM = 2
  nproc = 2
  se_grid, total_dims = init_test_grid()
  npt = total_dims["npt"]

  decomp = init_decomp(NELEM, nproc)
  proc_ids = elem_idx_global_to_proc_idx(np.array([0, 1]), decomp)
  # set associated vertices to 1,
  # then test that extract, communicate, accumulate returns multiplicity
  pairs = {}
  for target_face_idx in vert_redundancy_gll.keys():
    for (target_i, target_j) in vert_redundancy_gll[target_face_idx].keys():
      pair_key = (proc_ids[target_face_idx],
                  global_to_local(target_face_idx, proc_ids[target_face_idx], decomp),
                  target_i,
                  target_j)
      pair_set = set()
      for (source_face_idx, source_i, source_j) in vert_redundancy_gll[target_face_idx][(target_i, target_j)]:
        pair_set.add((proc_ids[source_face_idx],
                      global_to_local(source_face_idx, proc_ids[source_face_idx], decomp),
                      source_i,
                      source_j))
      pairs[pair_key] = pair_set.copy()
  grids = []
  dims = []

  for proc_idx in range(nproc):
    grid, dim = make_grid_mpi_ready(se_grid, total_dims, proc_idx, decomp=decomp)
    grids.append(grid)
    dims.append(dim)

  def zeros_f():
    fs = []
    for grid in grids:
      fs.append([np.zeros_like(grid["physical_coords"][:, :, :, 0])])
    return fs

  for pair_key in pairs.keys():
    fs = zeros_f()
    target_proc_idx, target_local_face_idx, target_i, target_j = pair_key
    fs[target_proc_idx][0][target_local_face_idx, target_i, target_j] = 1.0
    for source_proc_idx, source_local_face_idx, source_i, source_j in pairs[pair_key]:
      fs[source_proc_idx][0][source_local_face_idx, source_i, source_j] = 1.0
    fs_ref = [np.copy(f) for f in fs]
    fs_out_cont = project_scalar_for_stub(fs, grids)
    buffers = []
    for (f, grid) in zip(fs, grids):
      buffers.append(assemble_scalar_for_pack(f, grid))
    buffers = _exchange_buffers_stub(buffers)
    fs_out = []
    for (f, grid, buffer) in zip(fs, grids, buffers):
      fs_out.append(assemble_scalar_for_unpack(f, buffer, grid))
    if (source_i, source_j) in {(0, 0),
                                (npt - 1, npt - 1),
                                (0, npt - 1),
                                (npt - 1, 0)}:
      mult = 3.0
    elif source_j == 0 or source_j == npt - 1:
      mult = 2.0
    else:
      mult = 1.0
    assert np.allclose(fs_out[target_proc_idx][0][target_local_face_idx, target_i, target_j], mult)
    fs_out[target_proc_idx][0][target_local_face_idx, target_i, target_j] -= mult
    for source_proc_idx, source_local_face_idx, source_i, source_j in pairs[pair_key]:
      assert np.allclose(fs_out[source_proc_idx][0][source_local_face_idx, source_i, source_j], mult)
      fs_out[source_proc_idx][0][source_local_face_idx, source_i, source_j] -= mult
    for f in fs_out:
      assert(np.allclose(f[0], 0.0))
    for (f, f_new) in zip(fs_ref, fs_out_cont):
     assert (np.allclose(f, f_new))


def test_extract_fields_triples():
  for npt in test_npts:
    np.random.seed(global_seed)
    for nx in range(1, 3):
      global_grids_nowrapper = [init_uniform_grid(nx, nx + 1, npt, wrapped=False),
                                init_quasi_uniform_grid(nx, npt, wrapped=False)]
      global_grids = [init_uniform_grid(nx, nx + 1, npt, wrapped=use_wrapper),
                      init_quasi_uniform_grid(nx, npt, wrapped=use_wrapper)]
      for ((grid_total, dim_total),
           (grid_total_nowrapper, dim_total_nowrapper)) in zip(global_grids, global_grids_nowrapper):
        for random, num_iters in zip([False, True], [1, 10]):
          for _ in range(num_iters):
            nproc = 2
            decomp = init_decomp(dim_total["num_elem"], nproc)
            grids = []
            grids_nodevice = []
            dims = []
            fs = []
            total_elems = 0
            nlev = 2
            pairs = {}
            proc_ids = elem_idx_global_to_proc_idx(np.arange(dim_total["num_elem"]), decomp)
            vert_redundancy = vert_red_flat_to_hierarchy(grid_total_nowrapper["vertex_redundancy"])
            for target_face_idx in vert_redundancy.keys():
              for (target_i, target_j) in vert_redundancy[target_face_idx].keys():
                pair_key = (proc_ids[target_face_idx],
                            global_to_local(target_face_idx, proc_ids[target_face_idx], decomp),
                            target_i,
                            target_j)
                pair_set = set()
                for (source_face_idx, source_i, source_j) in vert_redundancy[target_face_idx][(target_i, target_j)]:
                  pair_set.add((proc_ids[source_face_idx],
                                global_to_local(source_face_idx, proc_ids[source_face_idx], decomp),
                                source_i,
                                source_j))
                pairs[pair_key] = pair_set.copy()
            for proc_idx in range(nproc):
              grid, dim = make_grid_mpi_ready(grid_total, dim_total, proc_idx, decomp=decomp)
              grid_nodevice, _ = make_grid_mpi_ready(grid_total_nowrapper,
                                                     dim_total_nowrapper,
                                                     proc_idx,
                                                     decomp=decomp)
              grids.append(grid)
              grids_nodevice.append(grid_nodevice)
              dims.append(dim)
              if random:
                fs.append([np.random.normal(size=(*grid_nodevice["physical_coords"][:, :, :, 0].shape, nlev)),
                           np.random.normal(size=(*grid_nodevice["physical_coords"][:, :, :, 0].shape, nlev + 1))])
              else:
                f1 = np.arange(1, grid_nodevice["physical_coords"][:, :, :, 0].size + 1)
                f1 = f1.reshape(grid_nodevice["physical_coords"][:, :, :, 0].shape)[:, :, :, np.newaxis]
                fs.append([f1 * np.ones((1, 1, 1, nlev)),
                           f1 * np.ones((1, 1, 1, nlev + 1))])
              total_elems += dim["num_elem"]
            fs_device = [[device_wrapper(f) for f in fs_local] for fs_local in fs]
            buffers_for = []
            buffers_device = []
            for (f, f_device, grid, grid_nodevice, dim) in zip(fs, fs_device, grids, grids_nodevice, dims):
              buffers_for.append(extract_fields_for(f, grid_nodevice["vertex_redundancy_send"]))
              buffer_for = buffers_for[-1]
              buffers_device.append(extract_fields(f_device, grid["triples_send"]))
              buffer_device = buffers_device[-1]
              for proc_idx in buffer_for.keys():
                assert proc_idx in buffer_device.keys()
                assert len(buffer_for[proc_idx]) == len(buffer_device[proc_idx])
                for k_idx in range(len(buffer_for[proc_idx])):
                  assert buffer_for[proc_idx][k_idx].shape == buffer_device[proc_idx][k_idx].shape
                  assert np.allclose(buffer_for[proc_idx][k_idx], device_unwrapper(buffer_device[proc_idx][k_idx]))
            buffers_for = _exchange_buffers_stub(buffers_for)
            buffers_device = _exchange_buffers_stub(buffers_device)
            for (f, f_device, grid, grid_nowrapper, dim, buffer_for, buffer_device) in zip(fs,
                                                                                           fs_device,
                                                                                           grids,
                                                                                           grids_nodevice,
                                                                                           dims,
                                                                                           buffers_for,
                                                                                           buffers_device):
              for remote_idx in buffer_for.keys():
                for field_idx in range(len(buffer_for[remote_idx])):
                  assert np.allclose(buffer_for[remote_idx][field_idx],
                                     device_unwrapper(buffer_device[remote_idx][field_idx]))
                  assert np.allclose(f[field_idx],
                                     device_unwrapper(f_device[field_idx]))
              fijk_fields_for = accumulate_fields_for(f, buffer_for, grid_nowrapper["vertex_redundancy_receive"])
              fijk_fields_triple = accumulate_fields(f_device, buffer_device, grid["triples_receive"], dim)
              assert len(fijk_fields_for) == len(fijk_fields_triple)
              for field_idx in range(len(fijk_fields_for)):
                assert np.allclose(f[field_idx], device_unwrapper(f_device[field_idx]))
                assert np.allclose(fijk_fields_for[field_idx], device_unwrapper(fijk_fields_triple[field_idx]))


def test_mpi_exchange_for():
  for npt in test_npts:
    for nx in range(1, 3):
      nproc = mpi_size
      local_proc_idx = mpi_rank
      print(f"dividing nx {nx} grid among {nproc} processors")
      grid_total, dim_total = init_quasi_uniform_grid(nx, npt, wrapped=False)
      decomp = init_decomp(dim_total["num_elem"], nproc)
      grids = []
      dims = []
      fs = []
      total_elems = 0
      pairs = {}
      proc_ids = elem_idx_global_to_proc_idx(np.arange(dim_total["num_elem"]), decomp)
      vert_redundancy = vert_red_flat_to_hierarchy(grid_total["vertex_redundancy"])
      for target_face_idx in vert_redundancy.keys():
        for (target_i, target_j) in vert_redundancy[target_face_idx].keys():
          pair_key = (proc_ids[target_face_idx],
                      global_to_local(target_face_idx, proc_ids[target_face_idx], decomp),
                      target_i,
                      target_j)
          pair_set = set()
          for (source_face_idx, source_i, source_j) in vert_redundancy[target_face_idx][(target_i, target_j)]:
            pair_set.add((proc_ids[source_face_idx],
                          global_to_local(source_face_idx, proc_ids[source_face_idx], decomp),
                          source_i,
                          source_j))
          pairs[pair_key] = pair_set.copy()
      for proc_idx in range(nproc):
        grid, dim = make_grid_mpi_ready(grid_total, dim_total, proc_idx, decomp=decomp)
        grids.append(grid)
        dims.append(dim)
        total_elems += dim["num_elem"]
      assert (dim_total["num_elem"] == total_elems)
      grid_local = grids[local_proc_idx]

      def zeros_f():
        fs = []
        for grid in grids:
          fs.append([np.zeros_like(grid["physical_coords"][:, :, :, 0])])
        return fs

      def rand_f(seed=global_seed):
        fs = []
        for grid_idx, grid in enumerate(grids):
          np.random.seed(grid_idx + seed)
          fs.append([np.random.uniform(grid["physical_coords"][:, :, :, 0])])
        return fs

      # test one scalar at a time
      for pair_key in pairs.keys():
        fs = zeros_f()
        target_proc_idx, target_local_face_idx, target_i, target_j = pair_key
        fs[target_proc_idx][0][target_local_face_idx, target_i, target_j] = 1.0
        for source_proc_idx, source_local_face_idx, source_i, source_j in pairs[pair_key]:
          fs[source_proc_idx][0][source_local_face_idx, source_i, source_j] = 1.0
        fs_ref = [[np.copy(f[0])] for f in fs]
        fs_out = project_scalar_for_stub(fs, grids)
        f_out = project_scalar_for_mpi(fs[local_proc_idx], grid_local)
        assert (np.allclose(fs_ref[local_proc_idx][0], fs_out[local_proc_idx][0]))
        assert (np.allclose(fs_ref[local_proc_idx][0], f_out))

      # test all scalars at once
      fs_ref = [[] for _ in range(nproc)]
      fs = [[] for _ in range(nproc)]
      for pair_key in pairs.keys():
        fs_tmp = zeros_f()
        target_proc_idx, target_local_face_idx, target_i, target_j = pair_key
        fs_tmp[target_proc_idx][0][target_local_face_idx, target_i, target_j] = 1.0
        for source_proc_idx, source_local_face_idx, source_i, source_j in pairs[pair_key]:
          fs_tmp[source_proc_idx][0][source_local_face_idx, source_i, source_j] = 1.0
        for proc_idx in range(nproc):
          fs_ref[proc_idx].append(np.copy(fs_tmp[proc_idx][0]))
          fs[proc_idx].append(np.copy(fs_tmp[proc_idx][0]))
      fs_out = project_scalar_for_stub(fs, grids)
      f_out = project_scalar_for_mpi(fs[local_proc_idx], grid_local)
      for (f_pair, f_new_pair) in zip(fs_out[local_proc_idx], f_out):
        assert (np.allclose(f_pair, f_new_pair))
      num_fields = 20
      fs_rand = [[] for _ in range(nproc)]
      for seed in range(num_fields):
        f_rand = rand_f()
        for proc_idx in range(nproc):
          fs_rand[proc_idx].append(np.copy(f_rand[proc_idx][0]))
      fs_stub_out = project_scalar_for_stub(fs_rand, grids)
      f_out = project_scalar_for_mpi(fs_rand[local_proc_idx], grid_local)
      for (f_stub, f_mpi) in zip(fs_stub_out[local_proc_idx], f_out):
        assert (np.allclose(f_stub, f_mpi))


def test_mpi_exchange_triple():
  for npt in test_npts:
    for nx in range(1, 3):
      nproc = mpi_size
      local_proc_idx = mpi_rank
      print(f"dividing nx {nx} grid among {nproc} processors")
      grid_total_nowrapper, dim_total_nowrapper = init_quasi_uniform_grid(nx, npt, wrapped=False)
      grid_total, dim_total = init_quasi_uniform_grid(nx, npt, wrapped=use_wrapper)
      decomp = init_decomp(dim_total["num_elem"], nproc)
      grids = []
      dims = []
      fs = []
      total_elems = 0
      pairs = {}
      proc_ids = elem_idx_global_to_proc_idx(np.arange(dim_total["num_elem"]), decomp)
      vert_redundancy = vert_red_flat_to_hierarchy(grid_total_nowrapper["vertex_redundancy"])
      for target_face_idx in vert_redundancy.keys():
        for (target_i, target_j) in vert_redundancy[target_face_idx].keys():
          pair_key = (proc_ids[target_face_idx],
                      global_to_local(target_face_idx, proc_ids[target_face_idx], decomp),
                      target_i,
                      target_j)
          pair_set = set()
          for (source_face_idx, source_i, source_j) in vert_redundancy[target_face_idx][(target_i, target_j)]:
            pair_set.add((proc_ids[source_face_idx],
                          global_to_local(source_face_idx, proc_ids[source_face_idx], decomp),
                          source_i,
                          source_j))
          pairs[pair_key] = pair_set.copy()
      for proc_idx in range(nproc):
        grid, dim = make_grid_mpi_ready(grid_total, dim_total, proc_idx, decomp=decomp)
        grids.append(grid)
        dims.append(dim)
        total_elems += dim["num_elem"]
      assert (dim_total["num_elem"] == total_elems)
      grid_local = grids[local_proc_idx]
      dim_local = dims[local_proc_idx]

      def zeros_f():
        fs = []
        for grid in grids:
          fs.append([np.zeros_like(grid["physical_coords"][:, :, :, 0])])
        return fs

      def rand_f(seed=0):
        fs = []
        for grid_idx, grid in enumerate(grids):
          np.random.seed(grid_idx + seed)
          fs.append([np.random.uniform(grid["physical_coords"][:, :, :, 0])])
        return fs

      # test one scalar at a time
      for pair_key in pairs.keys():
        fs = zeros_f()
        target_proc_idx, target_local_face_idx, target_i, target_j = pair_key
        fs[target_proc_idx][0][target_local_face_idx, target_i, target_j] = 1.0
        for source_proc_idx, source_local_face_idx, source_i, source_j in pairs[pair_key]:
          fs[source_proc_idx][0][source_local_face_idx, source_i, source_j] = 1.0
        for source_proc_idx in range(len(fs)):
          fs[source_proc_idx][0] = device_wrapper(fs[source_proc_idx][0])
        fs_ref = [[np.copy(f[0])] for f in fs]
        fs_out = _project_scalar_stub(fs, grids, dims)
        f_out = project_scalar_global(fs[local_proc_idx], grid_local, dim_local)[0]
        assert (np.allclose(fs_ref[local_proc_idx][0], device_unwrapper(fs_out[local_proc_idx][0])))
        assert (np.allclose(fs_ref[local_proc_idx][0], device_unwrapper(f_out)))

      # test all scalars at once
      fs_ref = [[] for _ in range(nproc)]
      fs = [[] for _ in range(nproc)]
      for pair_key in pairs.keys():
        fs_tmp = zeros_f()
        target_proc_idx, target_local_face_idx, target_i, target_j = pair_key
        fs_tmp[target_proc_idx][0][target_local_face_idx, target_i, target_j] = 1.0
        for source_proc_idx, source_local_face_idx, source_i, source_j in pairs[pair_key]:
          fs_tmp[source_proc_idx][0][source_local_face_idx, source_i, source_j] = 1.0
        for proc_idx in range(nproc):
          fs_ref[proc_idx].append(np.copy(fs_tmp[proc_idx][0]))
          fs[proc_idx].append(device_wrapper(fs_tmp[proc_idx][0]))
      fs_out = _project_scalar_stub(fs, grids, dims)
      f_out = project_scalar_global(fs[local_proc_idx], grid_local, dim_local)
      for (f_pair, f_new_pair) in zip(fs_out[local_proc_idx], f_out):
        assert (np.allclose(f_pair, f_new_pair))
      num_fields = 20
      fs_rand = [[] for _ in range(nproc)]
      for seed in range(num_fields):
        f_rand = rand_f()
        for proc_idx in range(nproc):
          fs_rand[proc_idx].append(device_wrapper(f_rand[proc_idx][0]))
      fs_stub_out = _project_scalar_stub(fs_rand, grids, dims)
      f_out = project_scalar_global(fs_rand[local_proc_idx], grid_local, dim_local)
      for (f_stub, f_mpi) in zip(fs_stub_out[local_proc_idx], f_out):
        assert (np.allclose(f_stub, f_mpi))
