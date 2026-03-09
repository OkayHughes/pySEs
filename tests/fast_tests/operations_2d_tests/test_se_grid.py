from src.mesh_generation.equiangular_metric import init_grid_from_topo, init_quasi_uniform_grid
from src.mesh_generation.cubed_sphere import init_cube_topo
from src.mesh_generation.mesh import init_element_corner_vert_redundancy
from src.mesh_generation.mesh import vert_red_flat_to_hierarchy, vert_red_hierarchy_to_flat
from src.mpi.processor_decomposition import init_decomp, local_to_global
from src.operations_2d.horizontal_grid import (triage_vert_redundancy_flat, init_assembly_global,
                                    init_assembly_local, make_grid_mpi_ready)
from ...test_data.handmade_grids import (vert_locals_ref,
                                         vert_recvs_ref,
                                         vert_sends_ref,
                                         vert_redundancy_gll,
                                         init_test_grid)
from src._config import get_backend as _get_backend
_be = _get_backend()
do_sharding = _be.do_sharding
import numpy as np
from ...context import test_npts


def test_vert_triage_artificial():
  NELEM = 2

  # check symmetry first
  nproc = 2
  decomp = init_decomp(NELEM, nproc)
  vert_locals = []
  vert_sends = []
  vert_recvs = []
  vert_red_gll_flat = vert_red_hierarchy_to_flat(vert_redundancy_gll)
  num_red = len(vert_red_gll_flat)
  assembly_triple = [np.array([0 for _ in range(num_red)]),
                     [np.array([vert_red_gll_flat[k_idx][0][0] for k_idx in range(num_red)], dtype=np.int64),
                      np.array([vert_red_gll_flat[k_idx][0][1] for k_idx in range(num_red)], dtype=np.int64),
                      np.array([vert_red_gll_flat[k_idx][0][2] for k_idx in range(num_red)], dtype=np.int64)],
                     [np.array([vert_red_gll_flat[k_idx][1][0] for k_idx in range(num_red)], dtype=np.int64),
                      np.array([vert_red_gll_flat[k_idx][1][1] for k_idx in range(num_red)], dtype=np.int64),
                      np.array([vert_red_gll_flat[k_idx][1][2] for k_idx in range(num_red)], dtype=np.int64)]]

  for proc_idx in range(nproc):
    vert_red_local, vert_red_send, vert_red_receive = triage_vert_redundancy_flat(assembly_triple, proc_idx, decomp)
    vert_locals.append(vert_red_flat_to_hierarchy(vert_red_local))
    vert_sends.append(vert_red_send)
    vert_recvs.append(vert_red_receive)

  # test pairwise first
  pairs_ref = {}
  for proc_idx in range(nproc):
    for target_face_idx in vert_locals_ref[proc_idx].keys():
      assert(target_face_idx in vert_locals[proc_idx].keys())
      for (target_i, target_j) in vert_locals_ref[proc_idx][target_face_idx].keys():
        assert (target_i, target_j) in vert_locals[proc_idx][target_face_idx].keys()
        for (source_face_idx, source_i, source_j) in vert_locals_ref[proc_idx][target_face_idx][(target_i, target_j)]:
          assert (source_face_idx, source_i, source_j) in vert_locals[proc_idx][target_face_idx][(target_i, target_j)]
          pair = ((target_face_idx, target_i, target_j),
                  (source_face_idx, source_i, source_j))
          if pair not in pairs_ref.keys():
            pairs_ref[pair] = 1
          else:
            pairs_ref[pair] += 1

  for proc_idx in range(nproc):
    for remote_proc_idx in vert_sends[proc_idx].keys():
      for (source_face_idx, source_i, source_j) in vert_sends[proc_idx][remote_proc_idx]:
        assert (source_face_idx, source_i, source_j) in vert_sends_ref[proc_idx][remote_proc_idx]
    for remote_proc_idx in vert_recvs[proc_idx].keys():
      for (target_face_idx, target_i, target_j) in vert_sends[proc_idx][remote_proc_idx]:
        assert (target_face_idx, target_i, target_j) in vert_recvs_ref[proc_idx][remote_proc_idx]

  # test handmade reference soln
  for proc_idx in range(nproc):
    for remote_proc_idx in vert_sends[proc_idx].keys():
      for k in range(len(vert_sends[proc_idx][remote_proc_idx])):
        send_face_idx, send_i, send_j = vert_sends_ref[proc_idx][remote_proc_idx][k]
        recv_face_idx, recv_i, recv_j = vert_recvs_ref[remote_proc_idx][proc_idx][k]
        send_global_idx = local_to_global(send_face_idx, proc_idx, decomp)
        recv_global_idx = local_to_global(recv_face_idx, remote_proc_idx, decomp)
        assert (recv_global_idx, recv_i, recv_j) in vert_redundancy_gll[send_global_idx][(send_i, send_j)]
        assert (send_global_idx, send_i, send_j) in vert_redundancy_gll[recv_global_idx][(recv_i, recv_j)]

  # test generated vert_redundancy structs
  for proc_idx in range(nproc):
    for remote_proc_idx in vert_sends[proc_idx].keys():
      for k in range(len(vert_sends[proc_idx][remote_proc_idx])):
        send_face_idx, send_i, send_j = vert_sends[proc_idx][remote_proc_idx][k]
        recv_face_idx, recv_i, recv_j = vert_recvs[remote_proc_idx][proc_idx][k]
        send_global_idx = local_to_global(send_face_idx, proc_idx, decomp)
        recv_global_idx = local_to_global(recv_face_idx, remote_proc_idx, decomp)
        assert (recv_global_idx, recv_i, recv_j) in vert_redundancy_gll[send_global_idx][(send_i, send_j)]
        assert (send_global_idx, send_i, send_j) in vert_redundancy_gll[recv_global_idx][(recv_i, recv_j)]


def test_vert_red_triage():
  if do_sharding:
    return
  for npt in test_npts:
    for nproc in range(1, 6):
      for nx in range(2, 5):
        face_connectivity, face_mask, face_position, face_position_2d = init_cube_topo(nx)
        vert_redundancy = init_element_corner_vert_redundancy(face_connectivity)
        grid, dims = init_grid_from_topo(face_connectivity,
                                         face_mask,
                                         face_position_2d,
                                         vert_redundancy,
                                         npt,
                                         wrapped=False)
        decomp = init_decomp(dims["num_elem"], nproc)
        vert_redundancy_gll = vert_red_flat_to_hierarchy(grid["vertex_redundancy"])
        vert_redundancy_check = {}
        for face_idx in vert_redundancy_gll.keys():
          for (i_idx, j_idx) in vert_redundancy_gll[face_idx].keys():
            for remote_face_idx, remote_i, remote_j in vert_redundancy_gll[face_idx][(i_idx, j_idx)]:
              pair = (face_idx, i_idx, j_idx)
              if pair not in vert_redundancy_check.keys():
                vert_redundancy_check[pair] = 1
              else:
                vert_redundancy_check[pair] += 1

        vert_locals = []
        vert_sends = []
        vert_recvs = []
        for proc_idx in range(nproc):
          vert_red_local, vert_red_send, vert_red_receive = triage_vert_redundancy_flat(grid["assembly_triple"],
                                                                                        proc_idx,
                                                                                        decomp)
          vert_red_local = vert_red_flat_to_hierarchy(vert_red_local)
          vert_locals.append(vert_red_local)
          vert_sends.append(vert_red_send)
          vert_recvs.append(vert_red_receive)
        # test that all redundancies are accounted for
        vert_red_decomp = {}
        for proc_idx in range(nproc):
          for face_idx in vert_locals[proc_idx].keys():
            for (i_idx, j_idx) in vert_locals[proc_idx][face_idx].keys():
              for remote_face_idx, remote_i, remote_j in vert_locals[proc_idx][face_idx][(i_idx, j_idx)]:
                pair = (local_to_global(face_idx, proc_idx, decomp), i_idx, j_idx)
                if pair not in vert_red_decomp.keys():
                  vert_red_decomp[pair] = 1
                else:
                  vert_red_decomp[pair] += 1
          for remote_proc_idx in vert_recvs[proc_idx].keys():
            for k in range(len(vert_recvs[proc_idx][remote_proc_idx])):
              local_face_idx, local_i, local_j = vert_recvs[proc_idx][remote_proc_idx][k]
              pair = (local_to_global(local_face_idx, proc_idx, decomp), local_i, local_j)
              if pair not in vert_red_decomp.keys():
                vert_red_decomp[pair] = 1
              else:
                vert_red_decomp[pair] += 1
        for key in vert_redundancy_check.keys():
          assert key in vert_red_decomp.keys()
          assert vert_redundancy_check[key] == vert_red_decomp[key]
        # test that ordering of send/receive buffers agree
        # using vert_redundancy_gll as ground truth.
        for proc_idx in range(nproc):
          for remote_proc_idx in vert_sends[proc_idx].keys():
            for k in range(len(vert_sends[proc_idx][remote_proc_idx])):
              send_face_idx, send_i, send_j = vert_sends[proc_idx][remote_proc_idx][k]
              recv_face_idx, recv_i, recv_j = vert_recvs[remote_proc_idx][proc_idx][k]
              send_global_idx = local_to_global(send_face_idx, proc_idx, decomp)
              recv_global_idx = local_to_global(recv_face_idx, remote_proc_idx, decomp)
              assert (recv_global_idx, recv_i, recv_j) in vert_redundancy_gll[send_global_idx][(send_i, send_j)]
              assert (send_global_idx, send_i, send_j) in vert_redundancy_gll[recv_global_idx][(recv_i, recv_j)]


def test_assembly_init():
  if do_sharding:
    return
  se_grid, dims = init_test_grid()
  assert np.allclose(np.sum(se_grid["metric_determinant"] *
                            (se_grid["gll_weights"][np.newaxis, :, np.newaxis] *
                             se_grid["gll_weights"][np.newaxis, np.newaxis, :])), 8.0)
  nproc = 2
  decomp = init_decomp(2, 2)

  def flip_triple(triple):
    return [x for x in zip(triple[0],
                           [fij for fij in zip(triple[1][0],
                                               triple[1][1],
                                               triple[1][2])],
                           [fij for fij in zip(triple[2][0],
                                               triple[2][1],
                                               triple[2][2])])]

  vert_red_total = flip_triple(se_grid["assembly_triple"])
  ct = len(vert_red_total)
  for proc_idx in range(nproc):
    vert_red_local, vert_red_send, vert_red_recv = triage_vert_redundancy_flat(se_grid["assembly_triple"],
                                                                               proc_idx, decomp)
    se_grid_subset, dims_subset = make_grid_mpi_ready(se_grid, dims, proc_idx, decomp, wrapped=False)
    assembly_triple = init_assembly_local(vert_red_local)
    ct -= len(flip_triple(assembly_triple))
    triples_send, triples_receive = init_assembly_global(vert_red_send, vert_red_recv)
    for remote_proc_idx in triples_send.keys():
      assert (remote_proc_idx in triples_receive.keys())
      # column refers to order of summation, which may not match.
      flip_trip_send = [x for x in flip_triple(triples_send[remote_proc_idx])]
      flip_double_send = [(x[0], x[2]) for x in flip_trip_send]
      ct -= len(flip_trip_send)
      flip_trip_recv = [x for x in flip_triple(triples_receive[remote_proc_idx])]
      flip_double_recv = [(x[0], x[1]) for x in flip_trip_recv]
      for (value, row, col) in flip_trip_send:
        assert (value, col) in flip_double_recv
      for (value, row, col) in flip_trip_recv:
        assert (value, row) in flip_double_send
  assert ct == 0


def test_triples_order():
  if do_sharding:
    return
  for npt in test_npts:
    for nx in range(1, 5):
      for nproc in range(1, 3):
        grid_total, dim_total = init_quasi_uniform_grid(nx, npt, wrapped=True)
        grid_total_n, dim_total_n = init_quasi_uniform_grid(nx, npt, wrapped=False)
        decomp = init_decomp(dim_total["num_elem"], nproc)
        grids = []
        grids_nowrapper = []
        dims = []
        for proc_idx in range(nproc):
          grid, dim = make_grid_mpi_ready(grid_total, dim_total, proc_idx, decomp, wrapped=True)
          grid_nowrapper, dim_nowrapper = make_grid_mpi_ready(grid_total_n,
                                                              dim_total_n,
                                                              proc_idx,
                                                              decomp,
                                                              wrapped=False)
          grids.append(grid)
          grids_nowrapper.append(grid_nowrapper)
          dims.append(dim)
        for local_proc_idx in range(nproc):
          # check that triples and vert redundancy structs are identical
          local_triples_send = grids[local_proc_idx]["triples_send"]
          local_triples_recv = grids[local_proc_idx]["triples_receive"]
          local_vert_red_send = grids_nowrapper[local_proc_idx]["vertex_redundancy_send"]
          local_vert_red_recv = grids_nowrapper[local_proc_idx]["vertex_redundancy_receive"]
          local_coords = grids[local_proc_idx]["physical_coords"]
          for remote_proc_idx in local_triples_send.keys():
            assert remote_proc_idx in local_triples_recv.keys()
            assert remote_proc_idx in local_vert_red_send.keys()
            assert remote_proc_idx in local_vert_red_recv.keys()
            remote_triples_send = grids[remote_proc_idx]["triples_send"]
            remote_triples_recv = grids[remote_proc_idx]["triples_receive"]
            remote_coords = grids[remote_proc_idx]["physical_coords"]
            for k_idx in range(list(local_triples_send[remote_proc_idx][0].shape)[0]):
              f_idx, i_idx, j_idx = local_vert_red_send[remote_proc_idx][k_idx]
              # test that vert_red_send struct and triples_send point to coincident local points

              def unwrap_rowcol(rowcol, k_idx):
                f_out = rowcol[0][k_idx]
                i_out = rowcol[1][k_idx]
                j_out = rowcol[2][k_idx]
                return f_out, i_out, j_out

              f_out, i_out, j_out = unwrap_rowcol(local_triples_send[remote_proc_idx][2], k_idx)
              assert(np.allclose(local_coords[f_idx, i_idx, j_idx, 0],
                                 local_coords[f_out, i_out, j_out, 0]))
              f_out, i_out, j_out = unwrap_rowcol(local_triples_send[remote_proc_idx][2], k_idx)
              assert(np.allclose(local_coords[f_idx, i_idx, j_idx, 1],
                                 local_coords[f_out, i_out, j_out, 1]))
              f_out, i_out, j_out = unwrap_rowcol(remote_triples_recv[local_proc_idx][1], k_idx)
              # test that local vert_red_send struct and remote triples_recv point to coincident points
              assert(np.allclose(local_coords[f_idx, i_idx, j_idx, 0],
                                 remote_coords[f_out, i_out, j_out, 0]))
              f_out, i_out, j_out = unwrap_rowcol(remote_triples_recv[local_proc_idx][1], k_idx)
              assert(np.allclose(local_coords[f_idx, i_idx, j_idx, 1],
                                 remote_coords[f_out, i_out, j_out, 1]))
              f_idx, i_idx, j_idx = local_vert_red_recv[remote_proc_idx][k_idx]
              # test that vert_red_recv struct and triples_recv point to coincident local points
              f_out, i_out, j_out = unwrap_rowcol(local_triples_recv[remote_proc_idx][1], k_idx)
              assert(np.allclose(local_coords[f_idx, i_idx, j_idx, 0],
                                 local_coords[f_out, i_out, j_out, 0]))
              f_out, i_out, j_out = unwrap_rowcol(local_triples_recv[remote_proc_idx][1], k_idx)
              assert(np.allclose(local_coords[f_idx, i_idx, j_idx, 1],
                                 local_coords[f_out, i_out, j_out, 1]))
              # test that local vert_red_recv struct and remote triples_send point to coincident points
              f_out, i_out, j_out = unwrap_rowcol(remote_triples_send[local_proc_idx][2], k_idx)
              assert(np.allclose(local_coords[f_idx, i_idx, j_idx, 0],
                                 remote_coords[f_out, i_out, j_out, 0]))
              f_out, i_out, j_out = unwrap_rowcol(remote_triples_send[local_proc_idx][2], k_idx)
              assert(np.allclose(local_coords[f_idx, i_idx, j_idx, 1],
                                 remote_coords[f_out, i_out, j_out, 1]))
