import numpy as np
from .._config import get_backend as _get_backend
from ..mpi.processor_decomposition import global_to_local, elem_idx_global_to_proc_idx
from scipy.sparse import coo_array
from functools import partial
_be = _get_backend()
jnp = _be.np
use_wrapper = _be.use_wrapper
jit = _be.jit
wrapper_type = _be.wrapper_type
DEBUG = _be.debug
device_wrapper = _be.array
shard_map = _be.shard_map
num_jax_devices = _be.num_jax_devices
projection_sharding = _be.projection_sharding
extraction_sharding = _be.extraction_sharding
usual_scalar_sharding = _be.usual_scalar_sharding
do_sharding = _be.do_sharding
if use_wrapper and wrapper_type == "jax":
  from jax.sharding import PartitionSpec
  import jax
  shard_map_extract = partial(jax.shard_map,
                              in_specs=(PartitionSpec("f", None, None, None),
                                        PartitionSpec("f", None),
                                        PartitionSpec("f", None),
                                        PartitionSpec("f", None),
                                        PartitionSpec("f", None)),
                              out_specs=PartitionSpec("f", None, None, None))
else:
  shard_map_extract = shard_map


def project_scalar_sparse(f,
                          grid,
                          matrix,
                          *args):
  """
  Project a potentially discontinuous scalar onto the continuous
  subspace using a sparse matrix, assuming all data is processor-local.

  *This is used for testing. Do not use in performance code*

  Parameters
  ----------
  f : `Array[tuple[elem_idx, gll_idx, gll_idx], Float]`
    Scalar field to project
  grid : `SpectralElementGrid`
    Spectral element grid struct that contains coordinate and metric data.
  scaled: `bool`, default=True
    Should f be scaled by the mass matrix before being summed?

  Notes
  -----
  * When using weak operators (i.e. in hyperviscosity),
  the resulting values are already scaled by the mass matrix.
  * In an ideal world, even performance device code would use a
  version of code that treats assembly, or Direct Stiffness Summation,
  as the application of a linear projection operator.
  However, support for sparse matrices in automatic differentiation libraries
  is bizarrely *ahem* sparse.

  Returns
  -------
  f_cont
      The globally continous scalar closest in norm to f.
  """
  vals_scaled = f * grid["mass_matrix"]
  ret = vals_scaled + (matrix @ (vals_scaled).flatten()).reshape(f.shape)
  return ret * grid["mass_matrix_denominator"]


def segment_sum(field,
                data,
                segment_ids):
  """
  A function that provides a numpy equivalent of the `segment_sum` function
  from Jax and TensorFlow.

  Parameters
  ----------
  data : Array[tuple[point_idx], Float]
      The floating point values to sum over
  segment_ids : Array[tuple[point_idx], Int]
      The indices in the result array into which to sum data.
      That is, `result[segment_idx[p]] += data[p]
  N: int
      The number of bins in which to sum

  Returns
  -------
  s: Array[tuple[N], Float]
      arrays into which segments have been summed.
  """
  data = np.asarray(data)
  np.add.at(field, (segment_ids[0], segment_ids[1], segment_ids[2]), data)


def segment_max(field,
                data,
                segment_ids):
  """
  A function that provides a numpy equivalent of the `segment_sum` function
  from Jax and TensorFlow.

  Parameters
  ----------
  data : Array[tuple[point_idx], Float]
      The floating point values to sum over
  segment_ids : Array[tuple[point_idx], Int]
      The indices in the result array into which to sum data.
      That is, `result[segment_idx[p]] += data[p]
  N: int
      The number of bins in which to sum

  Returns
  -------
  s: Array[tuple[N], Float]
      arrays into which segments have been summed.
  """

  data = np.asarray(data)
  s = np.copy(field)
  np.maximum.at(s, (segment_ids[0], segment_ids[1], segment_ids[2]), data)
  return s


@shard_map_extract
def do_sum_manual_sharding(scaled_f, elem_idx, i_idx, j_idx, relevant_data):
  """
  Sum redundant DOF values into a sharded field (JAX multi-device helper).

  Applied via ``shard_map`` so that each device accumulates only its own
  portion of the redundant values.

  Parameters
  ----------
  scaled_f : Array[tuple[1, local_elem_idx, gll_idx, gll_idx], Float]
      Mass-scaled field shard to accumulate into.
  elem_idx : Array[tuple[point_idx], Int]
      Element indices of target DOFs.
  i_idx : Array[tuple[point_idx], Int]
      First GLL indices of target DOFs.
  j_idx : Array[tuple[point_idx], Int]
      Second GLL indices of target DOFs.
  relevant_data : Array[tuple[point_idx], Float]
      Values to add at the target DOFs.

  Returns
  -------
  scaled_f : Array[tuple[1, local_elem_idx, gll_idx, gll_idx], Float]
      Updated field with redundant values accumulated.
  """
  scaled_f = scaled_f.at[0, elem_idx, i_idx, j_idx].add(relevant_data)
  return scaled_f


@partial(jit, static_argnames=["dims"])
def project_scalar_wrapper(f,
                           grid,
                           dims):
  """
  Project a potentially discontinuous scalar onto the continuous subspace using assembly triples,
  assuming all data is processor-local.

  Parameters
  ----------
  f : `Array[tuple[elem_idx, gll_idx, gll_idx], Float]`
    Scalar field to project
  grid : `SpectralElementGrid`
    Spectral element grid struct that contains coordinate and metric data.
  dims : frozendict[str, int]
      Scalar grid dimensions (``shape``, ``npt``).

  Notes
  -----
  * When using weak operators (i.e. in hyperviscosity),
  the resulting values are already scaled by the mass matrix.
  * This routine is allowed to depend on wrapper_type.

  Returns
  -------
  f_cont
      The globally continous scalar closest in norm to f.
  """
  (data, rows, cols) = grid["assembly_triple"]
  shape = f.shape

  scaled_f = f * grid["mass_matrix"]
  if use_wrapper and wrapper_type == "jax":
    if do_sharding:
      scaled_f = scaled_f.reshape((num_jax_devices, -1, dims["npt"], dims["npt"]), out_sharding=projection_sharding)
      extraction_struct = grid["shard_extraction_map"]

      relevant_data = (scaled_f).at[extraction_struct["extract_from"]["shard_idx"],
                                    extraction_struct["extract_from"]["elem_idx"],
                                    extraction_struct["extract_from"]["i_idx"],
                                    extraction_struct["extract_from"]["j_idx"]].get(out_sharding=extraction_sharding)
      relevant_data *= extraction_struct["mask"]
      scaled_f = do_sum_manual_sharding(scaled_f,
                                        extraction_struct["sum_into"]["elem_idx"],
                                        extraction_struct["sum_into"]["i_idx"],
                                        extraction_struct["sum_into"]["j_idx"],
                                        relevant_data)
      scaled_f = scaled_f.reshape(shape, out_sharding=usual_scalar_sharding)
    else:
      relevant_data = (scaled_f).at[cols[0], cols[1], cols[2]].get()
      scaled_f = scaled_f.at[rows[0], rows[1], rows[2]].add(relevant_data)
  elif use_wrapper and wrapper_type == "torch":
    # this is broken
    scaled_f = scaled_f.flatten()
    scaled_f = scaled_f.scatter_add_(0, rows, relevant_data)
    scaled_f = scaled_f.reshape(dims["shape"])
    relevant_data = scaled_f[cols[0], cols[1], cols[2]]
  else:
    relevant_data = scaled_f[cols[0], cols[1], cols[2]]
    segment_sum(scaled_f, relevant_data, rows)
  return scaled_f * grid["mass_matrix_denominator"]


@shard_map_extract
def do_max_manual_sharding(scaled_f, elem_idx, i_idx, j_idx, relevant_data):
  """
  Accumulate the element-wise maximum into a sharded field (JAX multi-device helper).

  Applied via ``shard_map`` so that each device computes the local maximum
  of its redundant DOF values.

  Parameters
  ----------
  scaled_f : Array[tuple[1, local_elem_idx, gll_idx, gll_idx], Float]
      Field shard to accumulate into.
  elem_idx : Array[tuple[point_idx], Int]
      Element indices of target DOFs.
  i_idx : Array[tuple[point_idx], Int]
      First GLL indices of target DOFs.
  j_idx : Array[tuple[point_idx], Int]
      Second GLL indices of target DOFs.
  relevant_data : Array[tuple[point_idx], Float]
      Values to take the maximum against at the target DOFs.

  Returns
  -------
  scaled_f : Array[tuple[1, local_elem_idx, gll_idx, gll_idx], Float]
      Updated field with element-wise maximums accumulated.
  """
  scaled_f = scaled_f.at[0, elem_idx, i_idx, j_idx].max(relevant_data)
  return scaled_f


@partial(jit, static_argnames=["dims", "max"])
def minmax_scalar(f,
                  grid,
                  dims,
                  max=True):
  """
  Project a potentially discontinuous scalar onto the continuous subspace using assembly triples,
  assuming all data is processor-local.

  Parameters
  ----------
  f : `Array[tuple[elem_idx, gll_idx, gll_idx], Float]`
    Scalar field to project
  grid : `SpectralElementGrid`
    Spectral element grid struct that contains coordinate and metric data.
  dims : frozendict[str, int]
    Scalar grid dimensions (``shape``, ``npt``).
  max : bool, default=True
    If true, compute maximum over redundant DOFs. Otherwise, compute min.

  Notes
  -----
  * When using weak operators (i.e. in hyperviscosity),
  the resulting values are already scaled by the mass matrix.
  * This routine is allowed to depend on wrapper_type.

  Returns
  -------
  f_cont
      The globally continous scalar closest in norm to f.
  """
  (data, rows, cols) = grid["assembly_triple"]
  shape = f.shape
  if max:
    scaled_f = 1.0 * f
  else:
    scaled_f = -1.0 * f
  if use_wrapper and wrapper_type == "jax":
    if do_sharding:
      scaled_f = scaled_f.reshape((num_jax_devices, -1, dims["npt"], dims["npt"]), out_sharding=projection_sharding)
      extraction_struct = grid["shard_extraction_map"]

      relevant_data = (scaled_f).at[extraction_struct["extract_from"]["shard_idx"],
                                    extraction_struct["extract_from"]["elem_idx"],
                                    extraction_struct["extract_from"]["i_idx"],
                                    extraction_struct["extract_from"]["j_idx"]].get(out_sharding=extraction_sharding)
      relevant_data *= extraction_struct["mask"]
      scaled_f = do_max_manual_sharding(scaled_f,
                                        extraction_struct["sum_into"]["elem_idx"],
                                        extraction_struct["sum_into"]["i_idx"],
                                        extraction_struct["sum_into"]["j_idx"],
                                        relevant_data)
      scaled_f = scaled_f.reshape(shape, out_sharding=usual_scalar_sharding)
    else:
      relevant_data = (scaled_f).at[cols[0], cols[1], cols[2]].get()
      scaled_f = scaled_f.at[rows[0], rows[1], rows[2]].max(relevant_data)
  else:
    relevant_data = scaled_f[cols[0], cols[1], cols[2]]
    scaled_f = segment_max(scaled_f, relevant_data, rows)
  if not max:
    scaled_f = -1.0 * scaled_f
  return scaled_f


project_scalar = project_scalar_wrapper


def init_assembly_matrix(NELEM,
                         npt,
                         assembly_triple):
  """
  Build a sparse DSS assembly matrix from an assembly triple.

  Constructs a COO-format sparse matrix whose application is equivalent to
  the Direct Stiffness Summation (DSS) projection over the global GLL DOFs.

  Parameters
  ----------
  NELEM : int
      Total number of elements in the grid.
  npt : int
      Number of GLL points per element edge.
  assembly_triple : tuple[Array, list[Array], list[Array]]
      Assembly triple ``(data, rows, cols)`` from :func:`init_assembly_local`.

  Returns
  -------
  assembly_matrix : scipy.sparse.coo_array
      Sparse assembly matrix of shape ``(NELEM*npt^2, NELEM*npt^2)``.
  """
  data, rows, cols = assembly_triple
  assembly_matrix = coo_array((data, (rows, cols)), shape=(NELEM * npt * npt, NELEM * npt * npt))
  return assembly_matrix


def init_assembly_local(vert_redundancy_local):
  """
  Build the processor-local DSS assembly triple from a flat redundancy list.

  Converts a flat list of ``(target, source)`` GLL DOF pairs into the
  ``(data, rows, cols)`` triple used by :func:`project_scalar_wrapper` and
  related assembly routines.  All DOF indices are processor-local.

  Parameters
  ----------
  vert_redundancy_local : list[tuple[tuple[int,int,int], tuple[int,int,int]]]
      Flat list of ``((target_elem, target_i, target_j), (source_elem, source_i, source_j))``
      pairs describing coincident GLL DOFs on the local processor.

  Returns
  -------
  assembly_triple : tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]
      ``(data, rows, cols)`` where ``data`` contains the assembly weights
      (all 1.0) and ``rows``/``cols`` are lists of three index arrays each
      ``[elem_idx, i_idx, j_idx]``.
  """
  # From this moment forward, we assume that
  # vert_redundancy_gll contains only the information
  # for processor-local GLL things,
  # and that remote_face_idx corresponds to processor local
  # ids.
  # hack: easier than figuring out indexing conventions

  data = []
  rows = [[], [], []]
  cols = [[], [], []]

  for ((local_face_idx, local_i, local_j),
       (remote_face_id, remote_i, remote_j)) in vert_redundancy_local:
    data.append(1.0)
    rows[0].append(remote_face_id)
    rows[1].append(remote_i)
    rows[2].append(remote_j)
    cols[0].append(local_face_idx)
    cols[1].append(local_i)
    cols[2].append(local_j)
  # print(f"nonzero entries: {dss_matrix.nnz}, total entries: {(NELEM * npt * npt)**2}")
  return (np.array(data, dtype=np.float64),
          [np.array(arr, dtype=np.int64) for arr in rows],
          [np.array(arr, dtype=np.int64) for arr in cols])


def init_assembly_global(vert_redundancy_send,
                         vert_redundancy_receive):
  """
  Build per-processor send/receive assembly triples for MPI DSS communication.

  Converts the send and receive vertex-redundancy lists (containing
  processor-local DOF indices) into ``(data, rows, cols)`` triples keyed
  by remote processor index, ready for use in :func:`extract_fields` and
  :func:`accumulate_fields`.

  Parameters
  ----------
  vert_redundancy_send : dict[int, list[tuple[int,int,int]]]
      Mapping from remote processor index to a list of local
      ``(elem_idx, i_idx, j_idx)`` DOFs to send to that processor.
  vert_redundancy_receive : dict[int, list[tuple[int,int,int]]]
      Mapping from remote processor index to a list of local
      ``(elem_idx, i_idx, j_idx)`` DOFs into which received values are
      summed.

  Returns
  -------
  triples_send : dict[int, tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]]
      Per-processor send triples ``(data, rows, cols)`` for extraction.
  triples_receive : dict[int, tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]]
      Per-processor receive triples ``(data, rows, cols)`` for accumulation.
  """
  # From this moment forward, we assume that
  # vert_redundancy_gll contains only the information
  # for processor-local GLL things,
  # and that remote_face_idx corresponds to processor local
  # ids.
  # hack: easier than figuring out indexing conventions

  # convention: when scaled=True, remote values are
  # pre-multiplied by numerator
  # divided by total mass matrix on receiving end
  triples_receive = {}
  triples_send = {}
  for vert_redundancy, triples, transpose in zip([vert_redundancy_receive, vert_redundancy_send],
                                                 [triples_receive, triples_send],
                                                 [False, True]):
    for source_proc_idx in vert_redundancy.keys():
      data = []
      rows = [[], [], []]
      cols = [[], [], []]
      for col_idx, (target_local_idx, target_i, target_j) in enumerate(vert_redundancy[source_proc_idx]):
        data.append(1.0)
        if transpose:
          cols[0].append(target_local_idx)
          cols[1].append(target_i)
          cols[2].append(target_j)
          rows[0].append(col_idx)
          rows[1].append(col_idx)
          rows[2].append(col_idx)
        else:
          rows[0].append(target_local_idx)
          rows[1].append(target_i)
          rows[2].append(target_j)
          cols[0].append(col_idx)
          cols[1].append(col_idx)
          cols[2].append(col_idx)
      triples[source_proc_idx] = (np.array(data, dtype=np.float64),
                                  [np.array(arr, dtype=np.int64) for arr in rows],
                                  [np.array(arr, dtype=np.int64) for arr in cols])
  # print(f"nonzero entries: {dss_matrix.nnz}, total entries: {(NELEM * npt * npt)**2}")
  return triples_send, triples_receive


def triage_vert_redundancy_flat(assembly_triple,
                                proc_idx,
                                decomp):
  """
  Partition a global flat redundancy list into local, send, and receive components.

  Classifies each ``(target, source)`` GLL DOF pair in the global assembly
  triple according to whether both DOFs are local to ``proc_idx``, the target
  is local but the source is remote (receive), or the source is local but the
  target is remote (send).

  Parameters
  ----------
  assembly_triple : tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]
      Global assembly triple ``(data, rows, cols)`` from
      :func:`init_assembly_local` on the full grid.
  proc_idx : int
      MPI rank of the calling processor.
  decomp : sequence of tuple[int, int]
      Element-range decomposition returned by :func:`init_decomp`.
      ``decomp[proc_idx]`` gives the ``(start, end)`` global element indices
      owned by processor ``proc_idx``.

  Returns
  -------
  vert_redundancy_local : list[tuple[tuple[int,int,int], tuple[int,int,int]]]
      Flat redundancy pairs where both DOFs are on ``proc_idx``
      (using local element indices).
  vert_redundancy_send : dict[int, list[tuple[int,int,int]]]
      Mapping from remote processor index to local DOFs that must be sent.
  vert_redundancy_receive : dict[int, list[tuple[int,int,int]]]
      Mapping from remote processor index to local DOFs into which received
      values will be summed.
  """
  # current understanding: this works because the outer
  # three for loops will iterate in exactly the same order for
  # the sending and recieving processor
  vert_redundancy_gll_flat = [((assembly_triple[1][0][k_idx],
                                assembly_triple[1][1][k_idx],
                                assembly_triple[1][2][k_idx]),
                               (assembly_triple[2][0][k_idx],
                                assembly_triple[2][1][k_idx],
                                assembly_triple[2][2][k_idx])) for k_idx in range(assembly_triple[0].shape[0])]
  vert_redundancy_local = []
  vert_redundancy_send = {}
  vert_redundancy_receive = {}

  for ((target_global_idx, target_i, target_j),
       (source_global_idx, source_i, source_j)) in vert_redundancy_gll_flat:
    target_proc_idx = elem_idx_global_to_proc_idx(target_global_idx, decomp)
    source_proc_idx = elem_idx_global_to_proc_idx(source_global_idx, decomp)
    if (target_proc_idx == proc_idx and source_proc_idx == proc_idx):
      target_local_idx = int(global_to_local(target_global_idx, proc_idx, decomp))
      source_local_idx = int(global_to_local(source_global_idx, proc_idx, decomp))
      vert_redundancy_local.append(((target_local_idx, target_i, target_j),
                                    (source_local_idx, source_i, source_j)))
    elif (target_proc_idx == proc_idx and not
          source_proc_idx == proc_idx):
      target_local_idx = int(global_to_local(target_global_idx, proc_idx, decomp))
      if source_proc_idx not in vert_redundancy_receive.keys():
        vert_redundancy_receive[source_proc_idx] = []
      vert_redundancy_receive[source_proc_idx].append(((target_local_idx, target_i, target_j)))
    elif (not target_proc_idx == proc_idx and
          source_proc_idx == proc_idx):
      source_local_idx = int(global_to_local(source_global_idx, proc_idx, decomp))
      if target_proc_idx not in vert_redundancy_send.keys():
        vert_redundancy_send[target_proc_idx] = []
      vert_redundancy_send[target_proc_idx].append(((source_local_idx, source_i, source_j)))
  return vert_redundancy_local, vert_redundancy_send, vert_redundancy_receive


def init_shard_extraction_map(assembly_triple, num_devices, nelem_padded, dims, wrapped=True):
  """
  Build the padded index arrays used for manual DSS sharding across JAX devices.

  For each entry in the global assembly triple, determines which device
  (shard) holds the source and target DOFs and assembles padded index
  tensors of shape ``(num_devices, max_dof)`` that can be passed to
  :func:`do_sum_manual_sharding` and :func:`do_max_manual_sharding` via
  ``shard_map``.

  Parameters
  ----------
  assembly_triple : tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]
      Global assembly triple ``(data, rows, cols)`` for the (possibly padded)
      element array.
  num_devices : int
      Number of JAX devices to shard across.
  nelem_padded : int
      Total number of elements after zero-padding; must be divisible by
      ``num_devices``.
  dims : frozendict[str, int]
      Grid dimension metadata (currently unused; reserved for future use).
  wrapped : bool, optional
      If ``True``, index arrays are wrapped with :func:`device_wrapper`
      before being stored.  Defaults to ``True``.

  Returns
  -------
  extraction_map : dict
      Dict with ``"sum_into"`` and ``"extract_from"`` sub-dicts, each
      containing padded ``shard_idx``, ``elem_idx``, ``i_idx``, ``j_idx``
      arrays of shape ``(num_devices, max_dof)``, plus a ``"mask"``
      coefficient array of the same shape.
  max_dof : int
      Maximum number of redundant DOFs on any single device.
  """
  # this will maybe eventually be rewritten to work for bigger grids?
  assert np.abs(np.round(nelem_padded / num_devices) - nelem_padded / num_devices) < 1e-6, "Did you pad your array?"

  if wrapped:
    wrapper = device_wrapper
  else:
    def wrapper(x, *args, **kwargs):
      return x

  shard_idx, elem_idx = np.meshgrid(np.arange(num_devices, dtype=np.int64),
                                    np.arange(np.round(nelem_padded / num_devices), dtype=np.int64))
  shard_flat = shard_idx.flatten(order="F")
  elem_flat = elem_idx.flatten(order="F")
  sum_into_shard = [[] for _ in range(num_devices)]
  extract_from_shard = [[] for _ in range(num_devices)]

  for ((f_row, i_row, j_row),
       (f_col, i_col, j_col)) in zip(zip(assembly_triple[1][0],
                                         assembly_triple[1][1],
                                         assembly_triple[1][2]),
                                     zip(assembly_triple[2][0],
                                         assembly_triple[2][1],
                                         assembly_triple[2][2])):
    sum_into_shard_idx = shard_flat[f_row]
    shard_local_elem_idx = elem_flat[f_row]
    extract_from_shard_idx = shard_flat[f_col]
    shard_remote_elem_idx = elem_flat[f_col]
    sum_into_shard[sum_into_shard_idx].append([sum_into_shard_idx, shard_local_elem_idx, i_row, j_row])
    extract_from_shard[sum_into_shard_idx].append([extract_from_shard_idx, shard_remote_elem_idx, i_col, j_col])
  max_dof = max([len(x) for x in sum_into_shard])
  if DEBUG:
    max_dof_maybe = max([len(x) for x in extract_from_shard])
    assert max_dof == max_dof_maybe
  size_of_comm = (num_devices, max_dof)
  coeff_mat = np.zeros(size_of_comm, dtype=np.float64)
  sum_into_shard_idxs = np.zeros(size_of_comm, dtype=np.int64)
  sum_into_elem_idxs = np.zeros(size_of_comm, dtype=np.int64)
  sum_into_i_idxs = np.zeros(size_of_comm, dtype=np.int64)
  sum_into_j_idxs = np.zeros(size_of_comm, dtype=np.int64)
  extract_from_shard_idxs = np.zeros(size_of_comm, dtype=np.int64)
  extract_from_elem_idxs = np.zeros(size_of_comm, dtype=np.int64)
  extract_from_i_idxs = np.zeros(size_of_comm, dtype=np.int64)
  extract_from_j_idxs = np.zeros(size_of_comm, dtype=np.int64)
  for shard_idx in range(num_devices):
    sum_into_data = np.array(sum_into_shard[shard_idx])
    extract_from_data = np.array(extract_from_shard[shard_idx])
    if sum_into_data.ndim == 1:
      sum_into_data = sum_into_data[np.newaxis, :]
    if extract_from_data.ndim == 1:
      extract_from_data = extract_from_data[np.newaxis, :]
    num_pts = sum_into_data.shape[0]
    if DEBUG:
      num_pts_maybe = extract_from_data.shape[0]
      assert num_pts == num_pts_maybe

    sum_into_shard_idxs[shard_idx, :num_pts] = sum_into_data[:, 0]
    sum_into_elem_idxs[shard_idx, :num_pts] = sum_into_data[:, 1]
    sum_into_i_idxs[shard_idx, :num_pts] = sum_into_data[:, 2]
    sum_into_j_idxs[shard_idx, :num_pts] = sum_into_data[:, 3]

    extract_from_shard_idxs[shard_idx, :num_pts] = extract_from_data[:, 0]
    extract_from_elem_idxs[shard_idx, :num_pts] = extract_from_data[:, 1]
    extract_from_i_idxs[shard_idx, :num_pts] = extract_from_data[:, 2]
    extract_from_j_idxs[shard_idx, :num_pts] = extract_from_data[:, 3]
    coeff_mat[shard_idx, :num_pts] = 1.0
  return {"sum_into": {"shard_idx": wrapper(sum_into_shard_idxs, dtype=jnp.int64, elem_sharding_axis=0),
                       "elem_idx": wrapper(sum_into_elem_idxs, dtype=jnp.int64, elem_sharding_axis=0),
                       "i_idx": wrapper(sum_into_i_idxs, dtype=jnp.int64, elem_sharding_axis=0),
                       "j_idx": wrapper(sum_into_j_idxs, dtype=jnp.int64, elem_sharding_axis=0)},
          "extract_from": {"shard_idx": wrapper(extract_from_shard_idxs, dtype=jnp.int64, elem_sharding_axis=0),
                           "elem_idx": wrapper(extract_from_elem_idxs, dtype=jnp.int64, elem_sharding_axis=0),
                           "i_idx": wrapper(extract_from_i_idxs, dtype=jnp.int64, elem_sharding_axis=0),
                           "j_idx": wrapper(extract_from_j_idxs, dtype=jnp.int64, elem_sharding_axis=0)},
          "mask": wrapper(coeff_mat, dtype=jnp.int64, elem_sharding_axis=0)}, max_dof
