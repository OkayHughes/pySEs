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
    relevant_data = _be.index_get(scaled_f, (cols[0], cols[1], cols[2]))
    scaled_f = _be.index_add(scaled_f, (rows[0], rows[1], rows[2]), relevant_data)
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
  if do_sharding:
    scaled_f = scaled_f.reshape((num_jax_devices, -1, dims["npt"], dims["npt"]), out_sharding=projection_sharding)
    extraction_struct = grid["shard_extraction_map"]

    relevant_data = (scaled_f).at[extraction_struct["extract_from"]["shard_idx"],
                                  extraction_struct["extract_from"]["elem_idx"],
                                  extraction_struct["extract_from"]["i_idx"],
                                  extraction_struct["extract_from"]["j_idx"]].get(out_sharding=extraction_sharding)
    # Padded extraction slots carry mask == 0 and default (zero) sum_into
    # indices; a multiplicative mask would leave payload 0 which then scatters
    # into DOF (0, 0, 0) of every shard via `.max`, corrupting the min/max
    # there (e.g. collapsing a strictly-positive tracer lower bound to 0).
    # -inf is the identity for the max reduction (used directly for `max` and
    # on the negated field for `min`), so padded slots become inert.
    relevant_data = jnp.where(extraction_struct["mask"] > 0, relevant_data, -jnp.inf)
    scaled_f = do_max_manual_sharding(scaled_f,
                                      extraction_struct["sum_into"]["elem_idx"],
                                      extraction_struct["sum_into"]["i_idx"],
                                      extraction_struct["sum_into"]["j_idx"],
                                      relevant_data)
    scaled_f = scaled_f.reshape(shape, out_sharding=usual_scalar_sharding)
  else:
    relevant_data = _be.index_get(scaled_f, (cols[0], cols[1], cols[2]))
    scaled_f = _be.index_max(scaled_f, (rows[0], rows[1], rows[2]), relevant_data)
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


def _edge_color_rounds(undirected_edges, num_devices):
  """
  Greedy edge-coloring of the (undirected) device communication graph.

  Each color class is a matching, so it can be realised as one ``lax.ppermute``
  round of disjoint transpositions.  Greedy coloring uses at most ``2*deg - 1``
  colors and in practice ``~deg`` where ``deg`` is the max neighbor count, which
  is small and independent of ``num_devices`` for spatially-local (e.g.
  Hilbert-curve) decompositions.

  Parameters
  ----------
  undirected_edges : list[tuple[int, int]]
      Sorted ``(a, b)`` device pairs (``a < b``) that must communicate.
  num_devices : int
      Total device count.

  Returns
  -------
  partner : list[list[int]]
      ``partner[r][d]`` is the device ``d`` is matched with in round ``r``
      (``d`` itself if idle that round).  ``len(partner)`` is the round count.
  """
  used = [set() for _ in range(num_devices)]
  edge_color = {}
  for (a, b) in undirected_edges:
    c = 0
    while c in used[a] or c in used[b]:
      c += 1
    edge_color[(a, b)] = c
    used[a].add(c)
    used[b].add(c)
  num_rounds = (max(edge_color.values()) + 1) if edge_color else 0
  partner = [[d for d in range(num_devices)] for _ in range(num_rounds)]
  for (a, b), c in edge_color.items():
    partner[c][a] = b
    partner[c][b] = a
  return partner


def init_global_comm_map(assembly_triple, num_devices, nelem_padded, wrapped=True):
  """
  Build the sparse ppermute DSS schedule (the global device/rank map).

  Classifies every redundancy entry in the global assembly triple by the
  ``(source_device, target_device)`` pair under a contiguous block element
  decomposition (device ``d`` owns elements ``[d*E, (d+1)*E)`` where
  ``E = nelem_padded / num_devices``).  Same-device pairs become a purely local
  gather/scatter (the "self" term).  The cross-device pairs form a symmetric
  communication graph, which is edge-colored into ``num_rounds`` matchings (see
  :func:`_edge_color_rounds`).  Each round is one ``lax.ppermute`` of disjoint
  transpositions ``{a <-> b}``: ``a`` sends bucket ``(a, b)`` while ``b`` sends
  bucket ``(b, a)`` in the same collective.  The round count is ``~max_neighbors``
  rather than ``num_devices - 1``.

  Per device ``d`` and round ``r`` (matched with ``q = partner[r][d]``):

  * ``send_*[d, r, p]`` — gather index on ``d`` for bucket ``(d, q)`` (what ``d``
    ships to ``q``); masked by ``send_mask`` so padded slots ship zeros.
  * ``recv_*[d, r, p]`` — scatter-add index on ``d`` for bucket ``(q, d)`` (where
    the payload from ``q`` lands); slot ``p`` aligns with ``q``'s send slot ``p``,
    so padded receives are zeros and scatter harmlessly.

  The diagonal ``self_*`` arrays carry bucket ``(d, d)`` for the local term.

  Parameters
  ----------
  assembly_triple : tuple[Array, list[Array], list[Array]]
      Global assembly triple ``(data, rows, cols)`` from
      :func:`init_assembly_local` over all ``nelem_padded`` elements.
  num_devices : int
      Number of mesh devices to decompose across.
  nelem_padded : int
      Total element count after zero-padding; must be divisible by
      ``num_devices``.
  wrapped : bool, optional
      If ``True`` (default), index/mask arrays are wrapped onto the device and
      sharded along the device axis.

  Returns
  -------
  comm_map : dict
      ``perms`` (static per-round ppermute permutations), the device-axis-sharded
      ``send_*`` / ``recv_*`` / ``send_mask`` arrays of shape
      ``(num_devices, num_rounds, max_points_edge)``, the ``self_*`` arrays of
      shape ``(num_devices, max_points_self)``, and the scalars
      ``elems_per_device``, ``num_rounds``, ``max_points_edge``,
      ``max_points_self`` and ``num_devices``.
  """
  assert nelem_padded % num_devices == 0, "nelem_padded must be divisible by num_devices"
  E = nelem_padded // num_devices
  N = num_devices

  data, rows, cols = assembly_triple
  rows = [np.asarray(_be.unwrap(arr)) for arr in rows]
  cols = [np.asarray(_be.unwrap(arr)) for arr in cols]
  num_entries = np.asarray(_be.unwrap(data)).shape[0]

  # bucket[(src_device, dst_device)] = list of (src_loc, si, sj, tgt_loc, ti, tj)
  buckets = {}
  for k in range(num_entries):
    s_elem, s_i, s_j = int(cols[0][k]), int(cols[1][k]), int(cols[2][k])
    t_elem, t_i, t_j = int(rows[0][k]), int(rows[1][k]), int(rows[2][k])
    a, s_loc = divmod(s_elem, E)
    b, t_loc = divmod(t_elem, E)
    buckets.setdefault((a, b), []).append((s_loc, s_i, s_j, t_loc, t_i, t_j))

  undirected = sorted({(min(a, b), max(a, b)) for (a, b) in buckets if a != b})
  partner = _edge_color_rounds(undirected, N)
  num_rounds = len(partner)
  perms = tuple(tuple((d, partner[r][d]) for d in range(N)) for r in range(num_rounds))

  cross_lens = [len(v) for (k, v) in buckets.items() if k[0] != k[1]]
  max_pts_edge = max(cross_lens) if cross_lens else 1
  self_lens = [len(buckets[(d, d)]) for d in range(N) if (d, d) in buckets]
  max_pts_self = max(self_lens) if self_lens else 1

  e_shape = (N, num_rounds, max_pts_edge)
  send_elem = np.zeros(e_shape, dtype=np.int64)
  send_i = np.zeros(e_shape, dtype=np.int64)
  send_j = np.zeros(e_shape, dtype=np.int64)
  send_mask = np.zeros(e_shape, dtype=np.float64)
  recv_elem = np.zeros(e_shape, dtype=np.int64)
  recv_i = np.zeros(e_shape, dtype=np.int64)
  recv_j = np.zeros(e_shape, dtype=np.int64)

  for r in range(num_rounds):
    for d in range(N):
      q = partner[r][d]
      if q == d:
        continue
      for p, (s_loc, si, sj, _, _, _) in enumerate(buckets.get((d, q), [])):
        send_elem[d, r, p] = s_loc
        send_i[d, r, p] = si
        send_j[d, r, p] = sj
        send_mask[d, r, p] = 1.0
      for p, (_, _, _, t_loc, ti, tj) in enumerate(buckets.get((q, d), [])):
        recv_elem[d, r, p] = t_loc
        recv_i[d, r, p] = ti
        recv_j[d, r, p] = tj

  s_shape = (N, max_pts_self)
  self_gather_elem = np.zeros(s_shape, dtype=np.int64)
  self_gather_i = np.zeros(s_shape, dtype=np.int64)
  self_gather_j = np.zeros(s_shape, dtype=np.int64)
  self_mask = np.zeros(s_shape, dtype=np.float64)
  self_scatter_elem = np.zeros(s_shape, dtype=np.int64)
  self_scatter_i = np.zeros(s_shape, dtype=np.int64)
  self_scatter_j = np.zeros(s_shape, dtype=np.int64)
  for d in range(N):
    for p, (s_loc, si, sj, t_loc, ti, tj) in enumerate(buckets.get((d, d), [])):
      self_gather_elem[d, p] = s_loc
      self_gather_i[d, p] = si
      self_gather_j[d, p] = sj
      self_mask[d, p] = 1.0
      self_scatter_elem[d, p] = t_loc
      self_scatter_i[d, p] = ti
      self_scatter_j[d, p] = tj

  if wrapped:
    def wrap(x, dtype):
      return device_wrapper(x, dtype=dtype, elem_sharding_axis=0)
  else:
    def wrap(x, dtype):
      return x

  return {"perms": perms,
          "num_rounds": num_rounds,
          "max_points_edge": max_pts_edge,
          "max_points_self": max_pts_self,
          "elems_per_device": E,
          "num_devices": N,
          "send_elem": wrap(send_elem, jnp.int64),
          "send_i": wrap(send_i, jnp.int64),
          "send_j": wrap(send_j, jnp.int64),
          "send_mask": wrap(send_mask, jnp.float64),
          "recv_elem": wrap(recv_elem, jnp.int64),
          "recv_i": wrap(recv_i, jnp.int64),
          "recv_j": wrap(recv_j, jnp.int64),
          "self_gather_elem": wrap(self_gather_elem, jnp.int64),
          "self_gather_i": wrap(self_gather_i, jnp.int64),
          "self_gather_j": wrap(self_gather_j, jnp.int64),
          "self_mask": wrap(self_mask, jnp.float64),
          "self_scatter_elem": wrap(self_scatter_elem, jnp.int64),
          "self_scatter_i": wrap(self_scatter_i, jnp.int64),
          "self_scatter_j": wrap(self_scatter_j, jnp.int64)}
