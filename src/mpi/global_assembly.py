import numpy as np
from .._config import get_backend as _get_backend
from .global_communication import exchange_buffers, _exchange_buffers_stub
from functools import partial
_be = _get_backend()
jnp = _be.np
use_wrapper = _be.use_wrapper
wrapper_type = _be.wrapper_type
jit = _be.jit
mpi_rank = _be.mpi_rank


def sum_into(fijk_field,
             buffer,
             rows,
             dims):
  """
  Sum redundant dofs into a 3D field using assembly triples.

  Parameters
  ----------
  fijk_field: `Array[tuple[elem_idx, gll_idx, gll_idx, level_idx], Float]`
      Field values into which the values in buffer will be summed
  buffer: `Array[tuple[point_idx, level_idx], Float]`
      Redundant DOFs
  rows: `Array[tuple[point_idx], Int]`
      Indexes of redundant dofs, namely
      `fijk_field[:, :, :, k_idx].flatten()[row[point_idx]]` corresponds to
      buffer[point_idx, k_idx].
  dims: `frozendict[str, Any]`
      Hashable dictionary containing key "shape", which is the shape of `fijk_field`.

  Returns
  -------
  fijk_field: `Array[tuple[elem_idx, gll_idx, gll_idx, level_idx], Float]`
      Field values into which the values in buffer have been summed

  Notes
  -----
  The implementation of this function is allowed to depend on wrapper_type.

  """
  if not use_wrapper:
    np.add.at(fijk_field, (rows[0], rows[1], rows[2], slice(None, None)), buffer.T)
  elif wrapper_type == "jax":
    fijk_field = fijk_field.at[rows[0], rows[1], rows[2], :].add(buffer.T)
  elif wrapper_type == "torch":
    ncol = fijk_field.shape[-1]
    fijk_field = fijk_field.reshape((-1, fijk_field.shape[-1]))
    fijk_field = fijk_field.scatter_add_(0, rows[:, np.newaxis] * jnp.ones((1, ncol), dtype=jnp.int64), buffer.T)
    fijk_field = fijk_field.reshape((*dims["shape"], ncol))
  return fijk_field


@jit
def extract_fields(fijk_fields,
                   triples_send):
  """
  Extract values from a list of processor-local 3D fields before
  inter-process communication using assembly triples.

  *This can be used in performance code.*

  Parameters
  ----------
  fijk_fields : `list[Array[tuple[elem_idx, gll_idx, gll_idx, level_idx], Float]]`
      List of fields from which to extract redundant dofs to ship off
      to remote processors.
  triples_send : `dict[remote_proc_idx, tuple[Array[tuple[point_idx], Float],\
                                            Array[tuple[point_idx], Int],\
                                            Array[tuple[point_idx], Int]]`
      Mapping from `remote_proc_idx` to an assembly triple describing info
      to ship off to `remote_proc_idx`.

      Values are extracted as `fijk_fields[field_idx][elem_idx, i_idx, j_idx, :]`

  Returns
  -------
  `dict[remote_proc_idx, list[Array[tuple[point_idx, level_idx], Float]]]`
      A mapping from `remote_proc_idx` to a list
      of buffers of redundant DOFs, preserving the number of levels in each field.

  Notes
  -----
  The number of levels in `fijk_fields[field_idx]` may vary across `field_idx`.

  """
  buffers = {}
  for remote_proc_idx in triples_send.keys():
    buffers[remote_proc_idx] = []
    for field_idx in range(len(fijk_fields)):
      (data, rows, cols) = triples_send[remote_proc_idx]
      if not use_wrapper:
        relevant_data = fijk_fields[field_idx][cols[0], cols[1], cols[2], :]
      else:
        relevant_data = fijk_fields[field_idx].at[cols[0], cols[1], cols[2], :].get()
      buffers[remote_proc_idx].append(relevant_data.T)
  return buffers


def accumulate_fields(fijk_fields,
                      buffers,
                      triples_receive,
                      dims):
  """
  Sum non-processor-local redundant DOFs into a list of processor-local 3D fields after
  inter-process communication using assembly triples.

  *Can be used in performance code.*

  Parameters
  ----------
  fijk_fields : `list[Array[tuple[elem_idx, gll_idx, gll_idx, level_idx], Float]]`
      List of fields from which to extract redundant dofs to ship off
      to remote processors.
  buffers: `dict[remote_proc_idx, [Array[tuple[point_idx, level_idx], Float]]]`
      a mapping from `remote_proc_idx` to a list
      of buffers of redundant DOFs, preserving the number of levels in each field.
  triples_receive: `dict[remote_proc_idx, tuple[Array[tuple[point_idx], Float],\
                                               Array[tuple[point_idx], Int],\
                                               Array[tuple[point_idx], Int]]`
      Mapping from remote_proc_idx to a list of tuples of processor-local indexes into which
      redundant DOFs will be summed.

  Notes
  -----
  The number of levels in `fijk_fields[field_idx]` may vary across `field_idx`.

  Returns
  -------
  fijk_fields : `list[Array[tuple[elem_idx, gll_idx, gll_idx, level_idx], Float]]`
      List of fields into which non-processor-local redundant dofs have been summed.
  """
  for remote_proc_idx in buffers.keys():
    for field_idx in range(len(fijk_fields)):
      (_, rows, _) = triples_receive[remote_proc_idx]
      fijk_fields[field_idx] = sum_into(fijk_fields[field_idx],
                                        buffers[remote_proc_idx][field_idx],
                                        rows, dims)
  return fijk_fields


@jit
def pack_scalar(fs_local,
                grid):
  """
  Extract processor-local list of scalars before
  inter-process communication using assembly triples.

  Assembly triples are designed for use in wrapped code, and
  have the best shot of being compatible with automatic differentiation.

  Parameters
  ----------
  fs_local : `list[Array[tuple[elem_idx, gll_idx, gll_idx, level_idx], Float]]`
      List of 2D scalar fields to extract for communication
  grid : `dict[str, Any]`
      Processor-local Spectral Element Grid struct.
      Contains "triples_send" key, which
      has type `dict[remote_proc_idx, tuple[Array[tuple[point_idx], Float],\
                                           Array[tuple[point_idx], Int],\
                                           Array[tuple[point_idx], Int]]]`

  Returns
  -------
  `dict[proc_idx, list[Array[tuple[local_point_idx, 1], Float]]`
      Mapping from the destination processor idx to an array
      of values where `buffer[remote_proc_idx][pt_idx, 1]` is being sent to
      `remote_proc_idx` and summed into
      `point_idx = triples_receive[local_proc_idx][1][pt_idx]`

  Notes
  -----
  See `se_grid.create_spectral_element_grid`
  for an example of what the grid struct looks like.

  """
  buffers = extract_fields([f for f in fs_local], grid["triples_send"])
  return buffers


def unpack_scalar(fs_local,
                  buffers,
                  grid,
                  dim,
                  *args):
  """
  Sum non-processor-local redundancies into list of scalars
  after inter-process communication using assembly triples.

  Assembly triples are designed for use in wrapped code, and
  have the best shot of being compatible with automatic differentiation.

  Parameters
  ----------
  fs_local : `list[Array[tuple[elem_idx, gll_idx, gll_idx, level_idx], Float]]`
      List of scalar fields to extract for communication
  buffers: `dict[proc_idx, list[Array[tuple[local_point_idx, 1], Float]]`
      Mapping from the (remote) source processor idx to an array
      of values where `buffer[remote_proc_idx][pt_idx, 1]` is being sent to
      `proc_idx` and summed on `proc_idx` into
      `point_idx = triples_receive[proc_idx][1][pt_idx]`
  grid : `dict[str, Any]`
      Processor-local Spectral Element Grid struct.
      Contains "triples_receive" key, which
      has type `dict[remote_proc_idx, tuple[Array[tuple[point_idx], Float],\
                                           Array[tuple[point_idx], Int],\
                                           Array[tuple[point_idx], Int]]]`

  Returns
  -------
  `list[Array[tuple[elem_idx, gll_idx, gll_idx, level_idx], Float]]`
      List of scalar fields into which redundant DOFS have been summed.

  Notes
  -----
  See `se_grid.create_spectral_element_grid`
  for an example of what this struct looks like.

  """
  return [f for f in accumulate_fields([f for f in fs_local],
                                       buffers,
                                       grid["triples_receive"],
                                       dim)]


def _project_scalar_stub(fs_global,
                         grids,
                         dims):
  """
  Perform continuity projection on a list of scalars assuming all data is processor local
  using stub communicator and projection triples.

  *Only intended for testing and debugging.*

  Parameters
  ----------
  fs_global : `list[list[Array[tuple[elem_idx, gll_idx, gll_idx], Float]]]`
      List of length `num_proc`, where fs_global[proc_idx]
      is a list of "processor-local" 2D scalars to perform continuity projection on.
  grids : `list[dict[str, Any]]`
      List of grids containing "triples_send",
      "triples_receive", and "assembly_triple".

  Returns
  -------
  `list[list[Array[tuple[elem_idx, gll_idx, gll_idx], Float]]]`
      List of length `num_proc`, where `fs_global[proc_idx]`
      is a list of C0 "processor-local" 2D scalars on which projection has been performed.

  Notes
  -----
  * See `se_grid.create_spectral_element_grid` with `proc_idx=local_proc_idx`
  for how to create the `grids[proc_idx]` entries.
  * To calculate `fs_global[proc_idx][field_idx]`,
  use `se_grid.subset_var` to subdivide a given `f` calculated
  using an entire grid, or re-calculate `f` using, e.g.,
  `grids[proc_idx]["physical_coords]`.
  * To create your own dummy grid, see `se_grid.vert_redundancy_triage`
  for how to create "vert_redundancy_send", "vert_redundancy_receive",
  and "vert_redundancy_local", then see
  `se_grid.init_assembly_global` and `se_grid.init_assembly_local` for how to generate
  ("triples_send", "triples_receive"), and "assembly_triple", respectively.
  """

  buffers = []
  data_scaled = []
  local_buffers = []
  for fs_local, grid, dim in zip(fs_global, grids, dims):
    data_scaled.append([(f * grid["mass_matrix"])[:, :, :, np.newaxis] for f in fs_local])
    buffers.append(pack_scalar([(f * grid["mass_matrix"])[:, :, :, np.newaxis]
                                for f in fs_local], grid))
    local_buffers.append(extract_fields([(f * grid["mass_matrix"])[:, :, :, np.newaxis]
                                         for f in fs_local],
                                        {mpi_rank: grid["assembly_triple"]})[mpi_rank])

  fs_out = []
  for (fs_local, grid, dim, buffer_list) in zip(data_scaled, grids, dims, local_buffers):
    fs_out.append([])
    for f_scaled, buffer in zip(fs_local, buffer_list):
      fs_out[-1].append(sum_into(f_scaled, buffer, grid["assembly_triple"][1], dim))

  buffers = _exchange_buffers_stub(buffers)

  for proc_idx in range(len(fs_out)):
    fs_out[proc_idx] = [f.squeeze() * grids[proc_idx]["mass_matrix_denominator"]
                        for f in unpack_scalar(fs_out[proc_idx],
                                               buffers[proc_idx],
                                               grids[proc_idx],
                                               dims[proc_idx])]

  return fs_out


@partial(jit, static_argnames=["dim", "two_d"])
def project_scalar_global(fs_in,
                          grid,
                          dim,
                          two_d=True):
  """
  Perform continuity projection on a list of processor-local scalars using projection triples.

  Can be used for performance code.

  Parameters
  ----------
  fs : `list[Array[tuple[elem_idx, gll_idx, gll_idx], Float]]`
    List of processor-local 2D scalars to perform continuity projection on.
  grids : `list[dict[str, Any]]`
    List of grids, each of which contains
    "triples_send", "triples_receive", and "assembly_triple"

  Returns
  -------
  `list[Array[tuple[elem_idx, gll_idx, gll_idx], Float]]`
      List of continuous processor-local scalars on which continuity projection was performed.

  Notes
  -----
  * See `se_grid.create_spectral_element_grid` with `proc_idx=local_proc_idx`
    for how to create the `grids[proc_idx]` entries.
  * To create your own dummy grid, see `se_grid.vert_redundancy_triage`
    for how to create "vert_redundancy_send", "vert_redundancy_receive",
    and "vert_redundancy_local", then see
    `se_grid.init_assembly_global` and `se_grid.init_assembly_local` for how to generate
    ("triples_send", "triples_receive"), and "assembly_triple", respectively.

  Raises
  ------
  Error
    Raises any error that can be raised by exchange_buffers_mpi function.

  """
  scale = grid["mass_matrix"][:, :, :, np.newaxis]

  if two_d:
    fs = [f.reshape(*f.shape, 1) for f in fs_in]
  else:
    fs = fs_in

  buffer = pack_scalar([f * scale for f in fs], grid)
  buffer = exchange_buffers(buffer)

  local_buffer = extract_fields([f * scale for f in fs], {mpi_rank: grid["assembly_triple"]})[mpi_rank]
  fs_out = []
  for f, local_buf in zip(fs, local_buffer):
      fs_out.append(sum_into(f * scale, local_buf, grid["assembly_triple"][1], dim))
  fs = [f * grid["mass_matrix_denominator"][:, :, :, np.newaxis] for f in unpack_scalar(fs_out,
                                                                                        buffer,
                                                                                        grid,
                                                                                        dim)]
  if two_d:
    fs = [f.squeeze() for f in fs]
  return fs
