from src._config import get_backend as _get_backend
import numpy as np
from src.mpi.global_communication import _exchange_buffers_stub, exchange_buffers
_be = _get_backend()


def summation_local_for(f, grid, *args):
  """
  Sum processor-local redundant Degrees of Freedom into a scalar using a for loop.

  *This is for testing purposes, and should not be used in performance code*

  Parameters
  ----------
  f : `Array[tuple[elem_idx, gll_idx, gll_idx], Float]`
    Processor local scalar field.
  grid : `SpectralElementGrid`
    Spectral element grid struct that contains coordinate and metric data.

  Notes
  -----
  * Return value is not normalized by, e.g., the global mass matrix.

  Returns
  -------
  f_summed: `Array[tuple[elem_idx, gll_idx, gll_idx], Float]`
    Scalar field with DOFs added.
  """
  vert_redundancy_gll = grid["vertex_redundancy"]
  workspace = f.copy()
  for ((local_face_idx, local_i, local_j),
       (remote_face_id, remote_i, remote_j)) in vert_redundancy_gll:
    workspace[remote_face_id, remote_i, remote_j] += f[local_face_idx, local_i, local_j]
  # this line works even for multi-processor decompositions.

  return workspace


def extract_fields_for(fijk_fields, vert_redundancy_send):
  """
  Extract values from a list of processor-local 3D fields before
  inter-process communication using a for loop.

  *Designed for testing and debugging only,
  do not use in performance code*

  Parameters
  ----------
  fijk_fields : `list[Array[tuple[elem_idx, gll_idx, gll_idx, level_idx], Float]]`
      List of fields from which to extract redundant dofs to ship off
      to remote processors.
  vert_redundancy_send : `dict[remote_proc_idx, list[tuple[elem_idx, gll_idx, gll_idx]]]`
      Mapping from `remote_proc_idx` to a list of tuples of processor_local info
      to ship off to `remote_proc_idx`.
      Values are extracted as `fijk_fields[field_idx][elem_idx, i_idx, j_idx, :]`

  Notes
  -----
  The number of levels in `fijk_fields[field_idx]` may vary across `field_idx`.

  Returns
  -------
  `dict[remote_proc_idx, [Array[tuple[point_idx, level_idx], Float]]]`
      a mapping from `remote_proc_idx` to a list
      of buffers of redundant DOFs, preserving the number of levels in each field.
  """
  buffers = {}
  for remote_proc_idx in vert_redundancy_send.keys():
    buffers[remote_proc_idx] = []
    for field_idx in range(len(fijk_fields)):
      data = []
      for (source_local_idx, source_i, source_j) in vert_redundancy_send[remote_proc_idx]:
        data.append(fijk_fields[field_idx][source_local_idx, source_i, source_j, :])
      buffers[remote_proc_idx].append(np.stack(data, axis=-1))
  return buffers


def accumulate_fields_for(fijk_fields, buffers, vert_redundancy_receive):
  """
  Sum non-processor-local redundant DOFs into a list of processor-local 3D fields after
  inter-process communication using a for loop.

  *Designed for testing and debugging only,
  do not use in performance code*

  Parameters
  ----------
  fijk_fields : `list[Array[tuple[elem_idx, gll_idx, gll_idx, level_idx], Float]]`
      List of fields from which to extract redundant dofs to ship off
      to remote processors.
  buffers: `dict[remote_proc_idx, [Array[tuple[point_idx, level_idx], Float]]]`
      a mapping from `remote_proc_idx` to a list
      of buffers of redundant DOFs, preserving the number of levels in each field.
  vert_redundancy_receive : `dict[remote_proc_idx, list[tuple[elem_idx, gll_idx, gll_idx]]]`
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
      for col_idx, (target_local_idx, target_i, target_j) in enumerate(vert_redundancy_receive[remote_proc_idx]):
          fijk_fields[field_idx][target_local_idx,
                                 target_i,
                                 target_j, :] += buffers[remote_proc_idx][field_idx][:, col_idx]
  return fijk_fields


def assemble_scalar_for_pack(fs_local, grid):
  """
  Extract processor-local list of scalars before
  inter-process communication using a for loop.

  *This is only used for testing correctness of assembly.
  Do not use this in performance-critical code.*

  Parameters
  ----------
  fs_local : `list[Array[tuple[local_elem_idx, gll_idx, gll_idx, level_idx], Float]]`
      List of 2D scalar fields to extract for communication
  grid : dict[str, Any]
      Processor-local Spectral Element Grid struct.
      Contains "vert_redundancy_send" key, which
      has type `dict[remote_proc_idx, list[tuple[elem_idx, gll_idx, gll_idx]]]`

  Returns
  -------
  `dict[proc_idx, list[Array[tuple[local_point_idx, 1], Float]]`
      Mapping from the destination processor idx to an array
      of values where `buffer[remote_proc_idx][pt_idx, 1]` is being sent to
      remote_proc_idx and summed on remote_proc_idx into
      `(elem_idx, i, j) = vert_redundancy_receive[local_proc_idx][pt_idx]`

  Notes
  -----
  Grid must be constructed with wrapped=False to contain
  vert_redundancy_send.

  See `se_grid.create_spectral_element_grid`
  for an example of what the grid struct looks like.

  """
  buffers = extract_fields_for([f.reshape((*f.shape, 1)) for f in fs_local], grid["vertex_redundancy_send"])
  return buffers


def assemble_scalar_for_unpack(fs_local, buffers, grid, *args):
  """
  Sum remote processor redundancies into a list of scalars
  after inter-process communication using a for loop.

  *This is only used for testing correctness of assembly.
  Do not use this in performance-critical code.*

  Parameters
  ----------
  fs_local : `list[Array[tuple[elem_idx, gll_idx, gll_idx, level_idx], Float]]`
      List of 2D scalar fields to sum redundant DOFs into.
  buffers: `dict[proc_idx, list[Array[tuple[local_point_idx, 1], Float]]`
      Mapping from source processor idx to an array
      of values where `buffer[remote_proc_idx][pt_idx, 1]` is being
      acumulated on the current process into
      `(elem_idx, i, j) = vert_redundancy_receive[remote_proc_idx][pt_idx]`
  grid : `dict[str, Any]`
      Processor-local Spectral Element Grid struct.
      Contains "vert_redundancy_receive" key, which
      has type `dict[remote_proc_idx, list[tuple[elem_idx, gll_idx, gll_idx]]]`

  Returns
  -------
  `list[Array[tuple[elem_idx, gll_idx, gll_idx, level_idx], Float]]`
      List of scalar fields into which redundant DOF info has been accumulated.

  Notes
  -----
  Grid must be constructed with wrapped=False to contain
  vert_redundancy_receive.

  See `se_grid.create_spectral_element_grid`
  for an example of what the grid struct looks like.

  """
  return [f[:, :, :, 0] for f in accumulate_fields_for([f.reshape((*f.shape, 1)) for f in fs_local],
                                                       buffers, grid["vertex_redundancy_receive"])]


def project_scalar_for_stub(fs_global, grids):
  """
  Perform continuity projection on a list of scalars assuming all data is processor local
  using stub communicator and a for loop.

  *Only intended for testing and debugging.*

  Parameters
  ----------
  fs_global : `list[list[Array[tuple[elem_idx, gll_idx, gll_idx], Float]]]`
      List of length `num_proc`, where fs_global[proc_idx]
      is a list of "processor-local" 2D scalars to perform continuity projection on.
  grids : `list[dict[str, Any]]`
      List of grids containing "vert_redundancy_send",
      "vert_redundancy_receive", and "vert_redundancy_local".

  Returns
  -------
  `list[list[Array[tuple[elem_idx, gll_idx, gll_idx], Float]]]`
      List of length `num_proc`, where `fs_global[proc_idx]`
      is a list of C0 "processor-local" globally continuous 2D scalar has been performed.

  Notes
  -----
  * See `se_grid.create_spectral_element_grid` with `proc_idx=local_proc_idx`
  for how to create the `grids[local_proc_idx]` entries.
  * To calculate `fs_global[proc_idx][field_idx]`,
  use `se_grid.subset_var` to subdivide a given `f` calculated
  using an entire grid, or re-calculate `f` using, e.g.,
  `grids[proc_idx]["physical_coords]`.
  * To create your own dummy grid, see `se_grid.vert_redundancy_triage`
  for how to create "vert_redundancy_send", "vert_redundancy_receive",
  and "vert_redundancy_local" entries.

  """

  buffers = []
  for fs_local, grid in zip(fs_global, grids):
    buffers.append(assemble_scalar_for_pack([f * grid["mass_matrix"] for f in fs_local], grid))

  fs_out = [[summation_local_for(f * grid["mass_matrix"], grid) for f in fs_local]
            for (fs_local, grid) in zip(fs_global, grids)]
  buffers = _exchange_buffers_stub(buffers)

  for proc_idx in range(len(fs_out)):
    fs_out[proc_idx] = [f * grids[proc_idx]["mass_matrix_denominator"]
                        for f in assemble_scalar_for_unpack(fs_out[proc_idx], buffers[proc_idx], grids[proc_idx])]

  return fs_out


def project_scalar_for_mpi(fs, grid):
  """
  Perform continuity projection on a list of processor-local scalars using a for loop.

  *Only used for testing, do not use in performance-critical code*

  Parameters
  ----------
  fs : `list[Array[tuple[elem_idx, gll_idx, gll_idx], Float]]`
    List of processor-local 2D scalars to perform continuity projection on.
  grids : `list[dict[str, Any]]`
    List of grids, each of which contains
    "vert_redundancy_send" and "vert_redundancy_receive", and "vert_redundancy_local"

  Returns
  -------
  `list[Array[tuple[elem_idx, gll_idx, gll_idx], Float]]`
    List of continuous processor-local scalars on which continuity projection was performed.

  Notes
  -----
  `create_spectral_element_grid` must be called with `wrapped=False`
  for it to contain "vert_redundancy_send", "vert_redundancy_receive",
  and "vert_redundancy_local"

  * See `se_grid.create_spectral_element_grid` with `proc_idx=local_proc_idx`
  for how to create the `grids[proc_idx]` entries.
  * To create your own dummy grid, see `se_grid.vert_redundancy_triage`
  for how to create "vert_redundancy_send", "vert_redundancy_receive",
  and "vert_redundancy_local".

  Raises
  ------
  Error
    Raises any error that can be raised by exchange_buffers_mpi function.
  """
  # This is primarily for testing!
  # do not use in model code!
  data_scaled = [f * grid["mass_matrix"] for f in fs]
  buffer = assemble_scalar_for_pack(data_scaled, grid)
  buffer = exchange_buffers(buffer)
  fs_out = [summation_local_for(f, grid) for f in data_scaled]
  fs = [f * grid["mass_matrix_denominator"]
        for f in assemble_scalar_for_unpack(fs_out, buffer, grid)]
  return fs
