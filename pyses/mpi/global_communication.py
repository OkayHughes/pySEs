import numpy as np
from .._config import get_backend as _get_backend
_be = _get_backend()
use_wrapper = _be.use_wrapper
wrapper_type = _be.wrapper_type
has_mpi = _be.has_mpi
device_wrapper = _be.array
jit = _be.jit
jnp = _be.np
mpi_comm = _be.mpi_comm
if has_mpi:
  from mpi4py import MPI
  if use_wrapper and wrapper_type == "jax":
    import mpi4jax


def exchange_buffers_mpi(buffer):
  """
  Exchange Spectral Element grid non-processor-local redundant DOFS
  between processes using the Message Passing Interface.

  Parameters
  ----------
  buffer: `dict[proc_idx, list[Array[tuple[point_idx, level_idx], Float]]]`
      A buffer struct that maps `proc_idx` to a
      list of arrays containing redundant DOFs to send to that processor.

  Returns
  -------
  buffer: `dict[proc_idx, list[Array[tuple[point_idx, level_idx], Float]]]`
      A buffer struct that maps `proc_idx` to a
      list of arrays containing redundant DOFs received from that processor.

  Notes
  -----
  mpi4py is designed to accept objects that buffer properties
  that resemble np.ndarrays. This function can almost certainly
  be designed in a way that can leverage gpu-aware MPI environments,
  but this functionality has not yet been tested.
  Divergence in how this is performed with different wrapper types
  is acceptable.

  Raises
  ------
  Error
    Any error that can be raised by the following two functions:
    * `mpi_comm.Isendrecv_replace`
    * `MPI.Request.Waitall`

  """
  reqs = []
  for source_proc_idx in buffer.keys():
    for k_idx in range(len(buffer[source_proc_idx])):
      reqs.append(mpi_comm.Isendrecv_replace(buffer[source_proc_idx][k_idx],
                                             source_proc_idx,
                                             source=source_proc_idx,
                                             sendtag=k_idx,
                                             recvtag=k_idx))
  MPI.Request.Waitall(reqs)
  return buffer


@jit
def exchange_buffers_jax(buffer):
  """
  Exchange Spectral Element grid non-processor-local redundant DOFS
  between processes using mpi4jax

  **This function is the only function in the entire codebase
  that will hang indefinitely in the event of, e.g., hardware failures
  on a remote processor, or other distributed-memory shenanigans.**

  Parameters
  ----------
  buffer: `dict[proc_idx, list[Array[tuple[point_idx, level_idx], Float]]]`
      A buffer struct that maps `proc_idx` to a
      list of arrays containing redundant DOFs to send to that processor.

  Returns
  -------
  buffer: `dict[proc_idx, list[Array[tuple[point_idx, level_idx], Float]]]`
      A buffer struct that maps `proc_idx` to a
      list of arrays containing redundant DOFs received from that processor.

  Notes
  -----
  mpi4py is designed to accept objects that buffer properties
  that resemble np.ndarrays. This function can almost certainly
  be designed in a way that can leverage gpu-aware MPI environments,
  but this functionality has not yet been tested.
  Divergence in how this is performed with different wrapper types
  is acceptable.

  Raises
  ------
  Error
    Any error that can be raised by:
      * mpi4jax.sendrecv

  """
  for source_proc_idx in buffer.keys():
    for k_idx in range(len(buffer[source_proc_idx])):
      buffer[source_proc_idx][k_idx] = mpi4jax.sendrecv(buffer[source_proc_idx][k_idx], buffer[source_proc_idx][k_idx],
                                                        source_proc_idx,
                                                        source_proc_idx,
                                                        sendtag=k_idx,
                                                        recvtag=k_idx)
  return buffer


def exchange_buffers_jax_unwrap(buffer):
  """
  Exchange Spectral Element grid non-processor-local redundant DOFS
  between processes using mpi4py with device-host copying.

  **This function is the only function in the entire codebase
  that will hang indefinitely in the event of, e.g., hardware failures
  on a remote processor, or other distributed-memory shenanigans.**

  Parameters
  ----------
  buffer: `dict[proc_idx, list[Array[tuple[point_idx, level_idx], Float]]]`
      A buffer struct that maps `proc_idx` to a
      list of arrays containing redundant DOFs to send to that processor.

  Returns
  -------
  buffer: `dict[proc_idx, list[Array[tuple[point_idx, level_idx], Float]]]`
      A buffer struct that maps `proc_idx` to a
      list of arrays containing redundant DOFs received from that processor.

  Notes
  -----
  mpi4py is designed to accept objects that buffer properties
  that resemble np.ndarrays. This function can almost certainly
  be designed in a way that can leverage gpu-aware MPI environments,
  but this functionality has not yet been tested.
  Divergence in how this is performed with different wrapper types
  is acceptable.

  Raises
  ------
  Error
    Any error that can be raised by the following two functions:
    * `mpi_comm.Isendrecv_replace`
    * `MPI.Request.Waitall`

  """
  reqs = []
  for source_proc_idx in buffer.keys():
    for k_idx in range(len(buffer[source_proc_idx])):
      buffer[source_proc_idx][k_idx] = np.array(buffer[source_proc_idx][k_idx])
  for source_proc_idx in buffer.keys():
    for k_idx in range(len(buffer[source_proc_idx])):
      reqs.append(mpi_comm.Isendrecv_replace(buffer[source_proc_idx][k_idx],
                                             source_proc_idx,
                                             source=source_proc_idx,
                                             sendtag=k_idx,
                                             recvtag=k_idx))
  MPI.Request.Waitall(reqs)
  for source_proc_idx in buffer.keys():
    for k_idx in range(len(buffer[source_proc_idx])):
      buffer[source_proc_idx][k_idx] = device_wrapper(buffer[source_proc_idx][k_idx])
  return buffer


if use_wrapper and wrapper_type == "jax":
  exchange_buffers = exchange_buffers_jax
else:
  exchange_buffers = exchange_buffers_mpi


def global_sum(summand):
  """
  Compute the global sum of a processor-local quantity
  such as a summed integrand.

  Parameters
  ----------
  summand : float
    Processor-local part of the quantity over which reduction is
    performed.

  Returns
  -------
  integral : float
    Global sum of quantity.
  """
  if has_mpi:
    send = np.array(summand)
    recv = np.copy(send)
    req = mpi_comm.Iallreduce(np.array(send),
                              recv,
                              MPI.SUM)
    MPI.Request.Wait(req)
    ret = recv.item()
  else:
    ret = jnp.sum(summand)
  return ret


def global_max(arg):
  """
  Compute the global maximum of a processor-local quantity.

  Parameters
  ----------
  arg : float
    Processor-local part of the quantity over which reduction is
    performed.

  Returns
  -------
  integral : float
    Global max of quantity.
  """
  if has_mpi:
    send = np.array(arg)
    recv = np.copy(send)
    req = mpi_comm.Iallreduce(np.array(send),
                              recv,
                              MPI.MAX)
    MPI.Request.Wait(req)
    ret = recv.item()
  else:
    ret = jnp.max(arg)
  return ret


def global_min(arg):
  """
  Compute the global minimum of a processor-local quantity.

  Parameters
  ----------
  arg : float
    Processor-local part of the quantity over which reduction is
    performed.

  Returns
  -------
  integral : float
    Global min of quantity.
  """
  if has_mpi:
    send = np.array(arg)
    recv = np.copy(send)
    req = mpi_comm.Iallreduce(np.array(send),
                              recv,
                              MPI.MIN)
    MPI.Request.Wait(req)
    ret = recv.item()
  else:
    ret = jnp.min(arg)
  return ret


def _exchange_buffers_stub(buffer_list):
  """
  Exchange buffers between source dofs and target dofs assuming that all grid is processor-local.

  *Only used for testing and debugging, do not use in performance
  code*

  Parameters
  ----------
  buffer_list: `list[dict[proc_idx, list[Array[tuple[point_idx, level_idx], Float]]]]`
      A list of length num_processors, each of which is a buffer struct
      that maps `proc_idx` to a list of arrays containing redundant DOFs to send.

  Returns
  -------
  `list[dict[proc_idx, list[Array[tuple[point_idx, level_idx], Float]]]]`
      A list of length num_processors, each of which is a buffer struct
      that maps proc_idx to a list of arrays containing redundant DOFs that were received.

  Notes
  ------
  This function exchanges the memory reffered to by `buffer_list[proc_idx][remote_proc_idx][field_idx]`
  with `buffer_list[remote_proc_idx][proc_idx][field_idx]`.
  The behavior should be almost identical to how exchange_buffers_mpi
  behaves when called when has_mpi=True, except for this difference.

  By construction, if any grid point `(elem_idx_source, i_idx_source, j_idx_source)`
  that has a redundancy with `(elem_idx_target, i_idx_target, j_idx_target)`,
  this relation is symmetric. Therefore, the number of grid points
  necessary to send from `proc_idx_1` to `proc_idx_2` is identical
  to the number to send from `proc_idx_2` to `proc_idx_1`.
  The indexes of points in the buffer that is sent
  will be different from those in the buffer that is received,
  but so long as both processes agree on the different orderings,
  this is fine.
  """
  pairs = set()
  for source_proc_idx in range(len(buffer_list)):
    buffer = buffer_list[source_proc_idx]
    for target_proc_idx in buffer.keys():
      if (target_proc_idx, source_proc_idx) not in pairs:
        # Python names and lists are counter-intuitive
        # so I'm leaving this ugly for the moment.
        (buffer[target_proc_idx],
         buffer_list[target_proc_idx][source_proc_idx]) = (buffer_list[target_proc_idx][source_proc_idx],
                                                           buffer[target_proc_idx])
        pairs.add((source_proc_idx, target_proc_idx))
  return buffer_list
