from .._config import get_backend as _get_backend
_be = _get_backend()


def exchange_buffers(send_buffers, neighbor_ranks=None):
  """
  Exchange Spectral Element grid non-processor-local redundant DOFS between
  processes using the active backend's native collective.

  This is a thin wrapper over :meth:`Backend.halo_exchange`.  NumPy uses mpi4py
  point-to-point sendrecv; JAX uses ``shard_map`` + ``lax.ppermute`` (multi-host
  path).  The per-field buffer lists are exchanged one field at a time so that
  buffers with differing level counts are handled uniformly.

  Parameters
  ----------
  send_buffers : `dict[proc_idx, list[Array[tuple[point_idx, level_idx], Float]]]`
      A buffer struct that maps `proc_idx` to a list of arrays containing
      redundant DOFs to send to that processor.
  neighbor_ranks : `tuple[int, ...]` or None, optional
      Static list of ranks this rank exchanges with.  Defaults to
      ``tuple(send_buffers.keys())``.

  Returns
  -------
  buffer : `dict[proc_idx, list[Array[tuple[point_idx, level_idx], Float]]]`
      A buffer struct that maps `proc_idx` to a list of arrays containing
      redundant DOFs received from that processor.
  """
  if neighbor_ranks is None:
    neighbor_ranks = tuple(send_buffers.keys())
  if not neighbor_ranks:
    return send_buffers
  num_fields = len(send_buffers[neighbor_ranks[0]])
  out = {k: [None] * num_fields for k in neighbor_ranks}
  for field_idx in range(num_fields):
    per_field = {k: send_buffers[k][field_idx] for k in neighbor_ranks}
    recv = _be.halo_exchange(per_field, neighbor_ranks)
    for k in neighbor_ranks:
      out[k][field_idx] = recv[k]
  return out


def global_sum(summand):
  """
  Compute the global sum of a processor-local quantity such as a summed
  integrand, via :meth:`Backend.all_reduce_sum`.

  Parameters
  ----------
  summand : float
    Processor-local part of the quantity over which reduction is performed.

  Returns
  -------
  integral : float
    Global sum of quantity.
  """
  return _be.all_reduce_sum(summand)


def global_max(arg):
  """
  Compute the global maximum of a processor-local quantity, via
  :meth:`Backend.all_reduce_max`.

  Parameters
  ----------
  arg : float
    Processor-local part of the quantity over which reduction is performed.

  Returns
  -------
  integral : float
    Global max of quantity.
  """
  return _be.all_reduce_max(arg)


def global_min(arg):
  """
  Compute the global minimum of a processor-local quantity, via
  :meth:`Backend.all_reduce_min`.

  Parameters
  ----------
  arg : float
    Processor-local part of the quantity over which reduction is performed.

  Returns
  -------
  integral : float
    Global min of quantity.
  """
  return _be.all_reduce_min(arg)


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
  The behavior should be almost identical to how exchange_buffers
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
