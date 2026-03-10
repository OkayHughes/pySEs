import numpy as np
from ..mesh_generation.mesh_definitions import (TOP_FACE, BOTTOM_FACE, FRONT_FACE,
                                                BACK_FACE, LEFT_FACE, RIGHT_FACE)


def eval_hilbert_curve(n_subdiv):
  """
  Generate a 2D array containing indices
  representing a 2D hilbert curve in euclidean space.

  Parameters
  ----------
  n_subdiv: int
      Number of times the initial 1x1 Hilbert curve should be subdivided

  Returns
  -------
  Array[tuple[2**n_subdiv, 2**n_subdiv, Int]
      Index of each grid cell in the hilbert curve.

  Notes
  -----
  Without loss of generality,
  the initial generator is assumed to be
  ^
  |  |
  |__|
  """
  A, B, C, D = 0, 1, 2, 3
  arr_prev = C * np.ones((1, 1), dtype=np.int64)
  idxs_prev = np.zeros((1, 1), dtype=np.int64)
  for iter in range(n_subdiv):
    idxs_next = np.zeros((2**(iter + 1), 2**(iter + 1)), dtype=np.int64)
    arr_next = np.zeros((2**(iter + 1), 2**(iter + 1)), dtype=np.int64)
    for i_idx in range(int(2**iter)):
      for j_idx in range(int(2**iter)):
        prev_ind = 4 * idxs_prev[i_idx, j_idx]
        if arr_prev[i_idx, j_idx] == A:
          arr_next[2 * i_idx:2 * (i_idx + 1),
                   2 * j_idx:2 * (j_idx + 1)] = [[A, A], [D, B]]
          idxs_next[2 * i_idx:2 * (i_idx + 1),
                    2 * j_idx:2 * (j_idx + 1)] = [[prev_ind + 1, prev_ind + 2],
                                                  [prev_ind + 0, prev_ind + 3]]
        if arr_prev[i_idx, j_idx] == B:
          arr_next[2 * i_idx:2 * (i_idx + 1),
                   2 * j_idx:2 * (j_idx + 1)] = [[B, C], [B, A]]
          idxs_next[2 * i_idx:2 * (i_idx + 1),
                    2 * j_idx:2 * (j_idx + 1)] = [[prev_ind + 1, prev_ind + 0],
                                                  [prev_ind + 2, prev_ind + 3]]
        if arr_prev[i_idx, j_idx] == C:
          arr_next[2 * i_idx:2 * (i_idx + 1),
                   2 * j_idx:2 * (j_idx + 1)] = [[D, B], [C, C]]
          idxs_next[2 * i_idx:2 * (i_idx + 1),
                    2 * j_idx:2 * (j_idx + 1)] = [[prev_ind + 3, prev_ind + 0],
                                                  [prev_ind + 2, prev_ind + 1]]
        if arr_prev[i_idx, j_idx] == D:
          arr_next[2 * i_idx:2 * (i_idx + 1), 2 * j_idx:2 * (j_idx + 1)] = [[C, D], [A, D]]
          idxs_next[2 * i_idx:2 * (i_idx + 1),
                    2 * j_idx:2 * (j_idx + 1)] = [[prev_ind + 3, prev_ind + 2],
                                                  [prev_ind + 0, prev_ind + 1]]
    idxs_prev, idxs_next = idxs_next, idxs_prev
    arr_prev, arr_next = arr_next, arr_prev
  return idxs_prev


def sphere_coord_to_face_idx_pos(lat,
                                 lon):
  """
  Calculate which face of the cubed sphere contain
  points given in spherical coordinates.

  Parameters
  ----------
  lat: Array[tuple[Any], Float]
      Latitude of points
  lon: Array[tuple[Any], Float]
      Longitude of points

  Notes
  -----
  See mesh_definitions.py for explanation of face conventions.

  Returns
  -------
  face_mask: Array[tuple[Any], Int]
      Which of TOP_FACE, BOTTOM_FACE, FRONT_FACE,
      BACK_FACE, LEFT_FACE, RIGHT_FACE each point lies on
  x: Array[tuple[Any], Int]
    Horizontal coordinates on cubed sphere face, [-1, 1]
  y: Array[tuple[Any], Int]
    Vertical coordinates on cubed sphere face, [-1, 1]
  """
  # assumes lon \in [0, 2*np.pi]
  lat[lat > np.pi / 2.0 - 1e-4] = np.pi / 2.0 - 1e-4
  lat[lat < -np.pi / 2.0 + 1e-4] = -np.pi / 2.0 + 1e-4

  equator_panel_centers = np.arange(0, 2 * np.pi - 1e-8, np.pi / 2) + np.pi / 4
  lon_diff = np.abs(lon.reshape((*lon.shape, 1)) -
                    equator_panel_centers.reshape((*np.ones_like(lon.shape),
                                                   equator_panel_centers.size)))
  face_idx = np.argmin(lon_diff, axis=-1)
  lon_loc = lon - equator_panel_centers[face_idx.flatten()].reshape(face_idx.shape)
  y_provisional = np.tan(lat) / np.cos(lon_loc)
  x_provisional = np.tan(lon_loc)
  top_mask = y_provisional > 1.0
  bottom_mask = y_provisional < -1.0
  front_mask = face_idx == 0
  left_mask = face_idx == 3
  right_mask = face_idx == 2
  back_mask = face_idx == 1

  y_provisional[top_mask] = -np.cos(lon[top_mask] + np.pi / 4) / np.tan(lat[top_mask])
  y_provisional[bottom_mask] = -np.cos(lon[bottom_mask] + np.pi / 4) / np.tan(lat[bottom_mask])
  x_provisional[top_mask] = np.sin(lon[top_mask] + np.pi / 4) / np.tan(lat[top_mask])
  x_provisional[bottom_mask] = -np.sin(lon[bottom_mask] + np.pi / 4) / np.tan(lat[bottom_mask])
  face_idx[left_mask] = LEFT_FACE
  face_idx[right_mask] = RIGHT_FACE
  face_idx[front_mask] = FRONT_FACE
  face_idx[back_mask] = BACK_FACE
  face_idx[top_mask] = TOP_FACE
  face_idx[bottom_mask] = BOTTOM_FACE
  return face_idx, x_provisional, y_provisional


def processor_id_to_range(proc_idx,
                          num_elems,
                          num_procs):
  """
  Calculate the number of elements to assign to a processor
  from the global grid size.

  Parameters
  ----------
  proc_idx: int
      The processor id (e.g. config.mpi_rank) for which to calculate
      the element assignment
  num_faces: int
      The total number of elements in the global grid.
  num_procs: int
      The number of processes (e.g. config.mpi_size) among which
      the grid will be divided.

  Returns
  -------
  tuple[int, int]
      Tuple containing (begin_elem_idx, end_elem_idx),
      where the final index is exclusive.

  Notes
  -----
  This must be able to calculate the number of grid points
  assigned to remote processors as well.

  Invoking code should be careful about using the specifics
  of the return value of this function, as there may eventually
  be a better way to do this.

  Raises
  ------
  ValueError
    Raises a value error if this processor decomposition strategy
    is used with more processors than elements.
  """
  if num_procs > num_elems:
    raise ValueError(("Naive processor decomposition strategy cannot be used for "
                      f"num_processors {num_procs} > num_elems {num_elems}"))
  stride = int(np.floor(num_elems / num_procs))
  beginning_idx = proc_idx * stride
  if proc_idx == num_procs - 1:
    end_idx = num_elems
  else:
    end_idx = (proc_idx + 1) * stride
  return beginning_idx, end_idx


def init_decomp(num_elems,
                num_procs):
  """
  Get decomposition of total number of faces by processor
  for all processors.

  Parameters
  ----------
  num_elems: int
      The number of elements in the global grid
  num_procs: int
      The total number of processors
      among which the grid will be divided,
      e.g. `config.mpi_size`.

  Notes
  -----
  Invoking code should be careful about using the specifics
  of the return value of get_decomp, as there may eventually
  be a better way to do this.

  Returns
  -------
  tuple[tuple[int, int], ...]
      A tuple of length num_procs where
      entry proc_idx is a tuple (begin_elem_idx, end_elem_idx)
  """
  segments = []
  for proc_idx in range(num_procs):
    segments.append(processor_id_to_range(proc_idx, num_elems, num_procs))
  return tuple(segments)


def init_mapping(n_subdiv,
                 latlons):
  """
  Use a cubed-sphere hilbert curve
  to create a reordering of the global grid
  that puts sequential elements close to each other

  Sequential elements may not be geometrically close
  at the endpoints of the hilbert curve on each
  face of the cubed sphere.

  Parameters
  ----------
  n_subdiv: int
      The number of subdivisions to use in the hilbert curve
  latlons: Array[tuple[elem_idx, 2], Float]
      Representative position of each element
      in lat-lon coordinates (e.g. centroid).
      Uses convention latlons[:, 0] is lat,
      latlons[:, 1] is lon.

  Returns
  -------
  index_map : Array[tuple[elem_idx], Int]
      Permutation array such that ``grid[index_map]`` reorders
      the global element array along the Hilbert curve,
      placing spatially adjacent elements close together in memory.

  Notes
  -----
  Although this decomposition method is based
  on the cubed sphere,
  it can also be used for unstructured grids on the sphere.
  """
  face_idx, x, y = sphere_coord_to_face_idx_pos(latlons[:, 0], latlons[:, 1])
  idxs = eval_hilbert_curve(n_subdiv)
  grid_pos_1d = np.linspace(-1.0, 1.0, 2**n_subdiv)
  id_x = np.argmin(np.abs(x[np.newaxis, :] - grid_pos_1d[:, np.newaxis]), axis=0)
  id_y = np.argmin(np.abs(y[np.newaxis, :] - grid_pos_1d[:, np.newaxis]), axis=0)
  idxs_coarse = idxs[id_x, id_y]
  idxs_coarse += face_idx * idxs.size
  index_map = np.argsort(idxs_coarse)
  return index_map


def local_to_global(elem_idxs_local,
                    proc_idx,
                    decomp):
  """
  Return global element indexes for a specific processor
  index, e.g., config.mpi_rank.

  Parameters
  ----------
  elem_idxs_local: `Array[*Shape, Float]`
      Element indexes used to index
      processor local data.
  proc_idx: `int`
      The processor index to which the local indexes
      apply.
  decomp: `tuple[tuple(int, int), ...]`
      Processor decomposition

  Returns
  -------
  elem_idxs_global: `Array[*Shape, Float]`
      Global element indexes
  """
  return elem_idxs_local + decomp[proc_idx][0]


def global_to_local(elem_idxs_global,
                    proc_idx,
                    decomp):
  """
  Return local element indexes for a specific processor
  index, e.g., config.mpi_rank.

  Parameters
  ----------
  elem_idxs_global: `Array[*Shape, Float]`
      Element indexes used to index
      global grid data.
  proc_idx: `int`
      The processor index to which the local indexes should
      apply.
  decomp: `tuple[tuple(int, int), ...]`
      Processor decomposition

  Returns
  -------
  elem_idxs_local: Array[*Shape, Float]
      Local element indexes
  """
  return elem_idxs_global - decomp[proc_idx][0]


def elem_idx_global_to_proc_idx(elem_idxs_global,
                                decomp):
  """
  Map global element indexes to the id of the
  processor they are assigned to.

  Parameters
  ----------
  elem_idxs_global: `Array[tuple[elem_idx], Int]`
      Global element indexes.
  decomp: `tuple[tuple(int, int), ...]`
      Processor decomposition

  Returns
  -------
  `Union[Int, Array[tuple[elem_idx], Int]]`
      Processor indexes of each element.

  Notes
  -----
  If an array with a single item is passed in,
  this function returns an unwrapped integer.
  This is for testing reasons and should
  probably be cleaned up.
  """
  out = np.zeros_like(elem_idxs_global, dtype=np.int64)
  for proc_idx, (begin, end) in enumerate(decomp):
    mask = np.logical_and(elem_idxs_global < end, elem_idxs_global >= begin)
    out[mask] = proc_idx
  if out.size == 1:
    out = out.item()
  return out
