from ..config import np, DEBUG, use_wrapper, do_sharding
from .bilinear_utils import eval_bilinear_mapping, eval_bilinear_jacobian
from .mesh_definitions import TOP_EDGE, LEFT_EDGE, RIGHT_EDGE, BOTTOM_EDGE, FORWARDS, MAX_VERT_DEGREE_UNSTRUCTURED
from ..spectral import init_spectral
from ..horizontal_grid import init_spectral_element_grid, smooth_tensor, shard_grid
from .spherical_coord_utils import unit_sphere_to_cart_coords_jacobian


def edge_to_vert(edge_id,
                 is_forwards=FORWARDS):
  """
  Map an edge id of oriented vertex ids of a given element edge.

  Parameters
  ----------
  edge_id : `int`
      Index of edge within an element
  is_forwards: `int`, default=FORWARDS
      Is the edge reversed from its
      default orientation.

  Returns
  -------
  `tuple[int, int]`
      (vert_idx_0, vert_idx_1)

  Notes
  --------
  See mesh_definitions for grid conventions on
  vertex_idx, edge enumeration, and default direction.
  """
  if edge_id == TOP_EDGE:
    v_idx_in_0 = 0
    v_idx_in_1 = 1
  elif edge_id == LEFT_EDGE:
    v_idx_in_0 = 0
    v_idx_in_1 = 2
  elif edge_id == RIGHT_EDGE:
    v_idx_in_0 = 1
    v_idx_in_1 = 3
  elif edge_id == BOTTOM_EDGE:
    v_idx_in_0 = 2
    v_idx_in_1 = 3
  if is_forwards != FORWARDS:
    return v_idx_in_1, v_idx_in_0
  else:
    return v_idx_in_0, v_idx_in_1


def mesh_to_cart_bilinear(face_position,
                          npt):
  """
  Bilinearly map cartesian elements corners to
  cartesian tensor-GLL mesh.

  Parameters
  ----------
  face_position : `Array[tuple[elem_idx, vert_idx, cart_idx], Float]`
      Array of cartesian position of element corners.
  npt : `int`
      number of Gauss-Lobatto-Legendre points in each
      dimension of an element.

  Notes
  -----
  This method works for general meshes, and is not
  specific to a particular topology.

  Returns
  -------
  gll_position : `Array[tuple[elem_idx, gll_idx, gll_idx, cart_idx], Float]`
      Position of GLL gridpoints in cartesian space, e.g. the (x, y) coordinates
      in a cubed sphere face
  gll_jacobian : `Array[tuple[elem_idx, gll_idx, gll_idx, cart_idx, ab], Float]`
      Jacobian of bilinear mapping from the reference element to cartesian space
      for the GLL mesh.
  """
  spectrals = init_spectral(npt)
  cart_dim = face_position.shape[2]
  NFACES = face_position.shape[0]

  gll_position = np.zeros(shape=(NFACES, npt, npt, cart_dim))
  gll_jacobian = np.zeros(shape=(NFACES, npt, npt, cart_dim, 2))

  for i_idx in range(npt):
    for j_idx in range(npt):
        alpha = spectrals["gll_points"][i_idx]
        beta = spectrals["gll_points"][j_idx]
        gll_position[:, i_idx, j_idx, :] = eval_bilinear_mapping(face_position[:, 0, :],
                                                                 face_position[:, 1, :],
                                                                 face_position[:, 2, :],
                                                                 face_position[:, 3, :],
                                                                 alpha,
                                                                 beta)

        dphys_dalpha, dphys_dbeta = eval_bilinear_jacobian(face_position[:, 0, :],
                                                           face_position[:, 1, :],
                                                           face_position[:, 2, :],
                                                           face_position[:, 3, :],
                                                           alpha,
                                                           beta)
        gll_jacobian[:, i_idx, j_idx, :, 0] = dphys_dalpha
        gll_jacobian[:, i_idx, j_idx, :, 1] = dphys_dbeta

  return gll_position, gll_jacobian


def init_spectral_grid_redundancy(vert_redundancy,
                                  npt):
  """
  Enumerate all redundant DOFs in a global
  SpectralElementGrid.

  Parameters
  ----------
  vert_redundancy : `dict[elem_idx, dict[corner_idx, set(tuple[elem_idx, corner_idx])]]`
      `dict[elem_idx][corner_idx]` is a set of element corners
      that are coincident with `face_position[elem_idx, corner_idx, :]`.
  npt : `int`
      number of Gauss-Lobatto-Legendre points in each
      dimension of an element.

  Notes
  -----
  This method works for general meshes, and is not
  specific to a particular topology.

  DOFs are not considered redundant with themselves.
  If assembling the Spectral Element projection operator
  from this struct, you must construct the diagonal entries yourself.

  Returns
  -------
  vert_redundancy_gll: `dict[elem_idx, dict[tuple(gll_idx, gll_idx),\
                                            set[tuple[elem_idx, gll_idx, gll_idx]]]]`
      `dict[elem_idx][(gll_idx,gll_idx)]` is a set
      of redundant DOFs on the global GLL grid.
  """
  # temporary note: we can assume here that this is mpi-local.
  # note:
  # count DOFs
  vert_redundancy_gll = {}

  def wrap(elem_idx, i_idx, j_idx):
    if elem_idx not in vert_redundancy_gll.keys():
      vert_redundancy_gll[elem_idx] = dict()
    if (i_idx, j_idx) not in vert_redundancy_gll[elem_idx].keys():
      vert_redundancy_gll[elem_idx][(i_idx, j_idx)] = set()

  correct_orientation = set([(0, 1), (0, 2), (2, 3), (1, 3)])

  def is_forwards(v0, v1):
    return (v0, v1) in correct_orientation

  def vert_to_i_j(vert_idx):
    if vert_idx == 0:
      return 0, 0
    elif vert_idx == 1:
      return npt - 1, 0
    elif vert_idx == 2:
      return 0, npt - 1
    elif vert_idx == 3:
      return npt - 1, npt - 1

  def infer_edge(elem_adj_loc, edge_idx, free_idx):
    if edge_idx == TOP_EDGE:
      idx0, idx1 = (0, 1)
    elif edge_idx == LEFT_EDGE:
      idx0, idx1 = (0, 2)
    elif edge_idx == RIGHT_EDGE:
      idx0, idx1 = (1, 3)
    elif edge_idx == BOTTOM_EDGE:
      idx0, idx1 = (2, 3)

    # find only element that overlaps both vertices on edge
    elems = [x[0] for x in elem_adj_loc[idx0]]
    elem_id = list(filter(lambda x: x[0] in elems, elem_adj_loc[idx1]))
    if DEBUG:
      assert (len(elem_id) == 1)
    elem_idx_pair = elem_id[0][0]
    # determine which vertices element is paired to
    v0 = list(filter(lambda x: x[0] == elem_idx_pair, elem_adj_loc[idx0]))
    v1 = list(filter(lambda x: x[0] == elem_idx_pair, elem_adj_loc[idx1]))
    if DEBUG:
      assert (len(v0) == 1)
      assert (len(v1) == 1)
    v0 = v0[0][1]
    v1 = v1[0][1]
    v0_i_idx, v0_j_idx = vert_to_i_j(v0)
    v1_i_idx, v1_j_idx = vert_to_i_j(v1)

    # calculate i, j indices on paired element,
    # accounting for edge direction.
    # Note: switch statement above orients local edge
    # in the forward direction, so orientation of paired
    # edge can be inferred from the ordered pair (v0, v1)
    if v0_i_idx == v1_i_idx:
      i_idx_pair = v0_i_idx
      if is_forwards(v0, v1):
        j_idx_pair = free_idx
      else:
        j_idx_pair = npt - 1 - free_idx
    elif v0_j_idx == v1_j_idx:
      j_idx_pair = v0_j_idx
      if is_forwards(v0, v1):
        i_idx_pair = free_idx
      else:
        i_idx_pair = npt - free_idx - 1
    else:
      raise ValueError("vertex-edge pairing is scuffed")

    return elem_idx_pair, i_idx_pair, j_idx_pair

  # Note: conforming grids should have no singleton vertices of elements.
  for elem_idx in vert_redundancy.keys():
    for i_idx in range(npt):
      for j_idx in range(npt):
        corner_idx = -1
        if i_idx == 0 and j_idx == 0:
          corner_idx = 0
        elif i_idx == npt - 1 and j_idx == npt - 1:
          corner_idx = 3
        elif i_idx == 0 and j_idx == npt - 1:
          corner_idx = 2
        elif i_idx == npt - 1 and j_idx == 0:
          corner_idx = 1

        if corner_idx != -1:
          wrap(elem_idx, i_idx, j_idx)
          for elem_idx_pair, vert_idx_pair in vert_redundancy[elem_idx][corner_idx]:
            i_idx_pair, j_idx_pair = vert_to_i_j(vert_idx_pair)
            vert_redundancy_gll[elem_idx][(i_idx, j_idx)].add((elem_idx_pair, i_idx_pair, j_idx_pair))

        edge_idx = -1
        if j_idx != 0 and j_idx != npt - 1:
          if i_idx == 0:
            edge_idx = LEFT_EDGE
            free_idx = j_idx
          elif i_idx == npt - 1:
            edge_idx = RIGHT_EDGE
            free_idx = j_idx
        if i_idx != 0 and i_idx != npt - 1:
          if j_idx == 0:
            edge_idx = TOP_EDGE
            free_idx = i_idx
          elif j_idx == npt - 1:
            edge_idx = BOTTOM_EDGE
            free_idx = i_idx
        # Note 1: some duplicate work done here, but
        # all grid building code is run at most once!
        # Optimization is not important.
        # Note 2: gll points lying on an edge share at most one neighbor,
        # since we're working in 2D
        if edge_idx != -1:
            elem_idx_pair, i_idx_pair, j_idx_pair = infer_edge(vert_redundancy[elem_idx], edge_idx, free_idx)
            wrap(elem_idx, i_idx, j_idx)
            vert_redundancy_gll[elem_idx][(i_idx, j_idx)].add((elem_idx_pair, i_idx_pair, j_idx_pair))
  return vert_redundancy_gll


def vert_red_flat_to_hierarchy(vert_redundancy_gll_flat):
  vert_redundancy = {}
  for ((target_idx, target_i, target_j),
       (source_idx, source_i, source_j)) in vert_redundancy_gll_flat:
    if target_idx not in vert_redundancy.keys():
      vert_redundancy[target_idx] = {}
    if (target_i, target_j) not in vert_redundancy[target_idx].keys():
      vert_redundancy[target_idx][(target_i, target_j)] = []
    vert_redundancy[target_idx][target_i, target_j].append((source_idx, source_i, source_j))
  return vert_redundancy


def vert_red_hierarchy_to_flat(vert_redundancy_gll):
  vert_redundancy = []

  for target_idx in vert_redundancy_gll.keys():
    for target_i, target_j in vert_redundancy_gll[target_idx].keys():
      for source_idx, source_i, source_j in vert_redundancy_gll[target_idx][(target_i, target_j)]:
        vert_redundancy.append(((target_idx, target_i, target_j),
                                (source_idx, source_i, source_j)))
  return vert_redundancy


def init_element_corner_vert_redundancy(face_connectivity):
  """
  Enumerate redundant DOFs on the elemental
  representation of an arbitrary mesh

  Parameters
  ----------
  nx : `int`
    The number of elements on a cubed-sphere edge.
  face_connectivity : `Array[tuple[elem_idx, edge_idx, 3], Int]`
    An array containing the topological information about the grid.
    It is unpacked as
    ```
    (remote_elem_idx, remote_edge_idx, same_direction) = face_connectivity[local_elem_idx,
                                                                           edge_idx, :]
    ```

  Returns
  -------
  `vert_redundancy: dict[local_elem_idx, dict[vert_idx, set[tuple[remote_elem_idx, vert_idx]]]]`
      `dict[local_elem_idx][vert_idx]` is a set of tuples
      `(remote_elem_idx, vert_idx_pair)` which represent vertices that
      share the same physical coordinates as `(local_elem_idx, vert_idx)`.
      Therefore, they represent redundant degrees of freedom.

  Notes
  -----
  This struct deliberately does not contain diagonal associations, i.e.
  (local_elem_idx, vert_idx) <-/-> (local_elem_idx, vert_idx)
  """
  vert_redundancy = dict()

  def wrap(elem_idx, vert_idx):
    if elem_idx not in vert_redundancy.keys():
      vert_redundancy[elem_idx] = dict()
    if vert_idx not in vert_redundancy[elem_idx].keys():
      vert_redundancy[elem_idx][vert_idx] = set()

  for elem_idx in range(len(face_connectivity)):
    for edge_idx in [TOP_EDGE, LEFT_EDGE, RIGHT_EDGE, BOTTOM_EDGE]:
      idx_pair, edge_idx_pair, is_forwards = face_connectivity[elem_idx, edge_idx, :]
      v0_local, v1_local = edge_to_vert(edge_idx)
      v0_pair, v1_pair = edge_to_vert(edge_idx_pair, is_forwards=is_forwards)
      wrap(elem_idx, v0_local)
      wrap(elem_idx, v1_local)
      vert_redundancy[elem_idx][v0_local].add((idx_pair, v0_pair))
      vert_redundancy[elem_idx][v1_local].add((idx_pair, v1_pair))
  # The following is a crude-but-concise way to ensure
  # (elem_idx_pair, v_idx_pair) in vert_redundancy[elem_idx][v_idx]  <=>
  # (elem_idx, v_idx) in vert_redundancy[elem_idx_pair][v_idx_pair]
  for _ in range(MAX_VERT_DEGREE_UNSTRUCTURED):
    for elem_idx in vert_redundancy.keys():
      for vert_idx in vert_redundancy[elem_idx].keys():
        for elem_idx_pair, vert_idx_pair in vert_redundancy[elem_idx][vert_idx]:
          vert_redundancy[elem_idx_pair][vert_idx_pair].update(vert_redundancy[elem_idx][vert_idx])
  # filter out diagonal
  for elem_idx in vert_redundancy.keys():
    for vert_idx in vert_redundancy[elem_idx].keys():
       if (elem_idx, vert_idx) in vert_redundancy[elem_idx][vert_idx]:
        vert_redundancy[elem_idx][vert_idx].remove((elem_idx, vert_idx))
  return vert_redundancy


def metric_terms_to_grid(gll_latlon,
                         gll_to_cartesian_jacobian,
                         cartesian_to_sphere_jacobian,
                         vert_redundancy_gll,
                         npt,
                         wrapped=use_wrapper,
                         calc_smooth_tensor=False):
  """
  Collate individual coordinate mappings into global SpectralElementGrid
  on an equiangular cubed sphere grid.

  Parameters
  ----------
  gll_latlon: `Array[tuple[elem_idx, gll_idx, gll_idx, phi_lambda], Float]`
      Grid point positions in spherical coordinates.
  gll_to_cube_jacobian : `Array[tuple[elem_idx, gll_idx, gll_idx, xy, ab]`
      Jacobian of mapping from reference element onto cube faces.
  cube_to_sphere_jacobian: `Array[tuple[elem_idx, gll_idx, gll_idx, phi_lambda, xy]`
      Jacobian of mapping from cube face to sphere
  vert_redundancy_gll: `dict[elem_idx, dict[tuple[gll_idx, gll_idx], set[tuple(elem_idx, gll_idx, gll_idx)]]]`
      Gridpoint redundancy struct.
  npt: `int`
      Number of 1D gll points used in grid.
  wrapped: `bool`, default=use_wrapper
      Flag that determines whether returned grid
      will use accelerator framework arrays
      or numpy arrays.

  Notes
  --------
  * See `mesh.gen_gll_redundancy` for a description of `vert_redundancy_gll`
  * See `se_grid.create_spectral_element_grid` for description of
  the grid data structure.

  Returns
  -------
  SpectralElementGrid
    Global spectral element grid.
  """

  gll_to_sphere_jacobian = np.einsum("fijpg,fijps->fijgs", cartesian_to_sphere_jacobian, gll_to_cartesian_jacobian)
  gll_to_sphere_jacobian[:, :, :, 1, :] *= np.cos(gll_latlon[:, :, :, 0])[:, :, :, np.newaxis]
  gll_to_sphere_jacobian_inv = np.linalg.inv(gll_to_sphere_jacobian)

  rmetdet = np.linalg.det(gll_to_sphere_jacobian_inv)

  metdet = 1.0 / rmetdet
  too_close_to_top = np.abs(gll_latlon[:, :, :, 0] - np.pi / 2) < 1e-8
  too_close_to_bottom = np.abs(gll_latlon[:, :, :, 0] + np.pi / 2) < 1e-8
  for i_idx, j_idx, entry in zip([0, 1, 0, 1],
                                 [0, 1, 1, 0],
                                 [1.0, 1.0, 0.0, 0.0]):
    gll_to_sphere_jacobian[:, :, :,
                           i_idx, j_idx] = np.where(np.logical_or(too_close_to_top,
                                                                  too_close_to_bottom),
                                                    entry,
                                                    gll_to_sphere_jacobian[:, :, :, i_idx, j_idx])
    gll_to_sphere_jacobian_inv[:, :, :,
                               i_idx, j_idx] = np.where(np.logical_or(too_close_to_top,
                                                                      too_close_to_bottom),
                                                        entry,
                                                        gll_to_sphere_jacobian_inv[:, :, :, i_idx, j_idx])
  spectrals = init_spectral(npt)

  mass_mat = metdet.copy() * (spectrals["gll_weights"][np.newaxis, :, np.newaxis] *
                              spectrals["gll_weights"][np.newaxis, np.newaxis, :])
  physical_coords_to_cartesian = unit_sphere_to_cart_coords_jacobian(gll_latlon)

  for local_face_idx in vert_redundancy_gll.keys():
    for local_i, local_j in vert_redundancy_gll[local_face_idx].keys():
      for remote_face_id, remote_i, remote_j in vert_redundancy_gll[local_face_idx][(local_i, local_j)]:
        mass_mat[remote_face_id, remote_i, remote_j] += (metdet[local_face_idx, local_i, local_j] *
                                                         (spectrals["gll_weights"][local_i] *
                                                          spectrals["gll_weights"][local_j]))

  inv_mass_mat = 1.0 / mass_mat
  vert_red_flat = vert_red_hierarchy_to_flat(vert_redundancy_gll)

  grid, dims = init_spectral_element_grid(gll_latlon,
                                          gll_to_sphere_jacobian,
                                          gll_to_sphere_jacobian_inv,
                                          physical_coords_to_cartesian,
                                          rmetdet,
                                          metdet,
                                          mass_mat,
                                          inv_mass_mat,
                                          vert_red_flat,
                                          wrapped=wrapped)
  if wrapped and do_sharding:
    grid = shard_grid(grid, dims)
  if calc_smooth_tensor:
    grid = smooth_tensor(grid, dims)
  return grid, dims
