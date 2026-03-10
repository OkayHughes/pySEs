import numpy as np
from .._config import get_backend as _get_backend
from .spectral import init_spectral
from ..operations_2d.horizontal_grid import init_spectral_element_grid, shard_grid, smooth_tensor
from .mesh import vert_red_hierarchy_to_flat
_be = _get_backend()
use_wrapper = _be.use_wrapper
do_sharding = _be.do_sharding


def init_periodic_plane(nx,
                        ny,
                        npt,
                        length_x=2.0,
                        length_y=2.0):
  """
  Generate grid topology for an axis-aligned
  regular grid on a doubly periodic plane.

  Parameters
  ----------
  nx : `int`
      Number of grid cells in the horizontal direction
  ny : `int`
      Number of grid cells in the vertical direction
  npt : `int`
      Number of Gauss-Lobatto-Legendre points per dimension in reference element.
  length_x : `float`, default=2.0
      Length of grid in the horizontal direction
  length_y : `float`, default=2.0
      Length of grid in the vertical direction

  Returns
  -------
  physical_coords : `Array[tuple[elem_idx, gll_idx, gll_idx, xy], Float]`
      Position of GLL gridpoints in the plane.
  ref_to_planar : `Array[tuple[elem_idx, gll_idx, gll_idx, xy, ab], Float]`
      Jacobian of bilinear mapping from the reference element to cartesian space
      for the GLL mesh.
  vert_redundancy_gll : `dict[elem_idx, dict[tuple(gll_idx, gll_idx),\
                                             set[tuple[elem_idx, gll_idx, gll_idx]]]]`
      `dict[elem_idx][(gll_idx,gll_idx)]` is a set
      of redundant DOFs on the global GLL grid.

  Raises
  ------
  KeyError
      when a key error
  """
  spectrals = init_spectral(npt)
  gll_pts = spectrals["gll_points"]
  elem_boundaries_x_1d = np.linspace(-length_x / 2.0, length_x / 2.0, nx + 1)
  elem_boundaries_y_1d = np.linspace(length_y / 2.0, -length_y / 2.0, ny + 1)
  elem_boundaries_x, elem_boundaries_y = np.meshgrid(elem_boundaries_x_1d,
                                                     elem_boundaries_y_1d)
  elem_boundaries_x = elem_boundaries_x.flatten()
  elem_boundaries_y = elem_boundaries_y.flatten()
  l_idxs_2d, m_idxs_2d = np.meshgrid(np.arange(nx), np.arange(ny))
  l_idxs = l_idxs_2d.flatten()
  m_idxs = m_idxs_2d.flatten()
  elem_centers_x_1d = 0.5 * (elem_boundaries_x_1d[1:] + elem_boundaries_x_1d[:-1])
  elem_centers_y_1d = 0.5 * (elem_boundaries_y_1d[1:] + elem_boundaries_y_1d[:-1])
  elem_centers_x, elem_centers_y = np.meshgrid(elem_centers_x_1d,
                                               elem_centers_y_1d)
  elem_centers_x = elem_centers_x.flatten()
  elem_centers_y = elem_centers_y.flatten()
  idx_hack = np.arange(l_idxs.size).reshape(l_idxs_2d.shape)

  dx = elem_boundaries_x_1d[1] - elem_boundaries_x_1d[0]
  dy = elem_boundaries_y_1d[1] - elem_boundaries_y_1d[0]
  gll_squished_x, gll_squished_y = np.meshgrid(dx / 2.0 * gll_pts,
                                               dy / 2.0 * gll_pts)
  x_pos = gll_squished_x[np.newaxis, :, :] + elem_centers_x[:, np.newaxis, np.newaxis]
  y_pos = gll_squished_y[np.newaxis, :, :] + elem_centers_y[:, np.newaxis, np.newaxis]
  x_pos = np.flip(np.swapaxes(x_pos, 1, 2), axis=1)
  y_pos = np.swapaxes(y_pos, 1, 2)
  physical_coords = np.stack((x_pos, y_pos), axis=-1)
  ref_to_planar_gs = np.array([[dx / 2.0, 0],
                               [0, -dy / 2.0]], dtype=np.float64)
  ref_to_planar = (ref_to_planar_gs[np.newaxis, np.newaxis, np.newaxis, :, :] *
                   np.ones_like(x_pos)[:, :, :, np.newaxis, np.newaxis])

  vert_redundancy_gll = {}

  def wrap(f_idx, i_idx, j_idx):
    if f_idx not in vert_redundancy_gll.keys():
      vert_redundancy_gll[f_idx] = {}
    if (i_idx, j_idx) not in vert_redundancy_gll[f_idx].keys():
      vert_redundancy_gll[f_idx][(i_idx, j_idx)] = set()

  last_pt = npt - 1
  first_pt = 0
  for f_idx in range(elem_centers_x.shape[0]):
    l_idx = l_idxs[f_idx]
    m_idx = m_idxs[f_idx]
    next_l = (l_idx + 1) % nx
    prev_l = l_idx - 1
    next_m = (m_idx + 1) % ny
    prev_m = m_idx - 1
    # handle left edge
    left_element_id = idx_hack[m_idx, prev_l]
    for j_idx in range(npt):
      wrap(f_idx, first_pt, j_idx)
      vert_redundancy_gll[f_idx][(first_pt, j_idx)].add((left_element_id, last_pt, j_idx))
    # handle top edge
    top_element_id = idx_hack[prev_m, l_idx]
    for i_idx in range(npt):
      wrap(f_idx, i_idx, last_pt)
      vert_redundancy_gll[f_idx][(i_idx, last_pt)].add((top_element_id, i_idx, first_pt))
    # handle right edge
    right_element_id = idx_hack[m_idx, next_l]
    for j_idx in range(npt):
      wrap(f_idx, last_pt, j_idx)
      vert_redundancy_gll[f_idx][(last_pt, j_idx)].add((right_element_id, first_pt, j_idx))
    # handle bottom edge
    bottom_element_id = idx_hack[next_m, l_idx]
    for i_idx in range(npt):
      wrap(f_idx, i_idx, first_pt)
      vert_redundancy_gll[f_idx][(i_idx, first_pt)].add((bottom_element_id, i_idx, last_pt))
    # handle top left corner
    top_left_element = idx_hack[prev_m, prev_l]
    wrap(f_idx, first_pt, last_pt)
    vert_redundancy_gll[f_idx][(first_pt, last_pt)].add((top_left_element, last_pt, first_pt))
    # handle top right corner
    top_right_element = idx_hack[prev_m, next_l]
    wrap(f_idx, last_pt, last_pt)
    vert_redundancy_gll[f_idx][(last_pt, last_pt)].add((top_right_element, first_pt, first_pt))
    # handle bottom right corner
    bottom_right_element = idx_hack[next_m, next_l]
    wrap(f_idx, last_pt, first_pt)
    vert_redundancy_gll[f_idx][(last_pt, first_pt)].add((bottom_right_element, first_pt, last_pt))
    # handle bottom left corner
    bottom_left_element = idx_hack[next_m, prev_l]
    wrap(f_idx, first_pt, first_pt)
    vert_redundancy_gll[f_idx][(first_pt, first_pt)].add((bottom_left_element, last_pt, last_pt))
  return physical_coords, ref_to_planar, vert_redundancy_gll


def metric_terms_to_grid(physical_coords,
                         gll_to_planar_jacobian,
                         vert_redundancy_gll,
                         npt,
                         calc_smooth_tensor=False,
                         wrapped=use_wrapper):
  """
    Collate individual coordinate mappings into into global SpectralElementGrid
    on a periodic plane.

  Parameters
  ----------
  physical_coords : `Array[tuple[elem_idx, gll_idx, gll_idx, phi_lambda], Float]`
      Grid point positions in (x, y) coordinates,
  gll_to_planar_jacobian : `Array[tuple[elem_idx, gll_idx, gll_idx, xy, ab]`
      Jacobian of mapping from reference element onto plane.
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
  See `init_periodic_plane` for how to initialize `physical_coords`,
  `gll_to_planar_jacobian`, and `vert_redundancy_grid`.

  Returns
  -------
  SpectralElementGrid
    Global spectral element grid.
  """

  spectrals = init_spectral(npt)
  NELEM = physical_coords.shape[0]

  gll_to_planar_jacobian_inv = np.linalg.inv(gll_to_planar_jacobian)

  physical_coords_to_cartesian = (np.eye(3)[np.newaxis, np.newaxis, np.newaxis, :, :2] *
                                  np.ones((NELEM, npt, npt, 1, 1)))

  rmetdet = np.linalg.det(gll_to_planar_jacobian_inv)

  metdet = 1.0 / rmetdet

  mass_mat = metdet.copy() * (spectrals["gll_weights"][np.newaxis, :, np.newaxis] *
                              spectrals["gll_weights"][np.newaxis, np.newaxis, :])

  for local_face_idx in vert_redundancy_gll.keys():
    for local_i, local_j in vert_redundancy_gll[local_face_idx].keys():
      for remote_face_id, remote_i, remote_j in vert_redundancy_gll[local_face_idx][(local_i, local_j)]:
        mass_mat[remote_face_id, remote_i, remote_j] += (metdet[local_face_idx, local_i, local_j] *
                                                         (spectrals["gll_weights"][local_i] *
                                                          spectrals["gll_weights"][local_j]))

  inv_mass_mat = 1.0 / mass_mat
  vert_red_flat = vert_red_hierarchy_to_flat(vert_redundancy_gll)

  grid, dims = init_spectral_element_grid(physical_coords,
                                          gll_to_planar_jacobian,
                                          gll_to_planar_jacobian_inv,
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


def init_uniform_grid(nx,
                      ny,
                      npt,
                      length_x=2.0,
                      length_y=2.0,
                      wrapped=use_wrapper,
                      calc_smooth_tensor=False):
  """
  Generate a uniform doubly periodic
  SpectralElementGrid on an axis-aligned cartesian plane.

  Parameters
  ----------
  nx : `int`
      Number of grid cells in the horizontal direction
  ny : `int`
      Number of grid cells in the vertical direction
  npt : `int`
      Number of Gauss-Lobatto-Legendre points per dimension in reference element.
  length_x : `float`, default=2.0
      Length of grid in the horizontal direction
  length_y : `float`, default=2.0
      Length of grid in the vertical direction
  wrapped: `bool`, default=use_wrapper
      Flag that determines whether returned grid
      will use accelerator framework arrays
      or numpy arrays.
  Returns
  -------
  SpectralElementGrid
    Global spectral element grid.
  """
  physical_coords, ref_to_planar, vert_red = init_periodic_plane(nx, ny, npt, length_x=length_x, length_y=length_y)
  grid, dims = metric_terms_to_grid(physical_coords,
                                    ref_to_planar,
                                    vert_red,
                                    npt,
                                    calc_smooth_tensor=calc_smooth_tensor,
                                    wrapped=wrapped)
  return grid, dims
