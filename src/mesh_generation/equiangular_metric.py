import numpy as np
from .._config import get_backend as _get_backend
from .mesh import mesh_to_cart_bilinear, init_spectral_grid_redundancy, metric_terms_to_grid
from .mesh_definitions import TOP_FACE, BOTTOM_FACE, FRONT_FACE, BACK_FACE, LEFT_FACE, RIGHT_FACE
from .cubed_sphere import init_cube_topo
from .mesh import init_element_corner_vert_redundancy
_be = _get_backend()
DEBUG = _be.debug
use_wrapper = _be.use_wrapper


def eval_metric_terms_equiangular(face_mask,
                                  cube_points_2d,
                                  npt):
  """
  Use the equiangular cubed sphere map
  to evaluate latitude, longitude and jacobian ∂(x, y)/∂(λ, φ)
  for a quasi-regular cubed-sphere grid.

  Parameters
  ----------
  face_mask: `Array[tuple[elem_idx], Int]`
    Integer mask denoting which cubed sphere face each element belongs to.
  cube_points_2d: `Array[tuple[elem_idx, gll_idx, gll_idx, xy], Float]`
    Local (x, y) coordinates of grid points
    in the cubed sphere face containing the element
  npt: `int`
    Number of 1d GLL points within each element.

  Returns
  -------
  gll_latlon: `Array[tuple[elem_idx, gll_idx, gll_idx, phi_lambda], Float]`
      Gridpoint locations in spherical coordinates.
  cube_to_sphere_jacobian: `Array[tuple[elem_idx, gll_idx, gll_idx, phi_lambda, xy], Float]`
      Jacobian of mapping from xy coordinates on cubed sphere face to the sphere.
      Namely,
      ```
      cube_to_sphere_jacobian[elem_idx, gll_idx, gll_idx, :, :] = [[∂ϕ/∂x, ∂ϕ/∂y],
                                                                   [∂λ/∂x, ∂λ/∂y]]
      ```


  Notes
  -----
  See `mesh_definitions.py` for a diagram of how the local coordinates
  are oriented on each face.

  See Ullrich, Lauritzen, and Jablonowski (https://doi.org/10.1175/2008MWR2817.1)
  for a thorough review of cubed sphere geometry.

  `cube_to_sphere_jacobian` is not C0 across cubed-sphere boundaries!
  """
  NFACES = cube_points_2d.shape[0]

  top_face_mask = (face_mask == TOP_FACE)[:, np.newaxis, np.newaxis]
  bottom_face_mask = (face_mask == BOTTOM_FACE)[:, np.newaxis, np.newaxis]
  left_face_mask = (face_mask == LEFT_FACE)[:, np.newaxis, np.newaxis]
  right_face_mask = (face_mask == RIGHT_FACE)[:, np.newaxis, np.newaxis]
  front_face_mask = (face_mask == FRONT_FACE)[:, np.newaxis, np.newaxis]
  back_face_mask = (face_mask == BACK_FACE)[:, np.newaxis, np.newaxis]

  gll_latlon = np.zeros(shape=(NFACES, npt, npt, 2))
  cube_to_sphere_jacobian = np.zeros(shape=(NFACES, npt, npt, 2, 2))
  if DEBUG:
    gll_latlon_pert = np.zeros(shape=(NFACES, npt, npt, 2))
    cube_points_pert = np.zeros_like(cube_points_2d)

  if DEBUG:
    n_mask = 0
    for m1, mask1 in enumerate([top_face_mask,
                                bottom_face_mask,
                                front_face_mask,
                                back_face_mask,
                                left_face_mask,
                                right_face_mask]):
      for m2, mask2 in enumerate([top_face_mask,
                                  bottom_face_mask,
                                  front_face_mask,
                                  back_face_mask,
                                  left_face_mask,
                                  right_face_mask]):
        if m1 != m2:
          ct = np.sum(np.logical_and(mask1, mask2))
          assert (ct == 0)
      n_mask += np.sum(mask1)
    assert (n_mask == NFACES)

  def set_jac_eq(jac, lat, lon, mask):
    """
    Calculate the equiangular cubed sphere jacobian on equatorial panels.

    Parameters
    ----------
    jac : `Array[tuple[elem_idx, gll_idx, gll_idx, phi_lambda, xy], Float]`
      Jacobian to set at gridpoints where mask is true
    lat : `Array[tuple[elem_idx, gll_idx, gll_idx], Float]`
      Latitude, [-π/2, π/2]
    lon :`Array[tuple[elem_idx, gll_idx, gll_idx], Float]`
      Longitude, [0, 2π]
    mask : `Array[tuple[elem_idx, gll_idx, gll_idx], Bool]`
      Mask which is True at grid points where jacobialn should be set
    """
    jac[:, :, :, 0, 1] += np.cos(lon[:, :, :])**2 * mask
    jac[:, :, :, 0, 0] += -1 / 4 * np.sin(2 * lon[:, :, :]) * np.sin(2 * lat[:, :, :]) * mask
    jac[:, :, :, 1, 0] += np.cos(lon[:, :, :]) * np.cos(lat[:, :, :])**2 * mask

  def set_jac_pole(jac, lat, lon, mask, k):
    """
    Calculate the equiangular cubed sphere jacobian on polar panels.

    Parameters
    ----------
    jac : `Array[tuple[elem_idx, gll_idx, gll_idx, phi_lambda, xy], Float]`
      Jacobian to set at gridpoints where mask is true
    lat : `Array[tuple[elem_idx, gll_idx, gll_idx], Float]`
      Latitude, [-π/2, π/2]
    lon : `Array[tuple[elem_idx, gll_idx, gll_idx], Float]`
      Longitude, [0, 2π]
    mask : `Array[tuple[elem_idx, gll_idx, gll_idx], Bool]`
      Mask which is True at grid points where jacobialn should be set
    k : float
      1.0 on top panel, -1.0 on bottom panel
    """
    jac[:, :, :, 0, 1] += k * np.cos(lon) * np.tan(lat) * mask
    jac[:, :, :, 0, 0] += -k * np.sin(lon) * np.sin(lat)**2 * mask
    jac[:, :, :, 1, 1] += np.sin(lon) * np.tan(lat) * mask
    jac[:, :, :, 1, 0] += np.cos(lon) * np.sin(lat)**2 * mask

  def dlatlon_dcube(latlon_fn, latlon_idx, cube_idx, mask):
    """
    Use finite differences to calculate approximate Jacobian

    Parameters
    ----------
    latlon_fn : `Callable[[Array, Array], Array]`
        Takes x, y and returns lat if `latlon_idx = 0`,
        lon if `latlon_idx = 1`
    latlon_idx: `int`
        0 if latlon_fn returns lat, 1 if latlon_fn returns lon.
        the 2nd param
    cube_idx: `int`
        0 if differentiating w.r.t. x, 1 if differentiating w.r.t. y
    mask: `Array`
        Mask that is true at points at which approximate
        jacobian should be calculated.
    """
    gll_latlon_pert[:] = 0
    cube_points_pert[:] = cube_points_2d[:]
    cube_points_pert[:, :, :, cube_idx] *= 0.99999
    gll_latlon_pert[:, :, :, latlon_idx] += latlon_fn(cube_points_pert[:, :, :, 0], cube_points_pert[:, :, :, 1]) * mask
    result = ((gll_latlon_pert[:, :, :, latlon_idx] - gll_latlon[:, :, :, latlon_idx]) /
              (cube_points_pert[:, :, :, cube_idx] - cube_points_2d[:, :, :, cube_idx]))
    return result

  def test_face(lat_fn, lon_fn, mask):
    """
    Test if analytic jacobian and
    approximate jacobian are approximately equal.

    Parameters
    ----------
    lat_fn: `Callable[[Array, Array], Array]`
        Takes (x, y) and returns latitude.
    lon_fn: `Callable[[Array, Array], Array]`
        Takes (x, y) and returns longitude.
    mask: `Array`
        Entries are true at gridpoints
        where equivalence should be tested.
    """
    dlat_dx = dlatlon_dcube(lat_fn, 0, 0, mask)
    dlat_dy = dlatlon_dcube(lat_fn, 0, 1, mask)
    dlon_dx = dlatlon_dcube(lon_fn, 1, 0, mask)
    dlon_dy = dlatlon_dcube(lon_fn, 1, 1, mask)
    jac_tmp = np.zeros((2, 2))
    check1 = mask * (dlat_dx - cube_to_sphere_jacobian[:, :, :, 0, 0])
    check2 = mask * (dlat_dy - cube_to_sphere_jacobian[:, :, :, 1, 0])
    check3 = mask * (dlon_dx - cube_to_sphere_jacobian[:, :, :, 0, 1])
    check4 = mask * (dlon_dy - cube_to_sphere_jacobian[:, :, :, 1, 1])
    try:
      assert (np.max(check1) < 1e-3)
      assert (np.max(check2) < 1e-3)
      assert (np.max(check3) < 1e-3)
      assert (np.max(check4) < 1e-3)
    except AssertionError:
      for face_idx in range(NFACES):
        if mask[face_idx]:
          i_idx, j_idx = (1, 1)
          jac_tmp[0, 0] = dlat_dx[face_idx, i_idx, j_idx]
          jac_tmp[0, 1] = dlon_dx[face_idx, i_idx, j_idx]
          jac_tmp[1, 0] = dlat_dy[face_idx, i_idx, j_idx]
          jac_tmp[1, 1] = dlon_dy[face_idx, i_idx, j_idx]
          err_str = (f"Face: {face_idx},\n"
                     "vvvvvvvvvvvvvvvvvvv\n"
                     "Numerical jac: \n"
                     f"{jac_tmp}, \n"
                     f"Analytic jac: \n"
                     f"{cube_to_sphere_jacobian[face_idx, i_idx, j_idx, :, :]}\n"
                     "^^^^^^^^^^^^^^^^^\n ")
          if False:
            print(err_str)

  # front face
  def front_lat(x, y):
    return (np.arctan2(y, np.sqrt(1 + x**2)))

  def front_lon(x, y):
    return np.mod((np.arctan(x) + 2 * np.pi), 2 * np.pi)

  gll_latlon[:, :, :, 0] += front_lat(cube_points_2d[:, :, :, 0], cube_points_2d[:, :, :, 1]) * front_face_mask
  gll_latlon[:, :, :, 1] += front_lon(cube_points_2d[:, :, :, 0], cube_points_2d[:, :, :, 1]) * front_face_mask
  set_jac_eq(cube_to_sphere_jacobian, gll_latlon[:, :, :, 0], gll_latlon[:, :, :, 1], front_face_mask)
  if DEBUG:
    test_face(front_lat, front_lon, front_face_mask)

  # right face
  def right_lat(x, y):
    return (np.arctan2(y, np.sqrt(1 + x**2)))

  def right_lon(x, y):
    return np.arctan(x) + np.pi / 2

  gll_latlon[:, :, :, 0] += right_lat(cube_points_2d[:, :, :, 0], cube_points_2d[:, :, :, 1]) * right_face_mask
  gll_latlon[:, :, :, 1] += right_lon(cube_points_2d[:, :, :, 0], cube_points_2d[:, :, :, 1]) * right_face_mask
  set_jac_eq(cube_to_sphere_jacobian, gll_latlon[:, :, :, 0], -np.pi / 2 + gll_latlon[:, :, :, 1], right_face_mask)
  if DEBUG:
    test_face(right_lat, right_lon, right_face_mask)

  # back face
  def back_lat(x, y):
    return (np.arctan2(y, np.sqrt(1 + x**2)))

  def back_lon(x, y):
    return (np.arctan(x) + np.pi)
  gll_latlon[:, :, :, 0] += back_lat(cube_points_2d[:, :, :, 0], cube_points_2d[:, :, :, 1]) * back_face_mask
  gll_latlon[:, :, :, 1] += back_lon(cube_points_2d[:, :, :, 0], cube_points_2d[:, :, :, 1]) * back_face_mask
  set_jac_eq(cube_to_sphere_jacobian, gll_latlon[:, :, :, 0], -np.pi + gll_latlon[:, :, :, 1], back_face_mask)
  if DEBUG:
    test_face(back_lat, back_lon, back_face_mask)

  # left face
  def left_lat(x, y):
    return (np.arctan2(y, np.sqrt(1 + x**2)))

  def left_lon(x, y):
    return (np.arctan(x) + 3 * np.pi / 2)

  gll_latlon[:, :, :, 0] += left_lat(cube_points_2d[:, :, :, 0], cube_points_2d[:, :, :, 1]) * left_face_mask
  gll_latlon[:, :, :, 1] += left_lon(cube_points_2d[:, :, :, 0], cube_points_2d[:, :, :, 1]) * left_face_mask
  set_jac_eq(cube_to_sphere_jacobian, gll_latlon[:, :, :, 0], -3 * np.pi / 2 + gll_latlon[:, :, :, 1], left_face_mask)
  if DEBUG:
    test_face(left_lat, left_lon, left_face_mask)

  # top face
  def top_lat(x, y):
    return (np.arctan2(1, np.sqrt(x**2 + y**2)))

  def top_lon(x, y):
    return np.mod((np.arctan2(x, -y)), 2 * np.pi)

  gll_latlon[:, :, :, 0] += top_lat(cube_points_2d[:, :, :, 0], cube_points_2d[:, :, :, 1]) * top_face_mask
  gll_latlon[:, :, :, 1] += top_lon(cube_points_2d[:, :, :, 0], cube_points_2d[:, :, :, 1]) * top_face_mask
  set_jac_pole(cube_to_sphere_jacobian, gll_latlon[:, :, :, 0], gll_latlon[:, :, :, 1], top_face_mask, 1.0)
  if DEBUG:
    test_face(top_lat, top_lon, top_face_mask)

  # bottom face
  def bottom_lat(x, y):
    return -np.arctan2(1, np.sqrt(x**2 + y**2))

  def bottom_lon(x, y):
    return np.mod((np.arctan2(x, y)), 2 * np.pi)

  gll_latlon[:, :, :, 0] += bottom_lat(cube_points_2d[:, :, :, 0], cube_points_2d[:, :, :, 1]) * bottom_face_mask
  gll_latlon[:, :, :, 1] += bottom_lon(cube_points_2d[:, :, :, 0], cube_points_2d[:, :, :, 1]) * bottom_face_mask
  set_jac_pole(cube_to_sphere_jacobian, gll_latlon[:, :, :, 0], gll_latlon[:, :, :, 1], bottom_face_mask, -1.0)

  if DEBUG:
    test_face(bottom_lat, bottom_lon, bottom_face_mask)

  gll_latlon[:, :, :, 1] = np.mod(gll_latlon[:, :, :, 1], 2 * np.pi - 1e-9)
  too_close_to_top = np.abs(gll_latlon[:, :, :, 0] - np.pi / 2) < 1e-8
  too_close_to_bottom = np.abs(gll_latlon[:, :, :, 0] + np.pi / 2) < 1e-8
  mask = np.logical_or(too_close_to_top,
                       too_close_to_bottom)
  gll_latlon[:, :, :, 1] = np.where(mask, 0.0, gll_latlon[:, :, :, 1])
  # if True:
  #   theta = 1e-5
  #   lat = gll_latlon[:, :, :, 0]
  #   lon = gll_latlon[:, :, :, 1]
  #   xyz = np.stack((np.cos(lon) * np.cos(lat),
  #                   np.sin(lon) * np.cos(lat),
  #                   np.sin(lat)), axis=-1)

  #   rotation_matrix = np.array([[1.0, 0.0, 0.0],
  #                               [0.0, np.cos(theta), -np.sin(theta)],
  #                               (0.0, np.sin(theta), np.cos(theta))])
  #   xyz_rotated = np.einsum("kl, fijk", rotation_matrix, xyz)
  #   gll_latlon = np.stack((np.asin(xyz_rotated[:, :, :, 2]),
  #                          np.atan2(xyz_rotated[:, :, :, 1],
  #                                   xyz_rotated[:, :, :, 0])), axis=-1)

  return gll_latlon, cube_to_sphere_jacobian


def init_grid_from_topo(face_connectivity,
                        face_mask,
                        face_position_2d,
                        vert_redundancy,
                        npt,
                        calc_smooth_tensor=False,
                        wrapped=use_wrapper,):
  """
  Generate SpectralElementGrid from topological information.

  Parameters
  ----------
  face_connectivity: `Array[tuple[elem_idx, edge_idx, 3], Int]`
    An array containing the topological information about the grid.
    It is unpacked as
    ```
    (remote_elem_idx, remote_edge_idx, same_direction) = face_connectivity[local_elem_idx,
                                                                           edge_idx, :]
    ```
  face_mask: `Array[tuple[elem_idx], Int]`
    An integer mask describing which face of the cubed sphere each element lies on.
  face_position_2d: `Array[tuple[elem_idx, vert_idx, xy], Float]`
    Positions of the element vertices within the local (x, y)
    coordinates on the cubed-sphere face that contains it.
  vert_redundancy: `dict[local_elem_idx, dict[vert_idx, set[tuple[remote_elem_idx, vert_idx]]]]`
      `dict[local_elem_idx][vert_idx]` is a set of tuples
      `(remote_elem_idx, vert_idx_pair)` which represent vertices that
      share the same physical coordinates as `(local_elem_idx, vert_idx)`.
      Therefore, they represent redundant degrees of freedom.
  wrapped: `bool`, default=use_wrapper
      Flag that determines whether returned grid
      will use accelerator framework arrays
      or numpy arrays.

  Notes
  --------
  * See `cubed_sphere.gen_cube_topo` to generate `face_connectivity`, `face_mask`, `face_position_2d`
  * See `cubed_sphere.gen_vert_redundancy` to generate `vert_redundancy`

  Returns
  -------
  SpectralElementGrid
    Global spectral element grid.
  """
  gll_position, gll_jacobian = mesh_to_cart_bilinear(face_position_2d, npt)
  cube_redundancy = init_spectral_grid_redundancy(vert_redundancy, npt)
  gll_latlon, cube_to_sphere_jacobian = eval_metric_terms_equiangular(face_mask, gll_position, npt)
  return metric_terms_to_grid(gll_latlon,
                              gll_jacobian,
                              cube_to_sphere_jacobian,
                              cube_redundancy,
                              npt,
                              calc_smooth_tensor=calc_smooth_tensor,
                              wrapped=wrapped)


def init_quasi_uniform_grid(nx,
                            npt,
                            wrapped=use_wrapper,
                            calc_smooth_tensor=False):
  """
  Generate an equiangular quasi-uniform cubed-sphere
  SpectralElementGrid.

  Parameters
  ----------
  nx : `int`
      Number of elements per edge of a cubed sphere face
  npt : `int`
      Number of 1D gll points to use within reference elements.
  wrapped: `bool`, default=use_wrapper
      Flag that determines whether returned grid
      will use accelerator framework arrays
      or numpy arrays.
  Returns
  -------
  SpectralElementGrid
    Global spectral element grid.
  """
  face_connectivity, face_mask, face_position, face_position_2d = init_cube_topo(nx)
  vert_redundancy = init_element_corner_vert_redundancy(face_connectivity)
  return init_grid_from_topo(face_connectivity,
                             face_mask,
                             face_position_2d,
                             vert_redundancy,
                             npt,
                             wrapped=wrapped,
                             calc_smooth_tensor=calc_smooth_tensor)
