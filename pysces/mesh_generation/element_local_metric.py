from ..config import use_wrapper, np
from .mesh import (init_element_corner_vert_redundancy,
                   init_spectral_grid_redundancy,
                   mesh_to_cart_bilinear,
                   metric_terms_to_grid)
from .equiangular_metric import eval_metric_terms_equiangular
from .cubed_sphere import init_cube_topo
from .spherical_coord_utils import (unit_sphere_to_cart_coords,
                                    cart_to_unit_sphere_coords_jacobian,
                                    cart_to_unit_sphere_coords)


def eval_metric_terms_elem_local(latlon_corners,
                                 npt,
                                 rotate=False):
  cart_corners = unit_sphere_to_cart_coords(latlon_corners)
  if rotate:
    theta = 1e-3

    rotation_matrix = np.array([[1.0, 0.0, 0.0],
                                [0.0, np.cos(theta), -np.sin(theta)],
                                (0.0, np.sin(theta), np.cos(theta))])
    cart_corners = np.einsum("kl, fvk->fvl", rotation_matrix, cart_corners)
  cart_points_3d, gll_to_cart_jacobian = mesh_to_cart_bilinear(cart_corners, npt)
  norm_cart = np.linalg.norm(cart_points_3d, axis=-1)
  gll_xyz = cart_points_3d / norm_cart[:, :, :, np.newaxis]
  # 1/‖p‖³ (‖p‖² I − pp⊤)
  cart_to_unit_sphere_jacobian = (1.0 / norm_cart[:, :, :, np.newaxis, np.newaxis]**3 *
                                  (norm_cart[:, :, :, np.newaxis, np.newaxis]**2 *
                                   np.eye(3)[np.newaxis, np.newaxis, np.newaxis, :, :] -
                                   np.einsum("fijc,fijd->fijcd", cart_points_3d, cart_points_3d)))

  gll_latlon = cart_to_unit_sphere_coords(gll_xyz)
  unit_sphere_to_sph_coords_jacobian = cart_to_unit_sphere_coords_jacobian(gll_xyz)

  cart_to_sphere_jacobian = np.einsum("fijcd,fijsc->fijds",
                                      cart_to_unit_sphere_jacobian,
                                      unit_sphere_to_sph_coords_jacobian)

  # gll_latlon[:, :, :, 1] = np.mod(gll_latlon[:, :, :, 1], 2 * np.pi - 1e-9)
  # too_close_to_top = np.abs(gll_latlon[:, :, :, 0] - np.pi / 2) < 1e-8
  # too_close_to_bottom = np.abs(gll_latlon[:, :, :, 0] + np.pi / 2) < 1e-8
  # mask = np.logical_or(too_close_to_top,
  #                      too_close_to_bottom)
  # gll_latlon[:, :, :, 1] = np.where(mask, 0.0, gll_latlon[:, :, :, 1])
  return gll_latlon, gll_to_cart_jacobian, cart_to_sphere_jacobian


def init_quasi_uniform_grid_elem_local(nx,
                                       npt,
                                       wrapped=use_wrapper,
                                       calc_smooth_tensor=False,
                                       rotate=True):
  face_connectivity, face_mask, face_position, face_position_2d = init_cube_topo(nx)
  vert_redundancy = init_element_corner_vert_redundancy(face_connectivity)
  gll_position_equi, gll_jacobian = mesh_to_cart_bilinear(face_position_2d, npt)
  cube_redundancy = init_spectral_grid_redundancy(vert_redundancy, npt)
  gll_latlon_equi, _ = eval_metric_terms_equiangular(face_mask, gll_position_equi, npt)
  latlon_corners = np.zeros((gll_latlon_equi.shape[0], 4, 2))
  for vert_idx, (i_in, j_in) in enumerate([(0, 0), (npt - 1, 0), (0, npt - 1), (npt - 1, npt - 1)]):
      latlon_corners[:, vert_idx, :] = gll_latlon_equi[:, i_in, j_in, :]

  # too_close_to_top = np.abs(latlon_corners[:, :, 0] - np.pi / 2) < 1e-8
  # too_close_to_bottom = np.abs(latlon_corners[:, :, 0] + np.pi / 2) < 1e-8
  # mask = np.logical_or(too_close_to_top,
  #                      too_close_to_bottom)
  # latlon_corners[:, :, 1] = np.where(mask, 0.0, latlon_corners[:, :, 1])

  gll_latlon, gll_to_cart_jacobian, cart_to_sphere_jacobian = eval_metric_terms_elem_local(latlon_corners,
                                                                                           npt,
                                                                                           rotate=rotate)

  return metric_terms_to_grid(gll_latlon,
                              gll_to_cart_jacobian,
                              cart_to_sphere_jacobian,
                              cube_redundancy,
                              npt,
                              calc_smooth_tensor=calc_smooth_tensor,
                              wrapped=wrapped)


def init_stretched_grid_elem_local(nx,
                                   npt,
                                   axis_dilation=None,
                                   orthogonal_transform=None,
                                   offset=None,
                                   wrapped=use_wrapper,
                                   calc_smooth_tensor=False,
                                   rotate=True):
  if axis_dilation is None:
    axis_dilation = np.ones((3,))
  if orthogonal_transform is None:
    orthogonal_transform = np.eye(3)
  if offset is None:
     offset = np.zeros((3,))
  face_connectivity, face_mask, face_position, face_position_2d = init_cube_topo(nx)
  vert_redundancy = init_element_corner_vert_redundancy(face_connectivity)
  gll_position_equi, gll_jacobian = mesh_to_cart_bilinear(face_position_2d, npt)
  cube_redundancy = init_spectral_grid_redundancy(vert_redundancy, npt)
  # generate base equiangular grid and extract corners
  gll_latlon_equi, _ = eval_metric_terms_equiangular(face_mask, gll_position_equi, npt)
  latlon_corners = np.zeros((gll_latlon_equi.shape[0], 4, 2))
  for vert_idx, (i_in, j_in) in enumerate([(0, 0), (npt - 1, 0), (0, npt - 1), (npt - 1, npt - 1)]):
      latlon_corners[:, vert_idx, :] = gll_latlon_equi[:, i_in, j_in, :]

  # Apply mapping x' = Q diag(s) x + c, then map x'' = x'/||x'||
  cart_corners = unit_sphere_to_cart_coords(latlon_corners)
  inverse_image_of_offset = np.einsum("c,dc,c->d", offset, orthogonal_transform, 1.0 / axis_dilation)
  assert np.allclose(np.dot(orthogonal_transform,
                            orthogonal_transform.T),
                     np.eye(3)), "Rotation matrix is not orthogonal"
  message = ("Mapping maps unit sphere to set that does not contain origin.\n"
             "The resulting transformation will be C0, but not bijective")
  assert np.linalg.norm(inverse_image_of_offset) < 1.0, message

  cart_corners = np.einsum("fvc,dc,c->fvd", cart_corners, orthogonal_transform, axis_dilation)
  cart_corners += offset[np.newaxis, np.newaxis, :]
  cart_corners /= np.linalg.norm(cart_corners, axis=-1)[:, :, np.newaxis]
  latlon_corners = cart_to_unit_sphere_coords(cart_corners[:, :, np.newaxis, :])[:, :, 0, :]

  gll_latlon, gll_to_cart_jacobian, cart_to_sphere_jacobian = eval_metric_terms_elem_local(latlon_corners,
                                                                                           npt,
                                                                                           rotate=rotate)

  return metric_terms_to_grid(gll_latlon,
                              gll_to_cart_jacobian,
                              cart_to_sphere_jacobian,
                              cube_redundancy,
                              npt,
                              wrapped=wrapped,
                              calc_smooth_tensor=calc_smooth_tensor)


def init_unstructured_grid(face_connectivity,
                           corner_cart_positions,
                           npt,
                           wrapped=use_wrapper,
                           calc_smooth_tensor=False,
                           rotate=False):
  vert_redundancy = init_element_corner_vert_redundancy(face_connectivity)
  print("vert redundancy finished")
  cube_redundancy = init_spectral_grid_redundancy(vert_redundancy, npt)
  print("spectral redundancy finished")

  # too_close_to_top = np.abs(latlon_corners[:, :, 0] - np.pi / 2) < 1e-8
  # too_close_to_bottom = np.abs(latlon_corners[:, :, 0] + np.pi / 2) < 1e-8
  # mask = np.logical_or(too_close_to_top,
  #                      too_close_to_bottom)
  # latlon_corners[:, :, 1] = np.where(mask, 0.0, latlon_corners[:, :, 1])
  latlon_corners = cart_to_unit_sphere_coords(corner_cart_positions[:, :, np.newaxis, :])[:, :, 0, :]

  gll_latlon, gll_to_cart_jacobian, cart_to_sphere_jacobian = eval_metric_terms_elem_local(latlon_corners,
                                                                                           npt,
                                                                                           rotate=rotate)

  return metric_terms_to_grid(gll_latlon,
                              gll_to_cart_jacobian,
                              cart_to_sphere_jacobian,
                              cube_redundancy,
                              npt,
                              calc_smooth_tensor=calc_smooth_tensor,
                              wrapped=wrapped)

