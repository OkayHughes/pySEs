import numpy as np
from .._config import get_backend as _get_backend
from .mesh import (init_element_corner_vert_redundancy,
                   init_spectral_grid_redundancy,
                   mesh_to_cart_bilinear,
                   metric_terms_to_grid)
from .equiangular_metric import eval_metric_terms_equiangular
from .cubed_sphere import init_cube_topo
from .spherical_coord_utils import (unit_sphere_to_cart_coords,
                                    cart_to_unit_sphere_coords_jacobian,
                                    cart_to_unit_sphere_coords)
_be = _get_backend()
use_wrapper = _be.use_wrapper


def eval_metric_terms_elem_local(latlon_corners,
                                 npt,
                                 rotate=False):
  """
  Compute metric terms for a general unstructured spherical element mesh
  using a bilinear mapping from element corners.

  Projects Cartesian element corners onto the unit sphere, bilinearly maps
  them to GLL nodes, then chains the reference-to-Cartesian and
  Cartesian-to-sphere Jacobians to produce the GLL grid positions and metric.

  Parameters
  ----------
  latlon_corners : Array[tuple[elem_idx, vert_idx, 2], Float]
      Spherical coordinates ``(lat, lon)`` (radians) of the four element
      corners in pyses vertex ordering.
  npt : int
      Number of GLL points per element edge.
  rotate : bool, optional
      If ``True``, apply a small rotation to the Cartesian element corners
      before projection (used to break symmetry for debugging).  Defaults
      to ``False``.

  Returns
  -------
  gll_latlon : Array[tuple[elem_idx, gll_idx, gll_idx, 2], Float]
      Spherical coordinates ``(lat, lon)`` of each GLL node.
  gll_to_cart_jacobian : Array[tuple[elem_idx, gll_idx, gll_idx, 3, 2], Float]
      Jacobian of the bilinear reference-to-Cartesian mapping.
  cart_to_sphere_jacobian : Array[tuple[elem_idx, gll_idx, gll_idx, 2, 3], Float]
      Jacobian of the Cartesian-to-sphere coordinate change.
  """
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
  """
  Build a quasi-uniform cubed-sphere spectral element grid using element-local bilinear metric.

  Generates the cubed-sphere topology for an ``nx``-element-per-face grid,
  computes element-corner latitudes/longitudes from the equiangular projection,
  then uses :func:`eval_metric_terms_elem_local` to produce the GLL metric.

  Parameters
  ----------
  nx : int
      Number of elements along one edge of a cubed-sphere face.
  npt : int
      Number of GLL points per element edge.
  wrapped : bool, optional
      If ``True``, arrays are moved onto the configured device after assembly.
      Defaults to ``use_wrapper``.
  calc_smooth_tensor : bool, optional
      If ``True``, smooth the hyperviscosity tensor after grid assembly.
      Defaults to ``False``.
  rotate : bool, optional
      If ``True``, apply a small rotation to break element symmetry.
      Defaults to ``True``.

  Returns
  -------
  grid : SpectralElementGrid
      Assembled spectral element grid struct.
  dims : frozendict[str, int]
      Grid dimension metadata ``{"N", "shape", "npt", "num_elem"}``.
  """
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
  """
  Build a stretched cubed-sphere spectral element grid using element-local metric.

  Applies an affine transformation ``x' = Q diag(s) x + c`` (followed by
  re-projection onto the unit sphere) to the equiangular cubed-sphere corners
  before computing the element-local bilinear metric.  This allows the grid
  to be concentrated toward a chosen region.

  Parameters
  ----------
  nx : int
      Number of elements along one edge of a cubed-sphere face.
  npt : int
      Number of GLL points per element edge.
  axis_dilation : Array[3] or None, optional
      Per-axis scaling factors ``s`` for the affine map.  Defaults to ones
      (no stretching).
  orthogonal_transform : Array[3, 3] or None, optional
      Orthogonal rotation matrix ``Q`` applied before axis dilation.
      Defaults to the identity.
  offset : Array[3] or None, optional
      Translation vector ``c`` added after dilation.  Must satisfy
      ``‖Q^T s^{-1} c‖ < 1`` so the mapping is bijective.  Defaults to
      zero.
  wrapped : bool, optional
      If ``True``, arrays are moved onto the configured device after assembly.
      Defaults to ``use_wrapper``.
  calc_smooth_tensor : bool, optional
      If ``True``, smooth the hyperviscosity tensor after grid assembly.
      Defaults to ``False``.
  rotate : bool, optional
      If ``True``, apply a small rotation to break element symmetry.
      Defaults to ``True``.

  Returns
  -------
  grid : SpectralElementGrid
      Assembled spectral element grid struct.
  dims : frozendict[str, int]
      Grid dimension metadata ``{"N", "shape", "npt", "num_elem"}``.
  """
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
  """
  Build a spectral element grid from an arbitrary unstructured mesh.

  Accepts pre-computed element corner Cartesian positions and a
  face-connectivity array, projects the corners onto the unit sphere,
  then uses :func:`eval_metric_terms_elem_local` to produce the GLL metric.

  Parameters
  ----------
  face_connectivity : Array[tuple[elem_idx, edge_idx, 3], Int]
      Topological connectivity array; each entry contains
      ``(remote_elem_idx, remote_edge_idx, same_direction)`` for the
      element edge at ``(elem_idx, edge_idx)``.
  corner_cart_positions : Array[tuple[elem_idx, vert_idx, 3], Float]
      Cartesian positions of the four element corners in pyses vertex
      ordering.  Will be projected onto the unit sphere internally.
  npt : int
      Number of GLL points per element edge.
  wrapped : bool, optional
      If ``True``, arrays are moved onto the configured device after
      assembly.  Defaults to ``use_wrapper``.
  calc_smooth_tensor : bool, optional
      If ``True``, smooth the hyperviscosity tensor after grid assembly.
      Defaults to ``False``.
  rotate : bool, optional
      If ``True``, apply a small rotation to break element symmetry.
      Defaults to ``False``.

  Returns
  -------
  grid : SpectralElementGrid
      Assembled spectral element grid struct.
  dims : frozendict[str, int]
      Grid dimension metadata ``{"N", "shape", "npt", "num_elem"}``.
  """
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
