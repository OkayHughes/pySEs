from ..config import np


def unit_sphere_to_cart_coords(latlon):
  """
  Convert spherical (lat, lon) coordinates to unit-sphere Cartesian coordinates.

  Parameters
  ----------
  latlon : Array[..., 2]
      Array whose last axis contains ``(latitude, longitude)`` in radians.

  Returns
  -------
  cart : Array[..., 3]
      Unit-sphere Cartesian coordinates ``(x, y, z)``.
  """
  lat = np.take(latlon, 0, axis=-1)
  lon = np.take(latlon, 1, axis=-1)
  cos_lat = np.cos(lat)
  cart = np.stack((cos_lat * np.cos(lon),
                   cos_lat * np.sin(lon),
                   np.sin(lat)), axis=-1)
  return cart


def cart_to_unit_sphere_coords(xyz):
  """
  Convert unit-sphere Cartesian coordinates to spherical (lat, lon).

  Parameters
  ----------
  xyz : Array[tuple[elem_idx, gll_idx, gll_idx, 3], Float]
      Unit-sphere Cartesian coordinates ``(x, y, z)``.

  Returns
  -------
  latlon : Array[tuple[elem_idx, gll_idx, gll_idx, 2], Float]
      Spherical coordinates ``(latitude, longitude)`` in radians,
      with longitude in ``[0, 2π)``.
  """
  latlon = np.stack((np.asin(xyz[:, :, :, 2]),
                     np.mod(np.atan2(xyz[:, :, :, 1],
                                     xyz[:, :, :, 0]) + 2 * np.pi,
                            2 * np.pi)), axis=-1)
  return latlon


def unit_sphere_to_cart_coords_jacobian(latlon):
  """
  Compute the Jacobian mapping spherical velocity components to Cartesian.

  Returns ``∂(x,y,z) / ∂(lat,lon)`` at each GLL node, i.e. the
  3×2 matrix that converts a physical ``(u_lat, u_lon)`` wind vector
  to the corresponding 3-D Cartesian velocity.

  Parameters
  ----------
  latlon : Array[tuple[elem_idx, gll_idx, gll_idx, 2], Float]
      Spherical coordinates ``(latitude, longitude)`` in radians.

  Returns
  -------
  jacobian : Array[tuple[elem_idx, gll_idx, gll_idx, 3, 2], Float]
      Jacobian ``∂(x,y,z)/∂(lat,lon)`` at each node.
  """
  lat = latlon[:, :, :, 0]
  lon = latlon[:, :, :, 1]
  unit_sphere_to_sph_coords_jacobian = np.zeros((*lat.shape[:3], 3, 2))
  unit_sphere_to_sph_coords_jacobian[:, :, :, 0, 0] = -np.sin(lat) * np.cos(lon)
  unit_sphere_to_sph_coords_jacobian[:, :, :, 0, 1] = np.cos(lat) * -np.sin(lon)
  unit_sphere_to_sph_coords_jacobian[:, :, :, 1, 0] = -np.sin(lat) * np.sin(lon)
  unit_sphere_to_sph_coords_jacobian[:, :, :, 1, 1] = np.cos(lat) * np.cos(lon)
  unit_sphere_to_sph_coords_jacobian[:, :, :, 2, 0] = np.cos(lat)
  unit_sphere_to_sph_coords_jacobian[:, :, :, 2, 1] = 0.0
  return unit_sphere_to_sph_coords_jacobian


def cart_to_unit_sphere_coords_jacobian(xyz):
  """
  Compute the Jacobian mapping Cartesian velocity components to spherical.

  Returns ``∂(lat,lon) / ∂(x,y,z)`` at each GLL node, i.e. the
  2×3 matrix that converts a 3-D Cartesian velocity to the corresponding
  physical ``(u_lat, u_lon)`` wind components.

  Parameters
  ----------
  xyz : Array[tuple[elem_idx, gll_idx, gll_idx, 3], Float]
      Unit-sphere Cartesian coordinates ``(x, y, z)``.

  Returns
  -------
  jacobian : Array[tuple[elem_idx, gll_idx, gll_idx, 2, 3], Float]
      Jacobian ``∂(lat,lon)/∂(x,y,z)`` at each node.
  """
  x = xyz[:, :, :, 0]
  y = xyz[:, :, :, 1]
  z = xyz[:, :, :, 2]
  normsq_2d = x**2 + y**2
  unit_sphere_to_sph_coords_jacobian = np.zeros((*x.shape[:3], 2, 3))
  unit_sphere_to_sph_coords_jacobian[:, :, :, 0, 2] = 1.0 / np.sqrt(1 - z**2)
  unit_sphere_to_sph_coords_jacobian[:, :, :, 1, 0] = -y / normsq_2d
  unit_sphere_to_sph_coords_jacobian[:, :, :, 1, 1] = x / normsq_2d
  return unit_sphere_to_sph_coords_jacobian
