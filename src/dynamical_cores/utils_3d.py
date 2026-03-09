import numpy as np
from .._config import get_backend as _get_backend
_be = _get_backend()
jnp = _be.np
jit = _be.jit
flip = _be.flip
from functools import partial
from .model_info import deep_atmosphere_models


@jit
def midlevel_to_interface_vel(field_model,
                              d_mass,
                              d_mass_int):
  """
  Interpolate a vector field from model mid-levels to interfaces using mass-weighted averaging.

  The two surface values (top and bottom) are copied unchanged; interior
  interfaces use a layer-mass-weighted average of the two adjacent mid-levels.

  Parameters
  ----------
  field_model : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx, 2], Float]
      Vector field on model mid-levels.
  d_mass : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Layer thickness on mid-levels.
  d_mass_int : Array[tuple[elem_idx, gll_idx, gll_idx, nlev+1], Float]
      Layer thickness on interface levels.

  Returns
  -------
  field_interface : Array[tuple[elem_idx, gll_idx, gll_idx, nlev+1, 2], Float]
      Vector field on interface levels.
  """
  scaled_sum = (d_mass[:, :, :, :-1, np.newaxis] * field_model[:, :, :, :-1, :] +
                d_mass[:, :, :, 1:, np.newaxis] * field_model[:, :, :, 1:, :])
  mid_levels = scaled_sum / (2.0 * d_mass_int[:, :, :, 1:-1, np.newaxis])
  return jnp.concatenate((field_model[:, :, :, 0:1, :],
                          mid_levels,
                          field_model[:, :, :, -1:, :]), axis=-2)


@jit
def midlevel_to_interface(field_model):
  """
  Linearly interpolate a scalar field from model mid-levels to interfaces.

  The two end values are copied unchanged; interior interfaces are the
  arithmetic mean of their neighbouring mid-level values.

  Parameters
  ----------
  field_model : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Scalar field on model mid-levels.

  Returns
  -------
  field_interface : Array[tuple[elem_idx, gll_idx, gll_idx, nlev+1], Float]
      Scalar field on interface levels.
  """
  mid_levels = (field_model[:, :, :, :-1] + field_model[:, :, :, 1:]) / 2.0
  return jnp.concatenate((field_model[:, :, :, 0:1],
                          mid_levels,
                          field_model[:, :, :, -1:]), axis=-1)


@jit
def interface_to_midlevel(field_interface):
  """
  Average a scalar field from interface levels to model mid-levels.

  Parameters
  ----------
  field_interface : Array[tuple[elem_idx, gll_idx, gll_idx, nlev+1], Float]
      Scalar field on interface levels.

  Returns
  -------
  field_model : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Scalar field on model mid-levels (arithmetic mean of bounding interfaces).
  """
  return (field_interface[:, :, :, 1:] +
          field_interface[:, :, :, :-1]) / 2.0


@jit
def interface_to_midlevel_vec(vec_interface):
  """
  Average a vector field from interface levels to model mid-levels.

  Parameters
  ----------
  vec_interface : Array[tuple[elem_idx, gll_idx, gll_idx, nlev+1, 2], Float]
      Vector field on interface levels.

  Returns
  -------
  vec_model : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx, 2], Float]
      Vector field on model mid-levels (arithmetic mean of bounding interfaces).
  """
  return (vec_interface[:, :, :, 1:, :] +
          vec_interface[:, :, :, :-1, :]) / 2.0


@jit
def interface_to_delta(field_interface):
  """
  Compute layer differences from an interface-level field.

  Parameters
  ----------
  field_interface : Array[tuple[elem_idx, gll_idx, gll_idx, nlev+1], Float]
      Scalar field on interface levels (e.g. geopotential or pressure).

  Returns
  -------
  delta : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Difference ``field[k+1] - field[k]`` for each mid-level ``k``.
  """
  return field_interface[:, :, :, 1:] - field_interface[:, :, :, :-1]


@jit
def cumulative_sum(dfield_model,
                   val_surf_top):
  """
  Reconstruct interface values from mid-level differences by downward cumulative summation.

  Integrates from the model top down, appending ``val_surf_top`` as the
  bottom (surface) interface value.

  Parameters
  ----------
  dfield_model : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Layer differences on mid-levels (e.g. ``d_phi`` or ``d_p``).
  val_surf_top : Array[tuple[elem_idx, gll_idx, gll_idx], Float]
      Value at the model-top interface (added as an offset after summation).

  Returns
  -------
  field_interface : Array[tuple[elem_idx, gll_idx, gll_idx, nlev+1], Float]
      Reconstructed field on interface levels.
  """
  return jnp.concatenate((flip(jnp.cumsum(flip(dfield_model, -1), axis=-1), -1) +
                          val_surf_top[:, :, :, np.newaxis],
                          val_surf_top[:, :, :, np.newaxis]), axis=-1)


@partial(jit, static_argnames=["model"])
def phi_to_z(phi,
             config,
             model):
  """
  Convert geopotential to geometric height.

  For shallow-atmosphere models ``z = phi / g``.  For deep-atmosphere
  models the full spherical relation is used.

  Parameters
  ----------
  phi : Array[..., Float]
      Geopotential (m^2 s^-2).
  config : dict[str, Any]
      Physics config with ``gravity`` and ``radius_earth``.
  model : model_info.models
      Model identifier; deep-atmosphere correction is applied if
      ``model in deep_atmosphere_models``.

  Returns
  -------
  z : Array[..., Float]
      Geometric height above the surface (m).
  """
  gravity = config["gravity"]
  radius_earth = config["radius_earth"]
  if model in deep_atmosphere_models:
    b = (2 * phi * radius_earth - gravity * radius_earth**2)
    z = -2 * phi * radius_earth**2 / (b - jnp.sqrt(b**2 - 4 * phi**2 * radius_earth**2))
  else:
    z = phi / gravity
  return z


@partial(jit, static_argnames=["model"])
def z_to_g(z,
           config,
           model):
  """
  Compute local gravitational acceleration from geometric height.

  For shallow-atmosphere models returns the constant ``gravity``.  For
  deep-atmosphere models applies the inverse-square law.

  Parameters
  ----------
  z : Array[..., Float]
      Geometric height above the surface (m).
  config : dict[str, Any]
      Physics config with ``gravity`` and ``radius_earth``.
  model : model_info.models
      Model identifier; depth correction is applied if
      ``model in deep_atmosphere_models``.

  Returns
  -------
  g : Array[..., Float]
      Local gravitational acceleration (m s^-2).
  """
  radius_earth = config["radius_earth"]
  if model in deep_atmosphere_models:
    g = config["gravity"] * (radius_earth /
                             (z + radius_earth))**2
  else:
    g = config["gravity"]
  return g


@partial(jit, static_argnames=["model"])
def phi_to_g(phi,
             config,
             model):
  """
  Compute local gravitational acceleration from geopotential.

  Convenience wrapper that chains ``phi_to_z`` and ``z_to_g``.

  Parameters
  ----------
  phi : Array[..., Float]
      Geopotential (m^2 s^-2).
  config : dict[str, Any]
      Physics config with ``gravity`` and ``radius_earth``.
  model : model_info.models
      Model identifier forwarded to ``phi_to_z`` and ``z_to_g``.

  Returns
  -------
  g : Array[..., Float]
      Local gravitational acceleration (m s^-2).
  """
  z = phi_to_z(phi, config, model)
  return z_to_g(z, config, model)


@partial(jit, static_argnames=["model"])
def phi_to_r_hat(phi,
                 config,
                 model):
  """
  Compute the non-dimensional radial scaling factor from geopotential.

  ``r_hat = (z + a) / a`` where ``a`` is ``radius_earth`` and ``z`` is
  the geometric height derived from ``phi``.  For shallow-atmosphere
  models returns an array of ones.

  Parameters
  ----------
  phi : Array[..., Float]
      Geopotential (m^2 s^-2).
  config : dict[str, Any]
      Physics config with ``gravity`` and ``radius_earth``.
  model : model_info.models
      Model identifier; deep-atmosphere scaling is applied if
      ``model in deep_atmosphere_models``.

  Returns
  -------
  r_hat : Array[..., Float]
      Non-dimensional radial scaling factor (dimensionless).
  """
  radius_earth = config["radius_earth"]
  if model in deep_atmosphere_models:
    r_hat = (phi_to_z(phi, config, model) + radius_earth) / radius_earth
  else:
    r_hat = jnp.ones_like(phi)
  return r_hat


@jit
def physical_dot_product(u, v):
  """
  Compute the element-wise dot product of two physical-space 2-component vector fields.

  Parameters
  ----------
  u : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx, 2], Float]
      First vector field.
  v : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx, 2], Float]
      Second vector field (same shape as ``u``).

  Returns
  -------
  dot : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Scalar dot product ``u[..., 0]*v[..., 0] + u[..., 1]*v[..., 1]``.
  """
  return (u[:, :, :, :, 0] * v[:, :, :, :, 0] +
          u[:, :, :, :, 1] * v[:, :, :, :, 1])
