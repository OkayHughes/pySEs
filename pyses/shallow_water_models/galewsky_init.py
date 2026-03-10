import numpy as np
from .._config import get_backend as _get_backend
_be = _get_backend()
device_wrapper = _be.array
jnp = _be.np


def init_galewsky_config(model_config):
  """
  Build the parameter struct for the Galewsky barotropic instability test case.

  Parameters
  ----------
  model_config : dict[str, Any]
      Physics configuration containing ``radius_earth``, ``angular_freq_earth``,
      and ``gravity``.

  Returns
  -------
  config : dict[str, Any]
      Test case configuration to be passed to the other ``eval_galewsky_*``
      functions in this module.

  Notes
  -----
  The test case is described in Galewsky, Scott, and Polvani (2004),
  *Tellus A*, 56, 429–440.
  """
  config = {}
  config["deg"] = 100
  pts, weights = device_wrapper(np.polynomial.legendre.leggauss(config["deg"]))
  pts = (pts + 1.0) / 2.0
  weights /= 2.0
  config["pts"] = pts
  config["weights"] = weights
  config["u_max"] = 80
  config["phi0"] = np.pi / 7
  config["phi1"] = np.pi / 2 - config["phi0"]
  config["e_norm"] = np.exp(-4 / (config["phi1"] - config["phi0"])**2)
  config["radius_earth"] = model_config["radius_earth"]
  config["angular_freq_earth"] = model_config["angular_freq_earth"]
  config["h0"] = 1e4
  config["hat_h"] = 120.0
  config["pert_alpha"] = 1.0 / 3.0
  config["pert_beta"] = 1.0 / 15.0
  config["pert_center"] = np.pi / 4
  config["gravity"] = model_config["gravity"]
  return config


def eval_galewsky_u(lat,
                    config):
  """
  Evaluate the zonal wind speed for the Galewsky test case.

  Parameters
  ----------
  lat : Array[tuple[elem_idx, gll_idx, gll_idx], Float]
      Latitude (radians).
  config : dict[str, Any]
      Test case configuration from ``init_galewsky_config``.

  Returns
  -------
  u : Array[tuple[elem_idx, gll_idx, gll_idx], Float]
      Zonal wind speed (m s^-1). Non-zero only between ``phi0`` and ``phi1``.
  """
  u = jnp.zeros_like(lat)
  mask = jnp.logical_and(lat > config["phi0"], lat < config["phi1"])
  u = jnp.where(mask, config["u_max"] / config["e_norm"] *
                jnp.exp(1 / ((lat - config["phi0"]) * (lat - config["phi1"]))), u)
  return u


def eval_galewsky_wind(lat,
                       lon,
                       config):
  """
  Evaluate the horizontal wind vector for the Galewsky test case.

  Parameters
  ----------
  lat : Array[tuple[elem_idx, gll_idx, gll_idx], Float]
      Latitude (radians).
  lon : Array[tuple[elem_idx, gll_idx, gll_idx], Float]
      Longitude (radians).
  config : dict[str, Any]
      Test case configuration from ``init_galewsky_config``.

  Returns
  -------
  wind : Array[tuple[elem_idx, gll_idx, gll_idx, 2], Float]
      Horizontal wind ``(u, v)`` in m s^-1. Meridional wind is identically zero.
  """
  u = jnp.stack((eval_galewsky_u(lat, config),
                 jnp.zeros_like(lat)), axis=-1)
  return u


def eval_galewsky_h(lat,
                    lon,
                    config):
  """
  Evaluate the free-surface height for the Galewsky test case.

  The background height is computed by integrating the gradient-wind balance
  condition using Gauss-Legendre quadrature. A perturbation is then added
  to trigger the barotropic instability.

  Parameters
  ----------
  lat : Array[tuple[elem_idx, gll_idx, gll_idx], Float]
      Latitude (radians).
  lon : Array[tuple[elem_idx, gll_idx, gll_idx], Float]
      Longitude (radians).
  config : dict[str, Any]
      Test case configuration from ``init_galewsky_config``.

  Returns
  -------
  h : Array[tuple[elem_idx, gll_idx, gll_idx], Float]
      Free-surface height (m).
  """
  quad_amount = lat + jnp.pi / 2.0
  weights_quad = quad_amount.reshape([*lat.shape, 1]) * config["weights"].reshape((*[1 for _ in lat.shape],
                                                                                   config["deg"]))
  phi_quad = quad_amount.reshape([*lat.shape, 1]) * config["pts"].reshape((*[1 for _ in lat.shape],
                                                                           config["deg"])) - np.pi / 2
  u_quad = eval_galewsky_u(phi_quad, config)
  f = 2.0 * config["angular_freq_earth"] * jnp.sin(phi_quad)
  integrand = config["radius_earth"] * u_quad * (f + jnp.tan(phi_quad) / config["radius_earth"] * u_quad)
  h = config["h0"] - 1.0 / config["gravity"] * jnp.sum(integrand * weights_quad, axis=-1)
  h_prime = (config["hat_h"] * jnp.cos(lat) * jnp.exp(-(lon / config["pert_alpha"])**2) *
             jnp.exp(-((config["pert_center"] - lat) / config["pert_beta"])**2))
  return h + h_prime


def eval_galewsky_hs(lat,
                     lon,
                     config):
  """
  Evaluate the surface topography for the Galewsky test case (identically zero).

  Parameters
  ----------
  lat : Array[tuple[elem_idx, gll_idx, gll_idx], Float]
      Latitude (radians).
  lon : Array[tuple[elem_idx, gll_idx, gll_idx], Float]
      Longitude (radians).
  config : dict[str, Any]
      Test case configuration from ``init_galewsky_config``.

  Returns
  -------
  hs : Array[tuple[elem_idx, gll_idx, gll_idx], Float]
      Surface topography (m). All zeros for this test case.
  """
  return jnp.zeros_like(lat)
