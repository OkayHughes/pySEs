"""Galewsky, Scott & Polvani (2004) barotropic-instability shallow-water test.

This is the canonical analytic initialisation for the *3-D shallow-water*
dynamical core (subsuming the old standalone shallow-water model).  The
zonal jet is centred between ``phi0 = pi/7`` and ``phi1 = pi/2 - phi0``;
its strength is normalised to ``u_max`` at the jet axis.  The free-surface
height is recovered by quadrature of the gradient-wind balance
``g dh/dphi = -a u (f + tan(phi) u / a)``, then a Gaussian bump
perturbation triggers the barotropic instability.

Reference
---------
J. Galewsky, R. K. Scott, L. M. Polvani,
*An initial-value problem for testing numerical models of the global
shallow-water equations*, Tellus A **56** (2004), 429-440.
"""
import numpy as np

from .._config import get_backend as _get_backend
from ..dynamical_cores.initialization import init_model_shallow_water

_be = _get_backend()
device_wrapper = _be.array
jnp = _be.np


def init_galewsky_config(model_config):
  """Build the Galewsky test-case parameter dict.

  Parameters
  ----------
  model_config : dict[str, Any]
      Physics configuration containing ``radius_earth``,
      ``angular_freq_earth`` and ``gravity``.

  Returns
  -------
  config : dict[str, Any]
      Test-case configuration consumed by the other ``eval_galewsky_*``
      and by :func:`init_galewsky_state`.
  """
  config = {}
  config["deg"] = 100
  pts, weights = device_wrapper(np.polynomial.legendre.leggauss(config["deg"]))
  # Map the Gauss-Legendre nodes from [-1, 1] to [0, 1] (the per-column
  # quadrature window used in the height integral below).
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


def eval_galewsky_u(lat, config):
  """Zonal wind speed of the Galewsky jet.

  ``u(phi) = (u_max / e_norm) * exp(1 / ((phi - phi0)(phi - phi1)))``
  in the jet window ``phi0 < phi < phi1`` and zero elsewhere.
  """
  u = jnp.zeros_like(lat)
  mask = jnp.logical_and(lat > config["phi0"], lat < config["phi1"])
  u = jnp.where(
      mask,
      config["u_max"] / config["e_norm"] *
      jnp.exp(1 / ((lat - config["phi0"]) * (lat - config["phi1"]))),
      u)
  return u


def eval_galewsky_wind(lat, lon, config):
  """Horizontal wind vector ``(u, 0)`` from the Galewsky zonal jet."""
  u = jnp.stack((eval_galewsky_u(lat, config),
                 jnp.zeros_like(lat)), axis=-1)
  return u


def eval_galewsky_h(lat, lon, config):
  """Free-surface height with the unstable Gaussian bump.

  The balanced part of ``h`` is obtained from the gradient-wind balance
  by Gauss-Legendre quadrature on the per-column window
  ``[-pi/2, lat]``.  A small zonally localised Gaussian bump
  ``hat_h * cos(lat) * exp(-(lon / pert_alpha)^2)
  * exp(-((pert_center - lat) / pert_beta)^2)``
  is added to seed the instability.
  """
  quad_amount = lat + jnp.pi / 2.0
  weights_quad = (quad_amount.reshape([*lat.shape, 1])
                  * config["weights"].reshape((*[1 for _ in lat.shape],
                                               config["deg"])))
  phi_quad = (quad_amount.reshape([*lat.shape, 1])
              * config["pts"].reshape((*[1 for _ in lat.shape], config["deg"]))
              - np.pi / 2)
  u_quad = eval_galewsky_u(phi_quad, config)
  f = 2.0 * config["angular_freq_earth"] * jnp.sin(phi_quad)
  integrand = (config["radius_earth"] * u_quad
               * (f + jnp.tan(phi_quad) / config["radius_earth"] * u_quad))
  h = config["h0"] - 1.0 / config["gravity"] * jnp.sum(integrand * weights_quad, axis=-1)
  h_prime = (config["hat_h"] * jnp.cos(lat)
             * jnp.exp(-(lon / config["pert_alpha"])**2)
             * jnp.exp(-((config["pert_center"] - lat) / config["pert_beta"])**2))
  return h + h_prime


def eval_galewsky_hs(lat, lon, config):
  """Surface topography — identically zero for Galewsky."""
  return jnp.zeros_like(lat)


def init_galewsky_state(h_grid, v_grid, model_config, test_config, dims, model):
  """Initialise the 3-D shallow-water model state for the Galewsky test.

  Wraps :func:`init_model_shallow_water` with the analytic surface +
  free-surface height and the horizontal wind function from the Galewsky
  jet.  The 3-D shallow-water core uses a single hybrid layer, so
  ``v_grid`` must satisfy ``len(hybrid_a_i) == 2`` (one interior layer
  between top and surface).

  Parameters
  ----------
  h_grid : `SpectralElementGrid`
      Horizontal spectral element grid struct.
  v_grid : `dict`
      Vertical grid struct containing hybrid coordinate coefficients.
  model_config : `dict`
      Model physics configuration dict.
  test_config : `dict`
      Test-case parameters from :func:`init_galewsky_config`.
  dims : frozendict[str, int]
      Grid dimension metadata.
  model : model_info.models
      Dynamical-core identifier (a shallow-water model).

  Returns
  -------
  model_state : model state struct
      Fully initialised model state for the Galewsky barotropic-instability
      test.
  """
  assert len(v_grid["hybrid_a_i"]) == 2, (
      "Galewsky init expects the single-layer 3-D shallow-water "
      "configuration; got len(hybrid_a_i) = "
      f"{len(v_grid['hybrid_a_i'])}.")

  def z_surf_total_height_func(lat, lon):
    return (eval_galewsky_hs(lat, lon, test_config),
            eval_galewsky_h(lat, lon, test_config))

  def u_func(lat, lon, v_grid):
    # Broadcast the analytic 2-D zonal wind to the (..., nlev, 2) interface
    # the 3-D core expects; the jet is layer-independent so we copy across
    # ``hybrid_a_m`` slots.
    u_2d = eval_galewsky_u(lat, test_config)
    ones_lev = jnp.ones_like(v_grid["hybrid_a_m"])[jnp.newaxis, jnp.newaxis,
                                                   jnp.newaxis, :]
    return u_2d[:, :, :, jnp.newaxis] * ones_lev

  def v_func(lat, lon, v_grid):
    return jnp.zeros_like(u_func(lat, lon, v_grid))

  return init_model_shallow_water(z_surf_total_height_func,
                                  u_func,
                                  v_func,
                                  h_grid,
                                  v_grid,
                                  model_config,
                                  dims,
                                  model)
