from ..config import jnp


def init_williamson_steady_config(model_config):
  """
  Build the parameter struct for Williamson shallow-water test case 2 (steady geostrophic flow).

  Parameters
  ----------
  model_config : dict[str, Any]
      Physics configuration containing ``radius_earth``, ``angular_freq_earth``,
      ``gravity``, and ``alpha`` (the flow angle in radians).

  Returns
  -------
  config : dict[str, Any]
      Test case configuration to be passed to the other ``eval_williamson_tc2_*``
      functions in this module.

  Notes
  -----
  The test case is described in Williamson et al. (1992),
  *J. Comput. Phys.*, 102, 211–224, Test Case 2.
  """
  config = {}
  config["u0"] = 2.0 * jnp.pi * model_config["radius_earth"] / (12.0 * 24.0 * 60.0 * 60.0)
  config["h0"] = 2.94e4 / model_config["gravity"]
  config["alpha"] = model_config["alpha"]
  config["gravity"] = model_config["gravity"]
  config["radius_earth"] = model_config["radius_earth"]
  config["angular_freq_earth"] = model_config["angular_freq_earth"]
  return config


def eval_williamson_tc2_u(lat,
                          lon,
                          config):
  """
  Evaluate the analytic wind field for Williamson test case 2.

  Parameters
  ----------
  lat : Array[tuple[elem_idx, gll_idx, gll_idx], Float]
      Latitude (radians).
  lon : Array[tuple[elem_idx, gll_idx, gll_idx], Float]
      Longitude (radians).
  config : dict[str, Any]
      Test case configuration from ``init_williamson_steady_config``.

  Returns
  -------
  wind : Array[tuple[elem_idx, gll_idx, gll_idx, 2], Float]
      Horizontal wind ``(u, v)`` in m s^-1.
  """
  wind = jnp.stack((config["u0"] * (jnp.cos(lat) * jnp.cos(config["alpha"]) +
                                    jnp.cos(lon) * jnp.sin(lat) * jnp.sin(config["alpha"])),
                    -config["u0"] * (jnp.sin(lon) * jnp.sin(config["alpha"]))), axis=-1)
  return wind


def eval_williamson_tc2_h(lat,
                          lon,
                          config):
  """
  Evaluate the analytic geopotential height for Williamson test case 2.

  Parameters
  ----------
  lat : Array[tuple[elem_idx, gll_idx, gll_idx], Float]
      Latitude (radians).
  lon : Array[tuple[elem_idx, gll_idx, gll_idx], Float]
      Longitude (radians).
  config : dict[str, Any]
      Test case configuration from ``init_williamson_steady_config``.

  Returns
  -------
  h : Array[tuple[elem_idx, gll_idx, gll_idx], Float]
      Free-surface height (m).
  """
  h = jnp.zeros_like(lat)
  h += config["h0"]
  second_factor = (-jnp.cos(lon) * jnp.cos(lat) * jnp.sin(config["alpha"]) +
                   jnp.sin(lat) * jnp.cos(config["alpha"]))**2
  h -= (config["radius_earth"] * config["angular_freq_earth"] *
        config["u0"] + config["u0"]**2 / 2.0) / config["gravity"] * second_factor
  return h


def eval_williamson_tc2_hs(lat,
                           lon,
                           config):
  """
  Evaluate the surface topography for Williamson test case 2 (identically zero).

  Parameters
  ----------
  lat : Array[tuple[elem_idx, gll_idx, gll_idx], Float]
      Latitude (radians).
  lon : Array[tuple[elem_idx, gll_idx, gll_idx], Float]
      Longitude (radians).
  config : dict[str, Any]
      Test case configuration from ``init_williamson_steady_config``.

  Returns
  -------
  hs : Array[tuple[elem_idx, gll_idx, gll_idx], Float]
      Surface topography (m). All zeros for this test case.
  """
  return jnp.zeros_like(lat)
