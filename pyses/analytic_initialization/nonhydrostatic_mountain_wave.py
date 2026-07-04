"""DCMIP-2012 non-hydrostatic orographic gravity-wave test cases (2-1 / 2-2).

This module provides the analytic initialisation for the Schar-type mountain
wave experiments of Ullrich et al. (2012), *DCMIP Test Case Document*,
section 2.X (tests 2-1 and 2-2).  A non-rotating, reduced-size planet
(``a = aref / X`` with ``X = 500`` by default) carries a background zonal jet
over Schar-type orography.

The governing analytic relations are

* topography   ``zs = h0 exp(-(r/d)^2) cos^2(pi r / xi)``
* great circle ``r  = a arccos(sin phic sin phi
                                        + cos phic cos phi cos(lon - lonc))``
* temperature  ``T(phi) = Teq (1 - c ueq^2 / g sin^2 phi)``
* pressure   ``p = peq exp(- ueq^2 / (2 Rd Teq) sin^2 phi
                                       - g z / (Rd T(phi)))``
* zonal wind  ``u = ueq cos(phi)
                             sqrt(2 Teq / T(phi) c z + T(phi) / Teq)``

with ``v = w = 0`` and density from the ideal gas law.

Reference
---------
P. A. Ullrich, C. Jablonowski, J. Kent, P. H. Lauritzen, R. D. Nair,
M. A. Taylor, *Dynamical Core Model Intercomparison Project (DCMIP) Test Case
Document*, version 1.7 (2012), section 2.X.
"""
from .._config import get_backend as _get_backend
import numpy as np
from ..dynamical_cores.initialization import init_analytic_state
from ..dynamical_cores.model_info import deep_atmosphere_models
_be = _get_backend()
jnp = _be.np
device_wrapper = _be.array

# DCMIP-2012 default surface wind shear for the sheared variant (test 2-2),
# ``cs`` in Table XIII (units m^-1).  The non-sheared variant (test 2-1) uses
# ``shear = 0``.
dcmip_surface_shear = 2.5e-4


def init_mountain_wave_config(Teq=300.0,
                              ueq=20.0,
                              peq=1e5,
                              shear=0.0,
                              mountain_height=250.0,
                              mountain_half_width=5000.0,
                              mountain_wavelength=4000.0,
                              mountain_lat=0.0,
                              mountain_lon=np.pi / 4.0,
                              reduction_factor=500.0,
                              radius_earth=6371e3,
                              angular_freq_earth=0.0,
                              Rgas=287.0,
                              gravity=9.80616,
                              model_config=None):
  """
  Create a struct containing all parameters necessary to initialise the
  DCMIP-2012 non-hydrostatic Schar-type mountain-wave test cases (2-1 / 2-2).

  The atmosphere is column-isothermal with a latitude-dependent temperature
  ``T(phi) = Teq (1 - shear * ueq^2 / g * sin^2 phi)``, a
  hydrostatically balanced pressure field carrying a background zonal jet
  (eq. 80), and a vertically sheared zonal wind.  The topography is a
  Schar-type ridge.

  The planet is non-rotating and reduced in size by ``reduction_factor`` (the
  DCMIP small-planet factor ``X``), so the *scaled* radius used everywhere in
  this test is ``a = radius_earth / reduction_factor``.  The reduced radius is
  what appears in the great-circle distance to the mountain center and
  must match the ``radius_earth`` used by the dynamical core; pass the same
  reduced-planet ``model_config`` here to keep the two consistent.

  Parameters
  ----------
  Teq : float
    Reference surface temperature at the equator ``Teq`` (K).
  ueq : float
    Reference (equatorial surface) zonal wind speed ``ueq`` (m s^-1).
  peq : float
    Reference surface pressure at the equator ``peq`` (Pa).
  shear : float
    Surface wind-shear parameter ``c`` (m^-1).  ``shear = 0`` selects the
    non-sheared test 2-1; ``shear = dcmip_surface_shear`` selects the sheared
    test 2-2.
  mountain_height : float
    Maximum Schar-type mountain height ``h0`` (m).
  mountain_half_width : float
    Schar-type mountain half-width ``d`` (m); the Gaussian envelope scale.
  mountain_wavelength : float
    Schar-type mountain wavelength ``xi`` (m); the ``cos^2`` oscillation scale.
  mountain_lat : float
    Latitude of the mountain centre ``phic`` (radians).  Defaults to the
    equator.
  mountain_lon : float
    Longitude of the mountain centre ``lonc`` (radians).  Defaults to
    ``pi / 4``.
  reduction_factor : float
    Small-planet reduction factor ``X``; the scaled radius is
    ``radius_earth / reduction_factor``.
  radius_earth : float
    Unscaled (reference) radius of the planet surface ``aref`` (m); the scaled
    radius ``a = radius_earth / reduction_factor`` is stored in the returned
    config.
  angular_freq_earth : float
    Angular frequency of the planet (s^-1).  Defaults to zero: the DCMIP
    mountain-wave tests use a non-rotating planet.
  Rgas : float
    Gas constant of dry air (J kg^-1 K^-1).
  gravity : float
    Constant strength of gravity at the surface (m s^-2).
  model_config :
    physics_config struct for your dynamical core.  When supplied,
    ``radius_earth``, ``angular_freq_earth``, ``Rgas`` and ``gravity`` are read
    from it so the analytic state is consistent with the simulation
    configuration.  Pass the *reduced-planet* physics_config here (i.e. one
    whose ``radius_earth`` is already ``aref / X``); its radius is used as the
    scaled radius ``a`` directly and ``reduction_factor`` is ignored in that
    case.

  Notes
  -----
  The non-sheared limit (``shear = 0``) coincides with the isothermal,
  non-rotating solid-body-rotation base state of
  :func:`hydrostatic_solid_body.init_solid_body_config` (with ``lapse = 0``,
  ``u_max = ueq``): the latitudinal pressure factor
  ``-ueq^2 / (2 Rd Teq) sin^2 phi`` and the wind ``ueq cos(phi)`` are identical.
  The sheared variant departs from that base state through the
  latitude-dependent column temperature and the height-dependent wind.

  Returns
  -------
  dict[str, Any]
      test_config to be passed to the other functions in this module.
  """
  if model_config:
    # A reduced-planet physics_config already carries the scaled radius a; use
    # it directly rather than re-dividing by ``reduction_factor``.
    a = model_config["radius_earth"]
    angular_freq_earth = model_config["angular_freq_earth"]
    Rgas = model_config["Rgas"]
    gravity = model_config["gravity"]
  else:
    a = radius_earth / reduction_factor
  return {"Teq": device_wrapper(Teq),
          "ueq": device_wrapper(ueq),
          "peq": device_wrapper(peq),
          "shear": device_wrapper(shear),
          "mountain_height": device_wrapper(mountain_height),
          "mountain_half_width": device_wrapper(mountain_half_width),
          "mountain_wavelength": device_wrapper(mountain_wavelength),
          "mountain_lat": device_wrapper(mountain_lat),
          "mountain_lon": device_wrapper(mountain_lon),
          "radius_earth": device_wrapper(a),
          "angular_freq_earth": device_wrapper(angular_freq_earth),
          "Rgas": device_wrapper(Rgas),
          "gravity": device_wrapper(gravity),
          }


def _eval_r_hat(z,
                config,
                deep=False):
  if deep:
    r_hat = (z + config["radius_earth"]) / config["radius_earth"]
  else:
    r_hat = jnp.ones_like(z)
  return r_hat


def _eval_column_temperature(lat,
                             config):
  """Column-isothermal, latitude-dependent temperature ``T(phi)`` .

  ``T(phi) = Teq (1 - shear * ueq^2 / g * sin^2 phi)``.  For the non-sheared
  variant (``shear = 0``) this reduces to the constant ``Teq``.
  """
  Teq = config["Teq"]
  ueq = config["ueq"]
  g = config["gravity"]
  return Teq * (1.0 - config["shear"] * ueq * ueq / g * jnp.sin(lat)**2)


def eval_great_circle_dist(lat,
                           lon,
                           config):
  """
  Evaluate the physical great-circle distance (m) from the mountain centre.

  Parameters
  ----------
  lat : Array[Float, tuple[elem_idx, i_idx, j_idx]]
    Latitude (radians).
  lon : Array[Float, tuple[elem_idx, i_idx, j_idx]]
    Longitude (radians).
  config : TestConfig
    Dict-like containing parameters for the test case.

  Notes
  -----
  * See :func:`init_mountain_wave_config` for how to initialise ``config``.
  * The distance uses the *scaled* radius ``a = aref / X`` (eq. 77), so the
    Schar ridge occupies a physically fixed footprint on the reduced planet.

  Returns
  -------
  r : Array[Float, tuple[elem_idx, i_idx, j_idx]]
      Great-circle distance from ``(mountain_lat, mountain_lon)`` in metres.
  """
  lat_m = config["mountain_lat"]
  lon_m = config["mountain_lon"]
  cos_d = (jnp.sin(lat_m) * jnp.sin(lat) +
           jnp.cos(lat_m) * jnp.cos(lat) * jnp.cos(lon - lon_m))
  # Clip guards against floating-point overshoot just outside [-1, 1].
  return config["radius_earth"] * jnp.arccos(jnp.clip(cos_d, -1.0, 1.0))


def eval_z_surface(lat,
                   lon,
                   config,
                   mountain=True):
  """
  Evaluate surface height (m).

  Parameters
  ----------
  lat : Array[Float, tuple[elem_idx, i_idx, j_idx]]
    Latitude (radians).
  lon : Array[Float, tuple[elem_idx, i_idx, j_idx]]
    Longitude (radians).
  config : TestConfig
    Dict-like containing parameters for the test case.
  mountain : bool, default=True
    If ``True``, use the Schar-type ridge (eq. 76); if ``False``, return a flat
    surface (a useful steady-state control since the balanced state should then
    remain at rest apart from the background jet).

  Notes
  -----
  * See :func:`init_mountain_wave_config` for how to initialise ``config``.
  * The Schar-type profile ``zs = h0 exp(-(r/d)^2) cos^2(pi r / xi)`` (eq. 76)
    combines a broad Gaussian envelope (half-width ``d``) with a short-scale
    ``cos^2`` oscillation (wavelength ``xi``), giving a quasi-two-dimensional
    ridge with compact support.

  Returns
  -------
  z_surf : Array[Float, tuple[elem_idx, i_idx, j_idx]]
      Surface height (m).  Identically zero when ``mountain=False``.
  """
  if not mountain:
    return jnp.zeros_like(lat)
  r = eval_great_circle_dist(lat, lon, config)
  d = config["mountain_half_width"]
  xi = config["mountain_wavelength"]
  return (config["mountain_height"] * jnp.exp(-(r / d)**2) *
          jnp.cos(np.pi * r / xi)**2)


def eval_pressure_temperature(z,
                              lat,
                              config,
                              deep=False):
  """
  Evaluate pressure and (dry) temperature on a 3-D grid.

  Parameters
  ----------
  z : Array[Float, tuple[elem_idx, i_idx, j_idx, level_idx]]
    Geometric height above the model surface in metres.
  lat : Array[Float, tuple[elem_idx, i_idx, j_idx]]
      Latitude (radians).
  config : TestConfig
    Dict-like containing parameters for the test case.
  deep : bool, default=False
    If true, use the deep-atmosphere base state (``r_hat = r / a`` in the
    height-to-pressure relation).  The DCMIP mountain-wave test is a
    shallow-atmosphere test, so this is normally False.

  Notes
  -----
  * See :func:`init_mountain_wave_config` for how to initialise ``config``.
  * Temperature is column-isothermal but latitude-dependent; the
    latitudinal surface-pressure factor ``-ueq^2 / (2 Rd Teq) sin^2 phi``
    carries the balance with the background zonal jet.

  Returns
  -------
  pressure : Array[Float, tuple[elem_idx, i_idx, j_idx, level_idx]]
    Moist pressure (Pa); for this dry test case it equals the dry pressure.
  temperature : Array[Float, tuple[elem_idx, i_idx, j_idx, level_idx]]
    Temperature (not virtual), in Kelvin.
  """
  Rd = config["Rgas"]
  g = config["gravity"]
  Teq = config["Teq"]
  ueq = config["ueq"]
  sin_lat = jnp.sin(lat)[:, :, :, np.newaxis]
  # Column temperature T(phi); broadcast to all levels (isothermal in z).
  temperature = _eval_column_temperature(lat, config)[:, :, :, np.newaxis] * jnp.ones_like(z)
  r_hat = _eval_r_hat(z, config, deep=deep)
  z_eff = r_hat * z
  log_p_lat = -ueq * ueq / (2.0 * Rd * Teq) * sin_lat**2
  log_p_vert = -g * z_eff / (Rd * temperature)
  pressure = config["peq"] * jnp.exp(log_p_lat + log_p_vert)
  return pressure, temperature


def eval_surface_state(lat,
                       lon,
                       config,
                       deep=False,
                       mountain=True):
  """
  Evaluate the surface height and (moist == dry) surface pressure.

  Parameters
  ----------
  lat : Array[Float, tuple[elem_idx, i_idx, j_idx]]
      Latitude (radians).
  lon : Array[Float, tuple[elem_idx, i_idx, j_idx]]
      Longitude (radians).
  config : TestConfig
    Dict-like containing parameters for the test case.
  deep : bool, default=False
    If true, use the deep-atmosphere base state.
  mountain : bool, default=True
    If ``True``, use the Schar-type ridge; see :func:`eval_z_surface`.

  Notes
  -----
  * See :func:`init_mountain_wave_config` for how to initialise ``config``.

  Returns
  -------
  z_surface : Array[Float, tuple[elem_idx, i_idx, j_idx]]
    Surface height in metres.  Identically zero when ``mountain=False``.
  p_surface : Array[Float, tuple[elem_idx, i_idx, j_idx]]
    Surface pressure in Pascal evaluated at ``z_surface``.
  """
  z_surface = eval_z_surface(lat, lon, config, mountain=mountain)
  p_surface = eval_pressure_temperature(z_surface[:, :, :, np.newaxis],
                                        lat,
                                        config,
                                        deep=deep)[0][:, :, :, 0]
  return z_surface, p_surface


def eval_state(lat,
               lon,
               z,
               config,
               deep=False):
  """
  Calculate zonal wind, meridional wind, pressure, temperature, and moisture.

  Parameters
  ----------
  lat : Array[Float, tuple[elem_idx, i_idx, j_idx]]
      Latitude (radians).
  lon : Array[Float, tuple[elem_idx, i_idx, j_idx]]
      Longitude (radians).
  z : Array[Float, tuple[elem_idx, i_idx, j_idx, level_idx]]
    Geometric height above the model surface in metres.
  config : TestConfig
    Dict-like containing parameters for the test case.
  deep : bool
    If true, use the deep-atmosphere base state for pressure/temperature.

  Notes
  -----
  * See :func:`init_mountain_wave_config` for how to initialise ``config``.
  * This is a dry test case: ``q_vapor`` is identically zero, so temperature
    and virtual temperature coincide.
  * The zonal wind is vertically sheared (eq. 82),
    ``u = ueq cos(phi) sqrt(2 Teq / T(phi) * shear * z + T(phi) / Teq)``.  For
    the non-sheared variant (``shear = 0``, ``T = Teq``) this reduces to the
    height-independent ``u = ueq cos(phi)``.

  Returns
  -------
  u : Array[Float, tuple[elem_idx, i_idx, j_idx, level_idx]]
    Zonal wind (m s^-1).
  v : Array[Float, tuple[elem_idx, i_idx, j_idx, level_idx]]
    Meridional wind (m s^-1); identically zero.
  pressure : Array[Float, tuple[elem_idx, i_idx, j_idx, level_idx]]
    Moist pressure (Pa); equal to dry pressure for this test.
  temperature : Array[Float, tuple[elem_idx, i_idx, j_idx, level_idx]]
    Virtual temperature (K); equal to dry temperature for this test.
  q : Array[Float, tuple[elem_idx, i_idx, j_idx, level_idx]]
    Specific humidity (kg kg^-1), identically zero.
  """
  pressure, temperature = eval_pressure_temperature(z, lat, config, deep=deep)
  Teq = config["Teq"]
  T_phi = _eval_column_temperature(lat, config)[:, :, :, np.newaxis]
  cos_lat = jnp.cos(lat)[:, :, :, np.newaxis]
  shear_term = 2.0 * Teq / T_phi * config["shear"] * z
  u = config["ueq"] * cos_lat * jnp.sqrt(shear_term + T_phi / Teq)
  v = jnp.zeros_like(u)
  q_vapor = jnp.zeros_like(z)
  return u, v, pressure, temperature, q_vapor


def init_mountain_wave_state(h_grid,
                             v_grid,
                             model_config,
                             test_config,
                             dims,
                             model,
                             mountain=True,
                             eps=1e-6,
                             enforce_hydrostatic=False):
  """
  Initialise model state for the DCMIP-2012 non-hydrostatic mountain-wave test.

  Wraps :func:`init_analytic_state` with the analytic surface state,
  pressure / temperature profile, and (sheared) wind functions of the
  Schar-type mountain-wave atmosphere.

  Parameters
  ----------
  h_grid : `SpectralElementGrid`
      Horizontal spectral element grid struct.
  v_grid : `dict`
      Vertical grid struct containing hybrid coordinate coefficients.
  model_config : `dict`
      Model physics configuration dict.  For a faithful DCMIP setup this should
      be a reduced-planet, non-rotating config (``radius_earth = aref / X`` and
      ``angular_freq_earth = 0``).
  test_config : `dict`
      Test-case parameters from :func:`init_mountain_wave_config`.
  dims : frozendict[str, int]
      Grid dimension metadata.
  model : model_info.models
      Dynamical core identifier (from ``model_info.models``).
  mountain : `bool`, default=True
      If ``True``, add the Schar-type ridge configured in ``test_config``.  The
      balanced background jet is exactly steady when ``mountain=False``; the
      ridge is what triggers the gravity-wave response.
  eps : `float`, default=1e-6
      Convergence tolerance passed to the height-inversion routines.
  enforce_hydrostatic : `bool`, default=False
      If ``True``, overwrite the initial interface geopotential with the
      hydrostatically balanced value (HOMME non-hydrostatic only).

  Returns
  -------
  model_state : model state struct
      Fully initialised model state struct for the mountain-wave test.
  """
  lat = h_grid["physical_coords"][:, :, :, 0]
  deep = model in deep_atmosphere_models

  def z_pi_surf_func(lat, lon):
    return eval_surface_state(lat, lon, test_config, deep=deep, mountain=mountain)

  def Q_func(lat, lon, z):
    return eval_state(lat, lon, z, test_config, deep=deep)[4]

  def p_func(z):
    return eval_pressure_temperature(z, lat, test_config, deep=deep)[0]

  def u_func(lat, lon, z):
    return eval_state(lat, lon, z, test_config, deep=deep)[0]

  def v_func(lat, lon, z):
    return eval_state(lat, lon, z, test_config, deep=deep)[1]

  def Tv_func(lat, lon, z):
    return eval_state(lat, lon, z, test_config, deep=deep)[3]

  def w_func(lat, lon, z):
    return jnp.zeros_like(z)

  model_state = init_analytic_state(z_pi_surf_func,
                                    p_func,
                                    Tv_func,
                                    u_func,
                                    v_func,
                                    Q_func,
                                    h_grid, v_grid,
                                    model_config,
                                    dims,
                                    model,
                                    w_func=w_func,
                                    eps=eps,
                                    enforce_hydrostatic=enforce_hydrostatic)
  return model_state
