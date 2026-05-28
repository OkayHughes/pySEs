from .._config import get_backend as _get_backend
import numpy as np
from ..dynamical_cores.initialization import init_model_pressure
from ..dynamical_cores.model_info import deep_atmosphere_models
_be = _get_backend()
jnp = _be.np
device_wrapper = _be.array


def init_solid_body_config(T0=300.0,
                           lapse=0.005,
                           u_max=0.0,
                           p0=1e5,
                           mountain_height=2000.0,
                           mountain_width=np.deg2rad(20.0),
                           mountain_lat=0.0,
                           mountain_lon=np.pi / 2.0,
                           radius_earth=6371e3,
                           angular_freq_earth=7.292e-5,
                           Rgas=287.0,
                           gravity=9.81,
                           model_config=None):
  """
  Create a struct containing all parameters necessary to initialise the
  hydrostatic solid body rotation test case.

  The atmosphere has a constant tropospheric lapse rate
  ``T(z) = T0 - lapse * z`` and (optionally) zonal solid body rotation
  ``u(lat) = u_max * cos(lat)``.  Surface pressure varies with latitude so
  that the flow is in geostrophic-cyclostrophic balance under the
  isothermal-limit Staniforth & White (2008) form,

      p_s(lat) = p0 * exp(- (u_max^2 + 2 * Omega * a * u_max)
                          * sin^2(lat) / (2 * Rgas * T0)).

  For ``u_max = 0`` the state reduces to a horizontally uniform,
  hydrostatic, constant-lapse atmosphere at rest -- the standard
  "dry hydrostatic" steady state used as a base for advection and
  initialisation tests.  For ``u_max > 0`` the balance is exact in the
  isothermal limit ``lapse -> 0`` and accurate to leading order in
  ``lapse * z / T0`` otherwise.

  Parameters
  ----------
  T0 : float
    Reference surface temperature (K).
  lapse : float
    Atmospheric lapse rate (K m^-1); ``T(z) = T0 - lapse * z``.
  u_max : float
    Maximum zonal wind amplitude (m s^-1); ``u(lat) = u_max * cos(lat)``.
    Set to zero for an atmosphere at rest.
  p0 : float
    Reference equatorial surface pressure (Pa).
  mountain_height : float
    Height of the optional radially symmetric Gaussian mountain (m); used
    only when ``init_solid_body_state`` (and downstream helpers) are called
    with ``mountain=True``.  Set to zero to disable.
  mountain_width : float
    Width of the mountain (radians); the half-width-half-tenth-maximum
    great-circle distance.  ``exp(-(d / sigma)^2) = 0.1`` at
    ``d = mountain_width / 2``, i.e.
    ``sigma = mountain_width / (2 * sqrt(ln 10))``.
  mountain_lat : float
    Latitude of the mountain centre (radians).  Defaults to the equator.
  mountain_lon : float
    Longitude of the mountain centre (radians).  Defaults to ``pi / 2``
    (90 deg E).
  radius_earth : float
    Nominal radius of the planet surface (m), `a` or `rearth` in most ESMs
  angular_freq_earth : float
    Angular frequency of the earth (1 / sec), `omega` in most ESMs
  Rgas : float
    Nominal gas constant of dry air (J kg^-1 K^-1)
  gravity : float
    Nominal constant strength of gravity at the surface (m s^-2)
  model_config :
    physics_config struct for your dynamical core. This allows
    other coefficients like Rgas to be read directly from your
    simulation configuration.

  Notes
  -----
  The constant-lapse rotating base state generalises the analytic
  isothermal solution of Staniforth & White (2008),
  *Q. J. Roy. Meteorol. Soc.* 134, 621-629, and is widely used as the
  base atmosphere for DCMIP advection tests (e.g. Ullrich et al. (2012),
  *DCMIP test case document*).

  Returns
  -------
  dict[str, Any]
      test_config to be passed to other functions in this module.
  """
  if model_config:
    radius_earth = model_config["radius_earth"]
    angular_freq_earth = model_config["angular_freq_earth"]
    Rgas = model_config["Rgas"]
    gravity = model_config["gravity"]
  # ``mountain_scale`` is the e-folding scale used inside the Gaussian.
  # The width convention matches moist_baroclinic_wave: ``mountain_width``
  # is the great-circle distance between the points where the mountain
  # falls to 10 % of its peak, so ``sigma = width / (2 * sqrt(ln 10))``.
  mountain_scale = mountain_width / (2.0 * np.sqrt(np.log(10.0)))
  return {"T0": device_wrapper(T0),
          "lapse": device_wrapper(lapse),
          "u_max": device_wrapper(u_max),
          "p0": device_wrapper(p0),
          "mountain_height": device_wrapper(mountain_height),
          "mountain_lat": device_wrapper(mountain_lat),
          "mountain_lon": device_wrapper(mountain_lon),
          "mountain_scale": device_wrapper(mountain_scale),
          "radius_earth": device_wrapper(radius_earth),
          "angular_freq_earth": device_wrapper(angular_freq_earth),
          "Rgas": device_wrapper(Rgas),
          "gravity": device_wrapper(gravity),
          }


def _eval_r_hat(z,
                config,
                deep=False):
  # Match the baroclinic-wave convention: ``deep`` switches on the
  # full deep-atmosphere geometric factor ``r/a``; otherwise the
  # shallow-atmosphere approximation r_hat = 1 is used.
  if deep:
    r_hat = (z + config["radius_earth"]) / config["radius_earth"]
  else:
    r_hat = jnp.ones_like(z)
  return r_hat


def _eval_log_lat_pressure_factor(lat,
                                  config):
  """Logarithm of the latitudinal surface-pressure modulation.

  Returns ``log(p_s(lat) / p0) = -(u_max^2 + 2 Omega a u_max)
  sin^2(lat) / (2 R T0)``, the geostrophic-cyclostrophic balance factor
  for the solid body rotation field at the reference temperature.
  """
  u_max = config["u_max"]
  omega = config["angular_freq_earth"]
  a = config["radius_earth"]
  R = config["Rgas"]
  T0 = config["T0"]
  return -(u_max * u_max + 2.0 * omega * a * u_max) * jnp.sin(lat)**2 / (2.0 * R * T0)


def eval_z_surface(lat,
                   lon,
                   config,
                   mountain=False):
  """
  Evaluate surface height (m).

  Parameters
  ----------
  lat : Array[Float, tuple[elem_idx, i_idx, j_idx]]
    Latitude (radians)
  lon : Array[Float, tuple[elem_idx, i_idx, j_idx]]
    Longitude (radians)
  config : TestConfig
    Dict-like containing parameters for the test case
  mountain : bool, default=False
    If ``True``, add a radially symmetric Gaussian mountain centred at
    ``(config["mountain_lat"], config["mountain_lon"])`` with peak height
    ``config["mountain_height"]`` and e-folding scale
    ``config["mountain_scale"]``.

  Notes
  -----
  * See :func:`init_solid_body_config` for how to initialise ``config``.
  * The Gaussian uses the spherical great-circle distance
    ``d = arccos(sin(lat_m) sin(lat) + cos(lat_m) cos(lat) cos(lon - lon_m))``
    so the bump is exactly axisymmetric on the sphere (no separable
    lat/lon distortion at the centre or near the poles).
  * Adding orography breaks the exact analytic solid body balance, but
    the imbalance is confined to the immediate vicinity of the mountain
    and is the standard way to seed mountain-induced wave tests on this
    base state.

  Returns
  -------
  z_surf : Array[Float, tuple[elem_idx, i_idx, j_idx]]
      Surface height (m).  Identically zero when ``mountain=False``.
  """
  if not mountain:
    return jnp.zeros_like(lat)
  lat_m = config["mountain_lat"]
  lon_m = config["mountain_lon"]
  cos_d = (jnp.sin(lat_m) * jnp.sin(lat) +
           jnp.cos(lat_m) * jnp.cos(lat) * jnp.cos(lon - lon_m))
  # Clip guards against floating-point overshoot just outside [-1, 1].
  gc_dist = jnp.arccos(jnp.clip(cos_d, -1.0, 1.0))
  return config["mountain_height"] * jnp.exp(-(gc_dist / config["mountain_scale"])**2)


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
    If true, use the deep-atmosphere base state (``r_hat = r / a`` in
    the height-to-pressure relation).

  Notes
  -----
  * See :func:`init_solid_body_config` for how to initialise ``config``.
  * Temperature is the dry temperature ``T(z) = T0 - lapse * z`` and is
    independent of latitude; the latitudinal surface-pressure factor
    carries the geostrophic-cyclostrophic balance with the solid body
    rotation field.

  Returns
  -------
  pressure : Array[Float, tuple[elem_idx, i_idx, j_idx, level_idx]]
    Moist pressure (Pa); for this dry test case it equals the dry
    pressure.
  temperature : Array[Float, tuple[elem_idx, i_idx, j_idx, level_idx]]
    Temperature (not virtual), in Kelvin.
  """
  T0 = config["T0"]
  lapse = config["lapse"]
  R = config["Rgas"]
  g = config["gravity"]
  r_hat = _eval_r_hat(z, config, deep=deep)
  # Temperature: constant lapse rate.  Independent of latitude.
  temperature = T0 - lapse * z
  # Vertical structure of pressure from hydrostatic balance.  Using the
  # constant-lapse closed form ``log(p/p_s) = (g / (R lapse))
  # * log(1 - lapse * z / T0)``, which reduces to ``-g z / (R T0)`` as
  # ``lapse -> 0``.  ``jnp.where`` selects the isothermal branch when
  # ``lapse`` is exactly zero so the gradient stays finite.
  z_eff = r_hat * z
  isothermal = -g * z_eff / (R * T0)
  with_lapse = (g / (R * lapse)) * jnp.log(jnp.maximum(1.0 - lapse * z_eff / T0, 1e-30))
  log_p_over_ps = jnp.where(lapse > 0.0, with_lapse, isothermal)
  log_p_s_factor = _eval_log_lat_pressure_factor(lat, config)[:, :, :, np.newaxis]
  pressure = config["p0"] * jnp.exp(log_p_s_factor + log_p_over_ps)
  return pressure, temperature


def eval_surface_state(lat,
                       lon,
                       config,
                       deep=False,
                       mountain=False):
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
    If true, use steady state for the deep-atmosphere equation set.
  mountain : bool, default=False
    If ``True``, add the radially symmetric Gaussian mountain configured
    in ``config``; see :func:`eval_z_surface` for the parameters.

  Notes
  -----
  * See :func:`init_solid_body_config` for how to initialise ``config``.

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
  Calculate zonal wind, meridional wind, pressure, temperature, and moisture vapour.

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
    If true, use steady state for the deep-atmosphere equation set.

  Notes
  -----
  * See :func:`init_solid_body_config` for how to initialise ``config``.
  * This is a dry test case: ``q_vapor`` is identically zero, so
    temperature and virtual temperature coincide.

  Returns
  -------
  u : Array[Float, tuple[elem_idx, i_idx, j_idx, level_idx]]
    Zonal wind (m s^-1); ``u_max * cos(lat)``.
  v : Array[Float, tuple[elem_idx, i_idx, j_idx, level_idx]]
    Meridional wind (m s^-1); identically zero.
  pressure : Array[Float, tuple[elem_idx, i_idx, j_idx, level_idx]]
    Moist pressure (Pa); equal to dry pressure for this test.
  virtual_temperature : Array[Float, tuple[elem_idx, i_idx, j_idx, level_idx]]
    Virtual temperature (K); equal to dry temperature for this test.
  q : Array[Float, tuple[elem_idx, i_idx, j_idx, level_idx]]
    Specific humidity (kg kg^-1), identically zero.
  """
  pressure, temperature = eval_pressure_temperature(z, lat, config, deep=deep)
  cos_lat = jnp.cos(lat)[:, :, :, np.newaxis]
  u = config["u_max"] * cos_lat * jnp.ones_like(z)
  v = jnp.zeros_like(u)
  q_vapor = jnp.zeros_like(z)
  return u, v, pressure, temperature, q_vapor


def init_solid_body_state(h_grid,
                          v_grid,
                          model_config,
                          test_config,
                          dims,
                          model,
                          mountain=False,
                          eps=1e-6,
                          enforce_hydrostatic=False):
  """
  Initialise model state for the hydrostatic solid body rotation test case.

  Wraps :func:`init_model_pressure` with the analytic surface state,
  pressure / temperature profile, and wind functions of the constant-lapse
  solid body rotation atmosphere.

  Parameters
  ----------
  h_grid : `SpectralElementGrid`
      Horizontal spectral element grid struct.
  v_grid : `dict`
      Vertical grid struct containing hybrid coordinate coefficients.
  model_config : `dict`
      Model physics configuration dict.
  test_config : `dict`
      Test-case parameters from :func:`init_solid_body_config`.
  dims : frozendict[str, int]
      Grid dimension metadata.
  model : model_info.models
      Dynamical core identifier (from ``model_info.models``).
  mountain : `bool`, default=False
      If ``True``, add the radially symmetric Gaussian mountain configured
      in ``test_config`` (see :func:`init_solid_body_config` and
      :func:`eval_z_surface`).  The base atmosphere is no longer exactly
      balanced in this case; the imbalance is localised around the
      mountain and seeds the standard mountain-induced wave tests.
  eps : `float`, default=1e-6
      Convergence tolerance passed to the height-inversion routines.
  enforce_hydrostatic : `bool`, default=False
      If ``True``, overwrite the initial interface geopotential with the
      hydrostatically balanced value (HOMME non-hydrostatic only).

  Returns
  -------
  model_state : model state struct
      Fully initialised model state struct for the hydrostatic solid
      body rotation test.
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

  model_state = init_model_pressure(z_pi_surf_func,
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
