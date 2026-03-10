from .._config import get_backend as _get_backend
from enum import Enum
import numpy as np
from ..dynamical_cores.initialization import init_model_pressure
from ..dynamical_cores.model_info import deep_atmosphere_models
_be = _get_backend()
jnp = _be.np
device_wrapper = _be.array
remainder = _be.remainder

perturbation_opts = Enum('perturbation_type',
                         [("none", 1),
                          ("exponential", 2),
                          ("streamfunction", 3)])


def init_baroclinic_wave_config(T0E=310,
                                T0P=240,
                                B=2.0,
                                K=3.0,
                                lapse=0.005,
                                pertu0=0.5,
                                pertr=1.0 / 6.0,
                                pertup=1.0,
                                pertexpr=0.1,
                                pertlon=np.pi / 9.0,
                                pertlat=2.0 * np.pi / 9.0,
                                pertz=15000,
                                moistqlat=2.0 * np.pi / 9.0,
                                moistqp=34000.0,
                                moisttr=0.1,
                                moistq0=0.018,
                                moistqr=0.9,
                                moisteps=0.622,
                                moistT0=273.16,
                                moistE0Ast=610.78,
                                p0=1e5,
                                radius_earth=6371e3,
                                angular_freq_earth=7.292e-5,
                                Rgas=287.1,
                                R_water_vapor=461.50,
                                gravity=9.81,
                                model_config=None,
                                alpha=0.5,
                                mountain_heights=[2000.0, 2000.0],
                                mountain_lats=[np.pi / 4.0, np.pi / 4.0],
                                mountain_lons=[(7.0 / 9.0) * np.pi,
                                               0.4 * np.pi],
                                mountain_lat_widths=[40.0 * np.pi / 180.0,
                                                     40.0 * np.pi / 180.0],
                                mountain_lon_widths=[7.0 * np.pi / 180.0,
                                                     7.0 * np.pi / 180.0]):
  """
  Create a struct containing all parameters necessary to initialize the
  moist/dry baroclinic wave

  Parameters
  ----------
  T0E : float
    The nominal value for surface temperature at the equator
  T0P : float
    The nominal value for the surface temperature at the pole
  B : float
    Vertical jet width parameter
  K : float
    Horizontal jet width parameter
  lapse : float
    Nominal atmospheric lapse rate at the surface
  pertu0 : float
    Streamfunction perturbation amplitude
  pertr : float
    Streamfunction perturbation radius
  pertup: float
    Exponential perturbation amplitude
  pertexpr: float
    Exponential perturbation width parameter
  pertlon : float
    Longitude of perturbation (either type) in radians.
  pertlat : float
    Latitude of perturbation (either type) in radians.
  pertz : float
    Maximum height of perturbation in height coordinates (m)
  moistqlat : float
    Meridional width parameter of moisture in radians
  moistqp : float
    Vertical width parameter of moisture in pressure units (Pa)
  moisttr : float
    Vertical cutoff for moisture (ignored for the moment)
  moistq0 : float
    Maximum specific humidity amplitude (moist mixing ratio, fraction not percentage)
  p0 : float
    Nominal constant thermodynamic moist surface pressure.
  radius_earth : float
    Nominal radius of the planet surface (m), `a` or `rearth` in most ESMs
  angular_freq_earth : float
    Angular frequency of the earth (1 / sec), `omega` in most ESMs
  Rgas : float
    Nominal gas constant of dry air (J kg^-1 K^-1)
  R_water_vapor : float
    Gas constant for water vapor (J kg^-1 K^-1)
  gravity : float
    Nominal constant strength of gravity at the surface (m s^-2)
  model_config :
    physics_config struct for your dynamical core. This allows
    other coefficients like Rgas to be read directly from your
    simulation configuration.
  alpha : float
    Skamarock, Ong, and Klemp correction for small-planet simulations.
  mountain_heights : list[float]
    Heights of mountains if present
  mountain_lons : list[float]
    Longitudinal position of mountain centers (radians)
  mountain_lats : list[float]
    Latitudinal position of mountain centers (radians)
  mountain_lat_widths : list[float]
    Distance (radians) between where surface height is 10% of its longitude-wise maximum
  mountain_lon_widths : list[float]
    Distance (radians) between where surface height is 10% of its latitude-wise maximum

  Notes
  -----
  The parameters here cover several test cases. The base state is derived in
  https://doi.org/10.1002/qj.2018 and developed into a baroclinic wave test
  case in https://doi.org/10.1002/qj.2241. Moisture was added for DCMIP2016,
  and a topographic variant was developed in https://doi.org/10.5194/gmd-16-6805-2023


  Returns
  -------
  dict[str, Any]
      test_config to be passed to other functions in this.

  """
  moistqs = 1e-12
  dx_epsilon = 1e-5
  if model_config:
    radius_earth = model_config["radius_earth"]
    angular_freq_earth = model_config["angular_freq_earth"]
    Rgas = model_config["Rgas"]
    R_water_vapor = model_config["moisture_species_Rgas"]["water_vapor"]
    gravity = model_config["gravity"]
  mountain_lat_scales = [lat_width / (2.0 * (-np.log(0.1))**(1.0 / 6.0))
                         for lat_width in mountain_lat_widths]
  mountain_lon_scales = [lon_width / (2.0 * (-np.log(0.1))**(1.0 / 2.0))
                         for lon_width in mountain_lon_widths]
  return {"T0E": device_wrapper(T0E),
          "T0P": device_wrapper(T0P),
          "B": device_wrapper(B),
          "K": device_wrapper(K),
          "lapse": device_wrapper(lapse),
          "pertu0": device_wrapper(pertu0),
          "pertr": device_wrapper(pertr),
          "pertup": device_wrapper(pertup),
          "pertexpr": device_wrapper(pertexpr),
          "pertlon": device_wrapper(pertlon),
          "pertlat": device_wrapper(pertlat),
          "pertz": device_wrapper(pertz),
          "dx_epsilon": device_wrapper(dx_epsilon),
          "moistqlat": device_wrapper(moistqlat),
          "moistqp": device_wrapper(moistqp),
          "moisttr": device_wrapper(moisttr),
          "moistqs": device_wrapper(moistqs),
          "moistq0": device_wrapper(moistq0),
          "moistqr": device_wrapper(moistqr),
          "moisteps": device_wrapper(moisteps),
          "moistT0": device_wrapper(moistT0),
          "moistE0Ast": device_wrapper(moistE0Ast),
          "p0": device_wrapper(p0),
          "radius_earth": device_wrapper(radius_earth),
          "angular_freq_earth": device_wrapper(angular_freq_earth),
          "Rgas": device_wrapper(Rgas),
          "R_water_vapor": device_wrapper(R_water_vapor),
          "gravity": device_wrapper(gravity),
          "alpha": device_wrapper(alpha),
          "mountain_heights": device_wrapper(mountain_heights),
          "mountain_lats": device_wrapper(mountain_lats),
          "mountain_lons": device_wrapper(mountain_lons),
          "mountain_lat_scales": device_wrapper(mountain_lat_scales),
          "mountain_lon_scales": device_wrapper(mountain_lon_scales),
          }


def _eval_T0(config):
  return (config["alpha"] * config["T0E"] +
          (1.0 - config["alpha"]) * config["T0P"])


def _eval_constH(config):
  return config["Rgas"] * _eval_T0(config) / config["gravity"]


def _eval_constA(config):
  return 1.0 / config["lapse"]


def _eval_constB(config):
  T0 = _eval_T0(config)
  return (T0 - config["T0P"]) / (T0 * config["T0P"])


def _eval_scaledZ(z,
                  config):
  return z / (config["B"] * _eval_constH(config))


def _eval_inttau2(z,
                  config):
  return (_eval_constC(config) * z *
          jnp.exp(-_eval_scaledZ(z, config)**2))


def _eval_constC(config):
  T0E = config["T0E"]
  T0P = config["T0P"]
  return (0.5 * (config["K"] + 2.0) *
          (T0E - T0P) / (T0E * T0P))


def _eval_r_hat(z,
                config,
                deep=False):
  # note: should be separate from model code.
  # so constant-g equation set can be used
  if deep:
    r_hat = (z + config["radius_earth"]) / config["radius_earth"]
  else:
    r_hat = jnp.ones_like(z)
  return r_hat


def eval_z_surface(lat,
                   lon,
                   config,
                   mountain=False):
  """
  Evaluate surface height (m)

  Parameters
  ----------
  lat : Array[Float, tuple[elem_idx, i_idx, j_idx]]
    Latitude (radians)
  lon : Array[Float, tuple[elem_idx, i_idx, j_idx]]
    Longitude (radians)
  config : TestConfig
    Dict-like containing parameters for test case
  mountain : bool, default=False
    If true, use the surface height from the topographic version of the test case.

  Notes
  -----
  * See init_baroclinic_wave_config for how to initialize config.
  * The topography is described in https://doi.org/10.5194/gmd-16-6805-2023

  Returns
  -------
  z_surf : Array[Float, tuple[elem_idx, i_idx, j_idx]]
      Surface height (m)
  """
  if mountain:
    zs = jnp.zeros_like(lat)
    for (mountain_height,
         mountain_lat,
         mountain_lon,
         mountain_lat_scale,
         mountain_lon_scale) in zip(config["mountain_heights"],
                                    config["mountain_lats"],
                                    config["mountain_lons"],
                                    config["mountain_lat_scales"],
                                    config["mountain_lon_scales"]):
      d0 = remainder(lon - mountain_lon, 2.0 * jnp.pi)
      d0 = jnp.minimum(d0, 2.0 * jnp.pi - d0)
      zs += mountain_height * jnp.exp(-(((lat - mountain_lat) / mountain_lat_scale)**6 +
                                        (d0 / mountain_lon_scale)**2))
  else:
    zs = jnp.zeros_like(lat)
  return zs


def eval_pressure_temperature(z,
                              lat,
                              config,
                              deep=False):
  """
  Evaluate pressure and (dry) temperature on a 3d grid.

  Parameters
  ----------
  z : Array[Float, tuple[elem_idx, i_idx, j_idx, level_idx]]
    Geometric height above the model surface in meters
  lat : Array[Float, tuple[elem_idx, i_idx, j_idx]]
      Latitude (radians)
  config : TestConfig
    Dict-like containing parameters for test case
  deep : bool, default=False
    If true, use the deep atmopsphere base state.

  Notes
  -----
  * See init_baroclinic_wave_config for how to initialize config.

  Returns
  -------
  pressure : Array[Float, tuple[elem_idx, i_idx, j_idx, level_idx]]
    Moist pressure (Pa)
  temperature : Array[Float, tuple[elem_idx, i_idx, j_idx, level_idx]]
    Temperature (not virtual), in Kelvin
  """
  lapse = config["lapse"]
  K = config["K"]
  T0 = _eval_T0(config)
  constA = _eval_constA(config)
  constB = _eval_constB(config)
  constC = _eval_constC(config)
  scaledZ = _eval_scaledZ(z, config)

  # note: this can be optimized for numpy so
  # scaledZ**2 quantities are not recomputed

  tau1 = (constA * lapse / T0 * jnp.exp(lapse * z / T0) +
          constB * (1.0 - 2.0 * scaledZ**2) * jnp.exp(-scaledZ**2))
  tau2 = constC * (1.0 - 2.0 * scaledZ**2) * jnp.exp(-scaledZ**2)

  inttau1 = (constA * (jnp.exp(lapse * z / T0) - device_wrapper(1.0)) +
             constB * z * jnp.exp(-scaledZ**2))
  inttau2 = _eval_inttau2(z, config)

  r_hat = _eval_r_hat(z, config, deep=deep)

  inttermT = ((r_hat * jnp.cos(lat)[:, :, :, np.newaxis])**K -
              K / (K + 2.0) * (r_hat * jnp.cos(lat)[:, :, :, np.newaxis])**(K + 2))

  temperature = 1.0 / (r_hat**2 * (tau1 - tau2 * inttermT))
  pressure = config["p0"] * jnp.exp(-config["gravity"] / config["Rgas"] *
                                    (inttau1 - inttau2 * inttermT))
  return pressure, temperature


def eval_surface_state(lat,
                       lon,
                       config,
                       deep=False,
                       mountain=False):
  """
  Evaluate the surface height and moist pressure.

  Parameters
  ----------
  lat : Array[Float, tuple[elem_idx, i_idx, j_idx]]
      Latitude (radians)
  lon : Array[Float, tuple[elem_idx, i_idx, j_idx]]
      Longitude(radians)
  config : TestConfig
    Dict-like containing parameters for test case
  deep : bool
    If true, use steady state for deep atmosphere equation set
  mountain : bool
    If true, use surface topography from the mountain version of the test.

  Notes
  -----
  * See init_baroclinic_wave_config for how to initialize config.

  Returns
  -------
  z_surface : Array[Float, tuple[elem_idx, i_idx, j_idx, level_idx]]
    Surface height in meters
  p_surface : Array[Float, tuple[elem_idx, i_idx, j_idx, level_idx]]
    Moist surface pressure in Pascal.

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
               deep=False,
               moist=False,
               pert_type=perturbation_opts.none):
  """
  Calculate zonal wind, meridional wind, pressure, virtual temperature, and moisture vapor.

  Parameters
  ----------
  lat : Array[Float, tuple[elem_idx, i_idx, j_idx]]
      Latitude (radians)
  lon : Array[Float, tuple[elem_idx, i_idx, j_idx]]
      Longitude (radians)
  z : Array[Float, tuple[elem_idx, i_idx, j_idx, level_idx]]
    Geometric height above the model surface in meters
  config : TestConfig
    Dict-like containing parameters for test case
  deep : bool
    If true, use steady state for deep-atmosphere equation set.
  moist : bool
    If true, use idealized moisture profile that can be used with Kessler physics.
  pert_type : perturbation_opts
    Type of perturbation to be used

  Notes
  -----
  * See init_baroclinic_wave_config for how to initialize config.
  * Valid perturbation types are
    * perturbation_opts.exponential: (used in DCMIP 2016 cases). This is a little noisier,
    but avoids the dycore-specific model spread that the streamfunction perturbation causes.
    * erturbation_opts.streamfunction: the perturbation used in the original dry test case paper.
    A less divergent perturbation that avoids initialization noise.
    * none: use the analytic steady state. Use this for steady state tests.

  Returns
  -------
  u : Array[Float, tuple[elem_idx, i_idx, j_idx, level_idx]]
    Zonal wind (m s^-1)
  v : Array[Float, tuple[elem_idx, i_idx, j_idx, level_idx]]
    Meridional wind (m s^-1)
  pressure : Array[Float, tuple[elem_idx, i_idx, j_idx, level_idx]]
    Moist pressure (Pa)
  virtual_temperature : Array[Float, tuple[elem_idx, i_idx, j_idx, level_idx]]
    Virtual temperature (K)
  q : Array[Float, tuple[elem_idx, i_idx, j_idx, level_idx]]
    Specific humidity (kg moisture / kg moist air, fraction not percentage)
  """
  K = config["K"]
  inttau2 = _eval_inttau2(z, config)
  r_hat = _eval_r_hat(z, config, deep=deep)
  cos_lat = jnp.cos(lat)[:, :, :, np.newaxis]
  inttermU = ((r_hat * cos_lat)**(K - 1.0) -
              (r_hat * cos_lat)**(K + 1.0))
  pressure, virtual_temperature = eval_pressure_temperature(z, lat, config, deep=deep)
  bigU = (config["gravity"] / config["radius_earth"] * K *
          inttau2 * inttermU * virtual_temperature)

  if deep:
    rcoslat = config["radius_earth"] * cos_lat
  else:
    rcoslat = (z + config["radius_earth"]) * cos_lat
  solid_body_rotation = config["angular_freq_earth"] * rcoslat
  u = -solid_body_rotation + jnp.sqrt(solid_body_rotation**2 +
                                      rcoslat * bigU)
  v = jnp.zeros_like(u)

  if moist:
    p0 = config["p0"]
    eta = pressure / p0
    q_vapor = config["moistq0"] * (jnp.exp(-(lat[:, :, :, np.newaxis] / config["moistqlat"])**4) *
                                   jnp.exp(-((eta - 1.0) * p0 / config["moistqp"])**2))
  else:
    q_vapor = jnp.zeros_like(z)

  # todo: handle pert
  if pert_type == perturbation_opts.exponential:
    print("using exponential perturbation type")
    u += eval_exponential(lat, lon, z, config)
  elif pert_type == perturbation_opts.streamfunction:
    print("using streamfunction perturbation type")
    eps = 1e-5
    sf_lat_above = eval_streamfunction(lat + eps, lon, z, config)
    sf_lat_below = eval_streamfunction(lat - eps, lon, z, config)
    sf_lon_above = eval_streamfunction(lat, lon + eps, z, config)
    sf_lon_below = eval_streamfunction(lat, lon - eps, z, config)
    u += - (sf_lat_above - sf_lat_below) / (2 * eps)
    v += (sf_lon_above - sf_lon_below) / (2 * eps)
  return u, v, pressure, virtual_temperature, q_vapor


def eval_great_circle_dist(lat,
                           lon,
                           config):
  """
  Evaluate the great circle distance in radians from the perturbation center

  Parameters
  ----------
  lat : Array[Float, tuple[elem_idx, i_idx, j_idx]]
    Latitude (radians)
  lon : Array[Float, tuple[elem_idx, i_idx, j_idx]]
    Longitude (radians)
  config : TestConfig
    Dict-like containing parameters for test case

  Notes
  -----
  * See init_baroclinic_wave_config for how to initialize config.

  Returns
  -------
  gc_dist : Array[Float, tuple[elem_idx, i_idx, j_idx]]
  """
  return (1.0 / config["pertexpr"] *
          jnp.arccos(jnp.sin(config["pertlat"]) *
                     jnp.sin(lat) +
                     jnp.cos(config["pertlat"]) *
                     jnp.cos(lat) *
                     jnp.cos(lon - config["pertlon"])))


def eval_taper_fn(z,
                  config):
  """
  Evaluate the vertical taper function used by both perturbation types

  Parameters
  ----------
  z : Array[Float, tuple[elem_idx, i_idx, j_idx, level_idx]]
    Geometric height above the model surface in meters
  config : TestConfig
    Dict-like containing parameters for test case

  Notes
  -----
  * See init_baroclinic_wave_config for how to initialize config.

  Returns
  -------
  taper : Array[Float, tuple[elem_idx, i_idx, j_idx, level_idx]]
    Unitless taper function that controls vertical structure of perturbation.
  """
  pertz = config["pertz"]
  taper_below_pertz = (1.0 - 3.0 * z**2 / pertz**2 + 2.0 * z**3 / pertz**3)
  return jnp.where(z < pertz,
                   taper_below_pertz,
                   jnp.zeros_like(z))


def eval_exponential(lat,
                     lon,
                     z,
                     config):
  """
  Evaluate exponential perturbation for zonal wind field

  Parameters
  ----------
  lat : Array[Float, tuple[elem_idx, i_idx, j_idx]]
    Latitude (radians)
  lon : Array[Float, tuple[elem_idx, i_idx, j_idx]]
    Longitude (radians)
  z : Array[Float, tuple[elem_idx, i_idx, j_idx, level_idx]]
    Geometric height above the model surface in meters
  config : TestConfig
    Dict-like containing parameters for test case

  Notes
  -----
  * See init_baroclinic_wave_config for how to initialize config.

  Returns
  -------
  u_perturbation : Array[Float, tuple[elem_idx, i_idx, j_idx, level_idx]]
    Perturbation in m s^-1
  """
  greatcircle_dist = eval_great_circle_dist(lat, lon, config)[:, :, :, np.newaxis]
  taper = eval_taper_fn(z, config)

  pert_inside_circle = (config["pertup"] *
                        taper *
                        jnp.exp(-greatcircle_dist**2))
  return jnp.where(greatcircle_dist < 1.0,
                   pert_inside_circle,
                   jnp.zeros_like(z))


def eval_streamfunction(lat,
                        lon,
                        z,
                        config):
  """
  Evaluate streamfunction from which perturbation can be computed.

  Parameters
  ----------
  lat : Array[Float, tuple[elem_idx, i_idx, j_idx]]
    Latitude (radians)
  lon : Array[Float, tuple[elem_idx, i_idx, j_idx]]
    Longitude (radians)
  z : Array[Float, tuple[elem_idx, i_idx, j_idx, level_idx]]
    Geometric height above the model surface in meters
  config : TestConfig
    Dict-like containing parameters for test case

  Notes
  -----
  * See init_baroclinic_wave_config for how to initialize config.
  * The resulting u, v values are typically computed by computing finite differences:
    * u = -∂ψ/∂ϕ
    * v = ∂ψ/∂λ

  Returns
  -------
  streamfunction : Array[Float, tuple[elem_idx, i_idx, j_idx, level_idx]]
    Streamfunction in m s^-1 (this is a mathematician's streamfunction).
  """
  greatcircle_dist = eval_great_circle_dist(lat, lon, config)
  taper = eval_taper_fn(z, config)
  pert_inside_circle = jnp.cos(0.5 * jnp.pi * greatcircle_dist)
  return jnp.where(greatcircle_dist < 1.0,
                   -config["pertu0"] * config["pertr"] * taper * pert_inside_circle**4,
                   jnp.zeros_like(z))


def init_baroclinic_wave_state(h_grid,
                               v_grid,
                               model_config,
                               test_config,
                               dims,
                               model,
                               mountain=False,
                               moist=False,
                               eps=1e-6,
                               pert_type=perturbation_opts.none,
                               enforce_hydrostatic=False):
  """
  Initialise model state for the Jablonowski-Williamson baroclinic
  wave test case.

  Wraps ``init_model_pressure`` with the analytic surface state,
  pressure/temperature profile, and wind functions from the
  moist baroclinic wave analytic initialisation module.

  Parameters
  ----------
  h_grid : `SpectralElementGrid`
      Horizontal spectral element grid struct.
  v_grid : `dict`
      Vertical grid struct containing hybrid coordinate coefficients.
  model_config : `dict`
      Model physics configuration dict.
  test_config : `dict`
      Test-case parameters passed to the analytic initialisation
      functions (e.g. rotation rate, lapse rate).
  dims : frozendict[str, int]
      Grid dimension metadata.
  model : model_info.models
      Dynamical core identifier (from ``model_info.models``).
  mountain : `bool`, default=False
      If ``True``, add an idealised surface topography perturbation.
  moist : `bool`, default=False
      If ``True``, include water vapour in the initial state.
  eps : `float`, default=1e-6
      Convergence tolerance passed to the height-inversion routines.
  pert_type : `perturbation_opts`, default=perturbation_opts.none
      Type of initial wind perturbation to apply.
  enforce_hydrostatic : `bool`, default=False
      If ``True``, overwrite the initial interface geopotential with the
      hydrostatically balanced value (HOMME non-hydrostatic only).

  Returns
  -------
  model_state : model state struct
      Fully initialised model state struct for the baroclinic wave test.
  """
  lat = h_grid["physical_coords"][:, :, :, 0]
  deep = model in deep_atmosphere_models

  def z_pi_surf_func(lat,
                     lon):
    return eval_surface_state(lat, lon, test_config, mountain=mountain)

  def Q_func(lat, lon, z):
    return eval_state(lat,
                      lon,
                      z,
                      test_config,
                      moist=moist,
                      deep=deep,
                      pert_type=pert_type)[4]

  def p_func(z):
    return eval_pressure_temperature(z,
                                     lat,
                                     test_config,
                                     deep=deep)[0]

  def u_func(lat, lon, z):
    return eval_state(lat,
                      lon,
                      z,
                      test_config,
                      moist=moist,
                      deep=deep,
                      pert_type=pert_type)[0]

  def v_func(lat, lon, z):
    return eval_state(lat,
                      lon,
                      z,
                      test_config,
                      moist=moist,
                      deep=deep,
                      pert_type=pert_type)[1]

  def Tv_func(lat, lon, z):
    return eval_state(lat,
                      lon,
                      z,
                      test_config,
                      moist=moist,
                      deep=deep)[3]

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
