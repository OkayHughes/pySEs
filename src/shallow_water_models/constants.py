from .._config import get_backend as _get_backend
_be = _get_backend()
device_wrapper = _be.array


def init_physics_config_shallow_water(radius_earth=6371e3,
                                      angular_freq_earth=7.292e-5,
                                      gravity=9.81,
                                      alpha=0.0):
  """
  Returns a struct that contains physical constants of the sphere on which
  you are simulating.

  Parameters
  ----------
  radius_earth : `float`, default=6371e3
      The radius of the sphere in meters, default is the nominal radius of the earth
  angular_freq_earth: `float`, default=7.292e-5.
      Reciprocal of sidereal day, units s^-1. Default is angular frequency of earth.
  gravity : `float`, default=9.81
      Gravitational acceleration in m s^-2, default is the nominal gravity at earth's surface
  alpha : `float`, default=0.0
      Angle in radians between physical lat-lon coords and axis of rotation of planet

  See Also
  --------
  run_shallow_water.simulate_sw

  Returns
  -------
  physics_config : dict[str, float]
      Physical constants to pass to a shallow water simulation.
  """
  return {"radius_earth": device_wrapper(radius_earth),
          "angular_freq_earth": device_wrapper(angular_freq_earth),
          "gravity": device_wrapper(gravity),
          "alpha": device_wrapper(alpha)}
