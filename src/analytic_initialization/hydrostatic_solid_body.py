# from ..config_alt import get_backend as _get_backend



# def init_test_config(T0=300,
#                      lapse=0.005,
#                      u_max=0.0,
#                      surface_pressure_equator=1e5,
#                      Rgas=287.0,
#                      model_config=None):
#   if model_config is not None:
#     Rgas = model_config["Rgas"]
#   return {"T0": T0,
#           "lapse": lapse,
#           "u_max": u_max,
#           "surface_pressure_equator": surface_pressure_equator,
#           "Rgas": Rgas}


# def wind(lat,
#          lon,
#          z):
#   # assume shallow atmosphere
#   u = jnp.ones_like(z) 
#   v = jnp.zeros_like(z)
#   return jnp.stack((u, v), axis=-1)


# def temperature(lat,
#                 lon,
#                 z,
#                 test_config):
#   return test_config["T0"] - test_config["lapse_rate"] * z


# def pressure(lat,
#              lon,
#              z):
#   pass
