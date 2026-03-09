from src._config import get_backend as _get_backend
_be = _get_backend()
jnp = _be.np
device_unwrapper = _be.unwrap
device_wrapper = _be.array
use_wrapper = _be.use_wrapper
wrapper_type = _be.wrapper_type
get_global_array = _be.get_global_array
import numpy as np

from src.mesh_generation.cubed_sphere import init_cube_topo
from src.mesh_generation.mesh import init_element_corner_vert_redundancy
from src.mesh_generation.equiangular_metric import init_grid_from_topo
from src.analytic_initialization.moist_baroclinic_wave import (init_baroclinic_wave_config,
                                                                  eval_pressure_temperature,
                                                                  eval_state)
from src.dynamical_cores.utils_3d import z_to_g
from src.dynamical_cores.model_info import models


def test_shallow():
  npt = 4
  if use_wrapper and wrapper_type == "torch":
    # getting double precision init is not a priority
    return
  nx = 31
  face_connectivity, face_mask, face_position, face_position_2d = init_cube_topo(nx)
  vert_redundancy = init_element_corner_vert_redundancy(face_connectivity)
  grid, dims = init_grid_from_topo(face_connectivity, face_mask, face_position_2d, vert_redundancy, npt)
  config_shallow = init_baroclinic_wave_config(pertu0=0.0,
                                               pertup=0.0)
  lat = grid["physical_coords"][:, :, :, 0]
  eps = device_wrapper(1e-3)

  for z in jnp.linspace(0, 40e3, 10):
    z_above = device_wrapper((z + eps) * jnp.ones((*lat.shape, 1)))
    pressure_above, _ = eval_pressure_temperature(z_above, lat, config_shallow, deep=False)
    z_below = device_wrapper((z - eps) * jnp.ones((*lat.shape, 1)))
    pressure_below, _ = eval_pressure_temperature(z_below, lat, config_shallow, deep=False)
    z_center = device_wrapper(z * jnp.ones((*lat.shape, 1)))
    pressure, temperature = eval_pressure_temperature(z_center, lat, config_shallow, deep=False)
    rho = pressure / (config_shallow["Rgas"] * temperature)
    dp_dz = (pressure_above - pressure_below) / (2 * eps)
    diff = z_to_g(z, config_shallow, models.homme_nonhydrostatic) * rho + dp_dz
    diff = get_global_array(diff, dims)
    assert (np.max(np.abs(device_unwrapper(diff))) < 0.001)


def test_moist_shallow():
  npt = 4
  if use_wrapper and wrapper_type == "torch":
    # getting double precision init is not a priority
    return
  nx = 31
  face_connectivity, face_mask, face_position, face_position_2d = init_cube_topo(nx)
  vert_redundancy = init_element_corner_vert_redundancy(face_connectivity)
  grid, dims = init_grid_from_topo(face_connectivity, face_mask, face_position_2d, vert_redundancy, npt)
  config_moist = init_baroclinic_wave_config(pertu0=0.0,
                                             pertup=0.0)
  config_pseudo_moist = init_baroclinic_wave_config(pertu0=0.0,
                                                    pertup=0.0,
                                                    moistq0=0.0)
  lat = grid["physical_coords"][:, :, :, 0]
  lon = grid["physical_coords"][:, :, :, 1]
  for z in jnp.linspace(0, 40e3, 10):
    z_2d = device_wrapper((z) * jnp.ones((*lat.shape, 1)))
    _, _, _, _, Q = eval_state(lat, lon, z_2d, config_moist, deep=False, moist=True)
    Q = get_global_array(Q, dims)
    assert jnp.all(Q > 0)
    assert jnp.all(Q <= config_moist["moistq0"])
    # test if we're accidentally returning temperature instead of virtual temperature
    _, _, _, virtual_temp_1, _ = eval_state(lat, lon, z_2d, config_moist, deep=False, moist=True)
    _, _, _, virtual_temp_2, Q = eval_state(lat, lon, z_2d, config_pseudo_moist, deep=False, moist=True)
    Q = get_global_array(Q, dims)
    virtual_temp_1 = get_global_array(virtual_temp_1, dims)
    virtual_temp_2 = get_global_array(virtual_temp_2, dims)
    assert jnp.allclose(Q, 0.0)
    assert jnp.allclose(virtual_temp_1, virtual_temp_2)


def test_deep():
  npt = 4
  if use_wrapper and wrapper_type == "torch":
    # getting double precision init is not a priority
    return
  nx = 31
  face_connectivity, face_mask, face_position, face_position_2d = init_cube_topo(nx)
  vert_redundancy = init_element_corner_vert_redundancy(face_connectivity)
  grid, dims = init_grid_from_topo(face_connectivity, face_mask, face_position_2d, vert_redundancy, npt)
  lat = grid["physical_coords"][:, :, :, 0]
  lon = grid["physical_coords"][:, :, :, 1]
  eps = device_wrapper(1e-3)
  for alpha in [0.4, 0.5, 0.8]:
    config_deep = init_baroclinic_wave_config(pertu0=0.0,
                                              pertup=0.0,
                                              radius_earth=6371e3 / 20.0,
                                              angular_freq_earth=7.292e-5 * 20.0,
                                              alpha=alpha)
    for z in jnp.linspace(0, 40e3, 10):
      z_above = device_wrapper((z + eps) * jnp.ones((*lat.shape, 1)))
      pressure_above, _ = eval_pressure_temperature(z_above, lat, config_deep, deep=True)
      z_below = device_wrapper((z - eps) * jnp.ones((*lat.shape, 1)))
      pressure_below, _ = eval_pressure_temperature(z_below, lat, config_deep, deep=True)
      z_center = device_wrapper(z * jnp.ones((*lat.shape, 1)))
      u, v, pressure, temperature, _ = eval_state(lat, lon, z_center, config_deep, deep=True)
      rho = pressure / (config_deep["Rgas"] * temperature)
      dp_dz = (pressure_above - pressure_below) / (2 * eps)
      metric_terms = -(u**2 + v**2) / (z_center + config_deep["radius_earth"])
      ncts = -u * 2.0 * config_deep["angular_freq_earth"] * jnp.cos(lat)[:, :, :, np.newaxis]
      g = z_to_g(z_center, config_deep, models.homme_nonhydrostatic_deep)
      diff = dp_dz / rho + g + metric_terms + ncts
      diff = get_global_array(diff, dims)
      assert np.max(np.abs(device_unwrapper(diff)) < 1e-3)
