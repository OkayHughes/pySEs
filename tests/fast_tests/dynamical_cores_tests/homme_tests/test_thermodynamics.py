from src.analytic_initialization.moist_baroclinic_wave import (init_baroclinic_wave_config,
                                                                  eval_pressure_temperature,
                                                                  init_baroclinic_wave_state)
from src.dynamical_cores.physics_config import init_physics_config
from src.mesh_generation.equiangular_metric import init_quasi_uniform_grid
from src._config import get_backend as _get_backend
_be = _get_backend()
jnp = _be.np
from ....test_data.mass_coordinate_grids import vertical_grid_finite_diff
from src.dynamical_cores.utils_3d import interface_to_midlevel
from src.dynamical_cores.mass_coordinate import init_vertical_grid
from src.dynamical_cores.homme.thermodynamics import eval_mu, eval_midlevel_pressure, eval_balanced_geopotential
from src.dynamical_cores.model_info import models


def test_eos_hydro():
  npt = 4
  nx = 5
  h_grid, dims = init_quasi_uniform_grid(nx, npt)
  lat = h_grid["physical_coords"][:, :, :, 0]
  vstruct = vertical_grid_finite_diff(200)
  model = models.homme_hydrostatic
  v_grid = init_vertical_grid(vstruct["hybrid_a_i"],
                              vstruct["hybrid_b_i"],
                              vstruct["p0"],
                              model)
  for mountain in [False, True]:
    model_config = init_physics_config(model)
    test_config = init_baroclinic_wave_config(model_config=model_config)
    lat = h_grid["physical_coords"][:, :, :, 0]
    model_state = init_baroclinic_wave_state(h_grid, v_grid, model_config, test_config, dims, model, mountain=mountain)
    p_mid = eval_midlevel_pressure(model_state["dynamics"],
                                   v_grid)
    phi_i = eval_balanced_geopotential(model_state["static_forcing"]["phi_surf"],
                                       p_mid,
                                       model_state["dynamics"]["theta_v_d_mass"], model_config)

    z_i = phi_i / model_config['gravity']
    z_mid = interface_to_midlevel(z_i)

    p_model, exner, r_hat_i, mu = eval_mu(model_state["dynamics"], phi_i, v_grid, model_config, model)
    Tv = (model_state["dynamics"]["theta_v_d_mass"] / model_state["dynamics"]["d_mass"]) * exner
    assert (jnp.max(jnp.abs(p_model - p_mid)) < 1e-8)
    assert (jnp.max(jnp.abs(p_model - p_mid)) < 1e-8)
    p_int_state, t_int_state = eval_pressure_temperature(z_mid, lat, test_config)
    p_mid_state, t_mid_state = eval_pressure_temperature(z_mid, lat, test_config)
    assert (jnp.max(jnp.abs(p_model - p_mid_state) / p_model) < 0.01)
    assert (jnp.max(jnp.abs(Tv - t_mid_state) / Tv) < 0.01)
    assert (jnp.max(jnp.abs(mu - 1)) < 1e-5)


def test_eos_nonhydro():
  npt = 4
  nx = 5
  model = models.homme_nonhydrostatic
  h_grid, dims = init_quasi_uniform_grid(nx, npt)
  lat = h_grid["physical_coords"][:, :, :, 0]
  vstruct = vertical_grid_finite_diff(100)
  v_grid = init_vertical_grid(vstruct["hybrid_a_i"],
                              vstruct["hybrid_b_i"],
                              vstruct["p0"],
                              model)
  for mountain in [False, True]:
    model_config = init_physics_config(model)
    test_config = init_baroclinic_wave_config(model_config=model_config)
    lat = h_grid["physical_coords"][:, :, :, 0]
    model_state = init_baroclinic_wave_state(h_grid, v_grid, model_config, test_config, dims, model,
                                             mountain=mountain, eps=1e-9)
    phi_i = model_state["dynamics"]["phi_i"]
    p_mid = eval_midlevel_pressure(model_state["dynamics"],
                                   v_grid)

    z_i = phi_i / model_config['gravity']
    z_mid = interface_to_midlevel(z_i)

    p_model, exner, r_hat_i, mu = eval_mu(model_state["dynamics"], phi_i, v_grid, model_config, model)
    p_mid_state, t_mid_state = eval_pressure_temperature(z_mid, lat, test_config)
    Tv = (model_state["dynamics"]["theta_v_d_mass"] / model_state["dynamics"]["d_mass"]) * exner
    assert (jnp.max(jnp.abs(p_model - p_mid) / p_model) < .01)
    assert (jnp.max(jnp.abs(p_model - p_mid) / p_model) < .01)
    assert (jnp.max(jnp.abs(p_model - p_mid_state) / p_model) < 0.01)
    assert (jnp.max(jnp.abs(Tv - t_mid_state) / Tv) < 0.01)
    assert (jnp.max(jnp.abs(mu - 1)) < 1e-2)
