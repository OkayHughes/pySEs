from pyses.mesh_generation.equiangular_metric import init_quasi_uniform_grid
from pyses.operations_2d.operators import horizontal_gradient
from pyses.operations_2d.local_assembly import project_scalar
from ...test_data.mass_coordinate_grids import cam30
from pyses.dynamical_cores.mass_coordinate import (init_vertical_grid,
                                                 surface_mass_to_interface_mass)
from pyses.dynamical_cores.initialization import (z_from_p_monotonic_moist,
                                                z_from_p_monotonic_dry,
                                                integrate_weight_of_vapor)
from pyses.dynamical_cores.physics_config import init_physics_config
from pyses.dynamical_cores.utils_3d import interface_to_delta
from pyses._config import get_backend as _get_backend
from pyses.analytic_initialization.moist_baroclinic_wave import (init_baroclinic_wave_config,
                                                               eval_surface_state,
                                                               init_baroclinic_wave_state)
from pyses.dynamical_cores.model_info import models
from ...context import allclose_global
_be = _get_backend()
jnp = _be.np
device_wrapper = _be.array
get_global_array = _be.get_global_array


def test_z_p_func():
  config = init_physics_config(models.homme_hydrostatic)
  pressures = device_wrapper(jnp.linspace(config["p0"], 100, 10))
  T0 = 300

  def p_given_z(z):
    return config["p0"] * jnp.exp(-config["gravity"] * z /
                                  (config["Rgas"] * T0))

  def z_given_p(p):
    return -T0 * config["Rgas"] / config["gravity"] * jnp.log(p / config["p0"])

  heights = z_from_p_monotonic_moist(pressures, p_given_z, eps=1e-10, z_top=80e3)
  assert (jnp.max(jnp.abs(heights - z_given_p(pressures)) / pressures) < 1e-5)


def test_z_p_dry_no_moisture():
  config = init_physics_config(models.homme_hydrostatic)
  pressures = device_wrapper(jnp.linspace(config["p0"], 100, 10))
  v_grid = init_vertical_grid(cam30["hybrid_a_i"],
                              cam30["hybrid_b_i"],
                              cam30["p0"],
                              models.cam_se)
  T0 = 300

  def p_given_z(z):
    return config["p0"] * jnp.exp(-config["gravity"] * z /
                                  (config["Rgas"] * T0))

  def z_given_p(p):
    return -T0 * config["Rgas"] / config["gravity"] * jnp.log(p / config["p0"])

  def Tv_given_z(z):
    return T0 * jnp.ones_like(z)

  def Q_given_z(z):
    return 0 * z

  heights = z_from_p_monotonic_dry(pressures, p_given_z, Q_given_z, Tv_given_z, v_grid, config, eps=1e-10, z_top=80e3)
  assert (jnp.max(jnp.abs(heights - z_given_p(pressures)) / pressures) < 1e-5)


def test_z_p_dry_with_moisture():
  config = init_physics_config(models.homme_hydrostatic)
  dry_weights = device_wrapper(jnp.linspace(config["p0"], 300, 20))
  v_grid = init_vertical_grid(cam30["hybrid_a_i"],
                              cam30["hybrid_b_i"],
                              cam30["p0"],
                              models.cam_se)
  T0 = 300

  def moist_p_given_z(z):
    return config["p0"] * jnp.exp(-config["gravity"] * z /
                                  (config["Rgas"] * T0))

  def Tv_given_z(z):
    return T0 * jnp.ones_like(z)

  def Q_given_z(z):
    return 3.0 / moist_p_given_z(z)

  p_top = v_grid["reference_surface_mass"] * v_grid["hybrid_a_i"][0]
  z_top = z_from_p_monotonic_moist(p_top * jnp.ones_like(dry_weights), moist_p_given_z)

  heights = z_from_p_monotonic_dry(dry_weights, moist_p_given_z, Q_given_z, Tv_given_z,
                                   v_grid, config, eps=1e-10, z_top=80e3)
  weight_of_Q_vs_z = 3.0 / (T0 * config["Rgas"]) * (z_top - heights) * config["gravity"]
  assert (jnp.max(jnp.abs(moist_p_given_z(heights) - (dry_weights + weight_of_Q_vs_z)) / dry_weights) < 1e-5)


def test_moisture_quadrature():
  model_config = init_physics_config(models.homme_hydrostatic)
  T0 = 300
  rho_water = .1

  def p_given_z(z):
    return model_config["p0"] * jnp.exp(-model_config["gravity"] * z /
                                        (model_config["Rgas"] * T0))

  def Tv_given_z(z):
    return T0 * jnp.ones_like(z)

  def q_given_z(z):
    return rho_water * model_config["Rgas"] * Tv_given_z(z) / p_given_z(z)

  zs = jnp.zeros((1, 1, 3))
  z_top = 1000.0 * jnp.ones_like(zs)
  analytic_answer = model_config["gravity"] * (rho_water) * (z_top - zs)
  numerical_answer = integrate_weight_of_vapor(p_given_z, Tv_given_z, q_given_z, zs, z_top, model_config)
  assert(jnp.allclose(analytic_answer, numerical_answer))


def test_init():
  npt = 4
  nx = 15
  h_grid, dims = init_quasi_uniform_grid(nx, npt)
  v_grid = init_vertical_grid(cam30["hybrid_a_i"],
                              cam30["hybrid_b_i"],
                              cam30["p0"],
                              models.homme_hydrostatic)
  lat = h_grid["physical_coords"][:, :, :, 0]
  lon = h_grid["physical_coords"][:, :, :, 1]
  for mountain in [False, True]:
    model_config = init_physics_config(models.homme_hydrostatic)
    test_config = init_baroclinic_wave_config(model_config=model_config)
    model_state = init_baroclinic_wave_state(h_grid, v_grid, model_config, test_config, dims,
                                             models.homme_hydrostatic, mountain=mountain)
    z_surf, ps = eval_surface_state(lat, lon, test_config, mountain=mountain)
    phi_surf = model_config["gravity"] * z_surf
    grad_phi_surf = horizontal_gradient(phi_surf, h_grid, a=model_config["radius_earth"])
    grad_phi_surf_cont = jnp.stack((project_scalar(grad_phi_surf[:, :, :, 0], h_grid, dims),
                                    project_scalar(grad_phi_surf[:, :, :, 1], h_grid, dims)), axis=-1)
    p_int = surface_mass_to_interface_mass(ps, v_grid)
    d_mass = interface_to_delta(p_int)
    p_top = v_grid["reference_surface_mass"] * v_grid["hybrid_a_i"][0]

    assert allclose_global(d_mass, model_state["dynamics"]["d_mass"], dims)
    assert allclose_global(phi_surf, model_state["static_forcing"]["phi_surf"], dims)
    assert allclose_global(grad_phi_surf_cont, model_state["static_forcing"]["grad_phi_surf"], dims)
    assert allclose_global(jnp.sum(get_global_array(d_mass, dims), axis=-1) + p_top, ps, dims)


def test_init_moist():
  npt = 4
  nx = 4
  h_grid, dims = init_quasi_uniform_grid(nx, npt)
  model = models.cam_se
  v_grid = init_vertical_grid(cam30["hybrid_a_i"],
                              cam30["hybrid_b_i"],
                              cam30["p0"],
                              model)
  lat = h_grid["physical_coords"][:, :, :, 0]
  lon = h_grid["physical_coords"][:, :, :, 1]
  for mountain in [False, True]:
    model_config = init_physics_config(model)
    test_config = init_baroclinic_wave_config(model_config=model_config)
    model_state = init_baroclinic_wave_state(h_grid,
                                             v_grid,
                                             model_config,
                                             test_config,
                                             dims,
                                             model,
                                             mountain=mountain,
                                             moist=True,
                                             eps=1e-8)
    z_surf, ps = eval_surface_state(lat, lon, test_config, mountain=mountain)
    # test dry pressure levels ()
    # test correct dry surface mass
    # test correct d mass values
    # test correct tracer_mass value using a moist coordinate
    pressure_moisture = model_state["tracers"]["moisture_species"]["water_vapor"] * model_state["dynamics"]["d_mass"]
    p_top = v_grid["reference_surface_mass"] * v_grid["hybrid_a_i"][0]
    moist_surface_pressure = jnp.sum(model_state["dynamics"]["d_mass"] + pressure_moisture, axis=-1) + p_top
    assert jnp.max(jnp.abs(get_global_array(ps, dims) - get_global_array(moist_surface_pressure, dims))) < 1e-3
