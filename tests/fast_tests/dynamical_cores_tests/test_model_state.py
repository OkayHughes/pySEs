from src._config import get_backend as _get_backend
import numpy as np
from src.dynamical_cores.model_info import (models,
                                            cam_se_models,
                                            thermodynamic_variable_names,
                                            hydrostatic_models,
                                            deep_atmosphere_models)
from src.mesh_generation.equiangular_metric import init_quasi_uniform_grid
from src.dynamical_cores.physics_config import init_physics_config
from src.analytic_initialization.moist_baroclinic_wave import init_baroclinic_wave_config, init_baroclinic_wave_state
from src.dynamical_cores.mass_coordinate import init_vertical_grid
from src.dynamical_cores.model_state import (sum_tracers,
                                             sum_tracers_series,
                                             wrap_dynamics,
                                             wrap_static_forcing,
                                             init_static_forcing,
                                             wrap_model_state,
                                             wrap_tracers,
                                             project_dynamics,
                                             sum_dynamics_series,
                                             sum_dynamics,
                                             check_dynamics_nan,
                                             check_tracers_nan,
                                             copy_model_state)
from .test_assembly_3d import is_3d_field_c0
from ...test_data.mass_coordinate_grids import cam30
from ...context import allclose_global
_be = _get_backend()
jnp = _be.np
device_wrapper = _be.array
get_global_array = _be.get_global_array


def test_copy_state():
  npt = 4
  nx = 4
  model = models.cam_se
  model_config = init_physics_config(model)
  v_grid = init_vertical_grid(cam30["hybrid_a_i"],
                              cam30["hybrid_b_i"],
                              cam30["p0"],
                              model)
  h_grid, dims = init_quasi_uniform_grid(nx, npt)
  test_config = init_baroclinic_wave_config(model_config=model_config)
  model_state = init_baroclinic_wave_state(h_grid,
                                           v_grid,
                                           model_config,
                                           test_config,
                                           dims,
                                           model,
                                           mountain=False,
                                           moist=True,
                                           eps=1e-3)
  trac_name = "fitzpatrick"
  water_vapor = model_state["tracers"]["moisture_species"]["water_vapor"]
  model_state["tracers"]["tracers"][trac_name] = 1.0 * jnp.ones_like(water_vapor)
  model_state_new = copy_model_state(model_state, model)
  assert allclose_global(model_state_new["dynamics"]["horizontal_wind"],
                         model_state["dynamics"]["horizontal_wind"],
                         dims)
  assert allclose_global(model_state_new["dynamics"]["T"], model_state["dynamics"]["T"], dims)
  assert allclose_global(model_state_new["dynamics"]["d_mass"], model_state["dynamics"]["d_mass"], dims)
  tracers_new = model_state_new["tracers"]
  tracers_old = model_state["tracers"]
  assert set(tracers_new["moisture_species"].keys()) == set(tracers_old["moisture_species"].keys())
  for species_name in tracers_new["moisture_species"].keys():
    assert allclose_global(tracers_new["moisture_species"][species_name],
                           tracers_old["moisture_species"][species_name], dims)
  assert set(tracers_new["tracers"].keys()) == set(tracers_old["tracers"].keys())
  for species_name in tracers_new["tracers"].keys():
    assert allclose_global(tracers_new["tracers"][species_name],
                           tracers_old["tracers"][species_name], dims)
  assert set(tracers_new["dry_air_species"].keys()) == set(tracers_old["dry_air_species"].keys())
  for species_name in model_state_new["tracers"]["dry_air_species"].keys():
    assert allclose_global(tracers_new["dry_air_species"][species_name],
                           tracers_old["dry_air_species"][species_name], dims)


def test_tracer_sums():
  npt = 4
  nx = 4
  h_grid, dims = init_quasi_uniform_grid(nx, npt)
  for model in [models.homme_hydrostatic, models.cam_se, models.cam_se_whole_atmosphere]:
    v_grid = init_vertical_grid(cam30["hybrid_a_i"],
                                cam30["hybrid_b_i"],
                                cam30["p0"],
                                model)
    model_config = init_physics_config(model)
    test_config = init_baroclinic_wave_config(model_config=model_config)
    model_state_1 = init_baroclinic_wave_state(h_grid,
                                               v_grid,
                                               model_config,
                                               test_config,
                                               dims,
                                               model,
                                               mountain=False,
                                               moist=True,
                                               eps=1e-3)
    model_state_2 = init_baroclinic_wave_state(h_grid,
                                               v_grid,
                                               model_config,
                                               test_config,
                                               dims,
                                               model,
                                               mountain=False,
                                               moist=True,
                                               eps=1e-3)
    val_1 = 1.0
    val_2 = 2.0
    coeff_1 = 0.25
    coeff_2 = 1.0 - coeff_1
    total = coeff_1 * val_1 + coeff_2 * val_2
    water_vapor = model_state_1["tracers"]["moisture_species"]["water_vapor"]
    model_state_1["tracers"]["moisture_species"]["water_vapor"] = val_1 * jnp.ones_like(water_vapor)
    model_state_2["tracers"]["moisture_species"]["water_vapor"] = val_2 * jnp.ones_like(water_vapor)
    tracer_sum = sum_tracers(model_state_1["tracers"]["moisture_species"],
                             model_state_2["tracers"]["moisture_species"],
                             coeff_1,
                             coeff_2)
    assert jnp.allclose(get_global_array(tracer_sum["water_vapor"], dims), total)
    trac_name = "fitzpatrick"
    model_state_1["tracers"]["tracers"][trac_name] = val_1 * jnp.ones_like(water_vapor)
    model_state_2["tracers"]["tracers"][trac_name] = val_2 * jnp.ones_like(water_vapor)
    tracer_sum = sum_tracers(model_state_1["tracers"]["tracers"],
                             model_state_2["tracers"]["tracers"],
                             coeff_1,
                             coeff_2)
    assert jnp.allclose(get_global_array(tracer_sum[trac_name], dims), total)
    if model in cam_se_models:
      tracer_sum = sum_tracers(model_state_1["tracers"]["dry_air_species"],
                               model_state_2["tracers"]["dry_air_species"],
                               coeff_1,
                               coeff_2,
                               is_dry_air_species=True)
      tracer_total = jnp.zeros_like(tracer_sum[next(iter(tracer_sum.keys()))])
      for tracer_name in tracer_sum.keys():
        tracer_total += tracer_sum[tracer_name]
      assert jnp.allclose(get_global_array(tracer_total, dims), 1.0)
    tracers_advance = sum_tracers_series([model_state_1["tracers"], model_state_2["tracers"]], [coeff_1, coeff_2], model)
    assert jnp.allclose(get_global_array(tracers_advance["moisture_species"]["water_vapor"], dims), total)
    assert jnp.allclose(get_global_array(tracers_advance["tracers"][trac_name], dims), total)
    if model in cam_se_models:
      dry_species = tracers_advance["dry_air_species"]
      tracer_total = jnp.zeros_like(dry_species[next(iter(dry_species.keys()))])
      for tracer_name in dry_species.keys():
        tracer_total += dry_species[tracer_name]
      assert jnp.allclose(get_global_array(tracer_total, dims), 1.0)


def test_wrappers():
  npt = 4
  nx = 4
  h_grid, dims = init_quasi_uniform_grid(nx, npt)
  for model in models:
    v_grid = init_vertical_grid(cam30["hybrid_a_i"],
                                cam30["hybrid_b_i"],
                                cam30["p0"],
                                model)
    model_config = init_physics_config(model)
    test_config = init_baroclinic_wave_config(model_config=model_config)
    model_state = init_baroclinic_wave_state(h_grid,
                                             v_grid,
                                             model_config,
                                             test_config,
                                             dims,
                                             model,
                                             mountain=False,
                                             moist=True,
                                             eps=1e-3)
    u = model_state["dynamics"]["horizontal_wind"]
    thermo = model_state["dynamics"][thermodynamic_variable_names[model]]
    d_mass = model_state["dynamics"]["d_mass"]
    if model not in hydrostatic_models:
      phi_i = model_state["dynamics"]["phi_i"]
      w_i = model_state["dynamics"]["w_i"]
    else:
      phi_i = None
      w_i = None
    dynamics_struct = wrap_dynamics(u, thermo, d_mass, model, phi_i=phi_i, w_i=w_i)
    for field in dynamics_struct.keys():
      assert allclose_global(dynamics_struct[field], model_state["dynamics"][field], dims)
    if model in cam_se_models:
      dry_air_species = model_state["tracers"]["dry_air_species"]
    else:
      dry_air_species = None
    tracer_struct = wrap_tracers(model_state["tracers"]["moisture_species"],
                                 model_state["tracers"]["tracers"],
                                 model,
                                 dry_air_species=dry_air_species)
    assert set(tracer_struct["moisture_species"].keys()) == set(model_state["tracers"]["moisture_species"].keys())
    for tracer_name in tracer_struct["moisture_species"].keys():
      assert allclose_global(tracer_struct["moisture_species"][tracer_name],
                             model_state["tracers"]["moisture_species"][tracer_name], dims)
    assert set(tracer_struct["tracers"].keys()) == set(model_state["tracers"]["tracers"].keys())
    for tracer_name in tracer_struct["tracers"].keys():
      assert allclose_global(tracer_struct["moisture_species"][tracer_name],
                             model_state["tracers"]["moisture_species"][tracer_name], dims)
    if model in cam_se_models:
      assert set(tracer_struct["dry_air_species"].keys()) == set(model_state["tracers"]["dry_air_species"].keys())
      for tracer_name in tracer_struct["dry_air_species"].keys():
        assert allclose_global(tracer_struct["dry_air_species"][tracer_name],
                               model_state["tracers"]["dry_air_species"][tracer_name], dims)
    phi_surf = model_state["static_forcing"]["phi_surf"]
    grad_phi_surf = model_state["static_forcing"]["grad_phi_surf"]
    coriolis_param = model_state["static_forcing"]["coriolis_param"]
    if model in deep_atmosphere_models:
      nontrad_coriolis_param = model_state["static_forcing"]["nontrad_coriolis_param"]
    else:
      nontrad_coriolis_param = None
    static_forcing = wrap_static_forcing(phi_surf,
                                         grad_phi_surf,
                                         coriolis_param,
                                         nontrad_coriolis_param=nontrad_coriolis_param)
    assert set(static_forcing.keys()) == set(model_state["static_forcing"].keys())
    for field in static_forcing.keys():
      assert allclose_global(static_forcing[field], model_state["static_forcing"][field], dims)
    phi_surf = jnp.cos(h_grid["physical_coords"][:, :, :, 0])
    static_forcing = init_static_forcing(phi_surf, h_grid, model_config, dims, model)
    assert set(static_forcing.keys()) == set(model_state["static_forcing"].keys())
    assert allclose_global(static_forcing["phi_surf"], phi_surf, dims)
    assert is_3d_field_c0(static_forcing["grad_phi_surf"], h_grid, dims)
    assert allclose_global(static_forcing["coriolis_param"], model_state["static_forcing"]["coriolis_param"], dims)
    if model in deep_atmosphere_models:
      assert allclose_global(static_forcing["nontrad_coriolis_param"],
                             model_state["static_forcing"]["nontrad_coriolis_param"], dims)
    else:
      assert "nontrad_coriolis_param" not in static_forcing.keys()
    model_state_new = wrap_model_state(model_state["dynamics"],
                                       model_state["static_forcing"],
                                       model_state["tracers"])
    assert allclose_global(model_state["dynamics"]["horizontal_wind"],
                           model_state_new["dynamics"]["horizontal_wind"], dims)
    assert allclose_global(model_state["static_forcing"]["phi_surf"],
                           model_state_new["static_forcing"]["phi_surf"], dims)
    assert allclose_global(model_state["tracers"]["moisture_species"]["water_vapor"],
                           model_state_new["tracers"]["moisture_species"]["water_vapor"], dims)


def test_project_dynamics_state():
  npt = 4
  nx = 4
  h_grid, dims = init_quasi_uniform_grid(nx, npt)
  for model in models:
    v_grid = init_vertical_grid(cam30["hybrid_a_i"],
                                cam30["hybrid_b_i"],
                                cam30["p0"],
                                model)
    model_config = init_physics_config(model)
    test_config = init_baroclinic_wave_config(model_config=model_config)
    model_state = init_baroclinic_wave_state(h_grid,
                                             v_grid,
                                             model_config,
                                             test_config,
                                             dims,
                                             model,
                                             mountain=False,
                                             moist=False,
                                             eps=1e-3)

    def noise_pert(field):
      return field + device_wrapper(np.random.normal(size=field.shape))

    for field in model_state["dynamics"].keys():
      model_state["dynamics"][field] = noise_pert(model_state["dynamics"][field])
      assert not is_3d_field_c0(model_state["dynamics"][field], h_grid, dims)
    dynamics_cont = project_dynamics(model_state["dynamics"], h_grid, dims, model)
    for field in dynamics_cont.keys():
      if field == "horizontal_wind":
        for comp_idx in range(2):
          assert is_3d_field_c0(dynamics_cont[field][:, :, :, :, comp_idx], h_grid, dims)
      else:
        assert is_3d_field_c0(dynamics_cont[field], h_grid, dims)


def test_advance_dynamics():
  npt = 4
  nx = 4
  h_grid, dims = init_quasi_uniform_grid(nx, npt)
  for model in models:
    v_grid = init_vertical_grid(cam30["hybrid_a_i"],
                                cam30["hybrid_b_i"],
                                cam30["p0"],
                                model)
    model_config = init_physics_config(model)
    test_config = init_baroclinic_wave_config(model_config=model_config)
    model_state = init_baroclinic_wave_state(h_grid,
                                             v_grid,
                                             model_config,
                                             test_config,
                                             dims,
                                             model,
                                             mountain=False,
                                             moist=False,
                                             eps=1e-3)
    coeff_1 = 1.0
    coeff_2 = 2.0
    coeff_3 = 3.0

    dynamics_state_out = sum_dynamics(model_state["dynamics"],
                                      model_state["dynamics"],
                                      coeff_1,
                                      coeff_2,
                                      model)
    assert set(dynamics_state_out.keys()) == set(model_state["dynamics"].keys())
    for field in dynamics_state_out.keys():
      assert allclose_global(dynamics_state_out[field],
                             (coeff_1 + coeff_2) * model_state["dynamics"][field], dims)
    dynamics_state_out = sum_dynamics_series([model_state["dynamics"],
                                              model_state["dynamics"],
                                              model_state["dynamics"]],
                                             [coeff_1, coeff_2, coeff_3], model)
    for field in dynamics_state_out.keys():
      assert allclose_global(dynamics_state_out[field],
                             (coeff_1 + coeff_2 + coeff_3) * model_state["dynamics"][field],
                             dims)


def test_check_nan():
  npt = 4
  nx = 4
  h_grid, dims = init_quasi_uniform_grid(nx, npt)
  for model in models:
    v_grid = init_vertical_grid(cam30["hybrid_a_i"],
                                cam30["hybrid_b_i"],
                                cam30["p0"],
                                model)
    model_config = init_physics_config(model)
    test_config = init_baroclinic_wave_config(model_config=model_config)
    model_state = init_baroclinic_wave_state(h_grid,
                                             v_grid,
                                             model_config,
                                             test_config,
                                             dims,
                                             model,
                                             mountain=False,
                                             moist=True,
                                             eps=1e-3)
    water_vapor = model_state["tracers"]["moisture_species"]["water_vapor"]
    model_state["tracers"]["tracers"]["fitzpatrick"] = jnp.ones_like(water_vapor)
    for field in model_state["dynamics"].keys():
      assert not check_dynamics_nan(model_state["dynamics"], h_grid, model)
      field_old = jnp.copy(model_state["dynamics"][field])
      model_state["dynamics"][field] = jnp.nan * jnp.ones_like(model_state["dynamics"][field])
      assert check_dynamics_nan(model_state["dynamics"], h_grid, model)
      model_state["dynamics"][field] = field_old
    for field in model_state["tracers"]["moisture_species"].keys():
      assert not check_tracers_nan(model_state["tracers"], h_grid, model)
      tracer_old = jnp.copy(model_state["tracers"]["moisture_species"][field])
      model_state["tracers"]["moisture_species"][field] = jnp.nan * jnp.ones_like(tracer_old)
      assert check_tracers_nan(model_state["tracers"], h_grid, model)
      model_state["tracers"]["moisture_species"][field] = tracer_old
    for field in model_state["tracers"]["tracers"].keys():
      assert not check_tracers_nan(model_state["tracers"], h_grid, model)
      tracer_old = jnp.copy(model_state["tracers"]["tracers"][field])
      model_state["tracers"]["tracers"][field] = jnp.nan * jnp.ones_like(model_state["tracers"]["tracers"][field])
      assert check_tracers_nan(model_state["tracers"], h_grid, model)
      model_state["tracers"]["tracers"][field] = tracer_old
    if model in cam_se_models:
      for field in model_state["tracers"]["dry_air_species"].keys():
        assert not check_tracers_nan(model_state["tracers"], h_grid, model)
        tracer_old = jnp.copy(model_state["tracers"]["dry_air_species"][field])
        model_state["tracers"]["dry_air_species"][field] = jnp.nan * jnp.ones_like(tracer_old)
        assert check_tracers_nan(model_state["tracers"], h_grid, model)
        model_state["tracers"]["dry_air_species"][field] = tracer_old
