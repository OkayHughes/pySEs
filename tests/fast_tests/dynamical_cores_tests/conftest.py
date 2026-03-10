from pyses.dynamical_cores.model_info import models
from pyses.mesh_generation.equiangular_metric import init_quasi_uniform_grid
from pyses.dynamical_cores.mass_coordinate import init_vertical_grid
from pyses.dynamical_cores.physics_config import init_physics_config
from ...test_data.mass_coordinate_grids import cam30
from pyses.analytic_initialization.moist_baroclinic_wave import init_baroclinic_wave_config, init_baroclinic_wave_state
from pytest import fixture


def quasi_uniform_test_states(nx, npt, model, mountain=False, moist=False):
  h_grid, dims = init_quasi_uniform_grid(nx, npt)
  v_grid = init_vertical_grid(cam30["hybrid_a_i"],
                              cam30["hybrid_b_i"],
                              cam30["p0"],
                              model)
  physics_config = init_physics_config(model)
  test_config = init_baroclinic_wave_config(model_config=physics_config)
  model_state = init_baroclinic_wave_state(h_grid,
                                           v_grid,
                                           physics_config,
                                           test_config,
                                           dims,
                                           model,
                                           mountain=mountain,
                                           moist=moist,
                                           enforce_hydrostatic=True,
                                           eps=1e-5)
  return {"h_grid": h_grid,
          "v_grid": v_grid,
          "dims": dims,
          "physics_config": physics_config,
          "model_state": model_state}


@fixture
def nx15_np4_dry_se():
  nx = 15
  npt = 4
  return quasi_uniform_test_states(nx, npt, models.cam_se)


@fixture
def nx15_np4_dry_se_whole():
  nx = 15
  npt = 4
  return quasi_uniform_test_states(nx, npt, models.cam_se_whole_atmosphere)


@fixture
def nx15_np4_dry_homme_hydro():
  nx = 15
  npt = 4
  return quasi_uniform_test_states(nx, npt, models.homme_hydrostatic)


@fixture
def nx15_np4_dry_homme_nonhydro():
  nx = 15
  npt = 4
  return quasi_uniform_test_states(nx, npt, models.homme_nonhydrostatic)


@fixture
def nx7_np4_dry_se():
  nx = 7
  npt = 4
  return quasi_uniform_test_states(nx, npt, models.cam_se)


@fixture
def nx7_np4_dry_se_whole():
  nx = 7
  npt = 4
  return quasi_uniform_test_states(nx, npt, models.cam_se_whole_atmosphere)


@fixture
def nx7_np4_dry_homme_hydro():
  nx = 7
  npt = 4
  return quasi_uniform_test_states(nx, npt, models.homme_hydrostatic)


@fixture
def nx7_np4_dry_homme_nonhydro():
  nx = 7
  npt = 4
  return quasi_uniform_test_states(nx, npt, models.homme_nonhydrostatic)


@fixture
def nx15_np4_moist_se():
  nx = 15
  npt = 4
  return quasi_uniform_test_states(nx, npt, models.cam_se, moist=True)


@fixture
def nx15_np4_moist_se_whole():
  nx = 15
  npt = 4
  return quasi_uniform_test_states(nx, npt, models.cam_se_whole_atmosphere, moist=True)


@fixture
def nx15_np4_moist_homme_hydro():
  nx = 15
  npt = 4
  return quasi_uniform_test_states(nx, npt, models.homme_hydrostatic, moist=True)


@fixture
def nx15_np4_moist_homme_nonhydro():
  nx = 15
  npt = 4
  return quasi_uniform_test_states(nx, npt, models.homme_nonhydrostatic, moist=True)


@fixture
def nx15_np4_mountain_se():
  nx = 15
  npt = 4
  return quasi_uniform_test_states(nx, npt, models.cam_se, mountain=True)


@fixture
def nx15_np4_mountain_se_whole():
  nx = 15
  npt = 4
  return quasi_uniform_test_states(nx, npt, models.cam_se_whole_atmosphere, mountain=True)


@fixture
def nx15_np4_mountain_homme_hydro():
  nx = 15
  npt = 4
  return quasi_uniform_test_states(nx, npt, models.homme_hydrostatic, mountain=True)


@fixture
def nx15_np4_mountain_homme_nonhydro():
  nx = 15
  npt = 4
  return quasi_uniform_test_states(nx, npt, models.homme_nonhydrostatic, mountain=True)
