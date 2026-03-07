from pysces.config import jnp
from pysces.time_step import time_step_options
from pysces.dynamical_cores.model_state import sum_dynamics, copy_dynamics
from pysces.model_info import models, cam_se_models, homme_models
from pysces.dynamical_cores.time_stepping import advance_dynamics_euler, advance_dynamics_ullrich_5stage
from frozendict import frozendict


def get_dummy_time_stepping_config(dynamics_tstep_type=time_step_options.RK3_5STAGE,
                                   dt_dynamics=1.0):
  return frozendict(dynamics=frozendict(step_type=dynamics_tstep_type,
                                        dt=dt_dynamics),
                    dynamics_subcycle=1)


epsilons = {"horizontal_wind": 0.02,
            "T": 5e-3,
            "theta_v_d_mass": 0.02,
            "d_mass": .2,
            "phi_i": 1.0,
            "w_i": 1e-2}


def test_steady_state_euler(nx7_np4_dry_homme_hydro, nx7_np4_dry_se):
  for model, struct in zip([models.cam_se, models.homme_hydrostatic],
                           [nx7_np4_dry_se, nx7_np4_dry_homme_hydro]):
    model_state = struct["model_state"]
    physics_config = struct["physics_config"]
    dims = struct["dims"]
    h_grid = struct["h_grid"]
    v_grid = struct["v_grid"]
    dt = 10
    timestep_config = get_dummy_time_stepping_config(dynamics_tstep_type=time_step_options.Euler, dt_dynamics=dt)
    if model in cam_se_models:
      dry_species = model_state["tracers"]["dry_air_species"]
      moisture_species = model_state["tracers"]["moisture_species"]
    else:
      dry_species = None
      moisture_species = None
    dynamics_new, _ = advance_dynamics_euler(model_state["dynamics"],
                                             model_state["static_forcing"],
                                             h_grid,
                                             v_grid,
                                             physics_config,
                                             timestep_config,
                                             dims,
                                             model,
                                             moisture_species=moisture_species,
                                             dry_air_species=dry_species)
    for s_idx in range(10):
      dynamics_new, _ = advance_dynamics_euler(dynamics_new,
                                               model_state["static_forcing"],
                                               h_grid,
                                               v_grid,
                                               physics_config,
                                               timestep_config,
                                               dims,
                                               model,
                                               moisture_species=moisture_species,
                                               dry_air_species=dry_species)
    dynamics_diff = sum_dynamics(dynamics_new, model_state["dynamics"], 1.0, -1.0, model)
    if model in homme_models:
      dynamics_diff["theta_v_d_mass"] = dynamics_diff["theta_v_d_mass"] / model_state["dynamics"]["d_mass"]
    for field in dynamics_diff.keys():
      assert jnp.max(jnp.abs(dynamics_diff[field])) < epsilons[field]


def test_steady_state_ullrich(nx7_np4_dry_homme_hydro, nx7_np4_dry_se):
  for model, struct in zip([models.cam_se, models.homme_hydrostatic],
                           [nx7_np4_dry_se, nx7_np4_dry_homme_hydro]):
    model_state = struct["model_state"]
    physics_config = struct["physics_config"]
    dims = struct["dims"]
    h_grid = struct["h_grid"]
    v_grid = struct["v_grid"]
    dt = 10
    timestep_config = get_dummy_time_stepping_config(dynamics_tstep_type=time_step_options.SSPRK3, dt_dynamics=dt)
    if model in cam_se_models:
      dry_species = model_state["tracers"]["dry_air_species"]
      moisture_species = model_state["tracers"]["moisture_species"]
    else:
      dry_species = None
      moisture_species = None
    dynamics_ref = copy_dynamics(model_state["dynamics"], model)
    for s_idx in range(10):
      model_state["dynamics"] = advance_dynamics_ullrich_5stage(model_state["dynamics"],
                                                                model_state["static_forcing"],
                                                                h_grid,
                                                                v_grid,
                                                                physics_config,
                                                                timestep_config,
                                                                dims,
                                                                model,
                                                                moisture_species=moisture_species,
                                                                dry_air_species=dry_species)[0]
    dynamics_diff = sum_dynamics(model_state["dynamics"], dynamics_ref, 1.0, -1.0, model)
    if model in homme_models:
      dynamics_diff["theta_v_d_mass"] = dynamics_diff["theta_v_d_mass"] / model_state["dynamics"]["d_mass"]
    for field in dynamics_diff.keys():
      assert jnp.max(jnp.abs(dynamics_diff[field])) < epsilons[field]


def test_steady_state_nonhydro(nx7_np4_dry_homme_nonhydro):
  model = models.homme_nonhydrostatic
  model_state = nx7_np4_dry_homme_nonhydro["model_state"]
  physics_config = nx7_np4_dry_homme_nonhydro["physics_config"]
  dims = nx7_np4_dry_homme_nonhydro["dims"]
  h_grid = nx7_np4_dry_homme_nonhydro["h_grid"]
  v_grid = nx7_np4_dry_homme_nonhydro["v_grid"]

  dt = .5
  timestep_config = get_dummy_time_stepping_config(dynamics_tstep_type=time_step_options.SSPRK3, dt_dynamics=dt)
  dry_species = None
  moisture_species = None
  dynamics_ref = copy_dynamics(model_state["dynamics"], model)
  for s_idx in range(100):
    model_state["dynamics"] = advance_dynamics_ullrich_5stage(model_state["dynamics"],
                                                              model_state["static_forcing"],
                                                              h_grid,
                                                              v_grid,
                                                              physics_config,
                                                              timestep_config,
                                                              dims,
                                                              model,
                                                              moisture_species=moisture_species,
                                                              dry_air_species=dry_species)[0]
  dynamics_diff = sum_dynamics(model_state["dynamics"], dynamics_ref, 1.0, -1.0, model)
  dynamics_diff["theta_v_d_mass"] = dynamics_diff["theta_v_d_mass"] / model_state["dynamics"]["d_mass"]
  for field in dynamics_diff.keys():
    assert jnp.max(jnp.abs(dynamics_diff[field])) < epsilons[field] / 2.0
