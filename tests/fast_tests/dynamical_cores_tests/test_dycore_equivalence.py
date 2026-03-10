from ...test_data.mass_coordinate_grids import cam30
from pyses._config import get_backend as _get_backend
import numpy as np
from pyses.dynamical_cores.physics_config import init_physics_config
from pyses.analytic_initialization.moist_baroclinic_wave import init_baroclinic_wave_config, init_baroclinic_wave_state
from pyses.mesh_generation.element_local_metric import init_quasi_uniform_grid_elem_local
from pyses.dynamical_cores.mass_coordinate import init_vertical_grid
from pyses.dynamical_cores.utils_3d import physical_dot_product
from pyses.dynamical_cores.model_info import models
from pyses.dynamical_cores.cam_se.explicit_terms import (eval_energy_gradient_term,
                                                       eval_temperature_horiz_advection_term,
                                                       eval_temperature_vertical_advection_term,
                                                       pressure_gradient_options,
                                                       eval_pressure_gradient_force_term)
from pyses.dynamical_cores.homme.explicit_terms import (eval_grad_kinetic_energy_h_term,
                                                      eval_pgrad_phi_term,
                                                      eval_theta_v_divergence_term,
                                                      eval_pgrad_pressure_term)
from pyses.dynamical_cores.cam_se.thermodynamics import eval_exner_function, eval_virtual_temperature, eval_sum_species
from pyses.dynamical_cores.cam_se.explicit_terms import init_common_variables as init_common_variables_se
from pyses.dynamical_cores.cam_se.explicit_terms import eval_d_mass_divergence_term as eval_d_mass_divergence_term_se
from pyses.dynamical_cores.cam_se.explicit_terms import eval_vorticity_term as eval_vorticity_term_se
from pyses.dynamical_cores.homme.explicit_terms import init_common_variables as init_common_variables_homme
from pyses.dynamical_cores.homme.explicit_terms import eval_d_mass_divergence_term as eval_d_mass_divergence_term_homme
from pyses.dynamical_cores.homme.explicit_terms import eval_vorticity_term as eval_vorticity_term_homme
from pyses.operations_2d.operators import inner_product
_be = _get_backend()
jnp = _be.np


def compare_equivalent_terms(model_state_se, model_state_homme,
                             physics_config_se, physics_config_homme,
                             v_grid_se, v_grid_homme,
                             model_se, model_homme, h_grid):
  dynamics = model_state_se["dynamics"]
  static_forcing = model_state_se["static_forcing"]
  moisture_species = model_state_se["tracers"]["moisture_species"]
  dry_air_species = model_state_se["tracers"]["dry_air_species"]
  common_variables_se = init_common_variables_se(dynamics, static_forcing,
                                                 moisture_species,
                                                 dry_air_species,
                                                 h_grid,
                                                 v_grid_se,
                                                 physics_config_se,
                                                 model_se)
  dynamics = model_state_homme["dynamics"]
  static_forcing = model_state_homme["static_forcing"]
  common_variables_homme = init_common_variables_homme(dynamics,
                                                       static_forcing,
                                                       h_grid,
                                                       v_grid_homme,
                                                       physics_config_homme,
                                                       model_homme)
  d_mass_div_se = eval_d_mass_divergence_term_se(common_variables_se, h_grid, physics_config_se)
  d_mass_div_homme = eval_d_mass_divergence_term_homme(common_variables_homme)
  vorticity_se = eval_vorticity_term_se(common_variables_se, h_grid, physics_config_se)
  vorticity_homme = eval_vorticity_term_homme(common_variables_homme, h_grid, physics_config_homme)
  grad_energy_se = eval_energy_gradient_term(common_variables_se, h_grid, physics_config_se)
  grad_energy_homme = (eval_grad_kinetic_energy_h_term(common_variables_homme, h_grid, physics_config_homme) +
                       eval_pgrad_phi_term(common_variables_homme))
  temperature_tend_se = (eval_temperature_horiz_advection_term(common_variables_se, h_grid, physics_config_se) +
                         eval_temperature_vertical_advection_term(common_variables_se, h_grid, physics_config_se))
  total_mixing_ratio = eval_sum_species(moisture_species)
  T_v_tend = eval_virtual_temperature(temperature_tend_se,
                                      moisture_species,
                                      total_mixing_ratio,
                                      common_variables_se["R_dry"],
                                      physics_config_se)
  theta_v_tend_se = T_v_tend / eval_exner_function(common_variables_se["pressure_midlevel"],
                                                   common_variables_se["R_dry"],
                                                   common_variables_se["cp_dry"],
                                                   physics_config_se)
  theta_v_tend_homme = eval_theta_v_divergence_term(common_variables_homme,
                                                    h_grid,
                                                    physics_config_homme) / model_state_homme["dynamics"]["d_mass"]
  pgrad_homme = eval_pgrad_pressure_term(common_variables_homme, h_grid, physics_config_homme)

  pgrads_se = {}
  for pgrad in pressure_gradient_options:
    pgrads_se[pgrad] = eval_pressure_gradient_force_term(common_variables_se,
                                                         h_grid,
                                                         v_grid_se,
                                                         physics_config_se,
                                                         pgrad)
  return {"d_mass_divergence": {"se": d_mass_div_se,
                                "homme": d_mass_div_homme},
          "vorticity": {"se": vorticity_se,
                        "homme": vorticity_homme},
          "grad_energy": {"se": grad_energy_se,
                          "homme": grad_energy_homme},
          "theta_v_divergence": {"se": theta_v_tend_se,
                                 "homme": theta_v_tend_homme},
          "pgrad": {"se": pgrads_se,
                    "homme": pgrad_homme}}


def compare_equivalent_tendency(model_state_se, model_state_homme):
  pass


def test_equivalent_terms_dry_steady_state(nx15_np4_dry_se,
                                           nx15_np4_dry_homme_hydro):
  return
  model_se = models.cam_se
  model_homme = models.homme_hydrostatic
  h_grid = nx15_np4_dry_homme_hydro["h_grid"]
  v_grid_se = nx15_np4_dry_se["v_grid"]
  v_grid_homme = nx15_np4_dry_homme_hydro["v_grid"]

  physics_config_se = nx15_np4_dry_se["physics_config"]
  model_state_se = nx15_np4_dry_se["model_state"]
  physics_config_homme = nx15_np4_dry_homme_hydro["physics_config"]
  model_state_homme = nx15_np4_dry_homme_hydro["model_state"]
  equivalent_terms = compare_equivalent_terms(model_state_se, model_state_homme,
                                              physics_config_se, physics_config_homme,
                                              v_grid_se, v_grid_homme, model_se, model_homme, h_grid)
  assert (jnp.max(jnp.abs(equivalent_terms["d_mass_divergence"]["se"] -
                          equivalent_terms["d_mass_divergence"]["homme"]))) < 1e-4
  assert (jnp.max(jnp.abs(equivalent_terms["vorticity"]["se"] -
                          equivalent_terms["vorticity"]["homme"]))) < 1e-4
  assert (jnp.max(jnp.abs(equivalent_terms["grad_energy"]["se"] +
                          equivalent_terms["grad_energy"]["homme"]))) < 1e-2
  assert (jnp.max(jnp.abs(equivalent_terms["theta_v_divergence"]["se"] +
                          equivalent_terms["theta_v_divergence"]["homme"]))) < 1e-4
  for pgrad in pressure_gradient_options:
    assert jnp.max(jnp.abs(equivalent_terms["pgrad"]["homme"] -
                           equivalent_terms["pgrad"]["se"][pgrad])) < 1e-4


def test_equivalent_terms_dry_perturbed():
  npt = 4
  nx = 16
  nlev = 30
  model_se = models.cam_se
  model_homme = models.homme_hydrostatic
  h_grid, dims = init_quasi_uniform_grid_elem_local(nx, npt)
  v_grid_se = init_vertical_grid(cam30["hybrid_a_i"],
                                 cam30["hybrid_b_i"],
                                 cam30["p0"],
                                 model_se)
  v_grid_homme = init_vertical_grid(cam30["hybrid_a_i"],
                                    cam30["hybrid_b_i"],
                                    cam30["p0"],
                                    model_homme)

  physics_config_se = init_physics_config(model_se)
  test_config = init_baroclinic_wave_config(model_config=physics_config_se)
  model_state_se = init_baroclinic_wave_state(h_grid,
                                              v_grid_se,
                                              physics_config_se,
                                              test_config,
                                              dims,
                                              model_se,
                                              mountain=False,
                                              moist=False)
  physics_config_homme = init_physics_config(model_homme)
  test_config = init_baroclinic_wave_config(model_config=physics_config_homme)
  model_state_homme = init_baroclinic_wave_state(h_grid,
                                                 v_grid_homme,
                                                 physics_config_homme,
                                                 test_config,
                                                 dims,
                                                 model_homme,
                                                 mountain=False,
                                                 moist=False)
  scaling = (v_grid_se["hybrid_a_m"] + v_grid_se["hybrid_b_m"])[np.newaxis, np.newaxis, np.newaxis, :]
  d_mass_pert = 1000.0 * jnp.cos(h_grid["physical_coords"][:, :, :, 0])[:, :, :, np.newaxis] * scaling
  pert_u = .1 * (jnp.cos(h_grid["physical_coords"][:, :, :, 0])[:, :, :, np.newaxis, np.newaxis] *
                 jnp.cos(h_grid["physical_coords"][:, :, :, 1])[:, :, :, np.newaxis, np.newaxis])
  model_state_homme["dynamics"]["d_mass"] += d_mass_pert
  model_state_se["dynamics"]["d_mass"] += d_mass_pert
  model_state_homme["dynamics"]["horizontal_wind"] += pert_u
  model_state_se["dynamics"]["horizontal_wind"] += pert_u
  equivalent_terms = compare_equivalent_terms(model_state_se, model_state_homme,
                                              physics_config_se, physics_config_homme,
                                              v_grid_se, v_grid_homme, model_se, model_homme, h_grid)
  print(jnp.max(jnp.abs(equivalent_terms["vorticity"]["se"] - equivalent_terms["vorticity"]["homme"])))
  print(jnp.max(jnp.abs(equivalent_terms["grad_energy"]["se"] + equivalent_terms["grad_energy"]["homme"])))
  reference_pgrad_method = pressure_gradient_options.basic
  for pgrad in [pressure_gradient_options.grad_exner, pressure_gradient_options.corrected_grad_exner]:
    pgrad_mag = jnp.sqrt(physical_dot_product(equivalent_terms["pgrad"]["se"][reference_pgrad_method],
                                              equivalent_terms["pgrad"]["se"][reference_pgrad_method]))
    for k in range(nlev):
      norm_const = inner_product(pgrad_mag[:, :, :, k], pgrad_mag[:, :, :, k], h_grid)
      diff = (equivalent_terms["pgrad"]["se"][reference_pgrad_method][:, :, :, k, :] -
              equivalent_terms["pgrad"]["se"][pgrad][:, :, :, k, :])
      diff_total = (inner_product(diff[:, :, :, 0] / norm_const, diff[:, :, :, 0], h_grid) +
                    inner_product(diff[:, :, :, 1] / norm_const, diff[:, :, :, 1], h_grid))
      assert diff_total < 1e-8
