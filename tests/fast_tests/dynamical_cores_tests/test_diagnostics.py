"""Tests for the model-agnostic mid-level geopotential diagnostic.

Each test builds an artificial (random but physically plausible) model state
for one dynamical core and checks that
``diagnose_midlevel_geopotential`` agrees with the manual sequence of
thermodynamics calls that the corresponding ``eval_common_variables``
performs internally.
"""
import numpy as np
import pytest

from pyses._config import get_backend as _get_backend
from pyses.dynamical_cores.model_info import (models,
                                              cam_se_models,
                                              cam_se_stable_models,
                                              shallow_water_models,
                                              hydrostatic_models,
                                              thermodynamic_variable_names)
from pyses.dynamical_cores.physics_config import init_physics_config
from pyses.dynamical_cores.mass_coordinate import init_vertical_grid
from pyses.dynamical_cores.diagnostics import diagnose_midlevel_geopotential
from pyses.dynamical_cores.cam_se import thermodynamics as cam_se_thermo
from pyses.dynamical_cores.homme import thermodynamics as homme_thermo
from pyses.dynamical_cores.shallow_water_3d import thermodynamics as sw_thermo
from pyses.dynamical_cores.utils_3d import interface_to_midlevel
from ...test_data.mass_coordinate_grids import cam30

_be = _get_backend()
jnp = _be.np
device_wrapper = _be.array

NELEM = 2
NPT = 3
NLEV = len(cam30["hybrid_a_i"]) - 1


def _artificial_state(model, physics_config, seed=0):
  """Build a synthetic model state with plausible magnitudes for ``model``."""
  rng = np.random.default_rng(seed)
  shape_m = (NELEM, NPT, NPT, NLEV)
  shape_surf = (NELEM, NPT, NPT)

  phi_surf = rng.uniform(0.0, 2.0e4, shape_surf)
  # positive layer masses roughly consistent with a 1e5 Pa column
  d_mass = rng.uniform(0.5, 1.5, shape_m) * 1.0e5 / NLEV

  dynamics = {"d_mass": device_wrapper(d_mass),
              "horizontal_wind": device_wrapper(
                  rng.uniform(-10.0, 10.0, shape_m + (2,)))}

  if model in shallow_water_models:
    tracers = {"moisture_species": {}, "tracers": {}}
  elif model in cam_se_models:
    if model in cam_se_stable_models:
      theta_d = rng.uniform(280.0, 340.0, shape_m)
      dynamics[thermodynamic_variable_names[model]] = device_wrapper(theta_d * d_mass)
    else:
      dynamics["T"] = device_wrapper(rng.uniform(220.0, 300.0, shape_m))
    moisture_species = {"water_vapor": device_wrapper(
        rng.uniform(0.0, 2.0e-2, shape_m))}
    # random positive dry-air mass fractions normalised to sum to one
    species_names = list(physics_config["dry_air_species_Rgas"].keys())
    fracs = rng.uniform(0.2, 1.0, (len(species_names),) + shape_m)
    fracs /= np.sum(fracs, axis=0)
    dry_air_species = {name: device_wrapper(fracs[species_idx])
                       for species_idx, name in enumerate(species_names)}
    tracers = {"moisture_species": moisture_species,
               "tracers": {},
               "dry_air_species": dry_air_species}
  else:  # HOMME
    theta_v = rng.uniform(280.0, 340.0, shape_m)
    dynamics["theta_v_d_mass"] = device_wrapper(theta_v * d_mass)
    if model not in hydrostatic_models:
      # monotone interface geopotential built from positive layer thicknesses
      d_phi = rng.uniform(500.0, 1500.0, shape_m)
      phi_i_above = np.flip(np.cumsum(np.flip(d_phi, axis=-1), axis=-1), axis=-1)
      phi_i = np.concatenate((phi_i_above + phi_surf[:, :, :, np.newaxis],
                              phi_surf[:, :, :, np.newaxis]), axis=-1)
      dynamics["phi_i"] = device_wrapper(phi_i)
      dynamics["w_i"] = device_wrapper(
          rng.uniform(-1.0, 1.0, (NELEM, NPT, NPT, NLEV + 1)))
    tracers = {"moisture_species": {}, "tracers": {}}

  static_forcing = {"phi_surf": device_wrapper(phi_surf)}
  return {"dynamics": dynamics,
          "static_forcing": static_forcing,
          "tracers": tracers}


def _manual_midlevel_geopotential(state, v_grid, physics_config, model):
  """Replicate the geopotential calls each model's eval_common_variables makes."""
  dynamics = state["dynamics"]
  phi_surf = state["static_forcing"]["phi_surf"]

  if model in shallow_water_models:
    # eval_geopotential returns layer-top interface values (phi_i, not phi_m);
    # shift down half a layer (d_phi = d_mass under the rho = 1 convention)
    return (sw_thermo.eval_geopotential(dynamics["d_mass"], phi_surf) -
            0.5 * dynamics["d_mass"])

  if model in cam_se_models:
    d_mass = dynamics["d_mass"]
    moisture_species = state["tracers"]["moisture_species"]
    dry_air_species = state["tracers"]["dry_air_species"]
    R_dry = cam_se_thermo.eval_Rgas_dry(dry_air_species, physics_config)
    sum_species = cam_se_thermo.eval_sum_species(moisture_species)
    p_top = v_grid["hybrid_a_i"][0] * v_grid["reference_surface_mass"]
    if model in cam_se_stable_models:
      # explicit_terms_theta.eval_common_variables
      cp_dry = cam_se_thermo.eval_cp_dry(dry_air_species, physics_config)
      p_int_dry = cam_se_thermo.eval_interface_pressure(d_mass, p_top)
      p_mid_dry = cam_se_thermo.eval_midlevel_pressure(p_int_dry)
      exner_dry = cam_se_thermo.eval_exner_function(p_mid_dry, R_dry, cp_dry,
                                                    physics_config)
      theta_d = dynamics[thermodynamic_variable_names[model]] / d_mass
      temperature = theta_d * exner_dry
    else:
      # explicit_terms.eval_common_variables
      temperature = dynamics["T"]
    virtual_temperature = cam_se_thermo.eval_virtual_temperature(
        temperature, moisture_species, sum_species, R_dry, physics_config)
    d_pressure = sum_species * d_mass
    p_int = cam_se_thermo.eval_interface_pressure(d_pressure, p_top)
    p_mid = cam_se_thermo.eval_midlevel_pressure(p_int)
    return cam_se_thermo.eval_balanced_geopotential(virtual_temperature,
                                                    d_pressure,
                                                    p_mid,
                                                    R_dry,
                                                    phi_surf)

  # HOMME: homme/explicit_terms.eval_common_variables
  if model in hydrostatic_models:
    p_mid = homme_thermo.eval_midlevel_pressure(dynamics, v_grid)
    phi_i = homme_thermo.eval_balanced_geopotential(phi_surf,
                                                    p_mid,
                                                    dynamics["theta_v_d_mass"],
                                                    physics_config)
  else:
    phi_i = dynamics["phi_i"]
  return interface_to_midlevel(phi_i)


@pytest.mark.parametrize("model", list(models), ids=lambda m: m.name)
def test_diagnose_midlevel_geopotential(model):
  physics_config = init_physics_config(model)
  v_grid = init_vertical_grid(cam30["hybrid_a_i"],
                              cam30["hybrid_b_i"],
                              cam30["p0"],
                              model)
  state = _artificial_state(model, physics_config, seed=0)

  phi_helper = diagnose_midlevel_geopotential(state,
                                              None,
                                              v_grid,
                                              physics_config,
                                              model)
  phi_manual = _manual_midlevel_geopotential(state, v_grid, physics_config, model)

  assert phi_helper.shape == phi_manual.shape
  assert jnp.allclose(phi_helper, phi_manual, rtol=1e-10, atol=1e-6)


@pytest.mark.parametrize("model",
                         [m for m in models if m not in hydrostatic_models],
                         ids=lambda m: m.name)
def test_nonhydrostatic_uses_prognostic_phi(model):
  """Non-hydrostatic models must report the prognostic phi_i, not a balanced one."""
  physics_config = init_physics_config(model)
  v_grid = init_vertical_grid(cam30["hybrid_a_i"],
                              cam30["hybrid_b_i"],
                              cam30["p0"],
                              model)
  state = _artificial_state(model, physics_config, seed=1)
  # perturb phi_i away from any balanced profile; the diagnostic must follow it
  state["dynamics"]["phi_i"] = state["dynamics"]["phi_i"] + 123.0
  phi_helper = diagnose_midlevel_geopotential(state,
                                              None,
                                              v_grid,
                                              physics_config,
                                              model)
  assert jnp.allclose(phi_helper,
                      interface_to_midlevel(state["dynamics"]["phi_i"]),
                      rtol=1e-10, atol=1e-6)
