from functools import partial
from pyses._config import get_backend as _get_backend
from pyses.dynamical_cores.cam_se import thermodynamics as cam_se_thermo
from pyses.dynamical_cores.homme import thermodynamics as homme_thermo
from pyses.dynamical_cores.shallow_water_3d import thermodynamics as sw_thermo
from pyses.dynamical_cores.utils_3d import interface_to_midlevel
from pyses.dynamical_cores import model_info
_be = _get_backend()
jnp = _be.np
jit = _be.jit


@partial(jit, static_argnames=["model"])
def diagnose_midlevel_geopotential(state,
                                   h_grid,
                                   v_grid,
                                   physics_config,
                                   model):
  """
  Diagnose the mid-level geopotential from a full model state, for any model.

  Reproduces the geopotential each dynamical core uses internally in its
  tendency computation:

  * CAM-SE models integrate the hydrostatic equation from the virtual
    temperature (diagnosed from ``theta_d_d_mass`` for the ``_stable``
    variants) and the moist pressure profile.
  * Hydrostatic HOMME models integrate the hydrostatic equation from
    ``theta_v_d_mass`` and average the interface result to mid-levels.
  * Non-hydrostatic HOMME models average the prognostic interface
    geopotential ``phi_i`` to mid-levels.
  * Shallow-water models integrate the constant-density hydrostatic
    relation from the layer mass; the thermodynamics routine returns the
    layer-top interface geopotential, so half a layer thickness
    (``d_phi = d_mass`` with ``rho = 1``) is subtracted to reach mid-levels.

  Parameters
  ----------
  state : dict
      Full model state from :func:`wrap_model_state` with keys
      ``"dynamics"``, ``"static_forcing"``, and ``"tracers"``.
  h_grid : SpectralElementGrid
      Horizontal grid struct (unused; accepted for API uniformity).
  v_grid : dict[str, Array]
      Vertical grid struct from :func:`init_vertical_grid`.
  physics_config : dict
      Physics configuration dict from :func:`init_physics_config`.
  model : model_info.models
      Model identifier; static JIT argument.

  Returns
  -------
  phi_m : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Mid-level geopotential (m^2 s^-2) in the model's native convention.
  """
  dynamics = state["dynamics"]
  phi_surf = state["static_forcing"]["phi_surf"]

  if model in model_info.shallow_water_models:
    # eval_geopotential returns the layer-top interface geopotential;
    # with rho = 1 the layer thickness is d_phi = d_mass.
    phi_layer_top = sw_thermo.eval_geopotential(dynamics["d_mass"], phi_surf)
    return phi_layer_top - 0.5 * dynamics["d_mass"]

  if model in model_info.cam_se_models:
    d_mass = dynamics["d_mass"]
    moisture_species = state["tracers"]["moisture_species"]
    dry_air_species = state["tracers"]["dry_air_species"]
    R_dry = cam_se_thermo.eval_Rgas_dry(dry_air_species, physics_config)
    sum_species = cam_se_thermo.eval_sum_species(moisture_species)
    p_top = v_grid["hybrid_a_i"][0] * v_grid["reference_surface_mass"]
    if model in model_info.cam_se_stable_models:
      # theta_d form: T = (theta_d_d_mass / d_mass) * Pi_d
      cp_dry = cam_se_thermo.eval_cp_dry(dry_air_species, physics_config)
      p_int_dry = cam_se_thermo.eval_interface_pressure(d_mass, p_top)
      p_mid_dry = cam_se_thermo.eval_midlevel_pressure(p_int_dry)
      exner_dry = cam_se_thermo.eval_exner_function(p_mid_dry,
                                                    R_dry,
                                                    cp_dry,
                                                    physics_config)
      thermo_name = model_info.thermodynamic_variable_names[model]
      temperature = dynamics[thermo_name] / d_mass * exner_dry
    else:
      temperature = dynamics["T"]
    virtual_temperature = cam_se_thermo.eval_virtual_temperature(temperature,
                                                                 moisture_species,
                                                                 sum_species,
                                                                 R_dry,
                                                                 physics_config)
    d_pressure = cam_se_thermo.eval_d_pressure(d_mass, moisture_species)
    p_int = cam_se_thermo.eval_interface_pressure(d_pressure, p_top)
    p_mid = cam_se_thermo.eval_midlevel_pressure(p_int)
    return cam_se_thermo.eval_balanced_geopotential(virtual_temperature,
                                                    d_pressure,
                                                    p_mid,
                                                    R_dry,
                                                    phi_surf)

  if model in model_info.homme_models:
    if model in model_info.hydrostatic_models:
      p_mid = homme_thermo.eval_midlevel_pressure(dynamics, v_grid)
      phi_i = homme_thermo.eval_balanced_geopotential(phi_surf,
                                                      p_mid,
                                                      dynamics["theta_v_d_mass"],
                                                      physics_config)
    else:
      phi_i = dynamics["phi_i"]
    return interface_to_midlevel(phi_i)

  raise ValueError(f"Unknown model: {model}")
