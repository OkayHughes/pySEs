from ..config import jnp, jit, flip, np, do_mpi_communication, vmap_1d_apply
from functools import partial
from ..operations_2d.operators import horizontal_gradient
from ..mpi.global_assembly import project_scalar_global
from ..operations_2d.local_assembly import project_scalar
from ..model_info import (f_plane_models,
                          deep_atmosphere_models,
                          thermodynamic_variable_names,
                          hydrostatic_models,
                          cam_se_models,
                          moist_mixing_ratio_models)
from .mass_coordinate import surface_mass_to_d_mass, surface_mass_to_midlevel_mass
from .homme.thermodynamics import eval_balanced_geopotential, eval_midlevel_pressure
from .utils_3d import interface_to_delta, cumulative_sum, phi_to_g
from .vertical_remap import zerroukat_remap
from ..mpi.global_communication import global_sum


@partial(jit, static_argnames=["is_dry_air_species"])
def sum_tracers(state1,
                state2,
                fold_coeff1,
                fold_coeff2,
                is_dry_air_species=False):
  """
  [Description]

  Parameters
  ----------
  [first] : array_like
      the 1st param name `first`
  second :
      the 2nd param
  third : {'value', 'other'}, optional
      the 3rd param, by default 'value'

  Returns
  -------
  string
      a value in a string

  Raises
  ------
  KeyError
      when a key error
  """
  state_out = {}
  for tracer_name in state1.keys():
    state_out[tracer_name] = fold_coeff1 * state1[tracer_name] + fold_coeff2 * state2[tracer_name]
  return state_out


@partial(jit, static_argnames=["model"])
def advance_tracers(tracer_states,
                    coeffs,
                    model):
  moisture_species = sum_tracers(tracer_states[0]["moisture_species"],
                                 tracer_states[1]["moisture_species"],
                                 coeffs[0],
                                 coeffs[1])
  passiveish_tracers = sum_tracers(tracer_states[0]["tracers"],
                                   tracer_states[1]["tracers"],
                                   coeffs[0],
                                   coeffs[1])
  if model in cam_se_models:
    dry_air_species = sum_tracers(tracer_states[0]["dry_air_species"],
                                  tracer_states[1]["dry_air_species"],
                                  coeffs[0],
                                  coeffs[1],
                                  is_dry_air_species=True)
  else:
    dry_air_species = None

  for coeff_idx in range(2, len(tracer_states)):
    moisture_species = sum_tracers(moisture_species,
                                   tracer_states[coeff_idx]["moisture_species"],
                                   coeffs[0],
                                   coeffs[1])
    passiveish_tracers = sum_tracers(passiveish_tracers,
                                     tracer_states[coeff_idx]["tracers"],
                                     1.0,
                                     coeffs[coeff_idx])
    if model in cam_se_models:
      dry_air_species = sum_tracers(dry_air_species,
                                    tracer_states[coeff_idx]["dry_air_species"],
                                    1.0,
                                    coeffs[coeff_idx],
                                    is_dry_air_species=True)

  return wrap_tracers(moisture_species,
                      passiveish_tracers,
                      model,
                      dry_air_species=dry_air_species)


@jit
def wrap_tracer_consist_dynamics(u_d_mass):
  """
  [Description]

  Parameters
  ----------
  [first] : array_like
      the 1st param name `first`
  second :
      the 2nd param
  third : {'value', 'other'}, optional
      the 3rd param, by default 'value'

  Returns
  -------
  string
      a value in a string

  Raises
  ------
  KeyError
      when a key error
  """
  return {"u_d_mass_avg": u_d_mass}


@jit
def wrap_tracer_consist_hypervis(d_mass_val,
                                 d_mass_tend):
  """
  [Description]

  Parameters
  ----------
  [first] : array_like
      the 1st param name `first`
  second :
      the 2nd param
  third : {'value', 'other'}, optional
      the 3rd param, by default 'value'

  Returns
  -------
  string
      a value in a string

  Raises
  ------
  KeyError
      when a key error
  """
  return {"d_mass_hypervis_tend": d_mass_tend,
          "d_mass_hypervis_avg": d_mass_val}


@jit
def sum_consistency_struct(struct_1, struct_2, coeff_1, coeff_2):
  res = {}
  for field in struct_1.keys():
    res[field] = struct_1[field] * coeff_1 + struct_2[field] * coeff_2
  return res


@partial(jit, static_argnames=["num_lev", "model"])
def remap_tracers(dynamics,
                  tracers,
                  v_grid,
                  num_lev,
                  model):
  tracer_list = []
  ct = 0
  if model in cam_se_models:
    dry_air_species = {}
    for species_name in tracers["dry_air_species"].keys():
      dry_air_species[species_name] = ct
      tracer_list.append(tracers["dry_air_species"][species_name])
      ct += 1
  else:
    dry_air_species = None
  moisture_species = {}
  for species_name in tracers["moisture_species"].keys():
    moisture_species[species_name] = ct
    tracer_list.append(tracers["moisture_species"][species_name])
    ct += 1
  tracers_new = {}
  for species_name in tracers["tracers"].keys():
    tracers_new[species_name] = ct
    tracer_list.append(tracers["tracers"][species_name])
    ct += 1
  pi_surf = dynamics_to_surface_mass(dynamics, v_grid)
  d_mass_ref = surface_mass_to_d_mass(pi_surf,
                                      v_grid)
  tracer_mass = jnp.stack(tracer_list, axis=-1) * dynamics["d_mass"][:, :, :, :, jnp.newaxis]
  tracers_out = zerroukat_remap(tracer_mass, dynamics["d_mass"], d_mass_ref, num_lev, filter=True) / d_mass_ref[:, :, :, :, jnp.newaxis]
  ct = 0
  if model in cam_se_models:
    for species_name in dry_air_species.keys():
      dry_air_species[species_name] = tracers_out[:, :, :, :, dry_air_species[species_name]]
  else:
    dry_air_species = None
  for species_name in moisture_species.keys():
    moisture_species[species_name] = tracers_out[:, :, :, :, moisture_species[species_name]]
  for species_name in tracers_new.keys():
    tracers_new[species_name] = tracers[:, :, :, :, tracers_new[species_name]]
  return wrap_tracers(moisture_species,
                      tracers_new,
                      model,
                      dry_air_species=dry_air_species)


@partial(jit, static_argnames=["model"])
def wrap_dynamics(horizontal_wind,
                  thermodynamic_variable,
                  d_mass,
                  model,
                  phi_i=None,
                  w_i=None):
  """
  [Description]

  Parameters
  ----------
  [first] : array_like
      the 1st param name `first`
  second :
      the 2nd param
  third : {'value', 'other'}, optional
      the 3rd param, by default 'value'

  Returns
  -------
  string
      a value in a string

  Raises
  ------
  KeyError
      when a key error
  """
  state = {"horizontal_wind": horizontal_wind,
           thermodynamic_variable_names[model]: thermodynamic_variable,
           "d_mass": d_mass
           }
  if phi_i is not None:
    state["phi_i"] = phi_i
  if w_i is not None:
    state["w_i"] = w_i
  return state


@jit
def wrap_model_state(dynamics,
                     static_forcing,
                     tracers):
  return {"dynamics": dynamics,
          "static_forcing": static_forcing,
          "tracers": tracers}


@partial(jit, static_argnames=["model"])
def copy_dynamics(dynamics,
                  model):
  if model not in hydrostatic_models:
    phi_i = dynamics["phi_i"]
    w_i = dynamics["w_i"]
  else:
    phi_i = None
    w_i = None
  return wrap_dynamics(jnp.copy(dynamics["horizontal_wind"]),
                       jnp.copy(dynamics[thermodynamic_variable_names[model]]),
                       jnp.copy(dynamics["d_mass"]),
                       model,
                       phi_i=phi_i,
                       w_i=w_i)


@partial(jit, static_argnames=["model"])
def copy_tracers(tracers,
                 model):
  if model in cam_se_models:
    dry_air_species = {}
    for species_name in tracers["dry_air_species"].keys():
      dry_air_species[species_name] = jnp.copy(tracers["dry_air_species"][species_name])
  else:
    dry_air_species = None
  moisture_species = {}
  for species_name in tracers["moisture_species"].keys():
    moisture_species[species_name] = jnp.copy(tracers["moisture_species"][species_name])
  tracers_new = {}
  for species_name in tracers["tracers"].keys():
    tracers_new[species_name] = jnp.copy(tracers["tracers"][species_name])
  return wrap_tracers(moisture_species,
                      tracers_new,
                      model,
                      dry_air_species=dry_air_species)


@partial(jit, static_argnames=["model"])
def copy_model_state(state,
                     model):
  return wrap_model_state(copy_dynamics(state["dynamics"], model),
                          state["static_forcing"],
                          copy_tracers(state["tracers"], model))


@partial(jit, static_argnames=["model"])
def _wrap_tracer_like(moisture_species,
                 tracers,
                 model,
                 dry_air_species=None):
  tracer_struct = {"moisture_species": moisture_species,
                   "tracers": tracers}
  if dry_air_species is not None:
    tracer_struct["dry_air_species"] = dry_air_species
  if model in moist_mixing_ratio_models:
    tracer_struct["moist_mixing_ratio"] = 1.0
  else:
    tracer_struct["dry_mixing_ratio"] = 1.0
  return tracer_struct


@partial(jit, static_argnames=["model"])
def wrap_tracers(moisture_species,
                 tracers,
                 model,
                 dry_air_species=None):
  tracer_struct = _wrap_tracer_like(moisture_species,
                                    tracers,
                                    model,
                                    dry_air_species=dry_air_species)
  if model in moist_mixing_ratio_models:
    tracer_struct["moist_mixing_ratio"] = 1.0
  else:
    tracer_struct["dry_mixing_ratio"] = 1.0
  return tracer_struct


@partial(jit, static_argnames=["model"])
def wrap_tracer_mass(moisture_species_mass,
                     tracers_mass,
                     model,
                     dry_air_species_mass=None):
  tracer_mass = _wrap_tracer_like(moisture_species_mass,
                                    tracers_mass,
                                    model,
                                    dry_air_species=dry_air_species_mass)
  tracer_mass["mass_quantity"] = 1.0
  return tracer_mass


@jit
def wrap_static_forcing(phi_surf,
                        grad_phi_surf,
                        coriolis_param,
                        nontrad_coriolis_param=None):
  static_forcing = {"phi_surf": phi_surf,
                    "grad_phi_surf": grad_phi_surf,
                    "coriolis_param": coriolis_param}
  if nontrad_coriolis_param is not None:
    static_forcing["nontrad_coriolis_param"] = nontrad_coriolis_param
  return static_forcing


def init_static_forcing(phi_surf,
                        h_grid,
                        physics_config,
                        dims,
                        model,
                        f_plane_center=jnp.pi / 4.0):
  grad_phi_surf_discont = horizontal_gradient(phi_surf, h_grid, a=physics_config["radius_earth"])
  if do_mpi_communication:
    grad_phi_surf = jnp.stack([project_scalar_global([grad_phi_surf_discont[:, :, :, 0]], h_grid, dims)[0],
                              project_scalar_global([grad_phi_surf_discont[:, :, :, 1]], h_grid, dims)[0]], axis=-1)
  else:
    grad_phi_surf = jnp.stack([project_scalar(grad_phi_surf_discont[:, :, :, 0], h_grid, dims),
                              project_scalar(grad_phi_surf_discont[:, :, :, 1], h_grid, dims)], axis=-1)
  if model in f_plane_models:
    coriolis_param = 2.0 * physics_config["angular_freq_earth"] * (jnp.sin(f_plane_center) *
                                                                   jnp.ones_like(h_grid["physical_coords"][:, :, :, 0]))
  else:
    coriolis_param = 2.0 * physics_config["angular_freq_earth"] * jnp.sin(h_grid["physical_coords"][:, :, :, 0])
  if model in deep_atmosphere_models:
    nontrad_coriolis_param = 2.0 * physics_config["angular_freq_earth"] * jnp.cos(h_grid["physical_coords"][:, :, :, 0])
  else:
    nontrad_coriolis_param = None
  return wrap_static_forcing(phi_surf, grad_phi_surf, coriolis_param, nontrad_coriolis_param=nontrad_coriolis_param)


@partial(jit, static_argnames=["dims", "model"])
def project_dynamics(dynamics_in,
                     h_grid,
                     dims,
                     model):
  """
  [Description]

  Parameters
  ----------
  [first] : array_like
      the 1st param name `first`
  second :
      the 2nd param
  third : {'value', 'other'}, optional
      the 3rd param, by default 'value'

  Returns
  -------
  string
      a value in a string

  Raises
  ------
  KeyError
      when a key error
  """
  u_cont = project_scalar_3d(dynamics_in["horizontal_wind"][:, :, :, :, 0], h_grid, dims)
  v_cont = project_scalar_3d(dynamics_in["horizontal_wind"][:, :, :, :, 1], h_grid, dims)
  thermo_var_cont = project_scalar_3d(dynamics_in[thermodynamic_variable_names[model]][:, :, :, :], h_grid, dims)
  d_mass_cont = project_scalar_3d(dynamics_in["d_mass"][:, :, :, :], h_grid, dims)
  if model not in hydrostatic_models:
    w_i_cont = project_scalar_3d(dynamics_in["w_i"], h_grid, dims)
    phi_i_cont = project_scalar_3d(dynamics_in["phi_i"], h_grid, dims)
  else:
    phi_i_cont = None
    w_i_cont = None
  return wrap_dynamics(jnp.stack((u_cont, v_cont), axis=-1),
                       thermo_var_cont,
                       d_mass_cont,
                       model,
                       phi_i=phi_i_cont,
                       w_i=w_i_cont)


@partial(jit, static_argnames=["dims"])
def project_scalar_3d(variable,
                      h_grid,
                      dims):
  if do_mpi_communication:
    variable_cont = project_scalar_global([variable],
                                          h_grid,
                                          dims,
                                          two_d=False)[0]
  else:
    op_2d = partial(project_scalar, grid=h_grid, dims=dims)
    variable_cont = vmap_1d_apply(op_2d, variable, -1, -1)
  return variable_cont


@jit
def dynamics_to_surface_mass(state_in,
                             v_grid):
  """
  [Description]

  Parameters
  ----------
  [first] : array_like
      the 1st param name `first`
  second :
      the 2nd param
  third : {'value', 'other'}, optional
      the 3rd param, by default 'value'

  Returns
  -------
  string
      a value in a string

  Raises
  ------
  KeyError
      when a key error
  """
  return jnp.sum(state_in["d_mass"], axis=-1) + v_grid["hybrid_a_i"][0] * v_grid["reference_surface_mass"]


@partial(jit, static_argnames=["num_lev", "model"])
def remap_dynamics(dynamics_in,
                   static_forcing,
                   v_grid,
                   physics_config,
                   num_lev,
                   model):
  """
  [Description]

  Parameters
  ----------
  [first] : array_like
      the 1st param name `first`
  second :
      the 2nd param
  third : {'value', 'other'}, optional
      the 3rd param, by default 'value'

  Returns
  -------
  string
      a value in a string

  Raises
  ------
  KeyError
      when a key error
  """
  pi_surf = dynamics_to_surface_mass(dynamics_in, v_grid)
  d_mass_ref = surface_mass_to_d_mass(pi_surf,
                                      v_grid)
  d_mass = dynamics_in["d_mass"]
  u_model = dynamics_in["horizontal_wind"][:, :, :, :, 0] * d_mass
  v_model = dynamics_in["horizontal_wind"][:, :, :, :, 1] * d_mass
  if model in cam_se_models:
    thermo_model = dynamics_in["T"] * d_mass
  else:
    thermo_model = dynamics_in["theta_v_d_mass"]
  if model not in hydrostatic_models:
    p_mid = eval_midlevel_pressure(dynamics_in, v_grid)
    phi_ref = eval_balanced_geopotential(static_forcing["phi_surf"],
                                         p_mid,
                                         dynamics_in["theta_v_d_mass"],
                                         physics_config)
    phi_pert = dynamics_in["phi_i"] - phi_ref
    d_phi = interface_to_delta(phi_pert)
    dw = interface_to_delta(dynamics_in["w_i"])
    Qdp = jnp.stack([u_model, v_model, thermo_model,
                     d_phi, dw], axis=-1)
  else:
    Qdp = jnp.stack([u_model, v_model, thermo_model], axis=-1)
  Qdp_out = zerroukat_remap(Qdp, dynamics_in["d_mass"], d_mass_ref, num_lev, filter=True)
  u_remap = jnp.stack((Qdp_out[:, :, :, :, 0] / d_mass_ref,
                       Qdp_out[:, :, :, :, 1] / d_mass_ref), axis=-1)
  if model in cam_se_models:
    thermo_remap = Qdp_out[:, :, :, :, 2] / d_mass_ref
  else:
    thermo_remap = Qdp_out[:, :, :, :, 2]

  if model not in hydrostatic_models:
    p_mid = surface_mass_to_midlevel_mass(pi_surf, v_grid)
    phi_ref_new = eval_balanced_geopotential(static_forcing["phi_surf"],
                                             p_mid,
                                             thermo_remap,
                                             physics_config)
    phi_i_remap = cumulative_sum(-Qdp_out[:, :, :, :, 3], jnp.zeros_like(static_forcing["phi_surf"])) + phi_ref_new
    w_i_surf = ((u_remap[:, :, :, -1, 0] * static_forcing["grad_phi_surf"][:, :, :, 0] +
                 u_remap[:, :, :, -1, 1] * static_forcing["grad_phi_surf"][:, :, :, 1]) /
                phi_to_g(static_forcing["phi_surf"], physics_config, model))
    w_i_upper = flip(jnp.cumsum(-flip(Qdp[:, :, :, :, 4], -1), axis=-1), -1) + dynamics_in["w_i"][:, :, :, -1:]
    w_i_remap = jnp.concatenate((w_i_upper, w_i_surf[:, :, :, np.newaxis]), axis=-1)
  else:
    phi_i_remap = None
    w_i_remap = None
  return wrap_dynamics(u_remap,
                       thermo_remap,
                       d_mass_ref,
                       model,
                       phi_i=phi_i_remap,
                       w_i=w_i_remap)


@partial(jit, static_argnames=["model"])
def sum_dynamics(state1,
                 state2,
                 fold_coeff1,
                 fold_coeff2,
                 model):
  """
  [Description]

  Parameters
  ----------
  [first] : array_like
      the 1st param name `first`
  second :
      the 2nd param
  third : {'value', 'other'}, optional
      the 3rd param, by default 'value'

  Returns
  -------
  string
      a value in a string

  Raises
  ------
  KeyError
      when a key error
  """
  if model not in hydrostatic_models:
    phi_i = state1["phi_i"] * fold_coeff1 + state2["phi_i"] * fold_coeff2
    w_i = state1["w_i"] * fold_coeff1 + state2["w_i"] * fold_coeff2
  else:
    phi_i = None
    w_i = None
  thermo_var_name = thermodynamic_variable_names[model]
  return wrap_dynamics(state1["horizontal_wind"] * fold_coeff1 + state2["horizontal_wind"] * fold_coeff2,
                       state1[thermo_var_name] * fold_coeff1 + state2[thermo_var_name] * fold_coeff2,
                       state1["d_mass"] * fold_coeff1 + state2["d_mass"] * fold_coeff2,
                       model,
                       phi_i=phi_i,
                       w_i=w_i)


@partial(jit, static_argnames=["model"])
def sum_dynamics_series(states,
                        coeffs,
                        model):
  """
  [Description]

  Parameters
  ----------
  [first] : array_like
      the 1st param name `first`
  second :
      the 2nd param
  third : {'value', 'other'}, optional
      the 3rd param, by default 'value'

  Returns
  -------
  string
      a value in a string

  Raises
  ------
  KeyError
      when a key error
  """
  state_out = sum_dynamics(states[0],
                           states[1],
                           coeffs[0],
                           coeffs[1], model)
  for coeff_idx in range(2, len(states)):
    state_out = sum_dynamics(state_out,
                             states[coeff_idx],
                             1.0,
                             coeffs[coeff_idx], model)
  return state_out


def apply_mask(field, h_grid):
  mask = h_grid["ghost_mask"]
  shape = list(h_grid["ghost_mask"].shape) + (field.ndim - mask.ndim) * [1]
  return jnp.where(mask.reshape(shape) > 0.5, field, 0.0)


def check_dynamics_nan(dynamics,
                       h_grid,
                       model):
  """
  [Description]

  Parameters
  ----------
  [first] : array_like
      the 1st param name `first`
  second :
      the 2nd param
  third : {'value', 'other'}, optional
      the 3rd param, by default 'value'

  Returns
  -------
  string
      a value in a string

  Raises
  ------
  KeyError
      when a key error
  """
  is_nan = False
  fields = ["horizontal_wind", thermodynamic_variable_names[model], "d_mass"]
  if model not in hydrostatic_models:
    fields += ["w_i", "phi_i"]
  for field in fields:
    is_nan = is_nan or jnp.any(jnp.isnan(apply_mask(dynamics[field], h_grid)))
  is_nan = int(is_nan)
  return global_sum(is_nan) > 0


def check_tracers_nan(tracers,
                      h_grid,
                      model):
  is_nan = False
  for field_name in tracers["moisture_species"].keys():
    is_nan = is_nan or jnp.any(jnp.isnan(apply_mask(tracers["moisture_species"][field_name], h_grid)))
  for field_name in tracers["tracers"].keys():
    is_nan = is_nan or jnp.any(jnp.isnan(apply_mask(tracers["tracers"][field_name], h_grid)))
  if model in cam_se_models:
    for field_name in tracers["dry_air_species"].keys():
      is_nan = is_nan or jnp.any(jnp.isnan(apply_mask(tracers["dry_air_species"][field_name], h_grid)))
  is_nan = int(is_nan)
  return global_sum(is_nan) > 0
