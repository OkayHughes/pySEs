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
  Compute a weighted sum of two tracer species dicts field-by-field.

  Parameters
  ----------
  state1 : dict[str, Array]
      First tracer species dict (e.g. ``tracers["moisture_species"]``).
  state2 : dict[str, Array]
      Second tracer species dict with the same keys as ``state1``.
  fold_coeff1 : float
      Scalar weight for ``state1``.
  fold_coeff2 : float
      Scalar weight for ``state2``.
  is_dry_air_species : bool, optional
      Currently unused; reserved for dry-air species handling (default: False).

  Returns
  -------
  state_out : dict[str, Array]
      Field-wise weighted sum ``fold_coeff1 * state1 + fold_coeff2 * state2``.
  """
  state_out = {}
  for tracer_name in state1.keys():
    state_out[tracer_name] = fold_coeff1 * state1[tracer_name] + fold_coeff2 * state2[tracer_name]
  return state_out


@partial(jit, static_argnames=["model"])
def advance_tracers(tracer_states,
                    coeffs,
                    model):
  """
  Compute a weighted linear combination of a list of tracer states.

  Parameters
  ----------
  tracer_states : list[dict]
      List of tracer state dicts, each containing ``"moisture_species"``,
      ``"tracers"``, and (for CAM-SE models) ``"dry_air_species"`` sub-dicts.
  coeffs : sequence[float]
      Scalar weights, one per element of ``tracer_states``.  The first two
      states are combined with ``coeffs[0]`` and ``coeffs[1]``; subsequent
      states are folded in with their corresponding coefficient.
  model : str
      Model identifier; determines whether ``"dry_air_species"`` is handled.

  Returns
  -------
  tracers : dict
      Assembled tracer state dict produced by :func:`wrap_tracers` containing
      the field-wise weighted sum across all input states.
  """
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
  Pack the tracer-consistency flux array into its named struct.

  Parameters
  ----------
  u_d_mass : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx, 2], Float]
      Mass-weighted horizontal wind ``u * d_mass`` accumulated during the
      dynamics sub-steps; used to transport tracers in a manner consistent
      with the dynamical-core mass update.

  Returns
  -------
  consist_dynamics : dict[str, Array]
      Dict ``{"u_d_mass_avg": u_d_mass}`` consumed by
      :func:`advance_tracers_rk2` and related tracer-transport routines.
  """
  return {"u_d_mass_avg": u_d_mass}


@jit
def wrap_tracer_consist_hypervis(d_mass_val,
                                 d_mass_tend):
  """
  Pack hyperviscosity consistency quantities into their named struct.

  Parameters
  ----------
  d_mass_val : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Layer-mass field used as the reference state for the hyperviscosity
      tracer-consistency correction (typically the time-averaged ``d_mass``).
  d_mass_tend : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Hyperviscosity tendency applied to ``d_mass``; communicated to the
      tracer solver so that tracers experience the same mass diffusion as
      the dynamics.

  Returns
  -------
  consist_hypervis : dict[str, Array]
      Dict with keys ``"d_mass_hypervis_tend"`` and ``"d_mass_hypervis_avg"``
      consumed by the tracer hyperviscosity routines.
  """
  return {"d_mass_hypervis_tend": d_mass_tend,
          "d_mass_hypervis_avg": d_mass_val}


@jit
def sum_consistency_struct(struct_1, struct_2, coeff_1, coeff_2):
  """
  Compute a weighted field-wise sum of two tracer-consistency structs.

  Parameters
  ----------
  struct_1 : dict[str, Array]
      First consistency struct (e.g. from :func:`wrap_tracer_consist_dynamics`
      or :func:`wrap_tracer_consist_hypervis`).
  struct_2 : dict[str, Array]
      Second consistency struct with the same keys as ``struct_1``.
  coeff_1 : float
      Scalar weight applied to ``struct_1``.
  coeff_2 : float
      Scalar weight applied to ``struct_2``.

  Returns
  -------
  result : dict[str, Array]
      Field-wise ``coeff_1 * struct_1 + coeff_2 * struct_2``.
  """
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
  """
  Vertically remap all tracer species to the reference hybrid-coordinate levels.

  Each tracer mixing ratio is converted to a mass quantity (``q * d_mass``),
  remapped conservatively via :func:`zerroukat_remap`, and then divided by
  the reference layer mass to recover the remapped mixing ratio.

  Parameters
  ----------
  dynamics : dict
      Dynamics state dict containing ``"d_mass"`` for the current model levels.
  tracers : dict
      Tracer state dict with sub-dicts ``"moisture_species"``, ``"tracers"``,
      and (for CAM-SE models) ``"dry_air_species"``.
  v_grid : dict[str, Array]
      Vertical grid struct from :func:`init_vertical_grid`.
  num_lev : int
      Number of vertical levels; used as a static argument for JIT tracing.
  model : str
      Model identifier; determines whether ``"dry_air_species"`` is remapped.

  Returns
  -------
  tracers_remapped : dict
      Tracer state dict assembled by :func:`wrap_tracers` with all species
      remapped onto the reference ``d_mass_ref`` levels.
  """
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
  Assemble the dynamics state dict from its prognostic fields.

  The thermodynamic variable is stored under the model-specific key given by
  ``thermodynamic_variable_names[model]`` (e.g. ``"T"`` for CAM-SE,
  ``"theta_v_d_mass"`` for HOMME).  The optional non-hydrostatic fields
  ``phi_i`` and ``w_i`` are included only when provided.

  Parameters
  ----------
  horizontal_wind : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx, 2], Float]
      Contra-variant horizontal wind components ``(u, v)``.
  thermodynamic_variable : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Model thermodynamic variable (temperature or virtual potential temperature
      times layer mass, depending on ``model``).
  d_mass : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Dry-air layer mass (pressure thickness in Pa).
  model : str
      Model identifier; selects the thermodynamic variable name and whether
      non-hydrostatic fields are expected.
  phi_i : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx+1], Float], optional
      Interface geopotential; included only for non-hydrostatic models.
  w_i : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx+1], Float], optional
      Interface vertical velocity; included only for non-hydrostatic models.

  Returns
  -------
  dynamics : dict[str, Array]
      State dict with keys ``"horizontal_wind"``, the model thermodynamic key,
      ``"d_mass"``, and optionally ``"phi_i"`` and ``"w_i"``.
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
  """
  Assemble the top-level model state dict.

  Parameters
  ----------
  dynamics : dict[str, Array]
      Prognostic dynamics state from :func:`wrap_dynamics`.
  static_forcing : dict[str, Array]
      Time-invariant forcing from :func:`init_static_forcing` (surface
      geopotential, its gradient, and Coriolis parameters).
  tracers : dict
      Tracer state from :func:`wrap_tracers`.

  Returns
  -------
  state : dict
      Dict with keys ``"dynamics"``, ``"static_forcing"``, and ``"tracers"``.
  """
  return {"dynamics": dynamics,
          "static_forcing": static_forcing,
          "tracers": tracers}


@partial(jit, static_argnames=["model"])
def copy_dynamics(dynamics,
                  model):
  """
  Deep-copy a dynamics state dict, allocating new arrays for each field.

  Parameters
  ----------
  dynamics : dict[str, Array]
      Dynamics state from :func:`wrap_dynamics`.
  model : str
      Model identifier; determines whether ``"phi_i"`` and ``"w_i"`` are copied.

  Returns
  -------
  dynamics_copy : dict[str, Array]
      New dynamics dict with independent copies of all arrays.
  """
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
  """
  Deep-copy a tracer state dict, allocating new arrays for each species.

  Parameters
  ----------
  tracers : dict
      Tracer state from :func:`wrap_tracers` with sub-dicts
      ``"moisture_species"``, ``"tracers"``, and optionally
      ``"dry_air_species"`` (CAM-SE models).
  model : str
      Model identifier; determines whether ``"dry_air_species"`` is copied.

  Returns
  -------
  tracers_copy : dict
      New tracer state dict with independent copies of all species arrays.
  """
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
  """
  Deep-copy the full model state (dynamics + static forcing + tracers).

  Parameters
  ----------
  state : dict
      Top-level model state from :func:`wrap_model_state`.
  model : str
      Model identifier forwarded to :func:`copy_dynamics` and
      :func:`copy_tracers`.

  Returns
  -------
  state_copy : dict
      New model state dict with independent copies of dynamics and tracers.
      The ``"static_forcing"`` sub-dict is shared (not copied) since it is
      time-invariant.
  """
  return wrap_model_state(copy_dynamics(state["dynamics"], model),
                          state["static_forcing"],
                          copy_tracers(state["tracers"], model))


@partial(jit, static_argnames=["model"])
def _wrap_tracer_like(moisture_species,
                 tracers,
                 model,
                 dry_air_species=None):
  """
  Build a tracer-like dict and attach the mixing-ratio convention flag.

  This is the shared implementation used by both :func:`wrap_tracers` and
  :func:`wrap_tracer_mass`.  It sets either ``"moist_mixing_ratio"`` or
  ``"dry_mixing_ratio"`` to ``1.0`` to indicate the convention used by
  ``model``.

  Parameters
  ----------
  moisture_species : dict[str, Array]
      Moisture mixing-ratio (or mass) fields keyed by species name.
  tracers : dict[str, Array]
      Passive tracer fields keyed by tracer name.
  model : str
      Model identifier; selects the mixing-ratio convention.
  dry_air_species : dict[str, Array] or None, optional
      Dry-air species fields (CAM-SE models only).

  Returns
  -------
  tracer_struct : dict
      Dict with ``"moisture_species"``, ``"tracers"``, optionally
      ``"dry_air_species"``, and a mixing-ratio convention key.
  """
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
  """
  Assemble the tracer state dict from mixing-ratio species dicts.

  Parameters
  ----------
  moisture_species : dict[str, Array]
      Moisture mixing-ratio fields keyed by species name.
  tracers : dict[str, Array]
      Passive tracer fields keyed by tracer name.
  model : str
      Model identifier; selects the mixing-ratio convention and whether
      ``"dry_air_species"`` is included.
  dry_air_species : dict[str, Array] or None, optional
      Dry-air species mixing-ratio fields (CAM-SE models only).

  Returns
  -------
  tracer_struct : dict
      Tracer state dict suitable for use in the full model state.
  """
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
  """
  Assemble the tracer *mass* dict (``q * d_mass``) from per-species arrays.

  Identical in structure to :func:`wrap_tracers` but marks the result with
  ``"mass_quantity": 1.0`` instead of a mixing-ratio flag, so downstream code
  can distinguish the two representations.

  Parameters
  ----------
  moisture_species_mass : dict[str, Array]
      Mass-weighted moisture fields (``q * d_mass``) keyed by species name.
  tracers_mass : dict[str, Array]
      Mass-weighted passive tracer fields keyed by tracer name.
  model : str
      Model identifier forwarded to :func:`_wrap_tracer_like`.
  dry_air_species_mass : dict[str, Array] or None, optional
      Mass-weighted dry-air species fields (CAM-SE models only).

  Returns
  -------
  tracer_mass : dict
      Tracer mass dict with key ``"mass_quantity": 1.0``.
  """
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
  """
  Assemble the static forcing dict from its precomputed fields.

  Parameters
  ----------
  phi_surf : Array[tuple[elem_idx, gll_idx, gll_idx], Float]
      Surface geopotential (m^2 s^-2).
  grad_phi_surf : Array[tuple[elem_idx, gll_idx, gll_idx, 2], Float]
      DSS-projected gradient of the surface geopotential.
  coriolis_param : Array[tuple[elem_idx, gll_idx, gll_idx], Float]
      Coriolis parameter ``f = 2 Omega sin(lat)`` (or constant for f-plane
      models).
  nontrad_coriolis_param : Array[tuple[elem_idx, gll_idx, gll_idx], Float] or None, optional
      Non-traditional Coriolis parameter ``2 Omega cos(lat)``; included only
      for deep-atmosphere models.

  Returns
  -------
  static_forcing : dict[str, Array]
      Dict with keys ``"phi_surf"``, ``"grad_phi_surf"``, ``"coriolis_param"``,
      and optionally ``"nontrad_coriolis_param"``.
  """
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
  """
  Compute and assemble the time-invariant static forcing fields.

  Computes the DSS-projected gradient of the surface geopotential and the
  Coriolis parameter (using a constant latitude for f-plane models, or the
  true latitude for spherical models).  Deep-atmosphere models also receive
  the non-traditional Coriolis parameter.

  Parameters
  ----------
  phi_surf : Array[tuple[elem_idx, gll_idx, gll_idx], Float]
      Surface geopotential (m^2 s^-2).
  h_grid : SpectralElementGrid
      Horizontal grid struct containing physical coordinates and DSS operators.
  physics_config : dict
      Physics configuration with keys ``"radius_earth"`` and
      ``"angular_freq_earth"``.
  dims : tuple[int, ...]
      Grid dimension tuple used for DSS projection.
  model : str
      Model identifier; selects f-plane vs spherical and shallow vs deep.
  f_plane_center : float, optional
      Latitude (radians) for f-plane Coriolis constant (default: pi/4).

  Returns
  -------
  static_forcing : dict[str, Array]
      Dict assembled by :func:`wrap_static_forcing`.
  """
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
  Apply DSS (Direct Stiffness Summation) projection to all dynamics fields.

  Each dynamics field is projected level-by-level via :func:`project_scalar_3d`
  to enforce C0 continuity across element boundaries.  Non-hydrostatic fields
  ``"phi_i"`` and ``"w_i"`` are projected only when present.

  Parameters
  ----------
  dynamics_in : dict[str, Array]
      Input dynamics state from :func:`wrap_dynamics`.
  h_grid : SpectralElementGrid
      Horizontal grid struct containing DSS operators.
  dims : tuple[int, ...]
      Grid dimension tuple used for DSS projection.
  model : str
      Model identifier; determines whether non-hydrostatic fields are projected.

  Returns
  -------
  dynamics_proj : dict[str, Array]
      Dynamics state with all fields replaced by their DSS projections.
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
  """
  Apply DSS projection to a 3-D scalar field, with optional MPI communication.

  When MPI communication is enabled the global DSS assembly is performed via
  :func:`project_scalar_global`; otherwise the purely local 2-D
  :func:`project_scalar` is mapped over the vertical axis via
  ``vmap_1d_apply``.

  Parameters
  ----------
  variable : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Discontinuous 3-D scalar field (one value per GLL node per level).
  h_grid : SpectralElementGrid
      Horizontal grid struct containing DSS operators.
  dims : tuple[int, ...]
      Grid dimension tuple used for DSS projection.

  Returns
  -------
  variable_cont : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      DSS-projected (C0-continuous) scalar field.
  """
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
  Recover the surface pressure from the dynamics ``d_mass`` field.

  Sums the layer masses over all vertical levels and adds the model-top
  pressure ``p_top = hybrid_a_i[0] * reference_surface_mass``.

  Parameters
  ----------
  state_in : dict[str, Array]
      Dynamics state dict containing ``"d_mass"`` with shape
      ``(elem, gll, gll, lev)``.
  v_grid : dict[str, Array]
      Vertical grid struct from :func:`init_vertical_grid` containing
      ``"hybrid_a_i"`` and ``"reference_surface_mass"``.

  Returns
  -------
  ps : Array[tuple[elem_idx, gll_idx, gll_idx], Float]
      Surface pressure (Pa).
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
  Vertically remap all dynamics fields to the reference hybrid-coordinate levels.

  Converts mass-weighted prognostic fields to layer-integrated form
  (``q * d_mass``), remaps them conservatively via :func:`zerroukat_remap`,
  then recovers mixing-ratio form on the reference levels.  Non-hydrostatic
  models additionally remap the geopotential perturbation and vertical
  velocity.

  Parameters
  ----------
  dynamics_in : dict[str, Array]
      Current dynamics state from :func:`wrap_dynamics`.
  static_forcing : dict[str, Array]
      Static forcing from :func:`init_static_forcing`; ``"phi_surf"`` and
      ``"grad_phi_surf"`` are used for the non-hydrostatic geopotential remap.
  v_grid : dict[str, Array]
      Vertical grid struct from :func:`init_vertical_grid`.
  physics_config : dict
      Physics configuration forwarded to thermodynamic helpers.
  num_lev : int
      Number of vertical levels; static JIT argument.
  model : str
      Model identifier; selects thermodynamic variable and non-hydrostatic
      handling.

  Returns
  -------
  dynamics_remapped : dict[str, Array]
      Dynamics state on the reference ``d_mass_ref`` levels.
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
  Compute a weighted sum of two dynamics states field-by-field.

  Parameters
  ----------
  state1 : dict[str, Array]
      First dynamics state from :func:`wrap_dynamics`.
  state2 : dict[str, Array]
      Second dynamics state with the same model fields as ``state1``.
  fold_coeff1 : float
      Scalar weight for ``state1``.
  fold_coeff2 : float
      Scalar weight for ``state2``.
  model : str
      Model identifier; selects the thermodynamic variable name and whether
      non-hydrostatic fields are combined.

  Returns
  -------
  dynamics_out : dict[str, Array]
      Field-wise ``fold_coeff1 * state1 + fold_coeff2 * state2``.
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
  Compute a weighted linear combination of a list of dynamics states.

  Applies :func:`sum_dynamics` iteratively: the first two states are combined
  with ``coeffs[0]`` and ``coeffs[1]``; each subsequent state is folded in
  with coefficient ``coeffs[i]`` (accumulated result weight is ``1.0``).

  Parameters
  ----------
  states : list[dict[str, Array]]
      List of dynamics state dicts from :func:`wrap_dynamics`.
  coeffs : sequence[float]
      Scalar weights, one per element of ``states``.
  model : str
      Model identifier forwarded to :func:`sum_dynamics`.

  Returns
  -------
  dynamics_out : dict[str, Array]
      Field-wise weighted sum over all input states.
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
  """
  Zero out ghost/halo GLL nodes using the grid mask.

  Broadcasts ``h_grid["ghost_mask"]`` to match ``field``'s rank and sets
  nodes where the mask is ``<= 0.5`` (ghost nodes) to zero.

  Parameters
  ----------
  field : Array
      Field array whose leading dimensions match ``(elem_idx, gll_idx, gll_idx)``.
  h_grid : SpectralElementGrid
      Horizontal grid struct containing ``"ghost_mask"`` of shape
      ``(elem_idx, gll_idx, gll_idx)``.

  Returns
  -------
  masked_field : Array
      Field with ghost-node values replaced by zero.
  """
  mask = h_grid["ghost_mask"]
  shape = list(h_grid["ghost_mask"].shape) + (field.ndim - mask.ndim) * [1]
  return jnp.where(mask.reshape(shape) > 0.5, field, 0.0)


def check_dynamics_nan(dynamics,
                       h_grid,
                       model):
  """
  Check whether any dynamics field contains a NaN on any MPI rank.

  Ghost nodes are excluded via :func:`apply_mask` before the NaN check.
  The per-rank boolean is reduced across all ranks via :func:`global_sum`.

  Parameters
  ----------
  dynamics : dict[str, Array]
      Dynamics state dict from :func:`wrap_dynamics`.
  h_grid : SpectralElementGrid
      Horizontal grid struct containing ``"ghost_mask"``.
  model : str
      Model identifier; determines which fields (including non-hydrostatic
      ones) are checked.

  Returns
  -------
  has_nan : bool
      ``True`` if any NaN is found in any dynamics field on any MPI rank.
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
  """
  Check whether any tracer field contains a NaN on any MPI rank.

  Ghost nodes are excluded via :func:`apply_mask` before the NaN check.
  The per-rank boolean is reduced across all ranks via :func:`global_sum`.

  Parameters
  ----------
  tracers : dict
      Tracer state dict from :func:`wrap_tracers` with sub-dicts
      ``"moisture_species"``, ``"tracers"``, and optionally
      ``"dry_air_species"`` (CAM-SE models).
  h_grid : SpectralElementGrid
      Horizontal grid struct containing ``"ghost_mask"``.
  model : str
      Model identifier; determines whether ``"dry_air_species"`` is checked.

  Returns
  -------
  has_nan : bool
      ``True`` if any NaN is found in any tracer field on any MPI rank.
  """
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
