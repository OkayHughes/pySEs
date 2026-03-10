import numpy as np
from .._config import get_backend as _get_backend
from ..operations_2d.operators import horizontal_weak_vector_laplacian, horizontal_weak_laplacian
from ..operations_2d.tensor_hyperviscosity import (eval_quasi_uniform_hypervisc_coeff,
                                                   eval_variable_resolution_hypervisc_coeff)
from ..operations_2d.horizontal_grid import eval_global_grid_deformation_metrics
from .model_state import wrap_dynamics, project_dynamics, wrap_tracer_consist_hypervis
from .utils_3d import interface_to_delta, interface_to_midlevel
from .homme.thermodynamics import eval_balanced_geopotential
from .mass_coordinate import surface_mass_to_interface_mass
from functools import partial
from .model_info import hydrostatic_models, thermodynamic_variable_names, homme_models, cam_se_models
_be = _get_backend()
vmap_1d_apply = _be.vmap_1d_apply
jit = _be.jit
jnp = _be.np
device_wrapper = _be.array


@partial(jit, static_argnames=["apply_tensor"])
def scalar_harmonic_3d(scalar,
                       h_grid,
                       physics_config,
                       apply_tensor=False):
  """
  Apply the weak-form horizontal scalar Laplacian to a 3-D field level-by-level.

  Parameters
  ----------
  scalar : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      3-D scalar field.
  h_grid : SpectralElementGrid
      Horizontal spectral element grid.
  physics_config : dict[str, Any]
      Physical constants; must contain ``radius_earth``.
  apply_tensor : bool, optional
      If True, use the tensor (variable-resolution) Laplacian weighting
      (default: False).

  Returns
  -------
  del2 : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Weak-form scalar Laplacian at each model level.
  """
  def lap_wk_onearg(scalar):
      return horizontal_weak_laplacian(scalar, h_grid, a=physics_config["radius_earth"], apply_tensor=apply_tensor)

  del2 = vmap_1d_apply(lap_wk_onearg, scalar, -1, -1)
  return del2


@jit
def vector_harmonic_3d(vector,
                       h_grid,
                       physics_config,
                       nu_div_factor):
  """
  Apply the weak-form horizontal vector Laplacian to a 3-D vector field level-by-level.

  Parameters
  ----------
  vector : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx, 2], Float]
      3-D horizontal vector field.
  h_grid : SpectralElementGrid
      Horizontal spectral element grid.
  physics_config : dict[str, Any]
      Physical constants; must contain ``radius_earth``.
  nu_div_factor : float
      Additional scaling applied to the divergence component of the
      vector Laplacian.

  Returns
  -------
  del2 : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx, 2], Float]
      Weak-form vector Laplacian at each model level.
  """
  def vec_lap_wk_onearg(vector):
      return horizontal_weak_vector_laplacian(vector, h_grid, a=physics_config["radius_earth"],
                                              nu_div_fact=nu_div_factor)

  del2 = vmap_1d_apply(vec_lap_wk_onearg, vector, -2, -2)
  return del2


@partial(jit, static_argnames=["apply_nu", "model"])
def eval_hypervis_harmonic(dynamics,
                           h_grid,
                           physics_config,
                           diffusion_config,
                           model,
                           apply_nu=True):
  """
  Evaluate one biharmonic (del^2) pass of hyperviscosity on a 3-D dynamics state.

  Used as the inner routine for the full fourth-order biharmonic operator:
  ``eval_hypervis_terms`` calls this twice (with and without ``apply_nu``).

  Parameters
  ----------
  dynamics : DynamicsState
      3-D dynamics state dict containing ``"horizontal_wind"``,
      thermodynamic variable, and ``"d_mass"`` (and ``"phi_i"``, ``"w_i"``
      for non-hydrostatic models).
  h_grid : SpectralElementGrid
      Horizontal spectral element grid.
  physics_config : dict[str, Any]
      Physical constants; must contain ``radius_earth``.
  diffusion_config : dict[str, Any]
      Hyperviscosity configuration.
  model : model_info.models
      Model identifier string.
  apply_nu : bool, optional
      If True, multiply by the viscosity coefficients ``nu``, ``nu_phi``,
      ``nu_d_mass`` from ``diffusion_config``; if False, use unit coefficients
      (default: True).

  Returns
  -------
  hypervis_tend : DynamicsState
      Intermediate hyperviscosity tendency (one harmonic pass).
  """
  apply_tensor = apply_nu and "tensor_hypervis" in diffusion_config.keys()

  if apply_nu:
    nu_default = diffusion_config["nu"]
    nu_phi = diffusion_config["nu_phi"]
    nu_d_mass = diffusion_config["nu_d_mass"]
  else:
    nu_default = 1.0
    nu_phi = 1.0
    nu_d_mass = 1.0

  if "tensor_hypervis" in diffusion_config.keys():
    u_cart = jnp.einsum("fijks,fijcs->fijkc", dynamics["horizontal_wind"], h_grid["physical_to_cartesian"])
    components = []
    for comp_idx in range(u_cart.shape[-1]):
      components.append(scalar_harmonic_3d(u_cart[:, :, :, :, comp_idx],
                                           h_grid,
                                           physics_config,
                                           apply_tensor=apply_tensor))
    hyperdiff_u = jnp.einsum("fijkc,fijcs->fijks", jnp.stack(components, axis=-1), h_grid["physical_to_cartesian"])
  elif "constant_hypervis" in diffusion_config.keys():
    nu_div_factor = diffusion_config["nu_div_factor"] if apply_nu else 1.0
    hyperdiff_u = vector_harmonic_3d(dynamics["horizontal_wind"],
                                     h_grid, physics_config, nu_div_factor)
  hyperdiff_d_mass = scalar_harmonic_3d(dynamics["d_mass"], h_grid, physics_config, apply_tensor=apply_tensor)
  if model not in hydrostatic_models:
    hyperdiff_phi_i = nu_phi * scalar_harmonic_3d(dynamics["phi_i"],
                                                  h_grid,
                                                  physics_config,
                                                  apply_tensor=apply_tensor)
    hyperdiff_w_i = nu_default * scalar_harmonic_3d(dynamics["w_i"],
                                                    h_grid,
                                                    physics_config,
                                                    apply_tensor=apply_tensor)
  else:
    hyperdiff_phi_i = None
    hyperdiff_w_i = None

  if model in homme_models:
    # THIS IS THE ONE POINT IN THE CODE WHERE dynamics["theta_v_d_mass"] MAY BE theta_v
    hyperdiff_thermo = scalar_harmonic_3d(dynamics["theta_v_d_mass"],
                                          h_grid,
                                          physics_config,
                                          apply_tensor=apply_tensor)
  else:
    hyperdiff_thermo = scalar_harmonic_3d(dynamics["T"],
                                          h_grid,
                                          physics_config,
                                          apply_tensor=apply_tensor)
  hypervis_tend = wrap_dynamics(nu_default * hyperdiff_u,
                                nu_default * hyperdiff_thermo,
                                nu_d_mass * hyperdiff_d_mass,
                                model,
                                phi_i=hyperdiff_phi_i,
                                w_i=hyperdiff_w_i)
  return hypervis_tend


@partial(jit, static_argnames=["dims", "model"])
def advance_sponge_layer(dynamics,
                         dt,
                         h_grid,
                         physics_config,
                         diffusion_config,
                         dims,
                         model):
  """
  Apply one Euler step of sponge-layer (top-of-model) diffusion.

  Only the uppermost ``n_sponge`` model levels are diffused.  Each level
  receives a viscosity coefficient scaled by ``nu_top * nu_ramp``, where
  ``nu_ramp`` decreases monotonically away from the model top.

  Parameters
  ----------
  dynamics : DynamicsState
      3-D dynamics state dict.
  dt : float
      Sponge-layer sub-step size in seconds.
  h_grid : SpectralElementGrid
      Horizontal spectral element grid.
  physics_config : dict[str, Any]
      Physical constants; must contain ``radius_earth``.
  diffusion_config : dict[str, Any]
      Hyperviscosity config; must contain ``"nu_top"`` and ``"nu_ramp"``.
  dims : frozendict[str, int]
      Grid dimension parameters.
  model : model_info.models
      Model identifier string.

  Returns
  -------
  dynamics_out : DynamicsState
      Updated dynamics state with the sponge-layer tendency applied.
  """
  nu_top = diffusion_config["nu_top"]
  nu_ramp = nu_top * diffusion_config["nu_ramp"]
  n_sponge = nu_ramp.size
  if model not in hydrostatic_models:
    hyperdiff_phi_i = nu_ramp * scalar_harmonic_3d(dynamics["phi_i"][:, :, :, :n_sponge],
                                                   h_grid, physics_config)
    hyperdiff_w_i = nu_ramp * scalar_harmonic_3d(dynamics["w_i"][:, :, :, :n_sponge],
                                                 h_grid, physics_config)
  else:
    hyperdiff_phi_i = None
    hyperdiff_w_i = None
  thermo_var_name = thermodynamic_variable_names[model]
  hyperdiff_thermo = scalar_harmonic_3d(dynamics[thermo_var_name][:, :, :, :n_sponge],
                                        h_grid, physics_config)
  hyperdiff_thermo *= nu_ramp
  hyperdiff_d_mass = scalar_harmonic_3d(dynamics["d_mass"][:, :, :, :n_sponge], h_grid, physics_config)
  hyperdiff_d_mass *= nu_ramp
  hyperdiff_u = vector_harmonic_3d(dynamics["horizontal_wind"][:, :, :, :n_sponge, :],
                                   h_grid, physics_config, 1.0)
  hyperdiff_u *= nu_ramp[:, :, :, :, np.newaxis]
  hyperdiff_state = wrap_dynamics(hyperdiff_u,
                                  hyperdiff_thermo,
                                  hyperdiff_d_mass,
                                  model,
                                  phi_i=hyperdiff_phi_i,
                                  w_i=hyperdiff_w_i)
  hyperdiff_state = project_dynamics(hyperdiff_state,
                                     h_grid,
                                     dims,
                                     model)

  u_out = jnp.concatenate((dt * hyperdiff_state["horizontal_wind"] + dynamics["horizontal_wind"][:, :, :, :n_sponge, :],
                           dynamics["horizontal_wind"][:, :, :, n_sponge:, :]),
                          axis=-2)
  thermo_out = jnp.concatenate((dt * hyperdiff_state[thermo_var_name] + dynamics[thermo_var_name][:, :, :, :n_sponge],
                                dynamics[thermo_var_name][:, :, :, n_sponge:]),
                               axis=-1)
  d_mass_out = jnp.concatenate((dt * hyperdiff_state["d_mass"] + dynamics["d_mass"][:, :, :, :n_sponge],
                                dynamics["d_mass"][:, :, :, n_sponge:]),
                               axis=-1)
  if model not in hydrostatic_models:
    phi_i_out = jnp.concatenate((dt * hyperdiff_state["phi_i"] + dynamics["phi_i"][:, :, :, :n_sponge],
                                 dynamics["phi_i"][:, :, :, n_sponge:]),
                                axis=-1)
    w_i_out = jnp.concatenate((dt * hyperdiff_state["w_i"] + dynamics["w_i"][:, :, :, :n_sponge],
                               dynamics["w_i"][:, :, :, n_sponge:]),
                              axis=-1)
  else:
    phi_i_out = None
    w_i_out = None

  struct = wrap_dynamics(u_out,
                         thermo_out,
                         d_mass_out,
                         model,
                         phi_i=phi_i_out,
                         w_i=w_i_out)
  return struct


@partial(jit, static_argnames=["n_sponge"])
def eval_nu_ramp(v_grid,
                 n_sponge):
  """
  Compute the level-dependent viscosity ramp profile for the sponge layer.

  Levels near the model top (where pressure ratio is large relative to the
  reference surface mass) receive a near-unity ramp, while levels further
  down receive smaller values capped at 8.

  Parameters
  ----------
  v_grid : dict[str, Array]
      Vertical grid from ``init_vertical_grid``; must contain
      ``"hybrid_a_i"``, ``"hybrid_b_i"``, and ``"reference_surface_mass"``.
  n_sponge : int
      Number of levels in the sponge layer (must be static for JIT).

  Returns
  -------
  nu_ramp : Array[tuple[1, 1, 1, n_sponge], Float]
      Dimensionless viscosity ramp coefficient for each sponge level.
  """
  pressure_ratio = ((v_grid["hybrid_a_i"][0] + v_grid["hybrid_b_i"][0]) /
                    (v_grid["hybrid_a_i"][:n_sponge] + v_grid["hybrid_b_i"][:n_sponge]))
  nu_ramp = jnp.minimum(device_wrapper(8.0),
                        (16.0 * pressure_ratio**2 /
                         (pressure_ratio**2 + 1.0))[np.newaxis, np.newaxis, np.newaxis, :])
  return nu_ramp


def init_hypervis_config_const(ne,
                               physics_config,
                               v_grid,
                               nu_top=2.5e5,
                               nu_base=-1.0,
                               nu_phi=-1.0,
                               nu_d_mass=-1.0,
                               nu_div_factor=2.5,
                               n_sponge=5,
                               T_ref=288.0,
                               T_ref_lapse=0.0065):
  """
  Initialize a quasi-uniform hyperviscosity configuration for the 3-D dynamical core.

  Parameters
  ----------
  ne : int
      Number of elements along one edge of the cubed-sphere face.
  physics_config : dict[str, Any]
      Physics config; must contain ``radius_earth``.
  v_grid : dict[str, Array]
      Vertical grid from ``init_vertical_grid``.
  nu_top : float, optional
      Sponge-layer viscosity coefficient at the model top (default: 2.5e5).
  nu_base : float, optional
      Biharmonic viscosity for horizontal wind (m^4 s^-1).
      Negative values auto-compute from resolution.
  nu_phi : float, optional
      Viscosity for geopotential.  Negative values match ``nu_base``.
  nu_d_mass : float, optional
      Viscosity for layer thickness.  Negative values match ``nu_base``.
  nu_div_factor : float, optional
      Additional factor for the divergence component (default: 2.5).
  n_sponge : int, optional
      Number of levels in the sponge layer (default: 5).
  T_ref : float, optional
      Reference temperature for the hyperviscosity reference state (K).
  T_ref_lapse : float, optional
      Reference temperature lapse rate (K m^-1).

  Returns
  -------
  diffusion_config : dict[str, Any]
      Hyperviscosity configuration dict for the 3-D dynamical core.
  """
  nu = eval_quasi_uniform_hypervisc_coeff(ne, physics_config["radius_earth"]) if nu_base <= 0 else nu_base
  nu_phi = nu if nu_phi < 0 else nu_phi
  nu_d_mass = nu if nu_d_mass < 0 else nu_d_mass
  nu_ramp = eval_nu_ramp(v_grid, n_sponge)
  diffusion_config = {"constant_hypervis": 1.0,
                      "nu": device_wrapper(nu),
                      "nu_phi": device_wrapper(nu_phi),
                      "nu_d_mass": device_wrapper(nu_d_mass),
                      "nu_tracer": device_wrapper(nu_d_mass),
                      "nu_div_factor": device_wrapper(nu_div_factor),
                      "reference_profiles": {"T_ref": device_wrapper(T_ref),
                                             "T_ref_lapse": device_wrapper(T_ref_lapse)}}

  if n_sponge > 0:
    diffusion_config["sponge_layer"] = 1.0
    diffusion_config["nu_top"] = device_wrapper(nu_top)
    diffusion_config["nu_ramp"] = device_wrapper(nu_ramp)
  return diffusion_config


def init_hypervis_config_stub():
  """
  Return a minimal diffusion config that disables all hyperviscosity.

  Returns
  -------
  diffusion_config : dict[str, Any]
      Config dict with the ``"disable_diffusion"`` flag set.
  """
  return {"disable_diffusion": 1.0}


def init_hypervis_config_tensor(h_grid,
                                v_grid,
                                dims,
                                config,
                                nu_top=2.5e5,
                                ad_hoc_scale=0.5,
                                n_sponge=5,
                                T_ref=288.0,
                                T_ref_lapse=0.0065):
  """
  Initialize a variable-resolution tensor hyperviscosity configuration for the 3-D dynamical core.

  Parameters
  ----------
  h_grid : SpectralElementGrid
      Horizontal spectral element grid (must contain ``"hypervis_scaling"``).
  v_grid : dict[str, Array]
      Vertical grid from ``init_vertical_grid``.
  dims : frozendict[str, int]
      Grid dimension parameters.
  config : dict[str, Any]
      Physics config; must contain ``radius_earth``.
  nu_top : float, optional
      Sponge-layer viscosity at the model top (default: 2.5e5).
  ad_hoc_scale : float, optional
      Empirical rescaling factor for the computed viscosity (default: 0.5).
  n_sponge : int, optional
      Number of sponge-layer levels (default: 5).
  T_ref : float, optional
      Reference temperature for the reference state (K).
  T_ref_lapse : float, optional
      Reference temperature lapse rate (K m^-1).

  Returns
  -------
  diffusion_config : dict[str, Any]
      Tensor hyperviscosity configuration dict for the 3-D dynamical core.
  """
  nu_ramp = eval_nu_ramp(v_grid, n_sponge)
  radius_earth = config["radius_earth"]
  _, max_min_dx, _ = eval_global_grid_deformation_metrics(h_grid, dims)
  nu_tens = eval_variable_resolution_hypervisc_coeff(radius_earth * max_min_dx,
                                                     h_grid["hypervis_scaling"],
                                                     dims["npt"])
  nu = device_wrapper(ad_hoc_scale * nu_tens)
  diffusion_config = {"tensor_hypervis": 1.0,
                      "nu": nu,
                      "nu_phi": nu,
                      "nu_d_mass": nu,
                      "nu_tracer": nu,
                      "reference_profiles": {"T_ref": device_wrapper(T_ref),
                                             "T_ref_lapse": device_wrapper(T_ref_lapse)}}
  if n_sponge > 0:
    diffusion_config["do_sponge_layer"] = 1.0
    diffusion_config["nu_top"] = device_wrapper(nu_top)
    diffusion_config["nu_ramp"] = device_wrapper(nu_ramp)
  return diffusion_config


def diffusion_config_for_tracer_consist(diffusion_config,
                                        v_grid):
  """
  Modify a 3-D diffusion config for use with tracer consistency.

  Disables mass diffusion (``"nu_d_mass" = 0``) and adds a
  ``"d_mass_tracer"`` scaling array derived from the vertical grid.

  Parameters
  ----------
  diffusion_config : dict[str, Any]
      Existing hyperviscosity configuration (modified in-place).
  v_grid : dict[str, Array]
      Vertical grid from ``init_vertical_grid``; must contain
      ``"hybrid_a_m"``, ``"hybrid_b_m"``, and ``"reference_surface_mass"``.

  Returns
  -------
  diffusion_config : dict[str, Any]
      The modified configuration dict (same object as input).
  """
  diffusion_config["d_mass_tracer"] = (v_grid["hybrid_a_m"] + v_grid["hybrid_b_m"]) * v_grid["reference_surface_mass"]
  diffusion_config["nu_d_mass"] = 0.0
  return diffusion_config


@partial(jit, static_argnames=["model", "dims"])
def eval_hypervis_terms(dynamics,
                        static_forcing,
                        h_grid,
                        v_grid,
                        dims,
                        physics_config,
                        diffusion_config,
                        model):
  """
  Evaluate the full fourth-order biharmonic hyperviscosity tendency.

  Computes perturbation fields relative to a reference state, applies two
  consecutive harmonic passes (``eval_hypervis_harmonic``), and assembles
  the tracer-consistency struct.

  Parameters
  ----------
  dynamics : DynamicsState
      3-D dynamics state dict.
  static_forcing : StaticForcing
      Surface geopotential and Coriolis fields.
  h_grid : SpectralElementGrid
      Horizontal spectral element grid.
  v_grid : dict[str, Array]
      Vertical grid from ``init_vertical_grid``.
  dims : frozendict[str, int]
      Grid dimension parameters.
  physics_config : dict[str, Any]
      Physical constants.
  diffusion_config : dict[str, Any]
      Hyperviscosity configuration; must contain ``"reference_profiles"``.
  model : model_info.models
      Model identifier string.

  Returns
  -------
  dynamics_tend : DynamicsState
      Fourth-order hyperviscosity tendency.
  tracer_consistency : dict[str, Array]
      Tracer-consistency struct from the mass diffusion tendency.
  """
  ref_state = eval_ref_state(static_forcing["phi_surf"], v_grid, physics_config, diffusion_config, model)
  d_mass_pert = dynamics["d_mass"] - ref_state["d_mass"]
  if model not in hydrostatic_models:
    phi_i_pert = dynamics["phi_i"] - ref_state["phi_i"]
    w_i = dynamics["w_i"]
  else:
    phi_i_pert = None
    w_i = None

  if model in cam_se_models:
    thermo_var = dynamics["T"] - ref_state["T"]
  else:
    thermo_var = dynamics["theta_v_d_mass"] / dynamics["d_mass"] - ref_state["theta_v"]

  hypervis_state = wrap_dynamics(dynamics["horizontal_wind"],
                                 thermo_var,
                                 d_mass_pert,
                                 model,
                                 phi_i=phi_i_pert,
                                 w_i=w_i)
  for apply_nu in [True, False]:
    hypervis_state = eval_hypervis_harmonic(hypervis_state,
                                            h_grid,
                                            physics_config,
                                            diffusion_config,
                                            model,
                                            apply_nu=apply_nu)

    hypervis_state = project_dynamics(hypervis_state, h_grid, dims, model)
  if model not in hydrostatic_models:
    phi_i = jnp.concatenate((-hypervis_state["phi_i"][:, :, :, :-1],
                             jnp.zeros_like(hypervis_state["phi_i"][:, :, :, -1:])),
                            axis=-1)
    w_i = -hypervis_state["w_i"]
  else:
    phi_i = None
    w_i = None

  if model in homme_models:
    thermo_tend = -hypervis_state[thermodynamic_variable_names[model]] * dynamics["d_mass"]
  else:
    thermo_tend = -hypervis_state[thermodynamic_variable_names[model]]

  dynamics_tend = wrap_dynamics(-hypervis_state["horizontal_wind"],
                                thermo_tend,
                                -hypervis_state["d_mass"],
                                model,
                                phi_i=phi_i,
                                w_i=w_i)
  # slightly silly way of handling tracer consistency
  hypervisc_tend_for_tracer = jnp.where(diffusion_config["nu_d_mass"] > 0.0,
                                        hypervis_state["d_mass"] / diffusion_config["nu_d_mass"],
                                        jnp.zeros_like(hypervis_state["d_mass"]))
  tracer_consistency = wrap_tracer_consist_hypervis(1.0 * dynamics["d_mass"],
                                                    hypervisc_tend_for_tracer)
  return dynamics_tend, tracer_consistency


@partial(jit, static_argnames=["model"])
def eval_ref_state(phi_surf,
                   v_grid,
                   physics_config,
                   diffusion_config,
                   model):
  """
  Compute the hydrostatic reference state used for hyperviscosity perturbation subtraction.

  Constructs a simple analytic reference atmosphere (linear lapse rate)
  from the surface geopotential and the hybrid vertical coordinate, to
  which perturbations are computed before applying the biharmonic operator.

  Parameters
  ----------
  phi_surf : Array[tuple[elem_idx, gll_idx, gll_idx], Float]
      Surface geopotential (m^2 s^-2).
  v_grid : dict[str, Array]
      Vertical grid from ``init_vertical_grid``.
  physics_config : dict[str, Any]
      Physical constants.
  diffusion_config : dict[str, Any]
      Hyperviscosity config; must contain ``"reference_profiles"`` with
      keys ``"T_ref"`` and ``"T_ref_lapse"``.
  model : model_info.models
      Model identifier string.

  Returns
  -------
  ref_profiles : dict[str, Array]
      Reference state with keys ``"d_mass"``, thermodynamic variable
      (``"T"`` or ``"theta_v"``), and optionally ``"phi_i"``.
  """
  # could eventually only be called once.
  # due to low cost, if we end up going the "vmap over nelem" route,
  # then this should probably be recomputed from element-local (and ideally SM-local) phi_surf
  reference_params = diffusion_config["reference_profiles"]
  dummy_thermo = physics_config
  ps_ref = v_grid["reference_surface_mass"] * jnp.exp(-phi_surf / (dummy_thermo["Rgas"] * reference_params["T_ref"]))
  pressure_int = surface_mass_to_interface_mass(ps_ref, v_grid)
  d_mass_ref = interface_to_delta(pressure_int)
  p_mid = interface_to_midlevel(pressure_int)
  exner = (p_mid / dummy_thermo["p0"])**(dummy_thermo["Rgas"] / dummy_thermo["cp"])
  T1 = reference_params["T_ref_lapse"] * reference_params["T_ref"] * dummy_thermo["cp"] / physics_config["gravity"]
  T0 = reference_params["T_ref"] - T1
  theta_ref = T0 + T0 * (1 - exner) + T1
  if model in cam_se_models:
    thermo_var_name = "T"
    thermo_profile = theta_ref * exner
  elif model in homme_models:
    thermo_var_name = "theta_v"
    thermo_profile = theta_ref
  ref_profiles = {"d_mass": d_mass_ref,
                  thermo_var_name: thermo_profile}
  if model not in hydrostatic_models:
    ref_profiles["phi_i"] = eval_balanced_geopotential(phi_surf, p_mid, theta_ref * d_mass_ref, physics_config)
  return ref_profiles
