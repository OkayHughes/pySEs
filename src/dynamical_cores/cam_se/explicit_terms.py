from ..operators_3d import horizontal_gradient_3d, horizontal_vorticity_3d, horizontal_divergence_3d
from ..utils_3d import physical_dot_product
import numpy as np
from ..._config import get_backend as _get_backend
from .thermodynamics import (eval_sum_species,
                             eval_midlevel_pressure,
                             eval_interface_pressure,
                             eval_cp_moist,
                             eval_balanced_geopotential,
                             eval_exner_function)
from .thermodynamics import eval_Rgas_dry
from .thermodynamics import eval_cp_dry
from .thermodynamics import eval_virtual_temperature
from ..model_state import wrap_dynamics, wrap_tracer_consist_dynamics
from enum import Enum
from functools import partial
_be = _get_backend()
jnp = _be.np
jit = _be.jit

pressure_gradient_options = Enum('pressure_gradient_options',
                                 [("basic", 1),
                                  ("grad_exner", 2),
                                  ("corrected_grad_exner", 3)])


@partial(jit, static_argnames=["model"])
def init_common_variables(dynamics,
                          static_forcing,
                          moisture_species,
                          dry_air_species,
                          h_grid,
                          v_grid,
                          physics_config,
                          model):
  """
  Pre-compute intermediate quantities shared across the CAM-SE adiabatic tendency terms.

  Derives effective gas constants, heat capacities, virtual temperature, mid-level
  and interface pressures, geopotential, and the inverse density from the prognostic
  state and tracer fields.

  Parameters
  ----------
  dynamics : dict[str, Array]
      Dynamics state dict containing ``"T"``, ``"horizontal_wind"``, and ``"d_mass"``.
  static_forcing : dict[str, Array]
      Static forcing struct from :func:`init_static_forcing`; must contain
      ``"phi_surf"`` and ``"coriolis_param"``.
  moisture_species : dict[str, Array]
      Moisture mixing-ratio fields (kg moisture / kg dry air) keyed by species name.
  dry_air_species : dict[str, Array]
      Dry-air species mass-fraction fields keyed by species name.
  h_grid : SpectralElementGrid
      Horizontal grid struct.
  v_grid : dict[str, Array]
      Vertical grid struct from :func:`init_vertical_grid`; must contain
      ``"hybrid_a_i"`` and ``"reference_surface_mass"``.
  physics_config : dict
      Physics configuration dict.
  model : model_info.models
      Model identifier; static JIT argument.

  Returns
  -------
  common_variables : dict[str, Array]
      Dict containing the following keys:

      - ``"horizontal_wind"`` — wind vector ``(u, v)``
      - ``"temperature"`` — temperature (K)
      - ``"phi"`` — hydrostatically balanced geopotential (m^2 s^-2)
      - ``"d_mass"`` — dry-air layer mass (Pa)
      - ``"d_pressure"`` — moist layer pressure thickness (Pa)
      - ``"virtual_temperature"`` — virtual temperature (K)
      - ``"cp"`` — effective moist heat capacity (J kg^-1 K^-1)
      - ``"cp_dry"`` — dry-air heat capacity (J kg^-1 K^-1)
      - ``"R_dry"`` — effective dry-air gas constant (J kg^-1 K^-1)
      - ``"grad_pressure"`` — horizontal gradient of mid-level pressure
      - ``"density_inv"`` — specific volume ``R_dry * T_v / p`` (m^3 kg^-1)
      - ``"pressure_midlevel"`` — mid-level pressure (Pa)
      - ``"phi_surf"`` — surface geopotential (m^2 s^-2)
      - ``"coriolis_param"`` — Coriolis parameter (s^-1)
  """
  temperature = dynamics["T"]
  wind = dynamics["horizontal_wind"]
  d_mass = dynamics["d_mass"]

  phi_surf = static_forcing["phi_surf"]
  coriolis_param = static_forcing["coriolis_param"]

  R_dry = eval_Rgas_dry(dry_air_species, physics_config)
  cp_dry = eval_cp_dry(dry_air_species, physics_config)
  cp = eval_cp_moist(moisture_species, cp_dry, physics_config)
  total_mixing_ratio = eval_sum_species(moisture_species)
  virtual_temperature = eval_virtual_temperature(temperature,
                                                 moisture_species,
                                                 total_mixing_ratio,
                                                 R_dry,
                                                 physics_config)

  ptop = v_grid["hybrid_a_i"][0] * v_grid["reference_surface_mass"]
  d_pressure = total_mixing_ratio * d_mass
  p_int = eval_interface_pressure(d_pressure, ptop)
  pressure_model_lev = eval_midlevel_pressure(p_int)
  grad_pressure = horizontal_gradient_3d(pressure_model_lev, h_grid, physics_config)
  density_inv = R_dry * virtual_temperature / pressure_model_lev
  phi = eval_balanced_geopotential(virtual_temperature,
                                   d_pressure,
                                   pressure_model_lev,
                                   R_dry,
                                   phi_surf)
  return {"horizontal_wind": wind,
          "temperature": temperature,
          "phi": phi,
          "d_mass": d_mass,
          "d_pressure": d_pressure,
          "virtual_temperature": virtual_temperature,
          "cp": cp,
          "cp_dry": cp_dry,
          "R_dry": R_dry,
          "grad_pressure": grad_pressure,
          "density_inv": density_inv,
          "pressure_midlevel": pressure_model_lev,
          "phi_surf": phi_surf,
          "coriolis_param": coriolis_param}


@jit
def eval_temperature_horiz_advection_term(common_variables,
                                          h_grid,
                                          physics_config):
  """
  Compute the horizontal advection contribution to the temperature tendency.

  Evaluates ``-u · grad(T)`` using the horizontal wind and the spectral-element
  gradient of temperature.

  Parameters
  ----------
  common_variables : dict[str, Array]
      Shared intermediate quantities from :func:`init_common_variables`; must
      contain ``"temperature"`` and ``"horizontal_wind"``.
  h_grid : SpectralElementGrid
      Horizontal grid struct.
  physics_config : dict
      Physics configuration dict.

  Returns
  -------
  horiz_advection : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Horizontal temperature advection tendency ``-u · grad(T)`` (K s^-1).
  """
  grad_temperature = horizontal_gradient_3d(common_variables["temperature"], h_grid, physics_config)
  u_dot_grad_temperature = physical_dot_product(common_variables["horizontal_wind"], grad_temperature)
  return -u_dot_grad_temperature


@jit
def eval_temperature_vertical_advection_term(common_variables,
                                             h_grid,
                                             physics_config):
  """
  Compute the adiabatic heating contribution to the temperature tendency.

  Estimates the vertical pressure velocity ``omega = dp/dt`` from horizontal
  mass-flux divergence and horizontal pressure advection, then converts to
  a temperature tendency via the ideal-gas adiabatic heating relation:
  ``dT/dt|_adiab = (1 / (rho * cp)) * omega``.

  The vertical pressure velocity is approximated as:
  ``omega ≈ -cumsum(div(dp * u)) + 0.5 * div(dp * u) + u · grad(p)``

  Parameters
  ----------
  common_variables : dict[str, Array]
      Shared intermediate quantities from :func:`init_common_variables`; must
      contain ``"horizontal_wind"``, ``"cp"``, ``"d_pressure"``,
      ``"grad_pressure"``, and ``"density_inv"``.
  h_grid : SpectralElementGrid
      Horizontal grid struct.
  physics_config : dict
      Physics configuration dict.

  Returns
  -------
  adiab_heating : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Adiabatic temperature tendency ``omega / (rho * cp)`` (K s^-1).
  """
  u = common_variables["horizontal_wind"]
  cp = common_variables["cp"]
  d_pressure_u = common_variables["d_pressure"][:, :, :, :, np.newaxis] * u
  div_d_pressure_u = horizontal_divergence_3d(d_pressure_u, h_grid, physics_config)
  u_dot_grad_pressure = physical_dot_product(u, common_variables["grad_pressure"])
  d_pressure_tend = -jnp.cumsum(div_d_pressure_u, axis=3) + 0.5 * div_d_pressure_u
  vertical_pressure_velocity = d_pressure_tend + u_dot_grad_pressure
  return common_variables["density_inv"] * vertical_pressure_velocity / cp


@jit
def eval_d_mass_divergence_term(common_variables,
                                h_grid,
                                physics_config):
  """
  Compute the horizontal divergence tendency for the dry-air layer mass.

  Evaluates ``-div(d_mass * u)`` using the spectral-element divergence
  operator applied to the horizontal mass flux.

  Parameters
  ----------
  common_variables : dict[str, Array]
      Shared intermediate quantities from :func:`init_common_variables`; must
      contain ``"d_mass"`` and ``"horizontal_wind"``.
  h_grid : SpectralElementGrid
      Horizontal grid struct.
  physics_config : dict
      Physics configuration dict.

  Returns
  -------
  d_mass_tend : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Layer-mass tendency ``-div(d_mass * u)`` (Pa s^-1).
  """
  d_mass_u = common_variables["d_mass"][:, :, :, :, np.newaxis] * common_variables["horizontal_wind"]
  div_d_mass_u = horizontal_divergence_3d(d_mass_u, h_grid, physics_config)
  return -div_d_mass_u


@jit
def eval_energy_gradient_term(common_variables,
                              h_grid,
                              physics_config):
  """
  Compute the gradient of total mechanical energy for the wind tendency.

  Evaluates ``-grad(KE + phi)`` where ``KE = |u|^2 / 2`` is the horizontal
  kinetic energy and ``phi`` is the geopotential.  Used in the vector-invariant
  momentum equation together with :func:`eval_vorticity_term`.

  Parameters
  ----------
  common_variables : dict[str, Array]
      Shared intermediate quantities from :func:`init_common_variables`; must
      contain ``"horizontal_wind"`` and ``"phi"``.
  h_grid : SpectralElementGrid
      Horizontal grid struct.
  physics_config : dict
      Physics configuration dict.

  Returns
  -------
  energy_grad : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx, 2], Float]
      Wind tendency ``-grad(KE + phi)`` (m s^-2).
  """
  u = common_variables["horizontal_wind"]
  phi = common_variables["phi"]
  kinetic_energy = physical_dot_product(u, u) / 2.0
  return -horizontal_gradient_3d(kinetic_energy[:, :, :] + phi, h_grid, physics_config)


@jit
def eval_vorticity_term(common_variables,
                        h_grid,
                        physics_config):
  """
  Compute the absolute-vorticity rotation term for the wind tendency.

  Evaluates ``(f + zeta) * u_perp`` in the vector-invariant form of the
  momentum equation, where ``zeta`` is the relative vorticity and ``u_perp``
  rotates the wind 90 degrees: ``u_perp = (v, -u)``.

  Parameters
  ----------
  common_variables : dict[str, Array]
      Shared intermediate quantities from :func:`init_common_variables`; must
      contain ``"horizontal_wind"`` and ``"coriolis_param"``.
  h_grid : SpectralElementGrid
      Horizontal grid struct.
  physics_config : dict
      Physics configuration dict.

  Returns
  -------
  vorticity_tend : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx, 2], Float]
      Wind tendency ``(f + zeta) * u_perp`` (m s^-2).
  """
  u = common_variables["horizontal_wind"]
  coriolis_parameter = common_variables["coriolis_param"]
  vorticity = horizontal_vorticity_3d(u, h_grid, physics_config)
  return jnp.stack((u[:, :, :, :, 1] * (coriolis_parameter[:, :, :, np.newaxis] + vorticity),
                    -u[:, :, :, :, 0] * (coriolis_parameter[:, :, :, np.newaxis] + vorticity)), axis=-1)


@partial(jit, static_argnames=["pgf_formulation"])
def eval_pressure_gradient_force_term(common_variables,
                                      h_grid,
                                      v_grid,
                                      physics_config,
                                      pgf_formulation):
  """
  Compute the pressure gradient force for the wind tendency.

  Supports three formulations selected by ``pgf_formulation``:

  - ``basic`` — ``-alpha * grad(p)`` where ``alpha = 1/rho``.
  - ``grad_exner`` — ``-cp_dry * theta_v * grad(pi)`` using the Exner function.
  - ``corrected_grad_exner`` — Exner formulation with the Simmons & Jiabin (1991)
    reference-profile correction near sigma levels; falls back to ``basic`` in
    purely pressure-based levels where ``hybrid_b_m <= 1e-9``.

  The reference-profile correction uses ``T_ref = 288 K`` and a standard
  tropospheric lapse rate of 0.0065 K m^-1.

  Parameters
  ----------
  common_variables : dict[str, Array]
      Shared intermediate quantities from :func:`init_common_variables`; must
      contain ``"cp_dry"``, ``"R_dry"``, ``"pressure_midlevel"``,
      ``"virtual_temperature"``, ``"density_inv"``, and ``"grad_pressure"``.
  h_grid : SpectralElementGrid
      Horizontal grid struct.
  v_grid : dict[str, Array]
      Vertical grid struct from :func:`init_vertical_grid`; must contain
      ``"hybrid_b_m"`` for the corrected formulation.
  physics_config : dict
      Physics configuration dict; must contain ``"cp"``, ``"gravity"``,
      and ``"p0"``.
  pgf_formulation : pressure_gradient_options
      Enum selecting the pressure gradient formulation; static JIT argument.

  Returns
  -------
  pgf_tend : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx, 2], Float]
      Pressure gradient force wind tendency (m s^-2).
  """
  cp_dry = common_variables["cp_dry"]
  R_dry = common_variables["R_dry"]
  exner = eval_exner_function(common_variables["pressure_midlevel"],
                              R_dry,
                              cp_dry,
                              physics_config)
  theta_v = common_variables["virtual_temperature"] / exner
  grad_exner = horizontal_gradient_3d(exner, h_grid, physics_config)
  lapse_rate = .0065
  # balanced ref profile correction:
  # reference temperature profile (Simmons and Jiabin, 1991, QJRMS, Section 2a)
  #
  #  Tref = T0+T1*Exner
  #  T1 = .0065*Tref*Cp/g ! = ~191
  #  T0 = Tref-T1         ! = ~97
  T_ref = 288.0

  T1 = (lapse_rate * T_ref * physics_config["cp"] / physics_config["gravity"])
  T0 = T_ref - T1
  pgf_term_grad_exner = cp_dry[:, :, :, :, np.newaxis] * theta_v[:, :, :, :, np.newaxis] * grad_exner
  grad_logexner = horizontal_gradient_3d(jnp.log(exner), h_grid, physics_config)
  pgf_correction = cp_dry[:, :, :, :, np.newaxis] * T0 * (grad_logexner - grad_exner / exner[:, :, :, :, np.newaxis])
  basic_pgf = common_variables["density_inv"][:, :, :, :, np.newaxis] * common_variables["grad_pressure"]
  if pgf_formulation == pressure_gradient_options.grad_exner:
      pgf_term = - pgf_term_grad_exner
  elif pgf_formulation == pressure_gradient_options.basic:
      pgf_term = - basic_pgf
  elif pgf_formulation == pressure_gradient_options.corrected_grad_exner:
      lower_levels_pgf = pgf_term_grad_exner + pgf_correction
      # only apply away from constant pressure levels
      pgf_term = -jnp.where(v_grid["hybrid_b_m"][np.newaxis, np.newaxis, np.newaxis, :, np.newaxis] > 1e-9,
                            lower_levels_pgf,
                            basic_pgf)
  else:
    raise ValueError("Pressure gradient not implemented")
  return pgf_term


@jit
def eval_tracer_consistency_term(common_variables):
  """
  Compute the mass-weighted wind flux for tracer consistency.

  Returns the horizontal mass flux ``d_mass * u`` used to keep tracer
  advection consistent with the dynamics layer-mass tendency.

  Parameters
  ----------
  common_variables : dict[str, Array]
      Shared intermediate quantities from :func:`init_common_variables`; must
      contain ``"d_mass"`` and ``"horizontal_wind"``.

  Returns
  -------
  d_mass_u : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx, 2], Float]
      Horizontal mass flux ``d_mass * u`` (Pa m s^-1).
  """
  return common_variables["d_mass"][:, :, :, :, jnp.newaxis] * common_variables["horizontal_wind"]


@partial(jit, static_argnames=["model", "pgf_formulation"])
def eval_explicit_tendency(dynamics,
                           static_forcing,
                           moist_species,
                           dry_air_species,
                           h_grid,
                           v_grid,
                           physics_config,
                           model,
                           pgf_formulation=pressure_gradient_options.corrected_grad_exner):
  """
  Assemble the full CAM-SE adiabatic explicit tendency.

  Calls :func:`init_common_variables` to pre-compute shared quantities, then
  sums the individual tendency terms for horizontal wind, temperature, and
  dry-air layer mass.  Also returns the tracer consistency mass flux for
  coupling with tracer advection.

  Parameters
  ----------
  dynamics : dict[str, Array]
      Dynamics state dict containing ``"T"``, ``"horizontal_wind"``, and
      ``"d_mass"``.
  static_forcing : dict[str, Array]
      Static forcing struct from :func:`init_static_forcing`.
  moist_species : dict[str, Array]
      Moisture mixing-ratio fields keyed by species name.
  dry_air_species : dict[str, Array]
      Dry-air species mass-fraction fields keyed by species name.
  h_grid : SpectralElementGrid
      Horizontal grid struct.
  v_grid : dict[str, Array]
      Vertical grid struct from :func:`init_vertical_grid`.
  physics_config : dict
      Physics configuration dict.
  model : model_info.models
      Model identifier; static JIT argument.
  pgf_formulation : pressure_gradient_options, optional
      Pressure gradient force formulation; defaults to
      ``corrected_grad_exner``.  Static JIT argument.

  Returns
  -------
  dynamics_tend : dict[str, Array]
      Dynamics tendency dict (same structure as ``dynamics``) with tendencies
      for ``"horizontal_wind"``, ``"T"``, and ``"d_mass"``.
  tracer_consistency : dict[str, Array]
      Tracer consistency struct from :func:`wrap_tracer_consist_dynamics`
      containing the mass-weighted wind flux ``u_d_mass``.
  """
  common_variables = init_common_variables(dynamics,
                                           static_forcing,
                                           moist_species,
                                           dry_air_species,
                                           h_grid,
                                           v_grid,
                                           physics_config,
                                           model)

  velocity_tend = (eval_vorticity_term(common_variables, h_grid, physics_config) +
                   eval_energy_gradient_term(common_variables, h_grid, physics_config) +
                   eval_pressure_gradient_force_term(common_variables,
                                                     h_grid,
                                                     v_grid,
                                                     physics_config,
                                                     pgf_formulation))
  temperature_tend = (eval_temperature_horiz_advection_term(common_variables, h_grid, physics_config) +
                      eval_temperature_vertical_advection_term(common_variables, h_grid, physics_config))
  d_mass_tend = eval_d_mass_divergence_term(common_variables, h_grid, physics_config)

  dynamics = wrap_dynamics(velocity_tend,
                           temperature_tend,
                           d_mass_tend,
                           model)
  tracer_consistency = wrap_tracer_consist_dynamics(eval_tracer_consistency_term(common_variables))
  return dynamics, tracer_consistency
