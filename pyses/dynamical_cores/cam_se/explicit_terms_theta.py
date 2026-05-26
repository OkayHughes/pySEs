"""
Adiabatic CAM-SE explicit step in dry-potential-temperature form.

Parallel to :mod:`pyses.dynamical_cores.cam_se.explicit_terms` (which is in
``T``-prognostic form) for the ``cam_se_stable`` / ``cam_se_whole_atmosphere_stable``
variants: the dynamics dict's thermodynamic variable is

    theta_d_d_mass  ==  mu * theta_d                                    (Sec. 1.2)

where ``mu`` is the dry layer mass and ``theta_d = T / Pi_d`` is the dry
potential temperature with the dry Exner ``Pi_d = (pi_d / p0)^(R_d / cp_d)``
evaluated at mid-level dry hydrostatic pressure ``pi_d``.

Implements

  * skew-symmetric flux-form advection of mu*theta_d                    (Sec. 4)
  * the omega/omega_d source term on the thermo equation                (Eq. 2.3)
  * energy-consistent Exner-form pressure gradient with moisture-loading
    correction                                                          (Eq. 5.2)

The dry mass-flux divergence ``X`` of (3.1) is computed once and reused for
the dry continuity tendency, the dry vertical pressure velocity ``omega_d``
(3.2), and the flux/divergence pieces of the skew-symmetric thermo advection
(4.1) -- per the reuse rule in Sec. 7.1.  Equations references are to the
attached math summary.

Hyperviscosity, vertical remap, physics coupling, and non-hydrostatic
geopotential / vertical velocity are out of scope here, matching the original
``explicit_terms.py``.
"""
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
from ..model_info import thermodynamic_variable_names
from functools import partial
_be = _get_backend()
jnp = _be.np
jit = _be.jit


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
  Pre-compute intermediate quantities shared across the theta-form tendency terms.

  Diagnoses ``theta_d``, the dry-air and moist mid-level pressures, the dry
  Exner function, temperature, virtual temperature, mixture gas constant /
  heat capacity sums, and the dry / moist horizontal mass-flux divergences and
  vertical pressure velocities ``omega_d`` and ``omega``.  The dry mass-flux
  divergence ``X`` is computed once and threaded through the dict so that the
  continuity, ``omega_d``, and skew-symmetric advection terms all reuse the
  exact same array (Sec. 7.1, rule C1).

  Parameters
  ----------
  dynamics : dict[str, Array]
      Dynamics state dict containing ``"horizontal_wind"``, ``"d_mass"`` and
      the prognostic ``mu * theta_d`` field (called ``theta_d_d_mass`` in this
      module; stored under the key ``thermodynamic_variable_names[model]``).
  static_forcing : dict[str, Array]
      Static forcing struct from :func:`init_static_forcing`; must contain
      ``"phi_surf"`` and ``"coriolis_param"``.
  moisture_species : dict[str, Array]
      Moisture mixing-ratio fields ``m^(ell)`` (kg ell / kg dry air) keyed by
      species name (Sec. 1.1).
  dry_air_species : dict[str, Array]
      Dry-air species mass-fraction fields keyed by species name.  Forwarded
      to :func:`eval_Rgas_dry` / :func:`eval_cp_dry`; reduces to constants for
      single-species CAM-SE and varies for ``cam_se_whole_atmosphere_stable``.
  h_grid : SpectralElementGrid
      Horizontal grid struct.
  v_grid : dict[str, Array]
      Vertical grid struct from :func:`init_vertical_grid`; supplies
      ``"hybrid_a_i"`` and ``"reference_surface_mass"`` for the model-top
      pressure ``p_top = hybrid_a_i[0] * reference_surface_mass``.
  physics_config : dict
      Physics configuration dict; must contain ``"p0"`` and the moisture
      species ``Rgas`` / ``cp`` look-ups for the mixture sums.
  model : model_info.models
      Model identifier (``cam_se_stable`` or ``cam_se_whole_atmosphere_stable``);
      static JIT argument used to look up the prognostic thermo key.

  Returns
  -------
  common_variables : dict[str, Array]
      Dict with the following keys (units in parentheses):

      - ``"horizontal_wind"``       -- wind vector ``(u, v)`` (m s^-1)
      - ``"theta_d_d_mass"``        -- prognostic ``mu * theta_d`` (Pa K)
      - ``"theta_d"``               -- dry potential temperature (K)
      - ``"d_mass"``                -- dry layer mass ``mu`` (Pa)
      - ``"d_pressure"``            -- moist layer pressure thickness (Pa)
      - ``"sum_species"``           -- ``sum_ell m^(ell)`` (1 + total moisture)
      - ``"temperature"``           -- ``T = theta_d * Pi_d`` (K)
      - ``"virtual_temperature"``   -- ``T_v`` (K)
      - ``"theta_v"``               -- ``T_v / Pi_d`` (K)
      - ``"R_dry"``                 -- dry-air gas constant (J kg^-1 K^-1)
      - ``"cp_dry"``                -- dry-air heat capacity (J kg^-1 K^-1)
      - ``"R_total"``               -- mixture-R numerator ``sum R^(ell) m^(ell)``
      - ``"cp_total"``              -- mixture-cp numerator ``sum cp^(ell) m^(ell)``
      - ``"pressure_midlevel_dry"`` -- ``pi_d`` at mid-levels (Pa)
      - ``"pressure_midlevel"``     -- moist pressure ``p`` at mid-levels (Pa)
      - ``"exner_dry"``             -- ``Pi_d`` at mid-levels (dimensionless)
      - ``"grad_exner_dry"``        -- ``grad(Pi_d)`` for the Exner-form PGF
      - ``"grad_sum_species"``      -- ``grad(sum_ell m^(ell))`` for the loading PGF
      - ``"omega_d"``               -- dry vertical pressure velocity (Pa s^-1)
      - ``"omega"``                 -- moist vertical pressure velocity (Pa s^-1)
      - ``"div_dry_mass_flux"``     -- ``X = div(mu * v)`` (Pa s^-1), reused
      - ``"phi"``                   -- balanced geopotential (m^2 s^-2)
      - ``"phi_surf"``              -- surface geopotential (m^2 s^-2)
      - ``"coriolis_param"``        -- Coriolis parameter (s^-1)
  """
  wind = dynamics["horizontal_wind"]
  d_mass = dynamics["d_mass"]
  theta_d_d_mass = dynamics[thermodynamic_variable_names[model]]
  theta_d = theta_d_d_mass / d_mass

  phi_surf = static_forcing["phi_surf"]
  coriolis_param = static_forcing["coriolis_param"]

  # mixture properties (Sec. 1.3); eval_cp_moist and the inline R_total below
  # are the *numerators* of the mass-weighted averages -- the sum_species
  # denominator cancels in the R/cp ratio used by the omega source term.
  R_dry = eval_Rgas_dry(dry_air_species, physics_config)
  cp_dry = eval_cp_dry(dry_air_species, physics_config)
  cp_total = eval_cp_moist(moisture_species, cp_dry, physics_config)
  R_total = 1.0 * R_dry
  for species_name in moisture_species.keys():
    R_total += (physics_config["moisture_species_Rgas"][species_name] *
                moisture_species[species_name])
  sum_species = eval_sum_species(moisture_species)

  # dry pressure: pi_d_top + cumsum(d_mass)  (mu has units of Pa so g already absorbed)
  p_top = v_grid["hybrid_a_i"][0] * v_grid["reference_surface_mass"]
  p_int_dry = eval_interface_pressure(d_mass, p_top)
  pressure_midlevel_dry = eval_midlevel_pressure(p_int_dry)
  exner_dry = eval_exner_function(pressure_midlevel_dry, R_dry, cp_dry, physics_config)
  temperature = theta_d * exner_dry

  # virtual temperature (Sec. 1.4) -- same helper as the T-based file, with T
  # diagnosed from theta_d * Pi_d above
  virtual_temperature = eval_virtual_temperature(temperature,
                                                 moisture_species,
                                                 sum_species,
                                                 R_dry,
                                                 physics_config)
  theta_v = virtual_temperature / exner_dry

  # moist pressure: p = pi_d * sum_species at the layer-thickness level, then
  # cumsum + average (Sec. 1.5 / 1.4 of the math summary)
  d_pressure = sum_species * d_mass
  p_int = eval_interface_pressure(d_pressure, p_top)
  pressure_midlevel = eval_midlevel_pressure(p_int)

  # gradients re-used by the PGF (Sec. 5.2): the same SE collocation gradient
  # is applied to both Pi_d and sum_species so the discrete cancellation in
  # Sec. 5.3 is preserved
  grad_exner_dry = horizontal_gradient_3d(exner_dry, h_grid, physics_config)
  grad_sum_species = horizontal_gradient_3d(sum_species, h_grid, physics_config)

  # dry and moist horizontal mass-flux divergences (Eq. 3.1).  X is the
  # quantity that must be reused for dry continuity, omega_d, and the
  # divergence half of the skew-symmetric advection.
  dry_mass_flux = d_mass[:, :, :, :, np.newaxis] * wind
  moist_mass_flux = d_pressure[:, :, :, :, np.newaxis] * wind
  div_dry_mass_flux = horizontal_divergence_3d(dry_mass_flux, h_grid, physics_config)
  div_moist_mass_flux = horizontal_divergence_3d(moist_mass_flux, h_grid, physics_config)

  # vertical pressure velocities (Eqs. 3.2 - 3.4) at full levels.  The
  # ``-cumsum + 0.5 * div`` combination evaluates the average of the two
  # adjacent half-level forms; u . grad(p_mid) is the same averaging applied
  # to the horizontal pressure advection piece.  Same kernel as in the T-form
  # explicit_terms.py, applied separately to the dry and moist fluxes.
  grad_p_dry = horizontal_gradient_3d(pressure_midlevel_dry, h_grid, physics_config)
  grad_p_moist = horizontal_gradient_3d(pressure_midlevel, h_grid, physics_config)
  cumsum_dry = -jnp.cumsum(div_dry_mass_flux, axis=3) + 0.5 * div_dry_mass_flux
  cumsum_moist = -jnp.cumsum(div_moist_mass_flux, axis=3) + 0.5 * div_moist_mass_flux
  omega_d = cumsum_dry + physical_dot_product(wind, grad_p_dry)
  omega = cumsum_moist + physical_dot_product(wind, grad_p_moist)

  # geopotential: use the existing hydrostatic integrator (T_v, dp_moist, p_moist);
  # this is the standard primitive-equation form, sufficient for the adiabatic
  # explicit step (the load-gradient correction in Eq. 2.5 is carried in the
  # PGF below rather than absorbed into phi).
  phi = eval_balanced_geopotential(virtual_temperature,
                                   d_pressure,
                                   pressure_midlevel,
                                   R_dry,
                                   phi_surf)

  return {"horizontal_wind": wind,
          "theta_d_d_mass": theta_d_d_mass,
          "theta_d": theta_d,
          "d_mass": d_mass,
          "d_pressure": d_pressure,
          "sum_species": sum_species,
          "temperature": temperature,
          "virtual_temperature": virtual_temperature,
          "theta_v": theta_v,
          "R_dry": R_dry,
          "cp_dry": cp_dry,
          "R_total": R_total,
          "cp_total": cp_total,
          "pressure_midlevel_dry": pressure_midlevel_dry,
          "pressure_midlevel": pressure_midlevel,
          "exner_dry": exner_dry,
          "grad_exner_dry": grad_exner_dry,
          "grad_sum_species": grad_sum_species,
          "omega_d": omega_d,
          "omega": omega,
          "div_dry_mass_flux": div_dry_mass_flux,
          "phi": phi,
          "phi_surf": phi_surf,
          "coriolis_param": coriolis_param}


@jit
def eval_thermo_skew_advection_term(common_variables,
                                    h_grid,
                                    physics_config):
  """
  Skew-symmetric flux-form advection of ``mu * theta_d`` (Eq. 4.1, 4.2).

  Returns the negative of the discrete skew-symmetric advection operator
  ``Adv^skew(mu * theta_d, v)`` so the result is directly the advective
  contribution to ``d(mu * theta_d)/dt``::

      Adv^skew(mu*theta_d, v) = 1/2 D[mu*theta_d v]                # flux
                              + 1/2 mu v . G[theta_d]              # gradient
                              + 1/2 theta_d X                      # X reuse

  ``X = div(mu * v)`` is taken straight from :func:`init_common_variables`
  rather than recomputed, preserving the discrete identity that links the
  flux- and gradient-half of the operator to the dry continuity tendency
  (Sec. 7.1, rule C1).

  Parameters
  ----------
  common_variables : dict[str, Array]
      Output of :func:`init_common_variables`; must contain
      ``"theta_d_d_mass"``, ``"theta_d"``, ``"d_mass"``, ``"horizontal_wind"``,
      and ``"div_dry_mass_flux"``.
  h_grid : SpectralElementGrid
      Horizontal grid struct.
  physics_config : dict
      Physics configuration dict.

  Returns
  -------
  thermo_advection : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      ``-Adv^skew(mu * theta_d, v)`` (Pa K s^-1).
  """
  u = common_variables["horizontal_wind"]
  theta_d = common_variables["theta_d"]
  theta_d_d_mass = common_variables["theta_d_d_mass"]
  d_mass = common_variables["d_mass"]
  X = common_variables["div_dry_mass_flux"]

  # flux half: 1/2 div(mu*theta_d v)
  flux = theta_d_d_mass[:, :, :, :, np.newaxis] * u
  div_flux = horizontal_divergence_3d(flux, h_grid, physics_config)

  # gradient half: 1/2 mu v . grad(theta_d)
  grad_theta_d = horizontal_gradient_3d(theta_d, h_grid, physics_config)
  mu_u_dot_grad_theta_d = d_mass * physical_dot_product(u, grad_theta_d)

  # X reuse: 1/2 theta_d X
  theta_d_X = theta_d * X

  return -0.5 * (div_flux + mu_u_dot_grad_theta_d + theta_d_X)


@jit
def eval_thermo_omega_source_term(common_variables):
  """
  Vertical-velocity source term on the thermodynamic equation (RHS of Eq. 2.3).

  Computes the explicit source

      mu * theta_d * [ (R / (cp * p)) * omega  -  (kappa_d / pi_d) * omega_d ]

  using the pre-computed mixture sums (``R_total``, ``cp_total`` -- the
  ``sum_species`` denominator cancels in their ratio), the dry constants
  (``R_dry``, ``cp_dry``), and the diagnosed vertical pressure velocities
  ``omega`` (moist) and ``omega_d`` (dry).

  Parameters
  ----------
  common_variables : dict[str, Array]
      Output of :func:`init_common_variables`; must contain
      ``"theta_d_d_mass"``, ``"R_total"``, ``"cp_total"``, ``"R_dry"``,
      ``"cp_dry"``, ``"pressure_midlevel"``, ``"pressure_midlevel_dry"``,
      ``"omega"``, and ``"omega_d"``.

  Returns
  -------
  thermo_source : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      The thermodynamic-equation source contribution (Pa K s^-1).
  """
  theta_d_d_mass = common_variables["theta_d_d_mass"]
  # R / (cp * p): the sum_species in the R and cp mixture means cancels, so
  # the ratio is R_total / cp_total
  R_over_cp_p = (common_variables["R_total"] /
                 (common_variables["cp_total"] * common_variables["pressure_midlevel"]))
  kappa_d_over_pi_d = ((common_variables["R_dry"] / common_variables["cp_dry"]) /
                       common_variables["pressure_midlevel_dry"])
  return theta_d_d_mass * (R_over_cp_p * common_variables["omega"] -
                           kappa_d_over_pi_d * common_variables["omega_d"])


@jit
def eval_d_mass_divergence_term(common_variables):
  """
  Horizontal divergence tendency for the dry-air layer mass (Eq. 2.1).

  Equals ``-X`` where ``X = div(mu * v)`` was pre-computed in
  :func:`init_common_variables`.  Reusing the cached ``X`` (rather than
  recomputing the divergence) guarantees the dry continuity tendency uses
  exactly the same discrete operator that fed ``omega_d`` and the skew-symmetric
  thermo advection (Sec. 7.1, rule C1).

  Parameters
  ----------
  common_variables : dict[str, Array]
      Output of :func:`init_common_variables`; must contain
      ``"div_dry_mass_flux"``.

  Returns
  -------
  d_mass_tend : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      ``-div(mu * v)`` (Pa s^-1).
  """
  return -common_variables["div_dry_mass_flux"]


@jit
def eval_energy_gradient_term(common_variables,
                              h_grid,
                              physics_config):
  """
  Gradient-of-mechanical-energy term ``-grad(KE + phi)`` (Eq. 2.4 third group).

  Identical in form to the T-based explicit step: the vector-invariant
  momentum equation pairs this gradient with the absolute-vorticity rotation
  (:func:`eval_vorticity_term`) and the pressure gradient force
  (:func:`eval_pressure_gradient_force_term`).

  Parameters
  ----------
  common_variables : dict[str, Array]
      Output of :func:`init_common_variables`; must contain
      ``"horizontal_wind"`` and ``"phi"``.
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
  return -horizontal_gradient_3d(kinetic_energy + phi, h_grid, physics_config)


@jit
def eval_vorticity_term(common_variables,
                        h_grid,
                        physics_config):
  """
  Absolute-vorticity rotation term ``(f + zeta) * u_perp`` (Eq. 2.4 first group).

  Identical to the T-based explicit step: only the horizontal wind and the
  Coriolis parameter are needed, and the dynamics state's prognostic thermo
  variable is irrelevant for this term.

  Parameters
  ----------
  common_variables : dict[str, Array]
      Output of :func:`init_common_variables`; must contain
      ``"horizontal_wind"`` and ``"coriolis_param"``.
  h_grid : SpectralElementGrid
      Horizontal grid struct.
  physics_config : dict
      Physics configuration dict.

  Returns
  -------
  vorticity_tend : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx, 2], Float]
      Wind tendency ``(f + zeta) * u_perp`` with ``u_perp = (v, -u)`` (m s^-2).
  """
  u = common_variables["horizontal_wind"]
  coriolis_parameter = common_variables["coriolis_param"]
  vorticity = horizontal_vorticity_3d(u, h_grid, physics_config)
  return jnp.stack((u[:, :, :, :, 1] * (coriolis_parameter[:, :, :, np.newaxis] + vorticity),
                    -u[:, :, :, :, 0] * (coriolis_parameter[:, :, :, np.newaxis] + vorticity)), axis=-1)


@jit
def eval_pressure_gradient_force_term(common_variables,
                                      h_grid,
                                      physics_config):
  """
  Splitform (Hamiltonian-adjoint) Exner pressure gradient force.

  The thermo equation in this file (:func:`eval_thermo_skew_advection_term`)
  is the splitform of the mu*theta_d advection -- half flux-divergence,
  half advective-plus-product-rule.  For the discrete energy / entropy
  structure to close, the PGF must be the matching splitform of the basic
  Hamiltonian PGF (the Exner form ``cp_d theta_v grad(Pi_d)`` from Eq. 5.2),
  i.e. the adjoint of the skew-symmetric thermo divergence::

      G_split = 1/2  cp_d theta_v G[Pi_d]                                 # full / flux half
              + 1/2  cp_d (G[theta_v * Pi_d] - Pi_d G[theta_v])           # split / product-rule half
              + (R_d T_v / p) pi_d G[sum_ell m^(ell)]                     # moisture loading (pointwise)

  Analytically the first two halves both equal ``cp_d theta_v G[Pi_d]``
  (product rule on G[theta_v Pi_d]); discretely they differ, and averaging
  the two preserves SBP / Hamiltonian adjointness with the splitform thermo
  divergence.  Pairing the splitform thermo with the basic
  (conservation-form) PGF (full ``cp_d theta_v G[Pi_d]``) violates the
  cancellation and is *worse* than either pure form -- this is exactly the
  HOMME ``eval_pgrad_pressure_term`` pattern, derived from CAM-SE/EAMv3
  splitform momentum equations.  The moisture-loading term is already
  pointwise and does not need splitting.

  Two extra horizontal gradients are taken here (``G[theta_v * Pi_d]`` and
  ``G[theta_v]``); ``G[Pi_d]`` is reused from :func:`init_common_variables`.

  Parameters
  ----------
  common_variables : dict[str, Array]
      Output of :func:`init_common_variables`; must contain ``"cp_dry"``,
      ``"theta_v"``, ``"exner_dry"``, ``"grad_exner_dry"``, ``"R_dry"``,
      ``"virtual_temperature"``, ``"pressure_midlevel"``,
      ``"pressure_midlevel_dry"``, and ``"grad_sum_species"``.
  h_grid : SpectralElementGrid
      Horizontal grid struct; used for the two splitform gradients
      (``G[theta_v Pi_d]`` and ``G[theta_v]``).
  physics_config : dict
      Physics configuration dict.

  Returns
  -------
  pgf_tend : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx, 2], Float]
      Pressure gradient force wind tendency (m s^-2).
  """
  cp_dry = common_variables["cp_dry"]
  R_dry = common_variables["R_dry"]
  theta_v = common_variables["theta_v"]
  T_v = common_variables["virtual_temperature"]
  p = common_variables["pressure_midlevel"]
  pi_d = common_variables["pressure_midlevel_dry"]
  exner_dry = common_variables["exner_dry"]
  grad_exner_dry = common_variables["grad_exner_dry"]
  grad_sum_species = common_variables["grad_sum_species"]

  # flux half: cp_d theta_v G[Pi_d]
  flux_half = (cp_dry[:, :, :, :, np.newaxis] *
               theta_v[:, :, :, :, np.newaxis] *
               grad_exner_dry)
  # split / product-rule half: cp_d (G[theta_v Pi_d] - Pi_d G[theta_v])
  grad_theta_v_exner = horizontal_gradient_3d(theta_v * exner_dry,
                                              h_grid, physics_config)
  grad_theta_v = horizontal_gradient_3d(theta_v, h_grid, physics_config)
  product_half = (cp_dry[:, :, :, :, np.newaxis] *
                  (grad_theta_v_exner -
                   exner_dry[:, :, :, :, np.newaxis] * grad_theta_v))
  exner_pgf = 0.5 * (flux_half + product_half)

  # moisture-loading correction (pointwise; identical to Eq. 5.2's G_load)
  loading_pgf = ((R_dry * T_v / p * pi_d)[:, :, :, :, np.newaxis] *
                 grad_sum_species)
  return -(exner_pgf + loading_pgf)


@jit
def eval_tracer_consistency_term(common_variables):
  """
  Mass-weighted horizontal wind flux ``mu * u`` for tracer consistency.

  Same shape and convention as the T-based explicit step: the dynamics-stage
  flux ``d_mass * u`` is handed to the tracer transport so dynamics and
  tracers see the same mass update (Sec. 7.1, rule C4).  Distinct from the
  per-species water-vapor mass fluxes used in (2.2).

  Parameters
  ----------
  common_variables : dict[str, Array]
      Output of :func:`init_common_variables`; must contain ``"d_mass"`` and
      ``"horizontal_wind"``.

  Returns
  -------
  d_mass_u : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx, 2], Float]
      Horizontal mass flux ``mu * u`` (Pa m s^-1).
  """
  return common_variables["d_mass"][:, :, :, :, jnp.newaxis] * common_variables["horizontal_wind"]


@partial(jit, static_argnames=["model"])
def eval_explicit_tendency(dynamics,
                           static_forcing,
                           moist_species,
                           dry_air_species,
                           h_grid,
                           v_grid,
                           physics_config,
                           model):
  """
  Assemble the full theta-form adiabatic explicit tendency.

  Drop-in parallel of ``cam_se.explicit_terms.eval_explicit_tendency`` for the
  stable variants.  Pre-computes shared diagnostics via
  :func:`init_common_variables`, sums the individual tendency terms for
  horizontal wind, ``mu * theta_d``, and dry layer mass, and packages the
  tracer-consistency mass flux.

  Parameters
  ----------
  dynamics : dict[str, Array]
      Dynamics state dict containing ``"horizontal_wind"``, ``"d_mass"``, and
      the ``mu * theta_d`` prognostic under ``thermodynamic_variable_names[model]``.
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
      Model identifier (``cam_se_stable`` or ``cam_se_whole_atmosphere_stable``);
      static JIT argument.

  Returns
  -------
  dynamics_tend : dict[str, Array]
      Dynamics tendency dict (same structure as ``dynamics``) with tendencies
      for ``"horizontal_wind"``, the model thermo key, and ``"d_mass"``.
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
                   eval_pressure_gradient_force_term(common_variables, h_grid, physics_config))
  thermo_tend = (eval_thermo_skew_advection_term(common_variables, h_grid, physics_config) +
                 eval_thermo_omega_source_term(common_variables))
  d_mass_tend = eval_d_mass_divergence_term(common_variables)

  dynamics_out = wrap_dynamics(velocity_tend,
                               thermo_tend,
                               d_mass_tend,
                               model)
  tracer_consistency = wrap_tracer_consist_dynamics(eval_tracer_consistency_term(common_variables))
  return dynamics_out, tracer_consistency
