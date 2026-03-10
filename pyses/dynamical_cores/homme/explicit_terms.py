import numpy as np
from ..._config import get_backend as _get_backend
from ..utils_3d import (midlevel_to_interface_vel,
                        midlevel_to_interface,
                        interface_to_midlevel,
                        interface_to_midlevel_vec)
from ..utils_3d import phi_to_z, z_to_g, phi_to_g, physical_dot_product
from .thermodynamics import eval_mu, eval_balanced_geopotential, eval_midlevel_pressure
from ..operators_3d import horizontal_gradient_3d, horizontal_vorticity_3d, horizontal_divergence_3d
from ..model_state import wrap_dynamics, wrap_tracer_consist_dynamics
from ..model_state import project_scalar_3d
from functools import partial
from ..model_info import hydrostatic_models, deep_atmosphere_models
_be = _get_backend()
jnp = _be.np
jit = _be.jit
device_wrapper = _be.array


@partial(jit, static_argnames=["model"])
def init_common_variables(dynamics,
                          static_forcing,
                          h_grid,
                          v_grid,
                          physics_config,
                          model):
  """
  Pre-compute intermediate quantities shared across all HOMME tendency terms.

  Evaluates interface geopotential, pressure, Exner function, radial scaling,
  mass-weighted interface velocities, horizontal divergence, and horizontal
  gradients.  Results are returned in a single dict so each term function can
  read what it needs without redundant computation.

  Parameters
  ----------
  dynamics : dict[str, Array]
      Dynamics state from :func:`wrap_dynamics`.
  static_forcing : dict[str, Array]
      Time-invariant forcing from :func:`init_static_forcing`.
  h_grid : SpectralElementGrid
      Horizontal grid struct.
  v_grid : dict[str, Array]
      Vertical grid struct from :func:`init_vertical_grid`.
  physics_config : dict
      Physics configuration dict.
  model : model_info.models
      Model identifier; selects hydrostatic/non-hydrostatic and deep/shallow
      branches.

  Returns
  -------
  common_variables : dict[str, Array]
      Dict of pre-computed quantities used by the individual tendency functions.
  """
  if model in hydrostatic_models:
    p_mid = eval_midlevel_pressure(dynamics, v_grid)
    phi_i = eval_balanced_geopotential(static_forcing["phi_surf"],
                                       p_mid,
                                       dynamics["theta_v_d_mass"],
                                       physics_config)
  else:
    phi_i = dynamics["phi_i"]
    w_i = dynamics["w_i"]

  d_mass = dynamics["d_mass"]
  u = dynamics["horizontal_wind"]
  radius_earth = physics_config["radius_earth"]
  theta_v_d_mass = dynamics["theta_v_d_mass"]

  d_mass_i = midlevel_to_interface(d_mass)
  phi = interface_to_midlevel(phi_i)
  pnh, exner, r_hat_i, mu = eval_mu(dynamics, phi_i, v_grid, physics_config, model)
  if model in deep_atmosphere_models:
    r_hat_m = interface_to_midlevel(r_hat_i)
    z = phi_to_z(phi_i, physics_config, model)
    r_m = interface_to_midlevel(z + radius_earth)
    g = z_to_g(z, physics_config, model)
  else:
    r_hat_m = device_wrapper(jnp.ones((1, 1, 1, 1)))
    r_m = radius_earth * device_wrapper(jnp.ones((1, 1, 1, 1)))
    g = physics_config["gravity"] * device_wrapper(jnp.ones((1, 1, 1, 1)))
  if model not in hydrostatic_models:
    w_m = interface_to_midlevel(w_i)
    grad_w_i = horizontal_gradient_3d(w_i, h_grid, physics_config)
  else:
    w_m = None
    grad_w_i = None

  grad_exner = horizontal_gradient_3d(exner, h_grid, physics_config) / r_hat_m
  theta_v = theta_v_d_mass / d_mass
  grad_phi_i = horizontal_gradient_3d(phi_i, h_grid, physics_config)
  v_over_r_hat_i = midlevel_to_interface_vel(u / r_hat_m[:, :, :, np.newaxis],
                                             d_mass,
                                             d_mass_i)
  div_dp = horizontal_divergence_3d(d_mass[:, :, :, :, np.newaxis] * u /
                                    r_hat_m[:, :, :, :, np.newaxis],
                                    h_grid,
                                    physics_config)
  u_i = midlevel_to_interface_vel(u, d_mass, d_mass_i)
  common_variables = {"phi_i": phi_i,
                      "phi": phi,
                      "d_mass_i": d_mass_i,
                      "pnh": pnh,
                      "exner": exner,
                      "r_hat_i": r_hat_i,
                      "mu": mu,
                      "r_hat_m": r_hat_m,
                      "r_m": r_m,
                      "g": g,
                      "coriolis_param": static_forcing["coriolis_param"],
                      "grad_exner": grad_exner,
                      "theta_v": theta_v,
                      "grad_phi_i": grad_phi_i,
                      "v_over_r_hat_i": v_over_r_hat_i,
                      "div_d_mass": div_dp,
                      "u_i": u_i,
                      "horizontal_wind": u,
                      "theta_v_d_mass": theta_v_d_mass,
                      "d_mass": d_mass}
  if model not in hydrostatic_models:
    common_variables["w_i"] = w_i
    common_variables["w_m"] = w_m
    common_variables["grad_w_i"] = grad_w_i
  if model in deep_atmosphere_models:
    common_variables["nontrad_coriolis_param"] = static_forcing["nontrad_coriolis_param"]
  else:
    common_variables["nontrad_coriolis_param"] = jnp.zeros_like(static_forcing["coriolis_param"])
  return common_variables


@jit
def eval_vorticity_term(common_variables,
                        h_grid,
                        config):
  """
  Evaluate the horizontal Coriolis + relative-vorticity tendency for momentum.

  Computes ``(f + zeta/r_hat_m) * u_perp`` in vector-invariant form, where
  ``zeta`` is the horizontal relative vorticity and ``r_hat_m`` is the
  mid-level radial scaling factor (``1`` for shallow-atmosphere models).

  Parameters
  ----------
  common_variables : dict[str, Array]
      Pre-computed quantities from :func:`init_common_variables`.
  h_grid : SpectralElementGrid
      Horizontal grid struct.
  config : dict
      Physics configuration dict.

  Returns
  -------
  vort_term : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx, 2], Float]
      Vorticity acceleration for the horizontal wind.
  """
  u = common_variables["horizontal_wind"]
  fcor = common_variables["coriolis_param"]
  vort = horizontal_vorticity_3d(u, h_grid, config)
  vort /= common_variables["r_hat_m"]
  vort_term = jnp.stack((u[:, :, :, :, 1] * (fcor[:, :, :, np.newaxis] + vort),
                         -u[:, :, :, :, 0] * (fcor[:, :, :, np.newaxis] + vort)), axis=-1)
  return vort_term


@jit
def eval_grad_kinetic_energy_h_term(common_variables,
                                    h_grid,
                                    config):
  """
  Evaluate the horizontal kinetic-energy gradient tendency for momentum.

  Computes ``-grad(KE_h) / r_hat_m`` where ``KE_h = (u^2 + v^2) / 2``
  is the horizontal kinetic energy.

  Parameters
  ----------
  common_variables : dict[str, Array]
      Pre-computed quantities from :func:`init_common_variables`.
  h_grid : SpectralElementGrid
      Horizontal grid struct.
  config : dict
      Physics configuration dict.

  Returns
  -------
  ke_h_term : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx, 2], Float]
      Horizontal kinetic-energy gradient acceleration.
  """
  u = common_variables["horizontal_wind"]
  grad_kinetic_energy = horizontal_gradient_3d((u[:, :, :, :, 0]**2 +
                                                u[:, :, :, :, 1]**2) / 2.0, h_grid, config)
  return -grad_kinetic_energy / common_variables["r_hat_m"]


@jit
def eval_grad_kinetic_energy_v_term(common_variables,
                                    h_grid,
                                    config):
  """
  Evaluate the vertical kinetic-energy gradient tendency for horizontal momentum.

  Computes ``-grad(KE_v) / r_hat_m`` where ``KE_v = w^2 / 2`` is the vertical
  kinetic energy, linearly interpolated to mid-levels from interfaces.
  Non-hydrostatic models only.

  Parameters
  ----------
  common_variables : dict[str, Array]
      Pre-computed quantities from :func:`init_common_variables`; requires
      ``"w_i"``.
  h_grid : SpectralElementGrid
      Horizontal grid struct.
  config : dict
      Physics configuration dict.

  Returns
  -------
  ke_v_term : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx, 2], Float]
      Vertical kinetic-energy gradient acceleration for the horizontal wind.
  """
  w_i = common_variables["w_i"]
  w_sq_m = interface_to_midlevel(w_i * w_i) / 2.0
  w2_grad_sph = horizontal_gradient_3d(w_sq_m, h_grid, config) / common_variables["r_hat_m"]
  return -w2_grad_sph


@jit
def eval_w_vorticity_correction_term(common_variables):
  """
  Evaluate the vertical-vorticity correction to horizontal momentum.

  Computes the interface-to-midlevel average of ``w * grad(w) / r_hat_m``,
  which arises from the vector-invariant form of the momentum equation when
  the flow has a non-negligible vertical component.  Non-hydrostatic models only.

  Parameters
  ----------
  common_variables : dict[str, Array]
      Pre-computed quantities from :func:`init_common_variables`; requires
      ``"w_i"`` and ``"grad_w_i"``.

  Returns
  -------
  w_vort_term : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx, 2], Float]
      Vertical-vorticity correction to the horizontal wind tendency.
  """
  w_grad_w_m = interface_to_midlevel_vec(common_variables["w_i"][:, :, :, :, np.newaxis] *
                                         common_variables["grad_w_i"])
  w_grad_w_m /= common_variables["r_hat_m"][:, :, :, :, np.newaxis]
  return w_grad_w_m


@jit
def eval_u_metric_term(common_variables):
  """
  Evaluate the metric (curvature) correction to horizontal momentum.

  Computes ``-w_m * u / r_m``, the correction to horizontal wind arising
  from spherical-geometry metric terms in the deep-atmosphere equations.

  Parameters
  ----------
  common_variables : dict[str, Array]
      Pre-computed quantities from :func:`init_common_variables`; requires
      ``"w_m"`` and ``"r_m"``.

  Returns
  -------
  u_metric : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx, 2], Float]
      Metric correction to the horizontal wind tendency.
  """
  return -(common_variables["w_m"][:, :, :, :, np.newaxis] * common_variables["horizontal_wind"] /
           common_variables["r_m"][:, :, :, np.newaxis])


@jit
def eval_u_nct_term(common_variables):
  """
  Evaluate the non-traditional Coriolis correction to horizontal momentum.

  Computes ``-w_m * f_cos`` for the zonal component (zero for meridional),
  where ``f_cos = 2 Omega cos(lat)`` is the non-traditional Coriolis parameter.
  Deep-atmosphere models only.

  Parameters
  ----------
  common_variables : dict[str, Array]
      Pre-computed quantities from :func:`init_common_variables`; requires
      ``"w_m"`` and ``"nontrad_coriolis_param"``.

  Returns
  -------
  u_nct : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx, 2], Float]
      Non-traditional Coriolis correction to the horizontal wind tendency.
  """
  w_m = common_variables["w_m"]
  fcorcos = common_variables["nontrad_coriolis_param"]
  return -jnp.stack((w_m, jnp.zeros_like(w_m)), axis=-1) * fcorcos[:, :, :, np.newaxis, np.newaxis]


@jit
def eval_pgrad_pressure_term(common_variables,
                             h_grid,
                             config):
  """
  Evaluate the Exner-pressure gradient force for horizontal momentum.

  Uses the symmetrised form ``-cp * (theta_v * grad(pi) + grad(theta_v * pi) -
  pi * grad(theta_v)) / 2`` to improve discrete energy conservation.

  Parameters
  ----------
  common_variables : dict[str, Array]
      Pre-computed quantities from :func:`init_common_variables`; requires
      ``"theta_v"``, ``"exner"``, ``"grad_exner"``, and ``"r_hat_m"``.
  h_grid : SpectralElementGrid
      Horizontal grid struct.
  config : dict
      Physics configuration dict with ``"cp"``.

  Returns
  -------
  pgrad_p : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx, 2], Float]
      Pressure-gradient acceleration for the horizontal wind.
  """
  theta_v = common_variables["theta_v"]
  exner = common_variables["exner"]
  r_hat_m = common_variables["r_hat_m"]
  grad_p_term_1 = config["cp"] * theta_v[:, :, :, :, np.newaxis] * common_variables["grad_exner"]
  grad_theta_v_exner = horizontal_gradient_3d(theta_v * exner, h_grid, config) / r_hat_m
  grad_theta_v = horizontal_gradient_3d(theta_v, h_grid, config) / r_hat_m
  grad_p_term_2 = config["cp"] * (grad_theta_v_exner - exner[:, :, :, :, np.newaxis] * grad_theta_v)
  return -(grad_p_term_1 + grad_p_term_2) / 2.0


@jit
def eval_pgrad_phi_term(common_variables):
  """
  Evaluate the geopotential-gradient pressure force for horizontal momentum.

  Computes the interface-to-midlevel average of ``-mu * grad(phi_i) / r_hat_m``,
  where ``mu = dp/d(d_mass)`` couples the non-hydrostatic pressure to the
  geopotential gradient.

  Parameters
  ----------
  common_variables : dict[str, Array]
      Pre-computed quantities from :func:`init_common_variables`; requires
      ``"mu"``, ``"grad_phi_i"``, and ``"r_hat_m"``.

  Returns
  -------
  pgrad_phi : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx, 2], Float]
      Geopotential-gradient pressure-force acceleration for the horizontal wind.
  """
  pgf_grad_phi_m = interface_to_midlevel_vec(common_variables["mu"][:, :, :, :, np.newaxis] *
                                             common_variables["grad_phi_i"])
  pgf_grad_phi_m /= common_variables["r_hat_m"][:, :, :, :, np.newaxis]
  return -pgf_grad_phi_m


@jit
def eval_w_advection_term(common_variables):
  """
  Evaluate the horizontal advection of vertical velocity at interfaces.

  Computes ``-v/r_hat_i · grad(w_i)`` at vertical interfaces, where the
  mass-weighted interface velocity ``v_over_r_hat_i`` is used.

  Parameters
  ----------
  common_variables : dict[str, Array]
      Pre-computed quantities from :func:`init_common_variables`; requires
      ``"v_over_r_hat_i"`` and ``"grad_w_i"``.

  Returns
  -------
  w_adv : Array[tuple[elem_idx, gll_idx, gll_idx, ilev_idx], Float]
      Horizontal advection of ``w`` at interfaces.
  """
  v_over_r_hat_i = common_variables["v_over_r_hat_i"]
  grad_w_i = common_variables["grad_w_i"]
  v_grad_w_i = (v_over_r_hat_i[:, :, :, :, 0] * grad_w_i[:, :, :, :, 0] +
                v_over_r_hat_i[:, :, :, :, 1] * grad_w_i[:, :, :, :, 1])
  return -v_grad_w_i


@jit
def eval_w_metric_term(common_variables):
  """
  Evaluate the metric (curvature) correction to vertical velocity at interfaces.

  Computes the mass-weighted interface average of ``(u^2 + v^2) / r_m``,
  the centrifugal contribution to the vertical momentum equation from
  spherical-geometry metric terms.  Deep-atmosphere models only.

  Parameters
  ----------
  common_variables : dict[str, Array]
      Pre-computed quantities from :func:`init_common_variables`; requires
      ``"horizontal_wind"``, ``"r_m"``, ``"d_mass"``, and ``"d_mass_i"``.

  Returns
  -------
  w_metric : Array[tuple[elem_idx, gll_idx, gll_idx, ilev_idx], Float]
      Metric correction to the interface vertical-velocity tendency.
  """
  v_sq_over_r_i = midlevel_to_interface_vel(common_variables["horizontal_wind"]**2 / common_variables["r_m"],
                                            common_variables["d_mass"],
                                            common_variables["d_mass_i"])
  return (v_sq_over_r_i[:, :, :, :, 0] + v_sq_over_r_i[:, :, :, :, 1])


@jit
def eval_w_nct_term(common_variables):
  """
  Evaluate the non-traditional Coriolis correction to vertical velocity.

  Computes ``u_i * f_cos`` at interfaces, the contribution to the vertical
  momentum equation from the non-traditional Coriolis parameter
  ``f_cos = 2 Omega cos(lat)``.  Deep-atmosphere models only.

  Parameters
  ----------
  common_variables : dict[str, Array]
      Pre-computed quantities from :func:`init_common_variables`; requires
      ``"u_i"`` and ``"nontrad_coriolis_param"``.

  Returns
  -------
  w_nct : Array[tuple[elem_idx, gll_idx, gll_idx, ilev_idx], Float]
      Non-traditional Coriolis correction to the interface vertical-velocity
      tendency.
  """
  fcorcos = common_variables["nontrad_coriolis_param"]
  return common_variables["u_i"][:, :, :, :, 0] * fcorcos[:, :, :, np.newaxis]


@jit
def eval_w_buoyancy_term(common_variables):
  """
  Evaluate the buoyancy tendency for vertical velocity at interfaces.

  Computes ``-g * (1 - mu)`` at interfaces, where ``g`` is local gravity and
  ``mu = dp/d(d_mass)`` is the non-hydrostatic coefficient.  For hydrostatic
  models this term is zero.

  Parameters
  ----------
  common_variables : dict[str, Array]
      Pre-computed quantities from :func:`init_common_variables`; requires
      ``"g"`` and ``"mu"``.

  Returns
  -------
  w_buoy : Array[tuple[elem_idx, gll_idx, gll_idx, ilev_idx], Float]
      Buoyancy acceleration at vertical interfaces.
  """
  return -common_variables["g"] * (1 - common_variables["mu"])


@jit
def eval_phi_advection_term(common_variables):
  """
  Evaluate the horizontal advection of interface geopotential.

  Computes ``-v/r_hat_i · grad(phi_i)`` at vertical interfaces, using the
  mass-weighted interface velocity ``v_over_r_hat_i``.

  Parameters
  ----------
  common_variables : dict[str, Array]
      Pre-computed quantities from :func:`init_common_variables`; requires
      ``"v_over_r_hat_i"`` and ``"grad_phi_i"``.

  Returns
  -------
  phi_adv : Array[tuple[elem_idx, gll_idx, gll_idx, ilev_idx], Float]
      Horizontal advection of ``phi_i`` at interfaces.
  """
  v_over_r_hat_i = common_variables["v_over_r_hat_i"]
  grad_phi_i = common_variables["grad_phi_i"]
  v_grad_phi_i = (v_over_r_hat_i[:, :, :, :, 0] * grad_phi_i[:, :, :, :, 0] +
                  v_over_r_hat_i[:, :, :, :, 1] * grad_phi_i[:, :, :, :, 1])
  return -v_grad_phi_i


@jit
def eval_phi_acceleration_v_term(common_variables):
  """
  Evaluate the vertical-velocity contribution to the interface geopotential tendency.

  Computes ``g * w_i`` at interfaces — the rate of change of geopotential due
  to vertical motion against gravity.

  Parameters
  ----------
  common_variables : dict[str, Array]
      Pre-computed quantities from :func:`init_common_variables`; requires
      ``"g"`` and ``"w_i"``.

  Returns
  -------
  phi_accel_v : Array[tuple[elem_idx, gll_idx, gll_idx, ilev_idx], Float]
      Vertical-motion contribution to the ``phi_i`` tendency.
  """
  return common_variables["g"] * common_variables["w_i"]


@jit
def eval_theta_v_divergence_term(common_variables,
                                 h_grid,
                                 config):
  """
  Evaluate the virtual-potential-temperature divergence tendency.

  Uses a symmetrised advection form that combines flux divergence and
  advective components to improve discrete energy conservation:
  ``-(div(u * theta_v_d_mass) + theta_v * div(d_mass * u) +
  d_mass * u · grad(theta_v)) / (2 * r_hat_m)``.

  Parameters
  ----------
  common_variables : dict[str, Array]
      Pre-computed quantities from :func:`init_common_variables`; requires
      ``"r_hat_m"``, ``"theta_v"``, ``"theta_v_d_mass"``, ``"horizontal_wind"``,
      ``"div_d_mass"``, and ``"d_mass"``.
  h_grid : SpectralElementGrid
      Horizontal grid struct.
  config : dict
      Physics configuration dict.

  Returns
  -------
  theta_v_tend : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Tendency of virtual potential temperature times layer mass.
  """
  r_hat_m = common_variables["r_hat_m"]
  theta_v = common_variables["theta_v"]
  u = common_variables["horizontal_wind"]
  div_d_mass = common_variables["div_d_mass"]
  d_mass = common_variables["d_mass"]
  v_theta_v = common_variables["horizontal_wind"] * common_variables["theta_v_d_mass"][:, :, :, :, np.newaxis]
  v_theta_v /= r_hat_m
  div_v_theta_v = horizontal_divergence_3d(v_theta_v, h_grid, config) / 2.0
  grad_theta_v = horizontal_gradient_3d(theta_v, h_grid, config)
  grad_theta_v /= r_hat_m

  div_v_theta_v += (theta_v * div_d_mass + (d_mass * (u[:, :, :, :, 0] * grad_theta_v[:, :, :, :, 0] +
                                                      u[:, :, :, :, 1] * grad_theta_v[:, :, :, :, 1]))) / 2.0
  return -div_v_theta_v


@jit
def eval_d_mass_divergence_term(common_variables):
  """
  Evaluate the layer-mass continuity tendency.

  Returns ``-div(d_mass * u / r_hat_m)``, the negative horizontal divergence
  of the mass flux, which drives changes in layer mass.

  Parameters
  ----------
  common_variables : dict[str, Array]
      Pre-computed quantities from :func:`init_common_variables`; requires
      ``"div_d_mass"``.

  Returns
  -------
  d_mass_tend : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Layer-mass continuity tendency.
  """
  return -common_variables["div_d_mass"]


@jit
def eval_tracer_velocity_term(common_variables):
  """
  Evaluate the mass-weighted tracer-consistency flux.

  Computes ``d_mass * u / r_hat_m``, the flux used to transport tracers
  consistently with the dynamical-core mass update.

  Parameters
  ----------
  common_variables : dict[str, Array]
      Pre-computed quantities from :func:`init_common_variables`; requires
      ``"d_mass"``, ``"horizontal_wind"``, and ``"r_hat_m"``.

  Returns
  -------
  tracer_flux : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx, 2], Float]
      Mass-weighted horizontal tracer-consistency flux.
  """
  return (common_variables["d_mass"][:, :, :, :, jnp.newaxis] *
          common_variables["horizontal_wind"] / common_variables["r_hat_m"])


@partial(jit, static_argnames=["model"])
def eval_explicit_tendency(dynamics,
                           static_forcing,
                           h_grid,
                           v_grid,
                           config,
                           model):
  """
  Evaluate the full explicit adiabatic tendency for HOMME.

  Combines all tendency terms: vorticity, horizontal and vertical kinetic-energy
  gradients, pressure-gradient forces, geopotential, virtual potential
  temperature, and layer-mass continuity.  Non-hydrostatic and deep-atmosphere
  terms are included only when applicable to ``model``.

  Parameters
  ----------
  dynamics : dict[str, Array]
      Current dynamics state from :func:`wrap_dynamics`.
  static_forcing : dict[str, Array]
      Time-invariant forcing from :func:`init_static_forcing`.
  h_grid : SpectralElementGrid
      Horizontal grid struct.
  v_grid : dict[str, Array]
      Vertical grid struct from :func:`init_vertical_grid`.
  config : dict
      Physics configuration dict.
  model : model_info.models
      Model identifier; static JIT argument.

  Returns
  -------
  dynamics_tend : dict[str, Array]
      Dynamics tendency dict from :func:`wrap_dynamics`.
  tracer_consistency : dict[str, Array]
      Tracer-consistency flux struct from :func:`wrap_tracer_consist_dynamics`.
  """

  common_variables = init_common_variables(dynamics,
                                           static_forcing,
                                           h_grid,
                                           v_grid,
                                           config,
                                           model)

  u_tend = (eval_vorticity_term(common_variables, h_grid, config) +
            eval_grad_kinetic_energy_h_term(common_variables, h_grid, config) +
            eval_pgrad_pressure_term(common_variables, h_grid, config) +
            eval_pgrad_phi_term(common_variables))

  if model not in hydrostatic_models:
    u_tend += (eval_grad_kinetic_energy_v_term(common_variables, h_grid, config) +
               eval_w_vorticity_correction_term(common_variables))
    w_tend = (eval_w_advection_term(common_variables) +
              eval_w_buoyancy_term(common_variables))
    phi_tend = (eval_phi_advection_term(common_variables) +
                eval_phi_acceleration_v_term(common_variables))
  else:
    w_tend = None
    phi_tend = None

  if model in deep_atmosphere_models:
      u_tend += (eval_u_metric_term(common_variables) +
                 eval_u_nct_term(common_variables))
      w_tend += (eval_w_metric_term(common_variables) +
                 eval_w_nct_term(common_variables))

  theta_v_d_mass_tend = eval_theta_v_divergence_term(common_variables, h_grid, config)
  d_mass_tend = eval_d_mass_divergence_term(common_variables)
  dynamics = wrap_dynamics(u_tend,
                           theta_v_d_mass_tend,
                           d_mass_tend,
                           model,
                           phi_i=phi_tend,
                           w_i=w_tend)
  tracer_consistency = wrap_tracer_consist_dynamics(eval_tracer_velocity_term(common_variables))
  return dynamics, tracer_consistency


@partial(jit, static_argnames=["dims", "model"])
def eval_energy_quantities(dynamics,
                           static_forcing,
                           h_grid,
                           v_grid,
                           config,
                           dims,
                           model):
  """
  Compute discrete energy transfer pairs and empirical energy tendencies.

  Evaluates all pairwise energy exchanges (KE-KE, KE-PE, KE-IE, PE-PE) between
  kinetic, potential, and internal energy reservoirs.  Also computes empirical
  total energy tendencies from the explicit tendency by direct inner products.
  Used for diagnosing discrete energy conservation.

  Parameters
  ----------
  dynamics : dict[str, Array]
      Current dynamics state from :func:`wrap_dynamics`.
  static_forcing : dict[str, Array]
      Time-invariant forcing from :func:`init_static_forcing`.
  h_grid : SpectralElementGrid
      Horizontal grid struct.
  v_grid : dict[str, Array]
      Vertical grid struct from :func:`init_vertical_grid`.
  config : dict
      Physics configuration dict.
  dims : frozendict[str, int]
      Grid dimension tuple used for DSS projection; static JIT argument.
  model : model_info.models
      Model identifier; static JIT argument.

  Returns
  -------
  pairs : dict[str, tuple[Array, Array]]
      Dict of ``(a_term, b_term)`` pairs; each pair should sum to zero for
      exact discrete energy conservation.
  empirical_tendencies : dict[str, Array]
      Dict with keys ``"ke"``, ``"ie"``, ``"pe"`` containing the total
      tendency of each energy reservoir integrated over the column.
  """
  common_variables = init_common_variables(dynamics,
                                           static_forcing,
                                           h_grid,
                                           v_grid,
                                           config,
                                           model)

  # !!!!!!!!!!!!!!!!!!!!!!!!!!
  # todo: incorporate mu correction.
  # !!!!!!!!!!!!!!!!!!!!!!!!!!
  d_mass_i = common_variables["d_mass_i"]

  d_mass_i_integral = jnp.concatenate((d_mass_i[:, :, :, 0:1] / 2.0,
                                       d_mass_i[:, :, :, 1:-1],
                                       d_mass_i[:, :, :, -1:] / 2.0), axis=-1)

  u = dynamics["horizontal_wind"]
  d_mass = dynamics["d_mass"]
  w_i = dynamics["w_i"]
  u1 = u[:, :, :, :, 0]
  u2 = u[:, :, :, :, 1]
  u_sq = physical_dot_product(u, u)
  g = common_variables["g"]
  mu = common_variables["mu"]
  exner = common_variables["exner"]
  phi = common_variables["phi"]

  grad_kinetic_energy_h = eval_grad_kinetic_energy_h_term(common_variables, h_grid, config)
  d_mass_divergence = eval_d_mass_divergence_term(common_variables)
  phi_acceleration_v = eval_phi_acceleration_v_term(common_variables)
  w_buoyancy = eval_w_buoyancy_term(common_variables)
  pgrad_pressure = eval_pgrad_pressure_term(common_variables, h_grid, config)
  pgrad_phi = eval_pgrad_phi_term(common_variables)
  theta_v_divergence = eval_theta_v_divergence_term(common_variables, h_grid, config)
  w_vorticity = eval_w_vorticity_correction_term(common_variables)
  w_advection = eval_w_advection_term(common_variables)
  u_metric = eval_u_metric_term(common_variables)
  w_metric = eval_w_metric_term(common_variables)
  u_nct = eval_u_nct_term(common_variables)
  w_nct = eval_w_nct_term(common_variables)
  grad_kinetic_energy_v = eval_grad_kinetic_energy_v_term(common_variables, h_grid, config)
  vorticity = eval_vorticity_term(common_variables, h_grid, config)
  phi_advection = eval_phi_advection_term(common_variables)

  ke_ke_1_a = jnp.sum(d_mass * physical_dot_product(u, grad_kinetic_energy_h), axis=-1)
  ke_ke_1_b = jnp.sum(1.0 / 2.0 * u_sq * project_scalar_3d(d_mass_divergence, h_grid, dims), axis=-1)

  ke_ke_2_a = jnp.sum(d_mass * (u1 * grad_kinetic_energy_v[:, :, :, :, 0] +
                                u2 * grad_kinetic_energy_v[:, :, :, :, 1]), axis=-1)
  ke_ke_2_b = jnp.sum(1.0 / 2.0 * interface_to_midlevel(w_i**2) * d_mass_divergence, axis=-1)

  ke_pe_1_a = jnp.sum(d_mass_i_integral * w_i * (w_buoyancy - mu * g), axis=-1)
  ke_pe_1_b = jnp.sum(d_mass_i_integral * phi_acceleration_v, axis=-1)

  ke_ie_1_a = jnp.sum(d_mass_i_integral * -mu * phi_acceleration_v, axis=-1)
  ke_ie_1_b = jnp.sum(d_mass_i_integral * w_i * (w_buoyancy + g), axis=-1)

  ke_ie_2_a = jnp.sum(d_mass * (u1 * pgrad_pressure[:, :, :, :, 0] +
                                u2 * pgrad_pressure[:, :, :, :, 1]), axis=-1)
  ke_ie_2_b = jnp.sum(config["cp"] * exner * theta_v_divergence, axis=-1)

  ke_ie_3_a = jnp.sum(d_mass * (u1 * pgrad_phi[:, :, :, :, 0] +
                                u2 * pgrad_phi[:, :, :, :, 1]), axis=-1)
  ke_ie_3_b = jnp.sum(d_mass_i_integral * -mu * phi_advection, axis=-1)

  ke_ke_3_a = jnp.sum(d_mass * (u1 * w_vorticity[:, :, :, :, 0] +
                                u2 * w_vorticity[:, :, :, :, 1]), axis=-1)
  ke_ke_3_b = jnp.sum(d_mass_i_integral * w_i * w_advection, axis=-1)

  ke_ke_4_a = jnp.sum(d_mass * u1 * vorticity[:, :, :, :, 0], axis=-1)
  ke_ke_4_b = jnp.sum(d_mass * u2 * vorticity[:, :, :, :, 1], axis=-1)

  pe_pe_1_a = jnp.sum(phi * d_mass_divergence, axis=-1)
  pe_pe_1_b = jnp.sum(d_mass_i_integral * phi_advection, axis=-1)

  ke_ke_5_a = jnp.sum(d_mass * (u1 * u_metric[:, :, :, :, 0] +
                                u2 * u_metric[:, :, :, :, 1]), axis=-1)
  ke_ke_5_b = jnp.sum(d_mass_i_integral * w_i * w_metric, axis=-1)

  ke_ke_6_a = jnp.sum(d_mass * (u1 * u_nct[:, :, :, :, 0] +
                                u2 * u_nct[:, :, :, :, 1]), axis=-1)
  ke_ke_6_b = jnp.sum(d_mass_i_integral * w_i * w_nct, axis=-1)

  tends, _ = eval_explicit_tendency(dynamics, static_forcing, h_grid, v_grid, config, model)
  u_tend = tends["horizontal_wind"]

  ke_tend_emp = jnp.sum(d_mass * (u1 * u_tend[:, :, :, :, 0] +
                                  u2 * u_tend[:, :, :, :, 1]), axis=-1)
  ke_tend_emp += jnp.sum(d_mass_i_integral * w_i * tends["w_i"], axis=-1)

  ke_tend_emp += jnp.sum(u_sq / 2.0 * tends["d_mass"], axis=-1)
  ke_tend_emp += jnp.sum(interface_to_midlevel(w_i**2) / 2.0 * tends["d_mass"], axis=-1)

  pe_tend_emp = jnp.sum(phi * tends["d_mass"], axis=-1)
  pe_tend_emp += jnp.sum(d_mass_i_integral * tends["phi_i"], axis=-1)

  ie_tend_emp = jnp.sum(config["cp"] * exner * tends["theta_v_d_mass"], axis=-1)
  ie_tend_emp -= jnp.sum(mu * d_mass_i_integral * tends["phi_i"], axis=-1)

  pairs = {"ke_ke_1": (ke_ke_1_a, ke_ke_1_b),
           "ke_ke_2": (ke_ke_2_a, ke_ke_2_b),
           "ke_ke_3": (ke_ke_3_a, ke_ke_3_b),
           "ke_ke_4": (ke_ke_4_a, ke_ke_4_b),
           "ke_ke_5": (ke_ke_5_a, ke_ke_5_b),
           "ke_ke_6": (ke_ke_6_a, ke_ke_6_b),
           "ke_pe_1": (ke_pe_1_a, ke_pe_1_b),
           "pe_pe_1": (pe_pe_1_a, pe_pe_1_b),
           "ke_ie_1": (ke_ie_1_a, ke_ie_1_b),
           "ke_ie_2": (ke_ie_2_a, ke_ie_2_b),
           "ke_ie_3": (ke_ie_3_a, ke_ie_3_b)}
  empirical_tendencies = {"ke": ke_tend_emp,
                          "ie": ie_tend_emp,
                          "pe": pe_tend_emp}
  return pairs, empirical_tendencies


@partial(jit, static_argnames=["model"])
def correct_state(dynamics,
                  static_forcing,
                  dt,
                  config,
                  model):
  """
  Apply the lower-boundary conservation correction to the dynamics state.

  For non-hydrostatic models, adjusts the lowest-level horizontal wind and
  interface vertical velocity so that the kinematic lower-boundary condition
  is satisfied.  For hydrostatic models this is a no-op.

  Parameters
  ----------
  dynamics : dict[str, Array]
      Dynamics state to be corrected.
  static_forcing : dict[str, Array]
      Time-invariant forcing; ``"grad_phi_surf"`` is used for the surface
      normal constraint.
  dt : float
      Timestep size (s).
  config : dict
      Physics configuration dict.
  model : model_info.models
      Model identifier; static JIT argument.

  Returns
  -------
  dynamics_corrected : dict[str, Array]
      Dynamics state with corrected lowest-level wind and vertical velocity.
  """
  if model in hydrostatic_models:
    return dynamics
  u_lowest_new, w_lowest_new, mu_update = eval_lower_boundary_correction(dynamics,
                                                                         static_forcing,
                                                                         dt,
                                                                         config,
                                                                         model)
  u_new = jnp.concatenate((dynamics["horizontal_wind"][:, :, :, :-1, :],
                           u_lowest_new[:, :, :, np.newaxis, :]), axis=-2)
  if model not in hydrostatic_models:
    w_new = jnp.concatenate((dynamics["w_i"][:, :, :, :-1],
                             w_lowest_new[:, :, :, np.newaxis]), axis=-1)
  else:
    w_new = dynamics["w_i"]
  return wrap_dynamics(u_new,
                       dynamics["theta_v_d_mass"],
                       dynamics["d_mass"],
                       model,
                       phi_i=dynamics["phi_i"],
                       w_i=w_new)


@partial(jit, static_argnames=["model"])
def eval_lower_boundary_correction(dynamics,
                                   static_forcing,
                                   dt,
                                   config,
                                   model):
  """
  Compute the lower-boundary correction to wind and vertical velocity.

  Determines a scalar multiplier ``mu_surf`` such that, after the correction,
  the lowest-level wind satisfies the kinematic no-penetration condition
  ``w_surf = u · grad(phi_surf) / g``.  Returns the corrected lowest-level
  horizontal wind, vertical velocity at the surface interface, and ``mu_surf``.

  Parameters
  ----------
  dynamics : dict[str, Array]
      Current dynamics state.
  static_forcing : dict[str, Array]
      Time-invariant forcing; requires ``"grad_phi_surf"`` and ``"phi_surf"``.
  dt : float
      Timestep size (s) used to scale the correction.
  config : dict
      Physics configuration dict.
  model : model_info.models
      Model identifier; hydrostatic models return the uncorrected values.

  Returns
  -------
  u_corrected : Array[tuple[elem_idx, gll_idx, gll_idx, 2], Float]
      Corrected lowest-level horizontal wind.
  w_corrected : Array[tuple[elem_idx, gll_idx, gll_idx], Float] or float
      Corrected surface-interface vertical velocity (``0.0`` for hydrostatic).
  mu_surf : Array[tuple[elem_idx, gll_idx, gll_idx], Float] or float
      Boundary correction multiplier (``1.0`` for hydrostatic).
  """
  # we need to pass in original state. Something is wrong here.
  if model in hydrostatic_models:
    u_corrected = dynamics["horizontal_wind"][:, :, :, -1, :]
    w_corrected = 0.0
    mu_surf = 1.0
  else:
    u_lowest = dynamics["horizontal_wind"][:, :, :, -1, :]
    w_lowest = dynamics["w_i"][:, :, :, -1]
    grad_phi_surf = static_forcing["grad_phi_surf"]
    g_surf = phi_to_g(static_forcing["phi_surf"], config, model)
    mu_surf = ((u_lowest[:, :, :, 0] * grad_phi_surf[:, :, :, 0] +
                u_lowest[:, :, :, 1] * grad_phi_surf[:, :, :, 1]) / g_surf - w_lowest)
    mu_surf /= (g_surf + 1.0 / (2.0 * g_surf) * (grad_phi_surf[:, :, :, 0]**2 +
                                                 grad_phi_surf[:, :, :, 1]**2))
    mu_surf /= dt
    mu_surf += 1.0

    w_corrected = w_lowest + dt * g_surf * (mu_surf - 1)
    u_corrected = u_lowest - dt * (mu_surf[:, :, :, np.newaxis] - 1) * grad_phi_surf / 2.0

  return u_corrected, w_corrected, mu_surf
