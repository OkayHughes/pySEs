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
from ..model_info import (hydrostatic_models,
                          deep_atmosphere_models,
                          vertically_buoyant_models,
                          eulerian_models)
_be = _get_backend()
jnp = _be.np
jit = _be.jit
device_wrapper = _be.array


# ---------------------------------------------------------------------------
# Vertical finite-difference and SB81 operator helpers (Eulerian height path)
# ---------------------------------------------------------------------------
# These follow Taylor et al. (2020), eqns (30)-(41).  The Lorenz staggering
# places ``midpoint quantities`` (u, Theta, d_mass, p, Pi, dphi/ds) at the
# ``i`` levels and ``interface quantities`` (phi, w, pi, s_dot, S_dot, mu,
# dp/ds) at the ``i + 1/2`` levels.
#
# We use the convention that the last array axis ranges over levels, with
# midpoint arrays of length ``n`` and interface arrays of length ``n + 1``.
# ``delta_s_m`` (shape ``(..., n)``) gives the layer thicknesses
# ``Delta s_i = s_{i+1/2} - s_{i-1/2}`` and ``delta_s_i`` (shape ``(..., n+1)``)
# gives the interface separations ``Delta s_{i+1/2} = s_{i+1} - s_i`` with
# the boundary conventions ``Delta s_{1/2} = Delta s_1`` and
# ``Delta s_{n+1/2} = Delta s_n`` (Taylor et al. 2020 sec. 4).
#
# The code's ``d_mass`` array holds ``(d pi/d s) * Delta s_i`` at midpoints,
# so ``(d pi/d s)`` at midpoints is recovered as ``d_mass / delta_s_m`` and
# the eqn (31) interface average ``(d pi/d s)-bar`` at interfaces is the
# weighted mean of ``d_mass / delta_s_m`` produced by
# :func:`midlevel_to_interface`.


def _vertical_diff_i_to_m(q_i, delta_s_m):
  """``(dq/ds)_i`` from an interface quantity ``q_i`` via eqn (32) left form.

  Output is at midpoints (length one less than ``q_i`` along the last axis).
  """
  return (q_i[..., 1:] - q_i[..., :-1]) / delta_s_m


def _vertical_diff_m_to_i(q_m, delta_s_i,
                          q_top=None, q_surf=None):
  """``(dq/ds)_{i+1/2}`` from a midpoint quantity ``q_m`` via eqn (32) right.

  Optional boundary values ``q_top`` and ``q_surf`` enable the one-sided
  differencing of eqn (33).  When omitted the boundary derivatives are set
  to zero -- appropriate when the result is later multiplied by ``Sdot``
  (which vanishes at top and surface).
  """
  dq_interior = (q_m[..., 1:] - q_m[..., :-1]) / delta_s_i[..., 1:-1]
  if q_top is not None:
    dq_top = (q_m[..., 0:1] - q_top[..., jnp.newaxis]) / (0.5 * delta_s_i[..., 0:1])
  else:
    dq_top = jnp.zeros_like(q_m[..., 0:1])
  if q_surf is not None:
    dq_surf = (q_surf[..., jnp.newaxis] - q_m[..., -1:]) / (0.5 * delta_s_i[..., -1:])
  else:
    dq_surf = jnp.zeros_like(q_m[..., -1:])
  return jnp.concatenate((dq_top, dq_interior, dq_surf), axis=-1)


def _zero_boundary_interface(q_i):
  """Set the top and surface interface values of ``q_i`` to exactly zero.

  Used to enforce ``Sdot_{1/2} = Sdot_{n+1/2} = 0`` in the Eulerian height
  coordinate.  The KE-KE cancellation in eqn (56) of Taylor et al. relies on
  this boundary condition holding bit-exactly, not just to truncation error.
  """
  interior = q_i[..., 1:-1]
  zeros_top = jnp.zeros_like(q_i[..., 0:1])
  zeros_surf = jnp.zeros_like(q_i[..., -1:])
  return jnp.concatenate((zeros_top, interior, zeros_surf), axis=-1)


def _sb81_advect_midpoint(q_m, s_dot_S, d_mass):
  """SB81 midpoint vertical advection operator (Taylor et al. eqn 40).

  Returns ``[s_dot d/ds] q`` at midpoints.  Expects ``s_dot_S`` to vanish
  at top and surface (call :func:`_zero_boundary_interface` first), so that
  the missing-neighbour contributions at the boundary midpoints have zero
  coefficient.

  ``(d pi/d s) [s_dot d/ds] q |_i  =  (1/Delta s_i) * (1/2)
        * [Sdot_{i+1/2} (q_{i+1} - q_i) + Sdot_{i-1/2} (q_i - q_{i-1})]``

  Dividing by ``(d pi/d s) = d_mass / delta_s_m`` and the explicit
  ``1/Delta s_i`` cancels into ``1/d_mass``.
  """
  # q_{i+1} - q_i for i in [0, n-2], padded with zero at i = n-1 so the
  # boundary contribution vanishes (Sdot is zero there anyway).
  upper_diff = jnp.concatenate(
    (q_m[..., 1:] - q_m[..., :-1], jnp.zeros_like(q_m[..., -1:])), axis=-1)
  # q_i - q_{i-1} for i in [1, n-1], padded with zero at i = 0.
  lower_diff = jnp.concatenate(
    (jnp.zeros_like(q_m[..., 0:1]), q_m[..., 1:] - q_m[..., :-1]), axis=-1)
  # s_dot_S has length n+1; align so [..., 1:] is at i+1/2 and [..., :-1]
  # is at i-1/2 from the midpoint i = 0..n-1 perspective.
  return 0.5 * (s_dot_S[..., 1:] * upper_diff +
                s_dot_S[..., :-1] * lower_diff) / d_mass


def _sb81_advect_interface(q_i, s_dot_S, delta_s_m, pi_deriv_avg_i):
  """SB81 interface vertical advection operator (Taylor et al. eqn 41).

  ``(d pi/d s)-bar [s_dot d/ds] q |_{i+1/2}
      = avg_to_i ( avg_to_m(Sdot) * (dq/ds)_m )_{i+1/2}``

  We then divide by ``(d pi/d s)-bar`` to return an acceleration at
  interfaces.  ``pi_deriv_avg_i`` is the eqn-31 interface average of
  ``d_mass / delta_s_m``.
  """
  s_dot_S_m = interface_to_midlevel(s_dot_S)
  dq_ds_m = _vertical_diff_i_to_m(q_i, delta_s_m)
  product_m = s_dot_S_m * dq_ds_m
  product_i = midlevel_to_interface(product_m)
  return product_i / pi_deriv_avg_i


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

  grad_exner = horizontal_gradient_3d(exner, h_grid, physics_config) / r_hat_m[:, :, :, :, jnp.newaxis]
  theta_v = theta_v_d_mass / d_mass
  grad_phi_i = horizontal_gradient_3d(phi_i, h_grid, physics_config)
  v_over_r_hat_i = midlevel_to_interface_vel(u / r_hat_m[:, :, :, :, np.newaxis],
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

  # -------------------------------------------------------------------------
  # Eulerian-height-coordinate diagnostics
  # -------------------------------------------------------------------------
  # In the Lagrangian branch all of these are unused (Sdot = 0 by construction
  # and phi is prognostic).  In the Eulerian-height branch they are required
  # by the four new vertical-transport tendency terms and by the four extra
  # cancellation pairs in :func:`eval_energy_quantities`.  Each definition
  # below traces to a specific equation in Taylor et al. (2020), section 5,
  # and is the *unique* form that closes one of the four discrete energy
  # cancellations (see the design note accompanying this module).
  if model in eulerian_models:
    delta_s_m = v_grid["delta_s_m"]
    delta_s_i = v_grid["delta_s_i"]

    # d_phi_ds at midpoints (eqn 32 left form).
    d_phi_ds_m = _vertical_diff_i_to_m(phi_i, delta_s_m)
    # d_phi_ds averaged to interfaces (eqn 31) -- the "phi-bar" derivative
    # that appears in eqn (45)'s vertical-transport term, the Sdot
    # diagnostic, and theta_v_tilde.
    d_phi_ds_i = midlevel_to_interface(d_phi_ds_m)

    # (d pi/d s)-bar at interfaces (eqn 31), defined as the weighted
    # interface average of (d pi / d s) = d_mass / delta_s_m at midpoints.
    # This is the explicit (d pi/d s)-bar that appears in Sdot = (d pi/d s)-
    # bar * s_dot and in the SB81 interface operator denominator.  Computing
    # it via midlevel_to_interface on (d_mass / delta_s_m) -- rather than
    # via midlevel_to_interface(d_mass) -- ensures we get the eqn-31 average
    # of the density itself, not of the layer-mass.
    pi_deriv_m = d_mass / delta_s_m
    pi_deriv_avg_i = midlevel_to_interface(pi_deriv_m)

    # Sdot diagnostic (eqn 16 of Taylor et al., specialised to the Eulerian
    # height coordinate where d phi / dt = 0).  The numerator is exactly the
    # combination already evaluated by eval_phi_acceleration_v_term and
    # eval_phi_advection_term; we reuse the same discrete objects so that the
    # phi-equation residual that defines Sdot and the phi-equation terms
    # that appear elsewhere in the energy budget are bit-identical.
    u_grad_phi_i = (v_over_r_hat_i[:, :, :, :, 0] * grad_phi_i[:, :, :, :, 0] +
                    v_over_r_hat_i[:, :, :, :, 1] * grad_phi_i[:, :, :, :, 1])
    phi_residual_i = g * w_i - u_grad_phi_i
    s_dot_i = phi_residual_i / d_phi_ds_i
    s_dot_S = pi_deriv_avg_i * s_dot_i

    # Enforce Sdot = 0 exactly at the top and surface interfaces.  This is
    # the boundary condition that closes the KE-KE cancellation in eqn (56)
    # of Taylor et al.  In the continuum these should already vanish via the
    # w-BCs; we zero them explicitly so that roundoff in those BCs does not
    # propagate into a per-step energy drift.
    s_dot_S = _zero_boundary_interface(s_dot_S)
    s_dot_i = _zero_boundary_interface(s_dot_i)
    # Sdot averaged to midpoints, used by the SB81 interface operator
    # (eqn 41) for the w-equation vertical advection.
    s_dot_S_m = interface_to_midlevel(s_dot_S)

    # Special interface average of theta_v from eqn (48).  This is the
    # exclusive choice of average that closes the I-K cancellation
    # (Taylor et al. eqn 52): cp Pi d(theta_v_tilde Sdot)/ds integrates by
    # parts to mu Sdot d(phi_bar)/ds.  An arithmetic mean or a
    # density-weighted mean would *not* close this cancellation -- the
    # specific form below is forced by the discrete identity.
    d_pi_ds_i = _vertical_diff_m_to_i(exner, delta_s_i)
    theta_v_tilde_i = -(mu / physics_config["cp"]) * d_phi_ds_i / d_pi_ds_i

    common_variables["delta_s_m"] = delta_s_m
    common_variables["delta_s_i"] = delta_s_i
    common_variables["d_phi_ds_m"] = d_phi_ds_m
    common_variables["d_phi_ds_i"] = d_phi_ds_i
    common_variables["d_pi_ds_i"] = d_pi_ds_i
    common_variables["pi_deriv_avg_i"] = pi_deriv_avg_i
    common_variables["s_dot_i"] = s_dot_i
    common_variables["s_dot_S"] = s_dot_S
    common_variables["s_dot_S_m"] = s_dot_S_m
    common_variables["theta_v_tilde_i"] = theta_v_tilde_i
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
  return -grad_kinetic_energy / common_variables["r_hat_m"][:, :, :, :, jnp.newaxis]


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
  w2_grad_sph = horizontal_gradient_3d(w_sq_m, h_grid, config) / common_variables["r_hat_m"][:, :, :, :, jnp.newaxis]
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
           common_variables["r_m"][:, :, :, :, np.newaxis])


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
  grad_theta_v_exner = horizontal_gradient_3d(theta_v * exner, h_grid, config) / r_hat_m[:, :, :, :, jnp.newaxis]
  grad_theta_v = horizontal_gradient_3d(theta_v, h_grid, config) / r_hat_m[:, :, :, :, jnp.newaxis]
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
  v_sq_over_r_i = midlevel_to_interface_vel(common_variables["horizontal_wind"]**2 / common_variables["r_m"][:, :, :, :, jnp.newaxis],
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


# ---------------------------------------------------------------------------
# Eulerian-height vertical-transport terms (SB81 operators)
# ---------------------------------------------------------------------------
# These four functions implement the non-Hamiltonian vertical-transport
# terms that appear in the Eulerian formulation of eqns (42)-(47) of
# Taylor et al. (2020).  Each is wrapped to return zero unless
# ``model in eulerian_models``; in the Lagrangian branch Sdot = 0 and the
# terms vanish.  See the design note for the cancellation pairs each term
# participates in.


@jit
def eval_u_vertical_advection_term(common_variables):
  """
  SB81 vertical advection of horizontal momentum (Taylor et al. eqn 40 / 42).

  Returns ``-[s_dot d/ds] u`` at midpoints using the midpoint-form SB81
  operator -- an acceleration, consistent with the units of the existing
  u-equation term functions.  Pairs with the ``1/2 u^2 dSdot/ds`` term from
  the continuity equation in the KE budget (cancellation group ``A``); the
  SB81 midpoint product rule (Taylor et al. eqn 39) is what makes the pair
  sum to zero to machine precision.
  """
  s_dot_S = common_variables["s_dot_S"]
  d_mass = common_variables["d_mass"]
  u = common_variables["horizontal_wind"]
  u1_adv = _sb81_advect_midpoint(u[..., 0], s_dot_S, d_mass)
  u2_adv = _sb81_advect_midpoint(u[..., 1], s_dot_S, d_mass)
  return -jnp.stack((u1_adv, u2_adv), axis=-1)


@jit
def eval_w_vertical_advection_term(common_variables):
  """
  SB81 vertical advection of vertical momentum (Taylor et al. eqn 41 / 44).

  Returns ``-[s_dot d/ds] w`` at interfaces -- an acceleration -- using the
  SB81 interface extension.  The interface-level normalisation is by
  ``(d pi/d s)-bar`` (the eqn-31 average of the density), *not* by the
  layer-mass interface average ``d_mass_i``; these differ by a factor of
  ``Delta s`` and getting it wrong shows up as an order-one drift in the
  KE-KE-2 cancellation.

  Pairs with ``1/2 w^2 dSdot/ds`` from the continuity equation in the KE
  budget (cancellation group ``B``); the SB81 interface product rule
  (eqn 38) plus the boundary condition ``Sdot = 0`` at top/surface close
  the pair via eqn (56) of Taylor et al.
  """
  delta_s_m = common_variables["delta_s_m"]
  s_dot_S = common_variables["s_dot_S"]
  pi_deriv_avg_i = common_variables["pi_deriv_avg_i"]
  w_i = common_variables["w_i"]
  return -_sb81_advect_interface(w_i, s_dot_S, delta_s_m, pi_deriv_avg_i)


@jit
def eval_theta_v_vertical_advection_term(common_variables):
  """
  Vertical advection of ``Theta = theta_v * d_mass`` (Taylor et al. eqn 46).

  Returns ``-Delta s * d(theta_v_tilde * Sdot)/ds`` at midpoints, i.e. the
  interface-difference of the ``theta_v_tilde * Sdot`` flux without
  dividing by ``Delta s_m`` -- consistent with the layer-mass tendency
  convention used by :func:`eval_theta_v_divergence_term`, where the
  horizontal divergence returns ``d(Theta)/dt`` per unit area, not per
  unit (area * Delta s).

  Uses the *special* interface average ``theta_v_tilde`` from eqn (48).
  Pairs with the ``-g mu`` contribution to the w-equation in the K-I
  cancellation group ``C``.  The discrete identity that closes this pair is
  the integration by parts
  ``sum cp Pi (Theta_v_tilde Sdot)_{i+1/2} - (...)_{i-1/2}
      = sum mu Sdot d(phi_bar)/ds * Delta s``,
  which holds *only* with the eqn (48) form of theta_v_tilde; any other
  average leaks at machine epsilon per step.
  """
  s_dot_S = common_variables["s_dot_S"]
  theta_v_tilde_i = common_variables["theta_v_tilde_i"]
  flux_i = theta_v_tilde_i * s_dot_S
  return -(flux_i[..., 1:] - flux_i[..., :-1])


@jit
def eval_d_mass_vertical_advection_term(common_variables):
  """
  Vertical advection of layer pseudodensity (Taylor et al. eqn 47).

  Returns ``-(Sdot_{i+1/2} - Sdot_{i-1/2})`` at midpoints -- the negative
  interface-difference of Sdot, equivalent to ``-Delta s * dSdot/ds`` per
  layer.  Same convention as :func:`eval_d_mass_divergence_term`.

  This single discrete object pairs against three different KE-budget
  terms: with ``A'`` (against u-vertical advection via the midpoint product
  rule), ``B'`` (against w-vertical advection via the interface product
  rule), and ``D_1'`` (against the potential-energy update via averaging-
  by-parts).  Triple duty is the reason the SB81 product rules are needed
  rather than just convenient.
  """
  s_dot_S = common_variables["s_dot_S"]
  return -(s_dot_S[..., 1:] - s_dot_S[..., :-1])


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
  v_theta_v /= r_hat_m[:, :, :, :, jnp.newaxis]
  div_v_theta_v = horizontal_divergence_3d(v_theta_v, h_grid, config) / 2.0
  grad_theta_v = horizontal_gradient_3d(theta_v, h_grid, config)
  grad_theta_v /= r_hat_m[:, :, :, :, jnp.newaxis]

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
          common_variables["horizontal_wind"] / common_variables["r_hat_m"][:, :, :, :, jnp.newaxis])


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

  # -------------------------------------------------------------------------
  # Eulerian-height vertical-transport contributions
  # -------------------------------------------------------------------------
  # In the Eulerian-height branch the four prognostic equations for u, w,
  # Theta, and d_mass each pick up a vertical-transport term (Taylor et al.
  # 2020 eqns 42-47).  The geopotential phi is no longer prognostic -- its
  # equation is inverted to define Sdot in :func:`init_common_variables`
  # -- so phi_tend is set to zero.  Sdot vanishes identically at top and
  # surface (enforced in init_common_variables), so the existing
  # lower-boundary correction remains the consistency condition that makes
  # the Sdot diagnostic return zero at the surface.
  if model in eulerian_models:
    u_tend += eval_u_vertical_advection_term(common_variables)
    w_tend += eval_w_vertical_advection_term(common_variables)
    theta_v_d_mass_tend += eval_theta_v_vertical_advection_term(common_variables)
    d_mass_tend += eval_d_mass_vertical_advection_term(common_variables)
    phi_tend = jnp.zeros_like(common_variables["phi_i"])

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

  # -------------------------------------------------------------------------
  # Eulerian-height vertical-transport cancellation pairs
  # -------------------------------------------------------------------------
  # In the Lagrangian branch Sdot = 0, every term below is identically zero,
  # and the pairs trivially satisfy ``a + b = 0``.  They are computed
  # unconditionally so the diagnostic dict has the same keys across model
  # variants; the cost is one Sdot-evaluation per energy diagnostic call.
  #
  # Group A (KE-KE via continuity, midpoint product rule, eqn 39):
  #     a:  sum d_mass * u . [u_vertical_advection]
  #     b:  sum (1/2) u^2 * [d_mass_vertical_advection]
  # Group B (KE-KE via continuity, interface product rule, eqn 38, plus
  #          Sdot = 0 boundary in eqn 56):
  #     a:  sum d_mass_i_integral * w_i * [w_vertical_advection]
  #     b:  sum (1/2) avg(w^2) * [d_mass_vertical_advection]
  # Group C (K-I, closed by theta_v_tilde from eqn 48):
  #     a:  sum mu * Sdot * d(phi_bar)/ds * Delta s_{i+1/2}   (interface)
  #     b:  sum cp * Pi * [theta_v_vertical_advection]
  # Group D (P-P, closed by averaging-by-parts eqn 37):
  #     a:  sum phi * [d_mass_vertical_advection]
  #     b:  sum d_mass_i_integral * (phi-bar contribution from phi-advection
  #         on interfaces, i.e. the new Sdot-mediated phi update piece).
  u_vertical_advection = eval_u_vertical_advection_term(common_variables) \
      if model in eulerian_models else jnp.zeros_like(u)
  w_vertical_advection = eval_w_vertical_advection_term(common_variables) \
      if model in eulerian_models else jnp.zeros_like(w_i)
  theta_v_vertical_advection = eval_theta_v_vertical_advection_term(common_variables) \
      if model in eulerian_models else jnp.zeros_like(d_mass)
  d_mass_vertical_advection = eval_d_mass_vertical_advection_term(common_variables) \
      if model in eulerian_models else jnp.zeros_like(d_mass)

  if model in eulerian_models:
    s_dot_S = common_variables["s_dot_S"]
    d_phi_ds_i = common_variables["d_phi_ds_i"]
  else:
    s_dot_S = jnp.zeros_like(w_i)
    d_phi_ds_i = jnp.zeros_like(w_i)

  # Group A: u-vertical-advection vs. d_mass-vertical-advection on u^2/2.
  ke_ke_v_u_a = jnp.sum(d_mass * physical_dot_product(u, u_vertical_advection), axis=-1)
  ke_ke_v_u_b = jnp.sum(0.5 * u_sq * d_mass_vertical_advection, axis=-1)

  # Group B: w-vertical-advection vs. d_mass-vertical-advection on w^2/2.
  # The w-term uses d_mass_i_integral (half-weighted at boundaries) which is
  # the consistent interface integration weight.  The d_mass-advection term
  # is a midpoint quantity, so we average w^2 to midpoints.
  ke_ke_v_w_a = jnp.sum(d_mass_i_integral * w_i * w_vertical_advection, axis=-1)
  ke_ke_v_w_b = jnp.sum(0.5 * interface_to_midlevel(w_i**2) * d_mass_vertical_advection, axis=-1)

  # Group C: K-I cancellation via theta_v_tilde.  Both sides are interface
  # quantities multiplied by Sdot, so they vanish at top and surface by
  # the boundary BC.  The pointwise identity from eqn 48 makes each
  # column entry of (a + b) algebraically zero.
  ke_ie_v_a = jnp.sum(d_mass_i_integral * mu * s_dot_S * d_phi_ds_i, axis=-1)
  ke_ie_v_b = jnp.sum(config["cp"] * exner * theta_v_vertical_advection, axis=-1)

  # Group D: P-budget cancellation.  d_mass_vertical_advection at midpoints
  # combines with the Sdot * d(phi-bar)/ds contribution at interfaces via
  # the averaging-by-parts identity (eqn 37).  The (b) side uses the
  # interface-integrated Sdot * d(phi_bar)/ds, with phi-bar derivative
  # being the interface-averaged form already stored in common_variables.
  pe_pe_v_a = jnp.sum(phi * d_mass_vertical_advection, axis=-1)
  pe_pe_v_b = jnp.sum(d_mass_i_integral * s_dot_S * d_phi_ds_i, axis=-1)

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
           "ke_ie_3": (ke_ie_3_a, ke_ie_3_b),
           "ke_ke_v_u": (ke_ke_v_u_a, ke_ke_v_u_b),
           "ke_ke_v_w": (ke_ke_v_w_a, ke_ke_v_w_b),
           "ke_ie_v":   (ke_ie_v_a,   ke_ie_v_b),
           "pe_pe_v":   (pe_pe_v_a,   pe_pe_v_b)}
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
