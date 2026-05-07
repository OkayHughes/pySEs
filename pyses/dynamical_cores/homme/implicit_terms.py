from ..._config import get_backend as _get_backend
from ..utils_3d import midlevel_to_interface_vel, physical_dot_product, phi_to_g, phi_to_z, interface_to_delta, cumulative_sum, calc_perceived_phi_tend
from ..model_state import wrap_dynamics
from .thermodynamics import eval_mu
from functools import partial
_be = _get_backend()
jit = _be.jit
jnp = _be.np
flip = _be.flip


@jit
def calc_dirk_jacobian(dt, d_mass, d_phi, pnh, physics_config):
  kappa = physics_config["R_gas"]/physics_config["cp"]
  a = (dt * physics_config)**2 / (1 - kappa)
  b = jnp.stack((a / d_mass[:, :, :, 0], 
                 2 * a / (d_mass[:, :, :, :-1] + d_mass[:, :, :, 1:])))
  c = pnh / d_phi
  jacL = b[:, :, :, 1:] * c[:, :, :, :-1]
  jacU = jnp.stack((2 * b[:, :, :, 0] * c[:, :, :, 0],
                    b[:, :, :, 1:] * c[:, :, :, 1:]), axis=-1)
  jacD = jnp.stack((1.0 - jacU[:, :, :, 0],
                    1.0 - jacL[:, :, :, :-1] - jacU[:, :, :, 1:],
                    1.0 - jacL[:, :, :, -1] - b[:, :, :, -1] * c[:, :, :, -1]), axis=-1)
  return jacL, jacD, jacU


@partial(jit, static_argnames=["model"])
def advance_buoyancy_explicit(alpha_dts, dynamics_series, static_forcing, physics_config, v_grid, model):
  for dynamics, alpha_dt in zip(dynamics_series,
                                alpha_dts):
    g = phi_to_g(dynamics["phi_i"], physics_config, model)
    _, _, _, mu = eval_mu(dynamics, dynamics["phi_i"], v_grid, physics_config, model)
    w_before_implicit += alpha_dt * g * (mu - 1.0)
    perceived_velocity = calc_perceived_phi_tend(dynamics["horizontal_wind"],
                                                 dynamics["d_mass"],
                                                 static_forcing["grad_phi_surf"],
                                                 v_grid)
    phi_before_implicit += alpha_dt * (g * dynamics["w_i"] - perceived_velocity)
  return {"w_i": w_before_implicit, 
          "phi_i": phi_before_implicit}


@jit
def init_search_dir(dt_implicit,
                    w_guess,
                    dphi_guess,
                    d_mass_before_implicit,
                    nh_pressure,
                    physics_config,
                    w_phi_compatibility_residual):
  jacobian_lower, jacobian_diag, jacobian_upper = calc_dirk_jacobian(dt_implicit,
                                                                      d_mass_before_implicit,
                                                                      dphi_guess,
                                                                      nh_pressure,
                                                                      physics_config)
  newton_rhs = -w_phi_compatibility_residual[:, :, :, :-1]
  w_search_dir = jnp.stack((solve_strict_diag_dominant_tridiag(jacobian_lower, jacobian_diag, jacobian_upper, newton_rhs),
                                  jnp.zeros_like(w_guess[:, :, :, -1:])), axis=-1)
  return w_search_dir


@partial(jit, static_argnames=["model"])
def take_limited_step(dt_implicit,
                       phi_guess,
                       w_search_dir,
                       dphi_no_implicit,
                       physics_config,
                       model):
  g_guess = phi_to_g(phi_guess, physics_config, model)
  # assume alpha_step = 1.0 in the following line
  phi_tend_interim = jnp.stack((g_guess[:, :, :, :-1] * ((w_guess[:, :, :, :-1] + w_search_dir[:, :, :, :-1])),
                                jnp.zeros_like(w_guess[:, :, :, -1:])),
                                axis=-1)

  d_phi_tend = interface_to_delta(phi_tend_interim)

  dphi_guess -= dt_implicit * d_phi_tend
  column_mask = jnp.sum(dphi_guess >= 0.0, axis=-1) > 0
  d_search_dir = jnp.stack((w_search_dir[:, :, :, 1:-1] - w_search_dir[:, :, :, 0:-2],
                            -w_search_dir[:, :, :, -1]), axis=-1)
  d_w_guess = jnp.stack((w_guess[:, :, :, 1:-1] - w_guess[:, :, :, 0:-2],
                          -w_guess[:, :, :, -1]), axis=-1)
  g_avg = (g_guess[:, :, :, :-1] + g_guess[:, :, :, 1:]) / 2.0
  max_safe_step_length = jnp.where(d_search_dir != 0.0, -(dphi_no_implicit + dt_implicit * g_avg * d_w_guess) / (dt_implicit * g_avg * d_search_dir), 1.0)
  max_safe_step_length = jnp.minimum(jnp.maximum(max_safe_step_length, 0.0), 1.0)
  safe_step_length = jnp.min(max_safe_step_length, axis=-1) / 2.0
  phi_tend_interim = jnp.stack((g_guess[:, :, :, 1:] * ((w_guess[:, :, :, 1:] + safe_step_length[:, :, :, jnp.newaxis] * w_search_dir[:, :, :, 1:])),
                                jnp.zeros_like(w_guess[:, :, :, -1:])),
                                axis=-1)
  d_phi_tend = interface_to_delta(phi_tend_interim)
  dphi_guess = jnp.where(column_mask[:, :, :, jnp.newaxis],
                          dphi_no_implicit + dt_implicit * d_phi_tend,
                          dphi_guess)
  step_length = jnp.where(column_mask, safe_step_length, 1.0)
  w_guess += step_length[:, :, :, jnp.newaxis] * w_search_dir
  return dphi_guess, w_guess


@partial(jit, static_argnames=["model"])
def calc_implicit_update(dt_implicit, nh_vars_before_implicit, dynamics_before_implicit, static_forcing, v_grid, physics_config, model):
  tol = 1.0e-11
  phi_before_implicit = 1.0 * nh_vars_before_implicit["phi_i"]
  w_before_implicit = 1.0 * nh_vars_before_implicit["w_i"]
  phi_guess = 1.0 * dynamics_before_implicit["phi_i"]
  w_guess = 1.0 * dynamics_before_implicit["w_i"]
  perceived_velocity = calc_perceived_phi_tend(dynamics_before_implicit["horizontal_wind"],
                                               dynamics_before_implicit["d_mass"],
                                               static_forcing["grad_phi_surf"],
                                               v_grid)
  phi_before_implicit -= dt_implicit * perceived_velocity
  # TODO: add other initial guess strategies
  w_next = (phi_to_z(phi_guess, physics_config, model) - phi_to_z(phi_before_implicit, physics_config, model))
  dphi_guess = interface_to_delta(phi_guess)
  dphi_no_implicit = interface_to_delta(phi_before_implicit)
  limit_value = -10.0 # m^2 s^{-2}
  mask_3d = dphi_guess < limit_value
  dphi_guess = jnp.where(mask_3d, limit_value, dphi_guess)
  mask = (jnp.sum(mask_3d, axis=-1) > 0)[:, :, :, jnp.newaxis]
  phi_guess = jnp.stack((jnp.where(mask, phi_guess[:, :, :, :-1] - dphi_guess, phi_guess[:, :, :, :-1]),
                         phi_guess[:, :, :, -1:]), axis=-1)
  w_guess = jnp.stack((jnp.where(mask, (phi_to_z(phi_guess[:, :, :, :-1], physics_config, model) - phi_to_z(phi_guess[:, :, :, :-1]) / dt_implicit, physics_config, model), w_guess[:, :, :, :-1]),
                       w_guess[:, :, :, -1:]), axis=-1)
  nh_pressure, _, _, mu = eval_mu(dynamics_before_implicit, phi_guess, v_grid, physics_config, model)
  w_phi_compatibility_residual = jnp.stack(((w_guess - (w_before_implicit + dt_implicit * phi_to_g(phi_guess, physics_config, model) * (mu - 1.0)))[:, :, :, :-1],
                                            jnp.zeros_like(w_guess[:, :, :, :-1])), axis=-1)
  max_itercount = 20
  for iter_idx in range(max_itercount):
    w_search_dir = init_search_dir(dt_implicit, w_guess, dphi_guess, dynamics_before_implicit["d_mass"], nh_pressure, physics_config, w_phi_compatibility_residual)
    dphi_guess, w_guess = take_limited_step(dt_implicit, phi_guess, w_search_dir, dphi_no_implicit, physics_config, model)
    phi_guess = cumulative_sum(dphi_guess, static_forcing["phi_surf"])
    nh_pressure, _, _, mu = eval_mu(dynamics_before_implicit, phi_guess, v_grid, physics_config, model)
    w_phi_compatibility_residual = w_guess - (w_before_implicit + dt_implicit * phi_to_g(phi_guess) * (mu - 1.0))
  w_next = w_guess
  phi_next = phi_before_implicit + dt_implicit * phi_to_g(phi_guess) * w_next
  phi_next[:, :, :, -1] = static_forcing["phi_surf"]
  return wrap_dynamics(dynamics_before_implicit["horizontal_wind"],
                       dynamics_before_implicit["v_theta_d_mass"],
                       dynamics_before_implicit["d_mass"],
                       model,
                       phi_i=phi_next,
                       w_i=w_next)


@jit
def solve_strict_diag_dominant_tridiag(jacL, jacD, jacU, rhs):
  updated_diag = [jacD[:, :, :, 0]]
  rhs_levs = [rhs[:, :, :, 0]]
  for lev_idx in range(jacD.shape[-1]-1):
    lower_rat = jacL[:, :, :, lev_idx] / updated_diag[-1]
    updated_diag.append(jacD[:, :, :, lev_idx+1] - lower_rat * jacU[:, :, :, lev_idx])
    rhs_levs.append(rhs[:, :, :, lev_idx+1] - lower_rat * rhs_levs[-1])
  # lower entries are zero at this point
  rhs_levs[-1] = rhs_levs[-1] / updated_diag[-1]
  rhs_out = [rhs_levs[-1]]
  ct = 0
  for lev_idx in reversed(range(jacD.shape[-1]-1)):
    rhs_out.append((rhs_levs[lev_idx] - jacU[:, :, :, lev_idx] * rhs_out[ct])/updated_diag[lev_idx])
    ct += 1
  return flip(jnp.stack(rhs_out, axis=-1), -1)
