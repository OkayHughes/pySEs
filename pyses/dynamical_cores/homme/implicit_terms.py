from ..._config import get_backend as _get_backend
from ..utils_3d import midlevel_to_interface_vel, physical_dot_product, phi_to_g, phi_to_z, interface_to_delta, cumulative_sum
from ..model_state import wrap_dynamics
from .thermodynamics import eval_mu
_be = _get_backend()
jnp = _be.np
flip = _be.flip


def compute_perceived_phi_tend(v, d_mass, grad_phi_surf, phi_i, physics_config, v_grid, model):
  v_d_mass = v * d_mass
  grad_z_surf = grad_phi_surf / phi_to_g(phi_i[:, :, :, -1], physics_config, model)
  v_i_exl_boundaries = ((v_d_mass[:, :, :, :-1] + v_d_mass[:, :, :, 1:]) / 
                        (d_mass[:, :, :, :-1] + d_mass[:, :, :, 1:]))
  w_i_exl_boundaries = (v_i_exl_boundaries[:, :, :, :, 0] * grad_z_surf[:, :, :, 0, jnp.newaxis] +
                         v_i_exl_boundaries[:, :, :, :, 1] * grad_z_surf[:, :, :, 1, jnp.newaxis]) * v_grid["hybrid_b_i"][jnp.newaxis, jnp.newaxis, jnp.newaxis, 1:-1]
  return jnp.stack((jnp.zeros_like(d_mass[:, :, :, 0:1]),
                    w_i_exl_boundaries,
                    jnp.zeros_like(d_mass[:, :, :, 0:1])), axis=-1) * phi_to_g(phi_i, physics_config, model)


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


def calc_implicit_update(alpha_dt_n_minus_1, alpha_dt_n, dt_implicit, dynamics_prev, dynamics_curr, dynamics_next, static_forcing, v_grid, physics_config, model):
  tol = 1.0e-11
  phi_before_implicit = 1.0 * dynamics_next["phi_i"]
  w_before_implicit = 1.0 * dynamics_next["w_i"]
  phi_guess = 1.0 * dynamics_next["phi_i"]
  w_guess = 1.0 * dynamics_next["w_i"]

  for dynamics, alpha_dt in zip([dynamics_curr, dynamics_prev],
                                [alpha_dt_n, alpha_dt_n_minus_1]):
    g = phi_to_g(dynamics["phi_i"], physics_config, model)
    _, _, _, mu = eval_mu(dynamics, dynamics["phi_i"], v_grid, physics_config, model)
    w_before_implicit += alpha_dt_n * g * (mu - 1.0)
    perceived_velocity = compute_perceived_phi_tend(dynamics["horizontal_wind"],
                                                    dynamics["d_mass"],
                                                    static_forcing["grad_phi_surf"],
                                                    v_grid)
    phi_before_implicit += alpha_dt * (g * dynamics["w_i"] - perceived_velocity)
  perceived_velocity = compute_perceived_phi_tend(dynamics_next["horizontal_wind"],
                                                  dynamics_next["d_mass"],
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
  nh_pressure, _, _, mu = eval_mu(dynamics_next, phi_guess, v_grid, physics_config, model)
  w_phi_compatibility_residual = w_guess - (w_before_implicit + dt_implicit * phi_to_g(phi_guess, physics_config, model) * (mu - 1.0))
  max_itercount = 20
  for iter_idx in range(max_itercount):
    jacobian_lower, jacobian_diag, jacobian_upper = calc_dirk_jacobian(dt_implicit,
                                                                       dynamics_next["d_mass"],
                                                                       dphi_guess,
                                                                       nh_pressure,
                                                                       physics_config)
    newton_rhs = -w_phi_compatibility_residual
    w_guess_search_dir = solve_strict_diag_dominant_tridiag(jacobian_lower, jacobian_diag, jacobian_upper, newton_rhs)
    g_guess = phi_to_g(phi_guess, physics_config, model)
    # assume alpha_step = 1.0 in the following line
    phi_tend_interim = jnp.stack((g_guess[:, :, :, 1:] * ((w_guess[:, :, :, 1:] + w_guess_search_dir[:, :, :, 1:])),
                                  jnp.zeros_like(w_guess[:, :, :, -1:])),
                                  axis=-1)
    
    d_phi_tend = interface_to_delta(phi_tend_interim)
    
    dphi_guess -= dt_implicit * d_phi_tend

    column_mask = jnp.sum(dphi_guess >= 0.0, axis=-1) > 0
    d_search_dir = jnp.stack((w_guess_search_dir[:, :, :, 1:-1] - w_guess_search_dir[:, :, :, 0:-2],
                              -w_guess_search_dir[:, :, :, -1]), axis=-1)
    d_w_guess = jnp.stack((w_guess[:, :, :, 1:-1] - w_guess[:, :, :, 0:-2],
                           -w_guess[:, :, :, -1]), axis=-1)
    g_avg = (g_guess[:, :, :, :-1] + g_guess[:, :, :, 1:]) / 2.0
    max_safe_step_length = jnp.where(d_search_dir != 0.0, -(dphi_no_implicit + dt_implicit * g_avg * d_w_guess) / (dt_implicit * g_avg * d_search_dir), 1.0)
    max_safe_step_length = jnp.minimum(jnp.maximum(max_safe_step_length, 0.0), 1.0)
    safe_step_length = jnp.min(max_safe_step_length, axis=-1) / 2.0
    phi_tend_interim = jnp.stack((g_guess[:, :, :, 1:] * ((w_guess[:, :, :, 1:] + safe_step_length[:, :, :, jnp.newaxis] * w_guess_search_dir[:, :, :, 1:])),
                                  jnp.zeros_like(w_guess[:, :, :, -1:])),
                                  axis=-1)
    d_phi_tend = interface_to_delta(phi_tend_interim)
    dphi_guess = jnp.where(column_mask[:, :, :, jnp.newaxis],
                           dphi_no_implicit + dt_implicit,
                           dphi_guess)
    step_length = jnp.where(column_mask, safe_step_length, 1.0)
    w_guess += step_length[:, :, :, jnp.newaxis] * w_guess_search_dir
    phi_guess = cumulative_sum(dphi_guess, static_forcing["phi_surf"])
    _, _, _, mu = eval_mu(dynamics_next, phi_guess, v_grid, physics_config, model)
    w_phi_compatibility_residual
  w_next = w_guess
  phi_next = phi_before_implicit + dt_implicit * phi_to_g(phi_guess) * w_next
  phi_next[:, :, :, -1] = static_forcing["phi_surf"]
  return wrap_dynamics(dynamics_next["horizontal_wind"],
                       dynamics_next["v_theta_d_mass"],
                       dynamics_next["d_mass"],
                       model,
                       phi_i=phi_next,
                       w_i=w_next)





def solve_strict_diag_dominant_tridiag(jacL, jacD, jacU, rhs):
  upper_diag_levs = []
  rhs_levs = []
  for lev_idx in range(jacD.shape[-1]-1):
    lower_rat = jacL[:, :, :, lev_idx] / jacD[:, :, :, lev_idx]
    upper_diag_levs.append(jacD[:, :, :, lev_idx+1] - lower_rat * jacD[:, :, :, lev_idx])
    rhs_levs.append(rhs[:, :, :, lev_idx+1] - lower_rat * rhs[:, :, :, lev_idx])
  rhs_levs.append(rhs[:, :, :, -1] / jacD[:, :, :, -1])
  rhs_out = [rhs_levs[-1]]
  for lev_idx in reversed(range(jacD.shape[-1]-1)):
    rhs_out.append(rhs_levs[lev_idx] - jacU[:, :, :, lev_idx] * rhs_levs[:, :, :, ])
  return flip(jnp.stack(rhs_out, axis=-1), -1)