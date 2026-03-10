import numpy as np
from ..._config import get_backend as _get_backend
from ..utils_3d import midlevel_to_interface, interface_to_delta, phi_to_r_hat
from functools import partial
from ..model_info import hydrostatic_models, deep_atmosphere_models
_be = _get_backend()
jnp = _be.np
jit = _be.jit
flip = _be.flip


@jit
def eval_r_hat_sq_avg(r_hat_i):
  """
  Compute the layer-averaged squared radial scaling factor.

  Uses the exact quadratic average of the interface values:
  ``r_hat_sq = (r_i^2 + r_i * r_{i+1} + r_{i+1}^2) / 3``.
  This corresponds to the integral average of ``r^2`` over the layer when
  ``r`` varies linearly between interfaces.

  Parameters
  ----------
  r_hat_i : Array[tuple[elem_idx, gll_idx, gll_idx, ilev_idx], Float]
      Radial scaling factor ``r_hat = r / a`` at vertical interfaces, from
      :func:`phi_to_r_hat`.

  Returns
  -------
  r_hat_sq : Array[tuple[elem_idx, gll_idx, gll_idx, ilev_idx], Float]
      Layer-averaged ``r_hat^2`` at mid-levels; used in the non-hydrostatic
      pressure calculation.
  """
  r_hat_sq = (r_hat_i[:, :, :, :-1] * r_hat_i[:, :, :, 1:] +
              r_hat_i[:, :, :, :-1] * r_hat_i[:, :, :, :-1] +
              r_hat_i[:, :, :, 1:] * r_hat_i[:, :, :, 1:]) / 3.0
  return r_hat_sq


@jit
def eval_pressure_exner_nonhydrostatic(theta_v_d_mass,
                                       d_phi,
                                       r_hat_sq_avg,
                                       config):
  """
  Compute mid-level pressure and the Exner function for non-hydrostatic HOMME.

  Derives pressure from the non-hydrostatic equation of state relating virtual
  potential temperature, geopotential layer thickness, and the deep-atmosphere
  radial scaling.

  Parameters
  ----------
  theta_v_d_mass : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Virtual potential temperature times layer mass.
  d_phi : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Layer geopotential thickness (interface differences of ``phi_i``).
  r_hat_sq_avg : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Layer-averaged squared radial scaling factor from :func:`eval_r_hat_sq_avg`;
      set to ``1.0`` for shallow-atmosphere models.
  config : dict
      Physics configuration dict with keys ``"p0"``, ``"Rgas"``, and ``"cp"``.

  Returns
  -------
  nh_pressure : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Non-hydrostatic mid-level pressure (Pa).
  exner : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Exner function ``pi = (p / p0)^(Rgas/cp)``.
  """
  p0 = config["p0"]
  nh_pressure_over_exner = -config["Rgas"] * theta_v_d_mass / d_phi
  nh_pressure_over_exner /= r_hat_sq_avg
  exponent = (1.0 / (1.0 - config["Rgas"] / config["cp"]))
  nh_pressure = p0 * (nh_pressure_over_exner / p0)**exponent
  return nh_pressure, nh_pressure / nh_pressure_over_exner


@partial(jit, static_argnames=["model"])
def eval_mu(state,
            phi_i,
            v_grid,
            config,
            model):
  """
  Compute pressure, Exner function, radial scaling, and the ``mu`` coefficient.

  ``mu`` is the derivative of the non-hydrostatic pressure with respect to
  layer mass (``d p / d (d_mass)``), needed for the implicit vertical
  velocity update.  For hydrostatic models ``mu`` is identically ``1``.  For
  deep-atmosphere models the pressure and ``mu`` are scaled by ``r_hat^2``.

  Parameters
  ----------
  state : dict[str, Array]
      Dynamics state dict containing ``"theta_v_d_mass"`` and ``"d_mass"``.
  phi_i : Array[tuple[elem_idx, gll_idx, gll_idx, ilev_idx], Float]
      Interface geopotential (assumed to be in hydrostatic balance on input).
  v_grid : dict[str, Array]
      Vertical grid struct from :func:`init_vertical_grid`.
  config : dict
      Physics configuration dict.
  model : model_info.models
      Model identifier; selects hydrostatic/non-hydrostatic and
      shallow/deep-atmosphere branches.

  Returns
  -------
  p_model : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Mid-level pressure (Pa).
  exner : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Exner function at mid-levels.
  r_hat_i : Array or float
      Interface radial scaling factor; ``1.0`` for shallow-atmosphere models.
  d_nh_pressure_d_mass : Array[tuple[elem_idx, gll_idx, gll_idx, ilev_idx], Float]
      ``mu = dp/d(d_mass)`` at interfaces; ``1.0`` array for hydrostatic models.
  """
  # note: assumes that phi_i is in hydrostatic balance.
  theta_v_d_mass = state["theta_v_d_mass"]
  d_phi = interface_to_delta(phi_i)
  if model in deep_atmosphere_models:
    r_hat_i = phi_to_r_hat(phi_i, config, model)
    r_hat_sq_avg = eval_r_hat_sq_avg(r_hat_i)
  else:
    r_hat_i = 1.0
    r_hat_sq_avg = 1.0
  p_model, exner = eval_pressure_exner_nonhydrostatic(theta_v_d_mass, d_phi, r_hat_sq_avg, config)
  if model in hydrostatic_models:
    d_nh_pressure_d_mass = jnp.ones_like(phi_i)
  else:
    p_top = v_grid["hybrid_a_i"][0] * v_grid["reference_surface_mass"]
    if model in deep_atmosphere_models:
      p_top /= r_hat_i[:, :, :, 0]**2
    d_mass_i = midlevel_to_interface(state["d_mass"])
    d_nh_pressure_d_mass_top = 2 * (p_model[:, :, :, 0] - p_top) / d_mass_i[:, :, :, 0]
    d_nh_pressure_d_mass_bottom = jnp.ones_like(p_model[:, :, :, 0])
    d_nh_pressure_d_mass_int = interface_to_delta(p_model) / d_mass_i[:, :, :, 1:-1]
    d_nh_pressure_d_mass = jnp.concatenate((d_nh_pressure_d_mass_top[:, :, :, np.newaxis],
                                            d_nh_pressure_d_mass_int,
                                            d_nh_pressure_d_mass_bottom[:, :, :, np.newaxis]),
                                           axis=-1)
    if model in deep_atmosphere_models:
      d_nh_pressure_d_mass *= r_hat_i**2
  return p_model, exner, r_hat_i, d_nh_pressure_d_mass


@jit
def eval_midlevel_pressure(state,
                           v_grid):
  """
  Compute mid-level pressure from the dynamics state using cumulative layer mass.

  Uses the hydrostatic approximation: ``p_mid[k] = sum(d_mass[0:k+1]) +
  p_top - d_mass[k] / 2``, where ``p_top = hybrid_a_i[0] *
  reference_surface_mass``.

  Parameters
  ----------
  state : dict[str, Array]
      Dynamics state dict containing ``"d_mass"`` with shape
      ``(elem, gll, gll, lev)``.
  v_grid : dict[str, Array]
      Vertical grid struct from :func:`init_vertical_grid`.

  Returns
  -------
  p_mid : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Hydrostatic mid-level pressure (Pa).
  """
  p = jnp.cumsum(state["d_mass"], axis=-1) + v_grid["hybrid_a_i"][0] * v_grid["reference_surface_mass"]
  p -= 0.5 * state["d_mass"]
  return p


@jit
def eval_balanced_geopotential(phi_surf,
                               p_mid,
                               theta_v_d_mass,
                               physics_config):
  """
  Compute the hydrostatically balanced interface geopotential.

  Integrates the hydrostatic equation upward from the surface using the
  virtual potential temperature and the Exner pressure relationship.  The
  result is the interface geopotential that would be in exact hydrostatic
  balance with the given ``theta_v_d_mass`` profile.

  Parameters
  ----------
  phi_surf : Array[tuple[elem_idx, gll_idx, gll_idx], Float]
      Surface geopotential (m^2 s^-2).
  p_mid : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Mid-level pressure (Pa) from :func:`eval_midlevel_pressure`.
  theta_v_d_mass : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Virtual potential temperature times layer mass.
  physics_config : dict
      Physics configuration dict with ``"Rgas"``, ``"cp"``, and ``"p0"``.

  Returns
  -------
  phi_i : Array[tuple[elem_idx, gll_idx, gll_idx, ilev_idx], Float]
      Interface geopotential (m^2 s^-2) in hydrostatic balance.
  """
  # p = get_p_mid(state, v_grid, config)
  exponent = (physics_config["Rgas"] / physics_config["cp"] - 1.0)
  d_phi = physics_config["Rgas"] * (theta_v_d_mass *
                                    (p_mid / physics_config["p0"])**exponent / physics_config["p0"])
  d_phi_augment = flip(jnp.concatenate((d_phi[:, :, :, :-1],
                                        (d_phi[:, :, :, -1] + phi_surf)[:, :, :, np.newaxis]),
                                       axis=-1), -1)
  phi_i_above_surf = jnp.cumsum(d_phi_augment, axis=-1)
  return jnp.concatenate((flip(phi_i_above_surf, -1), phi_surf[:, :, :, np.newaxis]), axis=-1)
