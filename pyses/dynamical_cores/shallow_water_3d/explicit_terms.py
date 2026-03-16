from ..operators_3d import horizontal_gradient_3d, horizontal_vorticity_3d, horizontal_divergence_3d
from ..utils_3d import physical_dot_product
import numpy as np
from ..._config import get_backend as _get_backend
from .thermodynamics import eval_geopotential
from ..model_state import wrap_dynamics, wrap_tracer_consist_dynamics
from enum import Enum
from functools import partial
_be = _get_backend()
jnp = _be.np
jit = _be.jit



@jit
def init_common_variables(dynamics,
                          static_forcing):
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

  Returns
  -------
  common_variables : dict[str, Array]
      Dict containing the following keys:

      - ``"horizontal_wind"`` — wind vector ``(u, v)``
      - ``"phi"`` — hydrostatically balanced geopotential (m^2 s^-2)
      - ``"d_mass"`` — dry-air layer mass (Pa)
      - ``"phi_surf"`` — surface geopotential (m^2 s^-2)
      - ``"coriolis_param"`` — Coriolis parameter (s^-1)
  """
  wind = dynamics["horizontal_wind"]
  d_mass = dynamics["d_mass"]

  phi_surf = static_forcing["phi_surf"]
  coriolis_param = static_forcing["coriolis_param"]
  phi = eval_geopotential(d_mass,
                          phi_surf)
  return {"horizontal_wind": wind,
          "phi": phi,
          "d_mass": d_mass,
          "phi_surf": phi_surf,
          "coriolis_param": coriolis_param}


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


@partial(jit, static_argnames=["model"])
def eval_explicit_tendency(dynamics,
                           static_forcing,
                           h_grid,
                           v_grid,
                           physics_config,
                           model):
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
  h_grid : SpectralElementGrid
      Horizontal grid struct.
  v_grid : dict[str, Array]
      Vertical grid struct from :func:`init_vertical_grid`.
  physics_config : dict
      Physics configuration dict.
  model : model_info.models
      Model identifier; static JIT argument.

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
                                           static_forcing)

  velocity_tend = (eval_vorticity_term(common_variables, h_grid, physics_config) +
                   eval_energy_gradient_term(common_variables, h_grid, physics_config))
  temperature_tend = jnp.zeros_like(common_variables["d_mass"])
  d_mass_tend = eval_d_mass_divergence_term(common_variables, h_grid, physics_config)

  dynamics = wrap_dynamics(velocity_tend,
                           temperature_tend,
                           d_mass_tend,
                           model)
  tracer_consistency = wrap_tracer_consist_dynamics(eval_tracer_consistency_term(common_variables))
  return dynamics, tracer_consistency
