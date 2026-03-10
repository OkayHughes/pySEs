from ..._config import get_backend as _get_backend
from ..model_state import wrap_model_state, wrap_tracers, init_static_forcing, wrap_dynamics
from functools import partial
_be = _get_backend()
jnp = _be.np
jit = _be.jit


@partial(jit, static_argnames=["dims", "model"])
def init_model_struct(u,
                      theta_v_d_mass,
                      d_mass,
                      phi_surf,
                      moisture_species,
                      tracers,
                      h_grid,
                      dims,
                      physics_config,
                      model,
                      phi_i=None,
                      w_i=None,
                      f_plane_center=jnp.pi / 4.0):
  """
  Initialise the full HOMME model state from raw prognostic arrays.

  Assembles the dynamics, static forcing, and tracer sub-structs and wraps
  them into the top-level model state dict.

  Parameters
  ----------
  u : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx, 2], Float]
      Horizontal wind components ``(u, v)``.
  theta_v_d_mass : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Virtual potential temperature times layer mass (HOMME thermodynamic
      variable).
  d_mass : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Dry-air layer mass (Pa).
  phi_surf : Array[tuple[elem_idx, gll_idx, gll_idx], Float]
      Surface geopotential (m^2 s^-2).
  moisture_species : dict[str, Array]
      Moisture mixing-ratio fields keyed by species name.
  tracers : dict[str, Array]
      Passive tracer fields keyed by tracer name.
  h_grid : SpectralElementGrid
      Horizontal grid struct.
  dims : frozendict[str, int]
      Grid dimension tuple; static JIT argument.
  physics_config : dict
      Physics configuration dict.
  model : model_info.models
      Model identifier; static JIT argument.
  phi_i : Array[tuple[elem_idx, gll_idx, gll_idx, ilev_idx], Float], optional
      Interface geopotential; required for non-hydrostatic models.
  w_i : Array[tuple[elem_idx, gll_idx, gll_idx, ilev_idx], Float], optional
      Interface vertical velocity; required for non-hydrostatic models.
  f_plane_center : float, optional
      Latitude (radians) for the f-plane Coriolis constant (default: pi/4).

  Returns
  -------
  state : dict
      Top-level model state dict from :func:`wrap_model_state`.
  """
  dynamics = wrap_dynamics(u,
                           theta_v_d_mass,
                           d_mass,
                           model,
                           phi_i=phi_i,
                           w_i=w_i)
  static_forcing = init_static_forcing(phi_surf,
                                       h_grid,
                                       physics_config,
                                       dims,
                                       model,
                                       f_plane_center=f_plane_center)
  tracers = wrap_tracers(moisture_species,
                         tracers,
                         model)
  return wrap_model_state(dynamics,
                          static_forcing,
                          tracers)


# TODO 12/23/25: add wrapper functions that apply
# summation, and project_scalar so model interface remains identical.
# also: refactor into separate file, since these are non-jittable
