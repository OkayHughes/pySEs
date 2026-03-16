from ..._config import get_backend as _get_backend
from functools import partial
from ..model_state import wrap_dynamics, init_static_forcing, wrap_tracers, wrap_model_state
_be = _get_backend()
jit = _be.jit
jnp = _be.np


@partial(jit, static_argnames=["dims", "model"])
def init_model_struct(u,
                      d_mass,
                      phi_surf,
                      tracers,
                      h_grid,
                      dims,
                      physics_config,
                      model):
  """
  Initialise the 3d shallow water model state from raw prognostic arrays.

  Assembles the dynamics, static forcing, and tracer sub-structs and wraps
  them into the top-level model state dict.

  Parameters
  ----------
  u : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx, 2], Float]
      Horizontal wind components ``(u, v)``.
  d_mass : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Dry-air layer mass (Pa).
  phi_surf : Array[tuple[elem_idx, gll_idx, gll_idx], Float]
      Surface geopotential (m^2 s^-2).
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

  Returns
  -------
  state : dict
      Top-level model state dict from :func:`wrap_model_state`.
  """
  dynamics = wrap_dynamics(u, jnp.zeros_like(d_mass), d_mass, model)
  static_forcing = init_static_forcing(phi_surf, h_grid, physics_config, dims, model)
  tracers = wrap_tracers({}, tracers, model)
  return wrap_model_state(dynamics, static_forcing, tracers)
