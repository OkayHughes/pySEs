from ..._config import get_backend as _get_backend
from functools import partial
from ..model_state import wrap_dynamics, init_static_forcing, wrap_tracers, wrap_model_state
from ..model_info import variable_kappa_models
_be = _get_backend()
jit = _be.jit
jnp = _be.np


@partial(jit, static_argnames=["dims", "model"])
def init_model_struct(u,
                      T,
                      d_mass,
                      phi_surf,
                      moisture_species,
                      tracers,
                      h_grid,
                      dims,
                      physics_config,
                      model,
                      dry_air_species=None):
  """
  Initialise the full CAM-SE model state from raw prognostic arrays.

  Assembles the dynamics, static forcing, and tracer sub-structs and wraps
  them into the top-level model state dict.  For models that do not use
  variable-kappa dry-air species the ``dry_air_species`` argument is ignored
  and replaced with a uniform ``{"dry_air": ones}`` array.

  Parameters
  ----------
  u : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx, 2], Float]
      Horizontal wind components ``(u, v)``.
  T : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Temperature (K); the CAM-SE thermodynamic prognostic variable.
  d_mass : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Dry-air layer mass (Pa).
  phi_surf : Array[tuple[elem_idx, gll_idx, gll_idx], Float]
      Surface geopotential (m^2 s^-2).
  moisture_species : dict[str, Array]
      Moisture mixing-ratio fields (kg moisture / kg dry air) keyed by
      species name.
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
  dry_air_species : dict[str, Array] or None, optional
      Dry-air species mass-fraction fields keyed by species name; only used
      for variable-kappa models.  Defaults to ``None``.

  Returns
  -------
  state : dict
      Top-level model state dict from :func:`wrap_model_state`.
  """
  if model not in variable_kappa_models:
    dry_air_species = {"dry_air": jnp.ones_like(T)}
  dynamics = wrap_dynamics(u, T, d_mass, model)
  static_forcing = init_static_forcing(phi_surf, h_grid, physics_config, dims, model)
  tracers = wrap_tracers(moisture_species, tracers, model, dry_air_species=dry_air_species)
  return wrap_model_state(dynamics, static_forcing, tracers)
