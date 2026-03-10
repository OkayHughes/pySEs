from .._config import get_backend as _get_backend
from ..operations_2d.operators import horizontal_divergence
from ..tracer_transport.eulerian_spectral import advance_tracers_rk2
from functools import partial
_be = _get_backend()
jit = _be.jit
jnp = _be.np


@jit
def stack_tracers_shallow_water(tracer_like):
  """
  Stack a dict of 2-D tracer arrays into a single 5-D array.

  Parameters
  ----------
  tracer_like : dict[str, Array[tuple[elem_idx, gll_idx, gll_idx], Float]]
      Named tracer fields on the shallow-water grid.

  Returns
  -------
  stacked : Array[tuple[n_tracers, elem_idx, gll_idx, gll_idx, 1], Float]
      All tracers concatenated along a leading tracer axis.
  tracer_names : dict[str, int]
      Mapping from tracer name to its index in ``stacked``.
  """
  tracer_names = {}
  tracer_mass_flat = []
  ct = 0
  for tracer_name in tracer_like.keys():
    tracer_names[tracer_name] = ct
    tracer_mass_flat.append(tracer_like[tracer_name])
    ct += 1
  return jnp.stack(tracer_mass_flat, axis=0)[:, :, :, :, jnp.newaxis], tracer_names


@jit
def unstack_tracers_shallow_water(tracer_like, tracer_names):
  """
  Unstack a 5-D tracer array back into a named dict of 2-D arrays.

  Parameters
  ----------
  tracer_like : Array[tuple[n_tracers, elem_idx, gll_idx, gll_idx, 1], Float]
      Stacked tracer array as returned by ``stack_tracers_shallow_water``.
  tracer_names : dict[str, int]
      Mapping from tracer name to index in ``tracer_like``.

  Returns
  -------
  tracers : dict[str, Array[tuple[elem_idx, gll_idx, gll_idx], Float]]
      Named tracer mixing-ratio fields.
  """
  tracers = {}
  for tracer_name, tracer_idx in tracer_names.items():
    tracers[tracer_name] = tracer_like[tracer_idx, :, :, :, 0]
  return tracers


@partial(jit, static_argnames=["dims", "timestep_config"])
def advance_tracers_shallow_water(tracers,
                                  tracer_consist_dyn,
                                  tracer_init_struct,
                                  grid,
                                  dims,
                                  physics_config,
                                  diffusion_config,
                                  timestep_config,
                                  tracer_consist_hypervis=None):
  """
  Advance all passive tracers by one physics timestep on the shallow-water grid.

  Parameters
  ----------
  tracers : dict[str, Array[tuple[elem_idx, gll_idx, gll_idx], Float]]
      Named tracer mixing-ratio fields at the beginning of the timestep.
  tracer_consist_dyn : dict[str, Array]
      Tracer-consistency struct from the dynamics step, containing
      ``"u_d_mass_avg"``.
  tracer_init_struct : dict[str, Array]
      Struct with keys ``"d_mass_init"`` and ``"d_mass_end"``, holding the
      layer thickness at the start and end of the dynamics step.
  grid : SpectralElementGrid
      Horizontal spectral element grid.
  dims : frozendict[str, int]
      Grid dimension parameters.
  physics_config : dict[str, Any]
      Physical constants (e.g. ``radius_earth``).
  diffusion_config : dict[str, Any]
      Hyperviscosity configuration.
  timestep_config : frozendict
      Time-step configuration from ``init_timestep_config``.
  tracer_consist_hypervis : dict[str, Array] or None, optional
      Tracer-consistency struct from the hyperviscosity step, or ``None``
      if hyperviscosity was not applied.

  Returns
  -------
  tracers_out : dict[str, Array[tuple[elem_idx, gll_idx, gll_idx], Float]]
      Updated tracer mixing-ratio fields after one physics timestep.
  """
  d_mass_init = tracer_init_struct["d_mass_init"]
  d_mass_end = tracer_init_struct["d_mass_end"]
  u_d_mass_avg = tracer_consist_dyn["u_d_mass_avg"]
  d_mass_dyn_tend = -horizontal_divergence(u_d_mass_avg, grid, a=physics_config["radius_earth"])
  if tracer_consist_hypervis is not None:
    d_mass_hypervis_tend = tracer_consist_hypervis["d_mass_hypervis_tend"][:, :, :, jnp.newaxis]
    d_mass_hypervis_avg = tracer_consist_hypervis["d_mass_hypervis_avg"][:, :, :, jnp.newaxis]
  else:
    d_mass_hypervis_tend = None
    d_mass_hypervis_avg = None
  stacked_tracers, tracer_names = stack_tracers_shallow_water(tracers)
  stacked_tracer_mass = stacked_tracers * d_mass_init[jnp.newaxis, :, :, :, jnp.newaxis]
  stacked_tracer_mass_out = advance_tracers_rk2(stacked_tracer_mass,
                                                d_mass_init[:, :, :, jnp.newaxis],
                                                u_d_mass_avg[:, :, :, jnp.newaxis, :],
                                                d_mass_dyn_tend[:, :, :, jnp.newaxis],
                                                grid,
                                                physics_config,
                                                diffusion_config,
                                                timestep_config,
                                                dims,
                                                d_mass_hypervis_tend=d_mass_hypervis_tend,
                                                d_mass_hypervis_avg=d_mass_hypervis_avg)
  stacked_tracer_out = stacked_tracer_mass_out / d_mass_end[jnp.newaxis, :, :, :, jnp.newaxis]
  return unstack_tracers_shallow_water(stacked_tracer_out, tracer_names)
