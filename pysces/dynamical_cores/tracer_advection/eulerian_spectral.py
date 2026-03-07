from ...config import jit, jnp
from functools import partial
from ..operators_3d import horizontal_divergence_3d
from ...tracer_transport.eulerian_spectral import advance_tracers_rk2
from ...model_info import cam_se_models, homme_models


@partial(jit, static_argnames=["model"])
def flatten_tracers(tracers, model):
  """
  Stack all tracer species from the nested tracer dict into a single array.

  Iterates over ``"moisture_species"``, ``"tracers"``, and (for CAM-SE)
  ``"dry_air_species"`` in order, building a flat list of arrays and a
  mapping from species names to their index positions.

  Parameters
  ----------
  tracers : dict
      Tracer state dict from :func:`wrap_tracers`.
  model : str
      Model identifier; determines whether ``"dry_air_species"`` is included.

  Returns
  -------
  tracers_flat : Array[tuple[n_species, elem_idx, gll_idx, gll_idx, lev_idx], Float]
      All species arrays stacked along axis 0.
  tracer_map : dict[str, dict[str, int]]
      Nested dict mapping each species name to its index in ``tracers_flat``.
  """
  tracers_flat = []
  tracer_map = {"moisture_species": {},
                "tracers": {}}
  if model in cam_se_models:
    tracer_map["dry_air_species"] = {}
  ct = 0
  for species_name in tracers["moisture_species"].keys():
    tracers_flat.append(tracers["moisture_species"][species_name])
    tracer_map["moisture_species"][species_name] = ct
    ct += 1
  for species_name in tracers["tracers"].keys():
    tracers_flat.append(tracers["tracers"][species_name])
    tracer_map["tracers"][species_name] = ct
    ct += 1
  if model in cam_se_models:
    for species_name in tracers["dry_air_species"].keys():
      tracers_flat.append(tracers["dry_air_species"][species_name])
      tracer_map["dry_air_species"][species_name] = ct
      ct += 1
  return jnp.stack(tracers_flat, axis=0), tracer_map


@partial(jit, static_argnames=["model"])
def ravel_tracers(tracers_flat, tracer_map, model):
  """
  Unstack a flat species array back into the nested tracer dict structure.

  Reverses :func:`flatten_tracers`: uses ``tracer_map`` to extract each
  species slice from ``tracers_flat`` and place it back in the appropriate
  sub-dict.

  Parameters
  ----------
  tracers_flat : Array[tuple[n_species, elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Stacked species array from :func:`flatten_tracers` (or after advection).
  tracer_map : dict[str, dict[str, int]]
      Nested index map returned by :func:`flatten_tracers`.
  model : str
      Model identifier; determines whether ``"dry_air_species"`` is populated.

  Returns
  -------
  tracers : dict
      Tracer state dict with sub-dicts ``"moisture_species"``, ``"tracers"``,
      and optionally ``"dry_air_species"``.
  """
  tracers = {"moisture_species": {},
             "tracers": {}}
  if model in cam_se_models:
    tracers["dry_air_species"] = {}
  for species_name in tracer_map["moisture_species"].keys():
    tracers["moisture_species"][species_name] = tracers_flat[tracer_map["moisture_species"][species_name]]
  for species_name in tracer_map["tracers"].keys():
    tracers["tracers"][species_name] = tracers_flat[tracer_map["tracers"][species_name]]
  if model in cam_se_models:
    for species_name in tracer_map["dry_air_species"].keys():
      tracers["dry_air_species"][species_name] = tracers_flat[tracer_map["dry_air_species"][species_name]]
  return tracers


@partial(jit, static_argnames=["dims", "timestep_config", "model"])
def advance_tracers(tracers,
                    tracer_consist_dyn,
                    tracer_init_struct,
                    grid,
                    dims,
                    physics_config,
                    diffusion_config,
                    timestep_config,
                    model,
                    tracer_consist_hypervis=None):
  """
  Advance all tracer species by one tracer time step using RK2 advection.

  Stacks tracers into a single array via :func:`flatten_tracers`, converts to
  mass form (``q * d_mass_init``), calls :func:`advance_tracers_rk2`, then
  divides by ``d_mass_end`` and unstacks via :func:`ravel_tracers`.
  Optionally applies a hyperviscosity consistency correction.

  Parameters
  ----------
  tracers : dict
      Tracer state dict from :func:`wrap_tracers`.
  tracer_consist_dyn : dict[str, Array]
      Dynamics tracer-consistency struct from
      :func:`wrap_tracer_consist_dynamics`; must contain ``"u_d_mass_avg"``.
  tracer_init_struct : dict[str, Array]
      Dict with ``"d_mass_init"`` and ``"d_mass_end"`` (layer mass at the
      beginning and end of the tracer interval).
  grid : SpectralElementGrid
      Horizontal grid struct.
  dims : tuple[int, ...]
      Grid dimension tuple; static JIT argument.
  physics_config : dict
      Physics configuration dict.
  diffusion_config : dict
      Hyperviscosity configuration dict.
  timestep_config : frozendict
      Time-stepping configuration; static JIT argument.
  model : str
      Model identifier; static JIT argument.
  tracer_consist_hypervis : dict[str, Array] or None, optional
      Hyperviscosity consistency struct from
      :func:`wrap_tracer_consist_hypervis`; ``None`` disables the correction.

  Returns
  -------
  tracers_out : dict
      Updated tracer state dict with the same structure as ``tracers``.
  """
  u_d_mass_avg = tracer_consist_dyn["u_d_mass_avg"]
  d_mass_init = tracer_init_struct["d_mass_init"]
  d_mass_end = tracer_init_struct["d_mass_end"]
  d_mass_dyn_tend = -horizontal_divergence_3d(u_d_mass_avg, grid, physics_config)
  if tracer_consist_hypervis is not None:
    d_mass_hypervis_tend = tracer_consist_hypervis["d_mass_hypervis_tend"]
    d_mass_hypervis_avg = tracer_consist_hypervis["d_mass_hypervis_avg"]
  else:
    d_mass_hypervis_tend = None
    d_mass_hypervis_avg = None
  stacked_tracers, tracer_names = flatten_tracers(tracers, model)
  stacked_tracer_mass = stacked_tracers * d_mass_init[jnp.newaxis, :, :, :, :]
  stacked_tracer_mass_out = advance_tracers_rk2(stacked_tracer_mass,
                                                d_mass_init,
                                                u_d_mass_avg,
                                                d_mass_dyn_tend,
                                                grid,
                                                physics_config,
                                                diffusion_config,
                                                timestep_config,
                                                dims,
                                                d_mass_hypervis_tend=d_mass_hypervis_tend,
                                                d_mass_hypervis_avg=d_mass_hypervis_avg)
  stacked_tracer_out = stacked_tracer_mass_out / d_mass_end[jnp.newaxis, :, :, :, :]
  return ravel_tracers(stacked_tracer_out, tracer_names, model)
