from ...config import jit, jnp
from functools import partial
from ..operators_3d import horizontal_divergence_3d
from ...tracer_transport.eulerian_spectral import advance_tracers_rk2
from ...model_info import cam_se_models, homme_models


@partial(jit, static_argnames=["model"])
def flatten_tracers(tracers, model):
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
