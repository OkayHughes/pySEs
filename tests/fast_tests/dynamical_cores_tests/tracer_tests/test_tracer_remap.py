from pysces.config import jnp, np
from pysces.mesh_generation.equiangular_metric import init_quasi_uniform_grid
from pysces.initialization import init_baroclinic_wave_state
from pysces.analytic_initialization.moist_baroclinic_wave import init_baroclinic_wave_config
from pysces.model_info import models
from pysces.dynamical_cores.mass_coordinate import init_vertical_grid
from pysces.dynamical_cores.physics_config import init_physics_config
from pysces.dynamical_cores.mass_coordinate import d_mass_to_surface_mass, surface_mass_to_d_mass
from pysces.dynamical_cores.model_state import remap_tracers
from ....test_data.mass_coordinate_grids import cam30

def test_remap_tracers():
  npt = 4
  nx = 4
  h_grid, dims = init_quasi_uniform_grid(nx, npt)
  model = models.cam_se_whole_atmosphere
  v_grid = init_vertical_grid(cam30["hybrid_a_i"],
                              cam30["hybrid_b_i"],
                              cam30["p0"],
                              model)
  model_config = init_physics_config(model)
  test_config = init_baroclinic_wave_config(model_config=model_config)
  model_state = init_baroclinic_wave_state(h_grid,
                                            v_grid,
                                            model_config,
                                            test_config,
                                            dims,
                                            model,
                                            mountain=False,
                                            moist=True,
                                            eps=1e-3)
  model_state["tracers"]["josh"] = jnp.array(np.random.uniform(size=model_state["tracers"]["moisture_species"]["water_vapor"].shape))
  model_state["dynamics"]["d_mass"] *= jnp.array(np.random.uniform(size=model_state["dynamics"]["d_mass"].shape, high=1.05, low=0.95))
  surface_mass = d_mass_to_surface_mass(model_state["dynamics"]["d_mass"], v_grid)
  d_mass = model_state["dynamics"]["d_mass"]
  d_mass_ref = surface_mass_to_d_mass(surface_mass, v_grid)
  remapped_tracers = remap_tracers(model_state["dynamics"],
                                   model_state["tracers"],
                                   v_grid,
                                   len(v_grid["hybrid_b_m"]),
                                   model)
  for species_name in model_state["tracers"]["moisture_species"].keys():
    assert jnp.allclose(jnp.sum(d_mass * model_state["tracers"]["moisture_species"][species_name]),
                        jnp.sum(d_mass_ref * remapped_tracers["moisture_species"][species_name]))
  for species_name in model_state["tracers"]["tracers"].keys():
    assert jnp.allclose(jnp.sum(d_mass * model_state["tracers"]["tracers"][species_name]),
                        jnp.sum(d_mass_ref * remapped_tracers["tracers"][species_name]))
  for species_name in model_state["tracers"]["dry_air_species"].keys():
    assert jnp.allclose(jnp.sum(d_mass * model_state["tracers"]["dry_air_species"][species_name]),
                        jnp.sum(d_mass_ref * remapped_tracers["dry_air_species"][species_name]))
