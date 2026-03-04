from pysces.config import jnp, np
from pysces.mesh_generation.equiangular_metric import init_quasi_uniform_grid
from pysces.initialization import init_baroclinic_wave_state
from pysces.analytic_initialization.moist_baroclinic_wave import init_baroclinic_wave_config, perturbation_opts
from pysces.model_info import models
from pysces.dynamical_cores.mass_coordinate import init_vertical_grid
from pysces.dynamical_cores.physics_config import init_physics_config
from pysces.dynamical_cores.mass_coordinate import d_mass_to_surface_mass, surface_mass_to_d_mass
from pysces.dynamical_cores.model_state import remap_tracers
from ....test_data.mass_coordinate_grids import cam30


def test_remap_tracers():
  npt = 4
  nx = 8
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
                                           eps=1e-3,
                                           pert_type=perturbation_opts.exponential)
  # todo: test that tracer consistency is correct
