from pysces.config import jnp
from ....test_data.mass_coordinate_grids import cam30
from ....context import get_figdir
from pysces.analytic_initialization.moist_baroclinic_wave import init_baroclinic_wave_config, perturbation_opts
from pysces.run_dycore import init_simulator
from pysces.dynamical_cores.hyperviscosity import diffusion_config_for_tracer_consist
from pysces.mesh_generation.equiangular_metric import init_quasi_uniform_grid
from pysces.dynamical_cores.mass_coordinate import init_vertical_grid
from pysces.model_info import models
from pysces.dynamical_cores.model_config import init_default_config, hypervis_opts
from pysces.initialization import init_baroclinic_wave_state


def test_():
  for model in [models.cam_se]:
    npt = 4
    nx = 7
    h_grid, dims = init_quasi_uniform_grid(nx, npt, calc_smooth_tensor=True)
    v_grid = init_vertical_grid(cam30["hybrid_a_i"],
                                cam30["hybrid_b_i"],
                                cam30["p0"],
                                model)

    total_time = (3600.0 * 6.0)
    for diffusion in [hypervis_opts.none, hypervis_opts.quasi_uniform, hypervis_opts.variable_resolution]:
      print("=" * 10)
      print(f"starting {diffusion}")
      print("=" * 10)
      physics_config, diffusion_config, timestep_config = init_default_config(nx, h_grid, v_grid,
                                                                              dims, model,
                                                                              hypervis_type=diffusion,
                                                                              physics_dt=600)
      if diffusion == hypervis_opts.none:
        diffusion_config = diffusion_config_for_tracer_consist(diffusion_config, v_grid)
      else:
        diffusion_config["nu_top"] = 0.0
      test_config = init_baroclinic_wave_config(model_config=physics_config)
      model_state = init_baroclinic_wave_state(h_grid, v_grid, physics_config, test_config, dims, model, mountain=False)
      simulator = init_simulator(h_grid, v_grid,
                                 physics_config,
                                 diffusion_config,
                                 timestep_config,
                                 dims,
                                 model)

      t = 0.0
      import matplotlib.pyplot as plt
      for t, state in simulator(model_state):
        print(t)
        print(jnp.max(state["tracers"]["dry_air_species"]["dry_air"]))
        print(jnp.min(state["tracers"]["dry_air_species"]["dry_air"]))
        plt.figure()
        plt.tricontourf(h_grid["physical_coords"][:, :, :, 1].flatten(),
                        h_grid["physical_coords"][:, :, :, 0].flatten(),
                        jnp.max(state["tracers"]["dry_air_species"]["dry_air"], axis=-1).flatten())
        plt.colorbar()
        plt.savefig(f"{get_figdir()}/{t}_max.pdf")
        plt.figure()
        plt.tricontourf(h_grid["physical_coords"][:, :, :, 1].flatten(),
                        h_grid["physical_coords"][:, :, :, 0].flatten(),
                        jnp.max(state["tracers"]["dry_air_species"]["dry_air"], axis=-1).flatten())
        plt.colorbar()
        plt.savefig(f"{get_figdir()}/{t}_min.pdf")
        if t > total_time:
          break
