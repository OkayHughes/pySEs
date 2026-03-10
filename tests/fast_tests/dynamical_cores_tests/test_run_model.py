from ...test_data.mass_coordinate_grids import cam30
from pyses._config import get_backend as _get_backend
from pyses.analytic_initialization.moist_baroclinic_wave import (init_baroclinic_wave_config,
                                                               perturbation_opts,
                                                               init_baroclinic_wave_state)
from pyses.dynamical_cores.run_dycore import init_simulator
from pyses.mesh_generation.element_local_metric import init_quasi_uniform_grid_elem_local
from pyses.dynamical_cores.mass_coordinate import init_vertical_grid
from pyses.dynamical_cores.model_info import models
from pyses.dynamical_cores.model_config import init_default_config, hypervis_opts
from ...context import get_figdir
_be = _get_backend()
jnp = _be.np


def test_theta_steady_state():
  for model in [models.homme_hydrostatic, models.cam_se]:
    npt = 4
    nx = 15
    h_grid, dims = init_quasi_uniform_grid_elem_local(nx, npt, calc_smooth_tensor=True)
    v_grid = init_vertical_grid(cam30["hybrid_a_i"],
                                cam30["hybrid_b_i"],
                                cam30["p0"],
                                model)

    total_time = (3600.0 * 6.0)
    for diffusion in [hypervis_opts.variable_resolution, hypervis_opts.quasi_uniform]:
      print("=" * 10)
      print(f"starting {diffusion}")
      print("=" * 10)
      physics_config, diffusion_config, timestep_config = init_default_config(nx, h_grid, v_grid,
                                                                              dims, model,
                                                                              hypervis_type=diffusion)
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

      for t, state in simulator(model_state):
        print(t)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.tricontourf(h_grid["physical_coords"][:, :, :, 1].flatten(),
                        h_grid["physical_coords"][:, :, :, 0].flatten(),
                        jnp.max(state["dynamics"]["horizontal_wind"][:, :, :, :, 0], axis=-1).flatten())
        plt.colorbar()
        plt.savefig(f"{get_figdir()}/u_{t}.pdf")
        plt.figure()
        plt.tricontourf(h_grid["physical_coords"][:, :, :, 1].flatten(),
                        h_grid["physical_coords"][:, :, :, 0].flatten(),
                        jnp.max(state["dynamics"]["horizontal_wind"][:, :, :, :, 1], axis=-1).flatten())
        plt.colorbar()
        plt.savefig(f"{get_figdir()}/v_{t}.pdf")
        if t > total_time:
          break


def test_theta_baro_wave_topo():
  npt = 4
  nx = 7
  h_grid, dims = init_quasi_uniform_grid_elem_local(nx, npt, calc_smooth_tensor=True)
  model = models.homme_hydrostatic
  v_grid = init_vertical_grid(cam30["hybrid_a_i"],
                              cam30["hybrid_b_i"],
                              cam30["p0"],
                              model)

  total_time = (3600.0 * 6.0)
  for diffusion in [hypervis_opts.variable_resolution, hypervis_opts.quasi_uniform, hypervis_opts.none]:
    physics_config, diffusion_config, timestep_config = init_default_config(nx, h_grid, v_grid, dims, model,
                                                                            hypervis_type=diffusion)
    test_config = init_baroclinic_wave_config(model_config=physics_config)
    model_state = init_baroclinic_wave_state(h_grid, v_grid, physics_config, test_config,
                                             dims, model, mountain=True,
                                             pert_type=perturbation_opts.none)
    simulator = init_simulator(h_grid, v_grid,
                               physics_config,
                               diffusion_config,
                               timestep_config,
                               dims,
                               model)

    t = 0.0
    for t, state in simulator(model_state):
      print(t)
      if t > total_time:
        break
