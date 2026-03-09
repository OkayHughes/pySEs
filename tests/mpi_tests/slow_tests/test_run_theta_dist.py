from ...test_data.mass_coordinate_grids import cam30
from src._config import get_backend as _get_backend
_be = _get_backend()
device_unwrapper = _be.unwrap
mpi_rank = _be.mpi_rank
is_main_proc = _be.is_main_proc
import numpy as np
from src.analytic_initialization.moist_baroclinic_wave import init_baroclinic_wave_config, perturbation_opts, init_baroclinic_wave_state
from src.dynamical_cores.run_dycore import init_simulator
from src.mesh_generation.equiangular_metric import init_quasi_uniform_grid
from src.dynamical_cores.mass_coordinate import init_vertical_grid
from src.operations_2d.horizontal_grid import make_grid_mpi_ready
from src.dynamical_cores.model_info import models
from src.dynamical_cores.model_config import init_default_config, hypervis_opts
from sys import stdout


def test_theta_steady_state():
  for model in [models.homme_hydrostatic, models.cam_se]:
    npt = 4
    nx = 15
    h_grid, dims = init_quasi_uniform_grid(nx, npt)
    h_grid, dims = make_grid_mpi_ready(h_grid, dims, mpi_rank)
    v_grid = init_vertical_grid(cam30["hybrid_a_i"],
                                cam30["hybrid_b_i"],
                                cam30["p0"],
                                model)

    total_time = (3600.0 * 24.0)
    diffusion = hypervis_opts.variable_resolution
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
      if is_main_proc:
        print(t)
        stdout.flush()
      if t > total_time:
        break

    end_state = state["dynamics"]

    assert(not np.any(np.isnan(device_unwrapper(end_state["horizontal_wind"][:, :, :, :, 1]))))


def test_theta_baro_wave():
  npt = 4
  nx = 15
  h_grid, dims = init_quasi_uniform_grid(nx, npt)
  h_grid, dims = make_grid_mpi_ready(h_grid, dims, mpi_rank)
  model = models.homme_hydrostatic
  v_grid = init_vertical_grid(cam30["hybrid_a_i"],
                              cam30["hybrid_b_i"],
                              cam30["p0"],
                              model)

  total_time = (3600.0 * 24.0)
  diffusion = hypervis_opts.variable_resolution
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
    if is_main_proc:
      print(t)
      stdout.flush()
    if t > total_time:
      break
