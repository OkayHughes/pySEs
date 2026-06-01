from ..test_data.mass_coordinate_grids import cam30
import pytest
from ..context import get_figdir, plot_grid
from pyses._config import get_backend as _get_backend
from pyses.analytic_initialization.moist_baroclinic_wave import (init_baroclinic_wave_config,
                                                               perturbation_opts,
                                                               init_baroclinic_wave_state)
from pyses.dynamical_cores.run_dycore import init_simulator
from pyses.mesh_generation.equiangular_metric import init_quasi_uniform_grid
from pyses.mesh_generation.element_local_metric import init_stretched_grid_elem_local
from pyses.dynamical_cores.mass_coordinate import init_vertical_grid
from pyses.dynamical_cores.model_info import models, cam_se_models, homme_models
from pyses.dynamical_cores.model_config import init_default_config, hypervis_opts


_be = _get_backend()
jnp = _be.np
get_global_array = _be.get_global_array


def test_theta_steady_state():
  for model in [models.homme_hydrostatic, models.cam_se]:
    npt = 4
    nx = 15
    h_grid, dims = init_quasi_uniform_grid(nx, npt, calc_smooth_tensor=True)
    v_grid = init_vertical_grid(cam30["hybrid_a_i"],
                                cam30["hybrid_b_i"],
                                cam30["p0"],
                                model)

    total_time = (3600.0 * 24.0 * 1.0)
    hv_type = hypervis_opts.variable_resolution
    physics_config, diffusion_config, timestep_config = init_default_config(nx, h_grid, v_grid,
                                                                            dims, model,
                                                                            hypervis_type=hv_type)
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
      if t > total_time:
        break

    end_state = state["dynamics"]
    ps = v_grid["hybrid_a_i"][0] * v_grid["reference_surface_mass"] + jnp.sum(end_state["d_mass"], axis=-1)
    ps_begin = v_grid["hybrid_a_i"][0] * v_grid["reference_surface_mass"] + jnp.sum(end_state["d_mass"], axis=-1)
    figdir = get_figdir()
    if model in homme_models:
      thermo = end_state["theta_v_d_mass"][:, :, :, 12] / end_state["d_mass"][:, :, :, 12]
    elif model in cam_se_models:
      thermo = end_state["T"][:, :, :, 12]
    plt.figure()
    plt.tricontourf(get_global_array(h_grid["physical_coords"][:, :, :, 1], dims).flatten(),
                    get_global_array(h_grid["physical_coords"][:, :, :, 0], dims).flatten(),
                    get_global_array(ps, dims).flatten())
    plt.colorbar()
    plot_grid(h_grid, plt.gca())
    plt.savefig(f"{figdir}/final_state_steady_{model}.pdf")
    plt.figure()
    plt.tricontourf(get_global_array(h_grid["physical_coords"][:, :, :, 1], dims).flatten(),
                    get_global_array(h_grid["physical_coords"][:, :, :, 0], dims).flatten(),
                    get_global_array(ps - ps_begin, dims).flatten())
    plt.colorbar()
    plt.savefig(f"{figdir}/ps_diff_steady_{model}.pdf")
    plt.figure()
    plot_grid(h_grid, plt.gca())
    plt.tricontourf(get_global_array(h_grid["physical_coords"][:, :, :, 1], dims).flatten(),
                    get_global_array(h_grid["physical_coords"][:, :, :, 0], dims).flatten(),
                    get_global_array(end_state["horizontal_wind"][:, :, :, 12, 1], dims).flatten())
    plt.colorbar()
    plot_grid(h_grid, plt.gca())
    plt.savefig(f"{figdir}/v_end_steady_{model}.pdf")
    plt.figure()
    plt.tricontourf(get_global_array(h_grid["physical_coords"][:, :, :, 1], dims).flatten(),
                    get_global_array(h_grid["physical_coords"][:, :, :, 0], dims).flatten(),
                    get_global_array(end_state["horizontal_wind"][:, :, :, 12, 0], dims).flatten())
    plt.colorbar()
    plot_grid(h_grid, plt.gca())
    plt.savefig(f"{figdir}/u_end_steady_{model}.pdf")
    plt.figure()
    plt.tricontourf(get_global_array(h_grid["physical_coords"][:, :, :, 1], dims).flatten(),
                    get_global_array(h_grid["physical_coords"][:, :, :, 0], dims).flatten(),
                    get_global_array(thermo, dims).flatten())
    plt.colorbar()
    plot_grid(h_grid, plt.gca())
    plt.savefig(f"{figdir}/thermo_end_steady_{model}.pdf")

@pytest.mark.parametrize("model", [models.cam_se,
                                   models.homme_hydrostatic])
def test_theta_baro_wave(model):
  npt = 4
  nx = 15
  h_grid, dims = init_stretched_grid_elem_local(nx,
                                                npt,
                                                axis_dilation=jnp.array([1.0, 1.5, 1.0]),
                                                calc_smooth_tensor=True)
  v_grid = init_vertical_grid(cam30["hybrid_a_i"],
                              cam30["hybrid_b_i"],
                              cam30["p0"],
                              model)

  total_time = (3600.0 * 24.0 * 1.0)
  hv_type = hypervis_opts.variable_resolution
  physics_config, diffusion_config, timestep_config = init_default_config(nx, h_grid, v_grid, dims, model,
                                                                          hypervis_type=hv_type)
  test_config = init_baroclinic_wave_config(model_config=physics_config)
  model_state = init_baroclinic_wave_state(h_grid, v_grid, physics_config, test_config,
                                           dims, model, mountain=False)
  simulator = init_simulator(h_grid, v_grid,
                             physics_config,
                             diffusion_config,
                             timestep_config,
                             dims,
                             model)
  import matplotlib.pyplot as plt
  from os import makedirs
  from os.path import join
  figdir = join(get_figdir(), f"baro_wave__{model}")
  makedirs(figdir, exist_ok=True)
  t = 0.0
  ct = 0
  for t, state in simulator(model_state):
    print(t)
    if t > total_time:
      break
    if ct % 5 == 0:
      end_state = state["dynamics"]
      ps = v_grid["hybrid_a_i"][0] * v_grid["reference_surface_mass"] + jnp.sum(end_state["d_mass"], axis=-1)
      if model in cam_se_models:
        thermo = end_state["T"][:, :, :, 12]
      else:
        thermo = end_state["theta_v_d_mass"][:, :, :, 12] / end_state["d_mass"][:, :, :, 12]

      plt.figure()
      plt.tricontourf(get_global_array(h_grid["physical_coords"][:, :, :, 1], dims).flatten(),
                      get_global_array(h_grid["physical_coords"][:, :, :, 0], dims).flatten(),
                      get_global_array(ps, dims).flatten())
      plot_grid(h_grid, plt.gca())
      plt.colorbar()
      plt.savefig(f"{figdir}/final_state_bw_topo.pdf")
      plt.figure()
      plt.tricontourf(get_global_array(h_grid["physical_coords"][:, :, :, 1], dims).flatten(),
                      get_global_array(h_grid["physical_coords"][:, :, :, 0], dims).flatten(),
                      get_global_array(end_state["horizontal_wind"][:, :, :, 12, 1], dims).flatten())
      plt.colorbar()
      plot_grid(h_grid, plt.gca())
      plt.savefig(f"{figdir}/v_end_bw_topo.pdf")
      plt.figure()
      plt.tricontourf(get_global_array(h_grid["physical_coords"][:, :, :, 1], dims).flatten(),
                      get_global_array(h_grid["physical_coords"][:, :, :, 0], dims).flatten(),
                      get_global_array(end_state["horizontal_wind"][:, :, :, 12, 0], dims).flatten())
      plt.colorbar()
      plot_grid(h_grid, plt.gca())
      plt.savefig(f"{figdir}/u_end_bw_topo.pdf")
      plt.figure()
      plt.tricontourf(get_global_array(h_grid["physical_coords"][:, :, :, 1], dims).flatten(),
                      get_global_array(h_grid["physical_coords"][:, :, :, 0], dims).flatten(),
                      get_global_array(thermo, dims).flatten())
      plt.colorbar()
      plot_grid(h_grid, plt.gca())
      plt.savefig(f"{figdir}/theta_v_end_bw_topo.pdf")
    ct += 1
