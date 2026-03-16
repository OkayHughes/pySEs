from ...test_data.mass_coordinate_grids import cam30
from pyses._config import get_backend as _get_backend
from pyses.analytic_initialization.williamson_init import (init_williamson_steady_state_config,
                                                           init_williamson_steady_state_state)
from pyses.dynamical_cores.run_dycore import init_simulator
from pyses.mesh_generation.element_local_metric import init_quasi_uniform_grid_elem_local
from pyses.dynamical_cores.mass_coordinate import init_vertical_grid, d_mass_to_surface_mass
from pyses.dynamical_cores.model_info import models
from pyses.dynamical_cores.model_config import init_default_config, hypervis_opts
from ...context import get_figdir
_be = _get_backend()
jnp = _be.np


def test_theta_baro_wave_topo():
  npt = 4
  nx = 9
  h_grid, dims = init_quasi_uniform_grid_elem_local(nx, npt, calc_smooth_tensor=True)
  model = models.shallow_water
  b_coeffs = jnp.linspace(0.0, 1.0, 2)
  a_coeffs = jnp.zeros_like(b_coeffs)
  reference_mass = 0.0
  v_grid = init_vertical_grid(a_coeffs,
                              b_coeffs,
                              reference_mass,
                              model)

  total_time = (3600.0 * 24.0) * 3
  diffusion = hypervis_opts.variable_resolution
  physics_config, diffusion_config, timestep_config = init_default_config(nx, h_grid, v_grid, dims, model,
                                                                          hypervis_type=diffusion)
  physics_config["alpha"] = jnp.pi / 4
  test_config = init_williamson_steady_state_config(model_config=physics_config)
  model_state = init_williamson_steady_state_state(h_grid, v_grid, physics_config, test_config,
                                                    dims, model)
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
    surf_mass = d_mass_to_surface_mass(state["dynamics"]["d_mass"], v_grid)
    plt.figure()
    plt.tricontourf(h_grid["physical_coords"][:, :, :, 1].flatten(),
                    h_grid["physical_coords"][:, :, :, 0].flatten(),
                    surf_mass.flatten())
    plt.colorbar()
    #plt.savefig(f"{get_figdir()}/h_{t}.pdf")
    plt.figure()
    plt.tricontourf(h_grid["physical_coords"][:, :, :, 1].flatten(),
                    h_grid["physical_coords"][:, :, :, 0].flatten(),
                    state["dynamics"]["horizontal_wind"][:, :, :, 0, 0].flatten())
    plt.colorbar()
    #plt.savefig(f"{get_figdir()}/u_{t}.pdf")
    for lev in range(model_state["dynamics"]["horizontal_wind"].shape[3]):
      plt.figure()
      plt.tricontourf(h_grid["physical_coords"][:, :, :, 1].flatten(),
                      h_grid["physical_coords"][:, :, :, 0].flatten(),
                      (state["dynamics"]["horizontal_wind"][:, :, :, lev, 1].flatten() -
                      model_state["dynamics"]["horizontal_wind"][:, :, :, lev, 1].flatten()))
      plt.colorbar()
      plt.savefig(f"{get_figdir()}/v_{t}_lev_{lev}.pdf")
    if t > total_time:
      break
