import numpy as np

from ...test_data.mass_coordinate_grids import cam30
from pyses._config import get_backend as _get_backend
from pyses.analytic_initialization.galewsky_init import (init_galewsky_config,
                                                          init_galewsky_state)
from pyses.analytic_initialization.williamson_init import (init_williamson_steady_state_config,
                                                           init_williamson_steady_state_state)
from pyses.dynamical_cores.run_dycore import init_simulator
from .conftest import cached_quasi_uniform_grid_elem_local
from pyses.dynamical_cores.mass_coordinate import init_vertical_grid, d_mass_to_surface_mass
from pyses.dynamical_cores.model_info import models
from pyses.dynamical_cores.model_config import init_default_config, hypervis_opts
from ...context import get_figdir, emit_plots
_be = _get_backend()
jnp = _be.np
unwrap = _be.unwrap


def test_theta_baro_wave_topo():
  npt = 4
  nx = 9
  h_grid, dims = cached_quasi_uniform_grid_elem_local(nx, npt, calc_smooth_tensor=True)
  model = models.shallow_water
  b_coeffs = jnp.linspace(0.0, 1.0, 2)
  a_coeffs = jnp.zeros_like(b_coeffs)
  reference_mass = 0.0
  v_grid = init_vertical_grid(a_coeffs,
                              b_coeffs,
                              reference_mass,
                              model)

  total_time = (3600.0 * 6.0)
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

  savedir = get_figdir(subdir="shallow_water_baro_wave_topo") if emit_plots() else None
  t = 0.0
  for t, state in simulator(model_state):
    print(t)
    if savedir is not None:
      import matplotlib.pyplot as plt
      surf_mass = d_mass_to_surface_mass(state["dynamics"]["d_mass"], v_grid)
      plt.figure()
      plt.tricontourf(h_grid["physical_coords"][:, :, :, 1].flatten(),
                      h_grid["physical_coords"][:, :, :, 0].flatten(),
                      surf_mass.flatten())
      plt.colorbar()
      plt.savefig(f"{savedir}/h_{t}.pdf")
      plt.close()
      plt.figure()
      plt.tricontourf(h_grid["physical_coords"][:, :, :, 1].flatten(),
                      h_grid["physical_coords"][:, :, :, 0].flatten(),
                      state["dynamics"]["horizontal_wind"][:, :, :, 0, 0].flatten())
      plt.colorbar()
      plt.savefig(f"{savedir}/u_{t}.pdf")
      plt.close()
      for lev in range(model_state["dynamics"]["horizontal_wind"].shape[3]):
        plt.figure()
        plt.tricontourf(h_grid["physical_coords"][:, :, :, 1].flatten(),
                        h_grid["physical_coords"][:, :, :, 0].flatten(),
                        (state["dynamics"]["horizontal_wind"][:, :, :, lev, 1].flatten() -
                        model_state["dynamics"]["horizontal_wind"][:, :, :, lev, 1].flatten()))
        plt.colorbar()
        plt.savefig(f"{savedir}/v_{t}_lev_{lev}.pdf")
        plt.close()
    if t > total_time:
      break


def test_galewsky_ne60():
  """Run the Galewsky barotropic-instability test on an ne60 single-layer
  shallow-water grid for the first 6 hours of simulated time.

  At ne60 the unstable jet is well-resolved by ~0.6 deg GLL spacing; six
  hours is enough for the perturbation to start rolling up but not so
  long that the test takes forever.  The assertion is purely on
  finiteness -- divergence here means the explicit RK3 + hypervis +
  remap loop crashed something fundamental, not that the long-term
  spin-up is wrong.
  """
  npt = 4
  nx = 60
  h_grid, dims = cached_quasi_uniform_grid_elem_local(
      nx, npt, calc_smooth_tensor=True)
  model = models.shallow_water
  # Single-layer 3-D shallow-water configuration: ``hybrid_a_i`` /
  # ``hybrid_b_i`` of length 2, ``reference_surface_mass`` = 0 so the
  # prognostic ``d_mass`` is just the free-surface height in kg / m^2.
  b_coeffs = jnp.linspace(0.0, 1.0, 2)
  a_coeffs = jnp.zeros_like(b_coeffs)
  reference_mass = 0.0
  v_grid = init_vertical_grid(a_coeffs, b_coeffs, reference_mass, model)

  total_time = 6.0 * 3600.0
  physics_config, diffusion_config, timestep_config = init_default_config(
      nx, h_grid, v_grid, dims, model,
      hypervis_type=hypervis_opts.variable_resolution)

  test_config = init_galewsky_config(physics_config)
  model_state = init_galewsky_state(h_grid, v_grid, physics_config,
                                    test_config, dims, model)
  # Sanity-check the initial state before running anything.
  for key in ("horizontal_wind", "d_mass"):
    arr = np.asarray(unwrap(model_state["dynamics"][key]))
    assert np.all(np.isfinite(arr)), f"non-finite {key} at init"

  simulator = init_simulator(h_grid, v_grid, physics_config,
                              diffusion_config, timestep_config, dims, model)

  state = model_state
  for t, state in simulator(model_state):
    if t >= total_time:
      break

  for key in ("horizontal_wind", "d_mass"):
    arr = np.asarray(unwrap(state["dynamics"][key]))
    assert np.all(np.isfinite(arr)), (
        f"non-finite {key} after {total_time / 3600.0:.0f} h of integration")
