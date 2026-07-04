from ...test_data.mass_coordinate_grids import cam30
from pyses._config import get_backend as _get_backend
from pyses.analytic_initialization.moist_baroclinic_wave import (init_baroclinic_wave_config,
                                                               perturbation_opts,
                                                               init_baroclinic_wave_state)
from pyses.analytic_initialization.nonhydrostatic_mountain_wave import (init_mountain_wave_config,
                                                                      init_mountain_wave_state,
                                                                      dcmip_surface_shear)
from pyses.dynamical_cores.run_dycore import init_simulator
from .conftest import cached_quasi_uniform_grid_elem_local
from pyses.dynamical_cores.mass_coordinate import init_vertical_grid
from pyses.dynamical_cores.model_info import models
from pyses.dynamical_cores.physics_config import init_physics_config
from pyses.dynamical_cores.model_config import init_default_config, hypervis_opts
_be = _get_backend()
jnp = _be.np

# DCMIP-2012 small-planet reduction factor X for the non-hydrostatic
# mountain-wave tests (Table XIII); the scaled radius is 6371 km / X.
MOUNTAIN_WAVE_X = 500.0


def _run_mountain_wave(model, shear, nx=7, nsteps=4,
                       enforce_hydrostatic=False,
                       hv=hypervis_opts.variable_resolution):
  """Initialise and step the DCMIP non-hydrostatic Schar mountain-wave test.

  Builds a non-rotating, reduced-size planet (``radius_earth = 6371 km / X``,
  ``angular_freq_earth = 0``), scales the coupling timestep by ``1 / X`` so the
  CFL-derived dynamics subcycling stays comparable to a full-planet run, and
  advances ``nsteps`` coupling steps.  The simulator asserts finiteness of the
  dynamics and tracer state after every step, so reaching ``nsteps`` without an
  exception is the pass condition.
  """
  npt = 4
  h_grid, dims = cached_quasi_uniform_grid_elem_local(nx, npt, calc_smooth_tensor=True)
  v_grid = init_vertical_grid(cam30["hybrid_a_i"],
                              cam30["hybrid_b_i"],
                              cam30["p0"],
                              model)
  # Reduced-size, non-rotating planet.  Passing this physics_config into
  # init_default_config keeps the diffusion and CFL time-step consistent with
  # the scaled radius.
  physics_config = init_physics_config(model,
                                       radius_earth=6371e3 / MOUNTAIN_WAVE_X,
                                       angular_freq_earth=0.0)
  physics_dt = (900.0 * 30.0 / nx) / MOUNTAIN_WAVE_X
  physics_config, diffusion_config, timestep_config = init_default_config(
      nx, h_grid, v_grid, dims, model, physics_dt=physics_dt,
      hypervis_type=hv, physics_config=physics_config)
  test_config = init_mountain_wave_config(shear=shear, model_config=physics_config)
  model_state = init_mountain_wave_state(h_grid, v_grid, physics_config, test_config,
                                         dims, model, mountain=True,
                                         enforce_hydrostatic=enforce_hydrostatic)
  simulator = init_simulator(h_grid, v_grid,
                             physics_config,
                             diffusion_config,
                             timestep_config,
                             dims,
                             model)
  ct = 0
  for t, state in simulator(model_state):
    print(t)
    ct += 1
    if ct >= nsteps:
      break


def test_mountain_wave_non_sheared():
  # DCMIP test 2-1: vertically uniform background jet (shear parameter c = 0).
  model = models.homme_nonhydrostatic
  _run_mountain_wave(model, shear=0.0,
                     nsteps=3, enforce_hydrostatic=True)


def test_mountain_wave_sheared():
  # DCMIP test 2-2: vertically sheared background jet (c = cs).
  model = models.homme_nonhydrostatic
  _run_mountain_wave(model, shear=dcmip_surface_shear,
                     nsteps=3, enforce_hydrostatic=True)


def test_mountain_wave_acid():
  # "Acid" stress test: the sheared jet driven over the Schar ridge on the fully
  # non-hydrostatic reduced planet -- the configuration that discriminates most
  # sharply between hydrostatic and non-hydrostatic responses.  Exercises the
  # non-hydrostatic HEVI path under the mountain forcing.
  _run_mountain_wave(models.homme_nonhydrostatic, shear=dcmip_surface_shear,
                     nsteps=3, enforce_hydrostatic=True)


def test_theta_steady_state():
  for model in [models.homme_hydrostatic, models.cam_se]:
    npt = 4
    nx = 7
    h_grid, dims = cached_quasi_uniform_grid_elem_local(nx, npt, calc_smooth_tensor=True)
    v_grid = init_vertical_grid(cam30["hybrid_a_i"],
                                cam30["hybrid_b_i"],
                                cam30["p0"],
                                model)

    total_time = (3600.0 * 2.0)
    for diffusion in [hypervis_opts.variable_resolution, hypervis_opts.quasi_uniform]:
      print(model)
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
        if t > total_time:
          break


def test_theta_baro_wave_topo():
  npt = 4
  nx = 7
  h_grid, dims = cached_quasi_uniform_grid_elem_local(nx, npt, calc_smooth_tensor=True)
  model = models.homme_hydrostatic
  v_grid = init_vertical_grid(cam30["hybrid_a_i"],
                              cam30["hybrid_b_i"],
                              cam30["p0"],
                              model)

  total_time = (3600.0 * 2.0)
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
