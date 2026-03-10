from src._config import get_backend as _get_backend
from ....test_data.mass_coordinate_grids import cam30
from ....context import get_figdir
from src.analytic_initialization.moist_baroclinic_wave import (init_baroclinic_wave_config,
                                                               perturbation_opts,
                                                               init_baroclinic_wave_state)
from src.dynamical_cores.run_dycore import init_simulator
from src.dynamical_cores.hyperviscosity import diffusion_config_for_tracer_consist
from src.mesh_generation.equiangular_metric import init_quasi_uniform_grid
from src.dynamical_cores.mass_coordinate import init_vertical_grid
from src.dynamical_cores.model_info import models
from src.dynamical_cores.model_config import init_default_config, hypervis_opts
_be = _get_backend()
jnp = _be.np


def test_tracer_consistency():
  model = models.homme_hydrostatic
  npt = 4
  nx = 7
  h_grid, dims = init_quasi_uniform_grid(nx, npt, calc_smooth_tensor=True)
  v_grid = init_vertical_grid(cam30["hybrid_a_i"],
                              cam30["hybrid_b_i"],
                              cam30["p0"],
                              model)

  total_time = (3600.0 * 3.0)
  for diffusion in [hypervis_opts.quasi_uniform, hypervis_opts.none, hypervis_opts.variable_resolution]:
    print("=" * 10)
    print(f"starting {diffusion}")
    print("=" * 10)
    physics_config, diffusion_config, timestep_config = init_default_config(nx, h_grid, v_grid,
                                                                            dims, model,
                                                                            hypervis_type=diffusion,
                                                                            physics_dt=600)
    diffusion_config = diffusion_config_for_tracer_consist(diffusion_config, v_grid)
    test_config = init_baroclinic_wave_config(model_config=physics_config)
    model_state = init_baroclinic_wave_state(h_grid,
                                             v_grid,
                                             physics_config,
                                             test_config,
                                             dims,
                                             model,
                                             mountain=False,
                                             pert_type=perturbation_opts.exponential)
    model_state["tracers"]["tracers"]["constant"] = jnp.ones_like(model_state["dynamics"]["d_mass"])
    coscos = (jnp.cos(h_grid["physical_coords"][:, :, :, 0])**2 *
              jnp.cos(h_grid["physical_coords"][:, :, :, 1])**2)[:, :, :, jnp.newaxis]
    model_state["tracers"]["tracers"]["coscos"] = jnp.ones_like(model_state["dynamics"]["d_mass"]) * coscos
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
      print(jnp.max(state["tracers"]["tracers"]["constant"]))
      print(jnp.min(state["tracers"]["tracers"]["constant"]))
      plt.figure()
      plt.tricontourf(h_grid["physical_coords"][:, :, :, 1].flatten(),
                      h_grid["physical_coords"][:, :, :, 0].flatten(),
                      jnp.sum(state["tracers"]["tracers"]["coscos"] *
                              state["dynamics"]["d_mass"], axis=-1).flatten())
      plt.colorbar()
      plt.savefig(f"{get_figdir()}/cos_cos_{t}_sum.pdf")
      if t > total_time:
        break
