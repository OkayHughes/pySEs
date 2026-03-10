from pyses._config import get_backend as _get_backend
from pyses.shallow_water_models.run_shallow_water import simulate_shallow_water
from pyses.shallow_water_models.model_state import wrap_model_state
from pyses.shallow_water_models.constants import init_physics_config_shallow_water
from pyses.shallow_water_models.time_stepping import init_timestep_config
from pyses.shallow_water_models.hyperviscosity import (init_hypervis_config_const,
                                                     init_hypervis_config_tensor,
                                                     diffusion_config_for_tracer_consist)
from pyses.shallow_water_models.williamson_init import (init_williamson_steady_config,
                                                      eval_williamson_tc2_h,
                                                      eval_williamson_tc2_hs,
                                                      eval_williamson_tc2_u)
from pyses.shallow_water_models.galewsky_init import (init_galewsky_config,
                                                    eval_galewsky_wind,
                                                    eval_galewsky_hs,
                                                    eval_galewsky_h)
from pyses.mesh_generation.equiangular_metric import init_quasi_uniform_grid
from pyses.mesh_generation.element_local_metric import init_stretched_grid_elem_local
from ...context import get_figdir
_be = _get_backend()
jnp = _be.np
device_wrapper = _be.array


def test_sw_model():
  npt = 4
  nx = 15
  grid, dims = init_quasi_uniform_grid(nx, npt)
  physics_config = init_physics_config_shallow_water(alpha=jnp.pi / 4)
  test_config = init_williamson_steady_config(physics_config)
  u_init = device_wrapper(eval_williamson_tc2_u(grid["physical_coords"][:, :, :, 0],
                                                grid["physical_coords"][:, :, :, 1],
                                                test_config))
  h_init = device_wrapper(eval_williamson_tc2_h(grid["physical_coords"][:, :, :, 0],
                                                grid["physical_coords"][:, :, :, 1],
                                                test_config))
  hs_init = device_wrapper(eval_williamson_tc2_hs(grid["physical_coords"][:, :, :, 0],
                                                  grid["physical_coords"][:, :, :, 1],
                                                  test_config))
  print(u_init.dtype)
  init_state = wrap_model_state(u_init, h_init, hs_init)
  tracers_in = {"constant": jnp.ones_like(grid["metric_determinant"]),
                "cos_sq_lon": (jnp.sin(grid["physical_coords"][:, :, :, 1])**2 *
                               jnp.cos(grid["physical_coords"][:, :, :, 0]))}
  T = 6000.0
  dt = 600
  diffusion_config = init_hypervis_config_const(nx, physics_config, nu_div_factor=1.0)
  diffusion_config = diffusion_config_for_tracer_consist(diffusion_config)
  # diffusion_config["nu_tracer"] = 0.0
  timestep_config = init_timestep_config(dt, grid, dims, physics_config,
                                         diffusion_config, sphere=True)
  final_state = simulate_shallow_water(T, init_state, grid,
                                       physics_config, diffusion_config, timestep_config,
                                       dims, diffusion=False, tracers_in=tracers_in)

  print(jnp.max(final_state["tracers"]["constant"]))
  print(jnp.min(final_state["tracers"]["constant"]))
  import matplotlib.pyplot as plt
  plt.figure()
  plt.tricontourf(grid["physical_coords"][:, :, :, 1].flatten(),
                  grid["physical_coords"][:, :, :, 0].flatten(),
                  final_state["tracers"]["constant"].flatten())
  plt.colorbar()
  plt.savefig(f"{get_figdir()}/constant_tracer.pdf")
  plt.figure()
  plt.tricontourf(grid["physical_coords"][:, :, :, 1].flatten(),
                  grid["physical_coords"][:, :, :, 0].flatten(),
                  final_state["tracers"]["cos_sq_lon"].flatten())
  plt.colorbar()
  plt.savefig(f"{get_figdir()}/cossq_lon_tracer.pdf")


def test_sw_model_diff():
  npt = 4
  nx = 15
  grid, dims = init_quasi_uniform_grid(nx, npt)
  physics_config = init_physics_config_shallow_water(alpha=jnp.pi / 4)
  test_config = init_williamson_steady_config(physics_config)
  u_init = device_wrapper(eval_williamson_tc2_u(grid["physical_coords"][:, :, :, 0],
                                                grid["physical_coords"][:, :, :, 1],
                                                test_config))
  h_init = device_wrapper(eval_williamson_tc2_h(grid["physical_coords"][:, :, :, 0],
                                                grid["physical_coords"][:, :, :, 1],
                                                test_config))
  hs_init = device_wrapper(eval_williamson_tc2_hs(grid["physical_coords"][:, :, :, 0],
                                                  grid["physical_coords"][:, :, :, 1],
                                                  test_config))
  print(u_init.dtype)
  init_state = wrap_model_state(u_init, h_init, hs_init)
  tracers_in = {"constant": jnp.ones_like(grid["metric_determinant"]),
                "cos_sq_lon": (jnp.sin(grid["physical_coords"][:, :, :, 1])**2 *
                               jnp.cos(grid["physical_coords"][:, :, :, 0]))}
  T = 6000.0
  dt = 600
  diffusion_config = init_hypervis_config_const(nx, physics_config, nu_div_factor=1.0)
  timestep_config = init_timestep_config(dt, grid, dims, physics_config,
                                         diffusion_config, sphere=True)
  final_state = simulate_shallow_water(T, init_state, grid,
                                       physics_config, diffusion_config, timestep_config,
                                       dims, diffusion=True, tracers_in=tracers_in)

  print(jnp.max(final_state["tracers"]["constant"]))
  print(jnp.min(final_state["tracers"]["constant"]))
  import matplotlib.pyplot as plt
  plt.figure()
  plt.tricontourf(grid["physical_coords"][:, :, :, 1].flatten(),
                  grid["physical_coords"][:, :, :, 0].flatten(),
                  final_state["tracers"]["constant"].flatten())
  plt.colorbar()
  plt.savefig(f"{get_figdir()}/constant_tracer_hvis.pdf")
  plt.figure()
  plt.tricontourf(grid["physical_coords"][:, :, :, 1].flatten(),
                  grid["physical_coords"][:, :, :, 0].flatten(),
                  final_state["tracers"]["cos_sq_lon"].flatten())
  plt.colorbar()
  plt.savefig(f"{get_figdir()}/cossq_lon_tracer_hvis.pdf")


def test_galewsky():
  npt = 4
  nx = 31
  grid, dims = init_stretched_grid_elem_local(nx, npt, axis_dilation=jnp.array([1.0, 1.5, 1.0]))

  physics_config = init_physics_config_shallow_water()
  test_config = init_galewsky_config(physics_config)

  dt = 300
  T = (24 * 3600) * 2
  diffusion_config_const = init_hypervis_config_const(nx, physics_config, nu_div_factor=1.0)
  diffusion_config_tensor = init_hypervis_config_tensor(grid, dims, physics_config)
  for diffusion_config in [diffusion_config_const, diffusion_config_tensor]:
    u_init = device_wrapper(eval_galewsky_wind(grid["physical_coords"][:, :, :, 0],
                                               grid["physical_coords"][:, :, :, 1],
                                               test_config))
    h_init = device_wrapper(eval_galewsky_h(grid["physical_coords"][:, :, :, 0],
                                            grid["physical_coords"][:, :, :, 1],
                                            test_config))
    hs_init = device_wrapper(eval_galewsky_hs(grid["physical_coords"][:, :, :, 0],
                                              grid["physical_coords"][:, :, :, 1],
                                              test_config))
    tracers_in = {"constant": jnp.ones_like(grid["metric_determinant"]),
                  "cos_sq_lon": (jnp.sin(grid["physical_coords"][:, :, :, 1])**2 *
                                 jnp.cos(grid["physical_coords"][:, :, :, 0]))}
    init_state = wrap_model_state(u_init, h_init, hs_init)

    timestep_config = init_timestep_config(dt, grid, dims, physics_config,
                                           diffusion_config, sphere=True)
    final_state = simulate_shallow_water(T, init_state, grid,
                                         physics_config, diffusion_config, timestep_config,
                                         dims, diffusion=True, tracers_in=tracers_in)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.tricontourf(grid["physical_coords"][:, :, :, 1].flatten(),
                    grid["physical_coords"][:, :, :, 0].flatten(),
                    final_state["tracers"]["constant"].flatten())
    plt.colorbar()
    plt.savefig(f"{get_figdir()}/constant_tracer_hvis.pdf")
    plt.figure()
    plt.tricontourf(grid["physical_coords"][:, :, :, 1].flatten(),
                    grid["physical_coords"][:, :, :, 0].flatten(),
                    final_state["tracers"]["cos_sq_lon"].flatten())
    plt.colorbar()
    plt.savefig(f"{get_figdir()}/cossq_lon_tracer_hvis.pdf")
