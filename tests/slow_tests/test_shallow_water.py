from src._config import get_backend as _get_backend
import numpy as np
from src.shallow_water_models.run_shallow_water import simulate_shallow_water
from src.shallow_water_models.model_state import wrap_model_state
from src.shallow_water_models.constants import init_physics_config_shallow_water
from src.shallow_water_models.time_stepping import init_timestep_config
from src.shallow_water_models.hyperviscosity import init_hypervis_config_const, init_hypervis_config_tensor
from src.shallow_water_models.williamson_init import (init_williamson_steady_config,
                                                      eval_williamson_tc2_h,
                                                      eval_williamson_tc2_hs,
                                                      eval_williamson_tc2_u)
from src.shallow_water_models.galewsky_init import (init_galewsky_config,
                                                    eval_galewsky_wind,
                                                    eval_galewsky_hs,
                                                    eval_galewsky_h)
from src.mesh_generation.equiangular_metric import init_quasi_uniform_grid
from src.mesh_generation.element_local_metric import init_stretched_grid_elem_local
from src.operations_2d.operators import inner_product, horizontal_vorticity
from src.operations_2d.local_assembly import project_scalar
from ..context import get_figdir, plot_grid
from os import makedirs
from os.path import join

_be = _get_backend()
jnp = _be.np
DEBUG = _be.debug
device_unwrapper = _be.unwrap
device_wrapper = _be.array


if DEBUG:
  import matplotlib.pyplot as plt


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

  T = 3600 * 72.0
  dt = 600
  diffusion_config = init_hypervis_config_const(nx, physics_config, nu_div_factor=1.0)
  timestep_config = init_timestep_config(dt, grid, dims, physics_config,
                                         diffusion_config, sphere=True)
  tracers_in = {"const": jnp.ones_like(grid["physical_coords"][:, :, :, 0])}
  final_state = simulate_shallow_water(T, init_state, grid,
                                       physics_config, diffusion_config, timestep_config,
                                       dims, diffusion=False, tracers_in=tracers_in)
  dynamics = final_state["dynamics"]
  diff_u = u_init - dynamics["horizontal_wind"]
  diff_h = h_init - dynamics["h"]
  assert (inner_product(diff_u[:, :, :, 0], diff_u[:, :, :, 0], grid) < 1e-5)
  assert (inner_product(diff_u[:, :, :, 1], diff_u[:, :, :, 1], grid) < 1e-5)
  assert (inner_product(diff_h, diff_h, grid) / jnp.max(h_init) < 1e-5)
  if DEBUG:
    fig_dir = get_figdir()
    makedirs(fig_dir, exist_ok=True)
    plt.figure()
    plt.title("U at time {t}")
    lon = device_unwrapper(grid["physical_coords"][:, :, :, 1])
    lat = device_unwrapper(grid["physical_coords"][:, :, :, 0])
    plt.tricontourf(lon.flatten(),
                    lat.flatten(),
                    device_unwrapper(dynamics["horizontal_wind"][:, :, :, 0].flatten()))
    plt.colorbar()
    plt.savefig(join(fig_dir, "U_final.pdf"))
    plt.figure()
    plt.title("V at time {t}")
    plt.tricontourf(lon.flatten(),
                    lat.flatten(),
                    device_unwrapper(dynamics["horizontal_wind"][:, :, :, 1].flatten()))
    plt.colorbar()
    plt.savefig(join(fig_dir, "V_final.pdf"))
    plt.figure()
    plt.title("h at time {t}")
    plt.tricontourf(lon.flatten(),
                    lat.flatten(),
                    device_unwrapper(dynamics["h"].flatten()))
    plt.colorbar()
    plt.savefig(join(fig_dir, "h_final.pdf"))
    # plt.figure()
    # plt.title("Q=1 at time {t}")
    # plt.tricontourf(lon.flatten(),
    #                 lat.flatten(),
    #                 device_unwrapper((tracers["const"]-tracers_in["const"]).flatten()))
    # plt.colorbar()
    # plt.savefig(join(fig_dir, "Q_const_final.pdf"))


def test_galewsky():
  npt = 4
  nx = 31
  grid, dims = init_stretched_grid_elem_local(nx, npt, axis_dilation=jnp.array([1.0, 1.5, 1.0]))

  physics_config = init_physics_config_shallow_water()
  test_config = init_galewsky_config(physics_config)

  dt = 300
  T = (144 * 3600) * 4
  u_init = device_wrapper(eval_galewsky_wind(grid["physical_coords"][:, :, :, 0],
                                             grid["physical_coords"][:, :, :, 1],
                                             test_config))
  h_init = device_wrapper(eval_galewsky_h(grid["physical_coords"][:, :, :, 0],
                                          grid["physical_coords"][:, :, :, 1],
                                          test_config))
  hs_init = device_wrapper(eval_galewsky_hs(grid["physical_coords"][:, :, :, 0],
                                            grid["physical_coords"][:, :, :, 1],
                                            test_config))
  init_state = wrap_model_state(u_init, h_init, hs_init)
  # diffusion_config = get_hypervis_config_const(nx, physics_config, nu_div_factor=1.0)
  tracers_in = {"const": jnp.ones_like(grid["physical_coords"][:, :, :, 0]),
                "cos": jnp.cos(grid["physical_coords"][:, :, :, 0])}
  diffusion_config = init_hypervis_config_tensor(grid, dims, physics_config)
  timestep_config = init_timestep_config(dt, grid, dims, physics_config,
                                         diffusion_config, sphere=True)
  final_struct = simulate_shallow_water(T, init_state, grid,
                                        physics_config, diffusion_config, timestep_config,
                                        dims, diffusion=True, tracers_in=tracers_in)
  final_state = final_struct["dynamics"]

  # mass_init = inner_product(h_init, h_init, grid)
  # mass_final = inner_product(final_state["h"], final_state["h"], grid)

  # assert (jnp.abs(mass_init - mass_final) / mass_final < 1e-6)
  # assert (not jnp.any(jnp.isnan(final_state["horizontal_wind"])))

  if DEBUG:
    fig_dir = get_figdir()
    makedirs(fig_dir, exist_ok=True)
    lon = device_unwrapper(grid["physical_coords"][:, :, :, 1])
    lat = device_unwrapper(grid["physical_coords"][:, :, :, 0])
    levels = np.arange(-10 + 1e-4, 101, 10)
    vort = project_scalar(horizontal_vorticity(final_state["horizontal_wind"],
                                               grid,
                                               a=physics_config["radius_earth"]),
                          grid,
                          dims)
    plt.figure()
    plt.title(f"U at time {T}s")
    plt.tricontourf(lon.flatten(), lat.flatten(),
                    device_unwrapper(final_state["horizontal_wind"][:, :, :, 0].flatten()), levels=levels)
    plt.colorbar()
    plt.savefig(join(fig_dir, "galewsky_U_final.pdf"))
    plt.figure()
    plt.title(f"V at time {T}s")
    plt.tricontourf(lon.flatten(), lat.flatten(),
                    device_unwrapper(final_state["horizontal_wind"][:, :, :, 1].flatten()))
    plt.colorbar()
    plt.savefig(join(fig_dir, "galewsky_V_final.pdf"))
    plt.figure()
    plt.title(f"h at time {T}s")
    plt.tricontourf(lon.flatten(), lat.flatten(),
                    device_unwrapper(final_state["h"].flatten()))
    plt.colorbar()
    plt.savefig(join(fig_dir, "galewsky_h_final.pdf"))
    plt.figure()
    # plt.title(f"Q at time {T}s")
    # plt.tricontourf(lon.flatten(), lat.flatten(),
    #                 device_unwrapper((tracers["const"]).flatten()))
    # plt.colorbar()
    # plt.savefig(join(fig_dir, "galewsky_q_final.pdf"))
    # plt.figure()
    # plt.title(f"Q_cos at time {T}s")
    # plt.tricontourf(lon.flatten(), lat.flatten(),
    #                 device_unwrapper((tracers["cos"]).flatten()))
    # plt.colorbar()
    # plt.savefig(join(fig_dir, "galewsky_cos_final.pdf"))
    plt.figure()
    plt.title(f"vorticity at time {T}s")
    plt.tricontourf(lon.flatten(), lat.flatten(),
                    device_unwrapper(vort.flatten()),
                    vmin=-0.0002, vmax=0.0002)
    plt.colorbar()
    plot_grid(grid, plt.gca())
    plt.savefig(join(fig_dir, "galewsky_vort_final.pdf"))
