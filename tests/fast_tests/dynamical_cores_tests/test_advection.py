from pyses.dynamical_cores.initialization import init_model_shallow_water
from pyses.grids import init
from pyses.mesh_generation.periodic_plane import init_uniform_grid
from pyses.simulate import init_simulator
from pyses.model_utils import mass_coordinate, model_info, model_config
from pyses.dynamical_cores.shallow_water_3d.thermodynamics import eval_geopotential
from ...context import get_figdir
from pyses._config import _reset_backend
import numpy as np
_reset_backend()
from pyses._config import get_backend as _get_backend
_be = _get_backend()
jnp = _be.np


def test_theta_baro_wave_topo():
  import matplotlib.pyplot as plt
  hybrid_b_i = np.linspace(0, 1, 3)
  hybrid_a_i = np.zeros_like(hybrid_b_i)
  p0 = 0.0  # reference surface pressure (Pa)

  # Generate horizontal grid
  npt = 4   # spectral-element polynomial order
  nx  = 30  # elements per edge
  # Note: the way dimensions work is that h_grid["physical_coords"] are on the unit plane [-1, 1]^2
  # then differential operators are calculated on physical_coords["radius_earth"] * h_grid["physical_coords"].
  # This means that the periodic plane and sphere behave similarly in this regard.

  h0 = 1.0e4
  u0 = 1.0 * 30.0
  h_pert = 1.0 * 10.0
  half_width = 0.05 # plane units are [-1, 1]^2


  def shear_u(x,
              y,
              v_grid):
      return u0 * -x[:, :, :, jnp.newaxis] * (1 - v_grid["hybrid_a_i"][jnp.newaxis, jnp.newaxis, jnp.newaxis, :-1])

  def v(x,
        y,
        v_grid):
      return u0 * y[:, :, :, jnp.newaxis] * v_grid["hybrid_a_i"][jnp.newaxis, jnp.newaxis, jnp.newaxis, :-1]

  def h_hs(x,
        y):
      return h_pert * jnp.exp(-(jnp.sqrt(x**2 + y**2) / half_width)**2), h0 * jnp.ones_like(x)


  h_grid, dims = init_uniform_grid(nx, nx, npt, calc_smooth_tensor=True)
  y = h_grid["physical_coords"][:, :, :, 0]
  x = h_grid["physical_coords"][:, :, :, 1]

  model = model_info.models.shallow_water_f_plane
  v_grid = mass_coordinate.init_vertical_grid(hybrid_a_i, hybrid_b_i, p0, model)

  # Use good default configuration. This can be customized!
  physics_config, diffusion_config, timestep_config = model_config.init_default_config(
      nx, h_grid, v_grid, dims, model,
      hypervis_type=model_config.hypervis_opts.variable_resolution, physics_dt=200
  )
  physics_config["angular_freq_earth"] = 0.0
  diffusion_config["nu_top"] = 0.0  # disable top-of-atmosphere sponge
  diffusion_config["nu_tracer"] *= 1e-2
  # Generate initial conditions
  model_state = init_model_shallow_water(h_hs, shear_u, v, h_grid, v_grid, physics_config, dims, model)
  model_state["tracers"]["tracers"]["frederick"] = jnp.exp(-(jnp.sqrt(x**2 + y**2) / 0.1)**2)[:, :, :, jnp.newaxis] * jnp.ones_like(model_state["dynamics"]["d_mass"])

  # initialize generator-based simulator
  simulator = init_simulator(
      h_grid, v_grid,
      physics_config, diffusion_config, timestep_config,
      dims, model,
  )


  # Time loop.
  ct = 0
  total_time = 3600.0 * 24.0 * 6.0  # one day in seconds
  for t, state in simulator(model_state):
      # User can perform archiving, analysis, etc here.
      if ct % 10 == 0:
        total_height = eval_geopotential(state["dynamics"]["d_mass"], state["static_forcing"]["phi_surf"])[:, :, :, 0]
        plt.figure()
        plt.tricontourf(h_grid["physical_coords"][:, :, :, 1].flatten(),
                    h_grid["physical_coords"][:, :, :, 0].flatten(),
                    total_height.flatten())
        plt.colorbar()
        plt.title(f"t={t}")
        plt.savefig(f"{get_figdir()}/h_t={t}.pdf")
        plt.figure()
        plt.tricontourf(h_grid["physical_coords"][:, :, :, 1].flatten(),
                    h_grid["physical_coords"][:, :, :, 0].flatten(),
                    state["tracers"]["tracers"]["frederick"][:, :, :, 0].flatten())
        plt.colorbar()
        plt.title(f"t={t}")
        plt.savefig(f"{get_figdir()}/fred_t={t}.pdf")
        print(t)
      ct += 1
      if t > total_time:
          break