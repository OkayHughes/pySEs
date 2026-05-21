from pyses.dynamical_cores.initialization import init_model_shallow_water
from pyses.grids import init
from pyses.mesh_generation.periodic_plane import init_uniform_grid
from pyses.dynamical_cores.model_state import project_dynamics
from pyses.operations_2d.operators import horizontal_gradient
from pyses.simulate import init_simulator
from pyses.model_utils import mass_coordinate, model_info, model_config
from pyses.dynamical_cores.shallow_water_3d.thermodynamics import eval_geopotential
from pyses.dynamical_cores.physics_dynamics_coupling import coupling_types
from pyses.dynamical_cores.time_step import time_step_options
from ...context import get_figdir
from pyses._config import _reset_backend
import numpy as np
from frozendict import frozendict
from PIL import Image
from scipy.interpolate import RegularGridInterpolator
from pyses._config import get_backend as _get_backend
_be = _get_backend()
jnp = _be.np


RGB_SCALE = 255
CMYK_SCALE = 1.0


def rgb_to_cmyk(r, g, b):
    max_norm = np.max(np.stack((r, g, b), axis=0), axis=0)
    mask = max_norm > 1e-8

    # rgb [0,255] -> cmy [0,1]
    c = 1 - r / RGB_SCALE
    m = 1 - g / RGB_SCALE
    y = 1 - b / RGB_SCALE

    # extract out k [0, 1]
    min_cmy = np.min(np.stack((c, m, y), axis=0), axis=0)
    c = (c - min_cmy) / (1 - min_cmy)
    m = (m - min_cmy) / (1 - min_cmy)
    y = (y - min_cmy) / (1 - min_cmy)
    k = min_cmy

    # rescale to the range [0,CMYK_SCALE]
    filt = lambda field: np.where(mask, field, 0.0)
    return filt(c), filt(m), filt(y), np.where(mask, k, 1.0)

def cmyk_to_rgb(c, m, y, k):
    r = RGB_SCALE * (1.0 - c ) * (1.0 - k )
    g = RGB_SCALE * (1.0 - m ) * (1.0 - k )
    b = RGB_SCALE * (1.0 - y ) * (1.0 - k )
    return r, g, b


def interp_field_bilinear(in_field, h_grid):
  x = np.linspace(-1, 1, in_field.shape[0])
  y = np.linspace(-1, 1, in_field.shape[1])
  interpolator = RegularGridInterpolator((x, y), in_field)
  x_eval = h_grid["physical_coords"][:, :, :, 0].flatten()
  y_eval = h_grid["physical_coords"][:, :, :, 1].flatten()
  return interpolator((x_eval, y_eval)).reshape(h_grid["physical_coords"][:, :, :, 0].shape)


def test_theta_baro_wave_topo():
  import matplotlib.pyplot as plt
  hybrid_b_i = np.linspace(0, 1, 2)
  hybrid_a_i = np.zeros_like(hybrid_b_i)
  p0 = 0.0  # reference surface pressure (Pa)

  # Generate horizontal grid
  npt = 4   # spectral-element polynomial order
  nx  = 240  # elements per edge
  # Note: the way dimensions work is that h_grid["physical_coords"] are on the unit plane [-1, 1]^2
  # then differential operators are calculated on physical_coords["radius_earth"] * h_grid["physical_coords"].
  # This means that the periodic plane and sphere behave similarly in this regard.
  I = np.asarray(Image.open('/Users/OstensiblyOwen/files/Images/fish.jpeg'))
  I_vect = np.asarray(Image.open('/Users/OstensiblyOwen/files/Images/manali.jpg'))
  field_norm = np.linalg.norm(I_vect, axis=-1)
  cyan, magenta, yellow, black = rgb_to_cmyk(I[:, :, 0], I[:, :, 1], I[:, :, 2])

  h0 = 1.0e4
  u0 = 1.0 * 80.0
  total_dt = 60.0
  timestep_config = frozendict(dynamics=frozendict(step_type=time_step_options.RK3_5STAGE,
                                                   dt=0.0),
                               hyperviscosity=frozendict(step_type=time_step_options.Euler,
                                                         dt=0.0),
                               tracer_advection=frozendict(step_type=time_step_options.RK2,
                                                           dt=total_dt),
                               dynamics_subcycle=1,
                               tracer_subcycle=1,
                               hypervis_subcycle=1,
                               physics_dt=total_dt,
                               physics_dynamics_coupling=coupling_types.none)

  scale = 20
  def shear_u(x,
              y,
              v_grid):
      return (u0 * -jnp.pi * jnp.sin(scale * jnp.pi / 2.0 * x) * jnp.cos(scale * jnp.pi / 2.0 * x) * jnp.cos(scale * jnp.pi / 2.0 * y)**2)[:, :, :, jnp.newaxis]

  def v(x,
        y,
        v_grid):
      return (u0 * jnp.pi * jnp.sin(scale * jnp.pi / 2.0 * y) * jnp.cos(scale * jnp.pi / 2.0 * y) * jnp.cos(scale * jnp.pi / 2.0 * x)**2)[:, :, :, jnp.newaxis]

  def hs_h(x,
        y):
      return 0.0 * jnp.ones_like(x), h0 * jnp.ones_like(x)

  h_grid, dims = init_uniform_grid(nx, nx, npt, calc_smooth_tensor=True)
  y = h_grid["physical_coords"][:, :, :, 0]
  x = h_grid["physical_coords"][:, :, :, 1]

  model = model_info.models.shallow_water_f_plane
  v_grid = mass_coordinate.init_vertical_grid(hybrid_a_i, hybrid_b_i, p0, model)

  # Use good default configuration. This can be customized!
  physics_config, diffusion_config, _timestep_config = model_config.init_default_config(
      nx, h_grid, v_grid, dims, model,
      hypervis_type=model_config.hypervis_opts.none, physics_dt=200
  )
  physics_config["angular_freq_earth"] = 0.0
  physics_config["radius_earth"] = 6371e3

  #diffusion_config["nu_top"] = 0.0  # disable top-of-atmosphere sponge
  #diffusion_config["nu_tracer"] *= 0.0
  # Generate initial conditions
  model_state = init_model_shallow_water(hs_h, shear_u, v, h_grid, v_grid, physics_config, dims, model)
  hamiltonian = interp_field_bilinear(field_norm, h_grid)
  grad_ham = horizontal_gradient(hamiltonian, h_grid, a=1.0)
  max_wind = jnp.max(jnp.abs(grad_ham))
  norm_const = 80.0 / max_wind
  horizontal_wind = jnp.stack((grad_ham[:, :, :, 1],
                               -grad_ham[:, :, :, 0]), axis=-1) * jnp.ones_like(model_state["dynamics"]["d_mass"])
  model_state["dynamics"]["horizontal_wind"] = horizontal_wind
  model_state["dynamics"] = project_dynamics(model_state["dynamics"], h_grid, dims, model)
  x_offset = 0.1
  y_offset = 0.1
  for key, field in zip(["cyan", "yellow", "magenta", "black"],
                        [cyan, yellow, magenta, black]):
    model_state["tracers"]["tracers"][key] = interp_field_bilinear(field, h_grid)[:, :, :, jnp.newaxis] * jnp.ones_like(model_state["dynamics"]["d_mass"])
    print()
  
  #model_state["tracers"]["tracers"]["frederick"] = jnp.exp(-(jnp.sqrt((x-x_offset)**2 + (y-y_offset)**2) / 0.1)**2)[:, :, :, jnp.newaxis] * jnp.ones_like(model_state["dynamics"]["d_mass"])
  #model_state["tracers"]["tracers"]["frederick"] = jnp.ones_like(model_state["dynamics"]["d_mass"])


  # initialize generator-based simulator
  simulator = init_simulator(
      h_grid, v_grid,
      physics_config, diffusion_config, timestep_config,
      dims, model,
  )

  # Time loop.
  ct = 0
  total_time = 3600.0 * 24.0 * 0.25  # one day in seconds
  for t, state in simulator(model_state):
      print(t)
      # User can perform archiving, analysis, etc here.
      if ct % 10 == 0:
        red, green, blue = cmyk_to_rgb(state["tracers"]["tracers"]["cyan"],
                                       state["tracers"]["tracers"]["magenta"],
                                       state["tracers"]["tracers"]["yellow"],
                                       state["tracers"]["tracers"]["black"])
        rgb = np.stack((red.flatten(), green.flatten(), blue.flatten()), axis=-1)/255.0
        for field in ["cyan", "magenta", "yellow", "black"]:
          print(f"{field}: {jnp.max(state["tracers"]["tracers"][field])} {jnp.min(state["tracers"]["tracers"][field])}")
        rgb = jnp.maximum(0.0, jnp.minimum(1.0, rgb))
        plt.figure()
        plt.scatter(h_grid["physical_coords"][:, :, :, 1].flatten(),
                    -h_grid["physical_coords"][:, :, :, 0].flatten(),
                    c=rgb, s=0.5)       
        plt.title(f"t={t}")
        plt.gca().set_aspect("equal")
        plt.savefig(f"{get_figdir()}/img_t={str(int(t)).zfill(10)}.png")
        print(t)
      ct += 1
      if t > total_time:
          break
  state["dynamics"]["horizontal_wind"] = -state["dynamics"]["horizontal_wind"]
  for t, state in simulator(state):
      print(t)
      # User can perform archiving, analysis, etc here.
      if ct % 10 == 0:
        red, green, blue = cmyk_to_rgb(state["tracers"]["tracers"]["cyan"],
                                       state["tracers"]["tracers"]["magenta"],
                                       state["tracers"]["tracers"]["yellow"],
                                       state["tracers"]["tracers"]["black"])
        rgb = np.stack((red.flatten(), green.flatten(), blue.flatten()), axis=-1)/255.0
        for field in ["cyan", "magenta", "yellow", "black"]:
          print(f"{field}: {jnp.max(state["tracers"]["tracers"][field])} {jnp.min(state["tracers"]["tracers"][field])}")
        rgb = jnp.maximum(0.0, jnp.minimum(1.0, rgb))
        plt.figure()
        plt.scatter(h_grid["physical_coords"][:, :, :, 1].flatten(),
                    -h_grid["physical_coords"][:, :, :, 0].flatten(),
                    c=rgb, s=0.5)       
        plt.title(f"t={total_time + t}")
        plt.gca().set_aspect("equal")
        plt.savefig(f"{get_figdir()}/img_t={str(int(total_time + t)).zfill(10)}.png")
        print(t)
      ct += 1
      if t > total_time:
          break