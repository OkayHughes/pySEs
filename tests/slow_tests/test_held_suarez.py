from ..test_data.mass_coordinate_grids import cam30
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
from pyses.dynamical_cores.mass_coordinate import d_mass_to_surface_mass, surface_mass_to_midlevel_mass
from pyses.dynamical_cores.model_state import wrap_model_state, wrap_dynamics, wrap_tracers
from pyses.operations_2d.operators import inner_product

_be = _get_backend()
jnp = _be.np
get_global_array = _be.get_global_array


sigma_b = 0.70
secpday = 86400
k_a = 1.0 / (40.0 * secpday)
k_f = 1.0 / (1.0 * secpday)
k_s = 1.0 / (4.0 * secpday)
dtheta_z = 10.0
dT_y = 60.0


# The temperature forcing for the Held-Suarez forcing,
# which is the simplest forcing that produces an earth-ish
# zonal wind climatology
# https://doi.org/10.1175/1520-0477(1994)075<1825:APFTIO>2.0.CO;2
# the temperature forcing is Newtonian relaxation to a zonally-symmetric
# earth-like reference profile
def hs_temperature(lat, lon, pi, T, v_grid, config):
  logprat = jnp.log(pi) - jnp.log(v_grid["reference_surface_mass"])
  etam = v_grid["hybrid_a_m"] + v_grid["hybrid_b_m"]
  pratk = jnp.exp(config["Rgas"] / config["cp"] * (logprat))
  k_t = (k_a + (k_s - k_a) * (jnp.cos(lat)**2 * jnp.cos(lat)**2)[:, :, :, jnp.newaxis] *
         jnp.maximum(0.0, ((etam - sigma_b) / (1.0 - sigma_b))[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]))
  Teq = jnp.maximum(200.0, (315.0 - dT_y * jnp.sin(lat)[:, :, :, jnp.newaxis]**2 -
                            dtheta_z * logprat * jnp.cos(lat)[:, :, :, jnp.newaxis]**2) * pratk)
  hs_T_frc = -k_t * (T - Teq)
  return hs_T_frc, Teq

# In the held-suarez forcing, energy cascade is modeled
# by Newtonian damping of velocity in the lower atmosphere
def hs_u(u, v_grid):
  etam = v_grid["hybrid_a_m"] + v_grid["hybrid_b_m"]
  k_v = k_f * jnp.maximum(0.0, (etam - sigma_b) / (1.0 - sigma_b))
  hs_v_frc = jnp.stack((-k_v[jnp.newaxis, jnp.newaxis, jnp.newaxis, :] * u[:, :, :, :, 0],
                        -k_v[jnp.newaxis, jnp.newaxis, jnp.newaxis, :] * u[:, :, :, :, 1]),
                       axis=-1)
  return hs_v_frc

# Assemble forcing into a dynamics struct so it
# can be added to the model state.
# Note that the thermodynamic variable is different
# in HOMME and CAM-SE
def hs_forcing(state_in, h_grid, v_grid, physics_config, model):
  # Note: we call the vertical coordinate a mass coordinate
  # not a pressure coordinate, because confusing mass for pressure
  # causes subtle errors that I personally lost 4 months of my life to.
  p_ish_surf = d_mass_to_surface_mass(state_in["dynamics"]["d_mass"], v_grid)
  p_ish_mid = surface_mass_to_midlevel_mass(p_ish_surf, v_grid)
  exner = (p_ish_mid / physics_config["p0"])**(physics_config["Rgas"]/physics_config["cp"])
  if model in homme_models:
    temperature = state_in["dynamics"]["theta_v_d_mass"] / state_in["dynamics"]["d_mass"] * exner
  elif model in cam_se_models:
    temperature = state_in["dynamics"]["T"]

  temperature_tend = hs_temperature(h_grid["physical_coords"][:, :, :, 0],
                                    h_grid["physical_coords"][:, :, :, 1],
                                    p_ish_mid,
                                    temperature,
                                    v_grid,
                                    physics_config)[0]

  if model in homme_models:
    thermo_tend = (temperature_tend / exner) * state_in["dynamics"]["d_mass"]
  elif model in cam_se_models:
    thermo_tend = temperature_tend
  moisture_tend = {}
  u_frc = hs_u(state_in["dynamics"]["horizontal_wind"], v_grid)
  for species in state_in["tracers"]["moisture_species"].keys():
    moisture_tend[species] = 0.0 * state_in["tracers"]["moisture_species"][species]
  # Dry air species are only treated as special tracers in CAM-SE
  if model in cam_se_models:
    dry_air_tend = {}
    for species in state_in["tracers"]["dry_air_species"].keys():
      dry_air_tend[species] = 0.0 * state_in["tracers"]["dry_air_species"][species]
  else:
    dry_air_tend = None
  # wrap_model_state, wrap_dynamics, and wrap_tracers are
  # how you turn numpy/jax arrays into a cross-dycore struct

  # ========================================================
  # NOTE: as of v0.0.1, tracer advection is not implemented.
  # This forcing is ignored
  # ========================================================
  return wrap_model_state(wrap_dynamics(1.0 * u_frc,
                                        1.0 * thermo_tend,
                                        0.0 * state_in["dynamics"]["d_mass"],
                                        model),
                          state_in["static_forcing"],
                          wrap_tracers(moisture_tend,
                                       {},
                                       model,
                                       dry_air_species=dry_air_tend))

def test_theta_held_suarez():
  npt = 4
  nx = 9
  h_grid, dims = init_stretched_grid_elem_local(nx,
                                                npt,
                                                offset=2.0 * jnp.array([.1, .1, .2]),
                                                calc_smooth_tensor=True)
  model = models.homme_hydrostatic
  v_grid = init_vertical_grid(cam30["hybrid_a_i"],
                              cam30["hybrid_b_i"],
                              cam30["p0"],
                              model)

  total_time = (3600.0 * 24.0 * 100.0)
  hv_type = hypervis_opts.variable_resolution
  physics_config, diffusion_config, timestep_config = init_default_config(nx, h_grid, v_grid, dims, model,
                                                                          hypervis_type=hv_type)
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
  generator = simulator(model_state)
  def calc_total_mass(state):
    total_mass = 0.0
    for lev_idx in range(state["dynamics"]["d_mass"].shape[-1]):
      sqrt_d_mass = jnp.sqrt(state["dynamics"]["d_mass"][:, :, :, lev_idx])
      total_mass += inner_product(sqrt_d_mass, sqrt_d_mass, h_grid)
    return total_mass

  initial_mass = calc_total_mass(model_state)
  t = 0.0
  ct = 0
  t, state = next(generator)
  while True:
    forcing = hs_forcing(state, h_grid, v_grid, physics_config, model)
    t, state = generator.send(forcing)
    print(f"time: {t / 86400} days")
    if ct % 10 == 0:
      ps = d_mass_to_surface_mass(state["dynamics"]["d_mass"], v_grid)
      import matplotlib.pyplot as plt
      total_mass = calc_total_mass(state)
      mass_error = total_mass-initial_mass
      print(f"Global mass change since beginning of integration: {mass_error}, relative error: {mass_error / initial_mass}")
      figdir = get_figdir()
      plt.figure()
      plt.tricontourf(get_global_array(h_grid["physical_coords"][:, :, :, 1], dims).flatten(),
                      get_global_array(h_grid["physical_coords"][:, :, :, 0], dims).flatten(),
                      get_global_array(ps, dims).flatten())
      plot_grid(h_grid, plt.gca())
      plt.colorbar()
      plt.savefig(f"{figdir}/ps_hs_topo.pdf")
      plt.figure()
      plt.tricontourf(get_global_array(h_grid["physical_coords"][:, :, :, 1], dims).flatten(),
                      get_global_array(h_grid["physical_coords"][:, :, :, 0], dims).flatten(),
                      get_global_array(state["dynamics"]["horizontal_wind"][:, :, :, 12, 1], dims).flatten())
      plt.colorbar()
      plot_grid(h_grid, plt.gca())
      plt.savefig(f"{figdir}/v_end_hs_topo.pdf")
      plt.figure()
      plt.tricontourf(get_global_array(h_grid["physical_coords"][:, :, :, 1], dims).flatten(),
                      get_global_array(h_grid["physical_coords"][:, :, :, 0], dims).flatten(),
                      get_global_array(state["dynamics"]["horizontal_wind"][:, :, :, 12, 0], dims).flatten())
      plt.colorbar()
      plot_grid(h_grid, plt.gca())
      plt.savefig(f"{figdir}/u_end_hs_topo.pdf")
    if t > total_time:
      break
    ct += 1
