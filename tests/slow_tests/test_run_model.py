from ..test_data.mass_coordinate_grids import cam30
import pytest
from ..context import get_figdir, plot_grid, emit_plots
from pyses._config import get_backend as _get_backend
from pyses.analytic_initialization.moist_baroclinic_wave import (init_baroclinic_wave_config,
                                                               perturbation_opts,
                                                               init_baroclinic_wave_state)
from pyses.analytic_initialization.nonhydrostatic_mountain_wave import (init_mountain_wave_config,
                                                                      init_mountain_wave_state,
                                                                      dcmip_surface_shear)
from pyses.dynamical_cores.run_dycore import init_simulator
from pyses.mesh_generation.element_local_metric import init_quasi_uniform_grid_elem_local
from pyses.mesh_generation.element_local_metric import init_stretched_grid_elem_local
from pyses.dynamical_cores.mass_coordinate import init_vertical_grid
from pyses.dynamical_cores.model_info import (models, cam_se_models, homme_models,
                                              vertically_buoyant_models)
from pyses.dynamical_cores.physics_config import init_physics_config
from pyses.dynamical_cores.model_config import init_default_config, hypervis_opts
from pyses.mesh_generation.mesh_io import exodus_to_pyses_grid_corners
from pyses.mesh_generation.element_local_metric import init_unstructured_grid
import numpy as np
from os.path import join
from ..context import get_data_dir

_be = _get_backend()
jnp = _be.np
get_global_array = _be.get_global_array

# DCMIP-2012 small-planet reduction factor X for the non-hydrostatic
# mountain-wave tests (Table XIII); the scaled radius is 6371 km / X.
MOUNTAIN_WAVE_X = 500.0


def _plot_equatorial_meridional_cross_section(state, h_grid, physics_config, dims,
                                              savedir, lat_eps=np.deg2rad(1.0)):
  """Longitude-height cross-section of meridional velocity near the equator.

  The cubed-sphere grid is unstructured, so there is no structured lon-height
  slice.  Instead we pick every column whose latitude lies within ``+/- lat_eps``
  of the equator, derive height from the geopotential (``z = phi / g`` for the
  shallow-atmosphere cores used here), co-located with the mid-level winds, and
  ``tricontourf`` the resulting scattered ``(lon, z, v)`` points.  Requires the
  interface geopotential ``phi_i`` (a prognostic field only for the
  non-hydrostatic cores).
  """
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  dynamics = state["dynamics"]
  gravity = float(_be.unwrap(physics_config["gravity"]))

  coords = get_global_array(h_grid["physical_coords"], dims)      # (nelem, npt, npt, 2)
  lat = coords[..., 0].reshape(-1)                               # (ncol,)
  lon = coords[..., 1].reshape(-1)                               # (ncol,)

  # Meridional wind (component 1 of horizontal_wind) at mid-levels.
  v = get_global_array(dynamics["horizontal_wind"][..., 1], dims)  # (nelem, npt, npt, nlev)
  nlev = v.shape[-1]
  v = v.reshape(-1, nlev)                                          # (ncol, nlev)

  # Height from the interface geopotential, averaged onto the mid-levels the
  # wind lives on.
  phi_i = get_global_array(dynamics["phi_i"], dims)                # (nelem, npt, npt, nlev+1)
  phi_mid = 0.5 * (phi_i[..., 1:] + phi_i[..., :-1])
  z = (phi_mid / gravity).reshape(-1, nlev)                        # (ncol, nlev)

  band = np.abs(lat) < lat_eps
  assert band.any(), f"no columns within {np.rad2deg(lat_eps):.2f} deg of the equator"
  # Broadcast each selected column's longitude over its levels, then flatten to
  # scattered (x=lon, y=z, c=v) points for the unstructured contour.
  x = np.repeat(np.rad2deg(lon[band])[:, np.newaxis], nlev, axis=1).ravel()
  y = (z[band] / 1e3).ravel()
  c = v[band].ravel()

  plt.figure(figsize=(7.0, 4.0))
  tcf = plt.tricontourf(x, y, c, levels=21, cmap="RdBu_r")
  plt.colorbar(tcf, label="meridional velocity (m s$^{-1}$)")
  plt.xlabel("longitude (deg)")
  plt.ylabel("height (km)")
  plt.title(f"equatorial meridional velocity ($|$lat$| < {np.rad2deg(lat_eps):.1f}^\\circ$, "
            f"{int(band.sum())} columns)")
  plt.tight_layout()
  plt.savefig(f"{savedir}/v_equator_cross_section.pdf")
  plt.close()


def _run_mountain_wave(model, shear, nx=15, nsteps=30, subdir="mountain_wave"):
  """Initialise and integrate the DCMIP non-hydrostatic Schar mountain-wave test.

  Runs the balanced background jet over a Schar-type ridge on a non-rotating,
  reduced-size planet (``radius_earth = 6371 km / X``, ``angular_freq_earth =
  0``), with the coupling timestep scaled by ``1 / X`` so the CFL-derived
  dynamics subcycling matches a full-planet run.  ``nsteps`` coupling steps are
  taken; the simulator asserts finiteness after every step, so completing the
  loop is the pass condition.  When plotting is enabled (``emit_plots()``) the
  surface pressure and a mid-level snapshot are written periodically.

  The geopotential is re-balanced hydrostatically at initialisation for the
  vertically buoyant (non-hydrostatic) cores; the flag is inert for the
  hydrostatic cores.
  """
  npt = 4
  enforce_hydrostatic = model in vertically_buoyant_models
  h_grid, dims = init_quasi_uniform_grid_elem_local(nx, npt, calc_smooth_tensor=True)
  v_grid = init_vertical_grid(cam30["hybrid_a_i"],
                              cam30["hybrid_b_i"],
                              cam30["p0"],
                              model)
  physics_config = init_physics_config(model,
                                       radius_earth=6371e3 / MOUNTAIN_WAVE_X,
                                       angular_freq_earth=0.0)
  physics_dt = (900.0 * 30.0 / nx) / MOUNTAIN_WAVE_X
  physics_config, diffusion_config, timestep_config = init_default_config(
      nx, h_grid, v_grid, dims, model, physics_dt=physics_dt,
      hypervis_type=hypervis_opts.variable_resolution, physics_config=physics_config)
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
  savedir = get_figdir(subdir=f"{subdir}__{model}") if emit_plots() else None
  ct = 0
  for t, state in simulator(model_state):
    print(t)
    ct += 1
    if ct >= nsteps:
      break
    if savedir is None or ct % 10 != 0:
      continue

    import matplotlib.pyplot as plt
    end_state = state["dynamics"]
    ps = (v_grid["hybrid_a_i"][0] * v_grid["reference_surface_mass"] +
          jnp.sum(end_state["d_mass"], axis=-1))
    if model in cam_se_models:
      thermo = end_state["T"][:, :, :, 12]
    else:
      thermo = end_state["theta_v_d_mass"][:, :, :, 12] / end_state["d_mass"][:, :, :, 12]
    lat = get_global_array(h_grid["physical_coords"][:, :, :, 0], dims).flatten()
    lon = get_global_array(h_grid["physical_coords"][:, :, :, 1], dims).flatten()
    plt.figure()
    plt.tricontourf(lon, lat, get_global_array(ps, dims).flatten())
    plt.colorbar()
    plot_grid(h_grid, plt.gca())
    plt.savefig(f"{savedir}/ps_mtn_wave.pdf")
    plt.close()
    plt.figure()
    plt.tricontourf(lon, lat, get_global_array(thermo, dims).flatten())
    plt.colorbar()
    plot_grid(h_grid, plt.gca())
    plt.savefig(f"{savedir}/thermo_mtn_wave.pdf")
    plt.close()

  # End-of-simulation longitude-height cross-section of meridional velocity at
  # the equator (non-hydrostatic cores only, since it derives height from the
  # prognostic geopotential ``phi_i``).
  if savedir is not None and "phi_i" in state["dynamics"]:
    _plot_equatorial_meridional_cross_section(state, h_grid, physics_config, dims, savedir)


@pytest.mark.parametrize("model", [models.homme_hydrostatic,
                                   models.homme_nonhydrostatic])
def test_mountain_wave_non_sheared(model):
  # DCMIP test 2-1: vertically uniform background jet (shear parameter c = 0).
  _run_mountain_wave(model, shear=0.0, subdir="mtn_wave_non_sheared")


@pytest.mark.parametrize("model", [models.homme_hydrostatic,
                                   models.homme_nonhydrostatic])
def test_mountain_wave_sheared(model):
  # DCMIP test 2-2: vertically sheared background jet (c = cs).
  _run_mountain_wave(model, shear=dcmip_surface_shear, subdir="mtn_wave_sheared")


def test_mountain_wave_acid():
  # "Acid" stress test: the sheared jet driven over the Schar ridge on the fully
  # non-hydrostatic reduced planet -- the DCMIP-faithful configuration that
  # discriminates most sharply between hydrostatic and non-hydrostatic
  # responses.  Integrated longer than the fast smoke test to build the wave.
  _run_mountain_wave(models.homme_nonhydrostatic, shear=dcmip_surface_shear,
                     nsteps=60, subdir="mtn_wave_acid")


def test_theta_steady_state():
  for model in [models.homme_hydrostatic, models.cam_se]:
    npt = 4
    nx = 15
    arr = np.load(join(get_data_dir(), "conus.npz"))
    cart_coords = arr["cart_coords"]
    connect_map = arr["connect_map"]
    element_permuation = arr["element_permutation"]
    vert_pos, face_connectivity = exodus_to_pyses_grid_corners(cart_coords, connect_map, element_permuation)
    h_grid, dims = init_unstructured_grid(face_connectivity, vert_pos, npt)
    # h_grid, dims = init_stretched_grid_elem_local(nx,
    #                                               npt,
    #                                               axis_dilation=jnp.array([1.0, 1.5, 1.0]),
    #                                               calc_smooth_tensor=True)
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

    savedir = get_figdir(subdir="run_model_steady_state") if emit_plots() else None
    t = 0.0
    for t, state in simulator(model_state):
      print(t)
      if t > total_time:
        break
      if savedir is None:
        continue

      import matplotlib.pyplot as plt
      end_state = state["dynamics"]
      ps = v_grid["hybrid_a_i"][0] * v_grid["reference_surface_mass"] + jnp.sum(end_state["d_mass"], axis=-1)
      ps_begin = v_grid["hybrid_a_i"][0] * v_grid["reference_surface_mass"] + jnp.sum(end_state["d_mass"], axis=-1)
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
      plt.savefig(f"{savedir}/final_state_steady_{model}.pdf")
      plt.close()
      plt.figure()
      plt.tricontourf(get_global_array(h_grid["physical_coords"][:, :, :, 1], dims).flatten(),
                      get_global_array(h_grid["physical_coords"][:, :, :, 0], dims).flatten(),
                      get_global_array(ps - ps_begin, dims).flatten())
      plt.colorbar()
      plt.savefig(f"{savedir}/ps_diff_steady_{model}.pdf")
      plt.close()
      plt.figure()
      plot_grid(h_grid, plt.gca())
      plt.tricontourf(get_global_array(h_grid["physical_coords"][:, :, :, 1], dims).flatten(),
                      get_global_array(h_grid["physical_coords"][:, :, :, 0], dims).flatten(),
                      get_global_array(end_state["horizontal_wind"][:, :, :, 12, 1], dims).flatten())
      plt.colorbar()
      plot_grid(h_grid, plt.gca())
      plt.savefig(f"{savedir}/v_end_steady_{model}.pdf")
      plt.close()
      plt.figure()
      plt.tricontourf(get_global_array(h_grid["physical_coords"][:, :, :, 1], dims).flatten(),
                      get_global_array(h_grid["physical_coords"][:, :, :, 0], dims).flatten(),
                      get_global_array(end_state["horizontal_wind"][:, :, :, 12, 0], dims).flatten())
      plt.colorbar()
      plot_grid(h_grid, plt.gca())
      plt.savefig(f"{savedir}/u_end_steady_{model}.pdf")
      plt.close()
      plt.figure()
      plt.tricontourf(get_global_array(h_grid["physical_coords"][:, :, :, 1], dims).flatten(),
                      get_global_array(h_grid["physical_coords"][:, :, :, 0], dims).flatten(),
                      get_global_array(thermo, dims).flatten())
      plt.colorbar()
      plot_grid(h_grid, plt.gca())
      plt.savefig(f"{savedir}/thermo_end_steady_{model}.pdf")
      plt.close()

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
  savedir = get_figdir(subdir=f"baro_wave__{model}") if emit_plots() else None
  t = 0.0
  ct = 0
  for t, state in simulator(model_state):
    print(t)
    if t > total_time:
      break
    if savedir is not None and ct % 5 == 0:
      import matplotlib.pyplot as plt
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
      plt.savefig(f"{savedir}/final_state_bw_topo.pdf")
      plt.close()
      plt.figure()
      plt.tricontourf(get_global_array(h_grid["physical_coords"][:, :, :, 1], dims).flatten(),
                      get_global_array(h_grid["physical_coords"][:, :, :, 0], dims).flatten(),
                      get_global_array(end_state["horizontal_wind"][:, :, :, 12, 1], dims).flatten())
      plt.colorbar()
      plot_grid(h_grid, plt.gca())
      plt.savefig(f"{savedir}/v_end_bw_topo.pdf")
      plt.close()
      plt.figure()
      plt.tricontourf(get_global_array(h_grid["physical_coords"][:, :, :, 1], dims).flatten(),
                      get_global_array(h_grid["physical_coords"][:, :, :, 0], dims).flatten(),
                      get_global_array(end_state["horizontal_wind"][:, :, :, 12, 0], dims).flatten())
      plt.colorbar()
      plot_grid(h_grid, plt.gca())
      plt.savefig(f"{savedir}/u_end_bw_topo.pdf")
      plt.close()
      plt.figure()
      plt.tricontourf(get_global_array(h_grid["physical_coords"][:, :, :, 1], dims).flatten(),
                      get_global_array(h_grid["physical_coords"][:, :, :, 0], dims).flatten(),
                      get_global_array(thermo, dims).flatten())
      plt.colorbar()
      plot_grid(h_grid, plt.gca())
      plt.savefig(f"{savedir}/theta_v_end_bw_topo.pdf")
      plt.close()
    ct += 1
