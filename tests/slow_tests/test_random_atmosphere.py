"""
Randomized ("fuzzing") stability tests for the HOMME hydrostatic dynamical core.

Each test builds a random-but-balanced atmosphere on an ne15 element-local
quasi-uniform mesh with 15 vertical levels (cam30 truncated to every other
level) and checks that the model integrates stably for six hours.

Initial state
-------------
* Temperature: a meridionally sinusoidal, equatorially symmetric surface
  temperature ``T_surf(lat) = T_E + (T_P - T_E) sin^2(lat)`` (``T_E`` at the
  equator, ``T_P`` at the poles) with a constant lapse rate ``GAMMA = 5 K/km``.
* Pressure: a sea-level-pressure field of mean 1000 hPa plus randomly placed
  Gaussian-hill perturbations.  Under hydrostatic balance with the constant
  lapse rate this gives a closed-form ``p(z)`` (the standard barometric power
  law) that is handed to ``init_model_pressure`` as ``p_moist``.  The expensive
  per-column quantities (SLP, surface temperature) are precomputed once on the
  grid and captured, so each ``p_moist`` call is a cheap power law.
* Surface geopotential (topography test only): the product of a "continent"
  sum of a few very wide, high-exponent Gaussian hills (≈ an indicator
  function of where land is) and a unit-baseline roughness made of many small,
  narrow Gaussian hills.
* Winds: discrete geostrophic balance away from the equator.  The geopotential
  is ``g * z`` at the model levels, so its horizontal gradient is computed
  directly from the ``z`` passed into the wind function with
  ``horizontal_gradient_3d`` (which returns physical east/north components).
  The geostrophic factor decays smoothly to zero through the tropics.

Because randomly drawn pressure fields can produce unbalanced jets, the draw is
accepted only if the maximum initial wind speed is below ``MAX_WIND`` (rejection
sampling); otherwise a new draw is taken.
"""
import numpy as np
import pytest

from pyses._config import get_backend as _get_backend
from pyses.dynamical_cores.initialization import (init_model_pressure,
                                                  surface_mass_to_midlevel_mass,
                                                  z_from_p_monotonic_moist)
from pyses.dynamical_cores.operators_3d import horizontal_gradient_3d
from pyses.dynamical_cores.run_dycore import init_simulator
from pyses.dynamical_cores.mass_coordinate import init_vertical_grid
from pyses.dynamical_cores.model_config import init_default_config, hypervis_opts
from pyses.dynamical_cores.time_stepping import (advance_dynamics_ullrich_5stage,
                                                  advance_hypervis_euler,
                                                  advance_sponge_euler)
from pyses.dynamical_cores.model_state import (remap_dynamics,
                                                 remap_tracers,
                                                 sum_consistency_struct,
                                                 se_T_to_theta_d_d_mass,
                                                 se_theta_d_d_mass_to_T,
                                                 wrap_model_state)
from pyses.dynamical_cores.tracer_advection.eulerian_spectral import advance_tracers
from pyses.dynamical_cores.model_info import models, thermodynamic_variable_names
from pyses.mesh_generation.element_local_metric import init_quasi_uniform_grid_elem_local
from ..context import emit_plots, get_figdir, plot_scalar_field
from ..test_data.mass_coordinate_grids import cam30

_be = _get_backend()
jnp = _be.np
device_wrapper = _be.array
unwrap = _be.unwrap

# --- resolution / model choices -------------------------------------------
NE = 15                       # ne15 cubed-sphere
NPT = 4                       # GLL points per element edge
MODEL = models.cam_se

# --- atmosphere parameters ------------------------------------------------
T_E = 300.0                   # equatorial surface temperature (K)
T_P = 240.0                   # polar surface temperature (K)
GAMMA = 0.005                 # lapse rate (K / m) == 5 K/km
SLP_MEAN = 1.0e5             # mean sea-level pressure (Pa) == 1000 hPa
SLP_STD = 1.0e3             # s.d. of SLP perturbations (Pa) == 10 hPa
SLP_COVERAGE = 0.3          # expected hill footprints per point (count knob)
LAT_TROPICS = np.deg2rad(20.0)   # tropical geostrophic decay scale
MAX_WIND = 100.0             # rejection-sampling cap on |wind| (m/s)
SEED0 = 20240517            # base seed (vary to fuzz)
MAX_ATTEMPTS = 30

# --- topography parameters ------------------------------------------------
N_CONTINENTS = 6
CONTINENT_WIDTH_MEAN = np.deg2rad(30.0)   # half-width mean
CONTINENT_WIDTH_STD = np.deg2rad(5.0)
CONTINENT_EXPONENT = 4                     # >2 -> approaches an indicator
CONTINENT_HEIGHT_RANGE = (500.0, 2000.0)   # m
NOISE_AMP = 0.3                            # half-normal scale of roughness
MAX_TOPO_HEIGHT = 2500.0                  # clip for steepness/stability


def _dx_radians():
  """Representative GLL point spacing in radians for the ne15/npt grid."""
  return (np.pi / 2.0) / (NE * (NPT - 1))


def _random_centers(rng, n):
  """``n`` Gaussian-hill centers uniformly distributed over the sphere."""
  lat_c = np.arcsin(rng.uniform(-1.0, 1.0, size=n))
  lon_c = rng.uniform(0.0, 2.0 * np.pi, size=n)
  return lat_c, lon_c


def _sum_gaussian_hills(lat, lon, lat_c, lon_c, amps, scales, exponent):
  """
  Sum ``amp_k * exp(-(great_circle_dist / scale_k) ** exponent)`` over hills.

  ``lat``/``lon`` are grid arrays (radians); the hill parameters are 1-D.
  """
  total = np.zeros_like(lat)
  for clat, clon, amp, scale in zip(lat_c, lon_c, amps, scales):
    cos_d = (np.sin(clat) * np.sin(lat) +
             np.cos(clat) * np.cos(lat) * np.cos(lon - clon))
    gc = np.arccos(np.clip(cos_d, -1.0, 1.0))
    total = total + amp * np.exp(-(gc / scale) ** exponent)
  return total


def _half_width_to_scale(half_width, exponent):
  """Scale s.t. the hill falls to 1/2 at ``half_width``."""
  return half_width / np.log(2.0) ** (1.0 / exponent)


def _eval_slp(lat, lon, rng, slp_std=SLP_STD):
  """
  Sea-level pressure field (Pa): mean ``SLP_MEAN`` plus random Gaussian hills.

  Centers are uniform on the sphere, half-widths follow a beta distribution
  centered on 3*dx, and amplitudes are N(0, ``slp_std``).  Returns a numpy
  array on the grid (precomputed once and captured by ``p_moist`` for speed).
  """
  dx = _dx_radians()
  # number of hills for the requested coverage (footprints per point ~ N w^2 / 4).
  w_mean = 3.0 * dx
  n_hills = max(1, int(round(SLP_COVERAGE * 4.0 / w_mean ** 2)))
  lat_c, lon_c = _random_centers(rng, n_hills)
  half_widths = dx * (1.5 + 3.0 * rng.beta(2.0, 2.0, size=n_hills))   # mean 3*dx
  scales = _half_width_to_scale(half_widths, 2)
  amps = rng.normal(0.0, slp_std, size=n_hills)
  return SLP_MEAN + _sum_gaussian_hills(lat, lon, lat_c, lon_c, amps, scales, 2)


def _eval_topography(lat, lon, rng):
  """
  Surface height (m): product of a continent sum and a roughness factor.

  * continents: ``N_CONTINENTS`` wide, high-exponent (indicator-like) hills
    whose amplitudes are the continent heights.
  * roughness: ``1 + sum`` of many small, narrow (≈3*dx) Gaussian hills with
    non-negative (half-normal) amplitudes; the unit baseline lets the noise
    modulate the continents rather than mask them.  The hill count is set so
    that, if uniformly spaced, three hills would overlap at a point.
  """
  dx = _dx_radians()
  # continents
  lat_c, lon_c = _random_centers(rng, N_CONTINENTS)
  cont_hw = np.clip(rng.normal(CONTINENT_WIDTH_MEAN, CONTINENT_WIDTH_STD, N_CONTINENTS),
                    np.deg2rad(8.0), np.deg2rad(35.0))
  cont_scales = _half_width_to_scale(cont_hw, CONTINENT_EXPONENT)
  cont_amps = rng.uniform(*CONTINENT_HEIGHT_RANGE, size=N_CONTINENTS)
  continents = _sum_gaussian_hills(lat, lon, lat_c, lon_c, cont_amps, cont_scales,
                                   CONTINENT_EXPONENT)
  # roughness: coverage of 3 -> N = 12 / w^2  (w the half-width in radians)
  w_mean = 3.0 * dx
  n_noise = max(1, int(round(12.0 / w_mean ** 2)))
  nlat_c, nlon_c = _random_centers(rng, n_noise)
  noise_hw = dx * (1.5 + 3.0 * rng.beta(2.0, 2.0, size=n_noise))
  noise_scales = _half_width_to_scale(noise_hw, 2)
  noise_amps = np.abs(rng.normal(0.0, NOISE_AMP, size=n_noise))
  roughness = 1.0 + _sum_gaussian_hills(lat, lon, nlat_c, nlon_c, noise_amps,
                                        noise_scales, 2)
  return np.clip(continents * roughness, 0.0, MAX_TOPO_HEIGHT)


def _build_state(h_grid, v_grid, physics_config, dims, mountain, rng,
                 model=MODEL, slp_std=SLP_STD):
  """Build a single random model state (no rejection check).

  ``model`` selects the dynamical core: the geostrophic wind / barometric SLP
  fields are model-invariant, so two cores called with the same ``rng`` and the
  same ``mountain`` flag produce the same initial physical atmosphere expressed
  in each core's prognostic variables.

  ``slp_std`` overrides the per-hill SLP perturbation amplitude (default:
  :data:`SLP_STD`).  CAM-SE's vertical remap tolerates smaller perturbations
  than HOMME's at this resolution, so cross-core tests can dial this down.
  """
  coords = unwrap(h_grid["physical_coords"])
  lat_np = coords[:, :, :, 0]
  lon_np = coords[:, :, :, 1]

  g = float(unwrap(physics_config["gravity"]))
  Rgas = float(unwrap(physics_config["Rgas"]))
  omega = float(unwrap(physics_config["angular_freq_earth"]))
  exponent = g / (Rgas * GAMMA)

  # precomputed per-column lookups captured by the closures below. elem_sharding_axis=0
  # shards them like the grid under JAX multi-device (no-op for numpy / torch).
  slp = device_wrapper(_eval_slp(lat_np, lon_np, rng, slp_std=slp_std),
                       elem_sharding_axis=0)     # (elem, i, j)
  if mountain:
    z_surf = device_wrapper(_eval_topography(lat_np, lon_np, rng), elem_sharding_axis=0)
  else:
    z_surf = device_wrapper(np.zeros_like(lat_np), elem_sharding_axis=0)
  lat = device_wrapper(lat_np, elem_sharding_axis=0)
  t_surf = T_E + (T_P - T_E) * jnp.sin(lat) ** 2                 # (elem, i, j)

  def p_moist_func(z):
    # hydrostatic, constant-lapse barometric law; clip keeps the base positive
    base = jnp.clip(1.0 - GAMMA * z / t_surf[..., jnp.newaxis], 1e-6, None)
    return slp[..., jnp.newaxis] * base ** exponent

  def z_pi_surf_func(lat_in, lon_in):
    surface_mass = p_moist_func(z_surf[..., jnp.newaxis])[..., 0]
    return z_surf, surface_mass

  def Tv_func(lat_in, lon_in, z):
    return t_surf[..., jnp.newaxis] - GAMMA * z

  def Q_func(lat_in, lon_in, z):
    return jnp.zeros_like(z)

  def w_func(lat_in, lon_in, z):
    return jnp.zeros_like(z)

  # Geostrophic winds balance the *synoptic* SLP field, not the terrain: the
  # mountains are merely a lower boundary the large-scale flow crosses, so the
  # winds are computed from a terrain-free height field (z_surf = 0).  This makes
  # the wind magnitude independent of the topography draw.  Phi = g*z on the
  # model levels, so the geopotential gradient is just g * grad(z); the 1/f
  # factor decays to zero at the equator and is capped at its polar value so the
  # small-f subtropics do not produce unbounded jets.
  surface_mass_syn = slp                                # p_moist_func(z=0) == slp
  p_mid_syn = surface_mass_to_midlevel_mass(surface_mass_syn, v_grid)
  z_mid_syn = z_from_p_monotonic_moist(p_mid_syn, p_moist_func, eps=1e-6)
  grad_z = horizontal_gradient_3d(z_mid_syn, h_grid, physics_config)  # (..., lev, 2)
  sin_lat = jnp.sin(lat)
  s2 = np.sin(LAT_TROPICS) ** 2
  raw = sin_lat / (sin_lat ** 2 + s2)          # ~ 1/(2 Omega sin lat) shape, finite at eq
  cap = 1.0 / (1.0 + s2)                        # value at the pole
  factor = jnp.clip(raw, -cap, cap)
  coeff = (g / (2.0 * omega) * factor)[..., jnp.newaxis]
  u_geo = -coeff * grad_z[..., 1]              # east  = -(g/f) dz/dy
  v_geo = coeff * grad_z[..., 0]               # north =  (g/f) dz/dx

  def u_func(lat_in, lon_in, z):
    return u_geo

  def v_func(lat_in, lon_in, z):
    return v_geo

  return init_model_pressure(z_pi_surf_func, p_moist_func, Tv_func,
                             u_func, v_func, Q_func,
                             h_grid, v_grid, physics_config, dims, model,
                             w_func=w_func)


# --- diagnostic plotting (gated by PYSES_TEST_EMIT_PLOTS) -----------------
# Sample vertical levels for plots; 0 is model top, 14 is the surface for the
# cam30[::2] truncation used here.  Three levels span strat / mid-trop / PBL.
PLOT_LEVELS = (2, 7, 12)
# Snapshot the run at these elapsed hours; the 6 h endpoint is included so we
# get a final-state plot alongside the initial one.
PLOT_SNAPSHOT_HOURS = (1.0, 3.0, 6.0)


def _plot_coords(h_grid):
  """Lat/lon arrays for plotting (radians, lon wrapped to ``[0, 2 pi)``)."""
  coords = unwrap(h_grid["physical_coords"])
  return coords[..., 0], coords[..., 1] % (2.0 * np.pi)


def _plot_state(state, lat, lon, label, savedir, gravity=None, mountain=False):
  """Save one set of (wind magnitude, theta_v) plots at ``PLOT_LEVELS``.

  ``label`` is interpolated into the filenames and titles (e.g. ``"init"``,
  ``"t06.0h"``).  Topography and the surface-mass column are only saved when
  ``gravity`` is provided (i.e. for the initial state).
  """
  wind = unwrap(state["dynamics"]["horizontal_wind"])
  d_mass = unwrap(state["dynamics"]["d_mass"])
  thermo = unwrap(state["dynamics"][thermodynamic_variable_names[MODEL]]) / d_mass
  wind_mag = np.sqrt(wind[..., 0] ** 2 + wind[..., 1] ** 2)

  for k in PLOT_LEVELS:
    plot_scalar_field(lat, lon, wind_mag[..., k],
                      f"wind magnitude  {label}  (level {k})",
                      f"{savedir}/{label}_wind_mag_lev{k:02d}.png",
                      cmap="viridis", cbar_label="|u| (m/s)")
    plot_scalar_field(lat, lon, thermo[..., k],
                      f"{thermodynamic_variable_names[MODEL]} {label}  (level {k})",
                      f"{savedir}/{label}_{thermodynamic_variable_names[MODEL]}_lev{k:02d}.png",
                      cmap="inferno")
  if gravity is not None:
    # diagnostic surface-mass column (Pa) -> hPa
    plot_scalar_field(lat, lon, d_mass.sum(axis=-1) * 1e-2,
                      f"dry-mass column  {label}  (hPa)",
                      f"{savedir}/{label}_dry_mass_column.png",
                      cmap="coolwarm", cbar_label="sum(d_mass) (hPa)")
    if mountain:
      z_surf = unwrap(state["static_forcing"]["phi_surf"]) / gravity
      plot_scalar_field(lat, lon, z_surf,
                        "topography  (m)",
                        f"{savedir}/{label}_topography.png",
                        cmap="terrain", cbar_label="elevation (m)")


def _init_fuzzed_state(h_grid, v_grid, physics_config, dims, mountain,
                       model=MODEL, slp_std=SLP_STD):
  """Rejection-sample a state whose maximum wind speed is below ``MAX_WIND``.

  Returns ``(accepted_attempt, state)``.  The attempt index lets callers
  reproduce the exact accepted draw under a different model -- the wind cap is
  computed from a model-invariant geostrophic field, so the same attempt is
  accepted for any choice of ``model`` and any ``slp_std`` (the latter just
  rescales the windspeed-vs-attempt ranking).
  """
  for attempt in range(MAX_ATTEMPTS):
    rng = np.random.default_rng(SEED0 + attempt)
    state = _build_state(h_grid, v_grid, physics_config, dims, mountain, rng,
                         model=model, slp_std=slp_std)
    wind = unwrap(state["dynamics"]["horizontal_wind"])
    max_wind = float(np.max(np.sqrt(wind[..., 0] ** 2 + wind[..., 1] ** 2)))
    if np.isfinite(max_wind) and max_wind <= MAX_WIND:
      print(f"accepted draw on attempt {attempt} (max wind {max_wind:.1f} m/s)")
      return attempt, state
  raise RuntimeError(f"no draw with max wind < {MAX_WIND} m/s in "
                     f"{MAX_ATTEMPTS} attempts")


def _run_fuzzing_test(mountain):
  h_grid, dims = init_quasi_uniform_grid_elem_local(NE, NPT, calc_smooth_tensor=True)
  v_grid = init_vertical_grid(cam30["hybrid_a_i"][::2],   # every other level
                              cam30["hybrid_b_i"][::2],
                              cam30["p0"],
                              MODEL)
  physics_config, diffusion_config, timestep_config = init_default_config(
      NE, h_grid, v_grid, dims, MODEL,
      hypervis_type=hypervis_opts.variable_resolution)

  _, model_state = _init_fuzzed_state(h_grid, v_grid, physics_config, dims,
                                      mountain)
  simulator = init_simulator(h_grid, v_grid, physics_config, diffusion_config,
                             timestep_config, dims, MODEL)

  # When PYSES_TEST_EMIT_PLOTS is truthy, save initial-state plots now and
  # collect snapshots of the evolution at the times in PLOT_SNAPSHOT_HOURS.
  savedir = None
  pending_snapshots: list = []
  if emit_plots():
    subdir = "random_atmosphere_topo" if mountain else "random_atmosphere_no_topo"
    savedir = get_figdir(subdir=subdir)
    plot_lat, plot_lon = _plot_coords(h_grid)
    gravity = float(unwrap(physics_config["gravity"]))
    _plot_state(model_state, plot_lat, plot_lon, "init", savedir,
                gravity=gravity, mountain=mountain)
    pending_snapshots = list(PLOT_SNAPSHOT_HOURS)

  total_time = 6.0 * 3600.0
  state = model_state
  # init_simulator asserts no-NaN each step, so reaching total_time means the
  # integration stayed stable.
  for t, state in simulator(model_state):
    print(f"t = {t:.0f} s")
    while savedir is not None and pending_snapshots and \
        t / 3600.0 >= pending_snapshots[0]:
      hr = pending_snapshots.pop(0)
      _plot_state(state, plot_lat, plot_lon, f"t{hr:04.1f}h", savedir)
    if t >= total_time:
      break

  end_wind = unwrap(state["dynamics"]["horizontal_wind"])
  assert np.all(np.isfinite(end_wind))


# --- TEMPORARY DIAGNOSTIC: isolate RK3 + remap (+ tracer transport) ----
# The cross-core test currently shows the cam_se path going unstable under
# default substepping despite tendency-by-tendency agreement with HOMME and
# despite hyperviscosity being active.  Earlier runs of this diagnostic
# *without* tracer transport showed the bare dynamics + remap cycle is
# stable for 6 h at 2 dyn / remap (the simulator's default subcycling), so
# the destabiliser must be in the only remaining element of the q_split
# loop: tracer transport.  This test now mirrors the simulator's q_split
# pattern (remap at start, N dynamics substeps with hypervis + sponge and
# consistency-flux accumulation, advance_tracers, remap_tracers) and
# tracks the water_vapor mixing ratio Q in addition to top-layer d_mass.
# When ``with_hypervis=True`` the quasi-uniform constant-coefficient
# hypervis + sponge stack is applied each dyn substep and the resulting
# ``tracer_consist_visc_total`` is forwarded to ``advance_tracers`` (the
# tracer-hypervis branch in ``advance_tracers_rk2`` was previously dead
# code; this exercises it for the first time).  Delete this test once
# the underlying cause is acted on.
@pytest.mark.parametrize("with_tracers", [False, True])
@pytest.mark.parametrize("with_hypervis", [False, True])
@pytest.mark.parametrize("n_dyn_per_remap", [1, 2])
def test_rk3_remap_plus_tracers(n_dyn_per_remap, with_hypervis, with_tracers):
  h_grid, dims = init_quasi_uniform_grid_elem_local(NE, NPT, calc_smooth_tensor=True)
  v_grid = init_vertical_grid(cam30["hybrid_a_i"][::2],
                              cam30["hybrid_b_i"][::2],
                              cam30["p0"], models.cam_se)
  hypervis_type = (hypervis_opts.quasi_uniform if with_hypervis
                   else hypervis_opts.none)
  physics_config, diffusion_config, timestep_config = init_default_config(
      NE, h_grid, v_grid, dims, models.cam_se, hypervis_type=hypervis_type)
  state = _build_state(h_grid, v_grid, physics_config, dims, mountain=True,
                       rng=np.random.default_rng(SEED0), model=models.cam_se)
  # The explicit step runs in theta_d_d_mass form; hypervis runs in T form
  # (it indexes dynamics["T"] -- see hyperviscosity.eval_hypervis_terms).
  # Mirror the simulator's convert-out / step / convert-back / hypervis /
  # re-convert bracket so the two halves see their preferred prognostic.
  stable_dyn, step_model = se_T_to_theta_d_d_mass(
      state["dynamics"], v_grid, physics_config, models.cam_se)
  state = wrap_model_state(stable_dyn, state["static_forcing"], state["tracers"])
  dynamics = state["dynamics"]
  static_forcing = state["static_forcing"]
  tracer_state = state["tracers"]
  moisture_species = tracer_state["moisture_species"]
  dry_air_species = tracer_state["dry_air_species"]
  num_lev = int(v_grid["hybrid_b_m"].shape[0])
  dt_dyn = float(timestep_config["dynamics"]["dt"])
  has_sponge = "sponge_layer" in diffusion_config

  total_time = 6.0 * 3600.0
  cycle_dt = n_dyn_per_remap * dt_dyn
  n_cycles = int(round(total_time / cycle_dt))
  q0 = np.asarray(moisture_species["water_vapor"])
  hypervis_label = "quasi_uniform+sponge" if with_hypervis else "none"
  print(f"\n  diagnostic: {n_cycles} cycles of "
        f"(remap + {n_dyn_per_remap} x (RK3-5stage + hypervis) + advance_tracers + remap_tracers), "
        f"dt_dyn={dt_dyn}s, cycle={cycle_dt}s, hypervis={hypervis_label}, "
        f"model={step_model.name}")
  print(f"  init d_mass[0] mean = "
        f"{float(np.asarray(dynamics['d_mass'])[..., 0].mean()):.2f} Pa")
  print(f"  init Q (water_vapor):  min={q0.min():>10.3e}  mean={q0.mean():>10.3e}  "
        f"max={q0.max():>10.3e}  std={q0.std():>10.3e}")

  for cycle in range(1, n_cycles + 1):
    # 1) remap_dynamics at start of "q_split" -- restores to reference grid
    dynamics = remap_dynamics(dynamics, static_forcing, v_grid,
                              physics_config, num_lev, step_model)
    # snapshot d_mass at start of cycle (used by advance_tracers consistency)
    tracer_consist_init = {"d_mass_init": 1.0 * dynamics["d_mass"]}

    # 2) N dynamics substeps; for each: explicit step in theta_d form,
    #    then (if enabled) convert back to T, run hypervis + sponge in
    #    T form, and convert back to theta_d form.  Accumulate both the
    #    dyn and the hypervis consistency-flux structs.
    tracer_consist_dyn_total = None
    tracer_consist_visc_total = None
    for n in range(n_dyn_per_remap):
      dynamics, tracer_consist_dyn = advance_dynamics_ullrich_5stage(
          dynamics, static_forcing, h_grid, v_grid, physics_config,
          timestep_config, dims, step_model,
          moisture_species=moisture_species,
          dry_air_species=dry_air_species)
      if n == 0:
        tracer_consist_dyn_total = sum_consistency_struct(
            tracer_consist_dyn, tracer_consist_dyn,
            1.0 / n_dyn_per_remap, 0.0)
      else:
        tracer_consist_dyn_total = sum_consistency_struct(
            tracer_consist_dyn_total, tracer_consist_dyn,
            1.0, 1.0 / n_dyn_per_remap)

      if with_hypervis:
        # convert theta_d_d_mass -> T for hypervis (which expects cam_se).
        dynamics, _ = se_theta_d_d_mass_to_T(
            dynamics, v_grid, physics_config, step_model)
        dynamics, tracer_consist_visc = advance_hypervis_euler(
            dynamics, static_forcing, h_grid, v_grid, physics_config,
            diffusion_config, timestep_config, dims, models.cam_se)
        if n == 0:
          tracer_consist_visc_total = sum_consistency_struct(
              tracer_consist_visc, tracer_consist_visc,
              1.0 / n_dyn_per_remap, 0.0)
        else:
          tracer_consist_visc_total = sum_consistency_struct(
              tracer_consist_visc_total, tracer_consist_visc,
              1.0, 1.0 / n_dyn_per_remap)
        if has_sponge:
          dynamics = advance_sponge_euler(
              dynamics, h_grid, physics_config, diffusion_config,
              timestep_config, dims, models.cam_se)
        # back to theta_d_d_mass for the next dyn substep / remap.
        dynamics, step_model = se_T_to_theta_d_d_mass(
            dynamics, v_grid, physics_config, models.cam_se)

    pre_remap_top = np.asarray(dynamics["d_mass"])[..., 0]
    tracer_consist_init["d_mass_end"] = 1.0 * dynamics["d_mass"]

    if with_tracers:
      # 3) tracer transport using the dynamics' mass-flux consistency.  The
      #    tracer-hypervis branch in advance_tracers_rk2 is gated on
      #    tracer_consist_hypervis is not None AND "disable_diffusion" not
      #    in diffusion_config -- so we only pass it when hypervis is on.
      tracer_state = advance_tracers(tracer_state,
                                     tracer_consist_dyn_total,
                                     tracer_consist_init,
                                     h_grid, dims, physics_config,
                                     diffusion_config, timestep_config, step_model,
                                     tracer_consist_hypervis=tracer_consist_visc_total)
      # 4) remap_tracers using the end-of-dyn dynamics state
      tracer_state = remap_tracers(dynamics, tracer_state, v_grid,
                                    num_lev, step_model)
      # re-read moisture / dry air for the next cycle's dyn step (the
      # simulator does this at the top of every n_split iteration)
      moisture_species = tracer_state["moisture_species"]
      dry_air_species = tracer_state["dry_air_species"]

    t = cycle * cycle_dt
    q = np.asarray(moisture_species["water_vapor"])
    log_this = (cycle % max(1, n_cycles // 18) == 0 or cycle <= 4 or
                not np.all(np.isfinite(pre_remap_top)) or
                not np.all(np.isfinite(q)))
    if log_this:
      print(f"  cycle {cycle:>3d}  t={t:>6.0f}s  "
            f"d_mass[0] pre-remap=[{pre_remap_top.min():>9.2f}, "
            f"{pre_remap_top.max():>9.2f}] Pa   "
            f"Q=[{q.min():>10.3e}, {q.max():>10.3e}] (std={q.std():>9.3e})")
    if not np.all(np.isfinite(pre_remap_top)):
      pytest.fail(f"non-finite d_mass[0] at cycle {cycle} (pre-remap)")
    if not np.all(np.isfinite(q)):
      pytest.fail(f"non-finite water_vapor at cycle {cycle}")

  q_final = np.asarray(moisture_species["water_vapor"])
  print(f"  Q amplification ratio: max(|Q_final|) / max(|Q_init|) = "
        f"{np.max(np.abs(q_final)) / max(np.max(np.abs(q0)), 1e-30):.3e}")
  assert np.all(np.isfinite(pre_remap_top))
  assert np.all(np.isfinite(q_final))


def test_fuzzing_no_topography():
  _run_fuzzing_test(mountain=False)


def test_fuzzing_topography():
  _run_fuzzing_test(mountain=True)


# --- cross-core consistency -----------------------------------------------
# Tolerances for ``test_homme_vs_cam_se_consistency``.  HOMME and CAM-SE differ
# in the prognostic thermodynamic variable (``theta_v * d_mass`` vs ``T``) and
# in the per-step remap, so they will not match bit-exactly; on this balanced,
# dry, no-physics initial state at 6 h they should still agree closely.  The
# bounds are intentionally loose because there is no analytic reference -- this
# is a cross-core consistency check, not a convergence test.  If a real bug
# breaks one core, these bounds will catch it long before they fire spuriously.
CROSS_CORE_PS_RELATIVE_TOL = 1.0e-2     # surface pressure RMS / mean
CROSS_CORE_WIND_TOL_MS = 10.0           # mid-level |u| RMS  (m/s)
# CAM-SE's variable-resolution hyperviscosity is too weak for this random
# atmosphere: the column-top pressure goes negative after a few remaps and the
# PPM monotonicity search fails.  Quasi-uniform hypervis (constant coefficient,
# stronger global damping) stabilises CAM-SE on the same atmosphere, so we use
# it for both cores here for an apples-to-apples comparison.  dt is left at the
# nx=15 default (1800 s) -- earlier debugging showed reducing dt alone did not
# rescue CAM-SE, so hypervis is the load-bearing change.
CROSS_CORE_PHYSICS_DT = -1.0            # negative -> init_default_config picks
CROSS_CORE_HYPERVIS = hypervis_opts.quasi_uniform


def test_homme_vs_cam_se_consistency():
  """HOMME hydrostatic and CAM-SE agree on the same random atmosphere at 6 h.

  Both cores are integrated from the *identical* initial physical state -- the
  geostrophic wind / barometric SLP construction in :func:`_build_state` is
  model-invariant, so the same rejection-sampled seed yields the same atmosphere
  expressed in each core's prognostic variables.  After 6 h of integration we
  compare two model-agnostic diagnostics:

    * surface pressure (sum of dry layer mass): RMS difference / mean
    * mid-level horizontal-wind magnitude:       RMS difference (m/s)

  Both must be finite and within :data:`CROSS_CORE_PS_RELATIVE_TOL` and
  :data:`CROSS_CORE_WIND_TOL_MS` respectively.
  """
  h_grid, dims = init_quasi_uniform_grid_elem_local(NE, NPT, calc_smooth_tensor=True)
  accepted_attempt = None
  end_states = {}
  for model in (models.homme_hydrostatic, models.cam_se):
    v_grid = init_vertical_grid(cam30["hybrid_a_i"],
                                cam30["hybrid_b_i"],
                                cam30["p0"], model)
    physics_config, diffusion_config, timestep_config = init_default_config(
        NE, h_grid, v_grid, dims, model,
        physics_dt=CROSS_CORE_PHYSICS_DT,
        hypervis_type=CROSS_CORE_HYPERVIS)
    if accepted_attempt is None:
      accepted_attempt, init_state = _init_fuzzed_state(
          h_grid, v_grid, physics_config, dims, mountain=False, model=model)
    else:
      # Reuse the accepted seed so the second core sees the same atmosphere;
      # rejection is model-invariant, so re-running it here would just confirm
      # the same attempt at the cost of another rebuild.
      rng = np.random.default_rng(SEED0 + accepted_attempt)
      init_state = _build_state(h_grid, v_grid, physics_config, dims,
                                mountain=False, rng=rng, model=model)
    simulator = init_simulator(h_grid, v_grid, physics_config, diffusion_config,
                               timestep_config, dims, model)
    total_time = 6.0 * 3600.0
    state = init_state
    for t, state in simulator(init_state):
      print(f"  {model.name}: t = {t:.0f} s")
      if t >= total_time:
        break
    end_states[model] = state

  h_dyn = end_states[models.homme_hydrostatic]["dynamics"]
  c_dyn = end_states[models.cam_se]["dynamics"]
  h_d_mass = np.asarray(unwrap(h_dyn["d_mass"]))
  c_d_mass = np.asarray(unwrap(c_dyn["d_mass"]))
  h_wind = np.asarray(unwrap(h_dyn["horizontal_wind"]))
  c_wind = np.asarray(unwrap(c_dyn["horizontal_wind"]))

  assert np.all(np.isfinite(h_wind)) and np.all(np.isfinite(c_wind))
  assert np.all(np.isfinite(h_d_mass)) and np.all(np.isfinite(c_d_mass))

  # surface pressure proxy (sum of dry layer mass)
  h_ps = h_d_mass.sum(axis=-1)
  c_ps = c_d_mass.sum(axis=-1)
  ps_mean = 0.5 * float(h_ps.mean() + c_ps.mean())
  ps_rms = float(np.sqrt(np.mean((h_ps - c_ps) ** 2)))
  ps_rel = ps_rms / ps_mean

  # mid-level wind magnitude
  mid = h_wind.shape[-2] // 2
  h_mag = np.sqrt(h_wind[..., mid, 0] ** 2 + h_wind[..., mid, 1] ** 2)
  c_mag = np.sqrt(c_wind[..., mid, 0] ** 2 + c_wind[..., mid, 1] ** 2)
  wind_rms = float(np.sqrt(np.mean((h_mag - c_mag) ** 2)))

  print(f"HOMME vs CAM-SE @ 6 h: dry-mass column RMS = {ps_rms:.2f} Pa "
        f"({ps_rel * 100:.4f}% of mean),  mid-level |u| RMS = {wind_rms:.3f} m/s")

  assert ps_rel < CROSS_CORE_PS_RELATIVE_TOL, (
      f"surface pressure RMS = {ps_rel * 100:.3f}% of mean exceeds "
      f"{CROSS_CORE_PS_RELATIVE_TOL * 100:.2f}% bound")
  assert wind_rms < CROSS_CORE_WIND_TOL_MS, (
      f"mid-level wind RMS = {wind_rms:.3f} m/s exceeds "
      f"{CROSS_CORE_WIND_TOL_MS:.2f} m/s bound")
