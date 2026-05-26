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
from pyses.dynamical_cores.model_info import models
from pyses.mesh_generation.element_local_metric import init_quasi_uniform_grid_elem_local
from ..test_data.mass_coordinate_grids import cam30

_be = _get_backend()
jnp = _be.np
device_wrapper = _be.array
unwrap = _be.unwrap

# --- resolution / model choices -------------------------------------------
NE = 15                       # ne15 cubed-sphere
NPT = 4                       # GLL points per element edge
MODEL = models.homme_hydrostatic

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
CONTINENT_WIDTH_MEAN = np.deg2rad(20.0)   # half-width mean
CONTINENT_WIDTH_STD = np.deg2rad(5.0)
CONTINENT_EXPONENT = 6                     # >2 -> approaches an indicator
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


def _eval_slp(lat, lon, rng):
  """
  Sea-level pressure field (Pa): mean ``SLP_MEAN`` plus random Gaussian hills.

  Centers are uniform on the sphere, half-widths follow a beta distribution
  centered on 3*dx, and amplitudes are N(0, SLP_STD).  Returns a numpy array on
  the grid (precomputed once and captured by ``p_moist`` for speed).
  """
  dx = _dx_radians()
  # number of hills for the requested coverage (footprints per point ~ N w^2 / 4).
  w_mean = 3.0 * dx
  n_hills = max(1, int(round(SLP_COVERAGE * 4.0 / w_mean ** 2)))
  lat_c, lon_c = _random_centers(rng, n_hills)
  half_widths = dx * (1.5 + 3.0 * rng.beta(2.0, 2.0, size=n_hills))   # mean 3*dx
  scales = _half_width_to_scale(half_widths, 2)
  amps = rng.normal(0.0, SLP_STD, size=n_hills)
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


def _build_state(h_grid, v_grid, physics_config, dims, mountain, rng):
  """Build a single random model state (no rejection check)."""
  coords = unwrap(h_grid["physical_coords"])
  lat_np = coords[:, :, :, 0]
  lon_np = coords[:, :, :, 1]

  g = float(unwrap(physics_config["gravity"]))
  Rgas = float(unwrap(physics_config["Rgas"]))
  omega = float(unwrap(physics_config["angular_freq_earth"]))
  exponent = g / (Rgas * GAMMA)

  # precomputed per-column lookups captured by the closures below. elem_sharding_axis=0
  # shards them like the grid under JAX multi-device (no-op for numpy / torch).
  slp = device_wrapper(_eval_slp(lat_np, lon_np, rng), elem_sharding_axis=0)     # (elem, i, j)
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
                             h_grid, v_grid, physics_config, dims, MODEL,
                             w_func=w_func)


def _init_fuzzed_state(h_grid, v_grid, physics_config, dims, mountain):
  """Rejection-sample a state whose maximum wind speed is below ``MAX_WIND``."""
  for attempt in range(MAX_ATTEMPTS):
    rng = np.random.default_rng(SEED0 + attempt)
    state = _build_state(h_grid, v_grid, physics_config, dims, mountain, rng)
    wind = unwrap(state["dynamics"]["horizontal_wind"])
    max_wind = float(np.max(np.sqrt(wind[..., 0] ** 2 + wind[..., 1] ** 2)))
    if np.isfinite(max_wind) and max_wind <= MAX_WIND:
      print(f"accepted draw on attempt {attempt} (max wind {max_wind:.1f} m/s)")
      return state
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

  model_state = _init_fuzzed_state(h_grid, v_grid, physics_config, dims, mountain)
  simulator = init_simulator(h_grid, v_grid, physics_config, diffusion_config,
                             timestep_config, dims, MODEL)

  total_time = 6.0 * 3600.0
  state = model_state
  # init_simulator asserts no-NaN each step, so reaching total_time means the
  # integration stayed stable.
  for t, state in simulator(model_state):
    print(f"t = {t:.0f} s")
    if t >= total_time:
      break

  end_wind = unwrap(state["dynamics"]["horizontal_wind"])
  assert np.all(np.isfinite(end_wind))


def test_fuzzing_no_topography():
  _run_fuzzing_test(mountain=False)


def test_fuzzing_topography():
  _run_fuzzing_test(mountain=True)
