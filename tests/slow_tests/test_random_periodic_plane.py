"""
Randomized ("fuzzing") stability test for the HOMME *non-hydrostatic* dynamical
core on a doubly-periodic Cartesian plane with idealized surface physics.

This is the planar analogue of ``test_random_atmosphere.py`` (which fuzzes the
HOMME *hydrostatic* core on the cubed sphere).  The geometry here is a
50 km x 50 km doubly-periodic plane (``init_uniform_grid``); the prognostic core
is ``models.homme_nonhydrostatic`` integrated with the HEVI (horizontally-
explicit / vertically-implicit) Runge-Kutta scheme.

Initial state
-------------
* **Topography** (two-step construction):
    1. *Rough field.*  Because the plane is periodic the rough topography is a
       sum of 2-D Fourier modes with a brown-noise (``k**-2`` power, i.e.
       ``k**-1`` amplitude) spectrum and random phases.  Modes whose wavelength
       is shorter than ``2*dx`` -- ``dx`` the (variable) minimum GLL spacing of
       the spectral-element grid -- are spectrally truncated so the field is
       representable.  The result is shifted to be non-negative and normalized
       so its maximum height is below 1 km.
    2. *Central plateau.*  With ``zbar`` the area-mean of the rough field,
       ``r`` the distance to the domain centre and ``f(r) = exp(-r**6 / lam)``
       (``lam`` chosen so ``f(12 km) = 0.1``), the surface is blended as
       ``z_surf = f*zbar + (1 - f)*z_rough`` -- a flat "ice" plateau in the
       middle, rough "land" outside.
* **Temperature**: piecewise-linear in height -- lapse 7 K/km from the surface
  to 10 km, 0.1 K/km from 10-20 km, then -2 K/km above (an inversion).  Sea-
  level (``z = 0``) temperature is 280 K.  Dry, so virtual temperature == T.
  The matching hydrostatic pressure ``p(z)`` (a per-segment barometric law) is
  handed to ``init_analytic_state`` as ``p_moist``.
* **Winds**: zonal wind a Gaussian in height centred at 10 km, peak 60 m/s and
  30 m/s at 10 +/- 5 km; meridional wind zero.
* **Moisture**: initially zero everywhere.  Water vapour, cloud water and rain
  water are carried as dry mixing-ratio tracers; the central lake (below) is the
  moisture source and the Kessler scheme (below) processes it into rain.

Surface physics forcing (operator-split ``lump_all`` coupling)
--------------------------------------------------------------
A sigmoid ``w_surf`` (large near the surface in sigma, small aloft) sets the
wind damping; the central ``f(r)`` region is a *lake* and the rest is *land*:

* **Winds.**  Where ``w_surf`` is large the horizontal *and* vertical winds are
  damped toward zero; where it is small the horizontal winds are damped toward
  their initial values.  Combined, the horizontal target is
  ``(1 - w_surf)*u_init`` and the vertical target is ``0`` (weighted by
  ``w_surf``), each relaxed with e-folding time ``TAU_WIND``.
* **Land** (weight ``1 - f(r)``).  The lowest ``N_RELAX`` layers relax toward a
  diurnal surface temperature ``T_surf_init + 10 K * cos(2*pi*t / SIDEREAL_DAY)``
  with e-folding time ``TAU_TEMP``.
* **Lake** (weight ``f(r)``).  Over the central region the lowest ``N_RELAX``
  layers' water vapour is linearly relaxed toward its saturation mixing ratio
  ``m_vs(T, p)`` (Tetens' formula) with e-folding time ``TAU_LAKE``; the latent
  heat of that evaporation is drawn from the same layers, cooling them by
  ``-(L/c_p) * dm_v/dt``.

Kessler warm-rain microphysics (Hughes & Jablonowski 2023, Appendix A)
----------------------------------------------------------------------
Applied at every level: saturation adjustment (condensation / cloud
evaporation, Eq. A15), autoconversion (A7) and accretion (A8) of cloud to rain,
rain evaporation in subsaturated air (A13-A14), and rain sedimentation with the
Kessler terminal velocity (A10-A11).  The net latent heating
``(L/c_p)(C_cond - E_r)`` and the moisture tendencies are added as forcing; the
temperature tendency is converted to the prognostic ``theta_v_d_mass`` tendency
via the diagnosed Exner function.

The test asserts the integration stays finite (the core's per-step NaN guards do
the rest), i.e. that this rough-terrain, strongly-sheared, moist, surface-forced
configuration integrates stably.
"""
import numpy as np
import pytest

from pyses._config import get_backend as _get_backend
from pyses.dynamical_cores.initialization import init_analytic_state
from pyses.dynamical_cores.mass_coordinate import init_vertical_grid
from pyses.dynamical_cores.physics_config import init_physics_config
from pyses.dynamical_cores.hyperviscosity import init_hypervis_config_tensor
from pyses.dynamical_cores.model_info import models
from pyses.dynamical_cores.model_state import wrap_dynamics, wrap_tracers
from pyses.dynamical_cores.homme.thermodynamics import eval_mu
from pyses.dynamical_cores.physics_dynamics_coupling import coupling_types
from pyses.dynamical_cores.run_dycore import advance_coupling_step
from pyses.dynamical_cores.time_step import time_step_options
from pyses.dynamical_cores.time_stepping import init_timestep_config
from pyses.mesh_generation.periodic_plane import init_uniform_grid
from ..context import emit_plots, get_figdir
from ..test_data.mass_coordinate_grids import cam30

_be = _get_backend()
jnp = _be.np
device_wrapper = _be.array
unwrap = _be.unwrap

# --- resolution / model choices -------------------------------------------
NX = NY = 20                      # elements per side
NPT = 4                          # GLL points per element edge
LENGTH = 50.0e3                  # 50 km square domain (m)
MODEL = models.homme_nonhydrostatic_f_plane   # f-plane: init_static_forcing
                                              # gives a constant Coriolis (the
                                              # sphere lat/lon formula is
                                              # meaningless on a Cartesian grid)

# --- topography parameters ------------------------------------------------
TOPO_MAX = 800.0                 # max rough-terrain height (m), below 1 km
PLATEAU_RADIUS = 12.0e3          # f(r) drops to 0.1 here (m)
TOPO_SEED = 20240601

# --- temperature profile (piecewise-linear lapse) -------------------------
T_SEA_LEVEL = 285.0              # surface temperature at z = 0 (K)
LAPSE_TROP = 7.0e-3              # 0 -> 10 km   (K/m)
LAPSE_TROPP = 0.1e-3             # 10 -> 20 km  (K/m)
LAPSE_STRAT = -2.0e-3             # > 20 km      (K/m, inversion)
Z_TROP = 10.0e3
Z_STRAT = 5.0e3
SLP_PA = 1.0e5                   # sea-level pressure (Pa)

# --- zonal wind (Gaussian in height) --------------------------------------
U_MAX = 60.0                     # peak zonal wind (m/s)
U_CENTER = 15.0e3                # height of the jet core (m)
U_HALF_OFFSET = 7.0e3            # at center +/- this, u = U_HALF_VALUE
U_HALF_VALUE = 30.0

# --- surface physics forcing ----------------------------------------------
SIGMA_C = 0.7                    # sigmoid centre in sigma
SIGMA_WIDTH = 0.08               # sigmoid width in sigma
TAU_WIND = 4 * 24.0 * 3600.0          # wind-damping e-folding time (s)
TAU_TEMP = 3.0 * 3600.0          # land temperature-relaxation e-folding time (s)
TAU_LAKE = 1800.0                # lake vapour -> saturation e-folding time (s)
N_RELAX = 2                      # number of lowest layers forced at the surface
T_DIURNAL_AMP = 20.0             # diurnal amplitude of the "land" target (K)

# --- Kessler warm-rain microphysics (paper Appendix A) --------------------
LATENT_HEAT = 2.5e6              # latent heat of vaporization L (J/kg)
K1_AUTO = 1.0e-3                 # autoconversion rate k1 (1/s), Eq. A7
AR_THRESH = 1.0e-3               # autoconversion cloud threshold a_r (kg/kg)
K2_ACCR = 2.2                    # accretion coefficient k2, Eq. A8
V_RAIN_COEF = 36.34              # rain terminal-velocity coefficient, Eq. A11

# --- time stepping --------------------------------------------------------
PHYSICS_DT = 0.1                # physics-coupling interval (s)
RUN_STEPS = 4000                   # number of coupling steps (20 min)


# ---------------------------------------------------------------------------
# Grid spacing
# ---------------------------------------------------------------------------
def _min_gll_spacing(coords):
  """Smallest adjacent GLL spacing in the plane (the 'variable' grid spacing).

  ``coords`` is ``(elem, i, j, 2)``; x varies along the ``i`` axis and y along
  the ``j`` axis within an element, so the in-element neighbour gaps are the
  successive differences along those axes.
  """
  dx = np.abs(np.diff(coords[..., 0], axis=1))
  dy = np.abs(np.diff(coords[..., 1], axis=2))
  gaps = np.concatenate([dx.ravel(), dy.ravel()])
  gaps = gaps[gaps > 1e-9]
  return float(gaps.min())


# ---------------------------------------------------------------------------
# Topography
# ---------------------------------------------------------------------------
def _rough_topography(coords, dx_min, rng):
  """Periodic brown-noise (``k**-2`` power) topography truncated below ``2*dx``.

  Built as an explicit Fourier sum over a half-plane of integer modes so the
  field is exactly periodic and can be evaluated directly at the (non-uniform)
  GLL nodes.  Amplitudes scale as ``1 / |k|`` (giving a ``k**-2`` power law)
  with random phases; modes whose wavelength is below ``2*dx_min`` are dropped.
  """
  x = coords[..., 0]
  y = coords[..., 1]
  # keep modes with wavelength >= 2*dx_min  <=>  |k_cycles/m| <= 1/(2*dx_min)
  k_cutoff = 1.0 / (4.0 * dx_min)
  kmax_x = int(np.floor(LENGTH * k_cutoff))
  kmax_y = int(np.floor(LENGTH * k_cutoff))
  field = np.zeros_like(x)
  for kx in range(0, kmax_x + 1):
    for ky in range(-kmax_y, kmax_y + 1):
      if kx == 0 and ky <= 0:
        continue                      # half-plane: skip k=0 and conjugates
      k_mag = np.sqrt((kx / LENGTH) ** 2 + (ky / LENGTH) ** 2)
      if k_mag > k_cutoff:
        continue
      amp = rng.normal() / k_mag      # |amp| ~ |k|^-1  ->  power ~ |k|^-2
      phase = rng.uniform(0.0, 2.0 * np.pi)
      field += amp * np.cos(2.0 * np.pi * (kx * x / LENGTH + ky * y / LENGTH)
                            + phase)
  # shift to non-negative elevation, then normalize the peak below 1 km
  field = field - field.min()
  field = field * (TOPO_MAX / field.max())
  return field


def _area_mean(values, h_grid):
  """Area-weighted mean over the plane using the assembled SE mass matrix.

  The mass matrix holds each *global* node's full quadrature weight, replicated
  on the duplicate GLL nodes that neighbouring elements share; summing over the
  geometrically-distinct nodes only (deduplicated by rounded coordinate) gives
  the exact spectral-element integral.
  """
  coords = unwrap(h_grid["physical_coords"]).reshape(-1, 2)
  mass = np.asarray(unwrap(h_grid["mass_matrix"])).reshape(-1)
  vals = np.asarray(values).reshape(-1)
  _, rep = np.unique(np.round(coords, 6), axis=0, return_index=True)
  return float(np.sum(mass[rep] * vals[rep]) / np.sum(mass[rep]))


def _topography(coords, h_grid, rng):
  """Two-step surface height: rough brown noise blended with a central plateau."""
  dx_min = _min_gll_spacing(coords)
  z_rough = _rough_topography(coords, dx_min, rng)
  zbar = _area_mean(z_rough, h_grid)
  f = _plateau_mask(coords)
  return f * zbar + (1.0 - f) * z_rough


def _domain_center(coords):
  """Centre of the (axis-aligned) periodic plane from the coordinate extent."""
  cx = 0.5 * (coords[..., 0].min() + coords[..., 0].max())
  cy = 0.5 * (coords[..., 1].min() + coords[..., 1].max())
  return cx, cy


def _plateau_radius_field(coords):
  cx, cy = _domain_center(coords)
  return np.sqrt((coords[..., 0] - cx) ** 2 + (coords[..., 1] - cy) ** 2)


def _plateau_mask(coords):
  """``f(r) = exp(-r**6 / lam)`` with ``lam`` set so ``f(PLATEAU_RADIUS)=0.1``."""
  r = _plateau_radius_field(coords)
  lam = PLATEAU_RADIUS ** 6 / np.log(10.0)
  return np.exp(-(r ** 6) / lam)


# ---------------------------------------------------------------------------
# Thermodynamic profile (piecewise-linear temperature + barometric pressure)
# ---------------------------------------------------------------------------
def _profile_constants(physics_config):
  """Per-segment base temperatures, base pressures and barometric exponents.

  The three layers have constant lapse rates, so hydrostatic balance integrates
  to a barometric power law ``p = p_base * (T / T_base)**(g / (R * lapse))``
  within each layer (an exponential where the lapse is zero -- not needed here,
  all three lapses are non-zero).  Base pressures are chained upward from the
  sea-level pressure.
  """
  g = float(unwrap(physics_config["gravity"]))
  R = float(unwrap(physics_config["Rgas"]))
  t_trop = T_SEA_LEVEL - LAPSE_TROP * Z_TROP            # T at 10 km
  t_strat = t_trop - LAPSE_TROPP * (Z_STRAT - Z_TROP)  # T at 20 km
  exp_trop = g / (R * LAPSE_TROP)
  exp_strat = g / (R * LAPSE_TROPP)
  exp_meso = g / (R * LAPSE_STRAT)
  p_trop = SLP_PA * (t_trop / T_SEA_LEVEL) ** exp_trop
  p_strat = p_trop * (t_strat / t_trop) ** exp_strat
  return dict(t_trop=t_trop, t_strat=t_strat,
              exp_trop=exp_trop, exp_strat=exp_strat, exp_meso=exp_meso,
              p_trop=p_trop, p_strat=p_strat)


def _temperature_of_z(z, c):
  """Piecewise-linear temperature (K) at height ``z`` (m)."""
  t_a = T_SEA_LEVEL - LAPSE_TROP * z
  t_b = c["t_trop"] - LAPSE_TROPP * (z - Z_TROP)
  t_c = c["t_strat"] - LAPSE_STRAT * (z - Z_STRAT)
  return jnp.where(z <= Z_TROP, t_a, jnp.where(z <= Z_STRAT, t_b, t_c))


def _pressure_of_z(z, c):
  """Hydrostatic pressure (Pa) matching the piecewise-linear temperature."""
  # clip the temperature ratios positive so the unused branches stay finite
  t_a = jnp.maximum(T_SEA_LEVEL - LAPSE_TROP * z, 1.0)
  t_b = jnp.maximum(c["t_trop"] - LAPSE_TROPP * (z - Z_TROP), 1.0)
  t_c = jnp.maximum(c["t_strat"] - LAPSE_STRAT * (z - Z_STRAT), 1.0)
  p_a = SLP_PA * (t_a / T_SEA_LEVEL) ** c["exp_trop"]
  p_b = c["p_trop"] * (t_b / c["t_trop"]) ** c["exp_strat"]
  p_c = c["p_strat"] * (t_c / c["t_strat"]) ** c["exp_meso"]
  return jnp.where(z <= Z_TROP, p_a, jnp.where(z <= Z_STRAT, p_b, p_c))


def _zonal_wind_of_z(z):
  """Gaussian zonal jet: peak ``U_MAX`` at ``U_CENTER``, ``U_HALF_VALUE`` at +/-offset."""
  two_sigma_sq = U_HALF_OFFSET ** 2 / np.log(U_MAX / U_HALF_VALUE)
  return U_MAX * jnp.exp(-(z - U_CENTER) ** 2 / two_sigma_sq)


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------
def _build_state(h_grid, v_grid, physics_config, dims, rng):
  """Build the homme_nonhydrostatic initial state described in the module docstring."""
  coords = unwrap(h_grid["physical_coords"])
  z_surf_np = _topography(coords, h_grid, rng)
  z_surf = device_wrapper(z_surf_np, elem_sharding_axis=0)
  c = _profile_constants(physics_config)

  def p_moist_func(z):
    return _pressure_of_z(z, c)

  def z_pi_surf_func(lat, lon):
    return z_surf, p_moist_func(z_surf)

  def Tv_func(lat, lon, z):
    return _temperature_of_z(z, c)

  def u_func(lat, lon, z):
    return _zonal_wind_of_z(z)

  def v_func(lat, lon, z):
    return jnp.zeros_like(z)

  def Q_func(lat, lon, z):
    return jnp.zeros_like(z)

  def w_func(lat, lon, z):
    return jnp.zeros_like(z)

  state = init_analytic_state(z_pi_surf_func, p_moist_func, Tv_func,
                              u_func, v_func, Q_func,
                              h_grid, v_grid, physics_config, dims, MODEL,
                              w_func=w_func, enforce_hydrostatic=True)
  # Carry cloud water and rain water as (initially zero) dry-mixing-ratio
  # tracers for the Kessler scheme; water vapour already lives in
  # moisture_species (zero here -- the lake is the only source).
  zeros = jnp.zeros_like(state["dynamics"]["d_mass"])
  state["tracers"]["tracers"]["cloud_water"] = 1.0 * zeros
  state["tracers"]["tracers"]["rain_water"] = 1.0 * zeros
  return state


# ---------------------------------------------------------------------------
# Diagnostics / surface physics forcing
# ---------------------------------------------------------------------------
def _diagnose_state(dynamics, v_grid, physics_config):
  """Mid-level temperature (K), Exner, and pressure (Pa) for nonhydrostatic HOMME.

  Moisture is carried passively for HOMME (it does not re-enter the dynamics
  step), so the prognostic ``theta_v`` is effectively dry potential temperature
  and ``T = theta_v * exner``.
  """
  p_model, exner, _, _ = eval_mu(dynamics, dynamics["phi_i"], v_grid,
                                 physics_config, MODEL)
  theta_v = dynamics["theta_v_d_mass"] / dynamics["d_mass"]
  return theta_v * exner, exner, p_model


def _sigmoid_sigma(sigma):
  """Sigmoid: ~1 near the surface (sigma -> 1), ~0 aloft (sigma -> 0)."""
  return 1.0 / (1.0 + np.exp(-(sigma - SIGMA_C) / SIGMA_WIDTH))


def _saturation_mixing_ratio(T, p):
  """Tetens' saturation mixing ratio (paper Eq. B8): ``380/p * exp(...)``.

  ``T`` in K and ``p`` in Pa; the result approximates the saturation dry mixing
  ratio ``m_vs`` used throughout the Kessler scheme.
  """
  return (380.0 / p) * jnp.exp(17.27 * (T - 273.0) / (T - 36.0))


def _kessler_tendencies(m_v, m_c, m_r, T, p, d_mass, c, dt):
  """Kessler warm-rain tendencies (paper Appendix A), per second.

  Returns ``(dm_v, dm_c, dm_r, dT)`` -- the water-vapour, cloud-water and
  rain-water dry-mixing-ratio rates and the latent heating rate (K/s).  The
  rain-evaporation and terminal-velocity formulas (Eqs. A11, A13, A14) are
  Klemp-Wilhelmson forms whose constants assume the dry-air density in
  ``g cm^-3``; sedimentation uses the SI density for the mass flux.  ``c`` holds
  ``g``/``Rd``/``cp`` as floats.
  """
  g, Rd, cp = c["g"], c["Rgas"], c["cp"]
  m_v = jnp.maximum(m_v, 0.0)
  m_c = jnp.maximum(m_c, 0.0)
  m_r = jnp.maximum(m_r, 0.0)
  m_vs = _saturation_mixing_ratio(T, p)
  rho = p / (Rd * T)                     # dry-air density (kg/m^3)
  rho_cgs = rho * 1.0e-3                 # g/cm^3 for the Kessler rain formulas

  # 1. saturation adjustment (Eq. A15): >0 condenses vapour, <0 evaporates cloud
  denom = 1.0 + m_vs * 4098.0 * LATENT_HEAT / (cp * (T - 36.0) ** 2)
  cond = (m_v - m_vs) / (dt * denom)
  cond = jnp.where(cond < 0.0, jnp.maximum(cond, -m_c / dt), cond)

  # 2. autoconversion (Eq. A7) + 3. accretion (Eq. A8), capped by cloud on hand
  auto = K1_AUTO * jnp.maximum(m_c - AR_THRESH, 0.0)
  accr = K2_ACCR * m_c * m_r ** 0.875
  cloud_to_rain = jnp.minimum(auto + accr, m_c / dt)

  # 4. rain evaporation in subsaturated air (Eqs. A13-A14)
  vent = 1.6 + 124.9 * (rho_cgs * m_r) ** 0.2046
  p_hpa = p / 100.0
  evap = ((1.0 / rho_cgs) * (1.0 - m_v / m_vs) * vent * (rho_cgs * m_r) ** 0.525
          / (5.4e5 + 2.55e6 / (p_hpa * m_vs)))
  evap = jnp.minimum(jnp.maximum(evap, 0.0), m_r / dt)

  # 5. sedimentation (Eqs. A10-A11): rain falls at the terminal velocity, with
  # an upstream flux divergence; the bottom layer's flux leaves as precipitation
  rho_surf = rho[..., -1:]
  v_r = V_RAIN_COEF * (rho_cgs * m_r) ** 0.1346 * jnp.sqrt(rho_surf / rho)
  flux = m_r * rho * v_r                 # downward rain-mass flux (kg/m^2/s)
  flux_above = jnp.concatenate((jnp.zeros_like(flux[..., :1]), flux[..., :-1]),
                               axis=-1)
  dm_r_sed = (g / d_mass) * (flux_above - flux)

  dm_v = -cond + evap
  dm_c = cond - cloud_to_rain
  dm_r = cloud_to_rain - evap + dm_r_sed
  dT = (LATENT_HEAT / cp) * (cond - evap)
  return dm_v, dm_c, dm_r, dT


class _SurfacePhysics:
  """Compute the ``lump_all`` physics-forcing dict for the current state.

  Captures the time-invariant pieces (initial winds, initial surface
  temperature, the ``f(r)`` blend mask, the vertical sigmoid weights and the
  bottom-``N_RELAX``-layer mask) so each call only forms the state-dependent
  tendencies.
  """

  def __init__(self, init_state, h_grid, v_grid, physics_config):
    self.v_grid = v_grid
    self.physics_config = physics_config
    dyn = init_state["dynamics"]
    nlev = int(unwrap(dyn["d_mass"]).shape[-1])

    self.u_init = dyn["horizontal_wind"]
    self.zeros_dmass = jnp.zeros_like(dyn["d_mass"])
    self.zeros_phi = jnp.zeros_like(dyn["phi_i"])

    # physical constants as floats, passed to the Kessler kernel
    self.consts = {k: float(unwrap(physics_config[k]))
                   for k in ("gravity", "Rgas", "cp")}
    self.consts["g"] = self.consts["gravity"]
    self.cp = self.consts["cp"]

    # initial near-surface temperature (lowest model layer, index -1)
    t_init, _, _ = _diagnose_state(dyn, v_grid, physics_config)
    self.t_surf_init = unwrap(t_init)[..., -1]                     # (E, i, j)
    self.t_surf_init = device_wrapper(self.t_surf_init, elem_sharding_axis=0)

    # f(r) blend mask, broadcast over the relaxed layers later
    coords = unwrap(h_grid["physical_coords"])
    self.f_mask = device_wrapper(_plateau_mask(coords), elem_sharding_axis=0)

    # vertical sigmoid weights from the hybrid sigma (a + b at p_surf = p0)
    sigma_m = np.asarray(unwrap(v_grid["hybrid_a_m"]) + unwrap(v_grid["hybrid_b_m"]))
    sigma_i = np.asarray(unwrap(v_grid["hybrid_a_i"]) + unwrap(v_grid["hybrid_b_i"]))
    self.w_surf_m = device_wrapper(_sigmoid_sigma(sigma_m).reshape(1, 1, 1, -1))
    self.w_surf_i = device_wrapper(_sigmoid_sigma(sigma_i).reshape(1, 1, 1, -1))

    # bottom-N surface-forcing mask over levels
    layer_mask = np.zeros(nlev)
    layer_mask[-N_RELAX:] = 1.0
    self.layer_mask = device_wrapper(layer_mask.reshape(1, 1, 1, -1))

    self.sidereal_day = 2.0 * np.pi / float(unwrap(physics_config["angular_freq_earth"]))

  def __call__(self, state, t):
    dynamics = state["dynamics"]
    d_mass = dynamics["d_mass"]
    m_v = state["tracers"]["moisture_species"]["water_vapor"]
    m_c = state["tracers"]["tracers"]["cloud_water"]
    m_r = state["tracers"]["tracers"]["rain_water"]
    f = self.f_mask[..., jnp.newaxis]                     # (E, i, j, 1)

    # --- wind damping -----------------------------------------------------
    u = dynamics["horizontal_wind"]
    w_m = self.w_surf_m[..., jnp.newaxis]                 # (...,lev,1) for (u,v)
    target_u = (1.0 - w_m) * self.u_init                  # 0 near surface, u_init aloft
    du = -(u - target_u) / TAU_WIND
    dw_i = -(self.w_surf_i * dynamics["w_i"]) / TAU_WIND  # vertical wind -> 0 near surface

    T, exner, p = _diagnose_state(dynamics, self.v_grid, self.physics_config)
    m_vs = _saturation_mixing_ratio(T, p)

    # --- land: diurnal temperature relaxation in the lowest N layers ------
    cos_t = float(np.cos(2.0 * np.pi * t / self.sidereal_day))
    t_land = self.t_surf_init + T_DIURNAL_AMP * cos_t      # (E, i, j)
    dT_land = ((1.0 - f) * self.layer_mask
               * -(T - t_land[..., jnp.newaxis]) / TAU_TEMP)

    # --- lake: relax vapour toward saturation, draw the latent heat -------
    dm_v_lake = f * self.layer_mask * (m_vs - m_v) / TAU_LAKE
    dT_lake = -(LATENT_HEAT / self.cp) * dm_v_lake
    
    kessler_flag = 0.0
    # --- Kessler warm-rain microphysics at every level --------------------
    dm_v_k, dm_c, dm_r, dT_k = _kessler_tendencies(
        m_v, m_c, m_r, T, p, d_mass, self.consts, PHYSICS_DT)

    dm_v = dm_v_lake + kessler_flag * dm_v_k
    dT = dT_land + dT_lake + kessler_flag * dT_k
    dtheta = (d_mass / exner) * dT                         # T tend -> theta tend

    dyn_forcing = wrap_dynamics(du, dtheta, self.zeros_dmass, MODEL,
                                phi_i=self.zeros_phi, w_i=dw_i)
    trac_forcing = wrap_tracers({"water_vapor": dm_v},
                                {"cloud_water": kessler_flag * dm_c, "rain_water": kessler_flag * dm_r}, MODEL)
    return {"dynamics": dyn_forcing, "tracers": trac_forcing}


# ---------------------------------------------------------------------------
# Configuration / run
# ---------------------------------------------------------------------------
def _build_configs(h_grid, v_grid, dims):
  """Physics/diffusion/timestep configs for the *planar* core.

  The horizontal operators were written for the cubed sphere; on the Cartesian
  plane two things must change for them to be consistent (``init_uniform_grid``
  pre-swaps the metric tensors so the operators' spherical component ``flip``
  resolves correctly -- see ``mesh_generation.periodic_plane``):

  * ``radius_earth = 1`` -- the plane's coordinates are already in metres, so
    the operators must use a unit length scale (no spurious ``1/a`` division).
    The same ``radius_earth`` drives ``eval_cfl``'s ``scale_inv`` and the
    variable-resolution hyperviscosity length scale, so setting it to 1 keeps
    the CFL-derived sub-cycling and the diffusion coefficient self-consistent
    with the metres-based metric.
  * ``lump_all`` coupling so the surface physics forcing is applied (the default
    is ``none``, which silently drops it).
  """
  physics_config = init_physics_config(MODEL, radius_earth=1.0)
  diffusion_config = init_hypervis_config_tensor(h_grid, v_grid, dims,
                                                 physics_config, n_sponge=5)
  timestep_config = init_timestep_config(
      PHYSICS_DT, h_grid, physics_config, diffusion_config, dims, MODEL,
      dynamics_tstep_type=time_step_options.RK3_5STAGE_HEVI,
      physics_dynamics_coupling=coupling_types.lump_all)
  return physics_config, diffusion_config, timestep_config


def _plot_topography(h_grid, init_state, physics_config):
  """Save a terrain map (only when PYSES_TEST_EMIT_PLOTS is set)."""
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  coords = unwrap(h_grid["physical_coords"])
  g = float(unwrap(physics_config["gravity"]))
  z_surf = unwrap(init_state["static_forcing"]["phi_surf"]) / g
  savedir = get_figdir(subdir="random_periodic_plane")
  fig, ax = plt.subplots(figsize=(5.5, 4.5))
  m = ax.tricontourf(coords[..., 0].ravel() / 1e3, coords[..., 1].ravel() / 1e3,
                     z_surf.ravel(), levels=21, cmap="terrain")
  ax.set_xlabel("x (km)")
  ax.set_ylabel("y (km)")
  ax.set_title("surface height (m)")
  fig.colorbar(m, ax=ax).set_label("elevation (m)")
  fig.tight_layout()
  fig.savefig(f"{savedir}/topography.png", dpi=90)
  plt.close(fig)


def _plot_field(stamp, h_grid, field, field_name, level=-3):
  """Save a w map (only when PYSES_TEST_EMIT_PLOTS is set)."""
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  coords = unwrap(h_grid["physical_coords"])
  w = unwrap(field)[..., level]
  savedir = get_figdir(subdir="random_periodic_plane")
  fig, ax = plt.subplots(figsize=(5.5, 4.5))
  m = ax.tricontourf(coords[..., 0].ravel() / 1e3, coords[..., 1].ravel() / 1e3,
                     w.ravel(), levels=21)
  ax.set_xlabel("x (km)")
  ax.set_ylabel("y (km)")
  ax.set_title(f"{field_name}")
  fig.colorbar(m, ax=ax)
  fig.tight_layout()
  fig.savefig(f"{savedir}/{field_name}_lev_{level}_time_{str(stamp).zfill(5)}.png", dpi=90)
  plt.close(fig)


def _run(steps=RUN_STEPS):
  h_grid, dims = init_uniform_grid(NX, NY, NPT, length_x=LENGTH, length_y=LENGTH,
                                   calc_smooth_tensor=True)
  v_grid = init_vertical_grid(cam30["hybrid_a_i"], cam30["hybrid_b_i"],
                              cam30["p0"], MODEL)
  physics_config, diffusion_config, timestep_config = _build_configs(
      h_grid, v_grid, dims)

  rng = np.random.default_rng(TOPO_SEED)
  state = _build_state(h_grid, v_grid, physics_config, dims, rng)

  init_wind = unwrap(state["dynamics"]["horizontal_wind"])
  assert np.all(np.isfinite(init_wind)), "initial wind is not finite"

  if emit_plots():
    _plot_topography(h_grid, state, physics_config)

  physics = _SurfacePhysics(state, h_grid, v_grid, physics_config)

  t = 0.0
  ct = 0
  for step in range(steps):
    forcing = physics(state, t)
    state = advance_coupling_step(state, h_grid, v_grid, physics_config,
                                  diffusion_config, timestep_config, dims, MODEL,
                                  physics_forcing=forcing)
    t += PHYSICS_DT
    if emit_plots() and ct % 10 == 0:
      _plot_field(ct, h_grid, state["dynamics"]["w_i"], "w_i")
      _plot_field(ct, h_grid, state["dynamics"]["horizontal_wind"][..., 0], "u", level=-1)

      _plot_field(ct, h_grid, state["tracers"]["moisture_species"]["water_vapor"], "water_vapor", level=-1)
      _plot_field(ct, h_grid, state["tracers"]["tracers"]["cloud_water"], "cloud_water", level=-12)
      _plot_field(ct, h_grid, state["tracers"]["tracers"]["rain_water"], "rain_water", level=-1)


    print(f"step {step + 1}/{steps}  t = {t:.0f} s")
    ct += 1

  dyn = state["dynamics"]
  trc = state["tracers"]
  checks = {
      "horizontal wind": dyn["horizontal_wind"],
      "thermodynamic variable": dyn["theta_v_d_mass"],
      "geopotential": dyn["phi_i"],
      "water vapor": trc["moisture_species"]["water_vapor"],
      "cloud water": trc["tracers"]["cloud_water"],
      "rain water": trc["tracers"]["rain_water"],
  }
  for name, field in checks.items():
    assert np.all(np.isfinite(unwrap(field))), f"{name} blew up"
  return state


def test_random_periodic_plane_runs():
  """The surface-forced rough-terrain plane integrates stably (state stays finite)."""
  _run()
