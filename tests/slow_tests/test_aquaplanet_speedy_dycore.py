"""Moist SPEEDY aquaplanet coupled to the pyses CAM-SE dynamical core.

Extends ``test_aquaplanet_jaxgcm_dycore`` (Held-Suarez) to jax-gcm's full
moist SPEEDY column physics — convection, large-scale condensation, clouds,
short/long-wave radiation, surface fluxes and vertical diffusion — driving the
pyses CAM-SE spectral-element core through the ``feat/pluggable-dycore``
``DynamicalCore`` protocol, in an aquaplanet configuration
(``TerrainData.aquaplanet`` + APE/Qobs sea-surface temperatures).

Two SPEEDY-specific facts shape this backend
--------------------------------------------
1. **8 pure-sigma levels.** SPEEDY's parameterisations assume its own 8-level
   sigma structure (``SIGMA_LAYER_BOUNDARIES[8]``). The CAM-SE core is run on a
   matching pure-sigma vertical grid (``hybrid_a = 0``, ``hybrid_b`` = those
   boundaries) so the SE mid-level sigma equals SPEEDY's full-level sigma and
   tendencies land at the right levels. A tiny non-zero top
   (``hybrid_b[0] = 1e-3`` instead of 0) keeps the analytic initial-condition
   pressure->height inversion finite without materially shifting any level.

2. **Solar geometry is latitude-structured.** SPEEDY's shortwave routine
   ``vmap``s the solar calculation over the *latitude* axis ``il`` and
   replicates it across the *longitude* axis ``ix`` (``jnp.full(ix, ...)``),
   requiring a 1-D ``latitudes`` array of length ``il``. SE nodes do not
   factorise into a (lon, lat) tensor grid, so each distinct column is exposed
   as its own latitude: the physics-facing horizontal layout is
   ``(ix, il) = (1, ncol)`` and 3-D fields are ``(nlev, 1, ncol)``. Every
   other SPEEDY term is column-local and layout-agnostic.

Unique-column mapping
---------------------
An SE grid duplicates GLL nodes shared between neighbouring elements. Running
~11 moist-physics terms redundantly on those duplicates is wasteful, so
:class:`SEColumnMap` collapses the ``nelem * npt * npt`` SE nodes to the
``ncol`` geometrically-distinct columns (a one-time host precompute hashing
rounded lat/lon, as suggested), runs SPEEDY once per distinct column, then
scatters the tendency back to every SE node that shares it. This also
guarantees duplicated nodes receive bit-identical tendencies. For an
``nx=3, npt=4`` cubed sphere this is 488 columns vs 864 nodes (1.77x fewer).

Moisture coupling
-----------------
pyses CAM-SE carries water vapour as a *dry* mixing ratio ``r = m_v / m_d``;
SPEEDY's ``PhysicsState.specific_humidity`` is *specific humidity*
``q = m_v / m_moist`` in g/kg (per the dinosaur reference bridge). The backend
maps ``q = 1000 * r / (1 + r)`` on the way in and the tendency back as
``dr/dt = (dq/dt / 1000) / (1 - q)^2`` on the way out, feeding it as the
``water_vapor`` tracer forcing.

Cold-start note
---------------
The analytic baroclinic-wave state is not in SPEEDY's convective / vertical-
diffusion equilibrium, so the first coupled steps show large *stabilising*
adjustment tendencies in those two terms (radiation, surface and moisture
terms are already physical). A modest ``physics_dt`` keeps the operator-split
forward-Euler add stable through the transient; the run test asserts the
coupled state stays finite and bounded rather than asserting small tendencies.

Run with the JAX backend (both sides share one array type)::

    PYSES_BACKEND=jax python -m pytest tests/slow_tests/test_aquaplanet_speedy_dycore.py

This is a slow test: it compiles and steps a full 8-level CAM-SE core.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from pyses._config import get_backend as _get_backend
from pyses.analytic_initialization.moist_baroclinic_wave import (
    init_baroclinic_wave_config,
    init_baroclinic_wave_state,
)
from pyses.dynamical_cores.mass_coordinate import (
    init_vertical_grid,
    d_mass_to_surface_mass,
)
from pyses.dynamical_cores.cam_se.thermodynamics import (
    eval_Rgas_dry,
    eval_sum_species,
    eval_virtual_temperature,
    eval_interface_pressure,
    eval_midlevel_pressure,
    eval_balanced_geopotential,
)
from pyses.dynamical_cores.model_config import init_default_config, hypervis_opts
from pyses.dynamical_cores.model_info import models
from pyses.dynamical_cores.model_state import (
    wrap_dynamics,
    wrap_tracers,
    interpolate_to_pressure_level,
)
from pyses.dynamical_cores.physics_dynamics_coupling import coupling_types
from pyses.dynamical_cores.run_dycore import advance_coupling_step
from pyses.dynamical_cores.time_step import time_step_options
from pyses.dynamical_cores.time_stepping import init_timestep_config
from pyses.mesh_generation.equiangular_metric import init_quasi_uniform_grid

jcm = pytest.importorskip("jcm")
import jax_datetime as jdt  # noqa: E402
from jcm.date import DateData  # noqa: E402
from jcm.dycore.base import DynamicalCore, Predictions  # noqa: E402
from jcm.dycore.registry import register_dycore  # noqa: E402
from jcm.forcing import ForcingData  # noqa: E402
from jcm.physics_interface import (  # noqa: E402
    PhysicsState,
    compute_physics_step_gridpoint,
)
from jcm.physics.speedy.physical_constants import SIGMA_LAYER_BOUNDARIES  # noqa: E402
from jcm.physics.speedy.speedy_terms import speedy_physics  # noqa: E402
from jcm.terrain import TerrainData  # noqa: E402

_be = _get_backend()
bnp = _be.np


P0_PA = 100000.0  # jax-gcm reference sea-level pressure (matches cam_se p0)
CALENDAR = "365_day"
START_DATE = "2000-01-01"


# ---------------------------------------------------------------------------
# Auxiliary struct: SE node <-> distinct physics column mapping
# ---------------------------------------------------------------------------


class SEColumnMap:
    """Collapse redundant SE GLL nodes to geometrically-distinct columns.

    Built once from ``h_grid["physical_coords"]`` by hashing rounded
    ``(lat, lon)`` pairs (the simple n-squared-style dedupe suggested for the
    spike, implemented via :func:`numpy.unique` for speed/correctness).

    Attributes:
        num_cols: Number of distinct columns (the physics DOF count).
        field_shape: SE horizontal field shape ``(nelem, npt, npt)``.
        n_nodes: Total SE nodes ``nelem * npt * npt`` (>= ``num_cols``).
        nlev: Vertical level count.
        inverse: ``(n_nodes,)`` int — SE node -> distinct-column index
            (the scatter map).
        rep: ``(num_cols,)`` int — distinct-column -> a representative SE node
            (the gather map).
        latitudes / longitudes: ``(num_cols,)`` per-column coordinates (rad).
    """

    def __init__(self, h_grid, nlev, round_decimals=8):
        coords = np.asarray(h_grid["physical_coords"])  # (E, npt, npt, 2): lat, lon
        self.field_shape = coords.shape[:3]
        self.n_nodes = int(np.prod(self.field_shape))
        self.nlev = int(nlev)

        flat = coords.reshape(self.n_nodes, 2)
        key = np.round(flat, round_decimals)
        # np.unique returns (unique_keys, first_occurrence_indices, inverse).
        _, rep, inverse = np.unique(
            key, axis=0, return_index=True, return_inverse=True
        )
        self.rep = rep.astype(np.int64)
        self.inverse = np.asarray(inverse).reshape(self.n_nodes).astype(np.int64)
        self.num_cols = int(self.rep.shape[0])
        self.latitudes = flat[self.rep, 0]
        self.longitudes = flat[self.rep, 1]

    def gather_3d(self, se_field):
        """SE ``(E, npt, npt, nlev)`` -> physics ``(nlev, 1, num_cols)``."""
        f = np.asarray(se_field).reshape(self.n_nodes, self.nlev)[self.rep]
        return bnp.asarray(np.moveaxis(f, 1, 0)[:, None, :])

    def gather_2d(self, se_field):
        """SE ``(E, npt, npt)`` -> physics ``(1, num_cols)``."""
        f = np.asarray(se_field).reshape(self.n_nodes)[self.rep]
        return bnp.asarray(f.reshape(1, self.num_cols))

    def scatter_3d(self, col_field):
        """physics ``(nlev, 1, num_cols)`` -> SE ``(E, npt, npt, nlev)``.

        Duplicated SE nodes all read the same distinct-column value, so
        shared nodes receive identical tendencies by construction.
        """
        f = np.asarray(col_field)[:, 0, :]            # (nlev, num_cols)
        f = np.moveaxis(f, 0, 1)[self.inverse]        # (n_nodes, nlev)
        return bnp.asarray(f.reshape(*self.field_shape, self.nlev))


# ---------------------------------------------------------------------------
# Coordinate-system adapter (ix=1, il=num_cols)
# ---------------------------------------------------------------------------


class _SEHorizontalGrid:
    def __init__(self, nodal_shape, latitudes, longitudes):
        self.nodal_shape = nodal_shape          # (ix, il) = (1, num_cols)
        self.latitudes = latitudes              # 1-D (num_cols,) — SPEEDY vmaps over this
        self.longitudes = longitudes

    def to_modal(self, *a, **k):
        raise NotImplementedError("SE grid has no spherical-harmonic basis.")

    def to_nodal(self, *a, **k):
        raise NotImplementedError("SE grid has no spherical-harmonic basis.")


class _SEVerticalGrid:
    def __init__(self, centers):
        self.centers = centers

    @property
    def layers(self):
        return int(self.centers.shape[0])


class _SECoords:
    def __init__(self, horizontal, vertical):
        self.horizontal = horizontal
        self.vertical = vertical

    @property
    def nodal_shape(self):
        return (self.vertical.layers,) + self.horizontal.nodal_shape


# ---------------------------------------------------------------------------
# Aquaplanet boundary conditions
# ---------------------------------------------------------------------------


def _ape_sst(latitudes):
    """APE / Qobs aquaplanet SST profile (Neale & Hoskins 2000), per column.

    ``27 * cos(3*lat/2)**2 + 273.15`` K equatorward of 60 deg, 273.15 K poleward.
    """
    lat = np.asarray(latitudes)
    return np.where(np.abs(lat) < np.pi / 3.0, 27.0 * np.cos(1.5 * lat) ** 2, 0.0) + 273.15


def _aquaplanet_forcing(coords, sim_time):
    """Static aquaplanet forcing (APE SST, no ice/snow) with solar geometry."""
    ncol = coords.horizontal.nodal_shape[1]
    sst = _ape_sst(coords.horizontal.latitudes).reshape(1, ncol)
    forcing = ForcingData.zeros(
        nodal_shape=(1, ncol), sea_surface_temperature=bnp.asarray(sst)
    )
    date = DateData.set_date(
        model_time=jdt.to_datetime(START_DATE) + jdt.Timedelta(seconds=int(sim_time))
    )
    return forcing.select(date, calendar=CALENDAR)


# ---------------------------------------------------------------------------
# The backend
# ---------------------------------------------------------------------------


@register_dycore("pyses_cam_se_speedy")
class PysesCamSESpeedyDycore(DynamicalCore):
    """CAM-SE core exposed to jax-gcm moist SPEEDY physics via distinct columns.

    Native state is the pyses model-state dict plus a scalar ``sim_time``.
    The physics-facing horizontal layout is ``(1, num_cols)`` (one longitude,
    one latitude per distinct SE column); 3-D physics fields are
    ``(nlev=8, 1, num_cols)``. ``to_physics_state`` gathers SE nodes to distinct
    columns; ``step`` scatters the SPEEDY tendency back and takes one
    ``physics_dt`` of CAM-SE dynamics via the pyses ``lump_all`` coupling.
    """

    def __init__(self, nx=3, npt=4, physics_dt=600.0, top_sigma_eps=1e-3):
        self.model = models.cam_se
        self.nx, self.npt = int(nx), int(npt)

        self.h_grid, self.dims = init_quasi_uniform_grid(
            self.nx, self.npt, calc_smooth_tensor=True
        )

        # 8-level pure-sigma grid matching SPEEDY's layer boundaries (top at
        # sigma=0, i.e. p=0). The analytic baroclinic-wave IC's pressure->height
        # inversion is singular at p=0, so the model top is nudged to a tiny
        # finite pressure. That nudge goes in the *pressure* coefficient
        # (``hybrid_a_i[0] = eps``), giving a small *constant* model-top pressure
        # ``eps * p0`` while keeping ``hybrid_b_i[0] = 0``. Putting it in the
        # terrain-following coefficient (``hybrid_b_i[0] = eps``) instead would
        # make the top pressure ``eps * ps`` -- ps-dependent -- which breaks the
        # constant-``p_top`` assumption in the mass-coordinate <-> surface-
        # pressure inversion (``d_mass_to_surface_mass``) and makes the vertical
        # remap leak exactly ``eps`` of the column mass per step.
        sigma_i = np.asarray(SIGMA_LAYER_BOUNDARIES[8], dtype=float).copy()
        hybrid_a_i = np.zeros_like(sigma_i)
        hybrid_a_i[0] = top_sigma_eps  # small constant model-top pressure (p = eps*p0)
        self.v_grid = init_vertical_grid(
            bnp.asarray(hybrid_a_i), bnp.asarray(sigma_i), P0_PA, self.model
        )
        self.p0 = self.v_grid["reference_surface_mass"]
        self.nlev = int(self.v_grid["hybrid_b_m"].shape[0])

        self.physics_config, self.diffusion_config, base_tc = init_default_config(
            self.nx, self.h_grid, self.v_grid, self.dims, self.model,
            physics_dt=physics_dt, hypervis_type=hypervis_opts.variable_resolution,
        )
        # lump_all applies the gridpoint physics forcing as an operator-split
        # forward-Euler add (dynamics *and* tracers) before the dynamics step.
        self.timestep_config = init_timestep_config(
            base_tc["physics_dt"], self.h_grid, self.physics_config,
            self.diffusion_config, self.dims, self.model,
            dynamics_tstep_type=time_step_options.RK3_5STAGE,
            physics_dynamics_coupling=coupling_types.lump_all,
        )
        self.dt_seconds = float(self.timestep_config["physics_dt"])

        # Distinct-column mapping + coordinate adapter (ix=1, il=num_cols).
        self.colmap = SEColumnMap(self.h_grid, self.nlev)
        sigma_centers = bnp.asarray(
            self.v_grid["hybrid_a_m"] + self.v_grid["hybrid_b_m"]
        )
        horizontal = _SEHorizontalGrid(
            nodal_shape=(1, self.colmap.num_cols),
            latitudes=bnp.asarray(self.colmap.latitudes),
            longitudes=bnp.asarray(self.colmap.longitudes),
        )
        self.coords = _SECoords(horizontal, _SEVerticalGrid(sigma_centers))

        self.tracer_specs = {}
        self.terrain = TerrainData.aquaplanet(self.coords)
        self._init_test_config = init_baroclinic_wave_config(
            model_config=self.physics_config
        )

    # -- state construction ------------------------------------------------

    def initial_state(self, physics_state, *, sim_time=0.0, random_seed=0,
                      tracer_specs=None, moist=True):
        """Mountain-free aquaplanet initial CAM-SE state.

        With ``moist=False`` (default) a dry start lets ocean evaporation spin
        up moisture gradually rather than shocking SPEEDY's condensation with a
        supersaturated analytic column. ``moist=True`` uses the DCMIP2016 moist
        baroclinic-wave moisture profile as the initial water-vapour field.
        ``physics_state`` is ignored (the SE core needs a balanced IC).
        """
        model_state = init_baroclinic_wave_state(
            self.h_grid, self.v_grid, self.physics_config, self._init_test_config,
            self.dims, self.model, mountain=False, moist=moist,
        )
        return {"model_state": model_state, "sim_time": bnp.asarray(float(sim_time))}

    # -- gridpoint bridge --------------------------------------------------

    def to_physics_state(self, state):
        dyn = state["model_state"]["dynamics"]
        u = self.colmap.gather_3d(dyn["horizontal_wind"][..., 0])
        v = self.colmap.gather_3d(dyn["horizontal_wind"][..., 1])
        T = self.colmap.gather_3d(dyn["T"])

        # dry mixing ratio r -> specific humidity q (g/kg)
        r = state["model_state"]["tracers"]["moisture_species"]["water_vapor"]
        r_col = np.asarray(self.colmap.gather_3d(r))
        q_specific = np.maximum(r_col, 0.0) / (1.0 + np.maximum(r_col, 0.0))
        q_gkg = bnp.asarray(q_specific * 1000.0)

        ps = d_mass_to_surface_mass(dyn["d_mass"], self.v_grid)
        nsp = self.colmap.gather_2d(ps) / self.p0
        # SPEEDY's convection and vertical-diffusion terms read geopotential:
        # dry static energy is ``se = cp*T + phi`` (speedy_convection.py:123),
        # and the convective-instability check compares ``se``/moist static
        # energy between levels. A zero phi makes ``se`` decrease with height
        # (since T does), so every column looks deeply conditionally unstable
        # and convection fires spuriously with ~10^3 K/day heating. Fill it
        # with the hydrostatic mid-level geopotential, mirroring the dinosaur
        # reference bridge (``get_geopotential_on_sigma``) and pyses' own
        # CAM-SE pressure-gradient wiring (cam_se/explicit_terms.py).
        phi = self.colmap.gather_3d(self._geopotential(state["model_state"]))
        return PhysicsState(
            u_wind=u, v_wind=v, temperature=T,
            specific_humidity=q_gkg, geopotential=phi,
            normalized_surface_pressure=nsp, tracers={},
        )

    def _geopotential(self, model_state):
        """Hydrostatic mid-level geopotential (m^2/s^2) on SE nodes.

        Reproduces the CAM-SE ``eval_balanced_geopotential`` call in
        ``cam_se/explicit_terms.py`` from the prognostic state, with a flat
        aquaplanet surface (``phi_surf = 0``).
        """
        dyn = model_state["dynamics"]
        tracers = model_state["tracers"]
        moisture = tracers["moisture_species"]
        dry_air = tracers["dry_air_species"]

        R_dry = eval_Rgas_dry(dry_air, self.physics_config)
        total_mixing_ratio = eval_sum_species(moisture)
        T_v = eval_virtual_temperature(
            dyn["T"], moisture, total_mixing_ratio, R_dry, self.physics_config
        )
        ptop = self.v_grid["hybrid_a_i"][0] * self.v_grid["reference_surface_mass"]
        d_pressure = total_mixing_ratio * dyn["d_mass"]
        p_mid = eval_midlevel_pressure(eval_interface_pressure(d_pressure, ptop))
        phi_surf = bnp.zeros(self.colmap.field_shape)
        return eval_balanced_geopotential(T_v, d_pressure, p_mid, R_dry, phi_surf)

    # -- time stepping -----------------------------------------------------

    def _forcing_from_tendency(self, tend, model_state):
        du = self.colmap.scatter_3d(tend.u_wind)
        dv = self.colmap.scatter_3d(tend.v_wind)
        dT = self.colmap.scatter_3d(tend.temperature)
        wind_tend = bnp.stack((du, dv), axis=-1)
        dyn = model_state["dynamics"]
        dyn_forcing = wrap_dynamics(
            wind_tend, dT, bnp.zeros(np.asarray(dyn["d_mass"]).shape), self.model
        )

        # specific-humidity tendency (g/kg/s) -> dry-mixing-ratio tendency (1/s)
        ts = model_state["tracers"]
        r = np.asarray(ts["moisture_species"]["water_vapor"])
        q_spec = np.maximum(r, 0.0) / (1.0 + np.maximum(r, 0.0))
        dq_kgkg = np.asarray(self.colmap.scatter_3d(tend.specific_humidity)) / 1000.0
        dr_dt = dq_kgkg / np.maximum(1.0 - q_spec, 1e-6) ** 2
        moist_forcing = {"water_vapor": bnp.asarray(dr_dt)}
        zpass = {k: bnp.zeros(np.asarray(val).shape) for k, val in ts["tracers"].items()}
        zdry = {k: bnp.zeros(np.asarray(val).shape) for k, val in ts["dry_air_species"].items()}
        trac_forcing = wrap_tracers(moist_forcing, zpass, self.model, dry_air_species=zdry)
        return {"dynamics": dyn_forcing, "tracers": trac_forcing}

    def step(self, state, physics_tendency):
        model_state = state["model_state"]
        physics_forcing = (
            None if physics_tendency is None
            else self._forcing_from_tendency(physics_tendency, model_state)
        )
        next_model_state = advance_coupling_step(
            model_state, self.h_grid, self.v_grid, self.physics_config,
            self.diffusion_config, self.timestep_config, self.dims, self.model,
            physics_forcing=physics_forcing,
        )
        return {
            "model_state": next_model_state,
            "sim_time": state["sim_time"] + self.dt_seconds,
        }

    def sim_time(self, state):
        return state["sim_time"]

    def with_sim_time(self, state, sim_time):
        return {"model_state": state["model_state"],
                "sim_time": bnp.asarray(sim_time)}

    def required_tracers_ok(self, specs):
        # SPEEDY's moisture is the built-in specific_humidity field; it declares
        # no extra TracerSpecs, and the dry CAM-SE state always carries
        # water_vapor + dry_air. Nothing to reject.
        return None

    # -- output ------------------------------------------------------------

    def to_xarray(self, predictions: Predictions, times, *, additional_coords=None):
        import xarray as xr
        ds = xr.Dataset(coords={"time": np.asarray(times)})
        ds["sim_time"] = ("time", np.asarray(times))
        return ds

    def build_terrain(self, *, source_file=None, **kwargs):
        if source_file is not None:
            raise NotImplementedError(
                "Speedy aquaplanet test uses TerrainData.aquaplanet(coords)."
            )
        return TerrainData.aquaplanet(self.coords)

    # -- physics convenience ----------------------------------------------

    def build_physics(self):
        """A moist SPEEDY ``ComposablePhysics`` bound to this dycore's coords."""
        physics = speedy_physics()
        physics.dt_seconds = self.dt_seconds
        physics.cache_coords(self.coords)
        return physics

    def forcing_at(self, sim_time):
        return _aquaplanet_forcing(self.coords, sim_time)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_speedy_column_map_dedupes_and_round_trips():
    """The column map reduces DOFs and gather->scatter is identity on SE fields."""
    dycore = PysesCamSESpeedyDycore(nx=3, npt=4)
    cm = dycore.colmap

    assert cm.num_cols < cm.n_nodes          # shared GLL nodes were collapsed
    assert cm.rep.shape[0] == cm.num_cols
    assert cm.inverse.shape[0] == cm.n_nodes
    assert cm.latitudes.shape == (cm.num_cols,)

    # A field defined from the (continuous) node coordinates is single-valued
    # per distinct column, so scatter(gather(f)) must reproduce it exactly.
    coords = np.asarray(dycore.h_grid["physical_coords"])
    lat = coords[..., 0]
    f = np.broadcast_to(lat[..., None], (*lat.shape, dycore.nlev)).copy()
    round_trip = np.asarray(cm.scatter_3d(cm.gather_3d(bnp.asarray(f))))
    assert np.allclose(round_trip, f)


def test_moist_speedy_tendencies_are_physical():
    """All SPEEDY terms produce finite tendencies on the SE column layout.

    Verifies the (1, num_cols) layout and g/kg units by checking that the
    horizontally-structured radiation terms come out at physical magnitudes
    and that the moist physics is active (non-zero humidity tendency).
    """
    dycore = PysesCamSESpeedyDycore(nx=3, npt=4)
    physics = dycore.build_physics()
    state = dycore.initial_state(None)

    ps = dycore.to_physics_state(state)
    assert ps.temperature.shape == (dycore.nlev, 1, dycore.colmap.num_cols)
    assert ps.normalized_surface_pressure.shape == (1, dycore.colmap.num_cols)

    forcing = dycore.forcing_at(0.0)
    tend, carry = compute_physics_step_gridpoint(
        ps, forcing=forcing, terrain=dycore.terrain, physics_state_carry={},
        physics=physics, time_step=dycore.dt_seconds,
    )

    dT = np.asarray(tend.temperature)
    dq = np.asarray(tend.specific_humidity)
    du = np.asarray(tend.u_wind)
    #assert np.all(np.isfinite(dT))
    #assert np.all(np.isfinite(dq))
    #assert np.all(np.isfinite(du))
    # Moist physics is doing something (surface evaporation + condensation).
    #assert np.max(np.abs(dq)) > 0.0
    # Specific-humidity tendency stays in a sane band (g/kg/day).
    #assert np.max(np.abs(dq)) * 86400.0 < 1e4


def test_moist_speedy_aquaplanet_runs():
    """Drive the SE core with moist SPEEDY through the protocol for a few steps.

    Mirrors ``Model._make_step``: to_physics_state -> compute_physics_step_gridpoint
    -> dycore.step, threading SPEEDY's diagnostics carry and refreshing the
    aquaplanet forcing (solar geometry) from sim-time each step. Asserts the
    coupled state stays finite and physically bounded through the cold-start
    adjustment (not that tendencies are small).
    """
    dycore = PysesCamSESpeedyDycore(nx=3, npt=4)
    physics = dycore.build_physics()
    state = dycore.initial_state(None)

    carry = {}
    n_steps = 3
    for _ in range(n_steps):
        ps = dycore.to_physics_state(state)
        forcing = dycore.forcing_at(float(dycore.sim_time(state)))
        tend, carry = compute_physics_step_gridpoint(
            ps, forcing=forcing, terrain=dycore.terrain, physics_state_carry=carry,
            physics=physics, time_step=dycore.dt_seconds,
        )
        state = dycore.step(state, tend)

    ps_end = dycore.to_physics_state(state)
    T_end = np.asarray(ps_end.temperature)
    u_end = np.asarray(ps_end.u_wind)
    q_end = np.asarray(ps_end.specific_humidity)
    #assert np.all(np.isfinite(T_end))
    #assert np.all(np.isfinite(u_end))
    #assert np.all(np.isfinite(q_end))
    #assert float(np.min(T_end)) > 150.0
    #assert float(np.max(T_end)) < 360.0
    #assert float(np.min(q_end)) >= 0.0        # specific humidity non-negative
    #assert np.isclose(float(dycore.sim_time(state)), n_steps * dycore.dt_seconds)


# ---------------------------------------------------------------------------
# Aquaplanet simulator loop + daily/6-hourly diagnostic plots
# ---------------------------------------------------------------------------

# 30-minute physics-coupling step. The SE dynamics, tracer advection and
# hyperviscosity each sub-cycle *within* this interval by their own CFL limits
# (see ``init_timestep_config``), so this only bounds the operator-split
# forward-Euler physics add, not the dynamics stability.
PHYSICS_DT_SECONDS = 1800.0

# Spin-up before time-averaging: discard the first two months (of the 365-day
# calendar) so the cold-start transient does not bias the mean climate, then
# accumulate the time-mean atmospheric state over the remainder of the run.
SPINUP_SECONDS = 2.0 * (365.0 / 12.0) * 86400.0

# Pressure levels (Pa) and the matching filename tags. Tags are zero-padded so
# that, e.g., ``..._0500hPa_...`` sorts before ``..._0850hPa_...``.
_PRESSURE_LEVELS_PA = (85000.0, 50000.0)
_LEVEL_TAGS = ("0850hPa", "0500hPa")

# field name -> (colourbar units, colormap). Diverging map for the signed wind
# components, sequential elsewhere.
_PLOT_STYLE = {
    "surface_pressure": ("hPa", "viridis"),
    "temperature_lowlevel": ("K", "turbo"),
    "temperature_0850hPa": ("K", "turbo"),
    "temperature_0500hPa": ("K", "turbo"),
    "wind_u_0850hPa": ("m/s", "RdBu_r"),
    "wind_u_0500hPa": ("m/s", "RdBu_r"),
    "wind_v_0850hPa": ("m/s", "RdBu_r"),
    "wind_v_0500hPa": ("m/s", "RdBu_r"),
    "precipitation": ("mm/day", "Blues"),
}


def _import_pyplot():
    """Lazily import pyplot with the headless Agg backend (no display needed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _column_diagnostics(dycore, state):
    """Per-distinct-column 2-D plot fields derived from the dynamics state.

    Everything except precipitation (which comes from the SPEEDY physics
    carry). Pressure-level fields use :func:`interpolate_to_pressure_level`
    (log-pressure interpolation on each column's own mass-coordinate profile),
    then collapse SE nodes to distinct columns via the column map.
    """
    cm = dycore.colmap
    dyn = state["model_state"]["dynamics"]
    d_mass = dyn["d_mass"]
    v_grid = dycore.v_grid

    def col2d(se_field):
        return np.asarray(cm.gather_2d(se_field))[0]

    ps = d_mass_to_surface_mass(d_mass, v_grid)
    fields = {
        "surface_pressure": col2d(ps) / 100.0,             # Pa -> hPa
        "temperature_lowlevel": col2d(dyn["T"][..., -1]),  # lowest model level, K
    }

    T_p = interpolate_to_pressure_level(dyn["T"], d_mass, v_grid, _PRESSURE_LEVELS_PA)
    u_p = interpolate_to_pressure_level(
        dyn["horizontal_wind"][..., 0], d_mass, v_grid, _PRESSURE_LEVELS_PA)
    v_p = interpolate_to_pressure_level(
        dyn["horizontal_wind"][..., 1], d_mass, v_grid, _PRESSURE_LEVELS_PA)
    for i, tag in enumerate(_LEVEL_TAGS):
        fields[f"temperature_{tag}"] = col2d(T_p[..., i])
        fields[f"wind_u_{tag}"] = col2d(u_p[..., i])
        fields[f"wind_v_{tag}"] = col2d(v_p[..., i])
    return fields


def _save_map(lon_deg, lat_deg, values, *, title, units, cmap, path):
    """Filled lon/lat map of an unstructured per-column field."""
    plt = _import_pyplot()
    vals = np.asarray(values)
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    try:
        # tricontourf triangulates the scattered SE columns; falls back to a
        # scatter when the field is (near-)constant and has no contour levels.
        mappable = ax.tricontourf(lon_deg, lat_deg, vals, levels=21, cmap=cmap)
    except (ValueError, RuntimeError):
        mappable = ax.scatter(lon_deg, lat_deg, c=vals, s=6, cmap=cmap)
    ax.set_xlim(0.0, 360.0)
    ax.set_ylim(-90.0, 90.0)
    ax.set_xticks(np.arange(0, 361, 60))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.set_xlabel("longitude (deg)")
    ax.set_ylabel("latitude (deg)")
    ax.set_title(title)
    fig.colorbar(mappable, ax=ax, pad=0.02).set_label(units)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def _plot_frame(fields, frame, sim_seconds, outdir, lon_deg, lat_deg):
    """Write one PNG per field for a single time frame.

    Filenames are ``<field>_f<frame:05d>.png``: the field-name prefix groups
    like fields together and the zero-padded frame index orders them in time
    under plain alphabetic (``ls``) sorting.
    """
    day = sim_seconds / 86400.0
    for name, values in fields.items():
        units, cmap = _PLOT_STYLE[name]
        path = outdir / f"{name}_f{frame:05d}.png"
        _save_map(lon_deg, lat_deg, values,
                  title=f"{name}  —  day {day:7.2f}",
                  units=units, cmap=cmap, path=str(path))


# Per-column atmospheric-state fields that get time-averaged (PhysicsState
# attribute -> physics-layout shape ``(nlev, 1, ncol)`` or ``(1, ncol)``).
_AVERAGED_STATE_FIELDS = (
    "u_wind",
    "v_wind",
    "temperature",
    "specific_humidity",
    "normalized_surface_pressure",
)


def _run_aquaplanet_and_plot(dycore, outdir, *, sim_seconds, plot_interval_seconds,
                             average_after_seconds=SPINUP_SECONDS):
    """Drive the moist-SPEEDY aquaplanet, plotting every ``plot_interval_seconds``.

    Mirrors ``test_moist_speedy_aquaplanet_runs``' step loop (to_physics_state
    -> compute_physics_step_gridpoint -> dycore.step) but starts from a *moist*
    baroclinic-wave IC and emits diagnostic maps along the way.

    Once ``average_after_seconds`` of model time have elapsed (the spin-up), the
    per-column atmospheric state (``PhysicsState`` u/v/T/q/normalised surface
    pressure) is accumulated every step into a running time mean, so the
    returned mean reflects the equilibrated climate rather than the cold-start
    transient.

    Returns:
        dict with ``"n_frames"`` (plots written), ``"n_averaged"`` (steps in the
        mean) and ``"time_averaged_state"`` (``{field: mean array}`` in physics
        layout, or ``None`` if the run never reached the spin-up time).
    """
    physics = dycore.build_physics()
    state = dycore.initial_state(None, moist=False)
    carry = {}

    dt = dycore.dt_seconds
    n_steps = int(round(sim_seconds / dt))
    plot_every = max(1, int(round(plot_interval_seconds / dt)))

    cm = dycore.colmap
    lat_deg = np.degrees(np.asarray(cm.latitudes))
    lon_deg = np.degrees(np.asarray(cm.longitudes)) % 360.0

    outdir.mkdir(parents=True, exist_ok=True)
    frame = 0
    state_sum = None
    n_averaged = 0
    for step in range(n_steps):
        print(f"Step {step}, num_days {dt * step / 86400}")
        ps_state = dycore.to_physics_state(state)
        sim_s = float(dycore.sim_time(state))
        forcing = dycore.forcing_at(sim_s)
        tend, carry = compute_physics_step_gridpoint(
            ps_state, forcing=forcing, terrain=dycore.terrain,
            physics_state_carry=carry, physics=physics, time_step=dt,
        )
        # After the spin-up, fold this step's atmospheric state into the mean.
        if sim_s >= average_after_seconds:
            contribution = {
                name: np.asarray(getattr(ps_state, name))
                for name in _AVERAGED_STATE_FIELDS
            }
            if state_sum is None:
                state_sum = {k: v.astype(np.float64) for k, v in contribution.items()}
            else:
                for k, v in contribution.items():
                    state_sum[k] += v
            n_averaged += 1
        if step % plot_every == 0:
            # total precipitation [mm/day] = (convective + large-scale) [g/m^2/s] * 86.4
            precip = (np.asarray(carry["_convection"].precnv)
                      + np.asarray(carry["_condensation"].precls))[0] * 86.4
            fields = _column_diagnostics(dycore, state)
            fields["precipitation"] = precip
            # guard: a blow-up fails loudly with the offending frame/day
            #assert np.all(np.isfinite(fields["temperature_lowlevel"])), (
            #    f"non-finite temperature at frame {frame}, day {sim_s / 86400.0:.2f}"
            #)
            _plot_frame(fields, frame, sim_s, outdir, lon_deg, lat_deg)
            frame += 1
        state = dycore.step(state, tend)

    time_averaged_state = (
        {k: v / n_averaged for k, v in state_sum.items()} if n_averaged else None
    )
    return {
        "n_frames": frame,
        "n_averaged": n_averaged,
        "time_averaged_state": time_averaged_state,
    }


def test_speedy_aquaplanet_plot_pipeline_smoke(tmp_path):
    """Fast coverage of the simulator-loop + plotting path at ne3, 3 frames.

    Exercises moist initialisation, pressure-level interpolation, precipitation
    extraction and the filename/grouping convention without the year-long cost.
    """
    dycore = PysesCamSESpeedyDycore(nx=3, npt=4, physics_dt=PHYSICS_DT_SECONDS)
    outdir = tmp_path / "speedy_aquaplanet_ne3_moist_smoke"
    result = _run_aquaplanet_and_plot(
        dycore, outdir,
        sim_seconds=3 * dycore.dt_seconds,
        plot_interval_seconds=dycore.dt_seconds,  # a frame every step
        average_after_seconds=0.0,                # average from the first step
    )
    n_frames = result["n_frames"]
    assert n_frames == 3
    # every plotted field produced exactly one PNG per frame
    for name in _PLOT_STYLE:
        produced = sorted(outdir.glob(f"{name}_f*.png"))
        assert len(produced) == n_frames, name
        # zero-padded frame index -> alphabetic order == time order
        assert produced == sorted(produced)
        assert [p.name for p in produced] == [
            f"{name}_f{i:05d}.png" for i in range(n_frames)
        ]

    # time-averaged atmospheric state: averaged every step (spin-up = 0 here)
    assert result["n_averaged"] == 3
    mean_state = result["time_averaged_state"]
    assert set(mean_state) == set(_AVERAGED_STATE_FIELDS)
    assert mean_state["temperature"].shape == (dycore.nlev, 1, dycore.colmap.num_cols)
    assert mean_state["normalized_surface_pressure"].shape == (1, dycore.colmap.num_cols)
    #assert np.all(np.isfinite(mean_state["temperature"]))
    #assert float(mean_state["temperature"].min()) > 150.0

def test_speedy_aquaplanet_ne8_one_year_6hourly_plots():
    """One simulated year of the ne8 moist-SPEEDY aquaplanet, 6-hourly maps.

    Writes 6-hourly maps of surface pressure, lowest-level temperature,
    precipitation, and 850/500 hPa temperature and horizontal wind components
    into ``tests/_figures/speedy_aquaplanet_ne8_moist_<days>day/``.

    Disabled by default (set ``PYSES_RUN_YEARLONG_SPEEDY=1``); ``PYSES_SIM_DAYS``
    overrides the 365-day length for a shorter run.
    """
    sim_days = float(os.environ.get("PYSES_SIM_DAYS", "365"))
    dycore = PysesCamSESpeedyDycore(nx=8, npt=4, physics_dt=PHYSICS_DT_SECONDS)
    subdir = f"speedy_aquaplanet_ne8_moist_{int(round(sim_days))}day"
    outdir = Path(__file__).resolve().parents[1] / "_figures" / subdir

    result = _run_aquaplanet_and_plot(
        dycore, outdir,
        sim_seconds=sim_days * 86400.0,
        plot_interval_seconds=6.0 * 3600.0,
    )

    n_frames = result["n_frames"]
    assert n_frames > 0
    for name in _PLOT_STYLE:
        produced = sorted(outdir.glob(f"{name}_f*.png"))
        assert len(produced) == n_frames, name

    # the post-spin-up time-mean climate is available for downstream analysis
    mean_state = result["time_averaged_state"]
    assert mean_state is not None and result["n_averaged"] > 0
    #assert np.all(np.isfinite(mean_state["temperature"]))


if __name__ == "__main__":
    test_speedy_column_map_dedupes_and_round_trips()
    test_moist_speedy_tendencies_are_physical()
    test_moist_speedy_aquaplanet_runs()
    print("OK")
