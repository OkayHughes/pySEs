Quickstart
====================

This page walks through two complete examples of setting up and running a 3-D
dynamical-core simulation with pySEs.  Both examples use the base state as described in
`Ullrich et al. (2014) <https://doi.org/10.1002/qj.2241>`_. The first is a steady state test that
uses an unperturbed base state. The second example uses the topographic profile described in `Hughes and Jablonowski, (2023) <https://doi.org/10.5194/gmd-16-6805-2023>`_
to trigger a baroclinic wave.

Imports
-------

Every simulation needs the following modules::

    from pyses.grids import init
    from pyses.model_utils import mass_coordinate, model_config, model_info
    from pyses.initialize import ullrich_baroclinic_wave
    from pyses.simulate import init_simulator

Vertical grid data (CAM-30 hybrid coefficients)
------------------------------------------------

The vertical coordinate is a hybrid sigma-pressure coordinate.  The CAM-30
level set is a common starting point::

    import numpy as np

    hybrid_a_i = np.array([
        0.00225523952394724, 0.00503169186413288, 0.0101579474285245,
        0.0185553170740604,  0.0306691229343414,  0.0458674766123295,
        0.0633234828710556,  0.0807014182209969,  0.0949410423636436,
        0.11169321089983,    0.131401270627975,   0.154586806893349,
        0.181863352656364,   0.17459799349308,    0.166050657629967,
        0.155995160341263,   0.14416541159153,    0.130248308181763,
        0.113875567913055,   0.0946138575673103,  0.0753444507718086,
        0.0576589405536652,  0.0427346378564835,  0.0316426791250706,
        0.0252212174236774,  0.0191967375576496,  0.0136180268600583,
        0.00853108894079924, 0.00397881818935275, 0.0, 0.0,
    ])
    hybrid_b_i = np.array([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0393548272550106, 0.0856537595391273, 0.140122056007385,
        0.204201176762581,  0.279586911201477,  0.368274360895157,
        0.47261056303978,   0.576988518238068,  0.672786951065063,
        0.753628432750702,  0.813710987567902,  0.848494648933411,
        0.881127893924713,  0.911346435546875,  0.938901245594025,
        0.963559806346893,  0.985112190246582,  1.0,
    ])
    p0 = 1e5  # reference surface pressure (Pa)

Example 1 – Quasi-uniform grid, steady-state check
----------------------------------------------------

This example runs the hydrostatic HOMME dynamical core on a quasi-uniform
cubed-sphere grid and verifies that the baroclinic-wave initial state is
close to steady state after one day.

::

    import numpy as np
    from pyses.grids import init
    from pyses.model_utils import mass_coordinate, model_config, model_info
    from pyses.initialize import ullrich_baroclinic_wave
    from pyses.simulate import init_simulator

    # Generate horizontal grid
    npt = 4   # spectral-element polynomial order
    nx  = 15  # elements per cube face

    h_grid, dims = init.init_quasi_uniform_grid(nx, npt, calc_smooth_tensor=True)

    model = model_info.models.homme_hydrostatic
    v_grid = mass_coordinate.init_vertical_grid(hybrid_a_i, hybrid_b_i, p0, model)

    # Use good default configuration. This can be customized!
    physics_config, diffusion_config, timestep_config = model_config.init_default_config(
        nx, h_grid, v_grid, dims, model,
        hypervis_type=model_config.hypervis_opts.variable_resolution,
    )
    diffusion_config["nu_top"] = 0.0  # disable top-of-atmosphere sponge

    # Generate initial conditions
    test_config  = ullrich_baroclinic_wave.init_baroclinic_wave_config(
        model_config=physics_config,
    )
    model_state = ullrich_baroclinic_wave.init_baroclinic_wave_state(
        h_grid, v_grid, physics_config, test_config, dims, model,
        mountain=False,
    )

    # initialize generator-based simulator
    simulator = init_simulator(
        h_grid, v_grid,
        physics_config, diffusion_config, timestep_config,
        dims, model,
    )

    # Time loop.
    total_time = 3600.0 * 24.0  # one day in seconds
    for t, state in simulator(model_state):
        # User can perform archiving, analysis, etc here.
        if t > total_time:
            break

The ``simulator`` object is a callable that returns ``(time, state)`` pairs.
``state["dynamics"]`` contains the prognostic variables (winds, thermodynamic
variable, mass).

Example 2 – Stretched grid, baroclinic wave with topography
------------------------------------------------------------

This example uses an element-local stretched grid that has increased
resolution over two antipodal points (``axis_dilation=[1.0, 1.5, 1.0]``).

::

    import numpy as np
    from pyses.grids import init
    from pyses.model_utils import mass_coordinate, model_config, model_info
    from pyses.initialize import ullrich_baroclinic_wave
    from pyses.simulate import init_simulator

    # Horizontal grid
    npt = 4
    nx  = 30

    h_grid, dims = init.init_stretched_grid(
        nx, npt,
        axis_dilation=np.array([1.0, 1.5, 1.0]),
        calc_smooth_tensor=True,
    )

    model = model_info.models.homme_hydrostatic
    v_grid = mass_coordinate.init_vertical_grid(hybrid_a_i, hybrid_b_i, p0, model)

    # --- model configuration ---
    physics_config, diffusion_config, timestep_config = model_config.init_default_config(
        nx, h_grid, v_grid, dims, model,
        hypervis_type=model_config.hypervis_opts.variable_resolution,
    )

    # --- initial state (mountain topography, no explicit perturbation) ---
    test_config  = ullrich_baroclinic_wave.init_baroclinic_wave_config(
        model_config=physics_config,
    )
    model_state = ullrich_baroclinic_wave.init_baroclinic_wave_state(
        h_grid, v_grid, physics_config, test_config, dims, model,
        mountain=True,
        pert_type=ullrich_baroclinic_wave.perturbation_opts.none,
    )

    # --- simulator ---
    simulator = init_simulator(
        h_grid, v_grid,
        physics_config, diffusion_config, timestep_config,
        dims, model,
    )

    # --- time loop ---
    total_time = 3600.0 * 24.0  # one day in seconds
    for t, state in simulator(model_state):
        if t > total_time:
            break

Key concepts
------------

``init_quasi_uniform_grid_grid`` vs ``init_stretched_grid``
    ``init.init_quasi_uniform_grid`` produces an equiangular cubed-sphere with
    uniform element sizes.  ``init.init_stretched_grid`` allows the element
    sizes to vary across the sphere by specifying an ``axis_dilation`` vector,
    which controls the stretching along each Cartesian axis of the sphere.

``hypervis_opts``
    The hyperviscosity type must match the grid.  Use
    ``hypervis_opts.variable_resolution`` for stretched grids and
    ``hypervis_opts.quasi_uniform`` for uniform grids (both work with the
    variable-resolution option, so it is a safe default).

``perturbation_opts``
    Controls how the initial velocity perturbation is applied.  Available
    values are ``perturbation_opts.none`` (no perturbation – wave grows from
    numerical noise), ``perturbation_opts.exponential`` (DCMIP 2016 style),
    and ``perturbation_opts.streamfunction``.

Coupling user-provided physics to the non-hydrostatic core
==========================================================

Examples 1 and 2 iterate ``init_simulator`` with a plain ``for`` loop, which
advances *pure dynamics*.  The generator it returns is also a coroutine: on each
step it can be handed a *physics-forcing dict* via ``send`` to couple your own
physics — microphysics, surface fluxes, radiative or Newtonian forcing,
idealised relaxation, etc. — to the core.  (``send`` simply forwards the dict to
the underlying ``advance_coupling_step`` primitive, which the test calls
directly.)

The non-hydrostatic planar fuzzing test
``tests/slow_tests/test_random_periodic_plane.py`` is a complete worked example:
it couples idealised surface drag, a diurnal land/lake surface and Kessler
warm-rain microphysics to the non-hydrostatic HOMME core on a doubly-periodic
plane.  This section distils its coupling pattern; see the test for the full
physics implementation.

The coupling contract
---------------------

Your physics is any callable ``physics(state, t) -> physics_forcing`` (the test
implements it as a class ``_SurfacePhysics`` that caches its time-invariant
pieces, but a plain function works too).  It receives the current model-state
dict and the elapsed time in seconds, and returns a nested dict ::

    physics_forcing = {"dynamics": <dynamics-tendency dict>,
                       "tracers":  <tracers-tendency dict>}

Every entry is a **tendency** — the field's rate of change per second.  Under
``lump_all`` coupling the core applies it as an operator-split forward-Euler
add, ``field += physics_dt * tendency``, *before* taking the dynamics step.  The
two sub-dicts are assembled with the ``wrap_dynamics`` and ``wrap_tracers``
helpers so the keys match the prognostic variables of the chosen ``model``.

Enabling the coupling (``lump_all``)
------------------------------------

The forcing is only applied if the timestep config selects a coupling mode that
consumes it.  The default (``coupling_types.none``) **silently ignores** the
``physics_forcing`` argument, so the essential switch is
``physics_dynamics_coupling=coupling_types.lump_all``.  This example uses pySEs'
friendly API throughout — the same ``pyses.grids`` / ``pyses.model_utils`` /
``pyses.simulate`` facades as Examples 1 and 2 ::

    from pyses.grids import init
    from pyses.model_utils import mass_coordinate, model_config, model_info, model_state
    from pyses.simulate import init_simulator
    from pyses._config import get_backend

    jnp = get_backend().np

    # A doubly-periodic 50 km plane with the non-hydrostatic HOMME core.
    npt = 4
    nx = ny = 20
    h_grid, dims = init.init_periodic_plane_uniform(
        nx, ny, npt, length_x=50e3, length_y=50e3, calc_smooth_tensor=True,
    )
    model = model_info.models.homme_nonhydrostatic_f_plane  # non-hydrostatic, constant Coriolis

    v_grid = mass_coordinate.init_vertical_grid(hybrid_a_i, hybrid_b_i, p0, model)

    # The planar operators use a unit length scale (the grid is already in
    # metres), so set radius_earth=1.  See the test for the matching top-sponge
    # rescaling (``nu_top``) this implies.
    physics_config = model_config.init_physics_config(model, radius_earth=1.0)
    diffusion_config = model_config.init_diffusion_config(
        h_grid, v_grid, dims, physics_config, n_sponge=5,
    )

    physics_dt = 3.0  # seconds between physics updates
    timestep_config = model_config.init_timestep_config(
        physics_dt, h_grid, physics_config, diffusion_config, dims, model,
        dynamics_tstep_type=model_config.time_step_options.RK3_5STAGE_HEVI,  # HEVI integrator
        physics_dynamics_coupling=model_config.coupling_types.lump_all,      # <-- apply the forcing
    )

``RK3_5STAGE_HEVI`` is the horizontally-explicit / vertically-implicit
Runge–Kutta integrator the non-hydrostatic core uses; the hydrostatic core uses
``RK3_5STAGE`` instead.

Building the forcing dict
-------------------------

Two non-hydrostatic specifics shape the ``wrap_dynamics`` call:

* The non-hydrostatic prognostic state carries the interface geopotential
  ``phi_i`` and interface vertical velocity ``w_i`` in addition to the
  hydrostatic fields, so ``wrap_dynamics`` accepts ``phi_i=`` and ``w_i=``
  tendencies.  Pass zeros for fields you do not force.
* HOMME's thermodynamic prognostic is ``theta_v_d_mass`` (virtual potential
  temperature times layer mass), **not** temperature.  A physical temperature
  tendency ``dT`` (K/s) must therefore be converted with the diagnosed Exner
  function, ``dtheta = (d_mass / exner) * dT``.

::

    def my_physics(state, t):
        dyn = state["dynamics"]
        d_mass = dyn["d_mass"]

        # Diagnose temperature / Exner / pressure from the prognostic state.
        # (``_diagnose_state`` in the test wraps ``eval_mu`` for this.)
        T, exner, p = diagnose_state(dyn, v_grid, physics_config)

        # --- your physics: compute tendencies of each prognostic field ---
        du   = ...        # (E, i, j, lev, 2)  horizontal-wind tendency (m/s / s)
        dw_i = ...        # (E, i, j, ilev)    vertical-velocity tendency
        dT   = ...        # (E, i, j, lev)     temperature tendency (K/s)
        dm_v = ...        # water-vapour mixing-ratio tendency (1/s)
        dm_c = ...        # cloud-water tendency
        dm_r = ...        # rain-water tendency

        # temperature tendency -> theta_v_d_mass tendency
        dtheta = (d_mass / exner) * dT

        zeros_mass = jnp.zeros_like(d_mass)
        zeros_phi  = jnp.zeros_like(dyn["phi_i"])
        dyn_forcing = model_state.wrap_dynamics(
            du, dtheta, zeros_mass, model,   # wind, thermodynamic, d_mass tendencies
            phi_i=zeros_phi,                 # not forcing the geopotential here
            w_i=dw_i,                        # non-hydrostatic vertical-velocity forcing
        )

        # Water vapour lives in ``moisture_species``; extra tracers (cloud,
        # rain, ...) in ``tracers``.  Both must be declared on the state.
        trac_forcing = model_state.wrap_tracers(
            {"water_vapor": dm_v},
            {"cloud_water": dm_c, "rain_water": dm_r},
            model,
        )
        return {"dynamics": dyn_forcing, "tracers": trac_forcing}

The driver loop
---------------

Build the simulator exactly as in Examples 1 and 2, then drive its generator as
a coroutine: prime it, and ``send`` the forcing recomputed from the freshly
yielded state on every step ::

    simulator = init_simulator(
        h_grid, v_grid, physics_config, diffusion_config, timestep_config,
        dims, model,
    )

    state = ...   # balanced non-hydrostatic IC, e.g. via
                  # pyses.initialize.custom_init.init_analytic_state (see the test)

    # Priming the generator takes the first step (using the forcing passed
    # here); ``send`` supplies the forcing for each step thereafter.
    sim = simulator(state, my_physics(state, 0.0))
    t, state = next(sim)
    total_time = 3600.0
    while t < total_time:
        t, state = sim.send(my_physics(state, t))
        # archiving / analysis here

Because the forcing is recomputed from the freshly yielded ``state`` every step,
state-dependent physics (saturation adjustment, surface fluxes that depend on
the current near-surface temperature, etc.) closes the loop with the dynamics
naturally.

If you do not need the ``send`` protocol you can equally call the lower-level
``advance_coupling_step(..., physics_forcing=forcing)`` in your own loop — this
is what ``test_random_periodic_plane.py`` does.

Coupling concepts
-----------------

``physics_forcing`` dict
    ``{"dynamics": ..., "tracers": ...}`` of **tendencies** (field per second).
    Built with ``wrap_dynamics`` / ``wrap_tracers`` so keys match the model's
    prognostic variables.  Recompute it from the live ``state`` each step.

``coupling_types.lump_all``
    Applies the forcing as an operator-split forward-Euler add to *both*
    dynamics and tracers before each dynamics step.  Required — the default
    ``coupling_types.none`` drops the ``physics_forcing`` argument silently.

``wrap_dynamics(..., phi_i=, w_i=)``
    The non-hydrostatic core adds the interface geopotential ``phi_i`` and
    vertical velocity ``w_i`` to the prognostic set; supply their tendencies
    (or zeros).  The hydrostatic core omits both.

``theta_v_d_mass`` thermodynamic variable
    HOMME evolves virtual potential temperature times layer mass.  Convert a
    temperature tendency with ``dtheta = (d_mass / exner) * dT`` using the
    Exner function diagnosed from the state.