How is the atmospheric state represented in pySEs?
==================================================

The top-level model state returned by the simulator at each time step is a
plain Python dict with three keys::

    state = {
        "dynamics":       ...,   # time-evolving prognostic fields
        "static_forcing": ...,   # time-invariant forcing
        "tracers":        ...,   # moisture and passive tracers
    }

All arrays have shape ``(n_elem, npt, npt, n_lev[, ...])`` in the horizontal
dimensions, where ``n_elem`` is the number of spectral elements, ``npt`` is
the number of GLL points used in the horizontal discretization,
and ``n_lev`` is the number of vertical levels.  Extra
trailing dimensions are documented per field.

.. contents:: Contents
   :local:
   :depth: 2

----

``state["dynamics"]`` — prognostic fields
------------------------------------------

The dynamics sub-dict holds the fields that the dynamical core integrates
forward in time.  Its keys depend on which model is selected.

Always-present fields
~~~~~~~~~~~~~~~~~~~~~

``"horizontal_wind"`` : ``(n_elem, npt, npt, n_lev, 2)``
    Contravariant horizontal wind components ``(u, v)``.

``"d_mass"`` : ``(n_elem, npt, npt, n_lev)``
    Dry-air layer mass (pressure thickness, Pa).  Summing over the level axis
    and adding the model-top pressure recovers surface pressure:

    .. code-block:: python

        ps = v_grid["hybrid_a_i"][0] * v_grid["reference_surface_mass"] \
             + state["dynamics"]["d_mass"].sum(axis=-1)

Thermodynamic variable
~~~~~~~~~~~~~~~~~~~~~~

The key used for the thermodynamic variable differs by model family because
the two dynamical cores use different formulations.

.. list-table::
   :header-rows: 1
   :widths: 35 25 40

   * - Model(s)
     - Key
     - Meaning
   * - ``homme_hydrostatic``, ``homme_nonhydrostatic``, ``homme_nonhydrostatic_deep``, f-plane variants
     - ``"theta_v_d_mass"``
     - Virtual potential temperature multiplied by layer mass: :math:`\tilde\theta_v = \theta_v \cdot \Delta m`.  Shape ``(n_elem, npt, npt, n_lev)``.
   * - ``cam_se``, ``cam_se_whole_atmosphere``
     - ``"T"``
     - Virtual temperature (K).  Shape ``(n_elem, npt, npt, n_lev)``.

To recover the unmultiplied virtual potential temperature from a HOMME state::

    theta_v = state["dynamics"]["theta_v_d_mass"] / state["dynamics"]["d_mass"]

Non-hydrostatic-only fields
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following fields are present **only** for ``homme_nonhydrostatic``,
``homme_nonhydrostatic_deep``, and ``homme_nonhydrostatic_f_plane``.  They
are defined on model *interfaces* (``n_ilev = n_lev + 1``).

``"phi_i"`` : ``(n_elem, npt, npt, n_ilev)``
    Interface geopotential (m² s⁻²).  Index 0 is the model top; index
    ``n_lev`` is the surface.

``"w_i"`` : ``(n_elem, npt, npt, n_ilev)``
    Interface vertical velocity (m s⁻¹).

----

``state["static_forcing"]`` — time-invariant forcing
------------------------------------------------------

The static-forcing sub-dict contains fields computed once at initialisation
and held fixed for the entire simulation.  It is assembled by
``pysces.initialize.custom_init.init_static_forcing``.

Always-present fields
~~~~~~~~~~~~~~~~~~~~~

``"phi_surf"`` : ``(n_elem, npt, npt)``
    Surface geopotential (m² s⁻²).  For a flat Earth this is zero everywhere.

``"grad_phi_surf"`` : ``(n_elem, npt, npt, 2)``
    DSS-projected horizontal gradient of the surface geopotential.  Used to
    impose the surface boundary condition and, in non-hydrostatic models, to
    diagnose the surface vertical velocity.

``"coriolis_param"`` : ``(n_elem, npt, npt)``
    Coriolis parameter :math:`f = 2\Omega\sin(\phi)`.

    * **Spherical models** — computed from the true latitude at each GLL node.
    * **F-plane models** — a spatially uniform constant evaluated at a
      prescribed reference latitude (default: 45° N).

Conditionally-present fields
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``"nontrad_coriolis_param"`` : ``(n_elem, npt, npt)``
    Non-traditional Coriolis parameter :math:`\tilde f = 2\Omega\cos(\phi)`.
    Present **only** for the deep-atmosphere model
    (``models.homme_nonhydrostatic_deep``).  Absent for all other models.

----

``state["tracers"]`` — moisture and passive tracers
----------------------------------------------------

The tracers sub-dict contains up to three sub-dicts of mixing-ratio fields,
plus a scalar flag that identifies the moisture convention.

Overview
~~~~~~~~

.. code-block:: python

    state["tracers"] = {
        "moisture_species":  {"water_vapor": array, ...},  # always present
        "tracers":           {"my_tracer":   array, ...},  # always present (may be empty)
        "dry_air_species":   {"N2": array, ...},            # CAM-SE whole-atmosphere only
        # exactly one of these flags is also present:
        "moist_mixing_ratio": 1.0,   # HOMME models
        "dry_mixing_ratio":   1.0,   # CAM-SE models
    }

All mixing-ratio arrays have shape ``(n_elem, npt, npt, n_lev)``.

``"moisture_species"``
~~~~~~~~~~~~~~~~~~~~~~

A dict of moisture species mixing ratios.  The default species initialised
by ``ullrich_baroclinic_wave`` is ``"water_vapor"``.  Additional moisture
species (cloud liquid, cloud ice, rain, …) can be added by the user and will
be advected alongside water vapour.

The mixing-ratio *convention* differs by model family:

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Model(s)
     - Flag key
     - Convention
   * - ``homme_*``
     - ``"moist_mixing_ratio"``
     - Moist mixing ratio: :math:`q_v = \rho_v / (\rho_v + \rho_d)`.
   * - ``cam_se``, ``cam_se_whole_atmosphere``
     - ``"dry_mixing_ratio"``
     - Dry mixing ratio: :math:`q_v = \rho_v / \rho_d`.

``"tracers"``
~~~~~~~~~~~~~

A dict of passive (or near-passive) tracer mixing ratios.  This sub-dict is
always present but is empty ``{}`` when no passive tracers have been defined.
Users can add arbitrary tracer fields to it; pySEs will advect and remap them
alongside the moisture species.

``"dry_air_species"``
~~~~~~~~~~~~~~~~~~~~~

A dict of dry-air constituent mixing ratios (e.g. N₂, O₂, CO₂).  This
sub-dict is present **only** for ``models.cam_se_whole_atmosphere``, which
uses a variable gas constant :math:`\kappa(z)` to model whole-atmosphere
thermodynamics accurately.  For HOMME models, this key is absent.

Mixing-ratio convention flag
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Exactly one of the following scalar sentinel keys is present in the tracer
dict, indicating which convention the stored mixing ratios follow:

* ``"moist_mixing_ratio": 1.0`` — HOMME model family
* ``"dry_mixing_ratio": 1.0`` — CAM-SE model family

Downstream code (e.g. the physics–dynamics coupling layer) reads this flag
to select the correct conversion when handing off water vapour between the
dynamical core and physics parameterisations. We do this by setting keys so that
model states can be passed as PyTrees in JAX.
