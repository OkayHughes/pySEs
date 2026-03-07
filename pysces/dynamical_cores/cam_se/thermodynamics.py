from ...config import jnp, np, jit


@jit
def eval_sum_species(moisture_species_per_dry_mass):
  """
  Compute the total mixing ratio of all moisture species plus dry air.

  Initialises the sum to ``1.0`` (representing dry air) then adds each
  moisture species mixing ratio, yielding the total moist air mass per unit
  dry air mass (``1 + sum_q``).

  Parameters
  ----------
  moisture_species_per_dry_mass : dict[str, Array]
      Moisture mixing-ratio fields (kg moisture / kg dry air) keyed by
      species name.

  Returns
  -------
  sum_species : Array
      Total moist mass fraction ``1 + sum(q_i)`` with the same shape as each
      species array.
  """
  sum_species = jnp.ones_like(next(iter(moisture_species_per_dry_mass.values())))
  for species_name in moisture_species_per_dry_mass.keys():
      sum_species += (moisture_species_per_dry_mass[species_name])
  return sum_species


@jit
def eval_cp_moist(moisture_species_per_dry_mass,
                  cp_dry,
                  physics_config):
  """
  Compute the effective moist heat capacity at constant pressure.

  Adds the moisture-weighted species heat capacities to the dry-air ``cp``:
  ``cp_moist = cp_dry + sum(q_i * cp_i)``.

  Parameters
  ----------
  moisture_species_per_dry_mass : dict[str, Array]
      Moisture mixing-ratio fields keyed by species name.
  cp_dry : Array
      Dry-air heat capacity at constant pressure (J kg^-1 K^-1).
  physics_config : dict
      Physics configuration dict; must contain ``"moisture_species_cp"``
      keyed by species name.

  Returns
  -------
  sum_cp : Array
      Effective moist ``cp`` (J kg^-1 K^-1) with the same shape as ``cp_dry``.
  """
  sum_cp = 1.0 * cp_dry
  for species_name in moisture_species_per_dry_mass.keys():
    sum_cp += physics_config["moisture_species_cp"][species_name] * moisture_species_per_dry_mass[species_name]
  return sum_cp


@jit
def eval_d_pressure(d_mass,
                    moisture_species_per_dry_mass):
  """
  Compute the moist layer pressure thickness from the dry layer mass.

  Converts dry-air layer mass to total (moist) pressure thickness:
  ``dp = d_mass * (1 + sum(q_i))``.

  Parameters
  ----------
  d_mass : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Dry-air layer mass (Pa).
  moisture_species_per_dry_mass : dict[str, Array]
      Moisture mixing-ratio fields keyed by species name.

  Returns
  -------
  dp_moist : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Moist layer pressure thickness (Pa).
  """
  dp_moist = 1.0 * d_mass
  for species_name in moisture_species_per_dry_mass.keys():
    dp_moist += d_mass * moisture_species_per_dry_mass[species_name]
  return dp_moist


@jit
def eval_surface_pressure(d_mass,
                          moisture_species_per_dry_mass,
                          p_top):
  """
  Compute the surface (total) pressure by summing moist layer thicknesses.

  Calls :func:`eval_d_pressure` to get moist layer thicknesses, sums over
  all vertical levels, and adds the model-top pressure.

  Parameters
  ----------
  d_mass : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Dry-air layer mass (Pa).
  moisture_species_per_dry_mass : dict[str, Array]
      Moisture mixing-ratio fields keyed by species name.
  p_top : float
      Model-top pressure (Pa).

  Returns
  -------
  ps : Array[tuple[elem_idx, gll_idx, gll_idx], Float]
      Total surface pressure (Pa).
  """
  ps = jnp.sum(eval_d_pressure(d_mass, moisture_species_per_dry_mass), axis=3) + p_top
  return ps


@jit
def eval_interface_pressure(d_pressure,
                            p_top):
  """
  Compute interface pressures by cumulative summation of layer thicknesses.

  Prepends the model-top pressure and cumulatively sums ``d_pressure`` from
  the top downward to produce pressures at each level interface.

  Parameters
  ----------
  d_pressure : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Moist layer pressure thickness (Pa) from :func:`eval_d_pressure`.
  p_top : float
      Model-top pressure (Pa).

  Returns
  -------
  p_int : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx+1], Float]
      Interface pressures (Pa) from top to surface.
  """
  p_int_lower = p_top + jnp.cumsum(d_pressure, axis=3)
  p_int = jnp.concatenate((p_top * jnp.ones_like(d_pressure[:, :, :, 0, jnp.newaxis]),
                           p_int_lower), axis=-1)
  return p_int


@jit
def eval_midlevel_pressure(p_int):
  """
  Compute mid-level pressures as the arithmetic mean of adjacent interfaces.

  Parameters
  ----------
  p_int : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx+1], Float]
      Interface pressures (Pa) from :func:`eval_interface_pressure`.

  Returns
  -------
  p_mid : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Mid-level pressures (Pa).
  """
  p_mid = (p_int[:, :, :, :-1] + p_int[:, :, :, 1:]) / 2.0
  return p_mid


@jit
def eval_virtual_temperature(temperature,
                             moisture_species_per_dry_mass,
                             sum_species,
                             R_dry,
                             physics_config):
  """
  Compute the virtual temperature from the actual temperature and moisture.

  Uses the effective gas constant accounting for moisture species:
  ``T_v = T * R_total / (R_dry * sum_species)`` where
  ``R_total = R_dry + sum(q_i * R_i)``.

  Parameters
  ----------
  temperature : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Actual temperature (K).
  moisture_species_per_dry_mass : dict[str, Array]
      Moisture mixing-ratio fields keyed by species name.
  sum_species : Array
      Total moist mass fraction from :func:`eval_sum_species`.
  R_dry : Array
      Dry-air specific gas constant (J kg^-1 K^-1).
  physics_config : dict
      Physics configuration dict; must contain ``"moisture_species_Rgas"``.

  Returns
  -------
  virtual_temperature : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Virtual temperature (K).
  """
  Rgas_total = 1.0 * R_dry
  for species_name in moisture_species_per_dry_mass.keys():
      Rgas_total += moisture_species_per_dry_mass[species_name] * physics_config["moisture_species_Rgas"][species_name]
  virtual_temperature = temperature * Rgas_total / (R_dry * sum_species)
  return virtual_temperature


@jit
def eval_Rgas_dry(dry_air_species_per_dry_mass,
                  physics_config):
  """
  Compute the effective dry-air specific gas constant for a mixture of species.

  Computes ``R_dry = sum(mass_frac_i * R_i)`` over all dry-air species.

  Parameters
  ----------
  dry_air_species_per_dry_mass : dict[str, Array]
      Dry-air species mass-fraction fields keyed by species name.
  physics_config : dict
      Physics configuration dict; must contain ``"dry_air_species_Rgas"``
      keyed by species name.

  Returns
  -------
  Rgas_total : Array
      Effective dry-air gas constant (J kg^-1 K^-1).
  """
  Rgas_total = jnp.zeros_like(next(iter(dry_air_species_per_dry_mass.values())))
  for species_name in dry_air_species_per_dry_mass.keys():
    Rgas_total += dry_air_species_per_dry_mass[species_name] * physics_config["dry_air_species_Rgas"][species_name]
  return Rgas_total


@jit
def eval_cp_dry(dry_air_species_per_dry_mass,
                physics_config):
  """
  Compute the effective dry-air heat capacity at constant pressure for a mixture.

  Computes ``cp_dry = sum(mass_frac_i * cp_i)`` over all dry-air species.

  Parameters
  ----------
  dry_air_species_per_dry_mass : dict[str, Array]
      Dry-air species mass-fraction fields keyed by species name.
  physics_config : dict
      Physics configuration dict; must contain ``"dry_air_species_cp"``
      keyed by species name.

  Returns
  -------
  cp_total : Array
      Effective dry-air ``cp`` (J kg^-1 K^-1).
  """
  cp_total = jnp.zeros_like(next(iter(dry_air_species_per_dry_mass.values())))
  for species_name in dry_air_species_per_dry_mass.keys():
    cp_total += dry_air_species_per_dry_mass[species_name] * physics_config["dry_air_species_cp"][species_name]
  return cp_total


@jit
def eval_exner_function(midpoint_pressure,
                        R_dry,
                        cp_dry,
                        physics_config):
  """
  Compute the Exner pressure function at mid-levels.

  ``pi = (p / p0)^(R_dry / cp_dry)``

  Parameters
  ----------
  midpoint_pressure : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Mid-level pressure (Pa).
  R_dry : Array
      Dry-air specific gas constant (J kg^-1 K^-1).
  cp_dry : Array
      Dry-air heat capacity at constant pressure (J kg^-1 K^-1).
  physics_config : dict
      Physics configuration dict with key ``"p0"`` (reference pressure, Pa).

  Returns
  -------
  exner : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Dimensionless Exner function.
  """
  return (midpoint_pressure / physics_config["p0"])**(R_dry / cp_dry)


@jit
def eval_balanced_geopotential(T_v,
                               dp,
                               p_mid,
                               R_dry,
                               phi_surf):
  """
  Compute the hydrostatically balanced mid-level geopotential for CAM-SE.

  Integrates the hydrostatic equation using the ideal-gas relation:
  ``d_phi = R_dry * T_v * dp / p_mid``, then cumulatively sums from the
  surface upward.

  Parameters
  ----------
  T_v : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Virtual temperature (K).
  dp : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Moist layer pressure thickness (Pa) from :func:`eval_d_pressure`.
  p_mid : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Mid-level pressure (Pa).
  R_dry : Array
      Dry-air specific gas constant (J kg^-1 K^-1).
  phi_surf : Array[tuple[elem_idx, gll_idx, gll_idx], Float]
      Surface geopotential (m^2 s^-2).

  Returns
  -------
  phi_m : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Hydrostatically balanced mid-level geopotential (m^2 s^-2).
  """
  d_phi = R_dry * T_v * dp / p_mid
  phi_i = jnp.cumsum(jnp.flip(d_phi, axis=-1), axis=-1) + phi_surf[:, :, :, np.newaxis]
  phi_m = jnp.flip(phi_i, axis=-1) - 0.5 * d_phi
  return phi_m
