from ...config import jnp, np, jit


@jit
def eval_sum_species(moisture_species_per_dry_mass):
  sum_species = jnp.ones_like(next(iter(moisture_species_per_dry_mass.values())))
  for species_name in moisture_species_per_dry_mass.keys():
      sum_species += (moisture_species_per_dry_mass[species_name])
  return sum_species


@jit
def eval_cp_moist(moisture_species_per_dry_mass,
                  cp_dry,
                  physics_config):
  sum_cp = 1.0 * cp_dry
  for species_name in moisture_species_per_dry_mass.keys():
    sum_cp += physics_config["moisture_species_cp"][species_name] * moisture_species_per_dry_mass[species_name]
  return sum_cp


@jit
def eval_d_pressure(d_mass,
                    moisture_species_per_dry_mass):
  dp_moist = 1.0 * d_mass
  for species_name in moisture_species_per_dry_mass.keys():
    dp_moist += d_mass * moisture_species_per_dry_mass[species_name]
  return dp_moist


@jit
def eval_surface_pressure(d_mass,
                          moisture_species_per_dry_mass,
                          p_top):
  ps = jnp.sum(eval_d_pressure(d_mass, moisture_species_per_dry_mass), axis=3) + p_top
  return ps


@jit
def eval_interface_pressure(d_pressure,
                            p_top):
  p_int_lower = p_top + jnp.cumsum(d_pressure, axis=3)
  p_int = jnp.concatenate((p_top * jnp.ones_like(d_pressure[:, :, :, 0, jnp.newaxis]),
                           p_int_lower), axis=-1)
  return p_int


@jit
def eval_midlevel_pressure(p_int):
  p_mid = (p_int[:, :, :, :-1] + p_int[:, :, :, 1:]) / 2.0
  return p_mid


@jit
def eval_virtual_temperature(temperature,
                             moisture_species_per_dry_mass,
                             sum_species,
                             R_dry,
                             physics_config):
  Rgas_total = 1.0 * R_dry
  for species_name in moisture_species_per_dry_mass.keys():
      Rgas_total += moisture_species_per_dry_mass[species_name] * physics_config["moisture_species_Rgas"][species_name]
  virtual_temperature = temperature * Rgas_total / (R_dry * sum_species)
  return virtual_temperature


@jit
def eval_Rgas_dry(dry_air_species_per_dry_mass,
                  physics_config):
  Rgas_total = jnp.zeros_like(next(iter(dry_air_species_per_dry_mass.values())))
  for species_name in dry_air_species_per_dry_mass.keys():
    Rgas_total += dry_air_species_per_dry_mass[species_name] * physics_config["dry_air_species_Rgas"][species_name]
  return Rgas_total


@jit
def eval_cp_dry(dry_air_species_per_dry_mass,
                physics_config):
  cp_total = jnp.zeros_like(next(iter(dry_air_species_per_dry_mass.values())))
  for species_name in dry_air_species_per_dry_mass.keys():
    cp_total += dry_air_species_per_dry_mass[species_name] * physics_config["dry_air_species_cp"][species_name]
  return cp_total


@jit
def eval_exner_function(midpoint_pressure,
                        R_dry,
                        cp_dry,
                        physics_config):
  return (midpoint_pressure / physics_config["p0"])**(R_dry / cp_dry)


@jit
def eval_balanced_geopotential(T_v,
                               dp,
                               p_mid,
                               R_dry,
                               phi_surf):
  d_phi = R_dry * T_v * dp / p_mid
  phi_i = jnp.cumsum(jnp.flip(d_phi, axis=-1), axis=-1) + phi_surf[:, :, :, np.newaxis]
  phi_m = jnp.flip(phi_i, axis=-1) - 0.5 * d_phi
  return phi_m
