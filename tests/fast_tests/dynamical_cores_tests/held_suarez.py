from pyses._config import get_backend as _get_backend
_be = _get_backend()
jnp = _be.np
jit = _be.jit

sigma_b = 0.70
secpday = 86400
k_a = 1.0 / 40.0
k_f = 1.0 / (1.0 * secpday)
k_s = 1.0 / (4.0 * secpday)
dtheta_z = 10.0
dT_y = 60.0


@jit
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


@jit
def hs_u(u, v_grid):
  etam = v_grid["hybrid_a_m"] + v_grid["hybrid_b_m"]
  k_v = k_f * jnp.maximum(0.0, (etam - sigma_b) / (1.0 - sigma_b))
  print(u.shape)
  hs_v_frc = jnp.stack((-k_v[jnp.newaxis, jnp.newaxis, jnp.newaxis, :] * u[:, :, :, :, 0],
                        -k_v[jnp.newaxis, jnp.newaxis, jnp.newaxis, :] * u[:, :, :, :, 1]),
                       axis=-1)
  print(hs_v_frc.shape)
  return hs_v_frc
