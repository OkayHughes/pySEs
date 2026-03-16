import numpy as np
from ..._config import get_backend as _get_backend
_be = _get_backend()
jnp = _be.np
jit = _be.jit


@jit
def eval_geopotential(dp,
                      phi_surf):
  """
  Compute the hydrostatically balanced mid-level geopotential for shallow water model.

  Integrates the hydrostatic equation using a constant density approximation, namely
  rho dphi = - dp, and we assume rho = 1 so that we can treat pressure and geopotential 
  interchangeably.

  Parameters
  ----------
  dp : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Moist layer pressure thickness (Pa) from :func:`eval_d_pressure`.
  phi_surf : Array[tuple[elem_idx, gll_idx, gll_idx], Float]
      Surface geopotential (m^2 s^-2).

  Returns
  -------
  phi_m : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Hydrostatically balanced mid-level geopotential (m^2 s^-2).
  """
  # note: here we are assuming rho_0 = 1 
  d_phi = dp 
  phi_i = jnp.cumsum(jnp.flip(d_phi, axis=-1), axis=-1) + phi_surf[:, :, :, np.newaxis]
  phi_i = jnp.flip(phi_i, axis=-1)
  return phi_i
