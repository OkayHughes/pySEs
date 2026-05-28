import numpy as np
from .._config import get_backend as _get_backend
from .model_info import moist_mixing_ratio_models
from .utils_3d import z_to_g
_be = _get_backend()
jit = _be.jit
jnp = _be.np


def init_vertical_grid(hybrid_a_i,
                       hybrid_b_i,
                       top_of_model_height,
                       model):
  """
  Build the vertical grid struct from hybrid σ-p coordinate coefficients.

  Parameters
  ----------
  hybrid_a_i : Array[tuple[nlev+1], Float]
      Pure-pressure part of the hybrid coordinate at interfaces
      (dimensionless, normalised by ``reference_surface_mass``).
  hybrid_b_i : Array[tuple[nlev+1], Float]
      Terrain-following part of the hybrid coordinate at interfaces
      (dimensionless, ranges from 1 at the surface to 0 at the model top).
  reference_surface_mass : float
      Nominal reference surface pressure / mass (Pa).
  model : model_info.models
      Model identifier; used to set ``"moist"`` or ``"dry"`` flag.

  Returns
  -------
  v_grid : dict[str, Array]
      Vertical grid struct containing ``"hybrid_a_i"``, ``"hybrid_b_i"``,
      ``"hybrid_a_m"``, ``"hybrid_b_m"``, ``"reference_surface_mass"``,
      and a moisture flag.
  """
  v_grid = {"top_of_model_height": top_of_model_height,
            "hybrid_a_i": hybrid_a_i,
            "hybrid_b_i": hybrid_b_i,
            "height": True}
  v_grid["hybrid_a_m"] = 0.5 * (hybrid_a_i[1:] + hybrid_a_i[:-1])
  v_grid["hybrid_b_m"] = 0.5 * (hybrid_b_i[1:] + hybrid_b_i[:-1])
  if model in moist_mixing_ratio_models:
    v_grid["moist"] = 1.0
  else:
    v_grid["dry"] = 1.0
  return v_grid


@jit
def surface_height_to_midlevel_height(z_surf,
                                      v_grid):
  """
  Compute mid-level pressure (mass) from surface pressure using the hybrid coordinate.

  Parameters
  ----------
  ps : Array[tuple[elem_idx, gll_idx, gll_idx], Float]
      Surface pressure (Pa).
  v_grid : dict[str, Array]
      Vertical grid from ``init_vertical_grid``.

  Returns
  -------
  p_mid : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Mid-level pressure (Pa) for each model level.
  """
  return (v_grid["top_of_model_height"] * v_grid["hybrid_a_m"][np.newaxis, np.newaxis, np.newaxis, :] +
          v_grid["hybrid_b_m"][np.newaxis, np.newaxis, np.newaxis, :] * z_surf[:, :, :, np.newaxis])

@jit
def surface_height_to_interface_height(z_surf,
                                       v_grid):
  """
  Compute mid-level pressure (mass) from surface pressure using the hybrid coordinate.

  Parameters
  ----------
  ps : Array[tuple[elem_idx, gll_idx, gll_idx], Float]
      Surface pressure (Pa).
  v_grid : dict[str, Array]
      Vertical grid from ``init_vertical_grid``.

  Returns
  -------
  p_mid : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Mid-level pressure (Pa) for each model level.
  """
  return (v_grid["top_of_model_height"] * v_grid["hybrid_a_m"][np.newaxis, np.newaxis, np.newaxis, :] +
          v_grid["hybrid_b_m"][np.newaxis, np.newaxis, np.newaxis, :] * z_surf[:, :, :, np.newaxis])
