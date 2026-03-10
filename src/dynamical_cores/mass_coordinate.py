import numpy as np
from .._config import get_backend as _get_backend
from .model_info import moist_mixing_ratio_models
_be = _get_backend()
jit = _be.jit
jnp = _be.np


def init_vertical_grid(hybrid_a_i,
                       hybrid_b_i,
                       reference_surface_mass,
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
  v_grid = {"reference_surface_mass": reference_surface_mass,
            "hybrid_a_i": hybrid_a_i,
            "hybrid_b_i": hybrid_b_i}
  v_grid["hybrid_a_m"] = 0.5 * (hybrid_a_i[1:] + hybrid_a_i[:-1])
  v_grid["hybrid_b_m"] = 0.5 * (hybrid_b_i[1:] + hybrid_b_i[:-1])
  if model in moist_mixing_ratio_models:
    v_grid["moist"] = 1.0
  else:
    v_grid["dry"] = 1.0
  return v_grid


@jit
def surface_mass_to_midlevel_mass(ps,
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
  return (v_grid["reference_surface_mass"] * v_grid["hybrid_a_m"][np.newaxis, np.newaxis, np.newaxis, :] +
          v_grid["hybrid_b_m"][np.newaxis, np.newaxis, np.newaxis, :] * ps[:, :, :, np.newaxis])


@jit
def surface_mass_to_d_mass(ps,
                           v_grid):
  """
  Compute layer thickness (dp) from surface pressure using the hybrid coordinate.

  Parameters
  ----------
  ps : Array[tuple[elem_idx, gll_idx, gll_idx], Float]
      Surface pressure (Pa).
  v_grid : dict[str, Array]
      Vertical grid from ``init_vertical_grid``.

  Returns
  -------
  d_mass : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Layer thickness (Pa) at each model level.
  """
  da = (v_grid["hybrid_a_i"][np.newaxis, np.newaxis, np.newaxis, 1:] -
        v_grid["hybrid_a_i"][np.newaxis, np.newaxis, np.newaxis, :-1])
  db = (v_grid["hybrid_b_i"][np.newaxis, np.newaxis, np.newaxis, 1:] -
        v_grid["hybrid_b_i"][np.newaxis, np.newaxis, np.newaxis, :-1])
  return (v_grid["reference_surface_mass"] * da +
          db * ps[:, :, :, np.newaxis])


@jit
def surface_mass_to_interface_mass(ps,
                                   v_grid):
  """
  Compute interface-level pressure (mass) from surface pressure using the hybrid coordinate.

  Parameters
  ----------
  ps : Array[tuple[elem_idx, gll_idx, gll_idx], Float]
      Surface pressure (Pa).
  v_grid : dict[str, Array]
      Vertical grid from ``init_vertical_grid``.

  Returns
  -------
  p_int : Array[tuple[elem_idx, gll_idx, gll_idx, nlev+1], Float]
      Interface-level pressure (Pa) for each model interface.
  """
  return (v_grid["reference_surface_mass"] * v_grid["hybrid_a_i"][np.newaxis, np.newaxis, np.newaxis, :] +
          v_grid["hybrid_b_i"][np.newaxis, np.newaxis, np.newaxis, :] * ps[:, :, :, np.newaxis])


@jit
def d_mass_to_surface_mass(d_mass,
                           v_grid):
  """
  Recover surface pressure from the layer-thickness column by summing all levels.

  Parameters
  ----------
  d_mass : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Layer thickness (Pa) at each model level.
  v_grid : dict[str, Array]
      Vertical grid from ``init_vertical_grid``.

  Returns
  -------
  ps : Array[tuple[elem_idx, gll_idx, gll_idx], Float]
      Surface pressure (Pa).
  """
  p_top = v_grid["reference_surface_mass"] * v_grid["hybrid_a_i"][0]
  return jnp.sum(d_mass, axis=-1) + p_top


@jit
def eval_top_interface_mass(v_grid):
  """
  Compute the pressure at the model-top interface from the hybrid coordinate.

  Parameters
  ----------
  v_grid : dict[str, Array]
      Vertical grid from ``init_vertical_grid``; must contain
      ``"hybrid_a_i"`` and ``"reference_surface_mass"``.

  Returns
  -------
  p_top : float
      Top-of-model interface pressure (Pa).
  """
  return v_grid["reference_surface_mass"] * v_grid["hybrid_a_i"][0]
