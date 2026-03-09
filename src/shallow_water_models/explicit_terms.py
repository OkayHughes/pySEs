import numpy as np
from .._config import get_backend as _get_backend
_be = _get_backend()
jit = _be.jit
jnp = _be.np
from ..operations_2d.operators import horizontal_vorticity, horizontal_gradient, horizontal_divergence
from .model_state import wrap_model_state
from functools import partial

@jit
def eval_explicit_terms(state_in,
                        grid,
                        physics_config):
  """
  Calculate the explicit right-hand-side of the rotating shallow water equations on a sphere.

  Parameters
  ----------
  state_in : `ShallowWaterModelState`
      the 1st param name `first`
  grid : `SpectralElementGrid`
      Spectral element grid struct that contains coordinate and metric data.
  physics_config : dict[str, float]
      Struct containing physical constants for the sphere on which simulation is performed.

  Notes
  -----
  These solve the equations
  ∂𝐮/∂t = −(ζ + f_α)𝐤×𝐮 − ∇(g(h+h_s) + (𝐮⋅𝐮)/2)
  ∂h/∂t = -∇⋅(h𝐮),
  where the planet's axis of rotation can be changed by setting
  f_α = 2Ω (−cos(λ)cos(ϕ)sin(α) + sin(ϕ)cos(alpha))

  Returns
  -------
  right_hand_side : `ShallowWaterModelState`
      The explicit time tendency, not scaled by ∆t.

  """
  h = state_in["h"]

  coriolis = (-jnp.cos(grid["physical_coords"][:, :, :, 1]) *
              jnp.cos(grid["physical_coords"][:, :, :, 0]) *
              jnp.sin(physics_config["alpha"]) +
              jnp.sin(grid["physical_coords"][:, :, :, 0]) *
              jnp.cos(physics_config["alpha"]))
  abs_vort = horizontal_vorticity(state_in["horizontal_wind"], grid, a=physics_config["radius_earth"])
  abs_vort += 2 * physics_config["angular_freq_earth"] * coriolis
  energy = 0.5 * (state_in["horizontal_wind"][:, :, :, 0]**2 +
                  state_in["horizontal_wind"][:, :, :, 1]**2) + physics_config["gravity"] * (h + state_in["hs"])
  energy_grad = horizontal_gradient(energy, grid, a=physics_config["radius_earth"])
  u_tend = abs_vort * state_in["horizontal_wind"][:, :, :, 1] - energy_grad[:, :, :, 0]
  v_tend = -abs_vort * state_in["horizontal_wind"][:, :, :, 0] - energy_grad[:, :, :, 1]
  h_tend = -horizontal_divergence(h[:, :, :, np.newaxis] * state_in["horizontal_wind"],
                                  grid,
                                  a=physics_config["radius_earth"])

  return wrap_model_state(jnp.stack((u_tend, v_tend), axis=-1),
                          h_tend,
                          state_in["hs"])
