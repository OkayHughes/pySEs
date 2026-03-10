from .._config import get_backend as _get_backend
from ..mpi.global_assembly import project_scalar_global
from ..operations_2d.local_assembly import project_scalar
from functools import partial
_be = _get_backend()
jit = _be.jit
jnp = _be.np
do_mpi_communication = _be.do_mpi_communication


def wrap_model_state(horizontal_wind,
                     h,
                     hs):
  """
  Assemble the shallow-water model state dict from its prognostic fields.

  Parameters
  ----------
  horizontal_wind : Array[tuple[elem_idx, gll_idx, gll_idx, 2], Float]
      Covariant horizontal wind ``(u, v)`` in m s^-1.
  h : Array[tuple[elem_idx, gll_idx, gll_idx], Float]
      Free-surface height (m).
  hs : Array[tuple[elem_idx, gll_idx, gll_idx], Float]
      Surface topography (m).

  Returns
  -------
  state : dict[str, Array]
      Shallow-water model state with keys ``"horizontal_wind"``, ``"h"``,
      and ``"hs"``.
  """
  return {"horizontal_wind": horizontal_wind,
          "h": h,
          "hs": hs}


@partial(jit, static_argnames=["dims"])
def project_model_state(state,
                        grid,
                        dims):
  """
  Project the shallow-water model state onto the continuous spectral element space (DSS).

  Parameters
  ----------
  state : dict[str, Array]
      Shallow-water model state, as returned by ``wrap_model_state``.
  grid : SpectralElementGrid
      Horizontal spectral element grid.
  dims : dict[str, int]
      Grid dimension parameters.

  Returns
  -------
  state_cont : dict[str, Array]
      Shallow-water model state with all fields projected to be
      globally continuous (C0).
  """
  if do_mpi_communication:
    u, v, h = project_scalar_global([state["horizontal_wind"][:, :, :, 0],
                                     state["horizontal_wind"][:, :, :, 1],
                                     state["h"][:, :, :]],
                                    grid, dims, two_d=True)
  else:
    u = project_scalar(state["horizontal_wind"][:, :, :, 0], grid, dims)
    v = project_scalar(state["horizontal_wind"][:, :, :, 1], grid, dims)
    h = project_scalar(state["h"][:, :, :], grid, dims)
  return wrap_model_state(jnp.stack((u, v), axis=-1), h, state["hs"])


@jit
def sum_avg_struct(struct_1, struct_2, coeff_1, coeff_2):
  """
  Compute a weighted sum of two tracer-consistency structs field-by-field.

  Parameters
  ----------
  struct_1 : dict[str, Array]
      First struct (e.g. a tracer-consistency accumulator).
  struct_2 : dict[str, Array]
      Second struct with the same keys as ``struct_1``.
  coeff_1 : float
      Scalar weight applied to each field of ``struct_1``.
  coeff_2 : float
      Scalar weight applied to each field of ``struct_2``.

  Returns
  -------
  struct_out : dict[str, Array]
      Field-wise weighted sum ``coeff_1 * struct_1 + coeff_2 * struct_2``.
  """
  struct_out = {}
  for field in struct_1.keys():
    struct_out[field] = struct_1[field] * coeff_1 + struct_2[field] * coeff_2
  return struct_out


@jit
def extract_average_dyn(state_in):
  """
  Compute the tracer-consistency flux ``u * h`` from a shallow-water state.

  Parameters
  ----------
  state_in : dict[str, Array]
      Shallow-water model state with keys ``"horizontal_wind"`` and ``"h"``.

  Returns
  -------
  tracer_consist : dict[str, Array]
      Dict with key ``"u_d_mass_avg"`` containing the layer-thickness-weighted
      wind ``h * u``.
  """
  out = {}
  out["u_d_mass_avg"] = state_in["h"][:, :, :, jnp.newaxis] * state_in["horizontal_wind"]
  return out


@jit
def extract_average_hypervis(state_in, state_tendency, diffusion_config):
  """
  Compute the tracer-consistency struct from a hyperviscosity tendency.

  Parameters
  ----------
  state_in : dict[str, Array]
      Shallow-water model state before the hyperviscosity step.
  state_tendency : dict[str, Array]
      Hyperviscosity tendency struct (output of ``eval_hypervis_*``).
  diffusion_config : dict[str, Any]
      Hyperviscosity configuration; must contain ``"nu_d_mass"``.

  Returns
  -------
  tracer_consist : dict[str, Array]
      Dict with keys ``"d_mass_hypervis_avg"`` (layer thickness) and
      ``"d_mass_hypervis_tend"`` (effective mass flux tendency for tracers).
  """
  out = {}
  out["d_mass_hypervis_avg"] = state_in["h"]
  nu = jnp.where(diffusion_config["nu_d_mass"] > 0.0, diffusion_config["nu_d_mass"], 1.0)
  out["d_mass_hypervis_tend"] = state_tendency["h"] / nu
  return out


@jit
def sum_state_series(states_in,
                     coeffs):
  """
  Compute a weighted linear combination of a sequence of shallow-water states.

  Parameters
  ----------
  states_in : list[dict[str, Array]]
      Sequence of shallow-water model states, each as returned by
      ``wrap_model_state``. Must have the same length as ``coeffs``.
  coeffs : list[float]
      Scalar coefficients for each state in ``states_in``.

  Returns
  -------
  state_res : dict[str, Array]
      Weighted sum of the input states; surface topography ``"hs"`` is
      taken from the first state unchanged.
  """
  state_res = wrap_model_state(states_in[0]["horizontal_wind"] * coeffs[0],
                               states_in[0]["h"] * coeffs[0],
                               states_in[0]["hs"])
  for state_idx in range(1, len(coeffs)):
    state = states_in[state_idx]
    coeff = coeffs[state_idx]
    state_res = wrap_model_state(state_res["horizontal_wind"] + state["horizontal_wind"] * coeff,
                                 state_res["h"] + state["h"] * coeff,
                                 state_res["hs"])
  return state_res
