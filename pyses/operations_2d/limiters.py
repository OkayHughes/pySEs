from .._config import get_backend as _get_backend
_be = _get_backend()
jnp = _be.np


def clip_and_sum_limiter(tracer_mass_tend, mass_matrix, tracer_min, tracer_max, d_mass):
  """
  Apply a clip-and-redistribute limiter to tracer mass tendencies.

  Clips over- and under-shooting tracer values to the element-local bounds
  ``[tracer_min, tracer_max]``, then redistributes the clipped mass to
  unsaturated DOFs within each element to conserve the element-integrated
  tracer mass.  If the element mean already falls outside the supplied
  bounds the bounds are relaxed to the element mean.

  Parameters
  ----------
  tracer_mass_tend : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Tracer mass (``q * d_mass``) after advection, to be limited.
  mass_matrix : Array[tuple[elem_idx, gll_idx, gll_idx], Float]
      GLL quadrature weights times metric determinant at each node.
  tracer_min : Array[tuple[elem_idx, lev_idx], Float]
      Lower bound for the tracer mixing ratio in each element.
  tracer_max : Array[tuple[elem_idx, lev_idx], Float]
      Upper bound for the tracer mixing ratio in each element.
  d_mass : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Dry-air layer mass (Pa) used to convert between mixing ratio and mass.

  Returns
  -------
  tracer_mass_tend_out : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Limited tracer mass with the same element-integral as the input.
  """
  # c -> scaled_mass
  # x -> tracer
  scaled_mass = mass_matrix[:, :, :, jnp.newaxis] * d_mass
  tracer = tracer_mass_tend / d_mass
  sum_scaled_mass = jnp.sum(scaled_mass, axis=(1, 2))
  sum_scaled_tracer = jnp.sum(tracer * scaled_mass, axis=(1, 2))
  tracer_min = jnp.where(sum_scaled_tracer < tracer_min * sum_scaled_mass,
                         sum_scaled_tracer / sum_scaled_mass,
                         tracer_min)
  tracer_max = jnp.where(sum_scaled_tracer > tracer_max * sum_scaled_mass,
                         sum_scaled_tracer / sum_scaled_mass,
                         tracer_max)
  add_mass = jnp.zeros_like(tracer_mass_tend)
  mask_overshoot = tracer > tracer_max[:, jnp.newaxis, jnp.newaxis, :]
  add_mass = jnp.where(mask_overshoot,
                       add_mass + (tracer - tracer_max[:, jnp.newaxis, jnp.newaxis, :]) * scaled_mass,
                       add_mass)
  tracer = jnp.where(mask_overshoot,
                     tracer_max[:, jnp.newaxis, jnp.newaxis, :],
                     tracer)
  mask_undershoot = tracer < tracer_min[:, jnp.newaxis, jnp.newaxis, :]
  add_mass = jnp.where(mask_undershoot,
                       add_mass + (tracer - tracer_min[:, jnp.newaxis, jnp.newaxis, :]) * scaled_mass,
                       add_mass)
  tracer = jnp.where(mask_undershoot,
                     tracer_min[:, jnp.newaxis, jnp.newaxis, :],
                     tracer)
  add_mass_per_lev = jnp.sum(add_mass, axis=(1, 2))
  modified = jnp.abs(add_mass_per_lev) > 0.0
  add_mask = (add_mass_per_lev > 0.0)[:, jnp.newaxis, jnp.newaxis, :]
  tracer_adjustment = jnp.where(add_mask,
                                tracer_max[:, jnp.newaxis, jnp.newaxis, :] - tracer,
                                tracer - tracer_min[:, jnp.newaxis, jnp.newaxis, :])
  tracer_adjustment = jnp.where(modified[:, jnp.newaxis, jnp.newaxis, :],
                                tracer_adjustment,
                                jnp.zeros_like(tracer))
  denominator = jnp.sum(tracer_adjustment * scaled_mass, axis=(1, 2))
  do_mass_adjustment = jnp.logical_and(modified, denominator > 0.0)
  tracer = jnp.where(do_mass_adjustment[:, jnp.newaxis, jnp.newaxis, :],
                     tracer + (add_mass_per_lev / denominator)[:, jnp.newaxis, jnp.newaxis, :] * tracer_adjustment,
                     tracer)
  tracer_mass_tend_out = tracer * d_mass
  return tracer_mass_tend_out


def full_limiter(tracer_mass_tend, mass_matrix, tracer_min, tracer_max, d_mass, tol_limiter=1e-10):
  """
  Apply an iterative mass-conservative limiter to tracer mass tendencies.

  Repeatedly clips over- and under-shooting tracer values to
  ``[tracer_min, tracer_max]`` and redistributes the surplus/deficit mass to
  unsaturated DOFs within the element, iterating ``npt^2 - 1`` times until
  convergence.  If the element mean already falls outside the supplied bounds
  the bounds are relaxed to the element mean.

  Parameters
  ----------
  tracer_mass_tend : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Tracer mass (``q * d_mass``) after advection, to be limited.
  mass_matrix : Array[tuple[elem_idx, gll_idx, gll_idx], Float]
      GLL quadrature weights times metric determinant at each node.
  tracer_min : Array[tuple[elem_idx, lev_idx], Float]
      Lower bound for the tracer mixing ratio in each element.
  tracer_max : Array[tuple[elem_idx, lev_idx], Float]
      Upper bound for the tracer mixing ratio in each element.
  d_mass : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Dry-air layer mass (Pa).
  tol_limiter : float, optional
      Unused tolerance parameter (reserved for future stopping criterion).
      Defaults to ``1e-10``.

  Returns
  -------
  tracer_mass_tend_out : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Limited tracer mass with the same element-integral as the input.
  """
  # c -> scaled_mass
  # x -> tracer
  npt = tracer_mass_tend.shape[1]
  scaled_mass = mass_matrix[:, :, :, jnp.newaxis] * d_mass
  tracer = tracer_mass_tend / d_mass
  sum_scaled_mass = jnp.sum(scaled_mass, axis=(1, 2))
  sum_scaled_tracer = jnp.sum(tracer * scaled_mass, axis=(1, 2))
  tracer_min = jnp.where(sum_scaled_tracer < tracer_min * sum_scaled_mass,
                         sum_scaled_tracer / sum_scaled_mass,
                         tracer_min)
  tracer_max = jnp.where(sum_scaled_tracer > tracer_max * sum_scaled_mass,
                         sum_scaled_tracer / sum_scaled_mass,
                         tracer_max)
  for iter_idx in range(npt * npt - 1):
    add_mass = jnp.zeros_like(tracer_mass_tend)
    mask_overshoot = tracer > tracer_max[:, jnp.newaxis, jnp.newaxis, :]
    add_mass = jnp.where(mask_overshoot,
                         add_mass + (tracer - tracer_max[:, jnp.newaxis, jnp.newaxis, :]) * scaled_mass,
                         add_mass)
    tracer = jnp.where(mask_overshoot,
                       tracer_max[:, jnp.newaxis, jnp.newaxis, :],
                       tracer)
    mask_undershoot = tracer < tracer_min[:, jnp.newaxis, jnp.newaxis, :]
    add_mass = jnp.where(mask_undershoot,
                         add_mass + (tracer - tracer_min[:, jnp.newaxis, jnp.newaxis, :]) * scaled_mass,
                         add_mass)
    tracer = jnp.where(mask_undershoot,
                       tracer_min[:, jnp.newaxis, jnp.newaxis, :],
                       tracer)
    add_mass_per_lev = jnp.sum(add_mass, axis=(1, 2))

    add_mask = (add_mass_per_lev > 0.0)[:, jnp.newaxis, jnp.newaxis, :]
    not_overshoot_mask = tracer < tracer_max[:, jnp.newaxis, jnp.newaxis, :]
    not_undershoot_mask = tracer > tracer_min[:, jnp.newaxis, jnp.newaxis, :]
    weight_sum = jnp.sum(jnp.where(jnp.logical_and(add_mask, not_overshoot_mask),
                                   scaled_mass,
                                   jnp.zeros_like(tracer)),
                         axis=(1, 2))

    tracer = jnp.where(jnp.logical_and(add_mask, not_overshoot_mask),
                       tracer + (add_mass_per_lev / weight_sum)[:, jnp.newaxis, jnp.newaxis, :],
                       tracer)
    not_add_mask = jnp.logical_not(add_mask)
    weight_sum = jnp.sum(jnp.where(jnp.logical_and(not_add_mask, not_undershoot_mask),
                                   scaled_mass,
                                   jnp.zeros_like(tracer)),
                         axis=(1, 2))
    tracer = jnp.where(jnp.logical_and(not_add_mask, not_undershoot_mask),
                       tracer + (add_mass_per_lev / weight_sum)[:, jnp.newaxis, jnp.newaxis, :],
                       tracer)
  tracer_mass_tend_out = tracer * d_mass
  return tracer_mass_tend_out
