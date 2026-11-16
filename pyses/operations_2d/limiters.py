from .._config import get_backend as _get_backend
_be = _get_backend()
jnp = _be.np


def _default_tau_A(smooth_tau, mass_matrix, d_mass):
  # A (the redistributed mass) has units of q times scaled mass, so its
  # switch width is tau times a typical scaled mass.
  return smooth_tau * jnp.mean(mass_matrix) * jnp.mean(d_mass)


def clip_and_sum_limiter(tracer_mass_tend, mass_matrix, tracer_min, tracer_max, d_mass,
                         smooth_tau=None, smooth_tau_A=None):
  """
  Apply a clip-and-redistribute limiter to tracer mass tendencies.

  With ``smooth_tau=None`` (default) this is exactly
  :func:`_clip_and_sum_exact`. With ``smooth_tau`` set (mixing-ratio units),
  the value is still the exact limiter's (up to one rounding), but the
  derivative is taken from :func:`_clip_and_sum_surrogate`, whose clip and
  redistribution switches are logistic functions of width ``smooth_tau``
  (``smooth_tau_A`` for the redistribution direction; default
  ``smooth_tau`` times a typical scaled mass). This is a category-C
  straight-through remedy (docs/ad_hardening_strategy.md §3): the exact
  limiter's tangent switches at equalities the model manufactures, so it is
  only defined to roundoff-selected one-sided derivatives. Both arguments
  must be static under jit.

  Parameters and returns are those of :func:`_clip_and_sum_exact`.
  """
  exact = _clip_and_sum_exact(tracer_mass_tend, mass_matrix, tracer_min, tracer_max, d_mass)
  if smooth_tau is None:
    return exact
  from ..smoothing import straight_through
  tau_A = (smooth_tau_A if smooth_tau_A is not None
           else _default_tau_A(smooth_tau, mass_matrix, d_mass))
  surrogate = _clip_and_sum_surrogate(tracer_mass_tend, mass_matrix, tracer_min, tracer_max,
                                      d_mass, smooth_tau, tau_A)
  return straight_through(exact, surrogate)


def _clip_and_sum_surrogate(tracer_mass_tend, mass_matrix, tracer_min, tracer_max, d_mass,
                            tau, tau_A):
  """
  Smooth-mask twin of :func:`_clip_and_sum_exact`, used only on the tangent
  path of :func:`clip_and_sum_limiter` (never with the flag off).

  The clip decisions become ``over = sigmoid((q - qmax) / tau)`` and
  ``under = sigmoid((qmin - q) / tau)``; the redistribution direction
  becomes ``s = sigmoid(A / tau_A)``, blending the two receiver patterns.
  The removed mass ``A`` is defined as the sum of what the smoothed clip
  removed and the clipped tracer is derived from it, and the same
  adjustment pattern appears in the numerator and the denominator of the
  redistribution, so element-level mass is conserved algebraically,
  whatever the masks; hence so is the tangent. The bound relaxation (to the
  element mean) is left exact, and the no-receiver guard stays hard.
  """
  from ..smoothing import stable_sigmoid as sig
  c = mass_matrix[:, :, :, jnp.newaxis] * d_mass                  # scaled mass
  q = tracer_mass_tend / d_mass
  csum = jnp.sum(c, axis=(1, 2))
  qsum = jnp.sum(q * c, axis=(1, 2))
  # Bound relaxation exactly as in the exact limiter (generic kink, left exact).
  qmin = jnp.where(qsum < tracer_min * csum, qsum / csum, tracer_min)[:, jnp.newaxis, jnp.newaxis, :]
  qmax = jnp.where(qsum > tracer_max * csum, qsum / csum, tracer_max)[:, jnp.newaxis, jnp.newaxis, :]

  # Smoothed clip: the signed mass removed per cell first, the clipped tracer from it.
  over = sig((q - qmax) / tau)
  under = sig((qmin - q) / tau)
  add = (over * (q - qmax) + under * (q - qmin)) * c
  q1 = q - add / c                     # == (1 - over - under) q + over qmax + under qmin
  A = jnp.sum(add, axis=(1, 2))        # exactly the mass removed, per element-level

  # Smoothed redistribution direction: receivers' room above (A > 0) or below (A < 0).
  s = sig(A / tau_A)[:, jnp.newaxis, jnp.newaxis, :]
  adj = s * (qmax - q1) + (1.0 - s) * (q1 - qmin)   # at A ~ 0: (qmax - qmin) / 2, uniform
  D = jnp.sum(adj * c, axis=(1, 2))
  thresh = 1e-12 * csum
  ok = (D > thresh)[:, jnp.newaxis, jnp.newaxis, :]
  safe_D = jnp.where(D > thresh, D, 1.0)[:, jnp.newaxis, jnp.newaxis, :]
  q2 = jnp.where(ok, q1 + (A[:, jnp.newaxis, jnp.newaxis, :] / safe_D) * adj, q1)
  return q2 * d_mass


def _clip_and_sum_exact(tracer_mass_tend, mass_matrix, tracer_min, tracer_max, d_mass):
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
  thresh = 1e-12 * sum_scaled_mass
  do_mass_adjustment = jnp.logical_and(modified, denominator > thresh)
  # The careful treatment of threshold here does not affect normal operation,
  # but prevents NaN poisoning in the adjoint.
  safe_denominator = jnp.where(denominator > thresh, denominator, 1.0)
  tracer = jnp.where(do_mass_adjustment[:, jnp.newaxis, jnp.newaxis, :],
                     tracer + (add_mass_per_lev / safe_denominator)[:, jnp.newaxis, jnp.newaxis, :] * tracer_adjustment,
                     tracer)
  tracer_mass_tend_out = tracer * d_mass
  return tracer_mass_tend_out


def full_limiter(tracer_mass_tend, mass_matrix, tracer_min, tracer_max, d_mass, tol_limiter=1e-10,
                 smooth_tau=None, smooth_tau_A=None):
  """
  Apply an iterative mass-conservative limiter to tracer mass tendencies.

  ``smooth_tau``/``smooth_tau_A`` as in :func:`clip_and_sum_limiter`: value of
  :func:`_full_limiter_exact`, derivative of :func:`_full_limiter_surrogate`.
  Other parameters and returns are those of :func:`_full_limiter_exact`.
  """
  exact = _full_limiter_exact(tracer_mass_tend, mass_matrix, tracer_min, tracer_max, d_mass,
                              tol_limiter=tol_limiter)
  if smooth_tau is None:
    return exact
  from ..smoothing import straight_through
  tau_A = (smooth_tau_A if smooth_tau_A is not None
           else _default_tau_A(smooth_tau, mass_matrix, d_mass))
  surrogate = _full_limiter_surrogate(tracer_mass_tend, mass_matrix, tracer_min, tracer_max,
                                      d_mass, smooth_tau, tau_A)
  return straight_through(exact, surrogate)


def _full_limiter_surrogate(tracer_mass_tend, mass_matrix, tracer_min, tracer_max, d_mass,
                            tau, tau_A):
  """
  Smooth-mask twin of :func:`_full_limiter_exact` (tangent path only).

  Each of the ``npt^2 - 1`` iterations applies the smoothed clip of
  :func:`_clip_and_sum_surrogate`; the receivers of the redistributed mass
  (cells strictly inside the bound, uniform increment in q) get the smooth
  weights ``sigmoid((qmax - q) / tau)`` (mass added) and
  ``sigmoid((q - qmin) / tau)`` (mass removed), blended by
  ``sigmoid(A / tau_A)``. Each branch is normalized by its own weight sum,
  so element-level mass is conserved whatever the masks. A cell left exactly
  on a bound by an earlier iteration gets half weight here where the exact
  limiter excludes it: that equality is one the iteration manufactures.
  """
  from ..smoothing import stable_sigmoid as sig
  npt = tracer_mass_tend.shape[1]
  c = mass_matrix[:, :, :, jnp.newaxis] * d_mass
  q = tracer_mass_tend / d_mass
  csum = jnp.sum(c, axis=(1, 2))
  qsum = jnp.sum(q * c, axis=(1, 2))
  qmin = jnp.where(qsum < tracer_min * csum, qsum / csum, tracer_min)[:, jnp.newaxis, jnp.newaxis, :]
  qmax = jnp.where(qsum > tracer_max * csum, qsum / csum, tracer_max)[:, jnp.newaxis, jnp.newaxis, :]
  for iter_idx in range(npt * npt - 1):
    over = sig((q - qmax) / tau)
    under = sig((qmin - q) / tau)
    add = (over * (q - qmax) + under * (q - qmin)) * c
    q1 = q - add / c
    A = jnp.sum(add, axis=(1, 2))[:, jnp.newaxis, jnp.newaxis, :]
    s = sig(A / tau_A)
    r_up = sig((qmax - q1) / tau)
    r_down = sig((q1 - qmin) / tau)
    w_up = jnp.sum(r_up * c, axis=(1, 2))[:, jnp.newaxis, jnp.newaxis, :]
    w_down = jnp.sum(r_down * c, axis=(1, 2))[:, jnp.newaxis, jnp.newaxis, :]
    # Hard no-receiver guards, as in the exact limiter.
    zero = jnp.zeros_like(q1)
    up = jnp.where(w_up > 0.0, A * r_up / jnp.where(w_up > 0.0, w_up, 1.0), zero)
    down = jnp.where(w_down > 0.0, A * r_down / jnp.where(w_down > 0.0, w_down, 1.0), zero)
    q = q1 + s * up + (1.0 - s) * down
  return q * d_mass


def _full_limiter_exact(tracer_mass_tend, mass_matrix, tracer_min, tracer_max, d_mass,
                        tol_limiter=1e-10):
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

    # Sanitize before dividing: weight_sum == 0 for element-levels with no
    # unsaturated DOF to receive the redistributed mass; a bare divide leaves
    # inf/NaN in the masked-out entries that poisons reverse-mode AD.
    safe_weight_sum = jnp.where(weight_sum > 0.0, weight_sum, 1.0)
    tracer = jnp.where(jnp.logical_and(add_mask, not_overshoot_mask),
                       tracer + (add_mass_per_lev / safe_weight_sum)[:, jnp.newaxis, jnp.newaxis, :],
                       tracer)
    not_add_mask = jnp.logical_not(add_mask)
    weight_sum = jnp.sum(jnp.where(jnp.logical_and(not_add_mask, not_undershoot_mask),
                                   scaled_mass,
                                   jnp.zeros_like(tracer)),
                         axis=(1, 2))
    safe_weight_sum = jnp.where(weight_sum > 0.0, weight_sum, 1.0)
    tracer = jnp.where(jnp.logical_and(not_add_mask, not_undershoot_mask),
                       tracer + (add_mass_per_lev / safe_weight_sum)[:, jnp.newaxis, jnp.newaxis, :],
                       tracer)
  tracer_mass_tend_out = tracer * d_mass
  return tracer_mass_tend_out
