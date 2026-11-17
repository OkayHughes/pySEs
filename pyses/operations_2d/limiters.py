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
  redistribution switches are softplus-smoothed over ``smooth_tau``
  (``smooth_tau_A`` for the redistribution direction; default
  ``smooth_tau`` times a typical scaled mass), every piece monotone with
  derivative in [0, 1] so the tangent stays non-expansive. This is a category-C
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


def _softplus(x, tau):
  """tau * log(1 + exp(x / tau)), overflow-free: -> x for x >> tau, -> tau exp(x / tau)
  for x << -tau. Monotone and convex, derivative sigmoid(x / tau) in (0, 1)."""
  return tau * jnp.logaddexp(jnp.zeros_like(x), x / tau)


def _relaxed_bounds(q, c, tracer_min, tracer_max):
  """Scaled-mass sums and the bounds relaxed to the element mean, exactly as in
  the exact limiter (a generic kink, left exact in the surrogate)."""
  csum = jnp.sum(c, axis=(1, 2))
  qsum = jnp.sum(q * c, axis=(1, 2))
  qmin = jnp.where(qsum < tracer_min * csum, qsum / csum, tracer_min)[:, jnp.newaxis, jnp.newaxis, :]
  qmax = jnp.where(qsum > tracer_max * csum, qsum / csum, tracer_max)[:, jnp.newaxis, jnp.newaxis, :]
  return csum, qmin, qmax


def _surrogate_pass(q, c, csum, qmin, qmax, tau, tau_A):
  """One smoothed clip-and-redistribute pass (v2.1: softplus throughout).

  Every smoothed piece is monotone with derivative in [0, 1], like the exact
  clip, so the surrogate tangent is non-expansive and any number of limiter
  applications composes to something bounded. (v1 wrote the clip as
  ``sigmoid((q - qmax) / tau) * (q - qmax)``, a swish whose derivative
  ranges over [-0.10, 1.10]: it expanded by up to 10% per application and
  exploded over long windows. Mask times value is a swish; use softplus.)
  See docs/limiter_surrogate.md for the derivation.

  * clip: excess = softplus(q - qmax), deficit = softplus(qmin - q),
    q1 = q - excess + deficit, so dq1/dq = 1 - s1 - s2 in [0, 1);
  * redistribution direction: A = A_plus - A_minus exactly, with
    A_plus = softplus(A, tau_A), A_minus = softplus(-A, tau_A), each sent
    through its own channel;
  * receiver weights: the room to each bound, floored smoothly,
    room_up = softplus(qmax - q1), room_dn = softplus(q1 - qmin). The floor
    keeps the channel denominators D >= ~ tau ln2 sum(c) on plateaus, where
    the exact rooms (and v2's D) are roundoff and the leaked softplus mass in
    A_plus, A_minus would otherwise be divided by ~0. No guard is needed, and
    mass is conserved exactly: each channel's weights sum to one. The price:
    a clipped cell sits exactly at room = 0 (the receive / do-not-receive
    switch) and gets weight ~ tau ln2, so far from every other switch the
    tangent differs from the exact one by O(tau / room) (~1e-8 at
    tau = 1e-8).

  A comes from ``add``, q1 from ``excess``/``deficit``, and both channels from
  the ``room_*`` used in ``D_*``: conservation rests on these couplings.
  """
  excess = _softplus(q - qmax, tau)                    # >= 0, ~ relu(q - qmax)
  deficit = _softplus(qmin - q, tau)                   # >= 0, ~ relu(qmin - q)
  add = (excess - deficit) * c                         # signed mass removed per cell
  q1 = q - excess + deficit
  A = jnp.sum(add, axis=(1, 2))
  A_plus = _softplus(A, tau_A)                         # softplus(x) - softplus(-x) == x
  A_minus = _softplus(-A, tau_A)
  room_up = _softplus(qmax - q1, tau)                  # receivers of positive excess
  room_dn = _softplus(q1 - qmin, tau)                  # receivers of a deficit
  D_plus = jnp.sum(room_up * c, axis=(1, 2))           # >= ~ tau ln2 csum > 0
  D_minus = jnp.sum(room_dn * c, axis=(1, 2))
  w_plus = (A_plus / D_plus)[:, jnp.newaxis, jnp.newaxis, :]
  w_minus = (A_minus / D_minus)[:, jnp.newaxis, jnp.newaxis, :]
  return q1 + w_plus * room_up - w_minus * room_dn


def _clip_and_sum_surrogate(tracer_mass_tend, mass_matrix, tracer_min, tracer_max, d_mass,
                            tau, tau_A):
  """
  Smooth twin of :func:`_clip_and_sum_exact`, used only on the tangent path
  of :func:`clip_and_sum_limiter` (never with the flag off): one
  :func:`_surrogate_pass`. Within ``tau`` of a bound (``tau_A`` of A = 0)
  the clip and redistribution-direction switches are softplus-smoothed;
  farther away the surrogate tangent equals the exact one.
  """
  c = mass_matrix[:, :, :, jnp.newaxis] * d_mass                  # scaled mass
  q = tracer_mass_tend / d_mass
  csum, qmin, qmax = _relaxed_bounds(q, c, tracer_min, tracer_max)
  return _surrogate_pass(q, c, csum, qmin, qmax, tau, tau_A) * d_mass


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
  Smooth twin of :func:`_full_limiter_exact` (tangent path only): the
  monotone :func:`_surrogate_pass` iterated ``npt^2 - 1`` times against the
  same relaxed bounds. Each pass is non-expansive, so the composition is.
  (Its redistribution follows the clip-and-sum pattern, room-weighted,
  rather than the full limiter's uniform increment.)
  """
  npt = tracer_mass_tend.shape[1]
  c = mass_matrix[:, :, :, jnp.newaxis] * d_mass
  q = tracer_mass_tend / d_mass
  csum, qmin, qmax = _relaxed_bounds(q, c, tracer_min, tracer_max)
  for iter_idx in range(npt * npt - 1):
    q = _surrogate_pass(q, c, csum, qmin, qmax, tau, tau_A)
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
