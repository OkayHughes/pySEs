"""Approximation-of-identity utilities for gradient remediation.

Category-C/D machinery from docs/ad_hardening_strategy.md §3: exact,
non-smooth forward models whose *derivatives* are taken from a smoothed
surrogate, plus the smooth primitives the surrogates are built from.
Everything here is opt-in — no default forward-model path may call these.
"""
from ._config import get_backend as _get_backend

_be = _get_backend()
jnp = _be.np


def stable_sigmoid(x):
  """Logistic sigmoid via tanh, overflow-free for large ``|x|``."""
  return 0.5 * (1.0 + jnp.tanh(0.5 * x))


def straight_through(exact, surrogate):
  """Value of ``exact``, derivative of ``surrogate``.

  The classic straight-through estimator, written backend-agnostically as
  ``surrogate + stop_gradient(exact - surrogate)``. The primal equals
  ``exact`` up to one floating-point rounding of the surrogate correction
  (exactly, when the surrogate is formed as ``exact + correction`` and the
  correction cancels); on backends without AD it degrades to the same
  ulp-level identity. Use when a piecewise-constant choice (integer
  search, hard mask) makes the exact derivative blind to a switch the
  optimization needs to feel.
  """
  return surrogate + _be.stop_gradient(exact - surrogate)


def _expand(m, axes, ndim):
  """``m`` reduced over ``axes`` of an ``ndim`` array, re-broadcastable to it
  (``None``-indexing; the torch shim's reductions take no ``keepdims``)."""
  return m[tuple(None if i in axes else slice(None) for i in range(ndim))]


def soft_reduce_max(x, axis, tau):
  """Value: exact ``max`` over ``axis``. Derivative: softmax-weighted mean.

  The weights ``exp((x - max) / tau)`` split the derivative evenly over
  values within ``tau`` of the maximum, so near-ties (which a 1-ulp change
  can make or break) share it smoothly instead of switching the subgradient.
  """
  axes = tuple(a % x.ndim for a in (axis if isinstance(axis, tuple) else (axis,)))
  m = jnp.max(x, axis=axis)
  m_const = _be.stop_gradient(m)
  # Shifted form: the softmax mean is m + sum(w (x - m)) / sum(w). The
  # differences x - m are exact (zero at ties); written as sum(w x) / sum(w),
  # the derivative would carry (x - mean) / tau with x - mean pure rounding
  # at ties, amplified by 1 / tau, and jvp and vjp would round it differently.
  delta = x - _expand(m_const, axes, x.ndim)
  weight = jnp.exp(delta / tau)
  surrogate = m_const + jnp.sum(weight * delta, axis=axis) / jnp.sum(weight, axis=axis)
  return straight_through(m, surrogate)


def soft_reduce_min(x, axis, tau):
  """Value: exact ``min`` over ``axis``; derivative: softmin-weighted mean."""
  return -soft_reduce_max(-x, axis, tau)
