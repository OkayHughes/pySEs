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
