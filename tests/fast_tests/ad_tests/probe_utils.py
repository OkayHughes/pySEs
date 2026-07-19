"""Reusable probes for kernel-level gradient stress tests.

Layer 1 of the AD-hardening harness (docs/ad_hardening_strategy.md §5):
given a kernel closed over its static configuration and a tuple of array
primals, these probes check

* ``probe_forward_reverse`` — the JVP and VJP are finite and satisfy the
  forward/reverse dot-product identity ``<w, J v> == <J^T w, v>``;
* ``probe_fd_directional`` — the JVP matches a central finite difference
  along the same tangent direction.

The FD probe compares AD against FD *inside* the branch selected by the
primal point: for piecewise-defined kernels both sides differentiate the
same branch unless the perturbation crosses a switching surface, so
agreement here does NOT rule out category-B/C hazards at the switches —
those need dedicated probes constructed at the switching surfaces.
"""
import numpy as np

from pyses._config import get_backend as _get_backend
from ...context import to_host

_be = _get_backend()
jnp = _be.np


def _leaves(tree):
  if isinstance(tree, dict):
    return [leaf for key in sorted(tree) for leaf in _leaves(tree[key])]
  if isinstance(tree, (tuple, list)):
    return [leaf for item in tree for leaf in _leaves(item)]
  return [tree]


def _map_structure(fn, tree):
  if isinstance(tree, dict):
    return {key: _map_structure(fn, val) for key, val in tree.items()}
  if isinstance(tree, tuple):
    return tuple(_map_structure(fn, val) for val in tree)
  if isinstance(tree, list):
    return [_map_structure(fn, val) for val in tree]
  return fn(tree)


def assert_all_finite(tree, what):
  for leaf in _leaves(tree):
    host = np.asarray(to_host(leaf))
    n_bad = int(np.count_nonzero(~np.isfinite(host)))
    assert n_bad == 0, f"{what}: {n_bad}/{host.size} non-finite values"


def scaled_tangents_like(primals, seed):
  """Deterministic pseudo-random tangents scaled by local magnitude.

  Scaling each component by ``|p| + 1`` keeps directional finite
  differences well conditioned when primals span many orders of magnitude
  (Pa-scale layer masses next to O(1) mixing ratios).
  """
  rng = np.random.default_rng(seed)
  return tuple(
      _be.array(rng.standard_normal(np.shape(to_host(p))) *
                (np.abs(np.asarray(to_host(p), dtype=float)) + 1.0))
      for p in primals)


def random_cotangent_like(out_tree, seed):
  rng = np.random.default_rng(seed)
  return _map_structure(
      lambda leaf: _be.array(rng.standard_normal(np.shape(to_host(leaf)))),
      out_tree)


def tree_dot(a, b):
  return sum(float(np.vdot(np.asarray(to_host(x), dtype=float),
                           np.asarray(to_host(y), dtype=float)))
             for x, y in zip(_leaves(a), _leaves(b)))


def probe_forward_reverse(fn, primals, seed=0, what="kernel", rtol=1e-9):
  """Assert finite JVP/VJP and the forward/reverse dot-product identity.

  Returns ``(out, jvp_tangent, vjp_cotangents)`` so callers can make
  further kernel-specific assertions.
  """
  tangents = scaled_tangents_like(primals, seed)
  out, tangent_out = _be.jvp(fn, primals, tangents)
  assert_all_finite(out, f"{what} primal output")
  assert_all_finite(tangent_out, f"{what} JVP tangent")

  _, vjp_fn = _be.vjp(fn, *primals)
  cotangent = random_cotangent_like(out, seed + 1)
  primal_cotangents = vjp_fn(cotangent)
  assert_all_finite(primal_cotangents, f"{what} VJP cotangent")

  lhs = tree_dot(cotangent, tangent_out)
  rhs = tree_dot(primal_cotangents, tangents)
  scale = max(abs(lhs), abs(rhs), 1.0)
  assert abs(lhs - rhs) <= rtol * scale, (
      f"{what}: forward/reverse mismatch <w, Jv> = {lhs!r} vs "
      f"<J^T w, v> = {rhs!r}")
  return out, tangent_out, primal_cotangents


def probe_fd_directional(fn, primals, seed=0, h=1e-7, rtol=2e-5,
                         atol_scale=1e-8, what="kernel"):
  """Assert the JVP matches central finite differences along one direction.

  ``h`` is a *relative* step because the tangents are magnitude-scaled;
  the absolute tolerance scales with the tangent magnitude so fields of
  very different units share one setting.
  """
  tangents = scaled_tangents_like(primals, seed)
  _, tangent_out = _be.jvp(fn, primals, tangents)

  hosts = [np.asarray(to_host(p), dtype=float) for p in primals]
  tans = [np.asarray(to_host(t), dtype=float) for t in tangents]
  out_plus = fn(*[_be.array(p + h * t) for p, t in zip(hosts, tans)])
  out_minus = fn(*[_be.array(p - h * t) for p, t in zip(hosts, tans)])

  for ad_leaf, plus_leaf, minus_leaf in zip(
      _leaves(tangent_out), _leaves(out_plus), _leaves(out_minus)):
    ad = np.asarray(to_host(ad_leaf), dtype=float)
    fd = (np.asarray(to_host(plus_leaf), dtype=float) -
          np.asarray(to_host(minus_leaf), dtype=float)) / (2.0 * h)
    atol = atol_scale * (1.0 + float(np.max(np.abs(ad))))
    np.testing.assert_allclose(
        ad, fd, rtol=rtol, atol=atol,
        err_msg=f"{what}: JVP disagrees with central FD")
