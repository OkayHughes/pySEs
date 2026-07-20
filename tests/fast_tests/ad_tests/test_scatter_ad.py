"""Gradient probes for the backend scatter ops (index_add / index_max).

On torch, ``index_max`` is a custom autograd.Function
(``_make_scatter_amax`` in pyses/_config.py): PyTorch's native
``index_reduce_`` has no forward-mode rule and its backward saves its own
output, which broke both AD modes through the minmax DSS (increment-9
findings). These probes pin the replacement's contract on every backend:
finiteness, the forward/reverse dot-product identity (including at exact
ties, where derivatives split evenly among winners), central-FD agreement
away from ties, and hand-computed winner gradients — the last of which
makes cross-backend parity checkable by construction, since jax runs the
same expected values through its native ``.at[].max``.
"""
import numpy as np
import pytest

from pyses._config import get_backend as _get_backend

from ...context import to_host
from .probe_utils import probe_fd_directional, probe_forward_reverse

_be = _get_backend()
jnp = _be.np
device_wrapper = _be.array

has_ad = _be.wrapper_type in ("jax", "torch")
requires_ad = pytest.mark.skipif(not has_ad,
                                 reason="backend has no AD support")

# Slot 2 and 4 are untouched by the scatter; slot 1 is won by arr; slots
# 0 and 3 are won by vals entries. No exact ties.
IDX_HOST = np.array([0, 0, 1, 3, 3, 3])
ARR_HOST = np.array([0.5, 2.5, 3.5, 1.0, -1.0])
VALS_HOST = np.array([1.2, 0.7, 2.0, 4.0, 3.0, 3.9])


def _idx():
  return (jnp.asarray(IDX_HOST),)


def _max_fn(arr, vals):
  return _be.index_max(arr, _idx(), vals)


def _add_fn(arr, vals):
  return _be.index_add(arr, _idx(), vals)


def _primals():
  return device_wrapper(ARR_HOST), device_wrapper(VALS_HOST)


def test_index_max_primal():
  out = to_host(_max_fn(*_primals()))
  np.testing.assert_array_equal(out, np.array([1.2, 2.5, 3.5, 4.0, -1.0]))


@requires_ad
def test_index_max_forward_reverse():
  probe_forward_reverse(_max_fn, _primals(), what="index_max")


@requires_ad
def test_index_max_fd():
  probe_fd_directional(_max_fn, _primals(), what="index_max")


@requires_ad
def test_index_max_winner_gradients():
  # Winners: slot0 <- vals[0], slot1 <- arr[1], slot2 <- arr[2] (untouched),
  # slot3 <- vals[3], slot4 <- arr[4] (untouched).
  arr, vals = _primals()
  cot = device_wrapper(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
  _, vjp_fn = _be.vjp(_max_fn, arr, vals)
  grad_arr, grad_vals = vjp_fn(cot)
  np.testing.assert_array_equal(to_host(grad_arr),
                                np.array([0.0, 2.0, 3.0, 0.0, 5.0]))
  np.testing.assert_array_equal(to_host(grad_vals),
                                np.array([1.0, 0.0, 0.0, 4.0, 0.0, 0.0]))

  d_arr = device_wrapper(np.array([10.0, 20.0, 30.0, 40.0, 50.0]))
  d_vals = device_wrapper(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
  _, tangent = _be.jvp(_max_fn, (arr, vals), (d_arr, d_vals))
  # allclose, not array_equal: jax's native scatter-max jvp carries
  # last-ulp noise (measured 4e-15 absolute) from its select formulation.
  np.testing.assert_allclose(to_host(tangent),
                             np.array([1.0, 20.0, 30.0, 4.0, 50.0]),
                             rtol=0, atol=1e-12)


@requires_ad
def test_index_max_ties_consistent():
  # Exact ties (arr and both vals equal the winner): derivatives split
  # among winners, and forward/reverse must stay mutual transposes.
  arr = device_wrapper(np.array([1.0, 5.0]))
  vals = device_wrapper(np.array([1.0, 1.0, 4.0]))
  idx = (jnp.asarray(np.array([0, 0, 1])),)

  def fn(a, v):
    return _be.index_max(a, idx, v)

  probe_forward_reverse(fn, (arr, vals), what="index_max ties")


@requires_ad
def test_index_max_multidim_index():
  # The production DSS call site uses a tuple of index arrays over the
  # leading (elem, i, j) axes; exercise the flattening path.
  rng = np.random.default_rng(3)
  arr = device_wrapper(rng.standard_normal((2, 3)))
  i0 = jnp.asarray(np.array([0, 1, 1]))
  i1 = jnp.asarray(np.array([2, 0, 0]))
  vals = device_wrapper(np.array([5.0, -0.25, 0.5]))

  def fn(a, v):
    return _be.index_max(a, (i0, i1), v)

  expected = np.asarray(to_host(arr)).copy()
  np.maximum.at(expected, (to_host(i0), to_host(i1)), to_host(vals))
  np.testing.assert_array_equal(to_host(fn(arr, vals)), expected)
  probe_forward_reverse(fn, (arr, vals), what="index_max multidim")
  probe_fd_directional(fn, (arr, vals), what="index_max multidim")


@requires_ad
def test_index_add_forward_reverse():
  probe_forward_reverse(_add_fn, _primals(), what="index_add")
  probe_fd_directional(_add_fn, _primals(), what="index_add")


@requires_ad
def test_index_max_under_vmap():
  # The production tracer path applies the minmax DSS under _be.vmap
  # (batched field and values, shared assembly indices) — the composition
  # that needs the torch Function's hand-written vmap rule.
  rng = np.random.default_rng(7)
  arr_b_host = rng.standard_normal((3, ARR_HOST.size)) + ARR_HOST
  vals_b_host = rng.standard_normal((3, VALS_HOST.size)) + VALS_HOST

  def per_tracer(a, v):
    return _be.index_max(a, _idx(), v)

  def fn(arr_b, vals_b):
    return _be.vmap(per_tracer)(arr_b, vals_b)

  arr_b = device_wrapper(arr_b_host)
  vals_b = device_wrapper(vals_b_host)

  expected = arr_b_host.copy()
  for b in range(3):
    np.maximum.at(expected[b], IDX_HOST, vals_b_host[b])
  np.testing.assert_allclose(to_host(fn(arr_b, vals_b)), expected,
                             rtol=0, atol=0)
  probe_forward_reverse(fn, (arr_b, vals_b), what="index_max under vmap")
  probe_fd_directional(fn, (arr_b, vals_b), what="index_max under vmap")
