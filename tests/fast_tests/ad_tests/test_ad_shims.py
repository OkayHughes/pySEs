"""Smoke tests for the backend AD shims (grad/jvp/vjp/stop_gradient/checkpoint).

These exercise the differentiation interface on closed-form toy functions so
that every backend's contract is pinned before the gradient stress harness
(built on top of these shims) starts probing model kernels:

* jax and torch must agree with the analytic derivatives, satisfy the
  forward/reverse dot-product identity, handle dict pytrees (the model-state
  container type), and be checkpoint- and jit-compatible;
* numpy must raise NotImplementedError from the differentiating entry points
  while keeping identity semantics for ``stop_gradient``/``checkpoint`` so
  backend-agnostic model code can call those unconditionally.
"""
from pyses._config import get_backend as _get_backend
import numpy as np
import pytest

from ...context import to_host

_be = _get_backend()
jnp = _be.np
device_wrapper = _be.array

has_ad = _be.wrapper_type in ("jax", "torch")
requires_ad = pytest.mark.skipif(not has_ad,
                                 reason="backend has no AD support")

X_HOST = np.array([0.3, 1.1, 2.0, -0.7])
V_HOST = np.array([1.0, -0.5, 2.0, 0.25])
W_HOST = np.array([0.7, 0.2, -1.0, 1.5])


def f_scalar(x):
  return jnp.sum(jnp.sin(x) * x)


def f_scalar_grad_analytic(x_host):
  return np.sin(x_host) + x_host * np.cos(x_host)


def f_vec(x):
  return jnp.cos(x) * x**2


def f_vec_jac_diag_analytic(x_host):
  return 2.0 * x_host * np.cos(x_host) - x_host**2 * np.sin(x_host)


@requires_ad
def test_grad_matches_analytic():
  x = device_wrapper(X_HOST)
  g = _be.grad(f_scalar)(x)
  np.testing.assert_allclose(to_host(g), f_scalar_grad_analytic(X_HOST),
                             rtol=1e-12, atol=1e-12)


@requires_ad
def test_grad_argnums():
  a = device_wrapper(X_HOST)
  b = device_wrapper(V_HOST)

  def f2(a, b):
    return jnp.sum(a * b)

  gb = _be.grad(f2, argnums=1)(a, b)
  np.testing.assert_allclose(to_host(gb), X_HOST, rtol=1e-12, atol=1e-12)
  ga, gb = _be.grad(f2, argnums=(0, 1))(a, b)
  np.testing.assert_allclose(to_host(ga), V_HOST, rtol=1e-12, atol=1e-12)
  np.testing.assert_allclose(to_host(gb), X_HOST, rtol=1e-12, atol=1e-12)


@requires_ad
def test_jvp_matches_analytic():
  x = device_wrapper(X_HOST)
  v = device_wrapper(V_HOST)
  out, tangent = _be.jvp(f_vec, (x,), (v,))
  np.testing.assert_allclose(to_host(out), np.cos(X_HOST) * X_HOST**2,
                             rtol=1e-12, atol=1e-12)
  np.testing.assert_allclose(to_host(tangent),
                             f_vec_jac_diag_analytic(X_HOST) * V_HOST,
                             rtol=1e-12, atol=1e-12)


@requires_ad
def test_vjp_matches_analytic():
  x = device_wrapper(X_HOST)
  w = device_wrapper(W_HOST)
  out, vjp_fn = _be.vjp(f_vec, x)
  np.testing.assert_allclose(to_host(out), np.cos(X_HOST) * X_HOST**2,
                             rtol=1e-12, atol=1e-12)
  (cotangent,) = vjp_fn(w)
  np.testing.assert_allclose(to_host(cotangent),
                             f_vec_jac_diag_analytic(X_HOST) * W_HOST,
                             rtol=1e-12, atol=1e-12)


@requires_ad
def test_jvp_vjp_dot_product_identity():
  # <w, J v> == <J^T w, v> — the consistency check the kernel probes will
  # run on every model subsystem.
  x = device_wrapper(X_HOST)
  v = device_wrapper(V_HOST)
  w = device_wrapper(W_HOST)
  _, tangent = _be.jvp(f_vec, (x,), (v,))
  _, vjp_fn = _be.vjp(f_vec, x)
  (cotangent,) = vjp_fn(w)
  lhs = to_host(jnp.sum(w * tangent))
  rhs = to_host(jnp.sum(cotangent * v))
  np.testing.assert_allclose(lhs, rhs, rtol=1e-12, atol=1e-12)


@requires_ad
def test_grad_matches_finite_differences():
  # The finite-difference oracle pattern used by the stress harness.
  h = 1e-6
  fd = np.array([
      (float(to_host(f_scalar(device_wrapper(X_HOST + h * e)))) -
       float(to_host(f_scalar(device_wrapper(X_HOST - h * e))))) / (2 * h)
      for e in np.eye(X_HOST.size)])
  g = to_host(_be.grad(f_scalar)(device_wrapper(X_HOST)))
  np.testing.assert_allclose(g, fd, rtol=1e-6, atol=1e-8)


@requires_ad
def test_grad_dict_pytree_input():
  # Model states are dicts of arrays; the shims must differentiate through
  # that container type on both AD backends.
  state = {"a": device_wrapper(X_HOST), "b": device_wrapper(V_HOST)}

  def f_dict(state):
    return jnp.sum(state["a"]**2 * state["b"])

  g = _be.grad(f_dict)(state)
  np.testing.assert_allclose(to_host(g["a"]), 2.0 * X_HOST * V_HOST,
                             rtol=1e-12, atol=1e-12)
  np.testing.assert_allclose(to_host(g["b"]), X_HOST**2,
                             rtol=1e-12, atol=1e-12)

  _, vjp_fn = _be.vjp(f_dict, state)
  (cot,) = vjp_fn(device_wrapper(1.0))
  np.testing.assert_allclose(to_host(cot["a"]), 2.0 * X_HOST * V_HOST,
                             rtol=1e-12, atol=1e-12)


@requires_ad
def test_stop_gradient():
  x = device_wrapper(X_HOST)

  def f_sg(x):
    return jnp.sum(x * _be.stop_gradient(x))

  g = _be.grad(f_sg)(x)
  # d/dx of x * const(x) is const(x), not 2x.
  np.testing.assert_allclose(to_host(g), X_HOST, rtol=1e-12, atol=1e-12)


@requires_ad
def test_stop_gradient_pytree():
  state = {"a": device_wrapper(X_HOST), "b": device_wrapper(V_HOST)}
  frozen = _be.stop_gradient(state)
  np.testing.assert_allclose(to_host(frozen["a"]), X_HOST)
  np.testing.assert_allclose(to_host(frozen["b"]), V_HOST)


@requires_ad
def test_checkpoint_preserves_value_and_grad():
  x = device_wrapper(X_HOST)
  ck = _be.checkpoint(f_scalar)
  np.testing.assert_allclose(to_host(ck(x)), to_host(f_scalar(x)),
                             rtol=1e-12, atol=1e-12)
  g = _be.grad(ck)(x)
  np.testing.assert_allclose(to_host(g), f_scalar_grad_analytic(X_HOST),
                             rtol=1e-12, atol=1e-12)


@requires_ad
def test_grad_composes_with_jit():
  x = device_wrapper(X_HOST)
  g = _be.jit(_be.grad(f_scalar))(x)
  np.testing.assert_allclose(to_host(g), f_scalar_grad_analytic(X_HOST),
                             rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(has_ad, reason="numpy-only contract")
def test_numpy_backend_ad_contract():
  x = device_wrapper(X_HOST)
  with pytest.raises(NotImplementedError):
    _be.grad(f_scalar)
  with pytest.raises(NotImplementedError):
    _be.jvp(f_vec, (x,), (x,))
  with pytest.raises(NotImplementedError):
    _be.vjp(f_vec, x)
  assert _be.stop_gradient(x) is x
  assert _be.checkpoint(f_scalar) is f_scalar
