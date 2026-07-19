"""Toy validation of the implicit-differentiation primitive (increment 5).

Two closed-form problems validate ``root_solve``/``fixed_point_solve`` on
both AD backends, forward and reverse, before increment 6 wires the DIRK
Newton solve onto them:

* ridge regression (the paper's Figure-1 example): the residual is the
  normal-equation gradient, the solver a dense solve, and the Jacobian of
  the solution w.r.t. the regularizer and the targets is known in closed
  form — including a dict-pytree ``theta``;
* a batched contraction fixed point ``x = a tanh(x) + b`` whose IFT system
  is diagonal, exercising the structured ``linear_solve`` hook against the
  CG-normal-equations fallback.

The numpy backend must run the primal and stay AD-free.
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

# --- ridge regression toy ---------------------------------------------------

PHI_HOST = np.array([[1.0, 2.0, 0.5],
                     [0.3, 1.5, 1.0],
                     [2.0, 0.1, 0.7],
                     [0.5, 0.5, 0.5]])
Y_HOST = np.array([1.0, 2.0, 0.5, 1.5])
THETA0 = 0.7

PHI = device_wrapper(PHI_HOST)
X0_RIDGE = device_wrapper(np.zeros(3))


def _ridge_residual(x, theta):
  # Stationarity of 0.5 ||Phi x - y||^2 + 0.5 theta ||x||^2.
  return (jnp.matmul(PHI.T, jnp.matmul(PHI, x) - theta["y"]) +
          theta["lam"] * x)


def _ridge_solver(x0, theta):
  A = jnp.matmul(PHI.T, PHI) + theta["lam"] * jnp.eye(3)
  return jnp.linalg.solve(A, jnp.matmul(PHI.T, theta["y"]))


def _ridge_fn(lam, y):
  return _be.root_solve(_ridge_residual, _ridge_solver, X0_RIDGE,
                        {"lam": lam, "y": y})


def _ridge_primals():
  return device_wrapper(THETA0), device_wrapper(Y_HOST)


def _ridge_closed_form(lam_host, y_host):
  A = PHI_HOST.T @ PHI_HOST + lam_host * np.eye(3)
  x_star = np.linalg.solve(A, PHI_HOST.T @ y_host)
  dx_dlam = -np.linalg.solve(A, x_star)
  dx_dy = np.linalg.solve(A, PHI_HOST.T)
  return x_star, dx_dlam, dx_dy


@requires_ad
def test_ridge_primal_matches_closed_form():
  x_star, _, _ = _ridge_closed_form(THETA0, Y_HOST)
  out = _ridge_fn(*_ridge_primals())
  np.testing.assert_allclose(to_host(out), x_star, rtol=1e-12, atol=1e-12)


@requires_ad
def test_ridge_forward_reverse():
  probe_forward_reverse(_ridge_fn, _ridge_primals(), what="ridge root_solve")


@requires_ad
def test_ridge_fd():
  probe_fd_directional(_ridge_fn, _ridge_primals(), what="ridge root_solve")


@requires_ad
def test_ridge_jacobian_analytic():
  lam, y = _ridge_primals()
  _, dx_dlam, dx_dy = _ridge_closed_form(THETA0, Y_HOST)

  # Forward mode against the closed-form Jacobian columns.
  _, tangent = _be.jvp(_ridge_fn, (lam, y),
                       (device_wrapper(1.0), device_wrapper(np.zeros(4))))
  np.testing.assert_allclose(to_host(tangent), dx_dlam, rtol=1e-9, atol=1e-12)
  dy = np.array([0.3, -1.0, 0.5, 2.0])
  _, tangent = _be.jvp(_ridge_fn, (lam, y),
                       (device_wrapper(0.0), device_wrapper(dy)))
  np.testing.assert_allclose(to_host(tangent), dx_dy @ dy,
                             rtol=1e-9, atol=1e-12)

  # Reverse mode against the closed-form gradient of sum(x*).
  glam, gy = _be.grad(lambda l, yy: jnp.sum(_ridge_fn(l, yy)),
                      argnums=(0, 1))(lam, y)
  np.testing.assert_allclose(float(to_host(glam)), np.sum(dx_dlam),
                             rtol=1e-9, atol=1e-12)
  np.testing.assert_allclose(to_host(gy), dx_dy.T @ np.ones(3),
                             rtol=1e-9, atol=1e-12)


@requires_ad
def test_ridge_grad_under_jit():
  lam, y = _ridge_primals()
  _, dx_dlam, _ = _ridge_closed_form(THETA0, Y_HOST)
  g = _be.jit(_be.grad(lambda l: jnp.sum(_ridge_fn(l, y))))(lam)
  np.testing.assert_allclose(float(to_host(g)), np.sum(dx_dlam),
                             rtol=1e-9, atol=1e-12)


# --- batched contraction fixed point ----------------------------------------

B_HOST = np.array([[0.4, -1.2, 0.7], [2.0, 0.1, -0.5]])
A0 = 0.3


def _fp_T(x, theta):
  return theta["a"] * jnp.tanh(x) + theta["b"]


def _fp_solver(x0, theta):
  x = x0
  for _ in range(60):  # |a| = 0.3 -> converged to f64 roundoff
    x = _fp_T(x, theta)
  return x


def _fp_fn(a, b, linear_solve=None):
  return _be.fixed_point_solve(_fp_T, _fp_solver,
                               device_wrapper(np.zeros(B_HOST.shape)),
                               {"a": a, "b": b}, linear_solve=linear_solve)


def _fp_primals():
  return device_wrapper(A0), device_wrapper(B_HOST)


def _fp_diag_solve(matvec, rhs, x_star, theta, transpose):
  # d1F = a sech^2(x*) - 1 is diagonal (and symmetric), so the IFT systems
  # are elementwise divisions — the structured-solver hook in miniature.
  diag = theta["a"] / jnp.cosh(x_star)**2 - 1.0
  return rhs / diag


@requires_ad
def test_fixed_point_is_fixed():
  a, b = _fp_primals()
  x_star = _fp_fn(a, b)
  np.testing.assert_allclose(to_host(x_star),
                             to_host(_fp_T(x_star, {"a": a, "b": b})),
                             rtol=1e-13, atol=1e-13)


@requires_ad
def test_fixed_point_forward_reverse():
  probe_forward_reverse(_fp_fn, _fp_primals(), what="tanh fixed_point_solve")


@requires_ad
def test_fixed_point_fd():
  probe_fd_directional(_fp_fn, _fp_primals(), what="tanh fixed_point_solve")


@requires_ad
def test_fixed_point_structured_linear_solve():
  # The structured hook must reproduce the fallback CG solver's derivatives.
  a, b = _fp_primals()
  da = device_wrapper(0.7)
  db = device_wrapper(np.linspace(-1.0, 1.0, B_HOST.size).reshape(B_HOST.shape))

  _, tan_default = _be.jvp(lambda aa, bb: _fp_fn(aa, bb), (a, b), (da, db))
  _, tan_diag = _be.jvp(lambda aa, bb: _fp_fn(aa, bb, _fp_diag_solve),
                        (a, b), (da, db))
  np.testing.assert_allclose(to_host(tan_diag), to_host(tan_default),
                             rtol=1e-9, atol=1e-11)

  loss_default = _be.grad(lambda aa: jnp.sum(_fp_fn(aa, b)**2))(a)
  loss_diag = _be.grad(
      lambda aa: jnp.sum(_fp_fn(aa, b, _fp_diag_solve)**2))(a)
  np.testing.assert_allclose(float(to_host(loss_diag)),
                             float(to_host(loss_default)),
                             rtol=1e-9, atol=1e-11)


@pytest.mark.skipif(has_ad, reason="numpy-only contract")
def test_numpy_backend_runs_primal():
  lam, y = _ridge_primals()
  x_star, _, _ = _ridge_closed_form(THETA0, Y_HOST)
  out = _be.root_solve(_ridge_residual, _ridge_solver, X0_RIDGE,
                       {"lam": lam, "y": y})
  np.testing.assert_allclose(to_host(out), x_star, rtol=1e-12, atol=1e-12)
  out = _be.fixed_point_solve(_fp_T, _fp_solver,
                              device_wrapper(np.zeros(B_HOST.shape)),
                              {"a": device_wrapper(A0),
                               "b": device_wrapper(B_HOST)})
  assert np.all(np.isfinite(to_host(out)))
