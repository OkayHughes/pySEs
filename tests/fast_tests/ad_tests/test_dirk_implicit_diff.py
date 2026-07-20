"""DIRK Newton solve under implicit differentiation (increment 6).

``calc_implicit_update(use_implicit_diff=True)`` runs the byte-identical
Newton sweep inside ``_be.root_solve`` with the analytic tridiagonal
Jacobian as the structured IFT solve. These tests pin the adoption
contract on the acoustic-wave state from the DIRK unit tests:

* the primal output is unchanged by the flag on every backend;
* gradients and directional derivatives from the implicit path match
  differentiate-through-the-unrolled-solver (they agree up to the Newton
  convergence level, per Theorem 1 of Blondel et al. 2022);
* the implicit-path gradient independently matches central finite
  differences — which also validates that ``calc_dirk_jacobian`` really
  is the residual's Jacobian (an inexact Jacobian would bias the IFT
  solve in a way the unrolled comparison alone might share).
"""
import numpy as np
import pytest

from pyses._config import get_backend as _get_backend
from pyses.dynamical_cores.homme.implicit_terms import calc_implicit_update

from ...context import to_host
from ..dynamical_cores_tests.homme_tests.test_implicit_terms import (
    _build_acoustic_initial_state)

_be = _get_backend()
jnp = _be.np
device_wrapper = _be.array

has_ad = _be.wrapper_type in ("jax", "torch")
requires_ad = pytest.mark.skipif(not has_ad,
                                 reason="backend has no AD support")

DT_IMPLICIT = 5.0
PHI_SCALE = 1.0e-3  # balances the phi and w terms in the scalar loss


@pytest.fixture(scope="module")
def acoustic():
  state, h_grid, v_grid, dims, physics_config, model = (
      _build_acoustic_initial_state())
  return {"dynamics": state["dynamics"],
          "static_forcing": state["static_forcing"],
          "v_grid": v_grid, "physics_config": physics_config,
          "model": model}


def _solve(a, w_i_n0, theta_v_d_mass, use_implicit_diff):
  dynamics = dict(a["dynamics"])
  dynamics["theta_v_d_mass"] = theta_v_d_mass
  nh_vars = {"w_i": w_i_n0, "phi_i": 1.0 * a["dynamics"]["phi_i"]}
  return calc_implicit_update(DT_IMPLICIT, nh_vars, dynamics,
                              a["static_forcing"], a["v_grid"],
                              a["physics_config"], a["model"],
                              use_implicit_diff=use_implicit_diff)


def _loss(a, w_i_n0, theta_v_d_mass, use_implicit_diff):
  out = _solve(a, w_i_n0, theta_v_d_mass, use_implicit_diff)
  phi_ref = a["dynamics"]["phi_i"]
  return (jnp.sum(out["w_i"]**2) +
          PHI_SCALE * jnp.sum((out["phi_i"] - phi_ref)**2))


def _controls(a):
  return 1.0 * a["dynamics"]["w_i"], 1.0 * a["dynamics"]["theta_v_d_mass"]


def test_primal_identical_flag_on_off(acoustic):
  w0, tv = _controls(acoustic)
  out_off = _solve(acoustic, w0, tv, False)
  out_on = _solve(acoustic, w0, tv, True)
  for key in ("w_i", "phi_i"):
    np.testing.assert_allclose(to_host(out_on[key]), to_host(out_off[key]),
                               rtol=1e-13, atol=1e-10)


@requires_ad
def test_gradient_equivalence_unrolled_vs_implicit(acoustic):
  w0, tv = _controls(acoustic)

  def grads(flag):
    return _be.grad(
        lambda w, t: _loss(acoustic, w, t, flag), argnums=(0, 1))(w0, tv)

  gw_off, gt_off = grads(False)
  gw_on, gt_on = grads(True)
  # Agreement is limited by the Newton convergence level after the fixed
  # 5-sweep budget (Theorem 1: gradient error ~ iterate error).
  np.testing.assert_allclose(to_host(gw_on), to_host(gw_off),
                             rtol=1e-6, atol=1e-9)
  scale_t = np.max(np.abs(to_host(gt_off)))
  np.testing.assert_allclose(to_host(gt_on), to_host(gt_off),
                             rtol=1e-6, atol=1e-6 * (scale_t + 1e-30))


@requires_ad
def test_jvp_equivalence_unrolled_vs_implicit(acoustic):
  w0, tv = _controls(acoustic)
  rng = np.random.default_rng(5)
  dw = device_wrapper(
      rng.standard_normal(np.shape(to_host(w0))) *
      (np.abs(np.asarray(to_host(w0), dtype=float)) + 1.0))
  dt = device_wrapper(
      rng.standard_normal(np.shape(to_host(tv))) *
      (np.abs(np.asarray(to_host(tv), dtype=float)) + 1.0) * 1e-4)

  def jvp_of(flag):
    _, tangent = _be.jvp(lambda w, t: _loss(acoustic, w, t, flag),
                         (w0, tv), (dw, dt))
    return float(to_host(tangent))

  tan_off = jvp_of(False)
  tan_on = jvp_of(True)
  np.testing.assert_allclose(tan_on, tan_off, rtol=1e-6)


@requires_ad
def test_implicit_gradient_matches_fd(acoustic):
  # Independent ground truth for the implicit path (including the
  # exactness of the analytic Jacobian used in the structured solve).
  w0, tv = _controls(acoustic)
  rng = np.random.default_rng(9)
  w0_host = np.asarray(to_host(w0), dtype=float)
  direction = rng.standard_normal(w0_host.shape) * (np.abs(w0_host) + 1.0)

  _, ad_dir = _be.jvp(lambda w: _loss(acoustic, w, tv, True),
                      (w0,), (device_wrapper(direction),))
  h = 1e-6
  f_plus = float(to_host(_loss(acoustic, device_wrapper(w0_host + h * direction),
                               tv, True)))
  f_minus = float(to_host(_loss(acoustic, device_wrapper(w0_host - h * direction),
                                tv, True)))
  fd_dir = (f_plus - f_minus) / (2.0 * h)
  np.testing.assert_allclose(float(to_host(ad_dir)), fd_dir, rtol=1e-5)
