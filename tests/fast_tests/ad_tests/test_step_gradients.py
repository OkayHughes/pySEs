"""Layers 2-3 of the gradient stress harness (strategy §5).

Layer 2 — single-step integration: JVP and VJP of the production
``advance_coupling_step`` with respect to initial-state fields, on two
configurations that between them exercise every audited subsystem:

* the moist CAM-SE baroclinic-wave case from the instrumented-coupling
  test (tracer transport + limiter + minmax DSS, both vertical remaps,
  hyperviscosity, sponge, physics forcing, CAM-SE thermodynamics);
* the HOMME non-hydrostatic acoustic-wave case under the RK3 HEVI
  stepper (NH thermodynamics, w/phi dynamics, the DIRK implicit solve on
  its default differentiate-through-the-solver path).

Probes assert finiteness of both AD modes and the forward/reverse
dot-product identity — exact properties that hold at kinks (full-step
elementwise FD was measured branch-noise-dominated; see the note above
the layer-3 section). When a probe reports non-finite values,
``tests/instrumented_coupling.py``'s per-sub-step flags are the
bisection tool for localizing the leak.

Layer 3 — short trajectories: reverse mode through a multi-step rollout
(driving the jitted step directly, bypassing the host-side NaN asserts
of the ``run_dycore`` generator loop), with the ``checkpoint`` shim
validated as value-preserving exactly and gradient-preserving up to the
model's measured re-linearization sensitivity.
"""
import numpy as np
import pytest

from pyses._config import get_backend as _get_backend
from pyses.dynamical_cores.model_config import init_default_config
from pyses.dynamical_cores.physics_dynamics_coupling import coupling_types
from pyses.dynamical_cores.run_dycore import advance_coupling_step
from pyses.dynamical_cores.time_step import time_step_options
from pyses.dynamical_cores.time_stepping import init_timestep_config

from ...context import to_host
from ..dynamical_cores_tests.homme_tests.test_implicit_terms import (
    _build_acoustic_initial_state)
from ..dynamical_cores_tests.test_instrumented_coupling_step import _build_case
from .probe_utils import (probe_forward_reverse, scaled_tangents_like,
                          tree_dot)

_be = _get_backend()
jnp = _be.np
device_wrapper = _be.array

pytestmark = pytest.mark.skipif(_be.wrapper_type not in ("jax", "torch"),
                                reason="backend has no AD support")


@pytest.fixture(scope="module")
def cam_case():
  return _build_case()


@pytest.fixture(scope="module")
def homme_case():
  state, h_grid, v_grid, dims, physics_config, model = (
      _build_acoustic_initial_state())
  _, diffusion_config, base_tc = init_default_config(3, h_grid, v_grid,
                                                     dims, model)
  timestep_config = init_timestep_config(
      base_tc["physics_dt"], h_grid, physics_config, diffusion_config, dims,
      model, dynamics_tstep_type=time_step_options.RK3_5STAGE_HEVI,
      physics_dynamics_coupling=coupling_types.lump_all)
  return dict(state_in=state, h_grid=h_grid, v_grid=v_grid,
              physics_config=physics_config,
              diffusion_config=diffusion_config,
              timestep_config=timestep_config, dims=dims, model=model,
              physics_forcing=None)


def _make_step_fn(case, control_keys, output_keys):
  rest = {key: val for key, val in case.items() if key != "state_in"}
  base_state = case["state_in"]

  def fn(*controls):
    dynamics = dict(base_state["dynamics"])
    for key, val in zip(control_keys, controls):
      dynamics[key] = val
    state = dict(base_state)
    state["dynamics"] = dynamics
    out = advance_coupling_step(state_in=state, **rest)
    return tuple(out["dynamics"][key] for key in output_keys)
  return fn


def _controls(case, keys):
  return tuple(1.0 * case["state_in"]["dynamics"][key] for key in keys)


# ---------------------------------------------------------------------------
# Layer 2 — single coupling step
# ---------------------------------------------------------------------------

# Measured layer-2 baseline (jax eager, 2026-07-19; root-caused
# 2026-07-20): the step's Jacobian is intrinsically eps-unstable — J.v
# moves by up to 4e-6 relative (broadly) under a 1-ulp input
# perturbation, identically on the scanned production step and the
# unrolled instrumented step, while the primal moves only ~5e-14. The
# model's structurally degenerate switches (limiter clips exactly onto
# bounds; fields compared against their own element extrema) flip on eps
# noise and their O(1) local Jacobian jumps mix globally within a step.
# The scanned path's ~2e-5 forward/reverse identity residual is one
# sampling of that floor (jvp- and vjp-of-scan compile two different
# primal programs); the unrolled eager identity of 8e-14 is the
# accidental exception (one bitwise primal shared by both modes). The
# tolerance pins the model-intrinsic consistency level with margin —
# tightening it requires smoothing the limiter switches, not AD changes.
STEP_IDENTITY_RTOL = 1e-4


# Both AD modes work on both backends: torch's scatter-max is a custom
# autograd.Function (_make_scatter_amax in pyses/_config.py) supplying
# the forward-mode rule PyTorch lacks for index_reduce_ and a backward
# that saves inputs rather than the mutation-prone output — the two
# increment-9 torch findings this suite originally pinned as xfails.
def test_cam_se_step_forward_reverse(cam_case):
  fn = _make_step_fn(cam_case, ("T", "horizontal_wind"),
                     ("T", "horizontal_wind", "d_mass"))
  probe_forward_reverse(fn, _controls(cam_case, ("T", "horizontal_wind")),
                        what="cam_se coupling step",
                        rtol=STEP_IDENTITY_RTOL)


def test_homme_nh_step_forward_reverse(homme_case):
  fn = _make_step_fn(homme_case, ("w_i", "theta_v_d_mass"),
                     ("w_i", "phi_i", "theta_v_d_mass"))
  probe_forward_reverse(fn,
                        _controls(homme_case, ("w_i", "theta_v_d_mass")),
                        what="homme NH coupling step",
                        rtol=STEP_IDENTITY_RTOL)


# Note on finite differences at this layer: full-step directional FD was
# measured (jax eager, 2026-07-19) and found branch-noise-dominated — the
# at-rest HEVI background puts the DIRK line-search switch quantities
# exactly at their thresholds, so an h=1e-7 full-step perturbation crosses
# branches freely (9% of dominant w_i tangent entries and 34% of phi_i
# entries deviate beyond 1e-2 from FD, individually up to 4x). Elementwise
# FD is therefore only asserted at layer 1, on branch-interior kernel
# states; at this layer the exact kink-safe properties (finiteness and
# the forward/reverse identity) are the contract.


# ---------------------------------------------------------------------------
# Layer 3 — short trajectory (reverse mode + checkpoint)
# ---------------------------------------------------------------------------

N_STEPS = 2  # enough to compose steps in reverse mode while fitting CI budgets


def _rollout_loss_fn(case, use_checkpoint):
  rest = {key: val for key, val in case.items() if key != "state_in"}
  base_state = case["state_in"]

  def step(state):
    return advance_coupling_step(state_in=state, **rest)

  if use_checkpoint:
    step = _be.checkpoint(step)

  def loss(T0, wind0):
    dynamics = dict(base_state["dynamics"])
    dynamics["T"] = T0
    dynamics["horizontal_wind"] = wind0
    state = dict(base_state)
    state["dynamics"] = dynamics
    for _ in range(N_STEPS):
      state = step(state)
    return (jnp.sum(state["dynamics"]["T"]**2) +
            1e-4 * jnp.sum(state["dynamics"]["horizontal_wind"]**2))
  return loss


def test_cam_se_trajectory_reverse(cam_case):
  primals = _controls(cam_case, ("T", "horizontal_wind"))
  loss = _rollout_loss_fn(cam_case, use_checkpoint=False)

  grads = _be.grad(loss, argnums=(0, 1))(*primals)
  for g, name in zip(grads, ("T", "horizontal_wind")):
    host = np.asarray(to_host(g))
    assert np.all(np.isfinite(host)), f"trajectory grad wrt {name} non-finite"

  # Forward/reverse consistency on the scalar loss: <grad, v> must equal
  # the directional JVP. Measured: 6.2e-4 relative at 2 steps — the
  # per-step ~2e-5 scan-consistency level compounds multiplicatively
  # (about 30x through the second step's linearized propagator), not
  # additively; expect roughly an order of magnitude per additional
  # step when budgeting trajectory-gradient quality.
  tangents = scaled_tangents_like(primals, seed=11)
  _, tangent_out = _be.jvp(loss, primals, tangents)
  lhs = float(to_host(tangent_out))
  rhs = tree_dot(grads, tangents)
  np.testing.assert_allclose(lhs, rhs, rtol=2e-3)


def test_cam_se_trajectory_checkpoint_equivalent(cam_case):
  # Rematerialization must be value-preserving exactly, and
  # gradient-preserving up to the model's re-linearization sensitivity:
  # jax remat re-linearizes the checkpointed region in a different op
  # order, and — the same phenomenon as the scan/unroll finding above —
  # that alone shifts this 2-step trajectory gradient by up to 2.6%
  # per entry (measured; norm-wise far smaller). torch degrades to the
  # unmaterialized path under torch.func and matches trivially.
  primals = _controls(cam_case, ("T", "horizontal_wind"))
  loss_plain = _rollout_loss_fn(cam_case, use_checkpoint=False)
  loss_ck = _rollout_loss_fn(cam_case, use_checkpoint=True)
  np.testing.assert_allclose(float(to_host(loss_ck(*primals))),
                             float(to_host(loss_plain(*primals))),
                             rtol=1e-12)
  g_plain = _be.grad(loss_plain)(*primals)
  g_ck = _be.grad(loss_ck)(*primals)
  g_plain_h = np.asarray(to_host(g_plain), dtype=float)
  g_ck_h = np.asarray(to_host(g_ck), dtype=float)
  rel_l2 = (np.linalg.norm(g_ck_h - g_plain_h) /
            np.linalg.norm(g_plain_h))
  assert rel_l2 < 0.05, (
      f"checkpointed trajectory gradient deviates {rel_l2:.3e} in "
      f"relative L2 from the plain gradient")
