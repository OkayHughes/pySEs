"""HOMME non-hydrostatic DIRK implicit-stage solver.

Direct port of ``compute_stage_value_dirk`` (and its helpers
``get_dirk_jacobian``, ``compute_gwphis``,
``solve_strict_diag_dominant_tridiag``) from HOMME's ``imex_mod`` /
``prim_advance_mod``.  The Fortran reference is reproduced verbatim in
``implicit_terms_ref.txt`` next to this file.

The DIRK stage equation is

  w_np1   = w_n0   + dt2 * g(phi_n0) * (mu - 1)
  phi_np1 = phi_n0 + dt2 * g(phi_n0) * w_np1

with ``w_n0`` / ``phi_n0`` already pre-loaded by the explicit
``alphadt_n0`` / ``alphadt_nm1`` contributions and the ``np1`` surface-
topography ``gw_phis`` term subtracted.  ``mu = d p_nh / d (d_mass)``
is recomputed from ``(theta_v_d_mass, d_mass, dphi)`` each Newton step.
"""
from functools import partial

from ..._config import get_backend as _get_backend
from ..model_state import wrap_dynamics
from ..utils_3d import (cumulative_sum, interface_to_delta, phi_to_g,
                        phi_to_z, calc_perceived_phi_tend)
from .thermodynamics import eval_mu

_be = _get_backend()
jit = _be.jit
jnp = _be.np


@jit
def calc_dirk_jacobian(dt, d_mass, d_phi, pnh, physics_config):
  """Exact analytic Jacobian of the DIRK fixed-point residual.

  Mirrors ``get_dirk_jacobian`` (exact branch).  ``jacD`` has length
  ``nlev``; the two off-diagonals have length ``nlev - 1``, matching
  the layout the tridiagonal solver expects.
  """
  kappa = physics_config["Rgas"] / physics_config["cp"]
  gravity = physics_config["gravity"]
  a = (dt * gravity) ** 2 / (1.0 - kappa)
  # b[0] = a / dp[0]; b[k] = 2 a / (dp[k-1] + dp[k])   for k = 1..nlev-1.
  b = jnp.concatenate(
      (a / d_mass[:, :, :, 0:1],
       2.0 * a / (d_mass[:, :, :, :-1] + d_mass[:, :, :, 1:])),
      axis=-1)
  c = pnh / d_phi
  # JacL[k-1] = b[k] * c[k-1]   for k = 1..nlev-1   ->  length nlev - 1.
  jacL = b[:, :, :, 1:] * c[:, :, :, :-1]
  # JacU[0]   = 2 b[0] c[0]
  # JacU[k]   = b[k] c[k]       for k = 1..nlev-2   ->  length nlev - 1.
  jacU = jnp.concatenate(
      (2.0 * b[:, :, :, 0:1] * c[:, :, :, 0:1],
       b[:, :, :, 1:-1] * c[:, :, :, 1:-1]),
      axis=-1)
  # JacD[0]      = 1 - JacU[0]
  # JacD[k]      = 1 - JacL[k-1] - JacU[k]   for k = 1..nlev-2
  # JacD[nlev-1] = 1 - JacL[nlev-2] - b[nlev-1] c[nlev-1]
  jacD = jnp.concatenate(
      (1.0 - jacU[:, :, :, 0:1],
       1.0 - jacL[:, :, :, :-1] - jacU[:, :, :, 1:],
       1.0 - jacL[:, :, :, -1:] - b[:, :, :, -1:] * c[:, :, :, -1:]),
      axis=-1)
  return jacL, jacD, jacU


@partial(jit, static_argnames=["model"])
def advance_buoyancy_explicit(alpha_dts,
                              dynamics_series,
                              w_init,
                              phi_init,
                              static_forcing,
                              physics_config,
                              v_grid,
                              model):
  """Accumulate the explicit ``alpha dt * S(u_nt)`` contributions.

  Mirrors the ``alphadt_n0`` / ``alphadt_nm1`` blocks of
  ``compute_stage_value_dirk``::

      w_n0   += alpha dt * g(phi_n0) * (mu(nt) - 1)
      phi_n0 += alpha dt * (g(phi_n0) * w(nt) - gw_phis(nt))

  ``g`` is evaluated at the *running* ``phi_n0`` accumulator (matching
  Fortran's ``g_from_phi(phi_n0)``).
  """
  w_n0 = 1.0 * w_init
  phi_n0 = 1.0 * phi_init
  for dynamics, alpha_dt in zip(dynamics_series, alpha_dts):
    g = phi_to_g(phi_n0, physics_config, model)
    _, _, _, mu = eval_mu(dynamics, dynamics["phi_i"], v_grid,
                          physics_config, model)
    w_n0 = w_n0 + alpha_dt * g * (mu - 1.0)
    gw_phis = calc_perceived_phi_tend(dynamics["horizontal_wind"],
                                      dynamics["d_mass"],
                                      static_forcing["grad_phi_surf"],
                                      phi_n0,
                                      physics_config, v_grid, model)
    phi_n0 = phi_n0 + alpha_dt * (g * dynamics["w_i"] - gw_phis)
  return {"w_i": w_n0,
          "phi_i": phi_n0}


@jit
def solve_strict_diag_dominant_tridiag(jacL, jacD, jacU, rhs):
  """Thomas-algorithm tridiagonal solver for a strictly diagonally
  dominant system.  Matches HOMME's
  ``solve_strict_diag_dominant_tridiag``."""
  # Forward elimination and back-substitution are sequential folds over the
  # vertical levels, expressed through the backend ``scan`` so the loop body
  # compiles once (``lax.scan`` on JAX) instead of unrolling ``nlev`` times.
  # jacD has length nlev; jacL/jacU are the length-(nlev-1) off-diagonals.
  nlev = jacD.shape[-1]

  def _forward(carry, level):
    jacL_k, jacU_k, jacD_next, rhs_next = level
    diag_prev, rhs_prev = carry
    lower_rat = jacL_k / diag_prev
    diag_k = jacD_next - lower_rat * jacU_k
    rhs_k = rhs_next - lower_rat * rhs_prev
    return (diag_k, rhs_k), (diag_k, rhs_k)

  diag0 = jacD[:, :, :, 0]
  rhs0 = rhs[:, :, :, 0]
  _, (diag_rest, rhs_rest) = _be.scan(
      _forward, (diag0, rhs0),
      (jacL[:, :, :, 0:nlev - 1], jacU[:, :, :, 0:nlev - 1],
       jacD[:, :, :, 1:nlev], rhs[:, :, :, 1:nlev]),
      axis=-1)
  updated_diag = jnp.concatenate((diag0[:, :, :, jnp.newaxis], diag_rest), axis=-1)
  rhs_levs = jnp.concatenate((rhs0[:, :, :, jnp.newaxis], rhs_rest), axis=-1)

  sol_last = rhs_levs[:, :, :, nlev - 1] / updated_diag[:, :, :, nlev - 1]

  def _backward(sol_next, level):
    rhs_lev_k, jacU_k, diag_k = level
    sol_k = (rhs_lev_k - jacU_k * sol_next) / diag_k
    return sol_k, sol_k

  _, sol_rest = _be.scan(
      _backward, sol_last,
      (rhs_levs[:, :, :, 0:nlev - 1], jacU[:, :, :, 0:nlev - 1],
       updated_diag[:, :, :, 0:nlev - 1]),
      reverse=True, axis=-1)
  return jnp.concatenate((sol_rest, sol_last[:, :, :, jnp.newaxis]), axis=-1)


@jit
def init_search_dir(dt_implicit,
                    w_guess,
                    dphi_guess,
                    d_mass,
                    nh_pressure,
                    physics_config,
                    w_phi_compatibility_residual):
  """Solve ``J x = -F`` on the interior nlev levels.

  Returned array is padded with a zero at the surface so its shape
  matches ``w_guess`` (length ``nlev + 1``).
  """
  jacL, jacD, jacU = calc_dirk_jacobian(dt_implicit, d_mass, dphi_guess,
                                        nh_pressure, physics_config)
  newton_rhs = -w_phi_compatibility_residual[:, :, :, :-1]
  x = solve_strict_diag_dominant_tridiag(jacL, jacD, jacU, newton_rhs)
  return jnp.concatenate(
      (x, jnp.zeros_like(w_guess[:, :, :, -1:])), axis=-1)


@partial(jit, static_argnames=["model"])
def take_limited_step(dt_implicit,
                      phi_n0,
                      dphi_n0,
                      w_guess,
                      w_search_dir,
                      physics_config,
                      model):
  """Backtracking-limited Newton update for ``(dphi, w)``.

  Computes the alpha-1 trial ``dphi_trial = dphi_n0 + dt * d/dt(dphi)``,
  detects unphysical (``dphi >= 0``) columns, and per such column
  halves the smallest non-negative ``alpha_k`` that would zero out
  some ``dphi`` -- matching Fortran's ``alpha = alpha/2`` rule.  All
  ``g`` evaluations follow Fortran and use ``g_from_phi(phi_n0)`` (the
  fixed RHS), not the current iterate.
  """
  # phi_to_g returns a scalar for shallow-atmosphere models; broadcast
  # so the per-interface slicing below works uniformly across shallow
  # and deep atmospheres.
  g_n0 = jnp.broadcast_to(phi_to_g(phi_n0, physics_config, model),
                           phi_n0.shape)
  # alpha = 1 candidate.  d_phi_tend[k] = g[k+1] u[k+1] - g[k] u[k] for
  # k = 0..nlev-2, and d_phi_tend[nlev-1] = -g[nlev-1] u[nlev-1] where
  # u = w + x.  The surface phi is fixed, so its tendency is zero.
  phi_tend_interim = jnp.concatenate(
      (g_n0[:, :, :, :-1] *
       (w_guess[:, :, :, :-1] + w_search_dir[:, :, :, :-1]),
       jnp.zeros_like(w_guess[:, :, :, -1:])),
      axis=-1)
  d_phi_tend = interface_to_delta(phi_tend_interim)
  dphi_trial = dphi_n0 + dt_implicit * d_phi_tend
  column_mask = jnp.sum(dphi_trial >= 0.0, axis=-1) > 0

  # Per-mid-level dx and dw used in the alpha_k formula.  Fortran's
  # ``k = nlev`` case is ``dx = -x(nlev)`` / ``dw = -w(nlev)``; in
  # 0-indexed Python with the surface entry stored at ``-1`` (and pinned
  # to zero by ``init_search_dir``), the active bottom interface is
  # ``-2``.
  d_search_dir = jnp.concatenate(
      (w_search_dir[:, :, :, 1:-1] - w_search_dir[:, :, :, :-2],
       -w_search_dir[:, :, :, -2:-1]),
      axis=-1)
  d_w_guess = jnp.concatenate(
      (w_guess[:, :, :, 1:-1] - w_guess[:, :, :, :-2],
       -w_guess[:, :, :, -2:-1]),
      axis=-1)
  # g_avg matches Fortran's ``g_from_phi((phi_n0(k)+phi_n0(k+1))/2)``;
  # under shallow + constant g this is just g, but the cancellation
  # below avoids needing the average explicitly.
  g_avg = (g_n0[:, :, :, :-1] + g_n0[:, :, :, 1:]) / 2.0
  numerator = -(dphi_n0 + dt_implicit * g_avg * d_w_guess)
  denominator = dt_implicit * g_avg * d_search_dir
  alpha_k = jnp.where(d_search_dir != 0.0,
                      numerator / jnp.where(denominator != 0.0,
                                            denominator, 1.0),
                      1.0)
  alpha_k = jnp.where(alpha_k >= 0.0, alpha_k, 1.0)
  safe_step_length = jnp.minimum(jnp.min(alpha_k, axis=-1), jnp.ones(1)) / 2.0

  # Recompute dphi with the limited step length.
  step_b = safe_step_length[:, :, :, jnp.newaxis]
  phi_tend_lim = jnp.concatenate(
      (g_n0[:, :, :, :-1] *
       (w_guess[:, :, :, :-1] + step_b * w_search_dir[:, :, :, :-1]),
       jnp.zeros_like(w_guess[:, :, :, -1:])),
      axis=-1)
  dphi_lim = dphi_n0 + dt_implicit * interface_to_delta(phi_tend_lim)
  dphi_out = jnp.where(column_mask[:, :, :, jnp.newaxis],
                       dphi_lim, dphi_trial)

  step_length = jnp.where(column_mask, safe_step_length, 1.0)
  w_out = w_guess + step_length[:, :, :, jnp.newaxis] * w_search_dir
  return dphi_out, w_out


# ---------------------------------------------------------------------------
# Implicit-differentiation wiring (docs/ad_hardening_strategy.md §4.2).
#
# The Newton carry ``(w, dphi, nh_pressure, residual)`` is fully derived
# from the interface velocity ``w``: dphi follows from the phi-update
# relation and everything else is recomputed from dphi.  ``w`` is therefore
# the root variable handed to ``_be.root_solve``; the surface entry, which
# the solver never moves (``init_search_dir`` pins its search direction to
# zero), is closed by an identity residual row against ``theta["w_surf"]``.
#
# Contract note: every array derived from differentiable inputs must reach
# these functions through ``theta`` (or ``w``), never through closures —
# closure-captured tracers inside the backend's custom-derivative rules
# either leak (JAX) or bypass ``save_for_backward`` (torch).  Closures may
# only carry static configuration: ``dt_implicit`` (python float),
# ``v_grid`` geometry, ``physics_config`` scalars, and ``model``.
# ---------------------------------------------------------------------------

def _dirk_phi_from_w(dt_implicit, w, theta, physics_config, model):
  """Reconstruct ``(g_n0, dphi, phi_i)`` from the velocity iterate ``w``.

  Mirrors the in-loop update exactly: ``dphi = dphi_n0 + dt * delta(g_n0 w)``
  with the surface tendency pinned to zero, and ``phi`` rebuilt from the
  surface upward.
  """
  phi_n0 = theta["phi_n0"]
  g_n0 = jnp.broadcast_to(phi_to_g(phi_n0, physics_config, model),
                          phi_n0.shape)
  phi_tend = jnp.concatenate(
      (g_n0[:, :, :, :-1] * w[:, :, :, :-1],
       jnp.zeros_like(w[:, :, :, -1:])), axis=-1)
  dphi = theta["dphi_n0"] + dt_implicit * interface_to_delta(phi_tend)
  phi = cumulative_sum(-dphi, theta["phi_surf"])
  return g_n0, dphi, phi


def _dirk_residual(dt_implicit, w, theta, v_grid, physics_config, model):
  """DIRK stage residual ``F(w, theta)`` on the ``nlev + 1`` interfaces.

  Interior rows are the ``w``/``phi`` compatibility residual
  ``w - (w_n0 + dt g_n0 (mu(phi(w)) - 1))``; the surface row pins ``w`` to
  ``theta["w_surf"]`` so the system is square and invertible.
  """
  g_n0, _, phi = _dirk_phi_from_w(dt_implicit, w, theta, physics_config,
                                  model)
  state = {"theta_v_d_mass": theta["theta_v_d_mass"],
           "d_mass": theta["d_mass"]}
  _, _, _, mu = eval_mu(state, phi, v_grid, physics_config, model)
  fn = w - (theta["w_n0"] + dt_implicit * g_n0 * (mu - 1.0))
  return jnp.concatenate(
      (fn[:, :, :, :-1], w[:, :, :, -1:] - theta["w_surf"]), axis=-1)


def _dirk_newton_sweeps(dt_implicit, w0, dphi0, nh_pressure0, resid0,
                        w_n0, phi_n0, dphi_n0, g_n0, phi_surf, state,
                        v_grid, physics_config, model, max_itercount):
  """The fixed-budget Newton sweep, shared verbatim by both solver paths."""
  def _newton_step(carry, _):
    w_guess, dphi_guess, nh_pressure, resid = carry
    w_search_dir = init_search_dir(dt_implicit, w_guess, dphi_guess,
                                   state["d_mass"], nh_pressure,
                                   physics_config, resid)
    dphi_guess, w_guess = take_limited_step(dt_implicit, phi_n0, dphi_n0,
                                            w_guess, w_search_dir,
                                            physics_config, model)
    phi_guess = cumulative_sum(-dphi_guess, phi_surf)
    nh_pressure, _, _, mu = eval_mu(state, phi_guess, v_grid,
                                    physics_config, model)
    fn = w_guess - (w_n0 + dt_implicit * g_n0 * (mu - 1.0))
    resid = jnp.concatenate(
        (fn[:, :, :, :-1], jnp.zeros_like(fn[:, :, :, -1:])), axis=-1)
    return (w_guess, dphi_guess, nh_pressure, resid), None

  carry, _ = _be.scan(_newton_step, (w0, dphi0, nh_pressure0, resid0),
                      jnp.arange(max_itercount))
  return carry


def _unrolled_thomas(jacL, jacD, jacU, rhs):
  """Thomas solve with the vertical recurrences unrolled at trace time.

  Same algorithm as :func:`solve_strict_diag_dominant_tridiag`, but as pure
  arithmetic instead of a backend ``scan``: inside the implicit-diff rule
  the solve graph must be structurally linear in ``rhs`` so reverse mode
  can transpose it, and ``lax.scan`` does not transpose in that position
  (its transposition hands the scan a cotangent accumulator).  The unroll
  is also the GPU-fusion-friendly form (cf. the Zerroukat remap's Thomas
  solve); ``nlev`` is static under jit.
  """
  nlev = jacD.shape[-1]
  diag = [jacD[..., 0]]
  r = [rhs[..., 0]]
  for k in range(1, nlev):
    lower_rat = jacL[..., k - 1] / diag[-1]
    diag.append(jacD[..., k] - lower_rat * jacU[..., k - 1])
    r.append(rhs[..., k] - lower_rat * r[-1])
  sol = [r[nlev - 1] / diag[nlev - 1]]
  for k in range(nlev - 2, -1, -1):
    sol.append((r[k] - jacU[..., k] * sol[-1]) / diag[k])
  sol.reverse()
  return jnp.stack(sol, axis=-1)


def _dirk_ift_linear_solve(dt_implicit, rhs, x_star, theta, v_grid,
                           physics_config, model, transpose):
  """Structured solve for the implicit-function-theorem systems.

  Rebuilds the analytic tridiagonal DIRK Jacobian at the solution and
  Thomas-solves the interior block; the adjoint system is the same solve
  with the off-diagonal bands swapped (the transpose of a tridiagonal
  matrix).  The surface residual row is the identity, so that component
  passes straight through.
  """
  _, dphi, phi = _dirk_phi_from_w(dt_implicit, x_star, theta,
                                  physics_config, model)
  state = {"theta_v_d_mass": theta["theta_v_d_mass"],
           "d_mass": theta["d_mass"]}
  nh_pressure, _, _, _ = eval_mu(state, phi, v_grid, physics_config, model)
  jacL, jacD, jacU = calc_dirk_jacobian(dt_implicit, theta["d_mass"], dphi,
                                        nh_pressure, physics_config)
  if transpose:
    jacL, jacU = jacU, jacL
  interior = _unrolled_thomas(jacL, jacD, jacU, rhs[:, :, :, :-1])
  return jnp.concatenate((interior, rhs[:, :, :, -1:]), axis=-1)


@partial(jit, static_argnames=["model", "use_implicit_diff"])
def calc_implicit_update(dt_implicit,
                         nh_vars_before_implicit,
                         dynamics_before_implicit,
                         static_forcing,
                         v_grid,
                         physics_config,
                         model,
                         use_implicit_diff=False):
  """Solve the DIRK stage equation for ``(phi_np1, w_np1)``.

  Direct port of ``compute_stage_value_dirk``.
  ``nh_vars_before_implicit`` carries the explicit RHS (``w_n0``,
  ``phi_n0`` augmented with the accumulated ``alpha dt S(u_nt)``
  contributions); ``dynamics_before_implicit["phi_i"]`` provides the
  hydrostatic initial guess.

  Newton iteration runs a fixed budget of ``max_itercount`` sweeps.  A
  JIT-compatible early exit on ``deltaerr < deltatol`` would require
  ``lax.while_loop`` and is left for a follow-up (legal under
  ``use_implicit_diff=True``, where the solver is opaque to AD).

  With ``use_implicit_diff=True`` the identical Newton sweep runs inside
  ``_be.root_solve``: the primal output is unchanged, but derivatives come
  from the implicit function theorem — one adjoint (or tangent)
  tridiagonal solve with the analytic DIRK Jacobian at the solution —
  instead of differentiating through the taped sweeps and line searches.
  """
  gravity = physics_config["gravity"]
  max_itercount = 5

  # phi_n0 / w_n0 : explicit RHS, then subtract the ``np1`` ``gw_phis``.
  phi_n0 = 1.0 * nh_vars_before_implicit["phi_i"]
  w_n0 = 1.0 * nh_vars_before_implicit["w_i"]
  gw_phis = calc_perceived_phi_tend(dynamics_before_implicit["horizontal_wind"],
                                    dynamics_before_implicit["d_mass"],
                                    static_forcing["grad_phi_surf"],
                                    phi_n0,
                                    physics_config, v_grid, model)
  phi_n0 = phi_n0 - dt_implicit * gw_phis

  # Hydrostatic initial guess for phi; w from the implicit relation.
  phi_guess = 1.0 * dynamics_before_implicit["phi_i"]
  w_guess = (phi_to_z(phi_guess, physics_config, model) -
             phi_to_z(phi_n0, physics_config, model)) / dt_implicit

  dphi_guess = interface_to_delta(phi_guess)
  dphi_n0 = interface_to_delta(phi_n0)

  # Pre-Newton limiter.  In HOMME's convention dphi < 0 (phi decreases
  # downward); any column with ``dphi > -g`` has an unphysically thin
  # / inverted layer in the initial guess.  Pin those entries to ``-g``
  # and rebuild phi_guess by the bottom-up scan
  # ``phi[k] = phi[k+1] - dphi[k]`` -- which is what
  # ``cumulative_sum(-dphi, phi_surf)`` does (the helper accumulates
  # the positive ``-dphi`` from the surface upward).  w_guess is then
  # recomputed from the rebuilt phi.
  limit_value = -gravity
  mask_3d = dphi_guess > limit_value
  dphi_guess = jnp.where(mask_3d, limit_value, dphi_guess)
  any_unsafe = (jnp.sum(mask_3d, axis=-1) > 0)[:, :, :, jnp.newaxis]
  phi_guess_rebuilt = cumulative_sum(-dphi_guess,
                                     static_forcing["phi_surf"])
  phi_guess = jnp.where(any_unsafe, phi_guess_rebuilt, phi_guess)
  w_guess_rebuilt = (phi_to_z(phi_guess, physics_config, model) -
                     phi_to_z(phi_n0, physics_config, model)) / dt_implicit
  w_guess = jnp.where(any_unsafe, w_guess_rebuilt, w_guess)

  # Initial residual:  F = w - (w_n0 + dt * g(phi_n0) * (mu - 1)).
  nh_pressure, _, _, mu = eval_mu(dynamics_before_implicit, phi_guess,
                                  v_grid, physics_config, model)
  # phi_to_g returns a scalar in shallow mode; broadcast for uniform
  # downstream arithmetic and indexing.
  g_n0 = jnp.broadcast_to(phi_to_g(phi_n0, physics_config, model),
                           phi_n0.shape)
  fn = w_guess - (w_n0 + dt_implicit * g_n0 * (mu - 1.0))
  # The tridiag solver expects the residual on nlev+1 interfaces with
  # a zero at the surface (the surface w is pinned by the boundary
  # condition); init_search_dir slices off this trailing zero.
  w_phi_compatibility_residual = jnp.concatenate(
      (fn[:, :, :, :-1], jnp.zeros_like(fn[:, :, :, -1:])), axis=-1)

  # Fixed-budget Newton sweep (shared helper so both solver paths run the
  # byte-identical loop body).  The loop-invariant state dict is the minimal
  # slice eval_mu / init_search_dir actually read.
  state_min = {"theta_v_d_mass": dynamics_before_implicit["theta_v_d_mass"],
               "d_mass": dynamics_before_implicit["d_mass"]}
  if use_implicit_diff:
    # dt_implicit and the v_grid leaves eval_mu reads are traced arguments
    # of this jitted function, so they must flow through theta like every
    # other traced value: the backend's custom-derivative rule is re-run
    # in later traces (e.g. pjit's jvp_jaxpr), where closure-captured
    # tracers from the original trace leak. The height-coordinate branch
    # of eval_mu only checks key presence, so it stays a static closure.
    is_height = "height" in v_grid
    if is_height:
      v_grid_theta = {}
    else:
      v_grid_theta = {
          "hybrid_a_i": jnp.asarray(v_grid["hybrid_a_i"]),
          "reference_surface_mass": jnp.asarray(
              v_grid["reference_surface_mass"])}

    def _theta_v_grid(th):
      return {"height": True} if is_height else th["v_grid"]

    # physics_config from init_physics_config carries array-valued leaves
    # (nested moisture-species dicts included), which are traced arguments
    # of this jitted function — so it must ride through theta as well.
    # Coerce every leaf to an array so the torch path flattens to tensors.
    def _as_array_tree(tree):
      if isinstance(tree, dict):
        return {key: _as_array_tree(val) for key, val in tree.items()}
      return jnp.asarray(tree)

    theta = {"w_n0": w_n0, "phi_n0": phi_n0, "dphi_n0": dphi_n0,
             "phi_surf": static_forcing["phi_surf"],
             "w_surf": w_guess[:, :, :, -1:],
             "theta_v_d_mass": state_min["theta_v_d_mass"],
             "d_mass": state_min["d_mass"],
             "dt": jnp.asarray(dt_implicit),
             "v_grid": v_grid_theta,
             "physics_config": _as_array_tree(physics_config),
             # Solver-only initialization data: consumed by _solver below,
             # ignored by the residual, so it carries no gradient path (the
             # exact root is independent of the starting carry).
             "init_dphi": dphi_guess,
             "init_nh_pressure": nh_pressure,
             "init_resid": w_phi_compatibility_residual}

    def _residual(wx, th):
      return _dirk_residual(th["dt"], wx, th, _theta_v_grid(th),
                            th["physics_config"], model)

    def _solver(x0_, th):
      g_n0_ = jnp.broadcast_to(
          phi_to_g(th["phi_n0"], th["physics_config"], model),
          th["phi_n0"].shape)
      state = {"theta_v_d_mass": th["theta_v_d_mass"],
               "d_mass": th["d_mass"]}
      w_out, _, _, _ = _dirk_newton_sweeps(
          th["dt"], x0_, th["init_dphi"], th["init_nh_pressure"],
          th["init_resid"], th["w_n0"], th["phi_n0"], th["dphi_n0"], g_n0_,
          th["phi_surf"], state, _theta_v_grid(th), th["physics_config"],
          model, max_itercount)
      return w_out

    def _linear(matvec, rhs, x_star, th, transpose):
      del matvec  # the analytic Jacobian replaces the matrix-free operator
      return _dirk_ift_linear_solve(th["dt"], rhs, x_star, th,
                                    _theta_v_grid(th), th["physics_config"],
                                    model, transpose)

    w_guess = _be.root_solve(_residual, _solver, w_guess, theta,
                             linear_solve=_linear)
  else:
    (w_guess, dphi_guess, nh_pressure,
     w_phi_compatibility_residual) = _dirk_newton_sweeps(
        dt_implicit, w_guess, dphi_guess, nh_pressure,
        w_phi_compatibility_residual, w_n0, phi_n0, dphi_n0, g_n0,
        static_forcing["phi_surf"], state_min, v_grid, physics_config,
        model, max_itercount)

  # Final phi update: phi_np1 = phi_n0 + dt * g(phi_n0) * w_np1.
  phi_next = phi_n0 + dt_implicit * g_n0 * w_guess
  # Surface phi must equal phi_surf.  JAX/torch arrays are immutable,
  # so rebuild via concatenate rather than in-place assignment.
  surf = static_forcing["phi_surf"][:, :, :, jnp.newaxis]
  phi_next = jnp.concatenate((phi_next[:, :, :, :-1], surf), axis=-1)

  return wrap_dynamics(dynamics_before_implicit["horizontal_wind"],
                       dynamics_before_implicit["theta_v_d_mass"],
                       dynamics_before_implicit["d_mass"],
                       model,
                       phi_i=phi_next,
                       w_i=w_guess)
