"""Shared machinery for implicit differentiation of solver outputs.

The backend method ``root_solve(residual_fn, solver_fn, x0, theta)`` returns
``x*`` with ``residual_fn(x*, theta) = 0``, differentiated via the implicit
function theorem (Blondel et al. 2022, arXiv:2105.15183) instead of through
``solver_fn``'s iterations:

  * the primal runs ``solver_fn(x0, theta)`` opaquely — its internal
    iterations, line searches, and (on JAX) even ``while_loop``s are never
    differentiated or taped;
  * the JVP solves    d1F dx = -d2F dtheta        for the output tangent;
  * the VJP solves    d1F^T w = v                 and returns -d2F^T w,

where ``d1F``/``d2F`` are the residual's Jacobians in ``x``/``theta``,
applied matrix-free through the residual's own JVP/VJP. The derivative is
exact at the solver's converged point; by Theorem 1 of the paper its error
is of the same order as the iterate error ``||x_hat - x*||``.

Contract (increment 5 scope):

  * ``x`` (and the residual output) is a single array; ``theta`` is an
    arbitrary pytree whose leaves are arrays. Close non-differentiable
    configuration (dicts of scalars, enums, grids) into ``residual_fn`` /
    ``solver_fn`` instead of passing it through ``theta``.
  * ``x0`` is passed through to the solver but its tangent is dropped: an
    exact root does not depend on the initial guess.
  * ``linear_solve(matvec, rhs, x_star, theta, transpose)``, if supplied,
    replaces the fallback solver for the IFT systems. It must return ``u``
    with ``d1F u = rhs`` (``transpose=False``) or ``d1F^T u = rhs``
    (``transpose=True``); ``matvec`` applies the requested operator, and
    ``x_star``/``theta`` are provided so structured implementations (e.g.
    a tridiagonal Thomas solve on an analytic Jacobian) can rebuild their
    band data and ignore ``matvec`` entirely.

``fixed_point_solve(T, solver_fn, x0, theta)`` is sugar for the residual
``F(x, theta) = T(x, theta) - x``.
"""


def cg_normal_equations(matvec, rmatvec, b, maxiter):
  """Solve the square system ``M u = b`` matrix-free via CG on the normal
  equations ``M^T M u = M^T b``.

  Backend-neutral: written with tensor operators and methods only, so the
  same code runs on JAX arrays and torch tensors inside either backend's
  custom-derivative rules. The iteration count is fixed (no data-dependent
  early exit), which keeps it jit-safe; each iteration costs one ``matvec``
  and one ``rmatvec``. When ``b`` batches many independent systems the
  global inner products still yield CG on the block-diagonal normal system,
  so the solve remains correct (if slower to converge than per-system CG).

  This is the *fallback* for toys and cold paths; hot-path consumers should
  supply a structured solver through ``root_solve``'s ``linear_solve`` hook
  (e.g. the DIRK tridiagonal Thomas solve) — both for speed and because a
  fixed iteration budget bounds, not guarantees, the residual reduction.
  """
  tiny = 1e-300  # guards 0/0 once the residual has converged to exactly zero
  atb = rmatvec(b)
  u = atb * 0.0
  r = atb
  p = r
  rs = (r * r).sum()
  for _ in range(maxiter):
    ap = rmatvec(matvec(p))
    alpha = rs / ((p * ap).sum() + tiny)
    u = u + alpha * p
    r = r - alpha * ap
    rs_new = (r * r).sum()
    p = r + (rs_new / (rs + tiny)) * p
    rs = rs_new
  return u
