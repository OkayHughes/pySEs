# Hardening automatic differentiation of the pySEs forward step

Status: strategy adopted; work tracked incrementally on the `ad-hardening` branch.

## 1. Goal

Make the full coupled forward step (`advance_coupling_step` in
`pyses/dynamical_cores/run_dycore.py`) reliably differentiable in both forward
(JVP) and reverse (VJP) mode on the JAX and torch backends, with

1. a stress-test harness that *measures* differentiability before we change any
   numerics,
2. targeted remediation of the operations that break or degrade gradients,
   including approximation-of-identity smoothing where (and only where) it is
   warranted, and
3. a backend-level implicit-differentiation feature (`custom_root` /
   fixed-point differentiation in the style of Blondel et al. 2022,
   arXiv:2105.15183) implemented natively for JAX **and** torch, with the HOMME
   DIRK Newton solve as its first consumer.

The default forward model must remain bitwise unchanged. Everything gradient-
motivated (smoothing temperatures, surrogate derivatives, implicit adjoints) is
opt-in via configuration so that existing science results are untouched.

## 2. Where we start (audit results, 2026-07-19)

### 2.1 There is no AD usage or testing today

- No `jax.grad/jvp/vjp`, no `torch.autograd` use, no `custom_vjp/custom_jvp`,
  no `stop_gradient`, and no gradient test anywhere in `pyses/` or `tests/`.
  AD is a stated goal (README, `pyproject.toml`) with no design doc.
- The backend layer (`pyses/_config.py`: `Backend` protocol + `NumpyBackend`,
  `JaxBackend`, `TorchBackend`) exposes `jit/vmap/scan/shard_map/index_*` but
  no differentiation interface at all.
- The code is already written *defensively* for reverse mode in places: the
  guard-the-denominator-inside-`where` idiom appears with explicit
  "poisons reverse-mode AD" comments in `operations_2d/limiters.py:75,157`,
  `dynamical_cores/hyperviscosity.py:566`, and
  `homme/implicit_terms.py:223-227`; the torch `halo_exchange` shim was built
  on `torch.distributed.nn.functional` specifically so DSS has a correct
  autograd adjoint (`_config.py:1196`). The conventions exist; nothing
  exercises them.
- Every iteration in the hot path is a **fixed-count** `scan` or unrolled loop
  (deliberately: `implicit_terms.py:262-264` defers a tolerance early-exit
  because `lax.while_loop` breaks reverse mode). So the step is *mechanically*
  differentiable end to end today; the open questions are NaN safety, gradient
  quality at kinks/discrete choices, and tape cost.

### 2.2 Hazard inventory of one forward step (ranked)

**Tier 1 — Zerroukat vertical remap** (`dynamical_cores/vertical_remap.py`,
runs for dynamics *and* tracers every subcycle). The densest hazard cluster:

- Cell search is an integer op: `idxs = clip(sum(below) - 1, ...)` at
  `vertical_remap.py:70-71` followed by data-dependent
  `take_along_axis` gathers (`:75-76`, `:274-278`). Zero gradient a.e. with
  respect to the interface pressures that decide the containing cell;
  discontinuous output when a reference interface crosses a model interface.
- Unguarded divisions by layer thickness: `:78` (`zgam`), `:85`
  (`h = 1/zhdp`), `:121` (Thomas pivot; the sequential recurrence spreads one
  bad pivot's NaN across the whole column's gradient).
- The monotonicity filter (`:135-268`) is ~40 data-dependent
  `where/abs/min/max/clip` selects assembling integer filter codes and
  piecewise parabola coefficients — step functions with zero or kinked
  gradients on the remapped fields.

**Tier 2 — tracer limiter chain** (every tracer Euler substep):
`clip_and_sum_limiter` clip-and-redistribute masks
(`operations_2d/limiters.py:47-79`), element bounds via `jnp.min/max`
reductions (`tracer_transport/eulerian_spectral.py:93-94`) and the scatter-MAX
minmax DSS (`operations_2d/local_assembly.py:245-246`). Piecewise-smooth;
zero gradient in saturated cells; nondifferentiable at ties.

**Tier 3 — thermodynamics smoothness boundary** (HOMME + CAM-SE tendencies):
fractional-power Exner functions, `log(exner)`, and pervasive division by
`d_mass` / `d_phi` / pressure (e.g. `homme/thermodynamics.py:73-77`,
`cam_se/explicit_terms.py:338-357`). Smooth on the physical interior but
NaN-under-differentiation at zero/negative thickness or pressure — these
matter because *perturbed* states in JVP/VJP stress tests and optimization
loops will visit the boundary.

**Tier 4 — HEVI implicit solve** (`homme/implicit_terms.py`, non-hydrostatic
`RK3_5STAGE_HEVI` only): fixed 5-sweep Newton `scan` (`:341-344`) with a
backtracking line search full of `where`/`min` switches (`:202-242`), an
analytic tridiagonal Jacobian (`calc_dirk_jacobian`, `:33`), and a Thomas
solve (`:106`). Differentiable today by unrolling the whole Newton tape.

**Tier 5 — deep-atmosphere geometry**: `phi_to_z`
(`dynamical_cores/utils_3d.py:177`) has the sqrt-at-zero /
`(b - sqrt(...))`-denominator trap.

Not hazards: the DSS projection itself (linear scatter-add), the spectral
operators (static metric constants), sponge/hyperviscosity linear algebra,
and initialization-only bisection loops (host-side, outside the jitted step).

## 3. Remediation taxonomy and decision rules

Classify every finding from the stress harness into one of four categories.
The category, not intuition, decides the fix. "Smoothing" is the *last*
resort, not the first.

**A. Latent NaN bugs (fix unconditionally, exact).** Unguarded divides,
`log`/`pow` of possibly-nonpositive arguments, `sqrt` at zero, where the
*primal* is fine but a cotangent/tangent becomes NaN or Inf. Remedy: the
existing safe-divide idiom (`where`-guard the operand, not the result) and
argument clamping with `stop_gradient`-free floors. These change no bits of
the forward model for physical states and need no flags. Tier 1 divides and
Tier 3 boundary guards are all category A.

**B. Piecewise-smooth kinks (leave exact by default).** `min/max/abs/clip`
where a subgradient is well defined and correct a.e. — limiter saturation,
line-search switches. AD frameworks return a valid element of the
subdifferential; for step-scale time integration that is usually fine.
Remediate only if Phase-1 evidence shows optimization-relevant gradient noise,
and then via category D machinery.

**C. Zero-gradient / discontinuous discrete choices.** Integer search +
gather (remap cell location), integer filter codes, scatter-max bound
selection. Two remedies, in order of preference:

1. **Freeze the discrete choice** (`stop_gradient` on the index/mask): primal
   is bitwise exact; the gradient treats the cell assignment / filter branch
   as locally constant — correct a.e., the standard treatment in
   differentiable semi-Lagrangian schemes. Cheap, safe, first thing to try.
2. **Surrogate-derivative ("straight-through") smoothing**: keep the exact
   primal, but define the derivative via a smoothed surrogate through
   `custom_jvp/custom_vjp` — e.g. the remap search becomes
   `sum(sigmoid((pi_ref - pi_model)/tau))` *on the tangent path only*, giving
   the gradient a sensitivity to interface motion that the frozen-index rule
   discards. Use when experiments show the frozen-choice gradient is too
   blind (e.g. optimizing quantities that move mass across many levels).

**D. True primal smoothing (opt-in only).** Replacing hard `where/min/max`
with sigmoid/softmax/log-sum-exp *in the forward model itself*. This changes
the physics (and can break the limiter's exact mass conservation and
monotonicity guarantees), so it is only ever enabled by an explicit
configuration (`ad_smoothing` config dict carrying temperatures, default
off), and every smoothed op must ship with (i) a bound on forward deviation
vs. the exact op as tau -> 0 and (ii) a conservation check where applicable
(softmax redistribution weights preserve the mass sum by construction; verify
in tests).

Shared infrastructure for C/D: a small library of smoothed primitives
(`soft_where`, `soft_clip`, `soft_min/soft_max` (LSE), `soft_sign`,
`smooth_cell_weights` for the searchsorted-equivalent), dispatched per
backend like `index_add` is today, plus straight-through wrappers built on the
new backend `custom_vjp` shim.

## 4. Implicit differentiation as a backend feature

### 4.1 Design (following Blondel et al. 2022)

Add to the `Backend` protocol a pair of primitives (implemented in
increment 5; full contract in `pyses/implicit_diff.py`):

```python
def root_solve(self, residual_fn, solver_fn, x0, theta,
               linear_solve=None, maxiter=50):
    """x* with F(x*, theta) = 0; differentiated via the implicit function
    theorem instead of through solver_fn's iterations. x is a single
    array; theta an arbitrary pytree of arrays."""

def fixed_point_solve(self, T, solver_fn, x0, theta, ...):
    # sugar: F(x, th) = T(x, th) - x
```

Semantics: the primal runs `solver_fn` (opaque to AD); derivatives come from
`-∂₁F(x*, θ) J = ∂₂F(x*, θ)`:

- JVP: solve `A (Jv) = B v` with `A = -∂₁F`, `B = ∂₂F` (products via `jax.jvp`
  / `torch.func.jvp` of the *residual*, never materializing Jacobians).
- VJP: solve `Aᵀ u = v`, return `uᵀ B` via one `vjp` of the residual.

`linear_solve(matvec, rhs, x_star, theta, transpose)` is caller-supplied so
structured solvers (the tridiagonal Thomas solve, rebuilt from
`x_star`/`theta`) can replace the default; the fallback is a fixed-iteration
matrix-free CG on the normal equations (`implicit_diff.cg_normal_equations`),
jit-safe and backend-neutral.

Per backend:

- **JAX**: a `jax.custom_jvp`/`custom_vjp`-wrapped function (equivalently
  `lax.custom_root`); residual derivatives via `jax.jvp`/`jax.vjp`.
- **torch**: the repo's first `torch.autograd.Function`, written with
  `setup_context` so it composes with `torch.func.vmap/jvp/vjp` (the torch
  backend's `vmap` is real `torch.func.vmap`, so this matters); `backward`
  solves the transposed system, and a `jvp` staticmethod provides forward
  mode. This is also where torch gains the most: torch's `scan` shim is a
  Python unroll, so differentiating *through* the Newton loop is exactly the
  regime the paper shows is dominated by implicit differentiation in both
  memory and error (their Fig. 3: implicit-diff Jacobian error tracks the
  iterate error bound; unrolling is strictly worse).
- **numpy**: runs the solver, raises on differentiation (the harness uses
  numpy only as the finite-difference oracle).

The implementation lives with its peers: protocol declaration in
`pyses/_config.py` beside `scan`, per-backend bodies in each backend class;
the reusable IFT math (residual-JVP/VJP plumbing, CG fallback) in a new
`pyses/implicit_diff.py` so the three backend methods stay thin.

### 4.2 First consumer: the DIRK Newton solve

`calc_implicit_update` (`homme/implicit_terms.py:247`) is the textbook target:

- residual: the `w`/`phi` compatibility residual already formed at
  `:306-317`;
- `A` is exactly the analytic tridiagonal Jacobian `calc_dirk_jacobian`
  (`:33`) the solver already builds every sweep;
- the linear solve is the existing `solve_strict_diag_dominant_tridiag`
  (`:106`); its adjoint is the same Thomas solve with the `jacL`/`jacU` bands
  swapped — no new numerics.

Payoff: reverse mode stops taping 5 Newton sweeps + line search per step
(the line-search `where` switches drop out of the adjoint entirely, removing
a whole cluster of Tier-4 kinks); gradients become exact at the solver's
converged point rather than "gradient of an approximate solver"; and the
deferred tolerance-based early exit (`:262-264`) becomes legal, because a
`while_loop` under `root_solve` no longer needs to be reverse-differentiated.
Rollout is flag-gated (`use_implicit_diff`, default off) with an equivalence
test against differentiate-through-unrolled on well-converged states, using
the paper's Theorem-1 bound (Jacobian error is O(iterate error)) to set
tolerances.

Later candidates once the primitive exists: the remap Thomas solve (linear
custom-transpose VJP, cheap tape win), and — if gradients through
initialization are ever needed — the host-side bisection height inversion in
`initialization.py`.

## 5. Stress-test harness (built before any numerics change)

New suite `tests/fast_tests/ad_tests/` (auto-collected by the existing
`run_test_matrix.py` configs and CI workflows), selected like every other
suite by `PYSES_BACKEND` with a capability skip-guard (numpy runs only the
finite-difference oracles and the not-implemented contract).

Backend prerequisite: minimal AD shims on the protocol — `grad`, `jvp`,
`vjp`, `stop_gradient`, and `checkpoint`/remat (JAX: the obvious bindings;
torch: `torch.func` + `detach` + `torch.utils.checkpoint`) — so tests are
written once against `_be`.

Three layers, all float64:

1. **Kernel probes.** For each subsystem in the audit table (remap, limiter +
   minmax DSS, thermodynamics, tendencies, hyperviscosity, DIRK solve,
   projection, deep-atmosphere geometry): seed with cached fixture states plus
   adversarial perturbations (near-zero layer thickness, saturated limiter
   cells, interface-crossing remap states, degenerate ties), then assert
   (a) JVP and VJP are finite, (b) forward/reverse agree
   (`⟨v, Jw⟩ = ⟨Jᵀv, w⟩` dot-product test), (c) AD matches central finite
   differences away from known kinks, with a step-size sweep to localize
   disagreements onto the hazard inventory. Known category-B/C sites start as
   `xfail(strict=True)` entries — the inventory becomes executable.
2. **Single-step integration.** JVP and VJP of `advance_coupling_step` w.r.t.
   initial state and physics parameters for each dycore configuration in the
   fixture matrix (hydrostatic/NH, HOMME/CAM-SE, tracers on/off, remap
   on/off, diffusion on/off) — the config flags double as the bisection tool
   for localizing any NaN to a subsystem. `jax.debug_nans` / torch anomaly
   mode in the failure path.
3. **Short-trajectory tests.** N-step rollout under `scan` (bypassing the
   host-side NaN-assert generator loop in `run_dycore.py:357-380`), reverse
   mode with the new `checkpoint` shim, watching for tape memory and gradient
   blowup — this is what data assimilation will actually run.

## 6. Incremental plan on `ad-hardening`

Each increment is a self-contained commit (or small stack) that keeps all
three backends' existing suites green; gradient tests land with the increment
that makes them pass (or as strict xfails documenting a known hazard).

1. **This document.**
2. **Backend AD shims** (`grad/jvp/vjp/stop_gradient/checkpoint`) + smoke
   tests on toy functions per backend.
3. **Harness layer 1** (kernel probes seeded from the audit) — expected to
   *fail/xfail* on Tier-1 divides and discrete ops; this pins the baseline.
4. **Category-A fixes**, scoped by the increment-3 measurements: the one
   proven bite is the `d_phi` divide in `homme/thermodynamics.py`
   (`eval_pressure_exner_nonhydrostatic`), guarded with the safe-divide
   idiom — its xfail flips to a pass. The suspected remap divides
   (`vertical_remap.py:78,85,121`) measured safe (finite gradients even at
   1e-12 layer thickness) and the `phi_to_z` sqrt discriminant only
   vanishes at z ≈ a/3, far outside the atmosphere — both stay untouched,
   with the layer-1 probes standing as sentinels. Bitwise-identical
   forward model for physical states (asserted by existing suites).
5. **Implicit-diff primitive**: `root_solve`/`fixed_point_solve` +
   `pyses/implicit_diff.py`, validated on toys (ridge regression as in the
   paper; a batched tridiagonal root problem mirroring the DIRK structure)
   on jax and torch, forward and reverse.
6. **DIRK adoption** behind `use_implicit_diff`, with
   unrolled-vs-implicit gradient equivalence tests and a tape-memory check.
7. **Discrete-choice remediation** (category C): frozen-index rule for the
   remap search/gather + minmax bounds; straight-through surrogate variants
   where layer-2 evidence demands them.
8. **Smoothing library** (category D, opt-in `ad_smoothing` config):
   `soft_*` primitives + conservation/deviation tests; applied only where
   categories A–C left documented gradient-quality gaps.
9. **Harness layers 2–3 in CI** (single-step + short-trajectory in the jax
   and torch workflows), closing the loop.

Ordering rationale: 2–3 give measurements before opinions; 4 is pure bug-fix;
5–6 remove the largest *structural* AD cost and de-risk the torch autograd
integration early; 7–8 are physics-adjacent and deliberately last, gated on
evidence from the harness rather than on the audit's suspicions.

## 7. Risks and non-goals

- **Sharded/MPI gradients are out of scope** for this branch: `shard_map`
  regions and the MPI torch path add collective-adjoint questions; the
  implicit-diff primitive is specified single-device first, and the harness
  runs unsharded. (The torch `halo_exchange` adjoint already exists, so the
  extension is natural later.)
- **`torch.compile` interaction**: increments land under eager torch
  (`PYSES_JIT_COMPILE=0`); compiling `autograd.Function` +
  `torch.func` compositions is validated separately before being claimed.
- **Conservation vs. smoothing**: any category-D change that cannot prove
  mass/energy conservation to round-off in tests stays off by default,
  permanently.
- **No solver behavior changes ride along**: tolerance early-exit for the
  Newton solve is enabled only after increment 6, as its own change.
