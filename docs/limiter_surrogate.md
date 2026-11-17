# The straight-through limiter surrogate

How pySEs differentiates the tracer limiter when `limiter_tau` is set: what
the exact limiter does, why its derivative is ill-defined, how the surrogate
replaces it, and what changed between the three versions (v1, v2, v2.1).
Code: `pyses/operations_2d/limiters.py` (`clip_and_sum_limiter`,
`_surrogate_pass`), `pyses/smoothing.py` (`straight_through`,
`soft_reduce_max/min`), `pyses/operations_2d/local_assembly.py`
(`minmax_scalar`), `pyses/tracer_transport/eulerian_spectral.py`
(`calc_minmax`). Tests: `tests/fast_tests/ad_tests/test_limiter_surrogate.py`
and the `*limiter_surrogate*` tests in `test_step_gradients.py`.

## 1. What the exact limiter does

The clip-and-sum limiter acts on one element and one level at a time. With
`c` the scaled mass of each node (mass-matrix weight times layer mass) and
`q` the mixing ratio after an advection stage:

1. **Bounds.** `qmin`, `qmax` are the smallest and largest stage-start values
   over the element and its neighbours (element min/max, then a DSS min/max
   over shared nodes, then min/max over the element's nodes). If the element
   mean lies outside them, the violated bound is relaxed to the mean.
2. **Clip.** `q1 = clamp(q, qmin, qmax)`; the mass removed is
   `A = Σ c (q − q1)`, positive if more was cut from overshoots than added to
   undershoots.
3. **Redistribute.** If `A > 0`, the removed mass goes back to the cells with
   room below `qmax`, in proportion to that room:
   `q2 = q1 + A · (qmax − q1) / Σ c (qmax − q1)`. If `A < 0`, mass is taken
   from cells with room above `qmin` in the same way. Element mass is
   conserved; values stay within the bounds.

## 2. Why its exact derivative is a problem

The limiter is continuous, and differentiable almost everywhere, but the
model puts it *on* its switches, not near them:

* the bounds tie by construction: an element's extremum on a shared edge
  node is the same value in both elements, and interior nodes all hold the
  element's own extremum after the DSS step; the derivative of a min/max
  over a tied set is split over whichever values happen to tie;
* a clipped cell is set exactly equal to its bound, and the next stage
  compares values against bounds computed from that output;
* on a plateau (a constant field, e.g. a background mixing ratio, or the
  dry-air species) every cell sits within ulps of both bounds, and the sign
  of `A` is roundoff.

At each of these, automatic differentiation picks one side, and roundoff
decides which. Measured on the moist baroclinic-wave step: a 1-ulp change of
the input moves J·v by 4e−6 (wind) to 8e−2 (water vapour); separately
compiled jvp and vjp programs disagree at 5e−6; and rematerialization
(checkpointing) changes a 2-step trajectory gradient by up to 140% per
entry. The derivative is valid (an element of the generalized gradient) but
not reproducible. Switching off just the bounds' derivative brings J·v to
roundoff, so the bound ties are the main source; the clip and redistribution
switches matter on plateaus.

## 3. Straight-through: exact value, surrogate derivative

With the flag on, the limiter returns

    output = S + stop_gradient(E − S)

where `E` is the exact limiter and `S` a smooth surrogate. The value is `E`
(up to one rounding); every derivative, forward or reverse, is the
derivative of `S`. Because both modes differentiate the same function, the
dot-product identity is exact, and because `S` is smooth on a scale `τ` much
larger than roundoff, a 1-ulp change of the state moves its derivative by
O(ulp / τ). With the flag off, `S` is never built and the code path is the
original one, bitwise.

## 4. The surrogate (v2.1)

Everything is built from one primitive, the softplus of width `τ`:

    sp_τ(x) = τ log(1 + exp(x / τ)),   sp_τ'(x) = σ(x / τ) ∈ (0, 1),
    sp_τ(x) − sp_τ(−x) = x,   sp_τ(x) → max(x, 0) as τ → 0.

It is a smoothed `max(x, 0)` whose slope is a smoothed step. For each
element-level, with the same relaxed bounds as the exact limiter:

| Step | Exact | Surrogate |
| --- | --- | --- |
| overshoot | `max(q − qmax, 0)` | `e = sp_τ(q − qmax)` |
| undershoot | `max(qmin − q, 0)` | `d = sp_τ(qmin − q)` |
| clipped value | `clamp(q, qmin, qmax)` | `q1 = q − e + d` |
| removed mass | `A = Σ c (q − q1)` | `A = Σ c (e − d)` (the same identity) |
| direction | `A > 0` or `A < 0` | `A₊ = sp_τA(A)`, `A₋ = sp_τA(−A)`, so `A = A₊ − A₋` exactly |
| room to receive | `qmax − q1`, `q1 − qmin` | `r↑ = sp_τ(qmax − q1)`, `r↓ = sp_τ(q1 − qmin)` |
| result | `q1 + A r↑ / Σ c r↑` (or the `A < 0` branch) | `q2 = q1 + A₊ r↑ / D↑ − A₋ r↓ / D↓` with `D↑ = Σ c r↑`, `D↓ = Σ c r↓` |

and the bounds take a soft tie split: their values are the exact min/max,
their derivative is a softmax-weighted mean of the candidates,
`Σ w x / Σ w` with `w = exp((x − max) / τ)` (written as
`max + Σ w (x − max) / Σ w`, so ties contribute exact zeros), at all three
reductions. The bound relaxation is left exact. `τ` is in mixing-ratio
units (absolute, not relative to the bound range) and
`τ_A = τ × mean(c)` by default.

**The tangent.** Differentiating the table gives, cell by cell,

    dq1 = (1 − σ_e − σ_d) dq + σ_e dqmax + σ_d dqmin,
    σ_e = σ((q − qmax) / τ),   σ_d = σ((qmin − q) / τ),

a convex combination of the cell's own perturbation and the bounds'
(`σ_e + σ_d ≤ 1` because `qmin ≤ qmax`). Far inside the bounds it is `dq`,
far outside it is the bound's perturbation, and within `τ` of a bound it
blends the two, where the exact derivative jumps. The redistribution
contributes `dA₊ = σ(A/τ_A) dA` and `dA₋ = −σ(−A/τ_A) dA` (summing to `dA`),
the channel weights `r↑ / D↑`, `r↓ / D↓` and their derivatives
`dr↑ = σ((qmax − q1)/τ)(dqmax − dq1)`, and so on; the bounds' perturbations
come from the soft tie split, a positive-weight average of the candidates'.

**The adjoint** is the transpose of this linear map, which reverse-mode AD
produces from the same surrogate. Read backwards: an adjoint sensitivity at
an output cell is shared, through the redistribution, with every cell of its
element-level in proportion to the smoothed rooms; at each cell it passes to
the cell's own input with weight `1 − σ_e − σ_d` and to the bounds with
weights `σ_e`, `σ_d`; from a bound it is spread by the softmax weights over
the neighbourhood values that nearly attain it.

**Properties** (pinned by the kernel tests):

1. *Non-expansive and monotone.* Every smoothed piece is monotone with slope
   in [0, 1], like the exact clip, so the tangent of one application never
   amplifies (`‖J v‖_c ≤ ‖v‖_c`) and a single-cell increase moves every output
   the same way or not at all. Any number of limiter applications therefore
   compose to something bounded.
2. *Exactly mass-conserving.* `Σ c q2 = Σ c q1 + A₊ − A₋ = Σ c q`, because
   each channel's weights `c r / D` sum to one; so the tangent conserves
   element mass to roundoff, plateaus included.
3. *Equal to the exact derivative away from switches,* up to terms of order
   `exp(−distance / τ)`, with one exception: a clipped cell sits exactly at
   zero room (the "receive / do not receive" switch), and the floored room
   gives it weight ≈ `τ ln 2`, an O(`τ / room`) relative difference (≈ 1e−8
   at `τ = 1e−8`).
4. *Roundoff-stable.* Every switch is replaced by a function that varies on
   the scale `τ`, so the derivative is defined independently of how roundoff
   lands.

## 5. What changed, and why

| | v1 | v2 | v2.1 (current) |
| --- | --- | --- | --- |
| clip | `σ((q−qmax)/τ)·(q − qmax)` | `sp_τ(q − qmax)` | same as v2 |
| direction | blend `s = σ(A/τ_A)`, scaled by `A` | `A₊ − A₋` softplus channels | same as v2 |
| rooms | exact | exact | **softplus-floored** |
| guard on `D` | hard `where(D > thresh)` | soft `A / (D + ε_D)` | **none needed** |
| tangent slope of the clip | `σ + x σ'` ∈ [−0.10, 1.10] | `1 − σ_e − σ_d` ∈ [0, 1] | same |
| failure | expanded ~10% per application: J·v grew to 3e16 over 200 steps on a plateau | on plateaus `D ≈ ε_D`, and the softplus leak in `A±` divided by ≈ 0: 14% mass error, wrong-sign responses | — |

* **v1 → v2.** `mask(x) · x` is a *swish*: its slope overshoots 1 just
  inside the bound and goes negative just outside. One application expands a
  perturbation by up to 10%; a 6-day window has thousands of applications.
  The exact clip has slope 0 or 1. Softplus has slope in (0, 1). The same
  applied to `A · σ(A)`, which v2 replaces by two softplus channels.
* **v2 → v2.1.** On a plateau the exact rooms are roundoff, so v2's
  denominators `D` were of the order of its guard `ε_D`, while softplus
  leaks `≈ τ ln 2` of mass into both `A₊` and `A₋` even with nothing
  clipped. The resulting weights (≈ 100) cancelled in value but not in the
  tangent. Flooring the rooms with the same softplus keeps
  `D ≥ ≈ τ ln 2 Σ c`, removes the guard, and makes conservation exact.

Rule for any future smoothed switch: never write `mask(x) · x`; use a
monotone convex primitive (softplus), and add the switch to the
non-expansiveness test.

## 6. What the surrogate gradient is, and is not

It approximates the gradient of a *mollified* limiter, `f_τ(x) ≈ E[f(x + τz)]`,
evaluated at the exact trajectory: reproducible, exactly transposed,
consistent across checkpointing and compilation, and bounded over long
windows. It is not the derivative of the exact model. A Taylor test of the
exact model against it has a first-order floor `‖(J − J_τ) v‖` as ε → 0, and
it is closest to finite differences at steps whose perturbation of `q` is
about `τ`. For optimization, pair it with a backtracking (Armijo) line
search rather than a strict Wolfe search, and choose `τ` well above roundoff
(1e−16 relative) and well below the physical overshoots (1e−3 to 1e−2 of
`q`); `1e−8` is the default used in the tests.

## 7. Using it

    timestep_config = init_timestep_config(..., limiter_tau=1e-8)        # limiter_tau_A optional

`limiter_tau` is stored as `timestep_config["tracer_advection"]["ad_smoothing"]`
and reaches the limiter and the bound reductions as a static argument; with
it absent every code path and compiled program is the original one.
