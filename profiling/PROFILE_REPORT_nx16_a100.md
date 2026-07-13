# pySES JAX-GPU profiling: NX16 moist baroclinic wave on an A100

**Date:** 2026-07-13 · **Machine:** NCAR Casper, 1× A100-SXM4-80GB · **JAX** 0.8.2 (cuda12 wheels)
**Case:** quasi-uniform cubed sphere `nx=16, npt=4` (1536 elements, 24 576 columns), 30 levels (`cam30`),
moist baroclinic wave (`init_baroclinic_wave_state`, `mountain=False`), default configs from
`init_default_config` (tensor hyperviscosity, RK3-5-stage dynamics).
Physics step 1700 s; subcycles: tracer 3 × dynamics 2 × hypervis 1 × sponge 1.

Reproduce with `profiling/profile_baro_wave.py` (+ `casper_a100_profile.pbs`,
`casper_nsys_rerun.pbs`, `casper_validate.pbs`). Raw outputs:
`/glade/derecho/scratch/owhughes/pyses_profiling/run_5184109` (main),
`.../validate_5185037` (validation micro-benchmarks and flag runs).

## 1. Headline numbers

| configuration | ms / coupling step | simulated years / wall-day | peak device mem |
|---|---|---|---|
| homme_hydrostatic, fp64 | **238.2** | 19.6 | 0.68 GiB |
| homme_hydrostatic, fp32 | **50.0** | 93.1 | 0.33 GiB |
| cam_se, fp64 | 282.0 | 16.5 | 0.74 GiB |

- First call (compile + 1 step, warm persistent cache): ~10 s. Setup (grid/init): ~9 s.
- Per-step timing with a sync every step equals free-running (pipelined) throughput
  → **host dispatch is not a bottleneck** (XLA runs the step as CUDA graphs).
- Device occupancy during steady state: **98.4 % busy** — the GPU is saturated,
  but with grossly inefficient kernels (see §3).
- XLA cost analysis of one compiled step: 9.74 GFLOP, **96 GB memory traffic**.
  Achieved: ~41 GFLOP/s (0.4 % of fp64 peak) and ~400 GB/s effective
  (~20 % of HBM peak) → the model is memory-traffic-bound, and most of that
  traffic moves through very inefficient kernels.

## 2. Where the time goes

Component wall-times (standalone jitted calls, median of 10; they reconstruct the
measured full step to within 1 %):

| component | ms/call | calls/step | ms/step | % of step |
|---|---|---|---|---|
| RK5 dynamics step (`advance_dynamics_ullrich_5stage`) | 19.4 | 6 | 116.2 | 48.8 |
| hyperviscosity (`advance_hypervis_euler`) | 12.7 | 6 | 76.5 | 32.1 |
| tracer advection (`advance_tracers`) | 7.9 | 3 | 23.7 | 9.9 |
| vertical remap dynamics (`remap_dynamics`) | 2.5 | 3 | 7.6 | 3.2 |
| sponge (`advance_sponge_euler`) | 1.1 | 6 | 6.5 | 2.7 |
| vertical remap tracers (`remap_tracers`) | 1.7 | 3 | 5.1 | 2.1 |
| renormalize dry-air species | 0.07 | 3 | 0.2 | 0.1 |
| single tendency eval (`dynamics_tendency`), for reference | 4.7 | (30) | — | — |
| NaN checks (host sync), for reference | 1.4 + 0.6 | interval | — | — |

Device-level view (JAX profiler trace and nsys `--cuda-graph-trace=node` agree):
~3 400–4 000 kernels/step, mean 60–70 µs, 1 620 D2D memcpys/step (3.8 ms).

**GEMM kernels consume 73.8 % of all device time** (176.8 ms/step across 371
launches/step: cutlass `d884gemm` and `sm80_xmma_gemm_f64` tensor-op kernels at
~880–950 µs each, plus smaller ones). The remaining fused-loop time attributes as:
RK5 tendency fusions 14.8 %, hypervis 3.6 %, vertical remaps 4.7 %, tracers 1.6 %,
sponge 0.5 %.

## 3. Root cause: tiny-inner-dimension contractions are dispatched to cuBLAS

The optimized HLO contains **283 `__cublas$gemm` custom calls** per step program.
Nearly all of the expensive ones are *batched contractions over a length-2 axis* —
per-GLL-point 2×2 metric-tensor transforms, batch = 24 576 points:

| einsum | source |
|---|---|
| `fijs,fijgs->fijg` (`physical_to_contravariant`) | [operators.py:402](../pyses/operations_2d/operators.py#L402) |
| `fijg,fijgs->fijs` (`contravariant_to_physical`-type gradient) | [operators.py:38](../pyses/operations_2d/operators.py#L38) |
| `fijs,fijsg->fijg` (`physical_to_covariant`) | [operators.py:429](../pyses/operations_2d/operators.py#L429) |
| `fijs,fijts->fijt` (viscosity-tensor application) | [operators.py:175](../pyses/operations_2d/operators.py#L175) |
| `fijkc,fijcs->fijks` / `fijks,fijcs->fijkc` (hypervis harmonic) | `eval_hypervis_harmonic` |

These run one thread-block per GLL point (`grid (1,1,24576)`, 25 % occupancy);
with an inner dimension of 2 a tensor-op GEMM tile is ~94 % wasted. Measured
micro-benchmark on the exact shapes (`bench_metric_transform.py`, A100 fp64):

| shape `[24576,K,2] × [24576,2,2]` | einsum → cuBLAS | unrolled multiply-add | speedup |
|---|---|---|---|
| K = 30 | 888 µs (27 GB/s) | 80 µs (303 GB/s) | **11×** |
| K = 150 | 1824 µs (65 GB/s) | 187 µs (635 GB/s) | **10×** |

The unrolled form (`jnp.sum(a[:, :, None, :] * m[:, None, :, :], -1)` or explicit
FMA) runs at ~copy speed and fuses into neighboring elementwise kernels.

Two important corollaries, both verified:

- `XLA_FLAGS=--xla_gpu_gemm_rewrite_size_threshold=10000` does **not** stop the
  rewrite (micro-benchmark still `cublas=True`; full fp64 model unchanged at
  237.9 ms/step, and still unchanged at threshold=1 000 000). There is no
  flag-only fix for fp64 — the einsums must be rewritten. (Curiously the same
  flag *does* buy fp32 a further 13 %: 50.0 → 43.7 ms/step, 107 SYPD.)
- In **fp32 XLA never routes these dots to cuBLAS** (they stay fused). That —
  not raw FLOP rate — is most of why fp32 is 4.8× faster than fp64 today.

The 1-D GLL derivative contractions (`fij,ki->fkj` with the 4×4 derivative
matrix, reshaped to `[4 × ~2.3M]` cuBLAS calls) are *not* pathological
(~220 GB/s, ~6 ms/step) but drag along `loop_transpose`/`loop_concatenate`
fusions (~15–20 ms/step) that exist only to feed cuBLAS-friendly layouts.

## 4. Secondary observations

1. **Vertical remap (Zerroukat) runs outside CUDA graphs.** Its data-dependent
   `while` loops execute ~738 microkernels/step of 2–3 µs each plus most of the
   1 620 D2D memcpys — the only work not captured into command buffers.
   ~11 ms/step now (~5 %), but it becomes 15–30 % once the GEMM problem is fixed,
   and it is the piece that prevents whole-step graph capture. Two ~910 µs
   `take_along_axis` gathers in `remap_tracers` also stand out.
2. **NaN guard** (`check_dynamics_nan` + `check_tracers_nan`) costs ~2 ms plus a
   host sync per physics step at the default `nan_check_interval=1`.
3. **cam_se** shows the same profile shifted up ~18 % (T↔θ conversions add work).
4. **Memory headroom is enormous** (0.7 / 80 GiB). NX16 already saturates the
   device with kernels, but there is room for much larger grids, more levels,
   or vmapped ensembles without touching the memory wall.
5. Compile times are a non-issue (~10 s warm, persistent cache enabled via
   `JAX_COMPILATION_CACHE_DIR`).

## 5. Ranked improvement avenues

| # | change | expected effect (NX16 fp64) | effort |
|---|---|---|---|
| 1 | **Unroll the 2×2 metric/viscosity-tensor einsums** into broadcast-multiply-sums (≈6 call sites in `operations_2d/operators.py` + hypervis) | GEMM time 177 → ~25 ms/step ⇒ step 238 → **~90 ms (≈2.6×)**; also helps torch backend (same `bmm` pathology) and fp32 (~25 %) | small, local |
| 2 | **Precision policy**: fp32 (or mixed fp32/fp64) production runs | measured **4.8×** today; combined with #1 est. step ~40–45 ms (**~5×**, ~110 SYPD) | small code-wise; needs numerics validation (conservation, DSS summation order) |
| 3 | **Restructure the Zerroukat remap** to fixed-trip `scan`/vectorized searchsorted+cumsum form | removes ~740 microkernels + most D2D copies per step; unlocks whole-step CUDA-graph capture; matters after #1/#2 (would be ~25 % of the remaining step) | medium |
| 4 | **Derivative-op layout**: unroll the npt=4 stencil (or one batched contraction per stage) to eliminate feeding transposes/concats | ~15–25 ms/step of transpose/concat/gemm traffic | medium |
| 5 | **Foundational: fused operator pipelines** (Pallas kernel or aggressive restructuring for gradient→metric→DSS chains; pack the per-field dict into one array to widen fusions) | XLA cost model shows 96 GB/step of traffic vs ~5–10 GB/step algorithmically necessary ⇒ ceiling ≈ **10–20 ms/step fp64** (~10×) if pursued to the end | large; profile-guided, incremental |
| 6 | Housekeeping: default `nan_check_interval>1` in production loops; keep buffer donation on | ~1–2 % now | trivial |

Suggested order: #1 (validated, local, big), then re-profile; #2 in parallel as a
numerics study; #3/#4 next; treat #5 as the long-term target informed by the
re-profile.

## 6. Post-fix re-profile (avenue #1 implemented)

Avenue #1 was implemented as `pointwise_matvec` in
[operators.py](../pyses/operations_2d/operators.py) (5 call sites) plus the two
`physical_to_cartesian` transforms in
[hyperviscosity.py](../pyses/dynamical_cores/hyperviscosity.py). Correctness:
all seven contractions match the original einsums to machine epsilon, and the
operator/hypervis/run-model fast tests pass 65/65 on both the numpy and jax
backends. Re-profiled on an A100 (run `reprofile_5185327`):

| configuration | baseline | post-fix | speedup |
|---|---|---|---|
| fp64 ms/step (SYPD) | 238.2 (19.6) | **84.4 (55.2)** | **2.82×** |
| fp32 ms/step (SYPD) | 50.0 (93) | **36.9 (126)** | 1.36× |

fp64-baseline → fp32-post-fix is **6.5×** end to end. New fp64 step budget:
RK5 dynamics 38.3 ms (46 %), hypervis 20.1 ms (24 %), tracer advection 7.0 ms,
vertical remaps 12.5 ms (15 %), sponge 2.8 ms. Device still 96.8 % busy with
3 148 kernels/step (mean 27.7 µs).

The ~900 µs 2×2 GEMMs are gone. Remaining cuBLAS time is ~31 ms/step (37 % of
device time), now dominated by the **4×4 GLL derivative-matrix contractions**
(cutlass ~180 µs × ~150/step + `gemmSN` thin GEMMs) and their feeding
transpose/concatenate fusions — i.e. avenue #4 is the new head of the list,
followed by the Zerroukat remap (#3, ~15 % incl. its ~970 µs `take_along_axis`
gathers) and precision (#2). One-time cost: first compile of the new program
rose to ~3 min (XLA fusion autotuning); cached thereafter.

## 7. Second fix: GLL derivative contractions unrolled (avenue #4)

The eight derivative-matrix einsums (`horizontal_gradient`, `horizontal_divergence`,
`horizontal_vorticity`, and the weak gradient/curl/divergence forms) were
rewritten through a `gll_matvec` helper in
[operators.py](../pyses/operations_2d/operators.py) — a batch-agnostic
broadcast-multiply-sum over one GLL axis, with the quadrature weights folded
into a premultiplied `(w[:,None]*D).T` matrix for the weak forms. Verified:
8/8 exact equivalence (with and without vmap-style batch dims) and 65/65 fast
tests on both numpy and jax backends.

Re-profiled on a **Derecho A100-SXM4-40GB** (develop queue; note: ~25 % less
HBM bandwidth than the Casper 80 GB part used above, so the speedups below are
conservative):

| configuration | after fix #1 (80 GB) | after fix #2 (40 GB) | speedup |
|---|---|---|---|
| fp64 ms/step (SYPD) | 84.4 (55.2) | **49.7 (93.7)** | **1.70×** |
| fp32 ms/step (SYPD) | 36.9 (126) | **26.7 (174.6)** | 1.38× |

Cumulative from the fp64 baseline: **4.8×** (238.2 → 49.7 ms); fp64-baseline →
fp32-now is **8.9×**. The compiled step now contains **zero cuBLAS custom
calls** — the entire coupling step is XLA fusions — and XLA's per-step memory
traffic dropped 81 → 48 GB (the GEMM-feeding transposes/concatenates are gone).
Device is still ~96 % busy with 2 623 kernels/step (mean 21 µs).

New fp64 device-time ranking: RK5 dynamics fusions 31.5 %, **vertical remaps
26.4 %** (`remap_dynamics` 17.6 % + `remap_tracers` 8.8 %, led by the ~1.1 ms
`take_along_axis` gathers and the out-of-graph while-loop microkernels),
hypervis 12.5 %, tracers 7.1 %. The Zerroukat remap restructure (avenue #3) is
now the clear next target, followed by wider fusion/precision work (#5/#2).

## 8. Methodology notes

- `profile_baro_wave.py` modes: `timing` (synced + pipelined step times),
  `components` (standalone jitted sub-functions × subcycle counts; reconstructs
  the step within 1 %), `cost` (XLA `cost_analysis`/`memory_analysis` + optimized
  HLO with `op_name` metadata), `trace` (jax.profiler → Perfetto/XSpace),
  `nsys` (cudaProfilerStart/Stop capture window).
- nsys **must** use `--cuda-graph-trace=node`, otherwise kernels inside XLA's
  command buffers are invisible (only the remap while-loops show up).
- `analyze_hlo_hotspots.py` joins nsys kernel rows to HLO `op_name`/source;
  `analyze_jax_trace.py` does the same for the JAX trace (works inside graphs,
  but cuBLAS kernels report `hlo_op=command_buffer` and stay unattributed —
  their identity was established from HLO shapes + grid signatures instead).
