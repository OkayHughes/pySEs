#!/usr/bin/env python
"""GPU profiling harness for the pySES JAX backend.

Runs an NX``--nx`` moist baroclinic wave (patterned on
``tests/slow_tests/test_run_model.py::test_theta_baro_wave`` but on the
quasi-uniform grid) and measures where time goes.

Modes (comma-separated in ``--mode``):

  timing      Wall-clock per coupling step: compile/first-call time, per-step
              stats with a per-step device sync, and a pipelined (no
              intermediate sync) throughput number to expose dispatch overhead.
  components  Times each already-jitted sub-function of the coupling step
              (RK5 dynamics, single tendency eval, hypervis, sponge, tracer
              advection, remaps, renormalization, NaN checks) standalone and
              reconstructs the per-step budget from the subcycle counts.
  cost        XLA cost/memory analysis of the full coupling step; saves the
              optimized HLO (with source metadata) for hotspot mapping.
  trace       Captures a jax.profiler trace (xplane + perfetto) of a few steps.
  nsys        Runs a few steps between cudaProfilerStart/Stop so that
              ``nsys profile -c cudaProfilerApi`` captures only steady-state.

Environment (PYSES_BACKEND etc.) is set *before* pyses/jax import, so run this
script directly rather than importing it.
"""
import argparse
import json
import os
import statistics
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--nx", type=int, default=16)
  p.add_argument("--npt", type=int, default=4)
  p.add_argument("--model", default="homme_hydrostatic")
  p.add_argument("--mode", default="timing")
  p.add_argument("--steps", type=int, default=20)
  p.add_argument("--warmup", type=int, default=3)
  p.add_argument("--repeats", type=int, default=10,
                 help="repeats per component in components mode")
  p.add_argument("--profile-steps", type=int, default=3,
                 help="steps captured in trace/nsys modes")
  p.add_argument("--float32", action="store_true")
  p.add_argument("--no-donate", action="store_true")
  p.add_argument("--outdir", default=None)
  return p.parse_args()


def setup_env(args):
  os.environ.setdefault("PYSES_BACKEND", "jax")
  if args.float32:
    os.environ["PYSES_USE_DOUBLE"] = "0"
  sys.path.insert(0, REPO)


def build_case(args):
  """Build grid, configs, and the initial baroclinic-wave state."""
  from tests.test_data.mass_coordinate_grids import cam30
  from pyses.analytic_initialization.moist_baroclinic_wave import (
      init_baroclinic_wave_config, init_baroclinic_wave_state)
  from pyses.mesh_generation.element_local_metric import init_quasi_uniform_grid_elem_local
  from pyses.dynamical_cores.mass_coordinate import init_vertical_grid
  from pyses.dynamical_cores.model_info import models
  from pyses.dynamical_cores.model_config import init_default_config, hypervis_opts

  model = models[args.model]
  t0 = time.perf_counter()
  h_grid, dims = init_quasi_uniform_grid_elem_local(args.nx, args.npt,
                                                    calc_smooth_tensor=True)
  v_grid = init_vertical_grid(cam30["hybrid_a_i"], cam30["hybrid_b_i"],
                              cam30["p0"], model)
  physics_config, diffusion_config, timestep_config = init_default_config(
      args.nx, h_grid, v_grid, dims, model,
      hypervis_type=hypervis_opts.variable_resolution)
  test_config = init_baroclinic_wave_config(model_config=physics_config)
  state = init_baroclinic_wave_state(h_grid, v_grid, physics_config,
                                     test_config, dims, model, mountain=False)
  setup_s = time.perf_counter() - t0
  return dict(model=model, h_grid=h_grid, dims=dims, v_grid=v_grid,
              physics_config=physics_config, diffusion_config=diffusion_config,
              timestep_config=timestep_config, state=state, setup_s=setup_s)


def log_run_info(args, case, out):
  import jax
  tc = case["timestep_config"]
  dims = case["dims"]
  dev = jax.devices()[0]
  info = {
      "jax_version": jax.__version__,
      "device": f"{dev.platform}:{dev.device_kind}",
      "num_devices": len(jax.devices()),
      "x64": jax.config.jax_enable_x64,
      "nx": args.nx, "npt": args.npt, "model": args.model,
      "num_elem": dims["num_elem"],
      "nlev": int(case["v_grid"]["hybrid_b_m"].shape[0]),
      "physics_dt_s": float(tc["physics_dt"]),
      "tracer_subcycle": int(tc["tracer_subcycle"]),
      "dynamics_subcycle": int(tc["dynamics_subcycle"]),
      "hypervis_subcycle": int(tc["hypervis_subcycle"]),
      "sponge_subcycle": int(tc["sponge_subcycle"]),
      "dynamics_dt_s": float(tc["dynamics"]["dt"]),
      "setup_s": case["setup_s"],
  }
  print("=== run info ===")
  for k, v in info.items():
    print(f"  {k}: {v}")
  out["run_info"] = info
  return info


def make_stepper(case, donate, nan_check_interval=10**9):
  import jax
  from pyses.dynamical_cores.run_dycore import init_simulator
  simulator = init_simulator(case["h_grid"], case["v_grid"],
                             case["physics_config"], case["diffusion_config"],
                             case["timestep_config"], case["dims"],
                             case["model"],
                             nan_check_interval=nan_check_interval,
                             donate_state=donate)
  # Donation invalidates the input buffers; hand each stepper its own copy so
  # later modes in the same process can still use case["state"].
  state0 = jax.tree_util.tree_map(
      lambda x: x.copy() if hasattr(x, "copy") else x, case["state"])
  return simulator(state0)


def run_timing(args, case, out):
  import jax
  gen = make_stepper(case, donate=not args.no_donate)

  t0 = time.perf_counter()
  _, st = next(gen)
  jax.block_until_ready(st)
  first_call_s = time.perf_counter() - t0
  print(f"first call (compile + 1 step): {first_call_s:.2f} s")

  for _ in range(args.warmup):
    _, st = next(gen)
  jax.block_until_ready(st)

  # Per-step timing with a sync every step.
  per_step = []
  for _ in range(args.steps):
    t0 = time.perf_counter()
    _, st = next(gen)
    jax.block_until_ready(st)
    per_step.append(time.perf_counter() - t0)

  # Pipelined: dispatch all steps, sync once. If markedly faster per step than
  # the synced loop, host-side dispatch latency is being hidden by the queue.
  t0 = time.perf_counter()
  for _ in range(args.steps):
    _, st = next(gen)
  jax.block_until_ready(st)
  pipelined_s = (time.perf_counter() - t0) / args.steps

  dev = jax.devices()[0]
  mem = dev.memory_stats() or {}
  physics_dt = float(case["timestep_config"]["physics_dt"])
  mean_s = statistics.mean(per_step)
  sypd = physics_dt / mean_s * 86400.0 / (86400.0 * 365.0)
  res = {
      "first_call_s": first_call_s,
      "per_step_s": per_step,
      "mean_s": mean_s,
      "median_s": statistics.median(per_step),
      "min_s": min(per_step),
      "max_s": max(per_step),
      "pipelined_mean_s": pipelined_s,
      "simulated_years_per_wallday": sypd,
      "peak_bytes_in_use": mem.get("peak_bytes_in_use"),
      "bytes_in_use": mem.get("bytes_in_use"),
  }
  print("=== timing ===")
  print(f"  per-step (synced):    mean {mean_s*1e3:.1f} ms  median {res['median_s']*1e3:.1f} ms  "
        f"min {res['min_s']*1e3:.1f} ms  max {res['max_s']*1e3:.1f} ms")
  print(f"  per-step (pipelined): {pipelined_s*1e3:.1f} ms")
  print(f"  simulated years / wall day: {sypd:.2f}")
  if res["peak_bytes_in_use"]:
    print(f"  peak device memory: {res['peak_bytes_in_use']/2**30:.2f} GiB")
  out["timing"] = res


def _advance_state(args, case, n_steps):
  """Advance the initial state n_steps without donation; return the new state."""
  import jax
  gen = make_stepper(case, donate=False)
  st = case["state"]
  for _ in range(n_steps):
    _, st = next(gen)
  jax.block_until_ready(st)
  return st


def run_components(args, case, out):
  import jax
  from pyses.dynamical_cores.run_dycore import advance_coupling_step
  from pyses.dynamical_cores.time_stepping import (advance_dynamics_ullrich_5stage,
                                                   advance_hypervis_euler,
                                                   advance_sponge_euler,
                                                   dynamics_tendency)
  from pyses.dynamical_cores.model_state import (remap_dynamics, remap_tracers,
                                                 renormalize_dry_air_species,
                                                 check_dynamics_nan,
                                                 check_tracers_nan,
                                                 se_T_to_theta_d_d_mass)
  from pyses.dynamical_cores.tracer_advection.eulerian_spectral import advance_tracers
  from pyses.dynamical_cores.model_info import cam_se_models, cam_se_stable_models

  state = _advance_state(args, case, args.warmup)
  h_grid, v_grid, dims, model = (case["h_grid"], case["v_grid"],
                                 case["dims"], case["model"])
  physics_config, diffusion_config, timestep_config = (
      case["physics_config"], case["diffusion_config"], case["timestep_config"])
  dyn = state["dynamics"]
  static = state["static_forcing"]
  trc = state["tracers"]
  nlev = int(v_grid["hybrid_b_m"].shape[0])

  if model in cam_se_models:
    ms, ds = trc["moisture_species"], trc["dry_air_species"]
  else:
    ms, ds = None, None
  if model in cam_se_models and model not in cam_se_stable_models:
    dyn_step, step_model = se_T_to_theta_d_d_mass(dyn, v_grid, physics_config, model)
  else:
    dyn_step, step_model = dyn, model

  # Structs needed as tracer-advection inputs.
  dyn_next, consist_dyn = advance_dynamics_ullrich_5stage(
      dyn_step, static, h_grid, v_grid, physics_config, timestep_config, dims,
      step_model, moisture_species=ms, dry_air_species=ds)
  _, consist_visc = advance_hypervis_euler(
      dyn_next, static, h_grid, v_grid, physics_config, diffusion_config,
      timestep_config, dims, model)
  consist_init = {"d_mass_init": 1.0 * dyn["d_mass"],
                  "d_mass_end": 1.0 * dyn_next["d_mass"]}
  jax.block_until_ready((dyn_next, consist_dyn, consist_visc))

  def timeit(fn):
    jax.block_until_ready(fn())  # compile
    ts = []
    for _ in range(args.repeats):
      t0 = time.perf_counter()
      jax.block_until_ready(fn())
      ts.append(time.perf_counter() - t0)
    return statistics.median(ts)

  tsub = int(timestep_config["tracer_subcycle"])
  dsub = int(timestep_config["dynamics_subcycle"])
  do_remap = v_grid["hybrid_a_m"].shape[0] > 1

  components = [
      ("full_coupling_step",
       lambda: advance_coupling_step(state, h_grid, v_grid, physics_config,
                                     diffusion_config, timestep_config, dims,
                                     model, physics_forcing=None),
       1),
      ("rk5_dynamics_step",
       lambda: advance_dynamics_ullrich_5stage(dyn_step, static, h_grid, v_grid,
                                               physics_config, timestep_config,
                                               dims, step_model,
                                               moisture_species=ms,
                                               dry_air_species=ds),
       tsub * dsub),
      ("single_dynamics_tendency",
       lambda: dynamics_tendency(dyn_step, static, h_grid, v_grid,
                                 physics_config, dims, step_model,
                                 moisture_species=ms, dry_air_species=ds),
       0),
      ("hypervis_all_subcycles",
       lambda: advance_hypervis_euler(dyn_next, static, h_grid, v_grid,
                                      physics_config, diffusion_config,
                                      timestep_config, dims, model),
       tsub * dsub),
      ("sponge_all_subcycles",
       lambda: advance_sponge_euler(dyn_next, h_grid, physics_config,
                                    diffusion_config, timestep_config, dims,
                                    model),
       tsub * dsub),
      ("tracer_advection",
       lambda: advance_tracers(trc, consist_dyn, consist_init, h_grid, dims,
                               physics_config, diffusion_config,
                               timestep_config, model,
                               tracer_consist_hypervis=consist_visc),
       tsub),
      ("remap_dynamics",
       lambda: remap_dynamics(dyn, static, v_grid, physics_config, nlev, model),
       tsub if do_remap else 0),
      ("remap_tracers",
       lambda: remap_tracers(dyn_next, trc, v_grid, nlev, model),
       tsub if do_remap else 0),
      ("renormalize_dry_air_species",
       lambda: renormalize_dry_air_species(trc, model),
       tsub),
      ("nan_check_dynamics_host_sync",
       lambda: check_dynamics_nan(dyn, h_grid, model),
       0),
      ("nan_check_tracers_host_sync",
       lambda: check_tracers_nan(trc, h_grid, model),
       0),
  ]
  if model in cam_se_models and model not in cam_se_stable_models:
    components.append(
        ("se_T_to_theta_conversion",
         lambda: se_T_to_theta_d_d_mass(dyn, v_grid, physics_config, model),
         2 * tsub * dsub))

  results = {}
  print("=== components (median of %d, ms) ===" % args.repeats)
  full_ms = None
  for name, fn, calls in components:
    ms_one = timeit(fn) * 1e3
    results[name] = {"ms_per_call": ms_one, "calls_per_step": calls,
                     "ms_per_step": ms_one * calls}
    if name == "full_coupling_step":
      full_ms = ms_one
  recon = sum(r["ms_per_step"] for n, r in results.items()
              if n != "full_coupling_step")
  for name, r in results.items():
    frac = (100.0 * r["ms_per_step"] / full_ms) if full_ms else float("nan")
    print(f"  {name:34s} {r['ms_per_call']:9.2f} ms/call x {r['calls_per_step']:3d} "
          f"= {r['ms_per_step']:9.2f} ms/step  ({frac:5.1f}% of full step)")
  print(f"  reconstructed step from components: {recon:.2f} ms "
        f"vs measured full step {full_ms:.2f} ms")
  results["_reconstructed_ms_per_step"] = recon
  out["components"] = results


def run_cost(args, case, out, outdir):
  import jax
  from pyses.dynamical_cores.run_dycore import (_advance_coupling_step,
                                                _COUPLING_STATIC_ARGNAMES)
  jfn = jax.jit(_advance_coupling_step, static_argnames=_COUPLING_STATIC_ARGNAMES)
  lowered = jfn.lower(case["state"], case["h_grid"], case["v_grid"],
                      case["physics_config"], case["diffusion_config"],
                      case["timestep_config"], case["dims"], case["model"],
                      physics_forcing=None)
  compiled = lowered.compile()
  cost = compiled.cost_analysis()
  if isinstance(cost, (list, tuple)):
    cost = cost[0]
  mem = compiled.memory_analysis()
  res = {"flops": cost.get("flops"),
         "bytes_accessed": cost.get("bytes accessed"),
         "transcendentals": cost.get("transcendentals"),
         "optimal_seconds": cost.get("optimal_seconds")}
  for attr in ("temp_size_in_bytes", "argument_size_in_bytes",
               "output_size_in_bytes", "alias_size_in_bytes",
               "generated_code_size_in_bytes"):
    res[attr] = getattr(mem, attr, None)
  hlo_path = os.path.join(outdir, "coupling_step_hlo.txt")
  with open(hlo_path, "w") as f:
    f.write(compiled.as_text())
  res["hlo_path"] = hlo_path
  print("=== cost analysis (one coupling step) ===")
  for k, v in res.items():
    print(f"  {k}: {v}")
  out["cost"] = res


def run_trace(args, case, out, outdir):
  import jax
  gen = make_stepper(case, donate=not args.no_donate)
  for _ in range(args.warmup):
    _, st = next(gen)
  jax.block_until_ready(st)
  trace_dir = os.path.join(outdir, "jax_trace")
  with jax.profiler.trace(trace_dir, create_perfetto_trace=True):
    for _ in range(args.profile_steps):
      _, st = next(gen)
      jax.block_until_ready(st)
  print(f"jax profiler trace written to {trace_dir}")
  out["trace_dir"] = trace_dir


def _load_cudart():
  import ctypes
  import glob
  candidates = []
  for pat in (os.path.join(sys.prefix, "lib/python*/site-packages/nvidia/cuda_runtime/lib/libcudart.so*"),
              os.path.join(sys.prefix, "lib/python*/site-packages/nvidia/*/lib/libcudart.so*")):
    candidates += sorted(glob.glob(pat))
  candidates += ["libcudart.so.13", "libcudart.so.12", "libcudart.so"]
  for c in candidates:
    try:
      return ctypes.CDLL(c)
    except OSError:
      continue
  return None


def run_nsys(args, case, out):
  import jax
  gen = make_stepper(case, donate=not args.no_donate)
  for _ in range(args.warmup):
    _, st = next(gen)
  jax.block_until_ready(st)
  cudart = _load_cudart()
  if cudart is None:
    print("WARNING: libcudart not found; profiling whole run (no capture range)")
  else:
    cudart.cudaProfilerStart()
  for _ in range(args.profile_steps):
    _, st = next(gen)
    jax.block_until_ready(st)
  if cudart is not None:
    cudart.cudaProfilerStop()
  print(f"nsys capture section done ({args.profile_steps} steps)")


def main():
  args = parse_args()
  setup_env(args)
  modes = [m.strip() for m in args.mode.split(",") if m.strip()]
  outdir = args.outdir or os.path.join(REPO, "profiling", "results")
  os.makedirs(outdir, exist_ok=True)

  case = build_case(args)
  out = {"argv": sys.argv[1:]}
  log_run_info(args, case, out)

  for mode in modes:
    if mode == "timing":
      run_timing(args, case, out)
    elif mode == "components":
      run_components(args, case, out)
    elif mode == "cost":
      run_cost(args, case, out, outdir)
    elif mode == "trace":
      run_trace(args, case, out, outdir)
    elif mode == "nsys":
      run_nsys(args, case, out)
    else:
      raise ValueError(f"unknown mode {mode}")

  tag = f"{args.model}_nx{args.nx}_{'fp32' if args.float32 else 'fp64'}"
  json_path = os.path.join(outdir, f"profile_{tag}.json")
  with open(json_path, "w") as f:
    json.dump(out, f, indent=2, default=str)
  print(f"results written to {json_path}")


if __name__ == "__main__":
  main()
