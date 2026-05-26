#!/usr/bin/env python3
"""
Strong-scaling driver for the random-atmosphere HOMME workload.

Launches ``tests/slow_tests/scaling_workload.py`` once per (backend, core-count)
point, fixing the problem size (ne15 x npt4 x 15 levels) and increasing the core
count, then reports time / speedup / parallel efficiency per backend.

Backends and how they parallelise:
  * numpy  -> MPI domain decomposition  (mpirun -n N)
  * torch  -> MPI + torch.distributed   (mpirun -n N)

JAX is intentionally *not* included.  On CPU, jaxlib sizes its TSL intra-op
thread pool to the OS core count and exposes no public knob to cap it
(``--xla_cpu_multi_thread_eigen=false`` only single-threads Eigen ops, not the
runtime pool), so an in-process "n-device" sharded run uses all cores at every
n -- the n=1 baseline is already multi-threaded and the speedup column is
meaningless.  ``scaling_workload.py`` itself still
runs under JAX for spot-check single-machine timings.

Single-core-per-worker for the MPI backends is enforced by single-threading
every process's math (OMP/OpenBLAS/MKL/veclib/numexpr = 1,
torch.set_num_threads(1)).  True hardware core-pinning needs OS affinity, which
macOS lacks; on Linux pass ``--bind`` to add ``mpirun -bind-to core``.

Examples
--------
    ./run_scaling.py                          # all backends, cores 1,2,4,8
    ./run_scaling.py --backends numpy --cores 1,2,4,6,12
    ./run_scaling.py --steps 8 --warmup 2 --mountain
    ./run_scaling.py --bind                   # Linux: pin MPI ranks to cores
"""
import argparse
import os
import re
import shutil
import signal
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
WORKLOAD = "tests/slow_tests/scaling_workload.py"

# one math thread per process so the only parallelism is across ranks/devices
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

# name -> (env, launch kind).  See module docstring for why jax is omitted.
CONFIGS = {
    "numpy": ({"PYSES_BACKEND": "numpy", "PYSES_USE_MPI": "1"}, "mpi"),
    "torch": ({"PYSES_BACKEND": "torch", "PYSES_USE_MPI": "1",
               "PYSES_TORCH_COMPILE": "0"}, "mpi"),
}
ORDER = ["numpy", "torch"]

_TIME_RE = re.compile(r"\btime_s=([0-9.]+)")


def resolve_python():
    venv_py = os.path.join(REPO_ROOT, ".venv", "bin", "python")
    if os.path.exists(venv_py):
        return venv_py
    try:
        out = subprocess.check_output(
            ["uv", "run", "python", "-c", "import sys; print(sys.executable)"],
            cwd=REPO_ROOT, text=True).strip().splitlines()[-1]
        if out and os.path.exists(out):
            return out
    except Exception:
        pass
    return sys.executable


def mpi_flavor():
    if shutil.which("mpirun") is None:
        return None
    out = subprocess.run(["mpirun", "--version"], capture_output=True, text=True)
    text = (out.stdout + out.stderr).lower()
    return "openmpi" if "open mpi" in text else "mpich"


def build_argv(kind, env, pyexe, n, args, flavor):
    """Return (argv, full_env) or (None, reason)."""
    full_env = dict(os.environ)
    full_env.update(THREAD_ENV)
    full_env.update(env)
    # the source tree (pyses/, tests/) must be importable by the bare interpreter
    existing_pp = full_env.get("PYTHONPATH", "")
    full_env["PYTHONPATH"] = REPO_ROOT + (os.pathsep + existing_pp if existing_pp else "")
    workload = [pyexe, WORKLOAD, "--nx", str(args.nx), "--steps", str(args.steps),
                "--warmup", str(args.warmup)]
    if args.mountain:
        workload.append("--mountain")
    if kind == "mpi":
        if flavor is None:
            return None, "mpirun not found"
        bind = ["-bind-to", "core"] if args.bind else []
        return ["mpirun", "-n", str(n), *bind, *workload], full_env
    raise ValueError(kind)


def run_point(name, pyexe, n, args, flavor):
    env_cfg, kind = CONFIGS[name]
    argv, full_env = build_argv(kind, env_cfg, pyexe, n, args, flavor)
    if argv is None:
        print(f"  [{name} n={n}] SKIPPED: {full_env}", flush=True)
        return None
    print(f"  [{name} n={n}] launching ({args.timeout}s budget) ...", flush=True)
    # start_new_session so a hung MPI run (mpirun + ranks) can be killed as a group
    proc = subprocess.Popen(argv, cwd=REPO_ROOT, env=full_env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.communicate()
        print(f"  [{name} n={n}] TIMEOUT after {args.timeout}s — killed "
              f"(likely a collective deadlock)", flush=True)
        return None
    m = None
    for line in out.splitlines():
        if "WORKLOAD" in line:
            m = _TIME_RE.search(line)
    if m is None:
        tail = "\n".join(out.strip().splitlines()[-6:])
        print(f"  [{name} n={n}] FAILED (exit {proc.returncode}):\n{tail}", flush=True)
        return None
    t = float(m.group(1))
    print(f"  [{name} n={n}] time_s={t:.4f}", flush=True)
    return t


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backends", nargs="*", choices=ORDER, default=ORDER)
    p.add_argument("--nx", type=int, default=15,
                   help="cubed-sphere resolution ne (default: 15 -> 1350 elems)")
    p.add_argument("--cores", default="1,2,4,8",
                   help="comma-separated core counts (default: 1,2,4,8)")
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--mountain", action="store_true")
    p.add_argument("--bind", action="store_true",
                   help="add 'mpirun -bind-to core' (Linux core pinning)")
    p.add_argument("--timeout", type=float, default=900.0,
                   help="per-run wall-clock budget in seconds (default: 900)")
    args = p.parse_args()

    cores = [int(c) for c in args.cores.split(",")]
    pyexe = resolve_python()
    flavor = mpi_flavor()
    ncpu = os.cpu_count()
    print(f"Interpreter: {pyexe}\nMPI: {flavor or 'not found'}   CPUs: {ncpu}")
    print(f"Backends: {args.backends}   Cores: {cores}   "
          f"steps={args.steps} warmup={args.warmup} mountain={args.mountain}")

    results = {b: {} for b in args.backends}
    for name in args.backends:
        print(f"\n=== {name} ===")
        for n in cores:
            if n > (ncpu or n):
                print(f"  [{name} n={n}] skipped (only {ncpu} CPUs)")
                continue
            t = run_point(name, pyexe, n, args, flavor)
            if t is not None:
                results[name][n] = t

    print("\n" + "=" * 64)
    print("STRONG SCALING (fixed ne15 x npt4 x 15 levels)")
    print("=" * 64)
    for name in args.backends:
        r = results[name]
        print(f"\n{name}")
        print(f"  {'cores':>5} {'time_s':>10} {'speedup':>9} {'efficiency':>11}")
        base = r.get(cores[0])
        for n in cores:
            if n not in r:
                continue
            t = r[n]
            if base:
                sp = base / t
                print(f"  {n:>5} {t:>10.4f} {sp:>8.2f}x {sp / n * 100:>10.1f}%")
            else:
                print(f"  {n:>5} {t:>10.4f} {'n/a':>9} {'n/a':>11}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
