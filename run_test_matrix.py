#!/usr/bin/env python3
"""
Run ``tests/fast_tests`` across backend / parallelism configurations.

Configurations
--------------
  numpy       NumPy, single process
  numpy-mpi   NumPy + MPI         (mpirun -n NP, PYSES_USE_MPI=1)
  torch       PyTorch, single process
  torch-mpi   PyTorch + torch.distributed (mpirun -n NP, PYSES_USE_MPI=1)
  jax         JAX, single device
  jax-shard   JAX, multi-device sharding  (PYSES_SHARD_CPU_COUNT=NSHARD)

By default every configuration runs, in series.  Pass a subset of names to run
only those, e.g.::

    ./run_test_matrix.py numpy torch jax
    ./run_test_matrix.py --np 4 numpy-mpi torch-mpi
    ./run_test_matrix.py --exclude-slow --pytest "-x -k project"

Notes
-----
* JAX cannot use MPI (the backend rejects PYSES_USE_MPI=1); its multi-process
  analogue is device sharding (``jax-shard``).
* The MPI configurations launch with ``mpirun``; if it is unavailable they are
  skipped with a warning rather than failing the whole run.
* ``tests/fast_tests`` are single-process science tests.  The MPI configs run
  the same files under multiple ranks (each rank runs the suite); point
  ``--tests`` at ``tests/mpi_tests`` to exercise the distributed assembly paths
  instead.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TESTS = "tests/fast_tests"
SLOW_IGNORES = [
    "--ignore=tests/fast_tests/dynamical_cores_tests/test_run_model.py",
    "--ignore=tests/fast_tests/shallow_water_tests",
]

# name -> (description, extra env, launch kind)
CONFIGS = {
    "numpy":     ("NumPy (single process)",
                  {"PYSES_BACKEND": "numpy"}, "serial"),
    "numpy-mpi": ("NumPy + MPI (mpirun -n NP)",
                  {"PYSES_BACKEND": "numpy", "PYSES_USE_MPI": "1"}, "mpi"),
    # torch.compile is on by default for end users, but the suite sweeps many
    # shapes (a Dynamo recompile each), so disable it here unless --torch-compile.
    "torch":     ("PyTorch (single process)",
                  {"PYSES_BACKEND": "torch", "PYSES_TORCH_COMPILE": "0"}, "serial"),
    "torch-mpi": ("PyTorch + torch.distributed (mpirun -n NP)",
                  {"PYSES_BACKEND": "torch", "PYSES_USE_MPI": "1",
                   "PYSES_TORCH_COMPILE": "0"}, "mpi"),
    "jax":       ("JAX (single device)",
                  {"PYSES_BACKEND": "jax"}, "serial"),
    "jax-shard": ("JAX (multi-device sharding)",
                  {"PYSES_BACKEND": "jax"}, "shard"),
}
ORDER = ["numpy", "numpy-mpi", "torch", "torch-mpi", "jax", "jax-shard"]

_SUMMARY_RE = re.compile(
    r"(\d+ (?:passed|failed|error|skipped|xfailed|xpassed|deselected)"
    r"[^\n]*?(?:in [\d.]+s|in [\d.]+m)[^\n]*)")


def detect_mpi_flavor():
    """Return 'openmpi', 'mpich', or None (mpirun absent)."""
    if shutil.which("mpirun") is None:
        return None
    try:
        out = subprocess.run(["mpirun", "--version"], capture_output=True,
                             text=True).stdout + \
              subprocess.run(["mpirun", "--version"], capture_output=True,
                             text=True).stderr
    except Exception:
        return "mpich"
    return "openmpi" if "Open MPI" in out or "open-mpi" in out.lower() else "mpich"


def resolve_python():
    """Resolve the project interpreter once (so mpirun ranks skip `uv` locking)."""
    venv_py = os.path.join(REPO_ROOT, ".venv", "bin", "python")
    if os.path.exists(venv_py):
        return [venv_py]
    try:
        out = subprocess.check_output(
            ["uv", "run", "python", "-c", "import sys; print(sys.executable)"],
            cwd=REPO_ROOT, text=True).strip().splitlines()[-1]
        if out and os.path.exists(out):
            return [out]
    except Exception:
        pass
    return [sys.executable]


def build_command(kind, env, pyexe, tests, pytest_args, nproc, nshard, mpi_flavor):
    """Return (argv, full_env, skip_reason). argv is None when the config is skipped."""
    full_env = dict(os.environ)
    full_env.update(env)
    base = [*pyexe, "-m", "pytest", tests, *pytest_args]
    if kind == "serial":
        return base, full_env, None
    if kind == "shard":
        full_env["PYSES_SHARD_CPU_COUNT"] = str(nshard)
        return base, full_env, None
    if kind == "mpi":
        if mpi_flavor is None:
            return None, full_env, "mpirun not found"
        forward = []
        if mpi_flavor == "openmpi":
            # OpenMPI does not propagate custom env vars without -x; MPICH/Hydra
            # forwards the launching environment by default, so it needs none.
            for var in list(env) + ["PYSES_USE_CPU", "PYSES_USE_DOUBLE"]:
                forward += ["-x", var]
        argv = ["mpirun", "-n", str(nproc), *forward, *base]
        return argv, full_env, None
    raise ValueError(kind)


def run_config(name, pyexe, tests, pytest_args, nproc, nshard, mpi_flavor,
               env_override=None):
    desc, env, kind = CONFIGS[name]
    env = {**env, **(env_override or {})}
    argv, full_env, skip_reason = build_command(
        kind, env, pyexe, tests, pytest_args, nproc, nshard, mpi_flavor)
    print("\n" + "=" * 78)
    print(f"[{name}] {desc}")
    if argv is None:
        print(f"  SKIPPED: {skip_reason}")
        print("=" * 78)
        return {"name": name, "status": "skipped", "summary": skip_reason,
                "seconds": 0.0, "returncode": None}
    print(f"  env: {' '.join(f'{k}={v}' for k, v in env.items())}"
          + (f" PYSES_SHARD_CPU_COUNT={nshard}" if kind == "shard" else ""))
    print(f"  cmd: {' '.join(argv)}")
    print("-" * 78)

    start = time.time()
    proc = subprocess.Popen(argv, cwd=REPO_ROOT, env=full_env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    summary = ""
    for line in proc.stdout or []:
        sys.stdout.write(line)
        sys.stdout.flush()
        m = _SUMMARY_RE.search(line)
        if m:
            summary = m.group(1).strip()
    proc.wait()
    seconds = time.time() - start
    status = "PASS" if proc.returncode == 0 else "FAIL"
    if not summary:
        summary = f"(no pytest summary; exit {proc.returncode})"
    return {"name": name, "status": status, "summary": summary,
            "seconds": seconds, "returncode": proc.returncode}


def main():
    p = argparse.ArgumentParser(
        description="Run tests/fast_tests across backend/parallelism configs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("configs", nargs="*", choices=ORDER + [],
                   help="Configurations to run (default: all, in series).")
    p.add_argument("--list", action="store_true", help="List configs and exit.")
    p.add_argument("--np", type=int, default=2,
                   help="MPI rank count for *-mpi configs (default: 2).")
    p.add_argument("--nshard", type=int, default=2,
                   help="JAX CPU device count for jax-shard (default: 2).")
    p.add_argument("--tests", default=DEFAULT_TESTS,
                   help=f"Test path passed to pytest (default: {DEFAULT_TESTS}).")
    p.add_argument("--exclude-slow", action="store_true",
                   help="Skip the slow test_run_model.py and shallow_water_tests.")
    p.add_argument("--torch-compile", action="store_true",
                   help="Enable torch.compile for the torch configs "
                        "(off by default here; expect long Dynamo recompiles).")
    p.add_argument("--pytest", default="-q -p no:cacheprovider",
                   help='Extra pytest args as one string '
                        '(default: "-q -p no:cacheprovider").')
    args = p.parse_args()

    if args.list:
        for name in ORDER:
            print(f"  {name:11s} {CONFIGS[name][0]}")
        return 0

    selected = args.configs or ORDER
    pytest_args = args.pytest.split()
    if args.exclude_slow:
        pytest_args = SLOW_IGNORES + pytest_args

    pyexe = resolve_python()
    mpi_flavor = detect_mpi_flavor()
    print(f"Interpreter: {' '.join(pyexe)}")
    print(f"MPI: {mpi_flavor or 'not found'}")
    print(f"Tests: {args.tests}   Configs: {', '.join(selected)}")

    override = {"PYSES_TORCH_COMPILE": "1"} if args.torch_compile else None
    results = []
    for name in selected:
        results.append(run_config(name, pyexe, args.tests, pytest_args,
                                  args.np, args.nshard, mpi_flavor, override))

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'config':12s} {'status':7s} {'time':>9s}  summary")
    print("-" * 78)
    for r in results:
        print(f"{r['name']:12s} {r['status']:7s} {r['seconds']:8.1f}s  {r['summary']}")

    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
