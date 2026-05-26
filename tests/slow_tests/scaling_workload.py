#!/usr/bin/env python3
"""
Single strong-scaling workload run: the random-atmosphere HOMME integration,
timed over a fixed number of coupling steps.

This is the *unit of work* the scaling driver (``run_scaling.py``) launches once
per (backend, core-count) point.  It is not a pytest test — run it directly:

    PYSES_BACKEND=numpy PYSES_USE_MPI=1 mpirun -n 4 python tests/slow_tests/scaling_workload.py
    PYSES_BACKEND=jax   PYSES_SHARD_CPU_COUNT=4 python tests/slow_tests/scaling_workload.py
    PYSES_BACKEND=torch PYSES_USE_MPI=1 mpirun -n 4 python tests/slow_tests/scaling_workload.py

Parallelism is the dynamical core's native decomposition for each backend:
  * numpy / torch + MPI -> domain decomposition via ``make_grid_mpi_ready`` (one
    MPI rank per subdomain; DSS halos via the backend collectives).
  * JAX -> element-axis device sharding (``PYSES_SHARD_CPU_COUNT`` virtual CPU
    devices); the grid auto-shards in ``init_quasi_uniform_grid_elem_local``.

The problem size is fixed (ne``NX`` x ``NPT`` x 15 levels), so this is *strong*
scaling: more cores divide the same total work.  Only the integration loop is
timed; grid build, initialisation, and the first ``--warmup`` steps (which
include JAX/Inductor compilation) are excluded.  The reported time is the
slowest rank's (an all-reduce max), printed on the main process as a
machine-parseable ``WORKLOAD ...`` line.

Run with single-threaded math (OMP_NUM_THREADS=1 etc., set by the driver) so the
only parallelism is across ranks/devices.
"""
import argparse
import time

import numpy as np

from pyses._config import get_backend
from pyses.mesh_generation.element_local_metric import init_quasi_uniform_grid_elem_local
from pyses.operations_2d.horizontal_grid import make_grid_mpi_ready
from pyses.dynamical_cores.mass_coordinate import init_vertical_grid
from pyses.dynamical_cores.model_config import init_default_config, hypervis_opts
from pyses.dynamical_cores.run_dycore import init_simulator
from tests.test_data.mass_coordinate_grids import cam30
from tests.slow_tests.test_random_atmosphere import (
    _build_state, NE, NPT, MODEL, SEED0, MAX_WIND, MAX_ATTEMPTS)

_be = get_backend()
_unwrap = _be.unwrap


def _sync(state):
    """Force completion of async work (JAX dispatches lazily; CPU torch is sync)."""
    if _be.wrapper_type == "jax":
        import jax
        jax.block_until_ready(jax.tree_util.tree_leaves(state))


def _barrier():
    if _be.mpi_comm is not None:
        _be.mpi_comm.Barrier()


def _reduce_max(x):
    if _be.mpi_comm is not None:
        from mpi4py import MPI
        return _be.mpi_comm.allreduce(x, op=MPI.MAX)
    return x


def _build_accepted_state(h_grid, v_grid, physics_config, dims, mountain):
    """Rejection-sample a state with global max wind <= MAX_WIND.

    Mirrors ``_init_fuzzed_state`` but is decomposition-safe: under MPI each rank
    only holds its subdomain, so the accept/reject test reduces the *global*
    maximum wind (``_reduce_max``) and every rank therefore selects the same
    accepted draw and builds a consistent piece of the same atmosphere.  All
    backends run the identical accepted draw (the seed schedule is fixed), so the
    scaling comparison is apples-to-apples.
    """
    for attempt in range(MAX_ATTEMPTS):
        rng = np.random.default_rng(SEED0 + attempt)
        state = _build_state(h_grid, v_grid, physics_config, dims, mountain, rng)
        wind = _unwrap(state["dynamics"]["horizontal_wind"])
        local = float(np.max(np.sqrt(wind[..., 0] ** 2 + wind[..., 1] ** 2)))
        if not np.isfinite(local):
            local = np.inf
        max_wind = _reduce_max(local)
        if max_wind <= MAX_WIND:
            if _be.is_main_proc:
                print(f"accepted draw on attempt {attempt} "
                      f"(global max wind {max_wind:.1f} m/s)", flush=True)
            return state
    raise RuntimeError(f"no draw with max wind < {MAX_WIND} m/s in "
                       f"{MAX_ATTEMPTS} attempts")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nx", type=int, default=NE)
    ap.add_argument("--npt", type=int, default=NPT)
    ap.add_argument("--steps", type=int, default=6, help="timed coupling steps")
    ap.add_argument("--warmup", type=int, default=2, help="untimed warmup steps")
    ap.add_argument("--mountain", action="store_true")
    args = ap.parse_args()

    # one math thread per process so parallelism is purely across ranks/devices
    if _be.wrapper_type == "torch":
        _be._torch.set_num_threads(1)
        try:
            _be._torch.set_num_interop_threads(1)
        except Exception:
            pass

    # --- grid: native decomposition per backend ---
    if _be.do_mpi_communication:                       # numpy / torch + MPI, >1 rank
        # build on the backend's native array type; make_grid_mpi_ready unwraps
        # the assembly triple internally for its (numpy) decomposition logic.
        g_glob, d_glob = init_quasi_uniform_grid_elem_local(
            args.nx, args.npt, calc_smooth_tensor=True)
        h_grid, dims = make_grid_mpi_ready(g_glob, d_glob, _be.mpi_rank)
        n_par = _be.mpi_size
    else:                                              # JAX (auto-sharded) or single rank
        h_grid, dims = init_quasi_uniform_grid_elem_local(
            args.nx, args.npt, calc_smooth_tensor=True)
        n_par = _be.num_devices

    v_grid = init_vertical_grid(cam30["hybrid_a_i"][::2], cam30["hybrid_b_i"][::2],
                                cam30["p0"], MODEL)
    physics_config, diffusion_config, timestep_config = init_default_config(
        args.nx, h_grid, v_grid, dims, MODEL,
        hypervis_type=hypervis_opts.variable_resolution)

    # rejection-sample a stable atmosphere (global max wind <= MAX_WIND); the
    # fixed seed schedule makes the accepted draw identical across backends and
    # decompositions, so this is the same well-posed problem everywhere.
    state = _build_accepted_state(h_grid, v_grid, physics_config, dims,
                                  args.mountain)

    simulator = init_simulator(h_grid, v_grid, physics_config, diffusion_config,
                               timestep_config, dims, MODEL)
    gen = simulator(state)

    st = state
    for _ in range(args.warmup):                       # compile / first-touch (untimed)
        _, st = next(gen)
    _sync(st)
    _barrier()

    t0 = time.perf_counter()
    for _ in range(args.steps):
        _, st = next(gen)
    _sync(st)
    elapsed = _reduce_max(time.perf_counter() - t0)

    if _be.is_main_proc:
        print(f"WORKLOAD backend={_be.wrapper_type} n_par={n_par} "
              f"nx={args.nx} npt={args.npt} steps={args.steps} "
              f"time_s={elapsed:.4f} per_step_s={elapsed / args.steps:.6f}")


if __name__ == "__main__":
    main()
