"""
Alternative configuration module using env vars + lazy singleton + Backend protocol.

Environment variables
---------------------
PYSES_BACKEND : str, default "numpy"
    Compute backend to use. One of "numpy" or "jax".
PYSES_USE_MPI : str, default "0"
    Set to "1" to enable MPI communication.
PYSES_USE_CPU : str, default "1"
    Set to "0" to allow GPU use (JAX backend only).
PYSES_USE_DOUBLE : str, default "1"
    Set to "0" to use single precision.
PYSES_DEBUG : str, default "0"
    Set to "1" to enable debug mode.
PYSES_SHARD_CPU_COUNT : str, default "1"
    Number of virtual CPU devices for JAX CPU sharding.

Usage
-----
    from config_alt import get_backend

    be = get_backend()
    x = be.array([1, 2, 3])
    y = be.flip(x, axis=0)

The backend is initialised once on first call to ``get_backend()`` and cached
for the lifetime of the process.  To override in tests, set env vars before
the first call or use ``_reset_backend()`` after patching the environment.
"""

import os
import functools
import numpy as np
from typing import Callable, Protocol, runtime_checkable, Literal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() not in ("0", "false", "no", "")


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None:
        return default
    return int(val)


# ---------------------------------------------------------------------------
# MPI (optional)
# ---------------------------------------------------------------------------

try:
    from mpi4py import MPI as _MPI
    _mpi_comm = _MPI.COMM_WORLD
    _mpi_rank: int = _mpi_comm.Get_rank()
    _mpi_size: int = _mpi_comm.Get_size()
    _has_mpi_lib = True
except ImportError:
    _mpi_comm = None
    _mpi_rank = 0
    _mpi_size = 1
    _has_mpi_lib = False


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Backend(Protocol):
    """
    Minimal interface that every compute backend must satisfy.

    Attributes
    ----------
    np :
        The array namespace (numpy, jax.numpy, …).
    eps : float
        Floating-point tolerance appropriate for the chosen precision.
    debug : bool
    use_double : bool
    do_mpi_communication : bool
    mpi_rank : int
    mpi_size : int
    is_main_proc : bool
    do_sharding : bool
    num_devices : int
    """

    np: object
    newaxis: Literal[None]
    eps: float
    debug: bool
    use_double: bool
    do_mpi_communication: bool
    mpi_rank: int
    mpi_size: int
    is_main_proc: bool
    do_sharding: bool
    num_devices: int

    def array(self, x, dtype=None, elem_sharding_axis: int | None = None):
        ...

    def unwrap(self, x) -> np.ndarray:
        ...

    def get_global_array(self, x, dims, elem_sharding_axis: int = 0) -> np.ndarray:
        ...

    def jit(self, func: Callable) -> Callable:
        ...

    def shard_map(self, func: Callable, *args, **kwargs) -> Callable:
        ...

    def index_get(self, arr, idx: tuple):
        ...

    def index_add(self, arr, idx: tuple, vals) -> object:
        ...

    def index_max(self, arr, idx: tuple, vals) -> object:
        ...

    def parallel_region(self, fn: Callable, in_specs=None, out_specs=None) -> Callable:
        ...

    def halo_exchange(self, send_buffers: dict, neighbor_ranks: tuple) -> dict:
        ...

    def all_reduce_sum(self, t):
        ...

    def all_reduce_max(self, t):
        ...

    def all_reduce_min(self, t):
        ...


# ---------------------------------------------------------------------------
# Numpy backend
# ---------------------------------------------------------------------------

class NumpyBackend:
    """Serial NumPy backend — no optional dependencies required."""

    np = np
    newaxis = None
    use_wrapper = False
    wrapper_type = "numpy"
    # JAX sharding attributes — not applicable; exposed as None for uniform access
    usual_scalar_sharding = None
    extraction_sharding = None
    projection_sharding = None

    def __init__(self,
                 use_double: bool,
                 debug: bool,
                 do_mpi_communication: bool,
                 mpi_rank: int,
                 mpi_size: int,
                 has_mpi: bool,
                 mpi_comm):
        self.use_double = use_double
        self.debug = debug
        self.do_mpi_communication = do_mpi_communication
        self.mpi_rank = mpi_rank
        self.mpi_size = mpi_size
        self.is_main_proc = mpi_rank == 0
        self.do_sharding = False
        self.num_devices = 1
        self.num_jax_devices = 1
        self.eps = 1e-11 if use_double else 1e-6
        self._default_dtype = np.float64 if use_double else np.float32
        self.has_mpi = has_mpi
        self.mpi_comm = mpi_comm
        if has_mpi:
            from mpi4py import MPI
            self._MPI = MPI
        else:
            self._MPI = None

    def array(self, x, dtype=None, elem_sharding_axis=None):
        return np.array(x, dtype=dtype if dtype is not None else self._default_dtype)

    def unwrap(self, x):
        return x

    def get_global_array(self, x, dims, elem_sharding_axis=0):
        return x

    def jit(self, func, *_, **__):
        return func

    def shard_map(self, func, *_, **__):
        return func

    def index_get(self, arr, idx):
        return arr[idx]

    def index_add(self, arr, idx, vals):
        result = arr.copy()
        np.add.at(result, idx, vals)
        return result

    def index_max(self, arr, idx, vals):
        result = arr.copy()
        np.maximum.at(result, idx, vals)
        return result

    def parallel_region(self, fn, in_specs=None, out_specs=None):
        return fn

    def halo_exchange(self, send_buffers, neighbor_ranks):
        recv_buffers = {k: np.ascontiguousarray(np.asarray(send_buffers[k]))
                        for k in neighbor_ranks}
        if not neighbor_ranks or self.mpi_comm is None:
            return recv_buffers
        reqs = []
        for k in neighbor_ranks:
            reqs.append(self.mpi_comm.Isendrecv_replace(recv_buffers[k],
                                                        dest=k, sendtag=0,
                                                        source=k, recvtag=0))
        self._MPI.Request.Waitall(reqs)
        return recv_buffers

    def all_reduce_sum(self, t):
        if not self.has_mpi:
            return np.sum(t)
        send = np.asarray(t)
        recv = np.copy(send)
        self.mpi_comm.Allreduce(send, recv, op=self._MPI.SUM)
        return recv.item()

    def all_reduce_max(self, t):
        if not self.has_mpi:
            return np.max(t)
        send = np.asarray(t)
        recv = np.copy(send)
        self.mpi_comm.Allreduce(send, recv, op=self._MPI.MAX)
        return recv.item()

    def all_reduce_min(self, t):
        if not self.has_mpi:
            return np.min(t)
        send = np.asarray(t)
        recv = np.copy(send)
        self.mpi_comm.Allreduce(send, recv, op=self._MPI.MIN)
        return recv.item()



# ---------------------------------------------------------------------------
# JAX backend
# ---------------------------------------------------------------------------

class JaxBackend:
    """JAX backend with optional CPU sharding and GPU support."""

    newaxis = None
    use_wrapper = True
    wrapper_type = "jax"

    def __init__(self,
                 use_double: bool,
                 debug: bool,
                 use_cpu: bool,
                 shard_cpu_count: int,
                 do_mpi_communication: bool,
                 mpi_rank: int,
                 mpi_size: int,
                 has_mpi: bool,
                 mpi_comm):
        import jax
        import jax.numpy as jnp
        from jax.sharding import PartitionSpec, NamedSharding, AxisType

        assert not do_mpi_communication, (
            "The JAX backend does not support MPI; it uses device sharding for "
            "parallelism. This should have been rejected in _build_backend.")

        self.use_double = use_double
        self.debug = debug
        self.do_mpi_communication = do_mpi_communication
        self.mpi_rank = mpi_rank
        self.mpi_size = mpi_size
        self.is_main_proc = mpi_rank == 0
        self.eps = 1e-11 if use_double else 1e-6
        self._default_dtype = jnp.float64 if use_double else jnp.float32
        self._jax = jax
        self.np = jnp
        self.has_mpi = has_mpi
        self.mpi_comm = mpi_comm

        # --- precision ---
        if use_double:
            jax.config.update("jax_enable_x64", True)

        # --- sharding / device setup ---
        # TODO: exhaustively analyse nonsensical configuration combinations.
        if not do_mpi_communication:
            self.do_sharding = True
            if use_cpu:
                self.num_devices = shard_cpu_count
                os.environ["XLA_FLAGS"] = (
                    f"--xla_force_host_platform_device_count={shard_cpu_count}"
                )
                devices = jax.devices(backend="cpu")
            else:
                maybe_devices = jax.devices(backend="gpu")
                devices = maybe_devices if len(maybe_devices) > 0 else jax.devices(backend="cpu")
                self.num_devices = len(devices)
        else:
            self.do_sharding = False
            self.num_devices = 1
            #if use_cpu:
            #    jax.config.update("jax_default_device", jax.local_devices("cpu")[0])
            devices = jax.local_devices()

        if debug:
            print(f"Using devices {devices}, num_devices: {self.num_devices}, "
                  f"do_sharding: {self.do_sharding}")

        # --- mesh / shardings ---
        elem_axis_name = "f"
        device_mesh = jax.make_mesh(
            (self.num_devices,), (elem_axis_name,),
            axis_types=(AxisType.Explicit,)
        )
        jax.set_mesh(device_mesh)
        self._device_mesh = device_mesh
        self._elem_axis_name = elem_axis_name
        self.usual_scalar_sharding = NamedSharding(
            device_mesh, PartitionSpec(elem_axis_name, None, None)
        )
        self.extraction_sharding = NamedSharding(
            device_mesh, PartitionSpec(elem_axis_name, None)
        )
        self.projection_sharding = NamedSharding(
            device_mesh, PartitionSpec(elem_axis_name, None, None, None)
        )

        self._NamedSharding = NamedSharding
        self._PartitionSpec = PartitionSpec
        self.num_jax_devices = self.num_devices

    # --- sharding helpers ---

    def _good_sharding(self, array, elem_sharding_axis):
        spec_names = [None] * len(array.shape)
        spec_names[elem_sharding_axis] = self._elem_axis_name
        return self._NamedSharding(self._device_mesh, self._PartitionSpec(*spec_names))

    # --- Backend interface ---

    def array(self, x, dtype=None, elem_sharding_axis=None):
        jnp = self.np
        x = jnp.array(x, dtype=dtype if dtype is not None else self._default_dtype)
        if elem_sharding_axis is not None:
            x = self._jax.device_put(x, self._good_sharding(x, elem_sharding_axis))
        return x

    def unwrap(self, x):
        return np.asarray(x)

    def get_global_array(self, x, dims, elem_sharding_axis=0):
        arr = np.asarray(self._jax.device_get(x))
        if dims is not None:
            slices = [slice(None)] * x.ndim
            slices[elem_sharding_axis] = slice(0, dims["num_elem"])
            return arr[*slices]
        return arr

    def jit(self, func, *args, **kwargs):
        return self._jax.jit(func, *args, **kwargs)

    def shard_map(self, func, *args, **kwargs):
        return self._jax.shard_map(func, *args, **kwargs)

    def index_get(self, arr, idx):
        return arr[idx]

    def index_add(self, arr, idx, vals):
        return arr.at[idx].add(vals)

    def index_max(self, arr, idx, vals):
        return arr.at[idx].max(vals)

    def parallel_region(self, fn, in_specs=None, out_specs=None):
        P = self._PartitionSpec
        if in_specs is None:
            in_specs = P(self._elem_axis_name)
        if out_specs is None:
            out_specs = P(self._elem_axis_name)
        return self._jax.shard_map(
            fn, mesh=self._device_mesh, in_specs=in_specs, out_specs=out_specs,
        )

    def all_to_all_ppermute(self, comm_matrix):
        """
        Transpose a device-indexed communication matrix via shard_map + ppermute.

        Parameters
        ----------
        comm_matrix : Array[tuple[num_devices, num_devices, *trailing], Float]
            ``comm_matrix[src, dst]`` is the payload device ``src`` sends to
            device ``dst``.  Sharded along axis 0 (the source axis).

        Returns
        -------
        Array[tuple[num_devices, num_devices, *trailing], Float]
            ``out[dst, src] == comm_matrix[src, dst]`` — each device's row holds
            what it received from every other device.  Sharded along axis 0.

        Notes
        -----
        Implemented as ``num_devices - 1`` ``lax.ppermute`` shift rounds inside
        a ``shard_map``.  Round ``d`` uses the static permutation
        ``[(s, (s + d) % N) for s in range(N)]``; the per-round payload is the
        column the local device owes its round-``d`` partner.  ``ppermute`` is
        its own transpose under autograd, so this is forward- and
        reverse-differentiable.  This is the native-collective building block on
        which a multi-host DSS halo exchange is built.
        """
        from jax import lax
        jnp = self.np
        N = self.num_devices
        axis = self._elem_axis_name

        def _core(row):
            r = lax.axis_index(axis)
            local = row[0]
            result = jnp.zeros_like(local)
            result = result.at[r].set(local[r])
            for d in range(1, N):
                payload = lax.dynamic_index_in_dim(local, (r + d) % N,
                                                   axis=0, keepdims=False)
                perm = [(s, (s + d) % N) for s in range(N)]
                recvd = lax.ppermute(payload, axis, perm)
                result = result.at[(r - d) % N].set(recvd)
            return result[jnp.newaxis]

        return self.parallel_region(_core)(comm_matrix)

    def halo_exchange(self, send_buffers, neighbor_ranks):
        """
        Sparse point-to-point exchange (multi-host JAX path).

        Single-host JAX achieves DSS through device sharding (see
        ``operations_2d.local_assembly.project_scalar``), so this dict-of-buffers
        API — a multi-process concept — is not exercised in single-host runs.
        A multi-host implementation packs ``send_buffers`` into a device-indexed
        communication matrix and calls :meth:`all_to_all_ppermute`; wiring that
        requires a globally consistent device/rank map threaded from grid
        construction, which is not yet in place.
        """
        if not neighbor_ranks:
            return dict(send_buffers)
        raise NotImplementedError(
            "Multi-host JAX halo_exchange is not wired. Single-host JAX uses "
            "device sharding for DSS (operations_2d.local_assembly.project_scalar); "
            "the ppermute all-to-all primitive is available as all_to_all_ppermute. "
            "See MIGRATION_PLAN.md §2.5.")

    def all_reduce_sum(self, t):
        return self.np.sum(t)

    def all_reduce_max(self, t):
        return self.np.max(t)

    def all_reduce_min(self, t):
        return self.np.min(t)



# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _build_backend() -> Backend:
    backend_name = os.environ.get("PYSES_BACKEND", "numpy").strip().lower()
    use_double = _env_bool("PYSES_USE_DOUBLE", default=True)
    debug = _env_bool("PYSES_DEBUG", default=False)
    use_mpi = _env_bool("PYSES_USE_MPI", default=False)
    use_cpu = _env_bool("PYSES_USE_CPU", default=True)
    shard_count = _env_int("PYSES_SHARD_CPU_COUNT", default=1)

    # MPI state
    if use_mpi and _has_mpi_lib:
        do_mpi = _mpi_size > 1
        rank, size = _mpi_rank, _mpi_size
        mpi_comm = _mpi_comm
    else:
        do_mpi, rank, size = False, 0, 1
        mpi_comm = None
    has_mpi = _has_mpi_lib and use_mpi

    if backend_name == "jax":
        if use_mpi:
            raise ValueError(
                "The JAX backend does not support MPI. JAX parallelism is "
                "provided by device sharding (set PYSES_SHARD_CPU_COUNT or use "
                "GPUs); unset PYSES_USE_MPI (or set PYSES_USE_MPI=0)."
            )
        return JaxBackend(
            use_double=use_double,
            debug=debug,
            use_cpu=use_cpu,
            shard_cpu_count=shard_count,
            do_mpi_communication=False,
            mpi_rank=0,
            mpi_size=1,
            has_mpi=False,
            mpi_comm=None,
        )
    elif backend_name == "numpy":
        return NumpyBackend(
            use_double=use_double,
            debug=debug,
            do_mpi_communication=do_mpi,
            mpi_rank=rank,
            mpi_size=size,
            has_mpi=has_mpi,
            mpi_comm=mpi_comm,
        )
    else:
        raise ValueError(
            f"Unknown PYSES_BACKEND={backend_name!r}. Must be 'numpy' or 'jax'."
        )


# Cache the singleton.  Call _reset_backend() in tests when you need a fresh
# instance after patching env vars.
@functools.cache
def get_backend() -> Backend:
    return _build_backend()


def _reset_backend():
    """Clear the cached backend (useful in tests)."""
    get_backend.cache_clear()


def runtime_assert(condition, message: str = "Assertion failed") -> None:
    """
    Assert that condition is True at runtime, including inside JIT.

    Under JAX jit, plain Python ``assert`` evaluates at trace time against an
    abstract value and either silently passes or raises ConcretizationTypeError.
    This helper uses ``jax.debug.callback`` so the check executes at actual
    kernel-launch time and is excluded from the autodiff graph.

    Under NumPy the check is a plain assert.  Future PyTorch support should
    replace the else-branch with ``torch._check(condition)``.
    """
    be = get_backend()
    if be.wrapper_type == "jax":
        import jax

        def _check(ok):
            if not bool(ok):
                raise RuntimeError(message)

        jax.debug.callback(_check, condition)
    else:
        assert condition, message
