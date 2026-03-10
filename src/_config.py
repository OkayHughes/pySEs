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

    def partial(self, func: Callable, *args, **kwargs) -> Callable:
        ...

    def vmap_1d_apply(self, func: Callable, vector, in_axis: int, out_axis: int):
        ...

    def flip(self, array, axis: int):
        ...

    def remainder(self, array, divisor):
        ...

    def take_along_axis(self, array, idxs, axis: int):
        ...

    def cast_type(self, arr, dtype):
        ...

    def assert_true(self, condition: bool) -> None:
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

    def partial(self, func, *args, **kwargs):
        from functools import partial
        return partial(func, *args, **kwargs)

    def vmap_1d_apply(self, func, scalar, in_axis, out_axis):
        levs = []
        for lev_idx in range(scalar.shape[in_axis]):
            scalar_2d = scalar.take(indices=lev_idx, axis=in_axis)
            levs.append(func(scalar_2d))
        return np.stack(levs, axis=out_axis)

    def flip(self, array, axis):
        return np.flip(array, axis=axis)

    def remainder(self, array, divisor):
        return np.mod(array, divisor)

    def take_along_axis(self, array, idxs, axis):
        return np.take_along_axis(array, idxs, axis=axis)

    def cast_type(self, arr, dtype):
        return arr.astype(dtype)

    def assert_true(self, condition):
        assert condition


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
        from jax.tree_util import Partial as jax_partial

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
            if use_cpu:
                jax.config.update("jax_default_device", jax.local_devices("cpu")[0])
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
        self._jax_partial = jax_partial
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

    def partial(self, func, *args, **kwargs):
        return self._jax_partial(func, *args, **kwargs)

    def vmap_1d_apply(self, func, vector, in_axis, out_axis):
        return self._jax.vmap(func, in_axes=(in_axis), out_axes=(out_axis))(vector)

    def flip(self, array, axis):
        return self.np.flip(array, axis=axis)

    def remainder(self, array, divisor):
        return self.np.mod(array, divisor)

    def take_along_axis(self, array, idxs, axis):
        return self.np.take_along_axis(array, idxs, axis=axis)

    def cast_type(self, arr, dtype):
        return arr.astype(dtype)

    def assert_true(self, condition):
        # JAX traces through asserts, so we skip them in jit-compiled code
        return


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
        assert not (shard_count > 1 and do_mpi), \
            "Sharding in an MPI environment is not presently supported"
        return JaxBackend(
            use_double=use_double,
            debug=debug,
            use_cpu=use_cpu,
            shard_cpu_count=shard_count,
            do_mpi_communication=do_mpi,
            mpi_rank=rank,
            mpi_size=size,
            has_mpi=has_mpi,
            mpi_comm=mpi_comm,
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
