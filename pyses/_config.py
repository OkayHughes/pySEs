"""
Alternative configuration module using env vars + lazy singleton + Backend protocol.

Environment variables
---------------------
PYSES_BACKEND : str, default "numpy"
    Compute backend to use. One of "numpy", "jax", or "torch".
PYSES_USE_MPI : str, default "0"
    Set to "1" to enable MPI communication. Not supported with the JAX backend
    (which uses device sharding); supported by numpy and torch.
PYSES_USE_CPU : str, default "0"
    Set to "0" to allow GPU use (JAX and torch backends).
PYSES_USE_DOUBLE : str, default "1"
    Set to "0" to use single precision.
PYSES_DEBUG : str, default "0"
    Set to "1" to enable debug mode.
PYSES_SHARD_CPU_COUNT : str, default "1"
    Number of virtual CPU devices for JAX CPU sharding.
PYSES_JIT_COMPILE : str, default "1"
    JAX and torch backends. Set to "0" to disable JIT compilation: the JAX
    backend skips ``jax.jit`` and the torch backend runs eager instead of
    ``torch.compile``, so both execute eagerly. 

Usage
-----
    from _config import get_backend

    be = get_backend()
    x = be.array([1, 2, 3])
    y = be.np.flip(x, axis=0)

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

    def vmap(self, func: Callable, in_axes=0, out_axes=0) -> Callable:
        ...

    def scan(self, f: Callable, init, xs, reverse: bool = False, axis: int = 0):
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

    def vmap(self, func, in_axes=0, out_axes=0):
        # No batching primitive; map in Python and stack, matching jax.vmap.
        # Supports several positional args with a shared int ``in_axes`` or a
        # per-arg tuple (``None`` marks an unmapped/broadcast arg), and
        # tuple-valued outputs.  A single mapped array with int ``in_axes`` --
        # the long-standing contract used across the dynamical core -- behaves
        # exactly as before.
        def wrapped(*args):
            axes = in_axes if isinstance(in_axes, (tuple, list)) else (in_axes,) * len(args)
            batch = next(a.shape[ax] for a, ax in zip(args, axes) if ax is not None)
            outs = []
            for k in range(batch):
                call_args = []
                for a, ax in zip(args, axes):
                    if ax is None:
                        call_args.append(a)
                    else:
                        idx = [slice(None)] * a.ndim
                        idx[ax] = k
                        call_args.append(a[tuple(idx)])
                outs.append(func(*call_args))
            if isinstance(outs[0], (tuple, list)):
                return tuple(np.stack([o[j] for o in outs], axis=out_axes) for j in range(len(outs[0])))
            return np.stack(outs, axis=out_axes)
        return wrapped

    def scan(self, f, init, xs, reverse=False, axis=0):
        # Sequential fold matching jax.lax.scan (carry threaded, ``y`` outputs
        # stacked along ``axis``; ``axis`` lets callers scan a non-leading axis
        # without transposing). xs is a single array or a tuple of arrays.
        is_tuple = isinstance(xs, (tuple, list))
        leaves = list(xs) if is_tuple else [xs]
        n = leaves[0].shape[axis]
        order = range(n - 1, -1, -1) if reverse else range(n)
        carry = init
        ys = []
        for i in order:
            sl = [slice(None)] * leaves[0].ndim
            sl[axis] = i
            x = tuple(a[tuple(sl)] for a in leaves) if is_tuple else leaves[0][tuple(sl)]
            carry, y = f(carry, x)
            ys.append(y)
        if reverse:
            ys = ys[::-1]
        if ys[0] is None:
            # A None y-output is a valid jax.lax.scan result (an empty pytree
            # leaf); mirror that instead of trying to stack Nones.
            stacked = None
        elif isinstance(ys[0], (tuple, list)):
            stacked = tuple(None if ys[0][j] is None
                            else np.stack([y[j] for y in ys], axis=axis)
                            for j in range(len(ys[0])))
        else:
            stacked = np.stack(ys, axis=axis)
        return carry, stacked

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
        # jax.jit on by default; PYSES_JIT_COMPILE=0 runs eager for debugging.
        self._use_jit = _env_bool("PYSES_JIT_COMPILE", default=True)
        self.has_mpi = has_mpi
        self.mpi_comm = mpi_comm

        # --- precision ---
        if use_double:
            jax.config.update("jax_enable_x64", True)

        # --- sharding / device setup ---
        # TODO: exhaustively analyse nonsensical configuration combinations.
        if not do_mpi_communication:
            if use_cpu:
                self.num_devices = shard_cpu_count
                # append (don't clobber) so externally-set XLA_FLAGS — e.g. the
                # scaling harness's thread-pinning flags — survive.
                existing_xla_flags = os.environ.get("XLA_FLAGS", "")
                os.environ["XLA_FLAGS"] = (
                    f"{existing_xla_flags} "
                    f"--xla_force_host_platform_device_count={shard_cpu_count}"
                ).strip()
                devices = jax.devices(backend="cpu")
                # --xla_force_host_platform_device_count only takes effect if
                # nothing has initialized the XLA CPU backend yet.  If something
                # touched jax.devices()/jnp before this Backend was built (e.g. a
                # notebook or an interactive _reset_backend), the flag is a silent
                # no-op and we quietly get 1 device instead of shard_cpu_count.
                assert len(devices) == shard_cpu_count, (
                    f"Requested {shard_cpu_count} host CPU devices but JAX "
                    f"initialized {len(devices)}. The XLA CPU backend was likely "
                    f"already initialized before this Backend was constructed, "
                    f"making XLA_FLAGS a no-op. Build the backend before touching "
                    f"JAX, or set --xla_force_host_platform_device_count in the "
                    f"environment before import."
                )
            else:
                try:
                    maybe_devices = jax.devices(backend="gpu")
                except RuntimeError:
                    maybe_devices = []
                devices = maybe_devices if len(maybe_devices) > 0 else jax.devices(backend="cpu")
                self.num_devices = len(devices)
            # Only shard (and pay the explicit-mesh tax) when there is more than
            # one device to shard across.
            self.do_sharding = self.num_devices > 1
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
        # An ``Explicit`` mesh propagates a sharded type onto every array's
        # element axis, which makes ordinary gather/scatter (``arr[idx]``,
        # ``arr.at[idx].add(...)``) illegal without an ``out_sharding=`` and
        # forces all DSS assembly through the ``shard_map`` path. That is only
        # worthwhile with real multiple devices, so on a single device we skip
        # the mesh entirely and run with JAX's default (unsharded) array model —
        # the same simple ``index_add`` assembly branch numpy and torch use.
        elem_axis_name = "f"
        self._NamedSharding = NamedSharding
        self._PartitionSpec = PartitionSpec
        self._elem_axis_name = elem_axis_name
        if self.do_sharding:
            device_mesh = jax.make_mesh(
                (self.num_devices,), (elem_axis_name,),
                axis_types=(AxisType.Explicit,)
            )
            jax.set_mesh(device_mesh)
            self._device_mesh = device_mesh
            self.usual_scalar_sharding = NamedSharding(
                device_mesh, PartitionSpec(elem_axis_name, None, None)
            )
            self.extraction_sharding = NamedSharding(
                device_mesh, PartitionSpec(elem_axis_name, None)
            )
            self.projection_sharding = NamedSharding(
                device_mesh, PartitionSpec(elem_axis_name, None, None, None)
            )
        else:
            self._device_mesh = None
            self.usual_scalar_sharding = None
            self.extraction_sharding = None
            self.projection_sharding = None
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
        # Without a sharding mesh (single device) ``elem_sharding_axis`` is a
        # no-op: the array already lives entirely on the one device.
        if elem_sharding_axis is not None and self.do_sharding:
            x = self._jax.device_put(x, self._good_sharding(x, elem_sharding_axis))
        return x

    def unwrap(self, x):
        return np.asarray(x)

    def get_global_array(self, x, dims, elem_sharding_axis=0):
        arr = np.asarray(self._jax.device_get(x))
        if dims is not None:
            slices = [slice(None)] * x.ndim
            slices[elem_sharding_axis] = slice(0, dims["num_elem"])
            return arr[tuple(slices)]
        return arr

    def jit(self, func, *args, **kwargs):
        # Eager unless PYSES_JIT_COMPILE is set (default on); returning the raw
        # function runs it op-by-op against concrete arrays for debugging.
        if not self._use_jit:
            return func
        return self._jax.jit(func, *args, **kwargs)

    def vmap(self, func, in_axes=0, out_axes=0):
        # The whole point: XLA compiles the mapped body once instead of the
        # caller unrolling a Python loop over (e.g.) the vertical axis.
        return self._jax.vmap(func, in_axes=in_axes, out_axes=out_axes)

    def scan(self, f, init, xs, reverse=False, axis=0):
        # lax.scan compiles the loop body once (no Python unroll). lax.scan
        # only scans axis 0, so move the requested axis to the front of each
        # xs leaf and move the stacked outputs back afterwards.
        jax = self._jax
        if axis != 0:
            xs = jax.tree_util.tree_map(lambda a: self.np.moveaxis(a, axis, 0), xs)
        carry, ys = jax.lax.scan(f, init, xs, reverse=reverse)
        if axis != 0:
            ys = jax.tree_util.tree_map(lambda a: self.np.moveaxis(a, 0, axis), ys)
        return carry, ys

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

    def dss_ppermute(self, field, comm_map):
        """
        Direct Stiffness Summation across the device mesh via shard_map + ppermute.

        Performs ``field[rows] += field[cols]`` (the assembly-triple DSS) for a
        field whose element axis is sharded across the device mesh, using only
        native collectives — no mpi4py / mpi4jax.  Uses the sparse edge-colored
        schedule from
        :func:`operations_2d.local_assembly.init_global_comm_map`: one
        ``lax.ppermute`` of disjoint transpositions per round
        (``num_rounds ~ max_neighbors``, independent of device count), plus a
        purely local self term for same-device redundancies.  All gathers read the
        original field, contributions accumulate, so every redundancy entry is
        applied exactly once.

        Parameters
        ----------
        field : Array[tuple[nelem_padded, npt, npt], Float]
            Scalar field, element axis length divisible by ``num_devices``.
        comm_map : dict
            Global device/rank map from
            :func:`operations_2d.local_assembly.init_global_comm_map`.

        Returns
        -------
        Array[tuple[nelem_padded, npt, npt], Float]
            ``field`` with every shared DOF accumulated across all its copies.
        """
        from jax import lax
        jnp = self.np
        N = self.num_devices
        E = comm_map["elems_per_device"]
        num_rounds = comm_map["num_rounds"]
        perms = comm_map["perms"]
        axis = self._elem_axis_name
        P = self._PartitionSpec
        npt_i, npt_j = field.shape[1], field.shape[2]

        field_resh = field.reshape(N, E, npt_i, npt_j)
        send_e, send_i, send_j = comm_map["send_elem"], comm_map["send_i"], comm_map["send_j"]
        send_m = comm_map["send_mask"]
        recv_e, recv_i, recv_j = comm_map["recv_elem"], comm_map["recv_i"], comm_map["recv_j"]
        slf_ge, slf_gi, slf_gj = (comm_map["self_gather_elem"], comm_map["self_gather_i"],
                                  comm_map["self_gather_j"])
        slf_m = comm_map["self_mask"]
        slf_se, slf_si, slf_sj = (comm_map["self_scatter_elem"], comm_map["self_scatter_i"],
                                  comm_map["self_scatter_j"])

        def _core(fr, send_e, send_i, send_j, send_m, recv_e, recv_i, recv_j,
                  slf_ge, slf_gi, slf_gj, slf_m, slf_se, slf_si, slf_sj):
            fl0 = fr[0]                      # (E, npt, npt) — original values
            fl = fl0
            for r in range(num_rounds):
                ge, gi, gj = send_e[0, r], send_i[0, r], send_j[0, r]   # (max_pts_edge,)
                payload = fl0[ge, gi, gj] * send_m[0, r]
                got = lax.ppermute(payload, axis, list(perms[r]))
                se, si, sj = recv_e[0, r], recv_i[0, r], recv_j[0, r]
                fl = fl.at[se, si, sj].add(got)
            # local (same-device) redundancies
            self_payload = fl0[slf_ge[0], slf_gi[0], slf_gj[0]] * slf_m[0]
            fl = fl.at[slf_se[0], slf_si[0], slf_sj[0]].add(self_payload)
            return fl[jnp.newaxis]

        spec = P(axis)
        out = self._jax.shard_map(
            _core, mesh=self._device_mesh,
            in_specs=(spec,) * 15, out_specs=spec,
        )(field_resh, send_e, send_i, send_j, send_m, recv_e, recv_i, recv_j,
          slf_ge, slf_gi, slf_gj, slf_m, slf_se, slf_si, slf_sj)
        return out.reshape(field.shape)

    def halo_exchange(self, send_buffers, neighbor_ranks):
        """
        Sparse rank-keyed point-to-point exchange (mpi4py-style API).

        This dict-of-buffers signature is the multi-process (NumPy/mpi4py)
        contract.  JAX expresses DSS over the device mesh with native
        collectives instead: build the global device/rank map with
        :func:`operations_2d.local_assembly.init_global_comm_map` and apply
        :meth:`dss_ppermute` (``shard_map`` + ``lax.ppermute``).
        """
        if not neighbor_ranks:
            return dict(send_buffers)
        raise NotImplementedError(
            "JAX does not use the rank-keyed halo_exchange API. Use "
            "init_global_comm_map + JaxBackend.dss_ppermute for mesh DSS via "
            "shard_map + lax.ppermute. See MIGRATION_PLAN.md §2.5.")

    def all_reduce_sum(self, t):
        return self.np.sum(t)

    def all_reduce_max(self, t):
        return self.np.max(t)

    def all_reduce_min(self, t):
        return self.np.min(t)



# ---------------------------------------------------------------------------
# PyTorch backend
# ---------------------------------------------------------------------------

def _install_torch_numpy_compat(torch, device):
    """
    Make torch tensors interoperate with numpy the way jax arrays do.

    numpy 2.0 dispatches mixed ``numpy <op> tensor`` and ``np.ufunc(tensor)``
    through ``__array_ufunc__`` (ignoring ``__array_priority__``).  Without a
    handler torch returns ``NotImplemented`` and the op raises.  We install a
    handler that converts numpy operands to device tensors and dispatches to the
    matching torch op, so host arrays can be freely mixed with tensors.  Also
    adds ``.astype`` (torch tensors only have ``.to``).
    """
    import numpy as _np
    if not hasattr(torch.Tensor, "astype"):
        torch.Tensor.astype = lambda self, dtype: self.to(dtype)
    if getattr(torch.Tensor, "_pyses_ufunc_installed", False):
        return
    mapping = {
        _np.add: torch.add, _np.subtract: torch.subtract, _np.multiply: torch.mul,
        _np.divide: torch.div, _np.true_divide: torch.div, _np.power: torch.pow,
        _np.negative: torch.neg, _np.absolute: torch.abs,
        _np.sqrt: torch.sqrt, _np.exp: torch.exp, _np.log: torch.log,
        _np.sin: torch.sin, _np.cos: torch.cos, _np.tan: torch.tan,
        _np.arccos: torch.arccos, _np.maximum: torch.maximum, _np.minimum: torch.minimum,
        _np.greater: torch.gt, _np.less: torch.lt, _np.greater_equal: torch.ge,
        _np.less_equal: torch.le, _np.equal: torch.eq, _np.not_equal: torch.ne,
        _np.logical_and: torch.logical_and, _np.logical_or: torch.logical_or,
        _np.logical_not: torch.logical_not, _np.isnan: torch.isnan, _np.isinf: torch.isinf,
        _np.floor: torch.floor, _np.ceil: torch.ceil, _np.sign: torch.sign,
        _np.remainder: torch.remainder,
    }

    def _array_ufunc(self, ufunc, method, *inputs, **kwargs):
        fn = mapping.get(ufunc)
        if method != "__call__" or fn is None:
            return NotImplemented
        if kwargs.get("out") is not None:
            # In-place into a numpy out-buffer (e.g. `numpy_array *= tensor`):
            # evaluate on the host so the numpy buffer is written, matching how
            # numpy interops with jax arrays.
            host_inputs = [x.detach().cpu().numpy() if torch.is_tensor(x) else x
                           for x in inputs]
            return ufunc(*host_inputs, **kwargs)
        conv = [torch.as_tensor(x, device=device) if isinstance(x, _np.ndarray) else x
                for x in inputs]
        return fn(*conv)

    torch.Tensor.__array_ufunc__ = _array_ufunc
    torch.Tensor._pyses_ufunc_installed = True


class _TorchNamespace:
    """
    NumPy-API-compatible namespace over torch, used as ``TorchBackend.np`` and
    aliased as ``jnp`` in the science modules.

    Names defined explicitly here adapt the torch signatures that differ from
    numpy/jax (``dim`` vs ``axis``, ``amax``/``amin`` vs ``max``/``min``,
    ``take_along_dim`` vs ``take_along_axis``, ``flip(dims=)``, ...).  Every other
    attribute falls through to ``torch`` via :meth:`__getattr__`, so the large set
    of name-and-signature-compatible ops (``einsum``, ``where``, ``zeros``,
    ``sqrt``, ``eye``, ``linalg``, ``pi``, dtypes, ...) needs no enumeration.
    Factory functions inherit the process-wide default device/dtype set in
    :class:`TorchBackend`, so created tensors land on the accelerator.
    """

    def __init__(self, torch, device):
        self._torch = torch
        self._device = device
        self.newaxis = None

    def _coerce(self, x):
        # Mirror jax.numpy, which promotes numpy arrays inside ops. Only
        # numpy.ndarray is coerced — tuples/lists are left alone so that shape
        # and axis arguments (e.g. jnp.ones((2, 3))) pass through untouched.
        if isinstance(x, np.ndarray):
            return self._torch.as_tensor(x, device=self._device)
        return x

    def _coerce_num(self, x):
        # For elementwise math (sin, sqrt, ...) numpy/jax accept python scalars;
        # torch needs tensors. Safe here because these ops never take shape args.
        if isinstance(x, (np.ndarray, int, float, complex)):
            return self._torch.as_tensor(x, device=self._device)
        return x

    def __getattr__(self, name):
        # Reached only for names not defined as instance/class attributes above
        # (e.g. einsum, where, zeros, sqrt, eye, linalg, pi, dtypes). Callables
        # are wrapped to coerce numpy/python array args to device tensors.
        attr = getattr(self._torch, name)
        if not callable(attr):
            return attr

        def _wrapped(*args, **kwargs):
            args = tuple(self._coerce(a) for a in args)
            kwargs = {k: self._coerce(v) for k, v in kwargs.items()}
            return attr(*args, **kwargs)
        return _wrapped

    def flip(self, x, axis=None):
        x = self._coerce(x)
        if axis is None:
            dims = tuple(range(x.ndim))
        elif isinstance(axis, int):
            dims = (axis,)
        else:
            dims = tuple(axis)
        return self._torch.flip(x, dims=dims)

    def cumsum(self, x, axis=-1):
        return self._torch.cumsum(self._coerce(x), dim=axis)

    def diff(self, a, n=1, axis=-1, prepend=None, append=None):
        # numpy/jax accept a scalar prepend/append that broadcasts along the
        # diff axis (e.g. jnp.diff(x, prepend=0.0)); torch.diff requires a tensor
        # shaped like ``a`` with size 1 along that axis. Coerce scalars to that
        # shaped tensor so callers written against the numpy API work unchanged.
        a = self._coerce(a)

        def _pad(val):
            if isinstance(val, (int, float, complex)):
                shape = list(a.shape)
                shape[axis] = 1
                return self._torch.full(shape, val, dtype=a.dtype, device=self._device)
            return self._coerce(val)

        kwargs = {}
        if prepend is not None:
            kwargs["prepend"] = _pad(prepend)
        if append is not None:
            kwargs["append"] = _pad(append)
        return self._torch.diff(a, n=n, dim=axis, **kwargs)

    def sum(self, x, axis=None):
        x = self._coerce(x)
        return self._torch.sum(x) if axis is None else self._torch.sum(x, dim=axis)

    def max(self, x, axis=None):
        x = self._coerce(x)
        return self._torch.max(x) if axis is None else self._torch.amax(x, dim=axis)

    def min(self, x, axis=None):
        x = self._coerce(x)
        return self._torch.min(x) if axis is None else self._torch.amin(x, dim=axis)

    def all(self, x, axis=None):
        x = self._coerce(x)
        return self._torch.all(x) if axis is None else self._torch.all(x, dim=axis)

    def any(self, x, axis=None):
        x = self._coerce(x)
        return self._torch.any(x) if axis is None else self._torch.any(x, dim=axis)

    def stack(self, arrays, axis=0):
        return self._torch.stack([self._coerce(a) for a in arrays], dim=axis)

    def concatenate(self, arrays, axis=0):
        return self._torch.cat([self._coerce(a) for a in arrays], dim=axis)

    def take_along_axis(self, arr, indices, axis):
        return self._torch.take_along_dim(self._coerce(arr), self._coerce(indices), dim=axis)

    def remainder(self, x, y):
        return self._torch.remainder(self._coerce(x), self._coerce(y))

    def mod(self, x, y):
        return self._torch.remainder(self._coerce(x), self._coerce(y))

    def allclose(self, a, b, **kwargs):
        a = self._torch.as_tensor(a, device=self._device)
        b = self._torch.as_tensor(b, device=self._device)
        if a.dtype != b.dtype:  # numpy/jax promote mixed dtypes; torch does not
            a, b = a.to(self._torch.float64), b.to(self._torch.float64)
        return self._torch.allclose(a, b, **kwargs)

    def sin(self, x):
        return self._torch.sin(self._coerce_num(x))

    def cos(self, x):
        return self._torch.cos(self._coerce_num(x))

    def tan(self, x):
        return self._torch.tan(self._coerce_num(x))

    def arccos(self, x):
        return self._torch.arccos(self._coerce_num(x))

    def sqrt(self, x):
        return self._torch.sqrt(self._coerce_num(x))

    def exp(self, x):
        return self._torch.exp(self._coerce_num(x))

    def log(self, x):
        return self._torch.log(self._coerce_num(x))

    def abs(self, x):
        return self._torch.abs(self._coerce_num(x))

    def mean(self, x, axis=None):
        x = self._coerce(x)
        return self._torch.mean(x) if axis is None else self._torch.mean(x, dim=axis)

    def nanmax(self, x, axis=None):
        x = self._torch.nan_to_num(self._coerce(x), nan=float("-inf"))
        return self.max(x, axis)

    def nanmin(self, x, axis=None):
        x = self._torch.nan_to_num(self._coerce(x), nan=float("inf"))
        return self.min(x, axis)

    def take(self, a, indices, axis=None):
        a = self._coerce(a)
        if axis is None:
            return self._torch.take(a, self._coerce(indices))
        if isinstance(indices, int):
            return self._torch.select(a, dim=axis, index=indices)
        return self._torch.index_select(a, axis, self._coerce(indices))

    def where(self, condition, x=None, y=None):
        # numpy/jax accept python-bool/scalar conditions and branches; torch
        # needs a tensor condition (a scalar comparison like `cfg_float > 0`
        # yields a python bool).
        if x is None and y is None:
            return self._torch.where(self._coerce_num(condition))
        return self._torch.where(self._coerce_num(condition),
                                 self._coerce_num(x), self._coerce_num(y))

    def maximum(self, x, y):
        # numpy/jax accept python scalars on either side; torch needs both
        # operands to be tensors (e.g. jnp.maximum(arr, 1e-30)).
        return self._torch.maximum(self._coerce_num(x), self._coerce_num(y))

    def minimum(self, x, y):
        return self._torch.minimum(self._coerce_num(x), self._coerce_num(y))

    def broadcast_to(self, x, shape):
        # The value being broadcast may be a python scalar (e.g. phi_to_g
        # returns g for shallow-atmosphere models); torch needs a tensor.
        return self._torch.broadcast_to(self._coerce_num(x), shape)

    def array(self, x, dtype=None):
        return self._torch.as_tensor(x, dtype=dtype, device=self._device)

    def asarray(self, x, dtype=None):
        return self._torch.as_tensor(x, dtype=dtype, device=self._device)

    def copy(self, x):
        return self._torch.as_tensor(x, device=self._device).clone()


class TorchBackend:
    """PyTorch backend with autograd-aware ``torch.distributed`` collectives."""

    newaxis = None
    use_wrapper = True
    wrapper_type = "torch"
    # JAX sharding attributes — not applicable; exposed as None for uniform access
    usual_scalar_sharding = None
    extraction_sharding = None
    projection_sharding = None
    do_sharding = False

    def __init__(self,
                 use_double: bool,
                 debug: bool,
                 use_cpu: bool,
                 do_mpi_communication: bool,
                 mpi_rank: int,
                 mpi_size: int,
                 has_mpi: bool,
                 mpi_comm):
        import torch

        self._torch = torch
        self.use_double = use_double
        self.debug = debug
        self.do_mpi_communication = do_mpi_communication
        self.mpi_rank = mpi_rank
        self.mpi_size = mpi_size
        self.is_main_proc = mpi_rank == 0
        self.has_mpi = has_mpi
        self.mpi_comm = mpi_comm
        self.num_devices = 1
        self.num_jax_devices = 1
        self.eps = 1e-11 if use_double else 1e-6
        # torch.compile is on by default (end users get compiled kernels); the
        # test suite — which sweeps many shapes and would pay Dynamo recompiles
        # on every one — disables it with PYSES_JIT_COMPILE=0.
        self._use_jit = _env_bool("PYSES_JIT_COMPILE", default=True)

        # --- precision (process-wide default so factory functions match) ---
        self._default_dtype = torch.float64 if use_double else torch.float32
        torch.set_default_dtype(self._default_dtype)

        # --- device selection and process-wide default ---
        if use_cpu:
            device = torch.device("cpu")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        self._device = device
        torch.set_default_device(device)

        # numpy interop (`.astype`, mixed numpy/tensor ops) — see helper.
        _install_torch_numpy_compat(torch, device)

        self.np = _TorchNamespace(torch, device)

        # --- distributed bootstrap (multi-process only) ---
        if do_mpi_communication:
            self._init_torch_dist(mpi_comm)
            import torch.distributed as dist
            self._dist = dist
            self._world_size = dist.get_world_size()
        else:
            self._dist = None
            self._world_size = 1

        if debug:
            print(f"Using torch device {device}, dtype {self._default_dtype}, "
                  f"distributed: {self._dist is not None}")

    # --- distributed setup ---

    def _init_torch_dist(self, mpi_comm):
        """Bootstrap torch.distributed via the existing mpi4py communicator."""
        import torch, torch.distributed as dist, os, socket
        rank = mpi_comm.Get_rank()
        world_size = mpi_comm.Get_size()
        if rank == 0:
            host = socket.gethostname()
            with socket.socket() as s:
                s.bind(("", 0))
                port = s.getsockname()[1]
            addr = (host, port)
        else:
            addr = None
        host, port = mpi_comm.bcast(addr, root=0)
        os.environ["MASTER_ADDR"] = host
        os.environ["MASTER_PORT"] = str(port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        local_rank = self._resolve_local_rank()
        os.environ["LOCAL_RANK"] = str(local_rank)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            backend = "nccl"
        else:
            backend = "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    @staticmethod
    def _resolve_local_rank():
        import os
        for var in ("SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK",
                    "MV2_COMM_WORLD_LOCAL_RANK", "PMI_LOCAL_RANK", "LOCAL_RANK"):
            if var in os.environ:
                return int(os.environ[var])
        return 0

    # --- Backend interface ---

    def array(self, x, dtype=None, elem_sharding_axis=None):
        dt = dtype if dtype is not None else self._default_dtype
        if not isinstance(dt, self._torch.dtype):
            # Accept numpy dtypes/strings (e.g. a restored array's arr.dtype);
            # torch.as_tensor only takes torch.dtype.
            dt = self._torch.as_tensor(np.empty(0, dtype=dt)).dtype
        return self._torch.as_tensor(x, dtype=dt, device=self._device)

    def unwrap(self, x):
        if self._torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def get_global_array(self, x, dims, elem_sharding_axis=0):
        arr = self.unwrap(x)
        if dims is not None:
            slices = [slice(None)] * arr.ndim
            slices[elem_sharding_axis] = slice(0, dims["num_elem"])
            return arr[tuple(slices)]
        return arr

    def jit(self, func, *_, static_argnames=(), **__):
        # Eager unless PYSES_JIT_COMPILE is set (default on). Eager mixes
        # numpy/tensor freely via __array_ufunc__, so no coercion is needed.
        if not self._use_jit:
            return func
        # This mirrors three jax.jit behaviours that plain torch.compile does
        # not provide and whose absence drove the Dynamo recompile/guard
        # problems:
        #   1. static_argnames select a separately-compiled graph per value.
        #      Dynamo's automatic guards do not reliably re-specialize on
        #      enum/struct values, so a graph traced for one ``model`` was
        #      reused for another -- silently dropping model-dependent dict
        #      keys. We key an explicit per-(static value) cache instead.
        #   2. Nested jit is transparent: a jitted call already inside an
        #      enclosing compile runs inline (traced into the outer graph)
        #      rather than forming its own guarded, separately-recompiling
        #      boundary.
        #   3. Each compiled entry gets a distinct code object. Dynamo caches by
        #      code object, so without this every jitted function (and every
        #      static-arg specialization) collapses into one polymorphic frame
        #      that recompiles per call signature until it hits recompile_limit.
        # numpy array inputs are coerced to device tensors at the boundary
        # (Dynamo cannot trace numpy; jax.jit converts them implicitly).
        import inspect
        from torch.utils._pytree import tree_map
        torch_mod = self._torch
        device = self._device
        if isinstance(static_argnames, str):
            static_argnames = (static_argnames,)
        static_argnames = tuple(static_argnames)
        try:
            params = inspect.signature(func).parameters
        except (TypeError, ValueError):
            params = {}
        param_names = list(params)
        fname = getattr(func, "__name__", "fn")

        def _to_tensor(x):
            if isinstance(x, np.ndarray):
                return torch_mod.as_tensor(x, device=device)
            return x

        def _static_key(args, kwargs):
            key = []
            for name in static_argnames:
                if name in kwargs:
                    key.append(kwargs[name])
                elif name in param_names and param_names.index(name) < len(args):
                    key.append(args[param_names.index(name)])
                elif name in params and params[name].default is not inspect.Parameter.empty:
                    key.append(params[name].default)
                else:
                    key.append(None)
            return tuple(key)

        cache = {}

        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            if torch_mod.compiler.is_compiling():
                return func(*args, **kwargs)
            try:
                key = _static_key(args, kwargs)
                compiled = cache.get(key)
            except TypeError:           # unhashable static value -> skip caching
                key, compiled = None, None
            if compiled is None:
                # Fresh closure per (static combo); the code-object replace gives
                # it a unique identity so Dynamo compiles it independently.
                def _entry(*a, **k):
                    return func(*a, **k)
                _entry.__code__ = _entry.__code__.replace(co_name=f"_jit_{fname}")
                compiled = torch_mod.compile(_entry)
                if key is not None:
                    cache[key] = compiled
            return compiled(*tree_map(_to_tensor, args),
                            **tree_map(_to_tensor, kwargs))
        return _wrapper

    def vmap(self, func, in_axes=0, out_axes=0):
        # Single-array contract matching jax.vmap, implemented with the real
        # batching transform torch.func.vmap: the mapped axis never enters the
        # traced graph, so torch.compile time is independent of its length. A
        # Python loop + stack would instead unroll that axis into the graph and
        # make compile time scale with it (e.g. the level count), which was the
        # dominant cost before this was switched over.
        torch = self._torch

        def wrapped(*args):
            args = tuple(self.np._coerce(a) for a in args)
            if len(args) == 1 and not isinstance(in_axes, (tuple, list)):
                # Single-array fast path (unchanged behaviour).
                in_dims = in_axes % args[0].ndim
                return torch.func.vmap(func, in_dims=in_dims, out_dims=out_axes)(args[0])
            # Multiple positional args: an int ``in_axes`` maps that axis on every
            # arg; a per-arg tuple gives each its own axis (``None`` = unmapped /
            # broadcast). Matches NumpyBackend.vmap and jax.vmap.
            axes = in_axes if isinstance(in_axes, (tuple, list)) else (in_axes,) * len(args)
            in_dims = tuple(None if ax is None else (ax % a.ndim)
                            for a, ax in zip(args, axes))
            return torch.func.vmap(func, in_dims=in_dims, out_dims=out_axes)(*args)
        return wrapped

    def scan(self, f, init, xs, reverse=False, axis=0):
        # Sequential fold matching jax.lax.scan (see NumpyBackend.scan).
        is_tuple = isinstance(xs, (tuple, list))
        leaves = list(xs) if is_tuple else [xs]
        n = leaves[0].shape[axis]
        order = range(n - 1, -1, -1) if reverse else range(n)
        carry = init
        ys = []
        for i in order:
            sl = [slice(None)] * leaves[0].ndim
            sl[axis] = i
            x = tuple(a[tuple(sl)] for a in leaves) if is_tuple else leaves[0][tuple(sl)]
            carry, y = f(carry, x)
            ys.append(y)
        if reverse:
            ys = ys[::-1]
        if ys[0] is None:
            # A None y-output is a valid jax.lax.scan result (an empty pytree
            # leaf); mirror that instead of trying to stack Nones.
            stacked = None
        elif isinstance(ys[0], (tuple, list)):
            stacked = tuple(None if ys[0][j] is None
                            else self.np.stack([y[j] for y in ys], axis=axis)
                            for j in range(len(ys[0])))
        else:
            stacked = self.np.stack(ys, axis=axis)
        return carry, stacked

    def shard_map(self, func, *_, **__):
        return func

    def parallel_region(self, fn, in_specs=None, out_specs=None):
        return fn

    # --- scatter / gather ---

    def _scatter(self, arr, idx, vals, reduce):
        torch = self._torch
        tensor_idx = [i for i in idx if not isinstance(i, slice)]
        n = len(tensor_idx)
        trailing = tuple(arr.shape[n:])
        flat_lead = 1
        for s in arr.shape[:n]:
            flat_lead *= s
        flat = tensor_idx[0]
        for d in range(1, n):
            flat = flat * arr.shape[d] + tensor_idx[d]
        flat = flat.reshape(-1).to(torch.long)
        arr_flat = arr.reshape((flat_lead,) + trailing)
        src = vals.reshape((-1,) + trailing)
        result = arr_flat.clone()
        if reduce == "sum":
            result.index_add_(0, flat, src)
        else:
            result.index_reduce_(0, flat, src, reduce=reduce, include_self=True)
        return result.reshape(arr.shape)

    def index_get(self, arr, idx):
        return arr[tuple(idx)]

    def index_add(self, arr, idx, vals):
        return self._scatter(arr, idx, vals, "sum")

    def index_max(self, arr, idx, vals):
        return self._scatter(arr, idx, vals, "amax")

    # --- collectives ---

    def halo_exchange(self, send_buffers, neighbor_ranks):
        """
        Sparse point-to-point exchange via ``all_to_all_single`` (zero splits for
        non-neighbors).  Its autograd adjoint swaps the split sizes — the
        mathematical adjoint of a halo exchange — so it is forward- and
        reverse-differentiable for DSS.
        """
        torch = self._torch
        if self._dist is None or not neighbor_ranks:
            return {k: send_buffers[k].clone() for k in neighbor_ranks}
        from torch.distributed.nn.functional import all_to_all_single

        # Buffers are (*payload, n_points) — the point count (last axis) varies
        # per neighbor; the leading payload (e.g. levels) is constant.  Move the
        # point axis to the front so each neighbor's chunk is contiguous for the
        # split-based all-to-all.
        sample = next(iter(send_buffers.values()))
        payload_shape = tuple(sample.shape[:-1])
        payload_numel = 1
        for s in payload_shape:
            payload_numel *= s

        input_split_sizes = [0] * self._world_size
        send_list = []
        for k in neighbor_ranks:
            buf = send_buffers[k]                              # (*payload, n_points_k)
            send_list.append(torch.movedim(buf, -1, 0).reshape(-1))
            input_split_sizes[k] = buf.shape[-1] * payload_numel
        # DSS bilateral symmetry: recv count from k == send count to k.
        output_split_sizes = list(input_split_sizes)

        input_tensor = (torch.cat(send_list) if send_list
                        else torch.zeros(0, dtype=sample.dtype, device=sample.device))
        # functional all_to_all_single needs a preallocated output (output, input, ...)
        output_tensor = torch.empty(int(sum(output_split_sizes)),
                                    dtype=input_tensor.dtype, device=input_tensor.device)
        output_tensor = all_to_all_single(
            output_tensor,
            input_tensor,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
        )

        recv_buffers = {}
        offset = 0
        for k in neighbor_ranks:
            n_flat = output_split_sizes[k]
            n_points = n_flat // payload_numel
            chunk = output_tensor[offset:offset + n_flat].reshape((n_points,) + payload_shape)
            recv_buffers[k] = torch.movedim(chunk, 0, -1)      # back to (*payload, n_points)
            offset += n_flat
        return recv_buffers

    def all_reduce_sum(self, t):
        t = self._torch.as_tensor(t, device=self._device)
        if self._dist is None:
            return self.np.sum(t)
        from torch.distributed.nn.functional import all_reduce
        return all_reduce(t)

    def all_reduce_max(self, t):
        t = self._torch.as_tensor(t, device=self._device)
        if self._dist is None:
            return self.np.max(t)
        out = t.clone()
        self._dist.all_reduce(out, op=self._dist.ReduceOp.MAX)
        return out

    def all_reduce_min(self, t):
        t = self._torch.as_tensor(t, device=self._device)
        if self._dist is None:
            return self.np.min(t)
        out = t.clone()
        self._dist.all_reduce(out, op=self._dist.ReduceOp.MIN)
        return out



# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _build_backend() -> Backend:
    backend_name = os.environ.get("PYSES_BACKEND", "numpy").strip().lower()
    use_double = _env_bool("PYSES_USE_DOUBLE", default=True)
    debug = _env_bool("PYSES_DEBUG", default=False)
    use_mpi = _env_bool("PYSES_USE_MPI", default=False)
    use_cpu = _env_bool("PYSES_USE_CPU", default=False)
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
    elif backend_name == "torch":
        return TorchBackend(
            use_double=use_double,
            debug=debug,
            use_cpu=use_cpu,
            do_mpi_communication=do_mpi,
            mpi_rank=rank,
            mpi_size=size,
            has_mpi=has_mpi,
            mpi_comm=mpi_comm,
        )
    else:
        raise ValueError(
            f"Unknown PYSES_BACKEND={backend_name!r}. Must be 'numpy', 'jax', or 'torch'."
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
        # jax.debug.callback inserts a device->host round-trip into the XLA
        # program, serializing the async dispatch pipeline (and it runs on every
        # invocation -- e.g. once per tracer sub-cycle from the remap).  Gate it
        # on PYSES_DEBUG so it compiles out of production runs.
        if not be.debug:
            return
        import jax

        def _check(ok):
            if not bool(ok):
                raise RuntimeError(message)

        jax.debug.callback(_check, condition)
    else:
        assert condition, message
