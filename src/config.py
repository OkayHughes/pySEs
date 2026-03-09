import os
from json import dumps
import numpy as np
from json import loads
from typing import Hashable
try:
  from mpi4py import MPI
  has_mpi = True
except ImportError:
  has_mpi = False


def get_config_filepath():
  """
  Return the path to the pysces configuration JSON file.

  The configuration file is located in the current working directory
  so that each run directory can have its own computational settings.

  Returns
  -------
  `str`
      Absolute path to ``pysces_config.json`` in the current
      working directory.
  """
  # return os.path.join(os.path.dirname(os.path.abspath(__file__)), "pysces_config.json")
  return os.path.join(os.getcwd(), "pysces_config.json")


def write_config(debug=True,
                 use_mpi=False,
                 use_wrapper=False,
                 wrapper_type="none",
                 use_cpu=True,
                 use_double=True,
                 shard_cpu_count=1):
  """
  Write a pysces computational configuration to ``pysces_config.json``
  in the current working directory.

  Parameters
  ----------
  debug : `bool`, default=True
      Enable debug mode (extra runtime checks and verbose output).
  use_mpi : `bool`, default=False
      Enable MPI distributed-memory parallelism.
  use_wrapper : `bool`, default=False
      Wrap arrays in the device library specified by ``wrapper_type``.
  wrapper_type : `str`, default="none"
      Device wrapper library to use.  Must be one of
      ``"none"``, ``"jax"``, or ``"torch"``.
  use_cpu : `bool`, default=True
      Force computation on CPU even when a GPU is available.
  use_double : `bool`, default=True
      Use double-precision (64-bit) floating point.
  shard_cpu_count : `int`, default=1
      Number of virtual CPU devices to shard across when using
      JAX CPU sharding.

  Notes
  -----
  The configuration is written as a JSON file and read back at
  import time by ``parse_config_file``.  Re-import (or restart)
  the Python process after calling this function for the new
  settings to take effect.
  """
  config_struct = {"debug": debug,
                   "use_mpi": use_mpi,
                   "use_wrapper": use_wrapper,
                   "wrapper_type": wrapper_type,
                   "use_cpu": use_cpu,
                   "use_double": use_double,
                   "shard_cpu_count": shard_cpu_count}
  with open(get_config_filepath(), "w") as config_file:
    config_file.write(dumps(config_struct, indent=2))


def parse_config_file():
  """
  Read the pysces configuration JSON file and return it as a dict.

  If no configuration file exists, a default serial NumPy configuration
  is written and the process is asked to restart.

  Returns
  -------
  config_vars : `dict`
      Configuration dictionary with keys ``"debug"``, ``"use_mpi"``,
      ``"use_wrapper"``, ``"wrapper_type"``, ``"use_cpu"``,
      ``"use_double"``, and ``"shard_cpu_count"``.
  """
  config_filename = get_config_filepath()
  try:
    assert os.path.isfile(config_filename)
  except AssertionError:
    write_config()
    print ("Config file is not written, writing serial numpy config file\n"
           "Run program again if this is acceptable, or run pysces.set_config\n"
           "with the appropriate computing environment configuration")
  with open(config_filename, "r") as f:
    config_vars = loads(f.read())
  return config_vars


config_vars = parse_config_file()

DEBUG = config_vars["debug"]

if DEBUG:
  print("=" * 20)
  print("For the next few days, using an even number of elements per cube face will cause.")
  print("a blowup at the pole point. This was due to a small error, and I've identified a fix.")
  print("=" * 20)


use_wrapper = config_vars["use_wrapper"]
wrapper_type = config_vars["wrapper_type"]
use_cpu = config_vars["use_cpu"]
use_double = config_vars["use_double"]


if use_double:
  eps = 1e-11
else:
  eps = 1e-6

if has_mpi:
  mpi_comm = MPI.COMM_WORLD
  mpi_rank = mpi_comm.Get_rank()
  mpi_size = mpi_comm.Get_size()
  is_main_proc = mpi_rank == 0
  do_mpi_communication = mpi_size > 1
else:
  do_mpi_communication = False
  mpi_rank = 0
  mpi_size = 1
  is_main_proc = True

num_jax_devices = 1
do_sharding = False

if wrapper_type == "jax" and use_wrapper:
  import jax

  # set jax global config
  # =====================
  # TODO: exhaustively analyze if this does the right thing for
  # nonsensical configurations.

  if not do_mpi_communication:
    do_sharding = True
    if use_cpu:
      num_jax_devices = config_vars["shard_cpu_count"]
      os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={num_jax_devices}"
      devices = jax.devices(backend="cpu")
    else:
      maybe_devices = jax.devices(backend="gpu")
      if len(maybe_devices) > 0:
        devices = maybe_devices
      else:
        devices = jax.devices(backend="cpu")
  else:
    do_sharding = False
    if use_cpu:
      jax.config.update("jax_default_device", jax.local_devices("cpu")[0])
      num_jax_devices = 1

  if use_double:
    jax.config.update("jax_enable_x64", True)

  # ========================================

  if DEBUG:
    print(f"Using devices {devices}, num_jax_devices: {num_jax_devices}, do_sharding: {do_sharding}")

  import jax.numpy as jnp
  import jax

  from jax.sharding import PartitionSpec, NamedSharding, AxisType
  elem_axis_name = "f"
  device_mesh = jax.make_mesh((num_jax_devices,), (elem_axis_name,), axis_types=(AxisType.Explicit,))
  jax.set_mesh(device_mesh)
  usual_scalar_sharding = NamedSharding(device_mesh, PartitionSpec(elem_axis_name, None, None))
  extraction_sharding = NamedSharding(device_mesh, PartitionSpec(elem_axis_name, None))
  projection_sharding = NamedSharding(device_mesh, PartitionSpec(elem_axis_name, None, None, None))

  def good_sharding(array, elem_sharding_axis):
    spec_names = [None for _ in range(len(array.shape))]
    spec_names[elem_sharding_axis] = elem_axis_name
    return NamedSharding(device_mesh, PartitionSpec(*spec_names))

  def device_wrapper(x,
                     dtype=jnp.float64,
                     elem_sharding_axis=None):
    x = jnp.array(x, dtype=dtype)
    if elem_sharding_axis is not None:
      x = jax.device_put(x, good_sharding(x, elem_sharding_axis))
    return x

  def get_global_array(x, dims, elem_sharding_axis=0):
    arr = np.asarray(jax.device_get(x))
    if dims is not None:
      slices = [slice(None, None) for _ in range(x.ndim)]
      slices[elem_sharding_axis] = slice(0, dims["num_elem"])
      res = arr[*slices]
    else:
      res = arr
    return res
  jit = jax.jit

  def device_unwrapper(x):
    return np.asarray(x)
  jit = jax.jit

  def versatile_assert(should_be_true):
    return

  from jax.tree_util import Partial as partial

  def vmap_1d_apply(func,
                    vector,
                    in_axis,
                    out_axis):
      return jax.vmap(func, in_axes=(in_axis), out_axes=(out_axis))(vector)

  def flip(array,
           axis):
    return jnp.flip(array, axis=axis)

  def remainder(array,
                divisor):
    return jnp.mod(array, divisor)

  def take_along_axis(array,
                      idxs,
                      axis):
    return jnp.take_along_axis(array, idxs, axis=axis)

  def cast_type(arr,
                dtype):
    return arr.astype(dtype)
  shard_map = jax.shard_map

else:
  import numpy as jnp

  def device_wrapper(x,
                     dtype=np.float64,
                     elem_sharding_axis=None):
    return np.array(x, dtype=dtype)

  def device_unwrapper(x):
    return x

  def jit(func, *_, **__):
    return func

  def shard_map(func, *_, **__):
    return func

  def versatile_assert(should_be_true):
    assert should_be_true

  from functools import partial

  def vmap_1d_apply(func,
                    scalar,
                    in_axis,
                    out_axis):
    levs = []
    for lev_idx in range(scalar.shape[in_axis]):
      scalar_2d = scalar.take(indices=lev_idx, axis=in_axis)
      levs.append(func(scalar_2d))
    return np.stack(levs, axis=out_axis)

  def flip(array,
           axis):
    return jnp.flip(array, axis=axis)

  def remainder(array,
                divisor):
    return jnp.mod(array, divisor)

  def take_along_axis(array,
                      idxs,
                      axis):
    return jnp.take_along_axis(array, idxs, axis=axis)

  def cast_type(arr,
                dtype):
    return arr.astype(dtype)

  def get_global_array(x, dims, elem_sharding_axis=0):
    return x


class grid_info():
  def __init__(self, **kwargs):
    for k, v in zip(kwargs.keys(), kwargs.values()):
      assert isinstance(k, Hashable), f"Unhashable key {k} passed to grid_info"
      assert isinstance(v, Hashable), f"Unhashable value {v} passed to grid_info"
    self._dict = frozenset([(k, v) for k, v in zip(kwargs.keys(), kwargs.values())])

  def __hash__(self):
    return hash(self._dict)

  def __getitem__(self, key):
    for k, v in self._dict:
      if k == key:
        return v


assert not (do_sharding and do_mpi_communication), "Sharding in an MPI environment is not presently supported"
