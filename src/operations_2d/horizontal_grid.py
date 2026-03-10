import numpy as np
from .._config import get_backend as _get_backend
from frozendict import frozendict
from .local_assembly import (triage_vert_redundancy_flat,
                             init_assembly_global,
                             init_assembly_local,
                             project_scalar,
                             init_shard_extraction_map)
from ..mpi.processor_decomposition import init_decomp
from ..mesh_generation.bilinear_utils import eval_bilinear_mapping
from ..mpi.global_communication import global_max, global_min
from .tensor_hyperviscosity import eval_hypervis_tensor
from ..mesh_generation.spectral import init_spectral
from math import floor
_be = _get_backend()
device_wrapper = _be.array
use_wrapper = _be.use_wrapper
device_unwrapper = _be.unwrap
jnp = _be.np
num_jax_devices = _be.num_jax_devices
get_global_array = _be.get_global_array
DEBUG = _be.debug
do_mpi_communication = _be.do_mpi_communication
mpi_size = _be.mpi_size


def reorder_parallel_axis(var, element_reordering=None, wrapped=use_wrapper):
  """
  Reorder the elements of an array along the first (element) axis
  according to a given index permutation.

  Parameters
  ----------
  var : `Array[tuple[elem_idx, ...], Any]`
      The array whose element axis is to be reordered.
  element_reordering : `Array[tuple[elem_idx], Int]`, optional
      Index permutation to apply along the element axis.
      If ``None``, the identity permutation is used.
  wrapped : `bool`, default=use_wrapper
      If ``True``, the array is unwrapped from the device
      wrapper before reordering and re-wrapped afterwards.

  Returns
  -------
  var_out : `Array[tuple[elem_idx, ...], Any]`
      A copy of ``var`` with elements permuted along the first axis.
  """
  NELEM_GLOBAL = var.shape[0]
  if element_reordering is None:
    element_reordering = np.arange(0, NELEM_GLOBAL)
  dtype = var.dtype
  if wrapped:
    var_np = device_unwrapper(var)
  else:
    var_np = var
  var_subset = np.take(var_np, element_reordering, axis=0)
  if wrapped:
    var_out = device_wrapper(var_subset, dtype=dtype)
  else:
    var_out = var_subset
  return var_out


def init_spectral_element_grid(latlon,
                               gll_to_sphere_jacobian,
                               gll_to_sphere_jacobian_inv,
                               physical_coords_to_cartesian,
                               rmetdet,
                               metdet,
                               mass_mat,
                               inv_mass_mat,
                               vert_redundancy_gll_flat,
                               element_reordering=None,
                               wrapped=use_wrapper):
  """
  Assemble a ``SpectralElementGrid`` dict from mesh metric data computed
  during mesh generation.

  Parameters
  ----------
  latlon : `Array[tuple[elem_idx, gll_idx, gll_idx, lat_lon], Float]`
      Physical (latitude, longitude) coordinates at every GLL node.
  gll_to_sphere_jacobian : `Array[tuple[elem_idx, gll_idx, gll_idx, lon_lat, alpha_beta], Float]`
      Jacobian mapping from reference-element contravariant components
      to physical (lon, lat) vector components.
  gll_to_sphere_jacobian_inv : `Array[tuple[elem_idx, gll_idx, gll_idx, alpha_beta, lon_lat], Float]`
      Inverse Jacobian mapping from physical to reference-element
      contravariant components.
  physical_coords_to_cartesian : `Array[tuple[elem_idx, gll_idx, gll_idx, xyz], Float]`
      Jacobian mapping physical velocity components to 3-D Cartesian
      velocity components.
  rmetdet : `Array[tuple[elem_idx, gll_idx, gll_idx], Float]`
      Reciprocal of the metric determinant at each GLL node.
  metdet : `Array[tuple[elem_idx, gll_idx, gll_idx], Float]`
      Metric determinant at each GLL node.
  mass_mat : `Array[tuple[elem_idx, gll_idx, gll_idx], Float]`
      Element-local mass matrix (quadrature weights times metric determinant).
  inv_mass_mat : `Array[tuple[elem_idx, gll_idx, gll_idx], Float]`
      Reciprocal of the mass matrix.
  vert_redundancy_gll_flat : flat redundancy structure
      Flat representation of vertex DOF redundancy
      (see ``init_assembly_local``).
  element_reordering : `Array[tuple[elem_idx], Int]`, optional
      Index permutation applied to all element-indexed arrays
      after wrapping.  If ``None``, the natural ordering is kept.
  wrapped : `bool`, default=use_wrapper
      If ``True``, arrays are moved onto the configured device
      (GPU / JAX shards) before being stored in the grid struct.

  Returns
  -------
  grid : `dict[str, Array]`
      ``SpectralElementGrid`` dict containing all metric, assembly,
      and spectral quantities needed by the operators in
      ``operations_2d``.
  dims : frozendict[str, int]
      Scalar grid dimensions (``N``, ``shape``, ``npt``, ``num_elem``).

  Notes
  -----
  This function assumes that the full global grid fits in memory.
  For distributed-memory parallelism, call ``make_grid_mpi_ready``
  after this function.
  """
  # note: test code sometimes sets wrapped=False to test wrapper library (jax, torch) vs stock numpy
  # this extra conditional is not extraneous.
  if wrapped:
    wrapper = device_wrapper
  else:
    def wrapper(x, dtype=None):
      return x

  def reorder_wrapper(field, dtype=None):
    return reorder_parallel_axis(wrapper(field, dtype=dtype),
                                 element_reordering=element_reordering, wrapped=wrapped)

  npt = metdet.shape[1]
  # This function currently assumes that the full grid can be loaded into memory.
  # This should be fine up to, e.g., quarter-degree grids.

  spectrals = init_spectral(npt)

  assembly_triple = init_assembly_local(vert_redundancy_gll_flat)

  met_inv = np.einsum("fijgs, fijhs->fijgh",
                      gll_to_sphere_jacobian_inv,
                      gll_to_sphere_jacobian_inv)
  mass_matrix = (metdet *
                 spectrals["gll_weights"][np.newaxis, :, np.newaxis] *
                 spectrals["gll_weights"][np.newaxis, np.newaxis, :])
  viscosity_tensor, hypervis_scaling = eval_hypervis_tensor(met_inv, gll_to_sphere_jacobian)
  ghost_mask = np.ones_like(mass_matrix)
  ret = {"physical_coords": reorder_wrapper(latlon),
         "physical_to_cartesian": reorder_wrapper(physical_coords_to_cartesian),
         "contra_to_physical": reorder_wrapper(gll_to_sphere_jacobian),
         "physical_to_contra": reorder_wrapper(gll_to_sphere_jacobian_inv),
         "recip_metric_determinant": reorder_wrapper(rmetdet),
         "metric_determinant": reorder_wrapper(metdet),
         "mass_matrix": reorder_wrapper(mass_matrix),
         "mass_matrix_denominator": reorder_wrapper(inv_mass_mat),
         "metric_inverse": reorder_wrapper(met_inv),
         "viscosity_tensor": reorder_wrapper(viscosity_tensor),
         "derivative_matrix": wrapper(spectrals["deriv"]),
         "gll_weights": wrapper(spectrals["gll_weights"]),
         "assembly_triple": (wrapper(assembly_triple[0]),
                             [wrapper(arr, dtype=jnp.int64) for arr in assembly_triple[1]],
                             [wrapper(arr, dtype=jnp.int64) for arr in assembly_triple[2]]),
         "hypervis_scaling": wrapper(hypervis_scaling),
         "ghost_mask": ghost_mask}
  if not wrapped:
    ret["vertex_redundancy"] = vert_redundancy_gll_flat
  metdet = ret["metric_determinant"]
  # if use_wrapper and wrapper_type == "torch":
  #   from .config import torch
  #   ret["dss_matrix"] = torch.sparse_coo_tensor((dss_triple[2], dss_triple[3]),
  #                                                dss_triple[0],
  #                                                size=(NELEM * npt * npt, NELEM * npt * npt))
  grid_dims = frozendict(N=metdet.size, shape=metdet.shape, npt=npt, num_elem=metdet.shape[0])
  return ret, grid_dims


def eval_grid_deformation_metrics(grid,
                                  npt):
  """
  Compute element-local grid deformation metrics from the metric inverse.

  Parameters
  ----------
  grid : `SpectralElementGrid`
      Spectral element grid struct containing ``metric_inverse``.
  npt : `int`
      Number of GLL points per element edge.

  Returns
  -------
  max_svd : `Array[tuple[elem_idx, gll_idx, gll_idx], Float]`
      Largest singular value of the Jacobian at each GLL node,
      corresponding to the most-compressed grid direction.
  dx_short : `Array[tuple[elem_idx, gll_idx, gll_idx], Float]`
      Shortest effective grid spacing at each GLL node (in reference units).
  dx_long : `Array[tuple[elem_idx, gll_idx, gll_idx], Float]`
      Longest effective grid spacing at each GLL node (in reference units).
  """
  eigs, _ = jnp.linalg.eigh(grid["metric_inverse"])
  max_svd = jnp.sqrt(jnp.max(eigs, axis=-1))
  min_svd = jnp.sqrt(jnp.min(eigs, axis=-1))
  dx_short = 1.0 / (max_svd * 0.5 * (npt - 1))
  dx_long = 1.0 / (min_svd * 0.5 * (npt - 1))
  return max_svd, dx_short, dx_long


def eval_global_grid_deformation_metrics(h_grid,
                                         dims):
  """
  Compute globally reduced grid deformation metrics across all MPI ranks.

  Parameters
  ----------
  h_grid : `SpectralElementGrid`
      Processor-local spectral element grid struct.
  dims : `frozendict`
      Grid dimension metadata (must contain ``"npt"``).

  Returns
  -------
  max_norm_jac_inv : `float`
      Global maximum of the largest singular value of the Jacobian inverse.
  max_min_dx : `float`
      Global maximum of the shortest effective grid spacing.
  min_max_dx : `float`
      Global minimum of the longest effective grid spacing.
  """
  L2_jac_inv, dx_short, dx_long = eval_grid_deformation_metrics(h_grid, dims["npt"])
  max_norm_jac_inv = global_max(jnp.max(L2_jac_inv))
  max_min_dx = global_max(jnp.max(dx_short))
  min_max_dx = global_min(jnp.min(dx_long))
  return max_norm_jac_inv, max_min_dx, min_max_dx


def eval_cfl(h_grid,
             radius_earth,
             diffusion_config,
             dims,
             sphere=True):
  """
  Estimate CFL-based stability limits for dynamics, tracer advection,
  and hyperviscosity time-stepping.

  Parameters
  ----------
  h_grid : `SpectralElementGrid`
      Processor-local spectral element grid struct.
  radius_earth : `float`
      Radius of the sphere (metres).  Used to non-dimensionalise
      spatial scales when ``sphere=True``.
  diffusion_config : `dict`
      Hyperviscosity configuration dict.  Recognised keys include
      ``"nu"``, ``"nu_div_factor"``, and ``"nu_d_mass"``.
  dims : `frozendict`
      Grid dimension metadata (must contain ``"npt"``).
  sphere : `bool`, default=True
      If ``True``, scale grid spacings by ``1/radius_earth``.

  Returns
  -------
  dt_limits : `dict[str, float]`
      Dictionary of stability-limited timestep estimates with keys:
      ``"dt_rkssp_euler"``, ``"dt_rk2_tracer"``, ``"dt_gravity_wave"``,
      ``"dt_hypervis_scalar"``, ``"dt_hypervis_vort"``,
      ``"dt_hypervis_div"``.
  grid_info : `dict[str, float]`
      Supporting grid metrics used in the calculation, including
      ``"max_norm_jac_inv"``, ``"max_min_dx"``, ``"min_min_dx"``,
      ``"lambda_vis"``, and ``"scale_inv"``.

  Notes
  -----
  Stability constants (``lambda_max``, ``lambda_vis``) are taken
  from Ullrich & Jablonowski (2012) as implemented in CAM-SE/HOMME.
  """
  #
  # estimate various CFL limits
  # Credit: This is basically copy-pasted from CAM-SE/HOMME

  # Courtesy of Paul Ullrich, Jared Whitehead
  lambda_maxs = {3: 1.5,
                 4: 2.74,
                 5: 4.18,
                 6: 5.86,
                 7: 7.79,
                 8: 10.0}

  lambda_viss = {3: 12.0,
                 4: 30.0,
                 5: 91.6742,
                 6: 190.117,
                 7: 374.7788,
                 8: 652.3015}

  npt = dims["npt"]
  scale_inv = 1.0 / radius_earth if sphere else 1.0

  assert npt in lambda_maxs.keys() and npt in lambda_viss.keys(), "Stability characteristics not calculated for {npt}"
  lambda_max = lambda_maxs[npt]
  lambda_vis = lambda_viss[npt]
  minimum_gauss_weight = jnp.min(h_grid["gll_weights"])

  hypervis_scaling = h_grid["hypervis_scaling"]

  max_norm_jac_inv, max_min_dx, min_min_dx = eval_global_grid_deformation_metrics(h_grid, dims)

  # tensorHV.  New eigenvalues are the eigenvalues of the tensor V
  # formulas here must match what is in cube_mod.F90
  # for tensorHV, we scale out the rearth dependency
  lam = max_norm_jac_inv**2

  norm_jac_inv_hvis_tensor = (lambda_vis**2) * (max_norm_jac_inv**4) * (lam**(-hypervis_scaling / 2.0))

  norm_jac_inv_hvis_const = (lambda_vis**2) * (1.0 / radius_earth * max_norm_jac_inv)**4
  # todo: figure out why this is wrong.
  if "tensor_hypervis" in diffusion_config.keys():
     norm_jac_inv_hvis = norm_jac_inv_hvis_tensor
  else:
     norm_jac_inv_hvis = norm_jac_inv_hvis_const

  nu_div_fact = 1.0 if "nu_div_factor" not in diffusion_config.keys() else diffusion_config["nu_div_factor"]
  nu_d_mass = 1.0 if "nu_d_mass" not in diffusion_config.keys() else diffusion_config["nu_d_mass"]
  nu = 1.0 if "nu" not in diffusion_config.keys() else diffusion_config["nu"]
  rkssp_euler_stability = minimum_gauss_weight / (120.0 * max_norm_jac_inv * scale_inv)
  rk2_tracer = 1.0 / (120.0 * max_norm_jac_inv * lambda_max * scale_inv)
  gravit_wave_stability = 1.0 / (342.0 * max_norm_jac_inv * lambda_max * scale_inv)
  hypervis_stability_dpi = 1.0 / (nu_d_mass * norm_jac_inv_hvis)
  hypervis_stability_vort = 1.0 / (nu * norm_jac_inv_hvis)
  hypervis_stability_div = 1.0 / (nu_div_fact * nu * norm_jac_inv_hvis)
  return ({"dt_rkssp_euler": rkssp_euler_stability,
           "dt_rk2_tracer": rk2_tracer,
           "dt_gravity_wave": gravit_wave_stability,
           "dt_hypervis_scalar": hypervis_stability_dpi,
           "dt_hypervis_vort": hypervis_stability_vort,
           "dt_hypervis_div": hypervis_stability_div},
          {"max_norm_jac_inv": max_norm_jac_inv,
           "max_min_dx": max_min_dx,
           "min_min_dx": min_min_dx,
           "lambda_vis": lambda_vis,
           "scale_inv": scale_inv})


def smooth_tensor(grid, dims):
  """
  Smooth the hyperviscosity tensor by projecting it onto a continuous
  bilinear representation over each element.

  The tensor is first projected onto the continuous spectral element
  space via DSS assembly, then re-sampled at each GLL node using
  bilinear interpolation from the element corners.

  Parameters
  ----------
  grid : `SpectralElementGrid`
      Spectral element grid struct whose ``"viscosity_tensor"`` field
      will be replaced with the smoothed version.
  dims : `frozendict`
      Grid dimension metadata (must contain ``"npt"``).

  Returns
  -------
  grid : `SpectralElementGrid`
      The same grid struct with an updated ``"viscosity_tensor"`` field.
  """
  npt = dims["npt"]
  spectral = init_spectral(npt)

  def project_matrix(matrix):
    upper_left = matrix[:, :, :, 0, 0]
    upper_right = matrix[:, :, :, 0, 1]
    lower_left = matrix[:, :, :, 1, 0]
    lower_right = matrix[:, :, :, 1, 1]
    upper_left_out = project_scalar(upper_left, grid, dims)
    upper_right_out = project_scalar(upper_right, grid, dims)
    lower_left_out = project_scalar(lower_left, grid, dims)
    lower_right_out = project_scalar(lower_right, grid, dims)
    left_col = jnp.stack((upper_left_out, lower_left_out), axis=-1)
    right_col = jnp.stack((upper_right_out, lower_right_out), axis=-1)
    return jnp.stack((left_col, right_col), axis=-1)

  tensor_cont = project_matrix(grid["viscosity_tensor"])
  tensor_bilinear = np.zeros_like(tensor_cont)
  for i_idx in range(npt):
    for j_idx in range(npt):
        beta = spectral["gll_points"][i_idx]
        alpha = spectral["gll_points"][j_idx]
        v0 = tensor_cont[:, 0, 0, :, :]
        v1 = tensor_cont[:, 0, npt - 1, :, :]
        v2 = tensor_cont[:, npt - 1, 0, :, :]
        v3 = tensor_cont[:, npt - 1, npt - 1, :, :]
        tensor_bilinear[:, i_idx, j_idx, :, :] = eval_bilinear_mapping(v0,
                                                                       v1,
                                                                       v2,
                                                                       v3,
                                                                       alpha,
                                                                       beta)
  grid["viscosity_tensor"] = device_wrapper(tensor_bilinear)
  return grid


def shard_grid(grid,
               dims):
  """
  Pad and shard the grid arrays across multiple JAX devices.

  Arrays whose element-axis length is not divisible by the number
  of devices are zero-padded (or NaN-padded in debug mode) before
  sharding.  A ghost mask is also added to the grid so that
  padded elements can be excluded from reductions.

  Parameters
  ----------
  grid : `SpectralElementGrid`
      Global spectral element grid struct to be sharded.
  dims : `frozendict`
      Grid dimension metadata (must contain ``"num_elem"`` and ``"npt"``).

  Returns
  -------
  grid : `SpectralElementGrid`
      Updated grid struct with all element arrays sharded across
      JAX devices and a ``"shard_extraction_map"`` and
      ``"ghost_mask"`` field added.

  Raises
  ------
  NotImplementedError
      If called in an MPI environment outside of debug mode.

  Notes
  -----
  This function is only relevant when using JAX multi-device sharding
  (``num_jax_devices > 1``).  For MPI parallelism, use
  ``make_grid_mpi_ready`` instead.
  """
  if not DEBUG and do_mpi_communication:
    raise NotImplementedError("Sharding with MPI parallelism is not tested.")

  def pad_array(arr):
    if DEBUG:
      value = jnp.nan
    else:
      value = 0.0
    if arr.shape[0] % num_jax_devices != 0:
      padded_size = (floor(arr.shape[0] / num_jax_devices) + 1) * num_jax_devices
      padding = [[0, 0] for _ in range(arr.ndim)]
      padding[0][1] = padded_size - arr.shape[0]
      arr = np.pad(arr, padding, mode="constant", constant_values=value)
    return device_wrapper(arr, elem_sharding_axis=0)

  grid["physical_coords"] = pad_array(grid["physical_coords"])
  grid["physical_to_cartesian"] = pad_array(grid["physical_to_cartesian"])
  grid["contra_to_physical"] = pad_array(grid["contra_to_physical"])
  grid["physical_to_contra"] = pad_array(grid["physical_to_contra"])
  grid["recip_metric_determinant"] = pad_array(grid["recip_metric_determinant"])
  grid["metric_determinant"] = pad_array(grid["metric_determinant"])
  grid["mass_matrix"] = pad_array(grid["mass_matrix"])
  grid["mass_matrix_denominator"] = pad_array(grid["mass_matrix_denominator"])
  grid["metric_inverse"] = pad_array(grid["metric_inverse"])
  grid["viscosity_tensor"] = pad_array(grid["viscosity_tensor"])

  nelem_fake = grid["physical_coords"].shape[0]
  extraction_map, max_dof_exchange = init_shard_extraction_map(grid["assembly_triple"],
                                                               num_jax_devices,
                                                               nelem_fake,
                                                               dims)
  grid["shard_extraction_map"] = extraction_map

  NELEM_FAKE = grid["physical_coords"].shape[0]
  ghost_mask = jnp.ones_like(grid["mass_matrix"])
  ghost_mask *= jnp.where(jnp.arange(NELEM_FAKE) < dims["num_elem"], 1.0, 0.0)[:, jnp.newaxis, jnp.newaxis]
  grid["ghost_mask"] = ghost_mask

  return grid


def get_global_grid(grid, dims):
  """
  Gather all sharded element arrays back into a single global grid.

  Parameters
  ----------
  grid : `SpectralElementGrid`
      A sharded or device-wrapped spectral element grid struct.
  dims : `frozendict`
      Grid dimension metadata (must contain ``"num_elem"``).
      Used to strip padding from sharded arrays.

  Returns
  -------
  global_grid : `dict[str, Array]`
      A grid struct where all element arrays have been moved to host
      memory and trimmed to ``dims["num_elem"]`` elements.
      Non-element fields are copied through unchanged.
  """
  global_grid = {}
  global_grid["physical_coords"] = get_global_array(grid["physical_coords"], dims)
  global_grid["physical_to_cartesian"] = get_global_array(grid["physical_to_cartesian"], dims)
  global_grid["contra_to_physical"] = get_global_array(grid["contra_to_physical"], dims)
  global_grid["physical_to_contra"] = get_global_array(grid["physical_to_contra"], dims)
  global_grid["recip_metric_determinant"] = get_global_array(grid["recip_metric_determinant"], dims)
  global_grid["metric_determinant"] = get_global_array(grid["metric_determinant"], dims)
  global_grid["mass_matrix"] = get_global_array(grid["mass_matrix"], dims)
  global_grid["mass_matrix_denominator"] = get_global_array(grid["mass_matrix_denominator"], dims)
  global_grid["metric_inverse"] = get_global_array(grid["metric_inverse"], dims)
  global_grid["viscosity_tensor"] = get_global_array(grid["viscosity_tensor"], dims)
  global_grid["ghost_mask"] = get_global_array(grid["ghost_mask"], dims)
  for field in grid.keys():
    if field not in global_grid.keys():
      global_grid[field] = grid[field]
  return global_grid


def extract_subset_parallel_dim(var, proc_idx, decomp):
  """
  Extract the processor-local slice of an array along its first (element) axis.

  Parameters
  ----------
  var : `Array[tuple[elem_idx, ...], Any]`
      Global array to slice.
  proc_idx : `int`
      MPI rank index of the target processor.
  decomp : sequence of tuple[int, int]
      Element-range decomposition returned by ``init_decomp``.
      ``decomp[proc_idx]`` gives the ``(start, end)`` indices
      of the slice assigned to processor ``proc_idx``.

  Returns
  -------
  `Array[tuple[local_elem_idx, ...], Any]`
      The sub-array belonging to processor ``proc_idx``.
  """
  slices = [slice(None, None) for _ in range(var.ndim)]
  slices[0] = slice(decomp[proc_idx][0], decomp[proc_idx][1])
  return var[*slices]


def make_grid_mpi_ready(grid, dims, proc_idx, decomp=None, wrapped=use_wrapper):
  """
  Partition a global grid into a processor-local grid for MPI parallelism.

  Splits the global element arrays along the element axis according to
  a 1-D domain decomposition, and partitions the DSS assembly triples
  into local, send, and receive components for halo exchanges.

  Parameters
  ----------
  grid : `SpectralElementGrid`
      Global spectral element grid struct (assembled on rank 0 or
      broadcast to all ranks).
  dims : `frozendict`
      Global grid dimension metadata.
  proc_idx : `int`
      MPI rank of the calling processor.
  decomp : sequence of tuple[int, int]`, optional
      Element-range decomposition.  If ``None``, a uniform
      decomposition over ``mpi_size`` ranks is computed via
      ``init_decomp``.
  wrapped : `bool`, default=use_wrapper
      If ``True``, local arrays are moved onto the configured device.

  Returns
  -------
  local_grid : `dict[str, Array]`
      Processor-local grid struct containing sliced element arrays,
      local/send/receive DSS assembly triples, and ghost masks.
  local_dims : `frozendict`
      Updated dimension metadata for the local grid, including
      ``"num_elem"`` set to the local element count and per-rank
      send buffer sizes.

  Raises
  ------
  NotImplementedError
      If called with both MPI communication and multi-device
      JAX sharding enabled outside of debug mode.
  """
  if not DEBUG and num_jax_devices > 1:
    raise NotImplementedError("Sharding with MPI parallelism is not tested.")

  if decomp is None:
    decomp = init_decomp(dims["num_elem"], mpi_size)

  if wrapped:
    wrapper = device_wrapper
  else:
    def wrapper(x, dtype=None):
      return x

  vert_red_local, vert_red_send, vert_red_recv = triage_vert_redundancy_flat(grid["assembly_triple"],
                                                                             proc_idx,
                                                                             decomp)
  triples_send, triples_recv = init_assembly_global(vert_red_send, vert_red_recv)
  triples_local = init_assembly_local(vert_red_local)
  for proc_idx_recv in triples_recv.keys():
    triples_recv[proc_idx_recv] = (wrapper(triples_recv[proc_idx_recv][0]),
                                   [wrapper(arr, dtype=jnp.int64) for arr in triples_recv[proc_idx_recv][1]],
                                   [wrapper(arr, dtype=jnp.int64) for arr in triples_recv[proc_idx_recv][2]])

  for proc_idx_send in triples_send.keys():
    triples_send[proc_idx_send] = (wrapper(triples_send[proc_idx_send][0]),
                                   [wrapper(arr, dtype=jnp.int64) for arr in triples_send[proc_idx_send][1]],
                                   [wrapper(arr, dtype=jnp.int64) for arr in triples_send[proc_idx_send][2]])

  local_grid = {}
  local_grid["physical_coords"] = extract_subset_parallel_dim(grid["physical_coords"],
                                                              proc_idx,
                                                              decomp)
  local_grid["physical_to_cartesian"] = extract_subset_parallel_dim(grid["physical_to_cartesian"],
                                                                    proc_idx,
                                                                    decomp)
  local_grid["contra_to_physical"] = extract_subset_parallel_dim(grid["contra_to_physical"],
                                                                 proc_idx,
                                                                 decomp)
  local_grid["physical_to_contra"] = extract_subset_parallel_dim(grid["physical_to_contra"],
                                                                 proc_idx,
                                                                 decomp)
  local_grid["recip_metric_determinant"] = extract_subset_parallel_dim(grid["recip_metric_determinant"],
                                                                       proc_idx,
                                                                       decomp)
  local_grid["metric_determinant"] = extract_subset_parallel_dim(grid["metric_determinant"],
                                                                 proc_idx,
                                                                 decomp)
  local_grid["mass_matrix"] = extract_subset_parallel_dim(grid["mass_matrix"],
                                                          proc_idx,
                                                          decomp)
  local_grid["mass_matrix_denominator"] = extract_subset_parallel_dim(grid["mass_matrix_denominator"],
                                                                      proc_idx,
                                                                      decomp)
  local_grid["metric_inverse"] = extract_subset_parallel_dim(grid["metric_inverse"],
                                                             proc_idx,
                                                             decomp)
  local_grid["viscosity_tensor"] = extract_subset_parallel_dim(grid["viscosity_tensor"],
                                                               proc_idx,
                                                               decomp)
  local_grid["ghost_mask"] = extract_subset_parallel_dim(grid["ghost_mask"],
                                                         proc_idx,
                                                         decomp)
  local_grid["triples_send"] = triples_send
  local_grid["triples_receive"] = triples_recv
  local_grid["assembly_triple"] = triples_local
  if not wrapped:
    local_grid["vertex_redundancy"] = vert_red_local
    local_grid["vertex_redundancy_send"] = vert_red_send
    local_grid["vertex_redundancy_receive"] = vert_red_recv

  for field in grid.keys():
    if field not in local_grid.keys():
      local_grid[field] = grid[field]

  send_dims = {}
  for proc_idx in triples_send.keys():
    send_dims[str(proc_idx)] = triples_send[proc_idx][0].size

  local_dims = {}
  for key in dims.keys():
    local_dims[key] = dims[key]
  local_dims["num_elem"] = local_grid["metric_determinant"].shape[0]
  for key in send_dims.keys():
    local_dims[key] = send_dims[key]
  return local_grid, local_dims
