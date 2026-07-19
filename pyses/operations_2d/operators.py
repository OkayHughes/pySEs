import numpy as np
from .._config import get_backend as _get_backend
from functools import partial
_be = _get_backend()
jnp = _be.np
jit = _be.jit


def pointwise_matvec(matrix, vec, transpose_matrix=False):
  """
  Apply a per-point (2x2-sized) matrix to a vector field, unrolled.

  Computes ``out[..., g] = sum_s matrix[..., g, s] * vec[..., s]`` (or the
  transposed contraction ``sum_s matrix[..., s, g] * vec[..., s]`` when
  ``transpose_matrix`` is set) as a broadcast-multiply-sum instead of
  ``einsum``.

  Parameters
  ----------
  matrix : `Array[tuple[..., g, s], Float]`
      Per-point matrix field (e.g. metric or viscosity tensor).  Leading axes
      broadcast against ``vec``, so an axis of length 1 may be inserted to
      apply one matrix across an extra ``vec`` axis (e.g. vertical levels).
  vec : `Array[tuple[..., s], Float]`
      Vector field whose trailing axis is contracted.
  transpose_matrix : `bool`, default=False
      Contract against the second-to-last ``matrix`` axis instead of the last.

  Returns
  -------
  out : `Array[tuple[..., g], Float]`
      The transformed vector field.

  Notes
  -----
  On GPU, XLA rewrites the equivalent ``einsum`` (a batched contraction over a
  length-2 axis) into one cuBLAS GEMM launch per grid point, which runs at a
  few percent of memory bandwidth; this form stays inside elementwise fusions
  on every backend.
  """
  if transpose_matrix:
    return jnp.sum(matrix * vec[..., :, None], axis=-2)
  return jnp.sum(matrix * vec[..., None, :], axis=-1)


def gll_matvec(mat, x, axis):
  """
  Contract one GLL axis of a nodal field with an (npt x npt) operator matrix.

  For a field whose trailing two axes are the intra-element GLL indices
  ``(i, j)``, computes

  * ``axis=-2``: ``out[..., k, j] = sum_i mat[k, i] * x[..., i, j]``
  * ``axis=-1``: ``out[..., i, k] = sum_j mat[k, j] * x[..., i, j]``

  as a broadcast-multiply-sum.  Any leading axes (elements, vertical levels,
  vmap batches) broadcast through unchanged.

  Parameters
  ----------
  mat : `Array[tuple[npt, npt], Float]`
      Operator matrix (e.g. the GLL derivative matrix, or a weight-scaled
      variant for weak-form contractions).
  x : `Array[tuple[..., npt, npt], Float]`
      Nodal field; the last two axes are the GLL tensor-product indices.
  axis : `int`
      Which GLL axis to contract: ``-2`` (first index) or ``-1`` (second).

  Returns
  -------
  out : `Array[tuple[..., npt, npt], Float]`
      The contracted field, same shape as ``x``.

  Notes
  -----
  On GPU, XLA rewrites the equivalent ``einsum`` into a cuBLAS GEMM with a
  4-wide inner dimension, forcing layout-shuffling transposes/concatenates
  around each call and running well below memory bandwidth; this form stays
  inside elementwise fusions on every backend (see ``pointwise_matvec``).
  """
  if axis == -2:
    return jnp.sum(mat[:, :, None] * x[..., None, :, :], axis=-2)
  if axis == -1:
    return jnp.sum(mat * x[..., None, :], axis=-1)
  raise ValueError("axis must be -1 or -2")


@jit
def horizontal_gradient(f,
                        grid,
                        a=1.0):
  """
  Calculate the element-local gradient of f in spherical coordinates.

  Parameters
  ----------
  f : `Array[tuple[elem_idx, gll_idx, gll_idx], Float]`
      The scalar field to calulate the gradient of
  grid : `SpectralElementGrid`
      Spectral element grid struct that contains coordinate and metric data.
  a : `float`, default=1.0
      Radius of sphere on which gradient is calculated.

  Notes
  -----
  One typically uses `se_grid.create_spectral_element_grid` to create
  the `grid` argument.

  Returns
  -------
  grad_f: `Array[tuple[elem_idx, gll_idx, gll_idx, lon_lat], Float]`
      The spherical gradient of f
  """
  df_da = gll_matvec(grid["derivative_matrix"], f, axis=-2)
  df_db = gll_matvec(grid["derivative_matrix"], f, axis=-1)
  df_dab = jnp.stack((df_da, df_db), axis=-1)
  return 1.0 / a * jnp.flip(pointwise_matvec(grid["physical_to_contra"], df_dab,
                                             transpose_matrix=True), -1)


@jit
def horizontal_divergence(u,
                          grid,
                          a=1.0):
  """
  Calculate the element-local spherical divergence of a physical vector.

  Parameters
  ----------
  u : `Array[tuple[elem_idx, gll_idx, gll_idx, lon_lat], Float]`
      Vector field (u, v) in spherical coordinates
      to apply divergence operator to
  grid : `SpectralElementGrid`
      Spectral element grid struct that contains coordinate and metric data.
  a : `float`, default=1.0
      Radius of sphere on which divergence is calculated.

  Returns
  -------
  div_u : `Array[tuple[elem_idx, gll_idx, gll_idx], Float]`
      Spherical divergence of `u`

  Notes
  -----
  One typically uses `se_grid.create_spectral_element_grid` to create
  the `grid` argument.
  """
  u_contra = 1.0 / a * grid["metric_determinant"][:, :, :, np.newaxis] * physical_to_contravariant(u, grid)
  du_da = gll_matvec(grid["derivative_matrix"], u_contra[..., 0], axis=-2)
  du_db = gll_matvec(grid["derivative_matrix"], u_contra[..., 1], axis=-1)
  div = grid["recip_metric_determinant"][:, :, :] * (du_da + du_db)
  return div


@jit
def horizontal_vorticity(u,
                         grid,
                         a=1.0):
  """
  Calculate the element-local spherical vorticity of a physical vector.

  Parameters
  ----------
  u : `Array[tuple[elem_idx, gll_idx, gll_idx, lon_lat], Float]`
      Vector field (u, v) in spherical coordinates
      to calculate vorticity of
  grid : `SpectralElementGrid`
      Spectral element grid struct that contains coordinate and metric data.
  a : `float`, default=1.0
      Radius of sphere on which vorticity is calculated.

  Returns
  -------
  vort_u : `Array[tuple[elem_idx, gll_idx, gll_idx], Float]`
    Spherical vorticity of `u`

  Notes
  -----
  One typically uses `se_grid.create_spectral_element_grid` to create
  the `grid` argument.
  """
  u_cov = physical_to_covariant(u, grid)
  dv_da = gll_matvec(grid["derivative_matrix"], u_cov[..., 1], axis=-2)
  du_db = gll_matvec(grid["derivative_matrix"], u_cov[..., 0], axis=-1)
  vort = 1.0 / a * grid["recip_metric_determinant"][:, :, :] * (du_db - dv_da)
  return vort


@jit
def horizontal_laplacian(f,
                         grid,
                         a=1.0):
  """
  Calculate the element-local spherical laplacian of f.

  Parameters
  ----------
  f : `Array[tuple[elem_idx, gll_idx, gll_idx], Float]`
      Scalar field to which to apply the laplacian operator
  grid : `SpectralElementGrid`
      Spectral element grid struct that contains coordinate and metric data.
  a : `float`, default=1.0
      Radius of sphere on which the laplacian is calculated.

  Returns
  -------
  laplace_f : `Array[tuple[elem_idx, gll_idx, gll_idx], Float]`
    Spherical laplacian of `f`

  Notes
  -----
  One typically uses `se_grid.create_spectral_element_grid` to create
  the `grid` argument.
  """
  grad = horizontal_gradient(f, grid, a=a)
  return horizontal_divergence(grad, grid, a=a)


@partial(jit, static_argnames=["apply_tensor"])
def horizontal_weak_laplacian(f,
                              grid,
                              a=1.0,
                              apply_tensor=False):
  """
  Calculate the element-local weak spherical laplacian of f.

  Use this function for hyperviscosity.

  Parameters
  ----------
  f : `Array[tuple[elem_idx, gll_idx, gll_idx], Float]`
      Scalar field to which to apply the weak laplacian operator
  grid : `SpectralElementGrid`
      Spectral element grid struct that contains coordinate and metric data.
  a : `float`, default=1.0
      Radius of sphere on which the weak laplacian is calculated.

  Returns
  -------
  wk_laplace_f : `Array[tuple[elem_idx, gll_idx, gll_idx], Float]`
    Weak spherical laplacian of `f`

  Notes
  -----
  When performing assembly, this is already scaled by mass matrix quantities
  due to how quadrature is computed in SE.

  [TODO] Explain how the math works

  One typically uses `se_grid.create_spectral_element_grid` to create
  the `grid` argument.
  """
  grad = horizontal_gradient(f, grid, a=a)
  if apply_tensor:
    grad = pointwise_matvec(grid["viscosity_tensor"], grad) * a**4
  lap_unscaled = horizontal_weak_divergence(grad, grid, a=a)
  lap_unscaled /= grid["mass_matrix"]
  return lap_unscaled


@jit
def horizontal_weak_gradient_covariant(s,
                                       grid,
                                       a=1.0):
  """
  Calculate the element-local weak gradient of s in spherical coordinates
  using covariant test functions.

  Parameters
  ----------
  s : `Array[tuple[elem_idx, gll_idx, gll_idx], Float]
      The scalar field to calulate the gradient of
  grid : `SpectralElementGrid`
      Spectral element grid struct that contains coordinate and metric data.
  a : `float`, default=1.0
      Radius of sphere on which weak gradient is calculated.

  Notes
  -----
  [TODO] Explain what's going on in the math here
  One typically uses `se_grid.create_spectral_element_grid` to create
  the `grid` argument.

  Returns
  -------
  wk_grad_s: Array[tuple[elem_idx, gll_idx, gll_idx, lon_lat], Float]
      The weak spherical gradient of s.
  """
  gll_weights = grid["gll_weights"]
  deriv = grid["derivative_matrix"]
  met_inv = grid["metric_inverse"]
  met_det = grid["metric_determinant"]
  # The four original 6-operand einsums reduce to two derivative contractions:
  # terms (1, 3) share D^T s ("j,fjn,jm->fmn") and terms (2, 4) share s D
  # ("j,fmj,jn->fmn").  The remaining per-node factors (the second GLL weight
  # and the metric) are j-independent, so apply them as elementwise multiplies
  # -- halving the FLOPs of the weak gradient, the inner kernel of all
  # hyperviscosity.
  # Fold the quadrature weight into the operator matrix once (npt x npt):
  # weighted_deriv_t[m, j] = gll_weights[j] * deriv[j, m].
  weighted_deriv_t = (gll_weights[:, None] * deriv).T
  dt_s = gll_matvec(weighted_deriv_t, s, axis=-2)
  s_d = gll_matvec(weighted_deriv_t, s, axis=-1)
  term_a = gll_weights[jnp.newaxis, jnp.newaxis, :] * dt_s
  term_b = gll_weights[jnp.newaxis, :, jnp.newaxis] * s_d
  ds_contra_0 = -met_det * (met_inv[:, :, :, 0, 0] * term_a + met_inv[:, :, :, 1, 0] * term_b)
  ds_contra_1 = -met_det * (met_inv[:, :, :, 0, 1] * term_a + met_inv[:, :, :, 1, 1] * term_b)
  ds_contra = jnp.stack((ds_contra_0, ds_contra_1), axis=-1)
  return 1.0 / a * contravariant_to_physical(ds_contra, grid)


@jit
def horizontal_weak_curl_covariant(s,
                                   grid,
                                   a=1.0):
  """
  Calculates weak horizontal spherical curl of the vector s𝐤 using covariant test functions.

  Parameters
  ----------
  s : `Array[tuple[elem_idx, gll_idx, gll_idx], Float]
      The scalar field to use for horizontal curl.
  grid : `SpectralElementGrid`
      Spectral element grid struct that contains coordinate and metric data.
  a : `float`, default=1.0
      Radius of sphere on which weak gradient is calculated.

  Notes
  -----
  [TODO] Explain what's going on in the math here
  One typically uses `se_grid.create_spectral_element_grid` to create
  the `grid` argument.

  Returns
  -------
  wk_curl_s: `Array[tuple[elem_idx, gll_idx, gll_idx, lon_lat], Float]`
      The weak spherical horizontal curl of s.
  """
  gll_weights = grid["gll_weights"]
  deriv = grid["derivative_matrix"]
  weighted_deriv_t = (gll_weights[:, None] * deriv).T
  ds_contra = jnp.stack(
      (gll_weights[:, None] * gll_matvec(weighted_deriv_t, s, axis=-1),
       -(gll_weights * gll_matvec(weighted_deriv_t, s, axis=-2))), axis=-1)
  return 1.0 / a * contravariant_to_physical(ds_contra, grid)


@partial(jit, static_argnames=["damp"])
def horizontal_weak_vector_laplacian(u,
                                     grid,
                                     a=1.0,
                                     nu_div_fact=1.0,
                                     damp=False):
  """
  Calculate the element-local weak spherical vector laplacian of a physical vector field u.

  Use this function for hyperviscosity.

  Parameters
  ----------
  u : `Array[tuple[elem_idx, gll_idx, gll_idx, lon_lat], Float]`
      Scalar field to which to apply the weak laplacian operator
  grid : `SpectralElementGrid`
      Spectral element grid struct that contains coordinate and metric data.
  a : `float`, default=1.0
      Radius of sphere on which the weak vector laplacian is calculated.

  Returns
  -------
  laplace_u : `Array[tuple[elem_idx, gll_idx, gll_idx, lon_lat], Float]`
    Weak spherical vector laplacian of `u`

  Notes
  -----
  When performing assembly, this is already scaled by mass matrix quantities
  due to how quadrature is computed in SE.

  [TODO] Explain how the math works

  One typically uses `se_grid.create_spectral_element_grid` to create
  the `grid` argument.
  """
  div = horizontal_divergence(u, grid, a=a) * nu_div_fact
  vor = horizontal_vorticity(u, grid, a=a)
  laplacian = horizontal_weak_gradient_covariant(div, grid, a=a) - horizontal_weak_curl_covariant(vor, grid, a=a)
  gll_weights = grid["gll_weights"]
  if damp:
    out = laplacian + jnp.stack((2 * (gll_weights[np.newaxis, :, np.newaxis] *
                                      gll_weights[np.newaxis, np.newaxis, :] *
                                      grid["metric_determinant"] * u[:, :, :, 0] * (1 / a)**2),
                                     (gll_weights[np.newaxis, :, np.newaxis] *
                                      gll_weights[np.newaxis, np.newaxis, :] *
                                      grid["metric_determinant"] * u[:, :, :, 1] * (1 / a)**2)), axis=-1)
  else:
    out = laplacian
  out /= grid["mass_matrix"][:, :, :, np.newaxis]
  return out


@jit
def horizontal_weak_divergence(u,
                               grid,
                               a=1.0):
  """
  Calculates weak spherical horizontal divergence of the vector u, given in spherical coordinates.

  Parameters
  ----------
  u : `Array[tuple[elem_idx, gll_idx, gll_idx, lon_lat], Float]`
      The vector field to apply divergence to.
  grid : `SpectralElementGrid`
      Spectral element grid struct that contains coordinate and metric data.
  a : `float`, default=1.0
      Radius of sphere on which weak gradient is calculated.

  Notes
  -----
  [TODO] Explain what's going on in the math here
  One typically uses `se_grid.create_spectral_element_grid` to create
  the `grid` argument.

  Returns
  -------
  wk_div_u: `Array[tuple[elem_idx, gll_idx, gll_idx], Float]`
      The weak spherical horizontal divergence of s.
  """
  contra = physical_to_contravariant(u, grid)
  gll_weights = grid["gll_weights"]
  met_det = grid["metric_determinant"]
  deriv = grid["derivative_matrix"]
  weighted_deriv_t = (gll_weights[:, None] * deriv).T
  du_da_wk = - (gll_weights * gll_matvec(weighted_deriv_t, met_det * contra[..., 0], axis=-2))
  du_db_wk = - (gll_weights[:, None] * gll_matvec(weighted_deriv_t, met_det * contra[..., 1], axis=-1))
  return 1.0 / a * (du_da_wk + du_db_wk)


@jit
def contravariant_to_physical(u,
                              grid):
  """
  Convert a vector given in contravariant coordinates on the local
  reference element to physical coordinates.

  Parameters
  ----------
  u : `Array[tuple[elem_idx, gll_idx, gll_idx, alpha_beta_super], Float]`
      The vector field in contravariant coordinates to map to physical coordinates
  grid : `SpectralElementGrid`
      Spectral element grid struct that contains coordinate and metric data.

  Returns
  -------
  u_physical : `Array[tuple[elem_idx, gll_idx, gll_idx, lon_lat], Float]`
      The vector in physical coordinates

  Notes
  -----
  One typically uses `se_grid.create_spectral_element_grid` to create
  the `grid` argument.
  """
  return jnp.flip(pointwise_matvec(grid["contra_to_physical"], u), -1)


@jit
def physical_to_contravariant(u,
                              grid):
  """
  Convert a vector given in physical coordinates to contravariant
  coordinates on the reference domain.

  Parameters
  ----------
  u : `Array[tuple[elem_idx, gll_idx, gll_idx, lon_lat], Float]`
      The vector field in physical coordinates to map to contravariant coordinates
  grid : `SpectralElementGrid`
      Spectral element grid struct that contains coordinate and metric data.

  Returns
  -------
  u_contra : `Array[tuple[elem_idx, gll_idx, gll_idx, alpha_beta_super], Float]
      The vector in physical coordinates

  Notes
  -----
  One typically uses `se_grid.create_spectral_element_grid` to create
  the `grid` argument.
  """
  return pointwise_matvec(grid["physical_to_contra"], jnp.flip(u, -1))


@jit
def physical_to_covariant(u,
                          grid):
  """
  Convert a vector given in physical coordinates to covariant
  coordinates on the reference domain.

  Parameters
  ----------
  u : `Array[tuple[elem_idx, gll_idx, gll_idx, lon_lat], Float]`
      The vector field in physical coordinates to map to covariant coordinates
  grid : `SpectralElementGrid`
      Spectral element grid struct that contains coordinate and metric data.

  Returns
  -------
  u_contra : `Array[tuple[elem_idx, gll_idx, gll_idx, alpha_beta_sub], Float]
      The vector in physical coordinates

  Notes
  -----
  One typically uses `se_grid.create_spectral_element_grid` to create
  the `grid` argument.
  """
  return pointwise_matvec(grid["contra_to_physical"], jnp.flip(u, -1),
                          transpose_matrix=True)


@jit
def inner_product(f,
                  g,
                  grid):
  """
  Calculate the Spectral Element discrete (processor-local) inner product of
  two scalars.

  Parameters
  ----------
  f: `Array[tuple[elem_idx, gll_idx, gll_idx], Float]`
      The first argument of the inner product
  g: `Array[tuple[elem_idx, gll_idx, gll_idx], Float]`
      The second argument of the inner product
  grid : `SpectralElementGrid`
      Spectral element grid struct that contains coordinate and metric data.

  Returns
  -------
  Float
      Inner product over elements contained in `grid`.

  Notes
  -----
  * By inner product, we mean the inner product of functions induced by global quadrature, namely
 〈f, g〉 = ∫f, g dA for real functions.
  * To calculate the inner product with distributed memory parallelism (e.g., MPI),
  simply call multiprocessing.global_sum on the result of `inner_product`.
  * To calculate the inner product of two vectors in physical coordinates, use
  inner_prod(u0[..., 0], u1[..., 0], grid) + inner_prod(u0[..., 1], u1[..., 1]).
  * The induced norm is simply `jnp.sqrt(inner_prod(f, f, grid))` (unless using distributed memory).
  One typically uses `se_grid.create_spectral_element_grid` to create
  the `grid` argument.
  """
  integrand = f * g * (grid["metric_determinant"] *
                       grid["gll_weights"][np.newaxis, :, np.newaxis] *
                       grid["gll_weights"][np.newaxis, np.newaxis, :])
  masked_integrand = jnp.where(grid["ghost_mask"] > 0.5, integrand, 0.0)
  return jnp.sum(masked_integrand)
