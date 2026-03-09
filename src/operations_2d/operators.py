import numpy as np
from .._config import get_backend as _get_backend
_be = _get_backend()
jnp = _be.np
jit = _be.jit
flip = _be.flip
from functools import partial


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
  df_da = jnp.einsum("fij,ki->fkj", f, grid["derivative_matrix"])
  df_db = jnp.einsum("fij,kj->fik", f, grid["derivative_matrix"])
  df_dab = jnp.stack((df_da, df_db), axis=-1)
  return 1.0 / a * flip(jnp.einsum("fijg,fijgs->fijs", df_dab, grid["physical_to_contra"]), -1)


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
  du_da = jnp.einsum("fij,ki->fkj", u_contra[:, :, :, 0], grid["derivative_matrix"])
  du_db = jnp.einsum("fij,kj->fik", u_contra[:, :, :, 1], grid["derivative_matrix"])
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
  dv_da = jnp.einsum("fij,ki->fkj", u_cov[:, :, :, 1], grid["derivative_matrix"])
  du_db = jnp.einsum("fij,kj->fik", u_cov[:, :, :, 0], grid["derivative_matrix"])
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
    grad = jnp.einsum("fijs,fijts->fijt", grad, grid["viscosity_tensor"]) * a**4
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
  ds_contra_term_1 = - jnp.einsum("j,n,fmn,fmn,fjn,jm->fmn",
                                  gll_weights,
                                  gll_weights,
                                  met_inv[:, :, :, 0, 0],
                                  met_det, s, deriv)
  ds_contra_term_2 = - jnp.einsum("m,j,fmn,fmn,fmj,jn->fmn",
                                  gll_weights,
                                  gll_weights,
                                  met_inv[:, :, :, 1, 0],
                                  met_det, s, deriv)
  ds_contra_term_3 = - jnp.einsum("j,n,fmn,fmn,fjn,jm->fmn",
                                  gll_weights,
                                  gll_weights,
                                  met_inv[:, :, :, 0, 1],
                                  met_det, s, deriv)
  ds_contra_term_4 = - jnp.einsum("m,j,fmn,fmn,fmj,jn->fmn",
                                  gll_weights,
                                  gll_weights,
                                  met_inv[:, :, :, 1, 1],
                                  met_det, s, deriv)
  ds_contra = jnp.stack((ds_contra_term_1 + ds_contra_term_2,
                         ds_contra_term_3 + ds_contra_term_4), axis=-1)
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
  ds_contra = jnp.stack((jnp.einsum("m,j,fmj,jn->fmn", gll_weights, gll_weights, s, deriv),
                         -jnp.einsum("j,n,fjn,jm->fmn", gll_weights, gll_weights, s, deriv)), axis=-1)
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
  du_da_wk = - jnp.einsum("n,j,fjn,fjn,jm->fmn", gll_weights, gll_weights, met_det, contra[:, :, :, 0], deriv)
  du_db_wk = - jnp.einsum("m,j,fmj,fmj,jn->fmn", gll_weights, gll_weights, met_det, contra[:, :, :, 1], deriv)
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
  return flip(jnp.einsum("fijg,fijsg->fijs", u, grid["contra_to_physical"]), -1)


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
  return jnp.einsum("fijs,fijgs->fijg", flip(u, -1), grid["physical_to_contra"])


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
  return jnp.einsum("fijs,fijsg->fijg", flip(u, -1), grid["contra_to_physical"])


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
