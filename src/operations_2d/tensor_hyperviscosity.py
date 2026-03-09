import numpy as np


def eval_quasi_uniform_hypervisc_coeff(ne,
                                       radius_earth=1.0):
  """
  Estimate the scalar hyperviscosity coefficient for a quasi-uniform grid.

  Scales from the reference NE30 full-planet coefficient of ``1e15``
  to the requested grid resolution and planet size using a logarithmic
  resolution scaling that accounts for the clustering of quadrature points
  near cubed-sphere corners.

  Parameters
  ----------
  ne : int
      Number of elements along one edge of a cubed-sphere face.
  radius_earth : float, optional
      Planet radius (m).  Defaults to ``1.0`` (unit sphere).

  Returns
  -------
  nu_base : float
      Scalar hyperviscosity coefficient (m^4 s^-1 for a full-radius planet).
  """
  ne_30_full_radius_coeff = 1e15
  small_planet_correction_factor = radius_earth / 6371e3
  # note: this power accounts for scrunched elements at corner points
  uniform_res_hypervis_scaling = 1.0 / np.log10(2.0)
  nu_base = ne_30_full_radius_coeff * small_planet_correction_factor * (30.0 / ne)**uniform_res_hypervis_scaling
  return nu_base


def eval_variable_resolution_hypervisc_coeff(smallest_gridpoint_dx,
                                             hypervis_scaling, npt,
                                             radius_earth=1.0):
  """
  Estimate the tensor hyperviscosity coefficient for a variable-resolution grid.

  Computes the minimum-resolution hyperviscosity coefficient by scaling the
  NE30 reference coefficient to the finest grid spacing, then applies the
  resolution-dependent scaling exponent ``hypervis_scaling`` to derive the
  tensor coefficient ``nu_tensor``.

  Parameters
  ----------
  smallest_gridpoint_dx : float
      Shortest GLL grid spacing in the mesh (m, on the unit sphere before
      scaling by ``radius_earth``).
  hypervis_scaling : float
      Exponent controlling how strongly hyperviscosity is reduced on
      fine-resolution regions (typically 3.2).
  npt : int
      Number of GLL points per element edge.
  radius_earth : float, optional
      Planet radius (m).  Defaults to ``1.0`` (unit sphere).

  Returns
  -------
  nu_tensor : float
      Tensor hyperviscosity coefficient scaled by ``radius_earth^{-4}``.
  """
  smallest_gridpoint_dx *= radius_earth
  ne30_elem_length = 110000.0
  small_planet_correction_factor = radius_earth / 6371e3
  uniform_res_hypervis_scaling = 1.0 / np.log10(2.0)
  ne_30_full_radius_coeff = 1e15 * small_planet_correction_factor
  ne_30_full_radius_coeff *= (smallest_gridpoint_dx / ne30_elem_length)**uniform_res_hypervis_scaling
  nu_min = ne_30_full_radius_coeff * (2.0 * radius_earth / ((npt - 1.0) * smallest_gridpoint_dx))**(hypervis_scaling)
  nu_tensor = nu_min / (radius_earth**4)
  return nu_tensor


def eval_hypervis_tensor(met_inv,
                         jacobian,
                         hypervis_scaling=3.2):
  """
  Initialize the metric tensor used to encode anisotropic resolution-dependent hyperviscosity
  for unstructured grids.

  Parameters
  ----------
  met_inv : `Array[tuple[elem_idx, gll_idx, gll_idx, alpha_beta_idx, alpha_beta_idx], Float]`
      The inverse metric tensor jacobian_inv*transpose(jacobian_inv)
  grid : `SpectralElementGrid`
      Spectral element grid struct that contains coordinate and metric data.
  jacobian : `Array[tuple[elem_idx, gll_idx, gll_idx, alpha_beta_idx, alpha_beta_idx], Float]`
      The jacobian matrix that takes covariant/contravariant vectors on the
      reference element to spherical coordinates.
  hypervis_scaling: `Float`, default=3.2
      Power to use in resolution-dependent hyperviscosity

  Notes
  -----
  * We assume here that tensor hyperviscosity is applied as
  ∇·(V ∇(∆^{2(lap_ord-1)} f). For example, for default fourth-order hyperviscosity
  `lap_ord = 2`, hyperviscosity is calculated as ∇·V ∇A[∇·∇f], where A is the
  spectral element projection operator, e.g., assembly.project_scalar.
  * The viscosity tensor can be understood in the following way. We will focus on
  fourth order hyperviscosity for simplicity. Recall that the point of hyperviscosity
  is to heavily damp unresolvable grid-scale flow features
  without artificially damping resolved flow.
  On a quasi-uniform grid, hyperviscosity can be applied as ν_const ∆^2 f (note that
  this laplacian is applied on the unit sphere).
  Using non-uniform grids introduces two complications. Firstly,
  some grid cells should have much smaller area than others,
  because this is the point of using variable-resolution modeling.
  This might motivate using a gridpoint-dependent ν, such as
  calculating the ratio of the area of the element that contains it to
  the average grid cell area of an NE30 grid. However, grid cells can also
  be quite distorted (especially in regions where the grid is transitioning from
  a coarser to a finer grid). In a grid cell that is twice as tall as it is wide
  (imagine we are woring on an x, y tangent plane)
  a 4∆y feature may be well resolved in the x direction, but poorly resolved due to the
  increased distance between quadrature points in the y direction. The
  standard spherical Laplacian ∇·∇f is constructed so that a grid with distorted elements
  will still converge to correct solution as grid resolution increases. This motivates the
  construction of a metric tensor V (recall: this is a positive definite matrix
  that induces a modified notion of, e.g., the length of a vector) that allows us to
  damp small-scale features in a way that re-introduces grid distortion.
  * To do this, we do an eigendomposition of the the (symmetric) inverse metric M= J^{-1}J^{-T}
  into M = E Λ E^T (recall, E are unit eigenvectors and M is symmetric, so E, E^T are inverses of each other),
  where Λ = diag((4/((npt-1)∆x)^2, 4/((npt-1)∆y)^2). Note that 4 is the area of the reference element [-1, 1]^2.
  Because most unstructured grids bilinearly map the reference element to the sphere
  in a way that is approximately bilinear on > nx=15 grids, ∆x and ∆y are
  essentially the cartesian length between grid points,
  and (npt-1)∆x, (npt-1)∆y are ∆x_elem, ∆y_elem, respectively. We then define
  Λ^* = diag(1/λ_0^(-hv_scaling/2.0), 1/λ_1^(-hv_scaling/2.0)), and it turns out that
  V = J E Λ Λ^* E^T J^T satisfies the properties we're looking for. In practice,
  the result of hyperviscosity must be scaled by radius_earth**4. Once that scaling is done,
  one can derive an equivalent ν_tensor given a ν_const, using the grid-dependent
  value h = ((np-1)*∆x/2)^{hv_scaling} * radius_earth**4, and find
  ν_tensor = ν_const / h. Therefore, a spatially uniform hyperviscosity with ν_const=10^15
  would have ν_tensor=7e-8 (though in practice most runs use 3.4e-8).
  [TODO] Make this explanation more accessible

  One typically uses `se_grid.create_spectral_element_grid` to create
  the `grid` argument.

  Returns
  -------
  viscosity_tensor: `Array[tuple[elem_idx, gll_idx, gll_idx, lon_lat], Float]`
      Anisotropy tensor applied in last application of laplacian within hyperviscosity.
  """

  eigs, evecs_normed = np.linalg.eigh(met_inv)
  lam_star = 1.0 / (eigs**(hypervis_scaling / 4.0))
  met_inv_scaled = np.einsum("fijmc, fijnc, fijc, fijc->fijmn", evecs_normed, evecs_normed, eigs, lam_star**2)

  # NOTE: missing rearth**4 scaling compared to HOMME code
  viscosity_tensor = np.einsum("fijmn, fijsm, fijrn -> fijsr",
                               met_inv_scaled,
                               jacobian,
                               jacobian)
  return viscosity_tensor, hypervis_scaling
