from .._config import get_backend as _get_backend
_be = _get_backend()
partial = _be.partial
jit = _be.jit
vmap_1d_apply = _be.vmap_1d_apply
from ..operations_2d.operators import horizontal_divergence, horizontal_vorticity
from ..operations_2d.operators import horizontal_gradient, horizontal_weak_laplacian, horizontal_weak_vector_laplacian


@jit
def horizontal_divergence_3d(vector,
                             h_grid,
                             physics_config):
  """
  Compute the horizontal divergence of a 3-D vector field level-by-level.

  Parameters
  ----------
  vector : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx, 2], Float]
      Horizontal vector field (covariant components).
  h_grid : SpectralElementGrid
      Horizontal spectral element grid.
  physics_config : dict[str, Any]
      Physical constants; must contain ``radius_earth``.

  Returns
  -------
  div : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Horizontal divergence (s^-1) at each model level.
  """
  sph_op = partial(horizontal_divergence, grid=h_grid, a=physics_config["radius_earth"])
  return vmap_1d_apply(sph_op, vector, -2, -1)


@jit
def horizontal_vorticity_3d(vector,
                            h_grid,
                            physics_config):
  """
  Compute the horizontal (vertical component of) vorticity of a 3-D vector field level-by-level.

  Parameters
  ----------
  vector : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx, 2], Float]
      Horizontal vector field (covariant components).
  h_grid : SpectralElementGrid
      Horizontal spectral element grid.
  physics_config : dict[str, Any]
      Physical constants; must contain ``radius_earth``.

  Returns
  -------
  vort : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Relative vorticity (s^-1) at each model level.
  """
  sph_op = partial(horizontal_vorticity, grid=h_grid, a=physics_config["radius_earth"])
  return vmap_1d_apply(sph_op, vector, -2, -1)


@jit
def horizontal_weak_laplacian_3d(scalar,
                                 h_grid,
                                 physics_config):
  """
  Compute the weak-form horizontal scalar Laplacian of a 3-D field level-by-level.

  Parameters
  ----------
  scalar : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      3-D scalar field.
  h_grid : SpectralElementGrid
      Horizontal spectral element grid.
  physics_config : dict[str, Any]
      Physical constants; must contain ``radius_earth``.

  Returns
  -------
  lap : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      Weak-form Laplacian at each model level.
  """
  sph_op = partial(horizontal_weak_laplacian, grid=h_grid, a=physics_config["radius_earth"])
  return vmap_1d_apply(sph_op, scalar, -1, -1)


@jit
def horizontal_weak_vector_laplacian_3d(vector,
                                        h_grid,
                                        physics_config):
  """
  Compute the weak-form horizontal vector Laplacian of a 3-D vector field level-by-level.

  Parameters
  ----------
  vector : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx, 2], Float]
      3-D horizontal vector field.
  h_grid : SpectralElementGrid
      Horizontal spectral element grid.
  physics_config : dict[str, Any]
      Physical constants; must contain ``radius_earth``.

  Returns
  -------
  lap_vec : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx, 2], Float]
      Weak-form vector Laplacian at each model level.
  """
  sph_op = partial(horizontal_weak_vector_laplacian, grid=h_grid, a=physics_config["radius_earth"])
  return vmap_1d_apply(sph_op, vector, -2, -2)


@jit
def horizontal_gradient_3d(scalar,
                           h_grid,
                           physics_config):
  """
  Compute the horizontal gradient of a 3-D scalar field level-by-level.

  Parameters
  ----------
  scalar : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx], Float]
      3-D scalar field.
  h_grid : SpectralElementGrid
      Horizontal spectral element grid.
  physics_config : dict[str, Any]
      Physical constants; must contain ``radius_earth``.

  Returns
  -------
  grad : Array[tuple[elem_idx, gll_idx, gll_idx, lev_idx, 2], Float]
      Covariant horizontal gradient at each model level.
  """
  sph_op = partial(horizontal_gradient, grid=h_grid, a=physics_config["radius_earth"])
  return vmap_1d_apply(sph_op, scalar, -1, -2)
