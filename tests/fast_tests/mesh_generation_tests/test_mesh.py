import numpy as np
from ...context import test_npts
from pyses.mesh_generation.cubed_sphere import init_cube_topo
from pyses.mesh_generation.mesh import init_element_corner_vert_redundancy
from pyses.mesh_generation.mesh_definitions import TOP_FACE, BOTTOM_FACE, FRONT_FACE
from pyses.mesh_generation.mesh_definitions import BACK_FACE, LEFT_FACE, RIGHT_FACE
from pyses.mesh_generation.cubed_sphere import elem_id_fn
from pyses.mesh_generation.mesh import mesh_to_cart_bilinear, init_spectral_grid_redundancy
from pyses.mesh_generation.equiangular_metric import init_quasi_uniform_grid
from pyses.mesh_generation.element_local_metric import init_quasi_uniform_grid_elem_local
from pyses.mesh_generation.spherical_coord_utils import unit_sphere_to_cart_coords


def test_gen_bilinear_grid_cs():
  nx = 7
  # note: test is only valid on quasi-uniform grid
  for npt in test_npts:
    face_connectivity, face_mask, face_position, face_position_2d = init_cube_topo(nx)
    vert_redundancy = init_element_corner_vert_redundancy(face_connectivity)
    gll_pos, gll_jacobian = mesh_to_cart_bilinear(face_position_2d, npt)
    vert_redundancy_gll = init_spectral_grid_redundancy(vert_redundancy, npt)
    for face_idx in [TOP_FACE, BOTTOM_FACE, FRONT_FACE, BACK_FACE, LEFT_FACE, RIGHT_FACE]:
      for x_idx in range(nx):
        for y_idx in range(nx):
          for i_idx in range(npt):
            for j_idx in range(npt):
              num_neighbors = 0
              if (((x_idx == 0 and y_idx == 0 and i_idx == 0 and j_idx == 0) or
                   (x_idx == 0 and y_idx == nx - 1 and i_idx == 0 and j_idx == npt - 1) or
                   (x_idx == nx - 1 and y_idx == nx - 1 and i_idx == npt - 1 and j_idx == npt - 1) or
                   (x_idx == nx - 1 and y_idx == 0 and i_idx == npt - 1 and j_idx == 0))):
                num_neighbors = 2
              elif ((i_idx == 0 and j_idx == 0) or
                    (i_idx == 0 and j_idx == npt - 1) or
                    (i_idx == npt - 1 and j_idx == 0) or
                    (i_idx == npt - 1 and j_idx == npt - 1)):
                num_neighbors = 3

              if j_idx != 0 and j_idx != npt - 1:
                if i_idx == 0 or i_idx == npt - 1:
                  num_neighbors = 1
              if i_idx != 0 and i_idx != npt - 1:
                if j_idx == 0 or j_idx == npt - 1:
                  num_neighbors = 1
              elem_idx = elem_id_fn(nx, face_idx, x_idx, y_idx)
              if (i_idx, j_idx) in vert_redundancy_gll[elem_idx].keys():
                assert (num_neighbors == len(vert_redundancy_gll[elem_idx][(i_idx, j_idx)]))
              else:
                assert (num_neighbors == 0)


def test_equiangular_element_local_grid_equivalence():
  """
  The equiangular and element-local metrics discretize the *same* cubed-sphere
  geometry two different ways, both funnelling through ``metric_terms_to_grid``:
  the equiangular metric uses the exact gnomonic equiangular map, while the
  element-local metric uses a bilinear map from the four (shared) element
  corners followed by projection onto the unit sphere.  On a quasi-uniform grid
  the two must agree in the fine-resolution limit, so we compare them on a
  large, odd-``nx`` grid.

  The element corners coincide exactly between the two metrics, so all
  differences live in the element interiors and shrink with resolution:

    * node positions differ at O(h^2)        (~7e-5 at nx=61)
    * the metric determinant differs at O(h)  (~3.4% at nx=61, concentrated at
      element-edge / cube-face-boundary nodes, where the equiangular Jacobian
      is intentionally only C0)

  An odd ``nx`` keeps the poles in the interior of a single element on each
  polar face rather than on a shared element corner, so no cube-vertex GLL node
  lands exactly on a pole -- this is the configuration that previously exposed
  the coordinate singularity the element-local metric is designed to remove.

  Two exact invariants are also asserted: both metrics must integrate the unit
  sphere to 4*pi, and both metric determinants must be strictly positive (a
  negative determinant would betray a lat/lon row swap in the assembled
  Jacobian).  Thresholds were calibrated at nx=61 and are stable across
  npt in {3, 4, 5, 6}.
  """
  nx = 61   # large and odd
  npt = 4   # p = 3 spectral elements, as in Guba et al. (2014)

  grid_equi, _ = init_quasi_uniform_grid(nx, npt, wrapped=False)
  grid_elem, _ = init_quasi_uniform_grid_elem_local(nx, npt, wrapped=False)

  metdet_equi = np.asarray(grid_equi["metric_determinant"])
  metdet_elem = np.asarray(grid_elem["metric_determinant"])

  # Exact invariant 1: consistent orientation (row order) -> positive determinant.
  assert np.all(metdet_equi > 0.0)
  assert np.all(metdet_elem > 0.0)

  # Exact invariant 2: both metrics integrate the unit sphere to 4*pi.
  gll_weights = np.asarray(grid_equi["gll_weights"])
  quad_weights = gll_weights[None, :, None] * gll_weights[None, None, :]
  for metdet in (metdet_equi, metdet_elem):
    assert np.isclose(np.sum(metdet * quad_weights), 4.0 * np.pi, rtol=1e-6)

  # Node positions agree to O(h^2); compare in Cartesian to sidestep lon wrap.
  xyz_equi = unit_sphere_to_cart_coords(grid_equi["physical_coords"])
  xyz_elem = unit_sphere_to_cart_coords(grid_elem["physical_coords"])
  max_position_error = np.max(np.linalg.norm(xyz_equi - xyz_elem, axis=-1))
  assert max_position_error < 5e-4

  # Metric determinant agrees to O(h) (measured max ~3.4% at nx=61 on the
  # equiangular corner placement, whose edge elements are larger than the
  # former equidistant placement's).
  metdet_rel_error = np.abs(metdet_elem - metdet_equi) / np.abs(metdet_equi)
  assert np.max(metdet_rel_error) < 5e-2
