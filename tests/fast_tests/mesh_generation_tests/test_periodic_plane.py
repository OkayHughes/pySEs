from pyses.mesh_generation.periodic_plane import init_periodic_plane, metric_terms_to_grid
from pyses._config import get_backend as _get_backend
import numpy as np
from ...context import test_npts
_be = _get_backend()
jnp = _be.np
get_global_array = _be.get_global_array


def test_init_periodic_plane():
  for npt in test_npts:
    for nx in range(3, 6):
      ny = nx + 1
      NELEM = nx * ny
      physical_coords, ref_to_planar, vert_red = init_periodic_plane(nx, ny, npt)
      physical_coords_test = (physical_coords + 1.0) % (2.0 - 1e-10)
      ct = 0
      for target_face_idx in vert_red.keys():
        for (target_i, target_j) in vert_red[target_face_idx].keys():
          for (source_face_idx, source_i, source_j) in vert_red[target_face_idx][(target_i, target_j)]:
            assert np.allclose(physical_coords_test[target_face_idx, target_i, target_j],
                               physical_coords_test[source_face_idx, source_i, source_j])
            ct += 1
      nsides = 4
      ncorners = 4
      assert ct == (nsides * npt + ncorners) * NELEM


def test_metric():
  for npt in test_npts:
    for nx in range(3, 6):
      ny = nx + 1
      physical_coords, ref_to_planar, vert_red = init_periodic_plane(nx, ny, npt)
      grid, dims = metric_terms_to_grid(physical_coords, ref_to_planar, vert_red, npt)
      assert (np.allclose(jnp.sum(get_global_array(grid["metric_determinant"], dims) *
                                  (grid["gll_weights"][np.newaxis, :, np.newaxis] *
                                   grid["gll_weights"][np.newaxis, np.newaxis, :])), 4.0))
