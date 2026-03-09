from src._config import get_backend as _get_backend
_be = _get_backend()
import numpy as np
from src.mpi.processor_decomposition import (init_decomp,
                                                sphere_coord_to_face_idx_pos,
                                                init_mapping)
from ...context import get_figdir


def test_get_decomp():
  for _ in range(20):
    num_faces = np.random.randint(0, int(1e3))
    for num_procs in range(1, int(num_faces / 3.0)):
      segments = init_decomp(num_faces, num_procs)

      for seg_idx, segment in enumerate(segments[1:]):
        assert segments[seg_idx][1] == segment[0]
        assert segment[1] - segment[0] > 0
      assert segments[0][0] == 0
      assert segments[-1][1] == num_faces


def test_mapping():
  num_lat = 200
  lats, lons = np.meshgrid(np.linspace(-np.pi / 2.0, np.pi / 2.0, num_lat),
                           np.linspace(0.0, 2.0 * np.pi, 2 * num_lat))
  lats_flat = lats.flatten()
  lons_flat = lons.flatten()
  latlons = np.stack((lats_flat,
                      lons_flat), axis=-1)
  face_idxs, x, y = sphere_coord_to_face_idx_pos(latlons[:, 0], latlons[:, 1])
  index_map = init_mapping(11, latlons)
  latlons = np.take(latlons, index_map, axis=0)
  face_idxs = np.take(face_idxs, index_map, axis=0)
  x = np.take(x, index_map, axis=0)
  y = np.take(y, index_map, axis=0)
  max_dist_x = 0.0
  max_dist_y = 0.0
  dists_x = []
  face_num = []
  dists_y = []
  lats = []
  lons = []
  for elem_idx, face_idx in enumerate(face_idxs[:-1]):
    if face_idxs[elem_idx + 1] == face_idxs[elem_idx]:
      x_dist = np.abs(x[elem_idx] - x[elem_idx + 1])
      y_dist = np.abs(y[elem_idx] - y[elem_idx + 1])
      lats.append(latlons[elem_idx, 0])
      lons.append(latlons[elem_idx, 1])
      dists_x.append(x_dist)
      dists_y.append(y_dist)
      face_num.append(face_idx)
      max_dist_x = max(x_dist, max_dist_x)
      max_dist_y = max(y_dist, max_dist_y)
  assert(max_dist_x < 0.1)
  assert(max_dist_y < 0.1)
  if False:
    import matplotlib.pyplot as plt
    for face in range(6):
      plt.figure()
      for elem_idx, face_idx in enumerate(face_idxs[:-1]):
        if face_idxs[elem_idx + 1] == face_idxs[elem_idx] and face_idxs[elem_idx] == face:
          lats_tmp = [y[elem_idx],
                      y[elem_idx + 1]]
          lons_tmp = [x[elem_idx],
                      x[elem_idx + 1]]
          plt.plot(lons_tmp, lats_tmp, c="k")
      plt.savefig(f"{get_figdir()}/pairs_{face}.pdf")
    print(max_dist_x)
    print(max_dist_y)
    plt.figure()
    plt.scatter(lons, lats, c=dists_x)
    plt.colorbar()
    plt.savefig(f"{get_figdir()}/dist_x.pdf")
    plt.figure()
    plt.scatter(lons, lats, c=dists_y)
    plt.colorbar()
    plt.savefig(f"{get_figdir()}/dist_y.pdf")
    plt.figure()
    plt.scatter(lons, lats, c=face_num)
    plt.colorbar()
    plt.savefig(f"{get_figdir()}/face_num.pdf")
    plt.figure()
    plt.scatter(lons, lats, c=np.arange(len(lons)))
    plt.colorbar()
    plt.savefig(f"{get_figdir()}/idxs.pdf")
