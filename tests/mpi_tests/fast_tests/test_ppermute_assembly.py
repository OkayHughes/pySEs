import numpy as np
import pytest
from pyses._config import get_backend as _get_backend
from pyses.mesh_generation.element_local_metric import init_quasi_uniform_grid_elem_local
from pyses.mpi.processor_decomposition import init_mapping
from pyses.operations_2d.local_assembly import init_global_comm_map
from ...context import test_npts, seed

_be = _get_backend()

# The ppermute DSS path is JAX-only and needs at least two mesh devices. Run
# with e.g. PYSES_BACKEND=jax PYSES_SHARD_CPU_COUNT=4 to exercise it.
_runnable = (_be.wrapper_type == "jax" and _be.num_devices > 1)
pytestmark = pytest.mark.skipif(
    not _runnable,
    reason="ppermute DSS requires the JAX backend with >1 device "
           "(set PYSES_BACKEND=jax PYSES_SHARD_CPU_COUNT>=2)")


def _ref_assemble(f, triple):
  """Serial reference DSS: out[rows] += out[cols] (no mass scaling)."""
  _, rows, cols = triple
  out = np.array(f).copy()
  relevant = out[cols[0], cols[1], cols[2]]
  np.add.at(out, (rows[0], rows[1], rows[2]), relevant)
  return out


def _run_case(nx, npt, n_subdiv=7, n_fields=5):
  num_devices = _be.num_devices
  grid, dim = init_quasi_uniform_grid_elem_local(nx, npt, wrapped=False)
  nelem = dim["num_elem"]
  data, rows, cols = grid["assembly_triple"]
  data = np.asarray(data)
  rows = [np.asarray(r) for r in rows]
  cols = [np.asarray(c) for c in cols]

  # Hilbert-curve patch decomposition: reorder elements so spatially adjacent
  # ones land in the same contiguous device block, then relabel the triple.
  c = npt // 2
  latlons = np.asarray(grid["physical_coords"])[:, c, c, :2].copy()
  index_map = init_mapping(n_subdiv, latlons)   # index_map[new] = old
  inv = np.argsort(index_map)                    # inv[old] = new
  rows_re = [inv[rows[0]], rows[1], rows[2]]
  cols_re = [inv[cols[0]], cols[1], cols[2]]
  triple_re = (data, rows_re, cols_re)

  nelem_padded = int(np.ceil(nelem / num_devices) * num_devices)
  comm_map = init_global_comm_map(triple_re, num_devices, nelem_padded, wrapped=True)

  rng = np.random.default_rng(seed)
  for _ in range(n_fields):
    f = rng.normal(size=(nelem, npt, npt))
    f_pad = np.zeros((nelem_padded, npt, npt))
    f_pad[:nelem] = f[index_map]

    expected = _ref_assemble(f_pad, triple_re)
    got = np.asarray(_be.unwrap(_be.dss_ppermute(_be.array(f_pad, elem_sharding_axis=0),
                                                 comm_map)))
    assert np.allclose(got[:nelem], expected[:nelem])


def test_dss_ppermute_matches_serial_reference():
  for npt in test_npts:
    for nx in range(1, 3):
      _run_case(nx, npt)
