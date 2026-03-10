from pyses._config import get_backend as _get_backend
import numpy as np
from ...reference_implementations.vert_remap_reference import for_loop_remap
from pyses.dynamical_cores.vertical_remap import zerroukat_remap
from ...context import seed as global_seed
_be = _get_backend()
jnp = _be.np
device_wrapper = _be.array


def get_testbed(seed=True, random=False, wrap=False):
  nF = 1
  npt = 1
  nlev = 10
  reference_levs = np.linspace(0, 1, nlev + 1)
  if seed:
    np.random.seed(0)
  else:
    np.random.seed(global_seed)
  ints = np.concatenate(([0.0],
                         np.sort(np.random.uniform(size=nlev - 1)),
                         [1.0]))
  deltas = (ints[1:] - ints[:-1])[np.newaxis, np.newaxis, np.newaxis, :] * np.ones((nF, npt, npt, 1))
  deltas_ref = ((reference_levs[1:] - reference_levs[:-1])[np.newaxis, np.newaxis, np.newaxis, :] *
                np.ones((nF, npt, npt, 1)))

  if random:
    Qs = np.random.uniform(size=(nF, npt, npt, nlev, 1))
  else:
    Qs = np.ones((nF, npt, npt, nlev, 1))
  if wrap:
    return (device_wrapper(deltas),
            device_wrapper(deltas_ref),
            device_wrapper(Qs),
            device_wrapper(Qs * deltas[:, :, :, :, np.newaxis]))
  else:
    return deltas, deltas_ref, Qs, Qs * deltas[:, :, :, :, np.newaxis]


def test_for_remap():
  for _ in range(100):
    for random in [True, False]:
      deltas, deltas_ref, Qs, Qdps = get_testbed(seed=False, random=random)
      Qdp_out = for_loop_remap(Qdps, deltas, deltas_ref)
      Qdp_out_filt = for_loop_remap(Qdps, deltas, deltas_ref, filter=True)
      if not random:
        assert(np.allclose(Qdp_out, deltas_ref[:, :, :, :, np.newaxis]))
      assert(np.allclose(np.sum(Qdps, axis=-2), np.sum(Qdp_out, axis=-2)))
      assert(np.allclose(np.sum(Qdps, axis=-2), np.sum(Qdp_out_filt, axis=-2)))
      assert(not np.any(Qdp_out_filt < 0))


def test_remap():
  for _ in range(100):
    deltas, deltas_ref, Qs, Qdps = get_testbed(seed=True, random=True, wrap=True)

    Qdp_out_for = device_wrapper(for_loop_remap(Qdps, deltas, deltas_ref, filter=False))
    Qdp_out_for_filt = device_wrapper(for_loop_remap(Qdps, deltas, deltas_ref, filter=True))

    Qdp_out = zerroukat_remap(Qdps, deltas, deltas_ref, Qdps.shape[-2])
    Qdp_out_filt = zerroukat_remap(Qdps, deltas, deltas_ref, Qdps.shape[-2], filter=True)

    assert (jnp.max(jnp.abs(Qdp_out_for - Qdp_out)) < 1e-10)
    assert (jnp.max(jnp.abs(Qdp_out_for_filt - Qdp_out_filt)) < 1e-10)

    assert(jnp.allclose(jnp.sum(Qdps, axis=-2), jnp.sum(Qdp_out, axis=-2)))
    assert(jnp.allclose(jnp.sum(Qdps, axis=-2), jnp.sum(Qdp_out_filt, axis=-2)))
    assert(jnp.all(Qdp_out_filt > 0))
