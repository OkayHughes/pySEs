from pyses._config import get_backend as _get_backend
import numpy as np
from pyses.operations_2d.limiters import clip_and_sum_limiter, full_limiter
from ...reference_implementations.limiters import clip_and_sum_limiter_for, full_limiter_for
from pyses.mesh_generation.spectral import init_spectral
_be = _get_backend()
jnp = _be.np


def test_conservation_bounds():
  npt = 4
  nelem = 6
  nlev = 5
  spectral = init_spectral(npt)
  tracer_shape = (nelem, npt, npt, nlev)
  mass_matrix = (np.ones((nelem, 1, 1)) *
                 spectral["gll_weights"][np.newaxis, :, np.newaxis] *
                 spectral["gll_weights"][np.newaxis, np.newaxis, :])
  pseudocoordinate = np.arange(nelem * npt * npt * nlev).reshape(tracer_shape)
  tracer_val = (np.cos(pseudocoordinate) + 1.0) / 2.0
  max_per_lev = np.max(tracer_val, axis=(0, 1, 2))[np.newaxis, :] * np.ones((nelem, nlev))
  min_per_lev = np.min(tracer_val, axis=(0, 1, 2))[np.newaxis, :] * np.ones((nelem, nlev))
  # mimic hyperviscosity perturbation
  tracer_val += 0.01 * np.sin(pseudocoordinate)
  d_mass = np.ones(tracer_shape) * np.arange(1, nlev + 1)[np.newaxis, np.newaxis, np.newaxis, :]
  tracer_mass_in = tracer_val * d_mass
  global_L2_norm_in = np.sum(mass_matrix[:, :, :, np.newaxis] * tracer_mass_in)
  tracer_mass_out = []
  for elem_idx in range(nelem):
    tracer_mass_out.append(clip_and_sum_limiter_for(tracer_mass_in[elem_idx, :, :, :],
                                                    mass_matrix[elem_idx, :, :],
                                                    min_per_lev[elem_idx, :],
                                                    max_per_lev[elem_idx, :],
                                                    d_mass[elem_idx, :, :, :]))
  tracer_mass_out = np.stack(tracer_mass_out, axis=0)
  global_L2_norm_out = np.sum(mass_matrix[:, :, :, np.newaxis] * tracer_mass_out)
  assert np.allclose(global_L2_norm_in, global_L2_norm_out)
  assert np.allclose(np.max(tracer_mass_out / d_mass, axis=(0, 1, 2)), max_per_lev[0, :])
  assert np.allclose(np.min(tracer_mass_out / d_mass, axis=(0, 1, 2)), min_per_lev[0, :])


def test_conservation_bounds_rand():
  npt = 4
  nelem = 6
  nlev = 5
  spectral = init_spectral(npt)
  tracer_shape = (nelem, npt, npt, nlev)
  for _ in range(100):
    mass_matrix = (np.random.uniform(size=tracer_shape[:-1]) *
                   spectral["gll_weights"][np.newaxis, :, np.newaxis] *
                   spectral["gll_weights"][np.newaxis, np.newaxis, :])
    tracer_val = np.random.uniform(size=tracer_shape)
    max_per_lev = np.max(tracer_val, axis=(0, 1, 2))[np.newaxis, :] * np.ones((nelem, nlev))
    min_per_lev = np.min(tracer_val, axis=(0, 1, 2))[np.newaxis, :] * np.ones((nelem, nlev))
    # mimic hyperviscosity perturbation
    tracer_val += np.random.normal(size=tracer_shape, scale=0.001)
    d_mass = np.random.uniform(size=tracer_shape)
    tracer_mass_in = tracer_val * d_mass
    global_L2_norm_in = np.sum(mass_matrix[:, :, :, np.newaxis] * tracer_mass_in)
    for fn in [full_limiter_for, clip_and_sum_limiter_for]:
      tracer_mass_out = []
      for elem_idx in range(nelem):
        tracer_mass_out.append(fn(tracer_mass_in[elem_idx, :, :, :],
                                  mass_matrix[elem_idx, :, :],
                                  min_per_lev[elem_idx, :],
                                  max_per_lev[elem_idx, :],
                                  d_mass[elem_idx, :, :, :]))
      tracer_mass_out = np.stack(tracer_mass_out, axis=0)
      global_L2_norm_out = np.sum(mass_matrix[:, :, :, np.newaxis] * tracer_mass_out)
      assert np.allclose(global_L2_norm_in, global_L2_norm_out)
      assert np.all(np.max(tracer_mass_out / d_mass, axis=(0, 1, 2)) <= max_per_lev[0, :] + 1e-9)
      assert np.all(np.min(tracer_mass_out / d_mass, axis=(0, 1, 2)) >= min_per_lev[0, :] - 1e-9)


def test_limiter_equiv():
  npt = 4
  nelem = 6
  nlev = 3
  spectral = init_spectral(npt)
  tracer_shape = (nelem, npt, npt, nlev)
  for _ in range(100):
    mass_matrix = (np.ones((nelem, 1, 1)) *
                   spectral["gll_weights"][np.newaxis, :, np.newaxis] *
                   spectral["gll_weights"][np.newaxis, np.newaxis, :])
    tracer_val = np.random.uniform(size=tracer_shape)
    max_per_lev = np.max(tracer_val, axis=(0, 1, 2))[np.newaxis, :] * np.ones((nelem, nlev))
    min_per_lev = np.min(tracer_val, axis=(0, 1, 2))[np.newaxis, :] * np.ones((nelem, nlev))
    # mimic hyperviscosity perturbation
    tracer_val += np.random.normal(size=tracer_shape, scale=0.001)
    d_mass = np.random.uniform(size=tracer_shape)
    tracer_mass_in = tracer_val * d_mass
    # global_L2_norm_in = np.sum(mass_matrix[:, :, :, np.newaxis] * tracer_mass_in)
    tracer_mass_out = []
    for elem_idx in range(nelem):
      tracer_mass_out.append(clip_and_sum_limiter_for(tracer_mass_in[elem_idx, :, :, :],
                                                      mass_matrix[elem_idx, :, :],
                                                      min_per_lev[elem_idx, :],
                                                      max_per_lev[elem_idx, :],
                                                      d_mass[elem_idx, :, :, :]))
    tracer_mass_out = np.stack(tracer_mass_out, axis=0)
    tracer_mass_out_numpy = clip_and_sum_limiter(tracer_mass_in, mass_matrix, min_per_lev, max_per_lev, d_mass)
    assert jnp.allclose(tracer_mass_out, tracer_mass_out_numpy)


def test_full_limiter_equiv():
  npt = 4
  nelem = 6
  nlev = 3
  spectral = init_spectral(npt)
  tracer_shape = (nelem, npt, npt, nlev)
  for _ in range(100):
    mass_matrix = (np.ones((nelem, 1, 1)) *
                   spectral["gll_weights"][np.newaxis, :, np.newaxis] *
                   spectral["gll_weights"][np.newaxis, np.newaxis, :])
    tracer_val = np.random.uniform(size=tracer_shape)
    max_per_lev = np.max(tracer_val, axis=(0, 1, 2))[np.newaxis, :] * np.ones((nelem, nlev))
    min_per_lev = np.min(tracer_val, axis=(0, 1, 2))[np.newaxis, :] * np.ones((nelem, nlev))
    # mimic hyperviscosity perturbation
    tracer_val += np.random.normal(size=tracer_shape, scale=0.001)
    d_mass = np.random.uniform(size=tracer_shape)
    tracer_mass_in = tracer_val * d_mass
    # global_L2_norm_in = np.sum(mass_matrix[:, :, :, np.newaxis] * tracer_mass_in)
    tracer_mass_out = []
    for elem_idx in range(nelem):
      tracer_mass_out.append(full_limiter_for(tracer_mass_in[elem_idx, :, :, :],
                                              mass_matrix[elem_idx, :, :],
                                              min_per_lev[elem_idx, :],
                                              max_per_lev[elem_idx, :],
                                              d_mass[elem_idx, :, :, :]))
    tracer_mass_out = np.stack(tracer_mass_out, axis=0)
    tracer_mass_out_numpy = full_limiter(tracer_mass_in, mass_matrix, min_per_lev, max_per_lev, d_mass)
    assert jnp.allclose(tracer_mass_out, tracer_mass_out_numpy)
