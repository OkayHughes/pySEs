from pyses._config import get_backend as _get_backend
import numpy as np
from pyses.mesh_generation.bilinear_utils import (eval_bilinear_jacobian, eval_bilinear_mapping)
from pyses.mesh_generation.spherical_coord_utils import (cart_to_unit_sphere_coords, unit_sphere_to_cart_coords,
                                                       cart_to_unit_sphere_coords_jacobian,
                                                       unit_sphere_to_cart_coords_jacobian)
from pyses.mesh_generation.cubed_sphere import init_cube_topo
from pyses.mesh_generation.spectral import init_spectral
from ...context import test_npts
_be = _get_backend()


def test_bilinear():
  NFACES = 1000
  np.random.seed(0)
  face_position_2d = np.random.uniform(size=(NFACES, 4, 2))
  jac_test = np.zeros(shape=(NFACES, 2, 2))
  diff_minus = np.zeros(shape=(NFACES, 2))
  diff_plus = np.zeros(shape=(NFACES, 2))
  ncheck = 5
  nfrac = np.linspace(-1, 1, ncheck)
  for i in range(ncheck):
    for j in range(ncheck):
      alpha = nfrac[i]
      beta = nfrac[j]
      eps = 1e-6
      diff_plus = eval_bilinear_mapping(face_position_2d[:, 0, :],
                                        face_position_2d[:, 1, :],
                                        face_position_2d[:, 2, :],
                                        face_position_2d[:, 3, :], alpha + eps, beta)
      diff_minus = eval_bilinear_mapping(face_position_2d[:, 0, :],
                                         face_position_2d[:, 1, :],
                                         face_position_2d[:, 2, :],
                                         face_position_2d[:, 3, :], alpha - eps, beta)
      dres_dalpha = (diff_plus - diff_minus) / (2 * eps)
      diff_plus = eval_bilinear_mapping(face_position_2d[:, 0, :],
                                        face_position_2d[:, 1, :],
                                        face_position_2d[:, 2, :],
                                        face_position_2d[:, 3, :], alpha, beta + eps)
      diff_minus = eval_bilinear_mapping(face_position_2d[:, 0, :],
                                         face_position_2d[:, 1, :],
                                         face_position_2d[:, 2, :],
                                         face_position_2d[:, 3, :], alpha, beta - eps)
      dphys_dalpha, dphys_dbeta = eval_bilinear_jacobian(face_position_2d[:, 0, :],
                                                         face_position_2d[:, 1, :],
                                                         face_position_2d[:, 2, :],
                                                         face_position_2d[:, 3, :], alpha, beta)
      jac_test[:, :, 0] = dphys_dalpha
      jac_test[:, :, 1] = dphys_dbeta
      dres_dbeta = (diff_plus - diff_minus) / (2 * eps)
      assert (np.max(np.abs(dres_dalpha - jac_test[:, :, 0])) < 1e-7)
      assert (np.max(np.abs(dres_dbeta - jac_test[:, :, 1])) < 1e-7)


def test_bilinear_cs():
  for npt in test_npts:
    spectrals = init_spectral(npt)
    for nx in [7, 8]:
      face_connectivity, face_mask, face_position, face_position_2d = init_cube_topo(nx)
      NFACES = face_position.shape[0]
      jac_test = np.zeros(shape=(NFACES, 2, 2))
      diff_minus = np.zeros(shape=(NFACES, 2))
      diff_plus = np.zeros(shape=(NFACES, 2))
      for i in range(npt):
        for j in range(npt):
          alpha = spectrals["gll_points"][i]
          beta = spectrals["gll_points"][j]
          eps = 1e-4
          diff_plus = eval_bilinear_mapping(face_position_2d[:, 0, :],
                                            face_position_2d[:, 1, :],
                                            face_position_2d[:, 2, :],
                                            face_position_2d[:, 3, :], alpha + eps, beta)
          diff_minus = eval_bilinear_mapping(face_position_2d[:, 0, :],
                                             face_position_2d[:, 1, :],
                                             face_position_2d[:, 2, :],
                                             face_position_2d[:, 3, :], alpha - eps, beta)
          dres_dalpha = (diff_plus - diff_minus) / (2 * eps)
          diff_plus = eval_bilinear_mapping(face_position_2d[:, 0, :],
                                            face_position_2d[:, 1, :],
                                            face_position_2d[:, 2, :],
                                            face_position_2d[:, 3, :], alpha, beta + eps)
          diff_minus = eval_bilinear_mapping(face_position_2d[:, 0, :],
                                             face_position_2d[:, 1, :],
                                             face_position_2d[:, 2, :],
                                             face_position_2d[:, 3, :], alpha, beta - eps)
          dphys_dalpha, dphys_dbeta = eval_bilinear_jacobian(face_position_2d[:, 0, :],
                                                             face_position_2d[:, 1, :],
                                                             face_position_2d[:, 2, :],
                                                             face_position_2d[:, 3, :], alpha, beta)
          jac_test[:, :, 0] = dphys_dalpha
          jac_test[:, :, 1] = dphys_dbeta
          dres_dbeta = (diff_plus - diff_minus) / (2 * eps)
          assert (np.max(np.abs(dres_dalpha - jac_test[:, :, 0])) < 1e-7)
          assert (np.max(np.abs(dres_dbeta - jac_test[:, :, 1])) < 1e-7)


def test_sphere_coords():
  lat_1d = np.linspace(-np.pi / 2.0, np.pi / 2.0, 10)
  lon_1d = np.linspace(0, 2 * np.pi, 20)
  lat, lon = np.meshgrid(lat_1d, lon_1d)
  lat = lat.reshape((-1, 1, 1))
  lon = lon.reshape((-1, 1, 1))
  latlon = np.stack((lat, lon), axis=-1)
  cart = unit_sphere_to_cart_coords(latlon)
  latlon_out = cart_to_unit_sphere_coords(cart)
  latlon[:, :, :, 1] = np.mod(latlon[:, :, :, 1], 2 * np.pi)
  assert np.allclose(latlon_out, latlon, atol=1e-7)


def test_sphere_coords_jacobian():
  npts = 100
  lat_1d = np.linspace(-np.pi / 2.0, np.pi / 2.0, npts + 2)[1:-1]
  np.random.seed(0)
  lon_1d = np.linspace(-np.pi, np.pi, 2 * npts)
  lat, lon = np.meshgrid(lat_1d, lon_1d)
  lat = lat.reshape((-1, 1, 1))
  lon = lon.reshape((-1, 1, 1))
  latlon = np.stack((lat, lon), axis=-1)
  cart = unit_sphere_to_cart_coords(latlon)
  cart_to_sphere = cart_to_unit_sphere_coords_jacobian(cart)
  sphere_to_cart = unit_sphere_to_cart_coords_jacobian(latlon)
  cart_to_sphere[:, :, :, 1, :] *= np.cos(latlon[:, :, :, 0])[:, :, :, np.newaxis]
  sphere_to_cart[:, :, :, :, 1] /= np.cos(latlon[:, :, :, 0])[:, :, :, np.newaxis]
  rand_vecs_latlon = np.random.normal(size=latlon.shape)
  cart_vecs = np.einsum("fijcs,fijs->fijc", sphere_to_cart, rand_vecs_latlon)
  sph_from_cart_vecs = np.einsum("fijsc,fijc->fijs", cart_to_sphere, cart_vecs)
  norm_sph_vecs = np.linalg.norm(rand_vecs_latlon, axis=-1)
  norm_cart_vecs = np.linalg.norm(cart_vecs, axis=-1)
  norm_sph_from_cart_vecs = np.linalg.norm(sph_from_cart_vecs, axis=-1)
  assert np.allclose(norm_cart_vecs, norm_sph_vecs)
  assert np.allclose(norm_sph_from_cart_vecs, norm_sph_vecs)
  iprod_sph = np.einsum("fijc, fijc->fij", cart_vecs / norm_cart_vecs[:, :, :, np.newaxis], cart)
  assert np.allclose(iprod_sph, 0.0)

  maybe_identity = np.einsum("fijsc,fijct->fijst", cart_to_sphere, sphere_to_cart)
  maybe_identity_2 = np.einsum("fijcs,fijct->fijst", sphere_to_cart, sphere_to_cart)
  assert np.allclose(maybe_identity_2, np.eye(2)[np.newaxis, np.newaxis, np.newaxis, :, :])
  assert np.allclose(maybe_identity, np.eye(2)[np.newaxis, np.newaxis, np.newaxis, :, :])
