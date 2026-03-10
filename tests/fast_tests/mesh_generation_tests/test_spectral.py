import numpy as np
from pyses.mesh_generation.spectral import init_spectral, _gll_points


def test_quadrature():
  for npt in _gll_points.keys():
    deriv = init_spectral(npt)
    assert (np.allclose(np.sum(deriv["gll_weights"] * deriv["gll_points"]**2), 2 / 3))


def test_generate_derivative():
  for npt in _gll_points.keys():
    deriv = init_spectral(npt)
    if npt > 3:
      assert (np.allclose(np.dot(deriv["deriv"], deriv["gll_points"]**2 - deriv["gll_points"]**3),
                          (2 * deriv["gll_points"] - 3 * deriv["gll_points"]**2)))
    assert (np.allclose(np.dot(deriv["deriv"], deriv["gll_points"]**2 - 4.0 * deriv["gll_points"]),
                        (2 * deriv["gll_points"] - 4.0)))
