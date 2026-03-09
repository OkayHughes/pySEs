import numpy as np


def init_deriv(gll_points):
  """
  Initialize the matrix that
  computes spectral derivatives within
  a reference element.

  Parameters
  ----------
  gll_points: Array[tuple[gll_idx], Float]
      The Gauss-Lobatto-Legendre nodes used to construct
      nodal interpolating functions.

  Returns
  -------
  deriv: Array[tuple[deriv_eval_idx, nodal_value_idx], Float]
      Derivative for calculating

  Notes
  -----
  Derivatives are calculated for the first and second dimensions
  in the reference element as
  df_da = np.einsum("ij,ki->kj", f, deriv)
  df_db = np.einsum("ij,kj->ik", f, deriv)
  respectively.

  Recall: this code uses a nodal representation
  of spectral data,
  so the values of a function expanded in
  the interpolation functions at the GLL points
  are precisely its coefficients.
  """
  # uses the lagrange interpolating polynomials
  npt = gll_points.size
  leg_eval = np.zeros(shape=(npt, npt))
  leg_der = np.zeros(shape=(npt, npt))

  for deg in range(npt):
    c = np.zeros(npt)
    c[deg] = 1.0
    leg_eval[:, deg] = np.polynomial.legendre.legval(gll_points, c)
    der = np.polynomial.legendre.legder(c, 1)
    leg_der[:, deg] = np.polynomial.legendre.legval(gll_points, der)

  coeffs = np.linalg.inv(leg_eval)
  return np.dot(leg_der, coeffs)


valid_npts = [3, 4, 5, 6]
_gll_points = {3: {"points": np.array([1.0, 0.0, -1.0]),
                   "weights": np.array([1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0])},
               4: {"points": np.array([1.0, np.sqrt(1 / 5), -np.sqrt(1 / 5), -1.0]),
                   "weights": np.array([1 / 6, 5 / 6, 5 / 6, 1 / 6])},
               5: {"points": np.array([1.0, np.sqrt(3 / 7), 0.0, -np.sqrt(3 / 7), -1.0]),
                   "weights": np.array([1.0 / 10.0, 49.0 / 90.0, 32.0 / 45.0, 49.0 / 90.0, 1.0 / 10.0])},
               6: {"points": np.array([1.0,
                                       np.sqrt(1.0 / 3.0 + 2.0 * np.sqrt(7) / 21.0),
                                       np.sqrt(1.0 / 3.0 - 2.0 * np.sqrt(7) / 21.0),
                                       -np.sqrt(1.0 / 3.0 - 2.0 * np.sqrt(7) / 21.0),
                                       -np.sqrt(1.0 / 3.0 + 2.0 * np.sqrt(7) / 21.0),
                                       -1.0]),
                   "weights": np.array([1.0 / 15.0,
                                        (14.0 - np.sqrt(7)) / 30.0,
                                        (14.0 + np.sqrt(7)) / 30.0,
                                        (14.0 + np.sqrt(7)) / 30.0,
                                        (14.0 - np.sqrt(7)) / 30.0,
                                        1.0 / 15.0])}}


def init_spectral(npt):
  """
  Return the necessary quantities
  to do spectral quadrature
  and differentiation for a given basis order.

  Parameters
  ----------
  npt:
      The number of Gauss-Lobatto-Legendre points
      to use in the reference interval

  Returns
  -------
  spectrals: dict[str, Array]
    Contains
    * "gll_points": Array[tuple[gll_idx], Float]
      1d GLL points
    * "gll_weights": Array[tuple[gll_idx], Float]
      1d weights for GLL quadrature
    * "deriv": Array[tuple[gll_idx, gll_idx], Float]
      Matrix for calculating spectral derivatives
      in the nodal basis.

  Raises
  ------
  KeyError
      if an invalid number of GLL points is requested.
  """
  return {"gll_points": _gll_points[npt]["points"],
          "gll_weights": _gll_points[npt]["weights"],
          "deriv": init_deriv(_gll_points[npt]["points"])}
