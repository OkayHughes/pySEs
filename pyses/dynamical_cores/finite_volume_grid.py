"""Element-local "pg-N" finite-volume physics grid for spectral elements.

This module implements the high-order, element-local remap operators of

    Hannah, Bradley, Guba, Tang, Golaz & Wolfe (2021), "Separating Physics and
    Dynamics Grids for Improved Computational Efficiency in Spectral Element
    Earth System Models", JAMES 13, e2020MS002419, doi:10.1029/2020MS002419

for mapping fields between the Gauss-Lobatto-Legendre (GLL, "np") dynamics grid
and a quasi-equal-area finite-volume (FV, "pg-N") physics grid that divides each
spectral element into ``nf x nf`` sub-cells (e.g. ``pg2`` -> ``nf = 2``).

The operators are purely *element-local*: each is a small per-element tensor
contraction over the in-element GLL indices ``(i, j)``, so the element axis
(axis 0) is carried straight through.  Any device sharding present on the
``h_grid`` arrays along the element axis is therefore inherited by the FV fields
and the remapped tendencies without any explicit resharding.

Conventions and the key identity
--------------------------------
Following the paper (Section 2.2), with ``b_i`` the GLL Lagrange basis on the
reference element ``[-1, 1]^2`` and ``J`` the metric determinant (area Jacobian)
at each node:

* ``M[c, i] = \\int_{R^f_c} b_i dr`` -- the integral of GLL basis ``i`` over FV
  sub-cell ``c`` (a constant reference matrix; the 2-D version is the tensor
  product of the 1-D matrix);
* GLL -> FV (area-weighted average, ``B^{p->f}``)::

      fv_c = ( sum_i M[c,i] J_i p_i ) / ( sum_i M[c,i] J_i )

  i.e. the metric-weighted (area-weighted) average of the GLL field over the
  sub-cell -- rows sum to one;
* FV -> GLL (``B^{f->p}``) uses the reference operator
  ``A^{f->p} = I^{p'->p} (A^{p->f} I^{p'->p})^{-1}`` (with ``np' = nf``),
  density-weighted by ``J``.

These satisfy the paper's idempotence requirement R1, so the round trip
**FV -> GLL -> FV is the identity** to machine precision, and GLL -> FV -> GLL
recovers any field that is exactly representable in the ``nf``-node basis
(bilinear for ``pg2``); see the regression tests for the checks.

The FV grid struct returned by :func:`init_fv_grid` carries the (small,
replicated) reference operators plus the per-element metric quantities (sharded
like ``h_grid``) and the area-averaged FV cell coordinates.
"""
import numpy as np

from .._config import get_backend as _get_backend
from ..mesh_generation.spectral import _gll_points

_be = _get_backend()
bnp = _be.np
device_wrapper = _be.array
unwrap = _be.unwrap


# ---------------------------------------------------------------------------
# Reference-element (host) construction of the linear remap operators
# ---------------------------------------------------------------------------
def _gll_nodes(n):
  """1-D GLL nodes for an ``n``-point rule, ordered as in :mod:`spectral`.

  ``spectral`` only tabulates ``n in {3, 4, 5, 6}``; the 2-point GLL rule is
  just the element endpoints, which the panel basis (``np' = nf``) needs for
  ``pg2``.
  """
  if n == 2:
    return np.array([1.0, -1.0])
  return np.asarray(_gll_points[n]["points"], dtype=np.float64)


def _lagrange_legendre_coeffs(gll_points):
  """Legendre coefficients of the GLL Lagrange basis.

  Returns ``coeffs`` such that the ``i``-th nodal basis function is
  ``L_i(x) = sum_deg coeffs[deg, i] P_deg(x)`` (``P_deg`` the Legendre
  polynomial), i.e. ``coeffs = inv(V)`` with ``V[j, deg] = P_deg(node_j)``.
  Mirrors the construction in ``spectral.init_deriv``.
  """
  npt = gll_points.size
  vandermonde = np.zeros((npt, npt))
  for deg in range(npt):
    c = np.zeros(npt)
    c[deg] = 1.0
    vandermonde[:, deg] = np.polynomial.legendre.legval(gll_points, c)
  return np.linalg.inv(vandermonde)


def _subcell_basis_integrals(gll_points, nf):
  """1-D matrix ``M[c, i] = \\int_{subcell_c} L_i(x) dx`` over ``[-1, 1]``.

  Sub-cells are the ``nf`` uniform sub-intervals of the reference interval.
  The integral is evaluated analytically from the Legendre series of ``L_i``
  (exact, since ``L_i`` is a polynomial), via ``legint``.
  """
  coeffs = _lagrange_legendre_coeffs(gll_points)   # (npt, npt): [deg, i]
  npt = gll_points.size
  edges = np.linspace(-1.0, 1.0, nf + 1)
  mat = np.zeros((nf, npt))
  for i in range(npt):
    antider = np.polynomial.legendre.legint(coeffs[:, i])
    vals = np.polynomial.legendre.legval(edges, antider)
    mat[:, i] = vals[1:] - vals[:-1]
  return mat, edges


def _interp_low_to_high(gll_low, gll_high):
  """``I[j, k] = L^{low}_k(x^{high}_j)`` -- the ``np'``-basis evaluated at the
  ``np`` nodes (the no-error projection ``I^{p'->p}`` for ``np' <= np``)."""
  coeffs_low = _lagrange_legendre_coeffs(gll_low)   # (nlow, nlow): [deg, k]
  out = np.zeros((gll_high.size, gll_low.size))
  for k in range(gll_low.size):
    out[:, k] = np.polynomial.legendre.legval(gll_high, coeffs_low[:, k])
  return out


def reference_fv_operators(npt, nf):
  """Constant reference-element remap matrices for an ``(np, nf)`` pair.

  Parameters
  ----------
  npt : int
      GLL points per element edge (``np``).
  nf : int
      FV sub-cells per element edge (``nf``); ``nf <= np``.

  Returns
  -------
  dict with the 1-D operators (all on ``[-1, 1]``):
      ``"subcell_integral"`` ``M`` ``(nf, np)``;
      ``"subcell_width"``     ``w^f`` ``(nf,)``;
      ``"gll_to_fv"``         ``A^{p->f} = diag(w^f)^{-1} M`` ``(nf, np)``;
      ``"fv_to_gll"``         ``A^{f->p} = I (A^{p->f} I)^{-1}`` ``(np, nf)``.

  ``A^{p->f} A^{f->p} == I_{nf}`` by construction (paper requirement R1), which
  is what makes the density-weighted FV -> GLL -> FV round trip exact.
  """
  if nf > npt:
    raise ValueError(f"pg-N requires nf <= np; got nf={nf}, np={npt}")
  gll_p = _gll_nodes(npt)
  subcell_integral, edges = _subcell_basis_integrals(gll_p, nf)   # (nf, np)
  subcell_width = np.diff(edges)                                  # (nf,)
  gll_to_fv = subcell_integral / subcell_width[:, None]           # (nf, np)
  # panel basis has np' = nf nodes (no projection error since nf <= np)
  interp = _interp_low_to_high(_gll_nodes(nf), gll_p)             # (np, nf)
  fv_to_gll = interp @ np.linalg.inv(gll_to_fv @ interp)          # (np, nf)
  return {
      "subcell_integral": subcell_integral,
      "subcell_width": subcell_width,
      "gll_to_fv": gll_to_fv,
      "fv_to_gll": fv_to_gll,
  }


# ---------------------------------------------------------------------------
# FV grid struct
# ---------------------------------------------------------------------------
def init_fv_grid(h_grid, dims, nf=2):
  """Build the pg-``nf`` finite-volume physics grid for a spectral-element grid.

  Parameters
  ----------
  h_grid : SpectralElementGrid
      Horizontal grid; must contain ``"metric_determinant"`` ``(E, np, np)``
      and ``"physical_coords"`` ``(E, np, np, 2)`` (lat, lon).
  dims : frozendict
      Grid dims; must contain ``"npt"`` and ``"num_elem"``.
  nf : int, optional
      FV sub-cells per element edge (default 2 -> ``pg2``).

  Returns
  -------
  fv_grid : dict
      Struct consumed by :func:`gll_to_fv` / :func:`fv_to_gll`.  Per-element
      metric arrays (``"metric_determinant"`` ``(E, np, np)`` and its FV
      counterpart ``(E, nf, nf)``) are wrapped with the same element-axis
      sharding as ``h_grid``; the small reference operators are replicated.
      ``"physical_coords"`` ``(E, nf, nf, 2)`` holds the area-averaged FV cell
      centres.
  """
  npt = int(dims["npt"])
  ops = reference_fv_operators(npt, nf)
  metdet = np.asarray(unwrap(h_grid["metric_determinant"]))      # (E, np, np)
  # FV metric (area) = A^{p->f} average of J over each sub-cell
  metdet_fv = np.einsum("ci,dj,eij->ecd",
                        ops["gll_to_fv"], ops["gll_to_fv"], metdet)

  fv_grid = {
      "nf": int(nf),
      "npt": npt,
      "num_elem": int(dims["num_elem"]),
      # replicated reference operators
      "subcell_integral": device_wrapper(ops["subcell_integral"]),
      "fv_to_gll_ref": device_wrapper(ops["fv_to_gll"]),
      # per-element metric, sharded on the element axis like h_grid
      "metric_determinant": device_wrapper(metdet, elem_sharding_axis=0),
      "metric_determinant_fv": device_wrapper(metdet_fv, elem_sharding_axis=0),
  }
  # area-averaged FV cell coordinates (lat, lon), inheriting the sharding
  fv_grid["physical_coords"] = gll_to_fv(h_grid["physical_coords"], fv_grid)
  return fv_grid


# ---------------------------------------------------------------------------
# The remap operators
# ---------------------------------------------------------------------------
def _safe_divide(num, den):
  """``num / den`` with ``0`` where ``den == 0`` (padded ghost elements)."""
  ok = den != 0.0
  return bnp.where(ok, num / bnp.where(ok, den, 1.0), 0.0)


def gll_to_fv(field, fv_grid):
  """Area-weighted average of a GLL field onto the FV grid (``B^{p->f}``).

  ``field`` is ``(E, np, np)`` or ``(E, np, np, K)`` (the trailing ``K`` may be
  levels, vector components, ...); the result is ``(E, nf, nf[, K])``.  Each FV
  value is the metric-weighted average of the GLL field over the sub-cell, so
  area-mean is preserved and the element axis (sharding) is untouched.
  """
  mat = fv_grid["subcell_integral"]                 # (nf, np)
  metdet = fv_grid["metric_determinant"]            # (E, np, np)
  weight = bnp.einsum("ci,dj,eij->ecd", mat, mat, metdet)        # (E, nf, nf)
  if field.ndim == 3:
    num = bnp.einsum("ci,dj,eij,eij->ecd", mat, mat, metdet, field)
    return _safe_divide(num, weight)
  num = bnp.einsum("ci,dj,eij,eijk->ecdk", mat, mat, metdet, field)
  return _safe_divide(num, weight[..., None])


def fv_to_gll(field_fv, fv_grid):
  """Element-local FV -> GLL remap (``B^{f->p}``).

  Inverse-companion of :func:`gll_to_fv`: ``gll_to_fv(fv_to_gll(x)) == x`` to
  machine precision (the paper's R1 identity).  ``field_fv`` is ``(E, nf, nf)``
  or ``(E, nf, nf, K)``; the result is the (element-discontinuous) GLL field
  ``(E, np, np[, K])``.  Apply DSS afterwards to make it continuous.
  """
  fv_to_gll_ref = fv_grid["fv_to_gll_ref"]          # (np, nf)
  metdet = fv_grid["metric_determinant"]            # (E, np, np)
  metdet_fv = fv_grid["metric_determinant_fv"]      # (E, nf, nf)
  if field_fv.ndim == 3:
    num = bnp.einsum("ic,jd,ecd,ecd->eij",
                     fv_to_gll_ref, fv_to_gll_ref, metdet_fv, field_fv)
    return _safe_divide(num, metdet)
  num = bnp.einsum("ic,jd,ecd,ecdk->eijk",
                   fv_to_gll_ref, fv_to_gll_ref, metdet_fv, field_fv)
  return _safe_divide(num, metdet[..., None])


# ---------------------------------------------------------------------------
# Reshape helpers between the (E, nf, nf, ...) FV layout and a flat column list
# ---------------------------------------------------------------------------
def fv_field_to_columns(field_fv):
  """``(E, nf, nf[, K]) -> (E*nf*nf[, K])`` -- flatten FV cells to columns."""
  nf = field_fv.shape[1]
  num_cols = field_fv.shape[0] * nf * nf
  return bnp.reshape(field_fv, (num_cols,) + tuple(field_fv.shape[3:]))


def columns_to_fv_field(columns, num_elem, nf):
  """Inverse of :func:`fv_field_to_columns`: ``(E*nf*nf[, K]) -> (E, nf, nf[, K])``."""
  return bnp.reshape(columns, (num_elem, nf, nf) + tuple(columns.shape[1:]))
