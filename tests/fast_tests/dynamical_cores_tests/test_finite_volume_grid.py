"""Regression tests for the element-local pg-N finite-volume remap operators.

Verifies the properties from Hannah et al. (2021) on a real quasi-uniform
cubed-sphere grid (ne3, np4, pg2):

* requirement R1 -- the reference operators satisfy ``A^{p->f} A^{f->p} = I``;
* the density-weighted round trip **FV -> GLL -> FV is the identity**;
* GLL -> FV -> GLL recovers any ``nf``-node-representable (bilinear for pg2)
  field exactly (R2), but is a genuine projection (not the identity) for a
  field with finer structure;
* the GLL -> FV average conserves the element-integrated (area-weighted) mass;
* shapes / column reshapes are consistent.
"""
import numpy as np

from pyses._config import get_backend as _get_backend
from pyses.mesh_generation.element_local_metric import init_quasi_uniform_grid_elem_local
from pyses.dynamical_cores.finite_volume_grid import (
    reference_fv_operators,
    init_fv_grid,
    gll_to_fv,
    fv_to_gll,
    fv_field_to_columns,
    columns_to_fv_field,
)

_be = _get_backend()
bnp = _be.np
W = _be.array
U = _be.unwrap

NX, NPT, NF, NLEV = 3, 4, 2, 5


def _grid():
  return init_quasi_uniform_grid_elem_local(NX, NPT, calc_smooth_tensor=True)


def test_reference_operators_satisfy_R1():
  """A^{p->f} A^{f->p} == I_nf (paper requirement R1) for pg2 and pg3."""
  for nf in (2, 3):
    ops = reference_fv_operators(NPT, nf)
    ident = ops["gll_to_fv"] @ ops["fv_to_gll"]
    assert np.allclose(ident, np.eye(nf), atol=1e-12), (nf, ident)
  # nf > np is rejected
  import pytest
  with pytest.raises(ValueError):
    reference_fv_operators(2, 4)


def test_fv_gll_fv_round_trip_is_identity():
  """FV -> GLL -> FV reproduces arbitrary FV data to machine precision."""
  h_grid, dims = _grid()
  fvg = init_fv_grid(h_grid, dims, nf=NF)
  ne = dims["num_elem"]
  rng = np.random.default_rng(0)
  fv0 = W(rng.standard_normal((ne, NF, NF, NLEV)))

  round_trip = gll_to_fv(fv_to_gll(fv0, fvg), fvg)
  assert np.allclose(np.asarray(U(round_trip)), np.asarray(U(fv0)), atol=1e-10)

  # also for a 3-D (single-level) FV field
  fv0_2d = W(rng.standard_normal((ne, NF, NF)))
  rt_2d = gll_to_fv(fv_to_gll(fv0_2d, fvg), fvg)
  assert np.allclose(np.asarray(U(rt_2d)), np.asarray(U(fv0_2d)), atol=1e-10)


def test_gll_fv_gll_recovers_representable_field():
  """GLL -> FV -> GLL is the identity on nf-representable fields (R2), and a
  genuine (non-identity) projection on a higher-order field."""
  h_grid, dims = _grid()
  fvg = init_fv_grid(h_grid, dims, nf=NF)
  ne = dims["num_elem"]
  rng = np.random.default_rng(1)

  # any field produced by fv_to_gll is exactly nf-node-representable
  representable = fv_to_gll(W(rng.standard_normal((ne, NF, NF, NLEV))), fvg)
  recovered = fv_to_gll(gll_to_fv(representable, fvg), fvg)
  assert np.allclose(np.asarray(U(recovered)), np.asarray(U(representable)), atol=1e-10)

  # a generic GLL field is NOT recovered (the map is a real projection)
  generic = W(rng.standard_normal((ne, NPT, NPT, NLEV)))
  projected = fv_to_gll(gll_to_fv(generic, fvg), fvg)
  assert not np.allclose(np.asarray(U(projected)), np.asarray(U(generic)), atol=1e-3)


def test_gll_to_fv_conserves_element_area_mass():
  """The area-weighted element integral is preserved by GLL -> FV (paper Eq 4):
  ``sum_ij w_i w_j J_ij f_ij == sum_cd wf_c wf_d Jf_cd f^f_cd``."""
  h_grid, dims = _grid()
  fvg = init_fv_grid(h_grid, dims, nf=NF)
  ne = dims["num_elem"]
  rng = np.random.default_rng(2)
  f = W(rng.standard_normal((ne, NPT, NPT, NLEV)))
  f_fv = gll_to_fv(f, fvg)

  mass_matrix = np.asarray(U(h_grid["mass_matrix"]))                 # w_i w_j J_ij
  gll_int = np.einsum("eij,eijl->el", mass_matrix, np.asarray(U(f)))

  ops = reference_fv_operators(NPT, NF)
  wf = ops["subcell_width"]                                          # (nf,)
  fv_weight = (wf[:, None] * wf[None, :])[None] * np.asarray(U(fvg["metric_determinant_fv"]))
  fv_int = np.einsum("ecd,ecdl->el", fv_weight, np.asarray(U(f_fv)))

  assert np.allclose(gll_int, fv_int, atol=1e-9), np.abs(gll_int - fv_int).max()


def test_shapes_and_column_reshape():
  """Output shapes, the (E, nf, nf, nlev) layout, and the column round trip."""
  h_grid, dims = _grid()
  fvg = init_fv_grid(h_grid, dims, nf=NF)
  ne = dims["num_elem"]

  f = W(np.zeros((ne, NPT, NPT, NLEV)))
  assert tuple(gll_to_fv(f, fvg).shape) == (ne, NF, NF, NLEV)

  fv = W(np.zeros((ne, NF, NF, NLEV)))
  assert tuple(fv_to_gll(fv, fvg).shape) == (ne, NPT, NPT, NLEV)

  # FV cell coordinates live on the (E, nf, nf, 2) grid
  assert tuple(fvg["physical_coords"].shape) == (ne, NF, NF, 2)

  # column flatten / unflatten round trips
  cols = fv_field_to_columns(fv)
  assert tuple(cols.shape) == (ne * NF * NF, NLEV)
  back = columns_to_fv_field(cols, ne, NF)
  assert np.allclose(np.asarray(U(back)), np.asarray(U(fv)))


def test_fv_cell_coords_are_area_average_of_nodes():
  """Each FV cell latitude is the area-weighted mean of its element's node
  latitudes, hence bounded by the element's node-coordinate range."""
  h_grid, dims = _grid()
  fvg = init_fv_grid(h_grid, dims, nf=NF)
  node_lat = np.asarray(U(h_grid["physical_coords"]))[..., 0]        # (E, np, np)
  fv_lat = np.asarray(U(fvg["physical_coords"]))[..., 0]             # (E, nf, nf)
  lo = node_lat.min(axis=(1, 2))[:, None, None]
  hi = node_lat.max(axis=(1, 2))[:, None, None]
  assert np.all(fv_lat >= lo - 1e-9) and np.all(fv_lat <= hi + 1e-9)
