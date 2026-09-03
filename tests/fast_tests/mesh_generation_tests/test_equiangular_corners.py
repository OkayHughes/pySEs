"""
Tests that the cubed-sphere constructors place element corners on the
equiangular cubed sphere used by HOMME/CAM-SE (``cube_mod``): corners
uniform in the equiangular coordinates ``(α, β) ∈ [-π/4, π/4]²``, so
element boundaries coincide with HOMME's.
"""
import numpy as np
from ...context import test_npts
from pyses.mesh_generation.cubed_sphere import init_cube_topo
from pyses.mesh_generation.mesh_definitions import FRONT_FACE
from pyses.mesh_generation.element_local_metric import init_quasi_uniform_grid_elem_local

NX = 6


def test_corners_are_equiangular():
  """Front-face element-corner longitudes form the uniform-in-angle ladder
  -π/4 + k (π/2) / nx (on the front face, lon = α exactly)."""
  _, face_mask, _, _ = init_cube_topo(NX)
  expected = -np.pi / 4 + (np.pi / 2) * np.arange(NX + 1) / NX
  for npt in test_npts:
    grid, _ = init_quasi_uniform_grid_elem_local(NX, npt, wrapped=False)
    latlon = np.asarray(grid["physical_coords"])
    corner_lons = latlon[face_mask == FRONT_FACE][:, [0, -1], :, 1][:, :, [0, -1]]
    corner_lons = np.where(corner_lons > np.pi, corner_lons - 2 * np.pi, corner_lons)
    found = np.unique(np.round(corner_lons, 12))
    assert found.size == NX + 1
    assert np.allclose(found, expected, atol=1e-11)


def test_grid_integrates_sphere_area():
  """Quadrature over the assembled grid recovers 4π to spectral accuracy."""
  for npt in test_npts:
    grid, dims = init_quasi_uniform_grid_elem_local(NX, npt, wrapped=False)
    metdet = np.asarray(grid["metric_determinant"])
    w = np.asarray(grid["gll_weights"])
    area = np.sum(metdet * w[None, :, None] * w[None, None, :])
    # tolerance set by the elem-local metric's discretization error at
    # nx=6, npt=3 (measured 1.1e-4); decays rapidly in both npt and nx
    assert abs(area - 4 * np.pi) / (4 * np.pi) < 5e-4
    assert dims["num_elem"] == 6 * NX**2
