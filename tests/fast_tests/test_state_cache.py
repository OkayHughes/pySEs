"""Unit tests for the disk-backed grid/state cache fixture.

Exercises a full save/load roundtrip on a tiny ne3/npt4 cam_se setup
with a hydrostatic-solid-body initial state, plus targeted checks on:

  * cache hits skip the ``build`` callback,
  * a change in ``test_config`` keys a fresh entry (no false hit),
  * non-array leaves (frozendict ``dims``, model enum, scalars) survive
    the roundtrip,
  * the backend snapshot is recorded in the meta file.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
from frozendict import frozendict

from pyses._config import get_backend as _get_backend
from pyses.analytic_initialization.hydrostatic_solid_body import (
    init_solid_body_config, init_solid_body_state)
from pyses.dynamical_cores.mass_coordinate import init_vertical_grid
from pyses.dynamical_cores.model_info import models
from pyses.dynamical_cores.physics_config import init_physics_config
from pyses.mesh_generation.equiangular_metric import init_quasi_uniform_grid
from ..state_cache import default_cache_dir, evict, load, save
from ..test_data.mass_coordinate_grids import cam30

_be = _get_backend()


# -- helpers ----------------------------------------------------------------


def _build_payload(nx=3, npt=4):
  """Tiny ne3/npt4 cam_se setup; cheap enough to rebuild in tests."""
  h_grid, dims = init_quasi_uniform_grid(nx, npt)
  v_grid = init_vertical_grid(cam30["hybrid_a_i"][::4],
                              cam30["hybrid_b_i"][::4],
                              cam30["p0"],
                              models.cam_se)
  physics_config = init_physics_config(models.cam_se)
  test_config = init_solid_body_config(model_config=physics_config, u_max=0.0)
  model_state = init_solid_body_state(h_grid, v_grid, physics_config,
                                      test_config, dims, models.cam_se)
  return {"h_grid": h_grid,
          "v_grid": v_grid,
          "model_state": model_state,
          "dims_inside_payload": dims,
          "model": models.cam_se,
          "scalar_flag": True,
          "scalar_int": 42}, dims, test_config


def _all_equal(a, b):
  """Element-wise equality across arbitrary nested backend arrays."""
  if isinstance(a, dict):
    return set(a) == set(b) and all(_all_equal(a[k], b[k]) for k in a)
  if isinstance(a, (list, tuple)):
    return len(a) == len(b) and all(_all_equal(x, y) for x, y in zip(a, b))
  try:
    a_np = np.asarray(_be.unwrap(a))
    b_np = np.asarray(_be.unwrap(b))
    return a_np.shape == b_np.shape and np.allclose(a_np, b_np, equal_nan=True)
  except (TypeError, ValueError):
    return a == b


@pytest.fixture
def tmp_cache_dir(tmp_path):
  return str(tmp_path / "cache")


# -- tests ------------------------------------------------------------------


def test_roundtrip_preserves_arrays_and_structure(tmp_cache_dir):
  payload, dims, test_config = _build_payload()
  test_id = "solid_body_ne3_npt4_dry"

  npz_path, meta_path = save(tmp_cache_dir, test_id, test_config, dims, payload)
  assert os.path.exists(npz_path)
  assert os.path.exists(meta_path)

  loaded = load(tmp_cache_dir, test_id, test_config, dims)
  assert loaded is not None
  meta = loaded.pop("_meta")
  # Top-level keys preserved.
  assert set(loaded) == set(payload)

  # Per-element grid + state arrays round-trip element-wise.
  for grid_key in ("physical_coords", "metric_determinant", "mass_matrix",
                   "derivative_matrix", "gll_weights"):
    assert _all_equal(loaded["h_grid"][grid_key], payload["h_grid"][grid_key]), \
        f"grid['{grid_key}'] mismatch after roundtrip"
  for dyn_key in payload["model_state"]["dynamics"]:
    assert _all_equal(loaded["model_state"]["dynamics"][dyn_key],
                      payload["model_state"]["dynamics"][dyn_key]), \
        f"dynamics['{dyn_key}'] mismatch after roundtrip"

  # Tuple-valued grid entries (the assembly_triple) survive.
  if "assembly_triple" in payload["h_grid"]:
    assert _all_equal(loaded["h_grid"]["assembly_triple"],
                      payload["h_grid"]["assembly_triple"])

  # Non-array leaves: enum, scalar bool, scalar int, frozendict dims.
  assert loaded["model"] is models.cam_se
  assert loaded["scalar_flag"] is True
  assert loaded["scalar_int"] == 42
  assert isinstance(loaded["dims_inside_payload"], frozendict)
  assert dict(loaded["dims_inside_payload"]) == dict(payload["dims_inside_payload"])

  # Backend snapshot recorded.
  assert "saved_backend" in meta and "loaded_backend" in meta
  assert meta["saved_backend"]["wrapper_type"] == _be.wrapper_type


def test_layout_restoration_matches_active_backend(tmp_cache_dir):
  """Reloaded arrays go through ``_be.array(..., elem_sharding_axis=0)`` for
  per-element fields, so the resulting type matches what a freshly built
  grid produces (the simulator can consume them directly)."""
  payload, dims, test_config = _build_payload()
  test_id = "solid_body_layout"

  save(tmp_cache_dir, test_id, test_config, dims, payload)
  loaded = load(tmp_cache_dir, test_id, test_config, dims)
  loaded.pop("_meta")

  # The sharded element axis (axis 0 of physical_coords) should have the
  # same length as ``dims["num_elem"]`` after restoration.
  pc = loaded["h_grid"]["physical_coords"]
  assert pc.shape[0] == dims["num_elem"]
  # And the restored array should be of the same Python type as a freshly
  # wrapped one (numpy.ndarray under numpy, DeviceArray under JAX, ...).
  freshly_wrapped = _be.array(np.zeros((dims["num_elem"], 1)),
                              elem_sharding_axis=0)
  assert type(pc) is type(freshly_wrapped), (
      f"restored physical_coords is {type(pc).__name__}, "
      f"expected {type(freshly_wrapped).__name__}")


def test_cache_hit_skips_build(tmp_cache_dir):
  """Inline get_or_build to verify the cache-hit short-circuit without
  depending on pytest's session-scoped fixture (whose cache dir we
  cannot retarget after instantiation)."""
  payload, dims, test_config = _build_payload()
  test_id = "solid_body_hit_skip"
  call_count = {"n": 0}

  def get_or_build():
    cached = load(tmp_cache_dir, test_id, test_config, dims)
    if cached is not None:
      cached.pop("_meta", None)
      return cached
    call_count["n"] += 1
    save(tmp_cache_dir, test_id, test_config, dims, payload)
    return payload

  result1 = get_or_build()
  result2 = get_or_build()
  assert call_count["n"] == 1, "second call should be a cache hit"
  assert _all_equal(result1["h_grid"]["physical_coords"],
                    result2["h_grid"]["physical_coords"])


def test_config_change_invalidates_cache(tmp_cache_dir):
  payload, dims, test_config = _build_payload()
  test_id = "solid_body_invalidation"

  save(tmp_cache_dir, test_id, test_config, dims, payload)

  # A different config value -> different key -> cache miss.
  new_config = init_solid_body_config(u_max=10.0)
  miss = load(tmp_cache_dir, test_id, new_config, dims)
  assert miss is None, "different test_config must produce a cache miss"

  # Original key still resolves.
  hit = load(tmp_cache_dir, test_id, test_config, dims)
  assert hit is not None


def test_meta_file_records_backend_snapshot(tmp_cache_dir):
  payload, dims, test_config = _build_payload()
  test_id = "solid_body_meta"
  _, meta_path = save(tmp_cache_dir, test_id, test_config, dims, payload)

  with open(meta_path) as f:
    meta = json.load(f)
  assert meta["backend"]["wrapper_type"] == _be.wrapper_type
  assert meta["backend"]["use_double"] == bool(_be.use_double)
  assert meta["backend"]["num_devices"] == int(_be.num_devices)
  assert meta["backend"]["mpi_size"] == int(_be.mpi_size)
  assert meta["format_version"] == 1


def test_evict_removes_cache_entry(tmp_cache_dir):
  payload, dims, test_config = _build_payload()
  test_id = "solid_body_evict"
  npz_path, meta_path = save(tmp_cache_dir, test_id, test_config, dims, payload)
  assert os.path.exists(npz_path) and os.path.exists(meta_path)
  assert evict(tmp_cache_dir, test_id, test_config, dims)
  assert not os.path.exists(npz_path) and not os.path.exists(meta_path)
  # Idempotent: evict returns False on a missing entry.
  assert not evict(tmp_cache_dir, test_id, test_config, dims)


def test_default_cache_dir_respects_env(monkeypatch):
  monkeypatch.setenv("PYSES_TEST_STATE_CACHE_DIR", "/tmp/some/custom/path")
  assert default_cache_dir() == "/tmp/some/custom/path"
  monkeypatch.delenv("PYSES_TEST_STATE_CACHE_DIR")
  assert default_cache_dir().endswith("pyses/state_cache")


def test_fixture_round_trip(cached_state_factory, monkeypatch, tmp_cache_dir):
  """Top-level integration test using the actual fixture."""
  monkeypatch.setenv("PYSES_TEST_STATE_CACHE_DIR", tmp_cache_dir)
  payload, dims, test_config = _build_payload()
  test_id = "solid_body_fixture_roundtrip"

  build_calls = {"n": 0}

  def build():
    build_calls["n"] += 1
    return payload

  out1 = cached_state_factory(test_id, test_config, dims, build)
  out2 = cached_state_factory(test_id, test_config, dims, build)
  # NB: ``cached_state_factory`` is session-scoped; the env var was only
  # applied for this test, so we can't reliably assert ``build_calls == 1``
  # if a previous test populated the *real* cache dir.  Instead, just
  # verify the payloads are equivalent.
  assert _all_equal(out1["h_grid"]["physical_coords"],
                    out2["h_grid"]["physical_coords"])
  assert _all_equal(out1["model_state"]["dynamics"]["horizontal_wind"],
                    out2["model_state"]["dynamics"]["horizontal_wind"])
