"""Top-level pytest fixtures shared across the whole test tree."""
from __future__ import annotations

import pytest

from .state_cache import default_cache_dir, load, save


@pytest.fixture(scope="session")
def cached_state_factory():
  """Disk-backed cache for expensive (grid + state) test fixtures.

  Use:

      def test_thing(cached_state_factory, ...):
          dims = ...                  # frozendict from grid construction
          test_config = {...}         # any JSON-canonicalisable dict
          payload = cached_state_factory(
              test_id="solid_body_ne15_npt4",
              test_config=test_config,
              dims=dims,
              build=lambda: {"h_grid": h_grid,
                             "v_grid": v_grid,
                             "model_state": model_state})

  On the first run the ``build`` callback is invoked and the returned
  dict is serialised under
  ``$PYSES_TEST_STATE_CACHE_DIR/<test_id>-<hash>.npz`` (+ ``.meta.json``);
  on subsequent runs the cached arrays are restored and re-wrapped
  through the active backend so the per-device memory layout (e.g. JAX
  sharding axes, torch device placement) matches what the simulator
  expects.

  The cache key folds ``test_id``, the canonical encoding of
  ``test_config`` and ``dims`` -- if any of them changes the cache
  misses and ``build`` is re-invoked.

  The factory is **session-scoped** so the cache directory is created
  once per ``pytest`` invocation; the underlying disk cache outlives
  the session (use :func:`tests.state_cache.evict` to drop entries).
  """
  cache_dir = default_cache_dir()

  def get_or_build(test_id, test_config, dims, build):
    payload = load(cache_dir, test_id, test_config, dims)
    if payload is not None:
      # Strip the bookkeeping field so the caller only sees what they saved.
      payload.pop("_meta", None)
      return payload
    payload = build()
    if not isinstance(payload, dict):
      raise TypeError(
          f"build() must return a dict (got {type(payload).__name__}); "
          "wrap your grid / state objects in named keys, e.g. "
          "{'h_grid': h_grid, 'v_grid': v_grid, 'model_state': model_state}")
    save(cache_dir, test_id, test_config, dims, payload)
    return payload

  return get_or_build
