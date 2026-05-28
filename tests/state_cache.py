"""Disk-backed cache for spectral-element grids and atmospheric states.

The cache lets a long-running test skip the (expensive) grid + balanced-
state construction on the second and later invocations.  All arrays are
stored in a single ``.npz`` archive; a sidecar ``.meta.json`` records the
structure (which fields are arrays vs scalars vs nested containers), the
backend that wrote the file (so the loader can restore the proper memory
layout for the *current* backend), and the test identifier / dims /
test-config used to key the entry.

Memory layout
-------------
This module assumes **shared-memory parallelism** (single process, no
MPI ranks splitting the element axis).  Element-sharded layouts under
JAX or torch are handled by re-wrapping each array through
``_be.array(..., elem_sharding_axis=0)`` on load, which restores the
per-device layout the rest of the code expects.

Heuristic for "is this array element-sharded?": ``arr.shape[0] ==
dims["num_elem"]`` (with the fallback to ``dims["num_elem_padded"]``
when that key exists).  This holds for every per-element quantity in
the codebase (grid metric arrays, dynamics fields, tracer fields);
small auxiliaries like ``derivative_matrix`` or ``gll_weights`` are
left un-sharded.
"""
from __future__ import annotations

import hashlib
import json
import os
from enum import Enum
from typing import Any

import numpy as np

from pyses._config import get_backend as _get_backend


# ---------------------------------------------------------------------------
# Constants / paths
# ---------------------------------------------------------------------------

_ENV_CACHE_DIR = "PYSES_TEST_STATE_CACHE_DIR"
_DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/pyses/state_cache")
# Bumped whenever the on-disk format changes incompatibly.
_FORMAT_VERSION = 1


def default_cache_dir() -> str:
  """Directory where ``.npz`` / ``.meta.json`` pairs are stored."""
  return os.environ.get(_ENV_CACHE_DIR, _DEFAULT_CACHE_DIR)


# ---------------------------------------------------------------------------
# Leaf classification
# ---------------------------------------------------------------------------

# Sentinel tags used in the manifest to distinguish leaf kinds.
_TAG_ARRAY = "array"          # numpy-serialized, key into the .npz
_TAG_SCALAR = "scalar"        # Python int / float / bool / str / None
_TAG_TUPLE = "tuple"          # children rebuilt as a Python tuple
_TAG_LIST = "list"            # children rebuilt as a Python list
_TAG_DICT = "dict"            # children rebuilt as a Python dict
_TAG_ENUM = "enum"            # stdlib Enum member, by class+name
_TAG_FROZENDICT = "frozendict"  # serialized as a dict, rebuilt via frozendict


def _is_array_like(obj: Any) -> bool:
  """True if ``obj`` looks like a numpy / jax / torch array."""
  if isinstance(obj, np.ndarray):
    return True
  # JAX / torch all expose ``shape`` + ``dtype`` and survive ``np.asarray``.
  return hasattr(obj, "shape") and hasattr(obj, "dtype") and not isinstance(obj, type)


def _to_numpy(obj: Any) -> np.ndarray:
  """Convert any backend-array (numpy, jax, torch) to a plain numpy array."""
  be = _get_backend()
  return np.asarray(be.unwrap(obj))


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------

def _canonicalize_for_hash(obj: Any) -> Any:
  """Recursively convert ``obj`` into a JSON-safe canonical form.

  Used only to derive the cache key (not for reconstruction), so we can
  freely flatten arrays to lists, sort dict keys, and turn Enums into
  their qualified name.
  """
  if obj is None or isinstance(obj, (bool, int, float, str)):
    return obj
  if isinstance(obj, Enum):
    return f"{type(obj).__module__}.{type(obj).__name__}.{obj.name}"
  if _is_array_like(obj):
    arr = _to_numpy(obj)
    return {"_shape": list(arr.shape),
            "_dtype": str(arr.dtype),
            "_data": arr.tolist()}
  if isinstance(obj, (list, tuple)):
    return [_canonicalize_for_hash(v) for v in obj]
  if hasattr(obj, "items"):
    return {str(k): _canonicalize_for_hash(v) for k, v in sorted(obj.items())}
  return repr(obj)


def cache_key(test_id: str, test_config: dict, dims: Any) -> str:
  """Stable cache key derived from ``test_id``, ``test_config`` and ``dims``.

  The key is a 16-char hex prefix of the SHA-256 of the canonical JSON
  encoding; the human-readable ``test_id`` is preserved as the filename
  prefix so the cache directory is easy to browse.
  """
  payload = {"format_version": _FORMAT_VERSION,
             "test_id": test_id,
             "test_config": _canonicalize_for_hash(test_config),
             "dims": _canonicalize_for_hash(dims)}
  blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
  digest = hashlib.sha256(blob).hexdigest()[:16]
  # Keep the file name filesystem-safe.
  safe_id = "".join(c if c.isalnum() or c in "-_." else "_" for c in test_id)
  return f"{safe_id}-{digest}"


# ---------------------------------------------------------------------------
# Recursive walker (flatten / rebuild)
# ---------------------------------------------------------------------------

def _path_to_key(path: tuple) -> str:
  """Encode a tuple of (str | int) into a slash-separated npz key.

  Integer indices are tagged with ``#`` so the rebuild step can tell
  them apart from string keys (e.g. ``grid/assembly_triple/#1/#0``).
  """
  parts = []
  for p in path:
    if isinstance(p, int):
      parts.append(f"#{p}")
    else:
      # NPZ archive entries are flat names; slashes are fine but avoid
      # the ``#`` prefix collision.
      s = str(p)
      if s.startswith("#"):
        s = "_" + s
      parts.append(s)
  return "/".join(parts) if parts else "_root"


def _flatten(struct: Any, path: tuple = ()) -> tuple[dict, dict]:
  """Walk ``struct`` once; return (arrays_by_key, manifest_node).

  The manifest mirrors the structure of ``struct`` with each leaf
  replaced by a ``{"_tag": ..., ...}`` descriptor that the loader can
  invert.  Array leaves carry an extra ``"key"`` pointing into the
  ``arrays`` dict, plus shape / dtype / sharding-hint metadata.
  """
  arrays: dict[str, np.ndarray] = {}

  def walk(node: Any, p: tuple):
    if node is None or isinstance(node, (bool, int, float, str)):
      return {"_tag": _TAG_SCALAR, "value": node}
    if isinstance(node, Enum):
      # The enum may have been built with the functional API
      # (``models = Enum("dynamical_core", ...)``), in which case
      # ``type(node).__name__`` is ``"dynamical_core"`` rather than the
      # ``models`` binding the rest of the code uses.  Walk the
      # module's namespace to find whichever attribute names point to
      # the enum class so the loader can re-bind reliably.
      import sys as _sys
      mod = _sys.modules.get(type(node).__module__)
      binding = None
      if mod is not None:
        for name in dir(mod):
          if getattr(mod, name, None) is type(node):
            binding = name
            break
      return {"_tag": _TAG_ENUM,
              "module": type(node).__module__,
              "type": type(node).__name__,
              "binding": binding,
              "name": node.name}
    if _is_array_like(node):
      arr = _to_numpy(node)
      key = _path_to_key(p)
      arrays[key] = arr
      return {"_tag": _TAG_ARRAY,
              "key": key,
              "shape": list(arr.shape),
              "dtype": str(arr.dtype)}
    if isinstance(node, tuple):
      return {"_tag": _TAG_TUPLE,
              "children": [walk(v, p + (i,)) for i, v in enumerate(node)]}
    if isinstance(node, list):
      return {"_tag": _TAG_LIST,
              "children": [walk(v, p + (i,)) for i, v in enumerate(node)]}
    if hasattr(node, "items"):
      # Distinguish frozendict from dict so we can rebuild the proper
      # immutable container if it's around.
      is_frozen = type(node).__name__ == "frozendict"
      children: dict[str, Any] = {}
      for k, v in node.items():
        children[str(k)] = walk(v, p + (str(k),))
      return {"_tag": _TAG_FROZENDICT if is_frozen else _TAG_DICT,
              "children": children}
    # Last-resort: try the array conversion; if that fails, dump repr().
    try:
      arr = np.asarray(node)
      key = _path_to_key(p)
      arrays[key] = arr
      return {"_tag": _TAG_ARRAY, "key": key,
              "shape": list(arr.shape), "dtype": str(arr.dtype)}
    except Exception:
      return {"_tag": _TAG_SCALAR, "value": repr(node)}

  manifest = walk(struct, path)
  return arrays, manifest


def _rebuild(manifest: dict, arrays: dict, num_elem: int) -> Any:
  """Invert ``_flatten``: assemble the original nested structure.

  Each array is re-wrapped through the *current* backend's ``array``
  with ``elem_sharding_axis=0`` whenever its leading shape matches
  ``num_elem``.  Smaller auxiliaries (e.g. ``gll_weights``) are wrapped
  without a sharding hint.
  """
  be = _get_backend()

  def restore_array(arr: np.ndarray) -> Any:
    sharded = (arr.ndim > 0 and arr.shape[0] == num_elem)
    return be.array(arr, dtype=arr.dtype,
                    elem_sharding_axis=0 if sharded else None)

  def go(node: Any) -> Any:
    tag = node["_tag"]
    if tag == _TAG_SCALAR:
      return node["value"]
    if tag == _TAG_ENUM:
      import importlib
      mod = importlib.import_module(node["module"])
      # Prefer the explicit module binding recorded at save time, fall
      # back to the type name (for "class Foo(Enum)" enums), and finally
      # to a scan of the module for any Enum class matching the name.
      cls = None
      binding = node.get("binding")
      if binding:
        cls = getattr(mod, binding, None)
      if cls is None:
        cls = getattr(mod, node["type"], None)
      if cls is None:
        for name in dir(mod):
          candidate = getattr(mod, name, None)
          if (isinstance(candidate, type) and issubclass(candidate, Enum)
              and candidate.__name__ == node["type"]):
            cls = candidate
            break
      if cls is None:
        raise LookupError(
            f"could not resolve enum {node['module']}.{node['type']}")
      return cls[node["name"]]
    if tag == _TAG_ARRAY:
      return restore_array(arrays[node["key"]])
    if tag == _TAG_TUPLE:
      return tuple(go(c) for c in node["children"])
    if tag == _TAG_LIST:
      return [go(c) for c in node["children"]]
    if tag == _TAG_DICT:
      return {k: go(v) for k, v in node["children"].items()}
    if tag == _TAG_FROZENDICT:
      from frozendict import frozendict
      return frozendict({k: go(v) for k, v in node["children"].items()})
    raise ValueError(f"unknown manifest tag: {tag!r}")

  return go(manifest)


# ---------------------------------------------------------------------------
# Backend snapshot
# ---------------------------------------------------------------------------

def _backend_snapshot() -> dict:
  """Capture the active backend's identity / device layout."""
  be = _get_backend()
  return {"wrapper_type": getattr(be, "wrapper_type", "numpy"),
          "use_double": bool(getattr(be, "use_double", True)),
          "do_sharding": bool(getattr(be, "do_sharding", False)),
          "num_devices": int(getattr(be, "num_devices", 1)),
          "mpi_size": int(getattr(be, "mpi_size", 1))}


# ---------------------------------------------------------------------------
# Top-level save / load
# ---------------------------------------------------------------------------

def cache_paths(cache_dir: str, key: str) -> tuple[str, str]:
  os.makedirs(cache_dir, exist_ok=True)
  return (os.path.join(cache_dir, f"{key}.npz"),
          os.path.join(cache_dir, f"{key}.meta.json"))


def save(cache_dir: str,
         test_id: str,
         test_config: dict,
         dims: Any,
         payload: dict,
         extra_meta: dict | None = None) -> tuple[str, str]:
  """Serialize ``payload`` (e.g. ``{"grid": ..., "state": ...}``) to disk.

  Returns ``(npz_path, meta_path)``.  The npz holds every array leaf
  (keyed by its dotted path); the meta json holds the manifest, the
  cache key inputs (so a stale file can be detected and rebuilt), and
  the backend snapshot.
  """
  key = cache_key(test_id, test_config, dims)
  npz_path, meta_path = cache_paths(cache_dir, key)

  arrays, manifest = _flatten(payload)
  meta = {"format_version": _FORMAT_VERSION,
          "key": key,
          "test_id": test_id,
          "test_config": _canonicalize_for_hash(test_config),
          "dims": _canonicalize_for_hash(dims),
          "backend": _backend_snapshot(),
          "manifest": manifest}
  if extra_meta:
    meta["extra"] = extra_meta

  # Write atomically: build a temp file then rename.  ``np.savez``
  # otherwise appends ``.npz`` to whatever name we hand it, so we open
  # the temp file ourselves and pass the handle in.
  tmp_npz = npz_path + ".tmp"
  tmp_meta = meta_path + ".tmp"
  with open(tmp_npz, "wb") as f:
    np.savez(f, **arrays)
  with open(tmp_meta, "w") as f:
    json.dump(meta, f, sort_keys=True)
  os.replace(tmp_npz, npz_path)
  os.replace(tmp_meta, meta_path)
  return npz_path, meta_path


def load(cache_dir: str,
         test_id: str,
         test_config: dict,
         dims: Any) -> dict | None:
  """Load a previously saved payload.  Returns ``None`` on cache miss.

  A "miss" means: no file exists for this key, or the file is from an
  older ``_FORMAT_VERSION``, or the recorded key parameters disagree
  with the requested ones (which can happen if a hash collision strips
  the truncated key, though this is astronomically rare).
  """
  key = cache_key(test_id, test_config, dims)
  npz_path, meta_path = cache_paths(cache_dir, key)
  if not (os.path.exists(npz_path) and os.path.exists(meta_path)):
    return None
  try:
    with open(meta_path) as f:
      meta = json.load(f)
  except (OSError, json.JSONDecodeError):
    return None
  if meta.get("format_version") != _FORMAT_VERSION:
    return None
  if meta.get("key") != key:
    return None

  # Load every array eagerly into a plain dict (np.savez yields a lazy
  # NpzFile that needs to outlive the consumer otherwise).
  with np.load(npz_path) as data:
    arrays = {k: data[k] for k in data.files}

  num_elem = int(meta["dims"]["num_elem"])
  payload = _rebuild(meta["manifest"], arrays, num_elem)
  payload.setdefault("_meta", {})
  payload["_meta"]["saved_backend"] = meta["backend"]
  payload["_meta"]["loaded_backend"] = _backend_snapshot()
  return payload


def evict(cache_dir: str, test_id: str, test_config: dict, dims: Any) -> bool:
  """Delete a cache entry; return True if a file was removed."""
  key = cache_key(test_id, test_config, dims)
  npz_path, meta_path = cache_paths(cache_dir, key)
  removed = False
  for p in (npz_path, meta_path):
    if os.path.exists(p):
      os.remove(p)
      removed = True
  return removed
