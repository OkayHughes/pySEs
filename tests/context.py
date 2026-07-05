import sys
import os


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def get_figdir(subdir=None):
  """Return ``tests/_figures`` (or a subdirectory thereof), creating it if needed."""
  figdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_figures")
  if subdir is not None:
    figdir = os.path.join(figdir, subdir)
  os.makedirs(figdir, exist_ok=True)
  return figdir


def get_scratch_dir(subdir=None):
  scratchdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_output")
  if scratchdir is not None:
    scratchdir = os.path.join(scratchdir, subdir)
  os.makedirs(scratchdir, exist_ok=True)
  return scratchdir


def emit_plots():
  """True when the ``PYSES_TEST_EMIT_PLOTS`` env var is truthy.

  Test-only switch: when set to ``1``/``true``/``yes``/``on`` (case-insensitive),
  tests may save diagnostic plots under :func:`get_figdir`.  Off by default so
  the standard test run produces no artifacts and does not import matplotlib.
  """
  return os.environ.get("PYSES_TEST_EMIT_PLOTS", "").strip().lower() in (
      "1", "true", "yes", "on")


def plot_scalar_field(lat, lon, values, title, savepath, cmap=None,
                      cbar_label=None, vmin=None, vmax=None, levels=21):
  """``tricontourf`` a scalar field on the cubed sphere and save it as PNG.

  ``lat``, ``lon``, and ``values`` are arrays of identical shape (typically
  ``(elem, npt, npt)``); they are flattened internally.  Coordinates are in
  radians but plotted in degrees so the file is skim-friendly.
  """
  import matplotlib.pyplot as plt
  import numpy as np
  r2d = 180.0 / np.pi
  fig, ax = plt.subplots(figsize=(8.5, 4.2))
  cs = ax.tricontourf(lon.flatten() * r2d, lat.flatten() * r2d,
                      values.flatten(), levels=levels, cmap=cmap,
                      vmin=vmin, vmax=vmax, extend="both")
  cb = fig.colorbar(cs, ax=ax)
  if cbar_label:
    cb.set_label(cbar_label)
  ax.set_xlabel("lon (deg)")
  ax.set_ylabel("lat (deg)")
  ax.set_xlim(0.0, 360.0)
  ax.set_ylim(-90.0, 90.0)
  ax.set_title(title)
  fig.tight_layout()
  fig.savefig(savepath, dpi=120)
  plt.close(fig)


def plot_grid(grid, ax):
  from matplotlib import collections as mc
  import numpy as np
  npt = grid["physical_coords"].shape[1]
  lines = []
  for i_idx in range(npt - 1):
    for j_idx in [0, npt - 1]:
      points_start = zip(grid["physical_coords"][:, i_idx, j_idx, 1].flatten(),
                         grid["physical_coords"][:, i_idx, j_idx, 0].flatten())
      points_end = zip(grid["physical_coords"][:, i_idx + 1, j_idx, 1].flatten(),
                       grid["physical_coords"][:, i_idx + 1, j_idx, 0].flatten())
      lines += zip(points_start, points_end)
  for j_idx in range(npt - 1):
    for i_idx in [0, npt - 1]:
      points_start = zip(grid["physical_coords"][:, i_idx, j_idx, 1].flatten(),
                         grid["physical_coords"][:, i_idx, j_idx, 0].flatten())
      points_end = zip(grid["physical_coords"][:, i_idx, j_idx + 1, 1].flatten(),
                       grid["physical_coords"][:, i_idx, j_idx + 1, 0].flatten())
      lines += zip(points_start, points_end)
  lines = list(filter(lambda line: np.abs(line[1][0] - line[0][0]) < np.pi, lines))
  lc = mc.LineCollection(lines, colors="k", alpha=0.5, linewidths=.05)
  ax.add_collection(lc)


extensive = False
test_division_factor = 1.0 if extensive else 1000.0
test_npts = [3, 4, 5, 6] if extensive else [3, 4]

seed = 0


def allclose_global(sharded_array_1, sharded_array_2, dims):
  from pyses._config import get_backend as _get_backend
  _be = _get_backend()
  return _be.np.allclose(_be.get_global_array(sharded_array_1, dims),
                         _be.get_global_array(sharded_array_2, dims))


def to_host(x, dims=None, elem_sharding_axis=0):
  """Explicit device->host egress of a backend array, returned as plain numpy.

  Use this at the assertion boundary so results can be compared with numpy
  reference functions (``np.allclose``, ``np.sum``, ...).  It makes the
  device->host copy *visible* in the test instead of relying on an implicit
  conversion that only happens to be legal on CPU: the torch backend refuses to
  turn a GPU tensor into numpy without an explicit ``.cpu()`` (which
  :meth:`Backend.unwrap` performs), so ``np.allclose(gpu_tensor, ...)`` raises.

  Pass ``dims`` to also gather sharded shards and strip element padding
  (delegates to :meth:`Backend.get_global_array`); omit it for unsharded or
  already-global arrays (delegates to :meth:`Backend.unwrap`).  Numpy arrays and
  numpy-backend values pass through unchanged.
  """
  from pyses._config import get_backend as _get_backend
  _be = _get_backend()
  if dims is not None:
    return _be.get_global_array(x, dims, elem_sharding_axis)
  return _be.unwrap(x)


def pretty_print_scalar(array, digits=5):
  num_pad = 20
  lines = ["=" * num_pad]
  num_elem = array.shape[0]
  npt = array.shape[1]
  number_format = f":.{digits}e"
  line_format = "[" + ", ".join(["{" + number_format + "}"] * npt) + "]"
  for elem_idx in range(num_elem):
    lines.append(f"Element {elem_idx}")
    for i_idx in range(npt):
      lines.append(line_format.format(*array[elem_idx, i_idx, :]))
  lines.append("=" * num_pad)
  print("\n".join(lines))


def get_data_dir():
  figdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")
  return figdir


def save_state_grids(path, model_state, h_grid, v_grid):
  """Serialize a ``(model_state, h_grid, v_grid)`` triple to a single ``.npz``.

  Reuses the recursive (de)serializer from :mod:`tests.state_cache`: every
  backend array leaf (numpy / jax / torch) is converted to plain numpy and
  stored under its dotted path, while the structure of the nested dicts is
  captured in a JSON manifest stashed inside the same archive.  ``path`` is
  returned (with ``.npz`` appended by ``np.savez`` if it was missing).
  """
  import json
  import numpy as np
  from .state_cache import _flatten
  arrays, manifest = _flatten(
      {"model_state": model_state, "h_grid": h_grid, "v_grid": v_grid})
  # The sharding heuristic on load compares each array's leading axis against
  # the element count; the grid's physical coords carry it verbatim.
  num_elem = int(np.asarray(h_grid["physical_coords"]).shape[0])
  payload = dict(arrays)
  payload["__manifest__"] = np.frombuffer(
      json.dumps(manifest).encode("utf-8"), dtype=np.uint8)
  payload["__num_elem__"] = np.asarray(num_elem)
  np.savez(path, **payload)
  return path


def load_state_grids(path):
  """Inverse of :func:`save_state_grids`; returns ``(model_state, h_grid, v_grid)``.

  Each array leaf is re-wrapped through the *current* backend (restoring
  element sharding where the leading axis matches the element count), so the
  loaded triple is ready to feed back into the model regardless of which
  backend wrote the file.
  """
  import json
  import numpy as np
  from .state_cache import _rebuild
  with np.load(path, allow_pickle=False) as data:
    arrays = {k: data[k] for k in data.files if not k.startswith("__")}
    manifest = json.loads(data["__manifest__"].tobytes().decode("utf-8"))
    num_elem = int(data["__num_elem__"])
  rebuilt = _rebuild(manifest, arrays, num_elem)
  return rebuilt["model_state"], rebuilt["h_grid"], rebuilt["v_grid"]
