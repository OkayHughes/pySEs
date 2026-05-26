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
