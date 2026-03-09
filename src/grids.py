from .operations_2d import horizontal_grid as _horizontal_grid
from .mesh_generation import equiangular_metric as _equiangular_metric
from .mesh_generation import element_local_metric as _element_local_metric
from .mesh_generation import mesh_io as _mesh_io
from .mesh_generation import periodic_plane as _periodic_plane
init_spectral_element_grid = _horizontal_grid.init_spectral_element_grid


class init:
  init_quasi_uniform_grid = _element_local_metric.init_quasi_uniform_grid_elem_local
  init_equiangular_grid = _equiangular_metric.init_quasi_uniform_grid
  init_stretched_grid = _element_local_metric.init_stretched_grid_elem_local
  init_unstructured_grid = _element_local_metric.init_unstructured_grid
  init_periodic_plane_grid = _periodic_plane.init_periodic_plane
  exodus_to_pysces_grid_corners = _mesh_io.exodus_to_pyses_grid_corners

class parallelism:
  shard_grid = _horizontal_grid.shard_grid
  get_global_grid = _horizontal_grid.get_global_grid
  make_grid_mpi_ready = _horizontal_grid.make_grid_mpi_ready
