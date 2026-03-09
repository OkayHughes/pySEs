from ._config import get_backend as _get_backend
_be = _get_backend()
from .dynamical_cores import mass_coordinate as _mass_coordinate
from .dynamical_cores import model_state as _model_state
from .dynamical_cores import utils_3d as _utils_3d
from .dynamical_cores import operators_3d as _operators_3d
from .dynamical_cores import model_config as _model_config
from .dynamical_cores import time_stepping as _time_stepping
from .dynamical_cores import physics_config as _physics_config
from .dynamical_cores import hyperviscosity as _hyperviscosity

from .dynamical_cores import model_info


class mass_coordinate:
  init_vertical_grid = _mass_coordinate.init_vertical_grid
  surface_mass_to_interface_mass = _mass_coordinate.surface_mass_to_interface_mass
  surface_mass_to_midlevel_mass = _mass_coordinate.surface_mass_to_midlevel_mass
  d_mass_to_surface_mass = _mass_coordinate.d_mass_to_surface_mass


class model_state:
  advance_dynamics = _model_state.sum_dynamics_series
  advance_tracers = _model_state.advance_tracers

  remap_tracers = _model_state.remap_tracers
  remap_dynamics = _model_state.remap_dynamics

  wrap_dynamics = _model_state.wrap_dynamics
  wrap_static_forcing = _model_state.wrap_static_forcing
  wrap_tracers = _model_state.wrap_tracers
  wrap_model_state = _model_state.wrap_model_state

  copy_dynamics = _model_state.copy_dynamics
  copy_tracers = _model_state.copy_tracers
  copy_model_state = _model_state.copy_model_state

  project_dynamics = _model_state.project_dynamics

  check_dynamics_nan = _model_state.check_dynamics_nan
  check_tracers_nan = _model_state.check_tracers_nan


class model_config:
  init_default_config = _model_config.init_default_config
  hypervis_opts = _model_config.hypervis_opts
  init_physics_config = _physics_config.init_physics_config
  init_timestep_config = _time_stepping.init_timestep_config
  init_diffusion_config = _hyperviscosity.init_hypervis_config_tensor


class parallel_utils:
  get_global_array = _be.get_global_array
  device_wrapper = _be.array
  device_unwrapper = _be.unwrap

class operators:
  project_scalar_3d = _model_state.project_scalar_3d
  horizontal_gradient_3d = _operators_3d.horizontal_gradient_3d
  horizontal_divergence_3d = _operators_3d.horizontal_divergence_3d
  horizontal_vorticity_3d = _operators_3d.horizontal_vorticity_3d
  horizontal_weak_laplacian_3d = _operators_3d.horizontal_weak_laplacian_3d
  horizontal_weak_vector_laplacian_3d = _operators_3d.horizontal_weak_vector_laplacian_3d
