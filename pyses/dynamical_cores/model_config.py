from .physics_config import init_physics_config
from .hyperviscosity import init_hypervis_config_stub, init_hypervis_config_tensor, init_hypervis_config_const
from .time_stepping import init_timestep_config
from math import floor, log10
from enum import Enum


hypervis_opts = Enum('hypervis_options',
                     [("variable_resolution", 1),
                      ("quasi_uniform", 2),
                      ("none", 3)])


def init_default_config(nx,
                        h_grid,
                        v_grid,
                        dims,
                        model,
                        physics_dt=-1.0,
                        hypervis_type=hypervis_opts.variable_resolution):
  """
  Build physics, diffusion, and time-stepping configs with sensible defaults.

  The coupling timestep is estimated as ``900 * (30 / nx)`` seconds (rounded
  to two significant figures) when ``physics_dt < 0``.  The hyperviscosity
  configuration is selected by ``hypervis_type``.

  Parameters
  ----------
  nx : int
      Number of elements per cube face (cubed-sphere refinement level); used
      to derive the default coupling timestep.
  h_grid : SpectralElementGrid
      Horizontal grid struct; forwarded to hyperviscosity and time-step
      config initializers.
  v_grid : dict[str, Array]
      Vertical grid struct from :func:`init_vertical_grid`.
  dims : frozendict[str, int]
      Grid dimension tuple forwarded to config initializers.
  model : model_info.models
      Model identifier forwarded to all sub-config initializers.
  physics_dt : float, optional
      Coupling interval in seconds.  If negative (default: ``-1.0``) the
      value is derived automatically from ``nx``.
  hypervis_type : hypervis_opts, optional
      Hyperviscosity variant.  ``variable_resolution`` (default) uses the
      tensor formulation; ``quasi_uniform`` uses the constant-coefficient
      formulation; ``none`` disables diffusion.

  Returns
  -------
  physics_config : dict
      Physics configuration from :func:`init_physics_config`.
  diffusion_config : dict
      Hyperviscosity/diffusion configuration.
  timestep_config : frozendict
      Time-stepping configuration from :func:`init_timestep_config`.
  """
  if physics_dt < 0:
    physics_dt = 900.0 * (30.0 / nx)
    physics_dt = round(physics_dt, -(int(floor(log10(abs(physics_dt)))) - 1))
  physics_config = init_physics_config(model)
  if hypervis_type is hypervis_opts.variable_resolution:
    diffusion_config = init_hypervis_config_tensor(h_grid, v_grid, dims, physics_config)
  elif hypervis_type is hypervis_opts.quasi_uniform:
    diffusion_config = init_hypervis_config_const(nx, physics_config, v_grid)
  elif hypervis_type is hypervis_opts.none:
    diffusion_config = init_hypervis_config_stub()
  timestep_config = init_timestep_config(physics_dt,
                                         h_grid,
                                         physics_config,
                                         diffusion_config,
                                         dims,
                                         model)
  return physics_config, diffusion_config, timestep_config
