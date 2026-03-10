from .dynamical_cores import initialization as _initialization
from .analytic_initialization import moist_baroclinic_wave as _moist_baroclinic_wave
from .dynamical_cores import model_state as _model_state
from .dynamical_cores.homme import homme_state as _homme_state
from .dynamical_cores.cam_se import se_state as _se_state


class custom_init:
  init_static_forcing = _model_state.init_static_forcing
  init_model_pressure = _initialization.init_model_pressure
  init_model_struct_homme = _homme_state.init_model_struct
  init_model_struct_cam_se = _se_state.init_model_struct


class ullrich_baroclinic_wave:
  init_baroclinic_wave_config = _moist_baroclinic_wave.init_baroclinic_wave_config
  init_baroclinic_wave_state = _moist_baroclinic_wave.init_baroclinic_wave_state
  perturbation_opts = _moist_baroclinic_wave.perturbation_opts
