from .dynamical_cores import initialization as _initialization
from .analytic_initialization import galewsky_init as _galewsky_init
from .analytic_initialization import moist_baroclinic_wave as _moist_baroclinic_wave
from .analytic_initialization import williamson_init as _williamson_init
from .dynamical_cores import model_state as _model_state
from .dynamical_cores.homme import homme_state as _homme_state
from .dynamical_cores.cam_se import se_state as _se_state


class custom_init:
  init_static_forcing = staticmethod(_model_state.init_static_forcing)
  init_analytic_state = staticmethod(_initialization.init_analytic_state)
  init_model_shallow_water = staticmethod(_initialization.init_model_shallow_water)
  init_model_struct_homme = staticmethod(_homme_state.init_model_struct)
  init_model_struct_cam_se = staticmethod(_se_state.init_model_struct)


class ullrich_baroclinic_wave:
  init_baroclinic_wave_config = staticmethod(_moist_baroclinic_wave.init_baroclinic_wave_config)
  init_baroclinic_wave_state = staticmethod(_moist_baroclinic_wave.init_baroclinic_wave_state)
  perturbation_opts = _moist_baroclinic_wave.perturbation_opts


class williamson_steady_state:
  """Williamson et al. (1992) test case 2 — steady-state geostrophic flow
  on the 3-D shallow-water core."""
  init_williamson_steady_state_config = staticmethod(_williamson_init.init_williamson_steady_state_config)
  init_williamson_steady_state_state = staticmethod(_williamson_init.init_williamson_steady_state_state)
  eval_williamson_tc2_u = staticmethod(_williamson_init.eval_williamson_tc2_u)
  eval_williamson_tc2_h = staticmethod(_williamson_init.eval_williamson_tc2_h)
  eval_williamson_tc2_hs = staticmethod(_williamson_init.eval_williamson_tc2_hs)


class galewsky_barotropic_instability:
  """Galewsky, Scott & Polvani (2004) barotropic-instability test on the
  3-D shallow-water core."""
  init_galewsky_config = staticmethod(_galewsky_init.init_galewsky_config)
  init_galewsky_state = staticmethod(_galewsky_init.init_galewsky_state)
  eval_galewsky_u = staticmethod(_galewsky_init.eval_galewsky_u)
  eval_galewsky_wind = staticmethod(_galewsky_init.eval_galewsky_wind)
  eval_galewsky_h = staticmethod(_galewsky_init.eval_galewsky_h)
  eval_galewsky_hs = staticmethod(_galewsky_init.eval_galewsky_hs)
