from .shallow_water_models import run_shallow_water as _run_shallow_water
from .shallow_water_models import time_stepping as _time_stepping_sw
from .shallow_water_models import hyperviscosity as _hyperviscosity_sw
from .shallow_water_models import model_state as _model_state_sw
from .shallow_water_models import williamson_init as _williamson_init
from .shallow_water_models import galewsky_init as _galewsky_init

simulate_shallow_water = _run_shallow_water.simulate_shallow_water


class configure:
  init_hypervis_config_quasi_uniform = _hyperviscosity_sw.init_hypervis_config_const
  init_hypervis_config_variable_res = _hyperviscosity_sw.init_hypervis_config_tensor
  init_timestep_config = _time_stepping_sw.init_timestep_config


class model_state:
  wrap_model_state = _model_state_sw.wrap_model_state
  project_model_state = _model_state_sw.project_model_state


class galewsky_init:
  init_galewsky_config = _galewsky_init.init_galewsky_config
  eval_galewky_wind = _galewsky_init.eval_galewsky_wind
  eval_galewky_h = _galewsky_init.eval_galewsky_h
  eval_galewky_hs = _galewsky_init.eval_galewsky_hs


class williamson_init:
  init_williamson_steady_config = _williamson_init.init_williamson_steady_config
  init_williamson_tc2_wind = _williamson_init.eval_williamson_tc2_u
  init_williamson_tc2_h = _williamson_init.eval_williamson_tc2_h
  init_williamson_tc2_hs = _williamson_init.eval_williamson_tc2_hs
