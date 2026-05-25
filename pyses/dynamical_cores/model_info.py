from enum import Enum

models = Enum('dynamical_core',
              [("cam_se", 1),
               ("cam_se_whole_atmosphere", 2),
               ("homme_hydrostatic", 3),
               ("homme_nonhydrostatic", 4),
               ("homme_nonhydrostatic_deep", 5),
               ("homme_quasi_hydrostatic", 6),
               ("homme_hydrostatic_f_plane", 7),
               ("homme_nonhydrostatic_f_plane", 8),
               ("shallow_water", 9),
               ("shallow_water_f_plane", 10)])

tracer_schemes = Enum('tracer_schemes',
                      [('eulerian_spectral', 1)])

homme_models = (models.homme_hydrostatic,
                models.homme_hydrostatic_f_plane,
                models.homme_nonhydrostatic,
                models.homme_quasi_hydrostatic,
                models.homme_nonhydrostatic_deep,
                models.homme_nonhydrostatic_f_plane)

cam_se_models = (models.cam_se,
                 models.cam_se_whole_atmosphere)

dynamical_cores = (*homme_models,
                   *cam_se_models)

shallow_water_models = (models.shallow_water,
                        models.shallow_water_f_plane)

spherical_models = (models.cam_se,
                    models.cam_se_whole_atmosphere,
                    models.homme_hydrostatic,
                    models.homme_nonhydrostatic,
                    models.homme_quasi_hydrostatic,
                    models.homme_nonhydrostatic_deep,
                    models.shallow_water)

f_plane_models = (models.homme_hydrostatic_f_plane,
                  models.homme_nonhydrostatic_f_plane,
                  models.shallow_water_f_plane)

hydrostatic_models = (models.cam_se,
                      models.cam_se_whole_atmosphere,
                      models.homme_hydrostatic,
                      models.homme_hydrostatic_f_plane,
                      models.shallow_water_f_plane,
                      models.shallow_water)
quasi_hydrostatic_models = (models.homme_quasi_hydrostatic,)

vertically_buoyant_models = (models.homme_nonhydrostatic,
                             models.homme_nonhydrostatic_deep,
                             models.homme_nonhydrostatic_f_plane)

deep_atmosphere_models = (models.homme_nonhydrostatic_deep,
                          models.homme_quasi_hydrostatic)

moist_mixing_ratio_models = (models.homme_hydrostatic,
                             models.homme_nonhydrostatic,
                             models.homme_nonhydrostatic_deep,
                             models.homme_quasi_hydrostatic,
                             models.homme_hydrostatic_f_plane,
                             models.homme_nonhydrostatic_f_plane)

dry_mixing_ratio_models = (models.cam_se,
                           models.cam_se_whole_atmosphere,
                           models.shallow_water,
                           models.shallow_water_f_plane)

variable_kappa_models = (models.cam_se_whole_atmosphere,)

_cam_se_thermo_name = "T"
_homme_thermo_name = "theta_v_d_mass"
_shallow_water_thermo_name = "const"

thermodynamic_variable_names = {models.cam_se: _cam_se_thermo_name,
                                models.cam_se_whole_atmosphere: _cam_se_thermo_name,
                                models.homme_hydrostatic: _homme_thermo_name,
                                models.homme_hydrostatic_f_plane: _homme_thermo_name,
                                models.homme_nonhydrostatic: _homme_thermo_name,
                                models.homme_nonhydrostatic_deep: _homme_thermo_name,
                                models.homme_quasi_hydrostatic: _homme_thermo_name,
                                models.homme_nonhydrostatic_f_plane: _homme_thermo_name,
                                models.shallow_water: _shallow_water_thermo_name,
                                models.shallow_water_f_plane: _shallow_water_thermo_name}
