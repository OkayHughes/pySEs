from enum import Enum

models = Enum('dynamical_core',
              [("cam_se", 1),
               ("cam_se_whole_atmosphere", 2),
               ("homme_hydrostatic", 3),
               ("homme_nonhydrostatic", 4),
               ("homme_nonhydrostatic_deep", 5),
               ("homme_hydrostatic_f_plane", 6),
               ("homme_nonhydrostatic_f_plane", 7)])

tracer_schemes = Enum('tracer_schemes',
                      [('eulerian_spectral', 1)])

homme_models = (models.homme_hydrostatic,
                models.homme_hydrostatic_f_plane,
                models.homme_nonhydrostatic,
                models.homme_nonhydrostatic_deep,
                models.homme_nonhydrostatic_f_plane)

cam_se_models = (models.cam_se,
                 models.cam_se_whole_atmosphere)

spherical_models = (models.cam_se,
                    models.cam_se_whole_atmosphere,
                    models.homme_hydrostatic,
                    models.homme_nonhydrostatic,
                    models.homme_nonhydrostatic_deep)

f_plane_models = (models.homme_hydrostatic_f_plane,
                  models.homme_nonhydrostatic_f_plane)

hydrostatic_models = (models.cam_se,
                      models.cam_se_whole_atmosphere,
                      models.homme_hydrostatic,
                      models.homme_hydrostatic_f_plane)


deep_atmosphere_models = (models.homme_nonhydrostatic_deep,)

moist_mixing_ratio_models = (models.homme_hydrostatic,
                             models.homme_nonhydrostatic,
                             models.homme_nonhydrostatic_deep,
                             models.homme_hydrostatic_f_plane,
                             models.homme_nonhydrostatic_f_plane)

dry_mixing_ratio_models = (models.cam_se,
                           models.cam_se_whole_atmosphere)

variable_kappa_models = (models.cam_se_whole_atmosphere,)

_cam_se_thermo_name = "T"
_homme_thermo_name = "theta_v_d_mass"

thermodynamic_variable_names = {models.cam_se: _cam_se_thermo_name,
                                models.cam_se_whole_atmosphere: _cam_se_thermo_name,
                                models.homme_hydrostatic: _homme_thermo_name,
                                models.homme_hydrostatic_f_plane: _homme_thermo_name,
                                models.homme_nonhydrostatic: _homme_thermo_name,
                                models.homme_nonhydrostatic_deep: _homme_thermo_name,
                                models.homme_nonhydrostatic_f_plane: _homme_thermo_name}
