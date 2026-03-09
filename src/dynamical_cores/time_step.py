from enum import Enum


time_step_options = Enum('time_step', [("Euler", 1),
                                       ("RK3_5STAGE", 2),
                                       ("RK2", 3),
                                       ("SSPRK3", 4)])


stability_info = {time_step_options.RK2: 2.0,
                  time_step_options.Euler: 2.0,
                  time_step_options.RK3_5STAGE: 3.87,
                  time_step_options.SSPRK3: 1.0}
