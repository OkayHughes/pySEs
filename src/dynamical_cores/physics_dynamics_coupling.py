from enum import Enum

coupling_types = Enum('phys_dyn_couple_opts',
                      [("lump_all", 1),
                       ("dribble_all", 2),
                       ("lump_tracers_dribble_dynamics", 3),
                       ("none", 4)])
