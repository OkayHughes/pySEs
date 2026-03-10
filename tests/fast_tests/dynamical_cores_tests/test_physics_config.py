from pyses.dynamical_cores.physics_config import init_physics_config, typical_mass_ratios
from pyses.dynamical_cores.model_info import models
from pyses._config import get_backend as _get_backend
import numpy as np
_be = _get_backend()


def test_mass_ratio_consistency():
  for molecule_set in typical_mass_ratios.keys():
    sum = 0.0
    for molecule in typical_mass_ratios[molecule_set].keys():
      sum += typical_mass_ratios[molecule_set][molecule]
    assert np.allclose(sum, 1.0)


def test_config_consistency():
  config_cam_se = init_physics_config(models.cam_se)
  default_Rgas = config_cam_se["dry_air_species_Rgas"]["dry_air"]
  default_cp = config_cam_se["dry_air_species_cp"]["dry_air"]
  config_se_whole_atmosphere = init_physics_config(models.cam_se_whole_atmosphere)
  dry_air_species_Rd = config_se_whole_atmosphere["dry_air_species_Rgas"]
  dry_air_species_cp = config_se_whole_atmosphere["dry_air_species_cp"]
  Rgas = 0.0
  cp = 0.0
  mass_ratios = typical_mass_ratios[frozenset(dry_air_species_Rd.keys())]
  for molecule_name in mass_ratios.keys():
    Rgas += dry_air_species_Rd[molecule_name] * mass_ratios[molecule_name]
    cp += dry_air_species_cp[molecule_name] * mass_ratios[molecule_name]
  assert np.abs(Rgas - default_Rgas) / Rgas < 0.01
  assert np.abs(cp - default_cp) / cp < 0.01
