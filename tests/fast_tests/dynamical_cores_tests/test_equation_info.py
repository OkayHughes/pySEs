import pytest

from pyses.dynamical_cores.model_info import (
    cam_se_models, dry_mixing_ratio_models, f_plane_models, homme_models,
    models, moist_mixing_ratio_models, shallow_water_models, spherical_models)


@pytest.mark.parametrize("model", list(models), ids=lambda m: m.name)
def test_moist_or_dry_classified(model):
  assert model in list(moist_mixing_ratio_models) + list(dry_mixing_ratio_models)


@pytest.mark.parametrize("model", list(models), ids=lambda m: m.name)
def test_geometry_classified(model):
  assert model in list(f_plane_models) + list(spherical_models)


@pytest.mark.parametrize("model", list(models), ids=lambda m: m.name)
def test_core_family_classified(model):
  assert model in (list(cam_se_models)
                   + list(homme_models)
                   + list(shallow_water_models))
