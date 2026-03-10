from src.dynamical_cores.model_info import (models, moist_mixing_ratio_models, dry_mixing_ratio_models,
                                            f_plane_models, spherical_models,
                                            cam_se_models, homme_models)


def test_all_eqns_classified():
  maybe_all_models = list(moist_mixing_ratio_models) + list(dry_mixing_ratio_models)
  for model in models:
    assert model in maybe_all_models
  maybe_all_models = list(f_plane_models) + list(spherical_models)
  for model in models:
    assert model in maybe_all_models
  maybe_all_models = list(cam_se_models) + list(homme_models)
  for model in models:
    assert model in maybe_all_models
