from src.dynamical_cores.mass_coordinate import (init_vertical_grid,
                                                 surface_mass_to_midlevel_mass,
                                                 surface_mass_to_interface_mass,
                                                 surface_mass_to_d_mass,
                                                 d_mass_to_surface_mass,
                                                 eval_top_interface_mass)
from src._config import get_backend as _get_backend
import numpy as np
from ...test_data.mass_coordinate_grids import cam30
from src.dynamical_cores.model_info import models
_be = _get_backend()
jnp = _be.np


def test_vcoord():
  for model in models:
    v_grid = init_vertical_grid(cam30["hybrid_a_i"],
                                cam30["hybrid_b_i"],
                                cam30["p0"],
                                model)
    assert jnp.allclose(eval_top_interface_mass(v_grid), v_grid["reference_surface_mass"] * v_grid["hybrid_a_i"][0])
    n_test = 12
    ps = ((1 + .05 * jnp.sin(jnp.linspace(0, jnp.pi, n_test))[:, np.newaxis, np.newaxis]) *
          v_grid["reference_surface_mass"])
    p_mid = surface_mass_to_midlevel_mass(ps, v_grid)
    p_int = surface_mass_to_interface_mass(ps, v_grid)
    assert (p_mid.shape == (n_test, 1, 1, 30))
    assert (p_int.shape == (n_test, 1, 1, 31))
    for lev_isobaric in range(0, 12):
      assert (jnp.max(jnp.abs(p_int[0:1, 0, 0, lev_isobaric] -
                              p_int[:, 0, 0, lev_isobaric])) < 1e-8)
      assert (jnp.max(jnp.abs(p_mid[0:1, 0, 0, lev_isobaric] -
                              p_mid[:, 0, 0, lev_isobaric])) < 1e-8)
    for lev_terrain in range(29, 31):
      p_int_scaled = p_int[:, :, :, lev_terrain] / ps
      assert (jnp.max(jnp.abs(p_int_scaled[0:1, 0, 0] -
                              p_int_scaled[:, 0, 0])) < 1e-8)
    analytic_p_mid = 0.5 * (p_int[:, :, :, :-1] + p_int[:, :, :, 1:])
    assert (jnp.allclose(analytic_p_mid, p_mid))
    d_mass = surface_mass_to_d_mass(ps, v_grid)
    surface_mass = d_mass_to_surface_mass(d_mass, v_grid)
    assert jnp.allclose(ps, surface_mass)
