from pyses.dynamical_cores.homme.implicit_terms import (solve_strict_diag_dominant_tridiag,
                                                        calc_implicit_update,
                                                        take_limited_step,
                                                        init_search_dir,
                                                        advance_buoyancy_explicit,
                                                        calc_dirk_jacobian)
from pyses._config import get_backend as _get_backend
import numpy as np
_be = _get_backend()
jnp = _be.np


def test_calc_dirk_jacobian():
  pass


def test_advance_buoyancy_explicit():
  pass


def test_init_search_dir():
  pass


def test_take_limited_step():
  pass


def test_implicit_update_nocrash():
  pass


def test_solve_tridiagonal():
  def test_soln(jac_L, jac_D, jac_U, rhs, soln):
    diag_contrib = jac_D * soln
    lower_contrib = jnp.concatenate((jnp.zeros_like(rhs[:, :, :, -1:]),
                                     jac_L * soln[:, :, :, :-1]),
                                    axis=-1)
    upper_contrib = jnp.concatenate((jac_U * soln[:, :, :, 1:],
                                     jnp.zeros_like(rhs[:, :, :, -1:])),
                                    axis=-1)
    assert jnp.allclose(rhs, diag_contrib + lower_contrib + upper_contrib)
  for lower_val, upper_val in [(0.0, 0.01), (0.01, 0.0)]:
    print(lower_val)
    for diag_val in range(1, 10):
      jac_L = lower_val * jnp.array(np.minimum(0.2, np.random.uniform(size=(10, 4, 4, 29))))
      jac_U = upper_val * jnp.array(np.minimum(0.2, np.random.uniform(size=(10, 4, 4, 29))))
      jac_D = diag_val * jnp.array(np.minimum(0.2, np.random.uniform(size=(10, 4, 4, 30))))
      rhs = jnp.array(np.random.normal(scale=0.05, loc=1.0, size=(10, 4, 4, 30)))
      soln = solve_strict_diag_dominant_tridiag(jac_L,
                                                jac_D,
                                                jac_U,
                                                rhs)
      test_soln(jac_L, jac_D, jac_U, rhs, soln)
    
