from pyses.analytic_initialization.moist_baroclinic_wave import init_baroclinic_wave_config, init_baroclinic_wave_state
from pyses.dynamical_cores.physics_config import init_physics_config
from ..conftest import cached_quasi_uniform_grid_elem_local
from pyses._config import get_backend as _get_backend
import numpy as np
from ....test_data.mass_coordinate_grids import cam30
from pyses.dynamical_cores.utils_3d import phi_to_g
from pyses.dynamical_cores.mass_coordinate import init_vertical_grid
from pyses.dynamical_cores.model_state import remap_dynamics
from pyses.dynamical_cores.model_info import models
_be = _get_backend()
jnp = _be.np
device_wrapper = _be.array
unwrap = _be.unwrap


def test_remap_state():
  npt = 4
  nx = 5
  h_grid, dims = cached_quasi_uniform_grid_elem_local(nx, npt)
  model = models.homme_nonhydrostatic
  v_grid = init_vertical_grid(cam30["hybrid_a_i"],
                              cam30["hybrid_b_i"],
                              cam30["p0"],
                              model)
  for mountain in [False]:
    model_config = init_physics_config(model)
    test_config = init_baroclinic_wave_config(model_config=model_config)
    model_state = init_baroclinic_wave_state(h_grid, v_grid, model_config, test_config, dims, model,
                                             mountain=mountain)
    dynamics = model_state["dynamics"]
    static_forcing = model_state["static_forcing"]
    u = dynamics["horizontal_wind"]
    w_i = np.random.normal(size=dynamics["w_i"].shape)
    w_i[:, :, :, -1] = ((u[:, :, :, -1, 0] * static_forcing["grad_phi_surf"][:, :, :, 0] +
                         u[:, :, :, -1, 1] * static_forcing["grad_phi_surf"][:, :, :, 1]) /
                        phi_to_g(static_forcing["phi_surf"], model_config, model))
    dynamics["w_i"] = device_wrapper(w_i, elem_sharding_axis=0)
    dynamics_remapped = remap_dynamics(dynamics, static_forcing, v_grid, model_config, len(v_grid["hybrid_a_m"]), model)

    for field in ["horizontal_wind", "theta_v_d_mass",
                  "d_mass", "phi_i",
                  "w_i"]:
      assert(jnp.max(jnp.abs(dynamics[field] - dynamics_remapped[field])) < 1e-5)


def test_remap_w_surface_anchor():
  """Exercise a *non-identity* NH remap and check the w reconstruction.

  ``test_remap_state`` initialises a state that already lives on the reference
  (hybrid) levels, so ``d_mass == d_mass_ref`` and the remap is the identity --
  every field round-trips regardless of how ``w_i`` is reconstructed.  To catch
  the surface-anchor bug in ``remap_dynamics`` (integrating the remapped
  ``w`` increments from the *pre-remap* surface value instead of the
  kinematically consistent post-remap ``w_i_surf``), we must perturb ``d_mass``
  off the hybrid grid so the remap actually moves mass and the lowest-level
  wind -- hence the kinematic surface ``w`` -- genuinely changes.

  The invariant we assert is a consequence of (a) the conservative remap
  preserving the column sum of the layer increments ``dw = interface_to_delta(w_i)``
  and (b) anchoring the reconstruction at ``w_i_surf``:

      w_i[surface] - w_i[top]  ==  sum_k dw[k]  (invariant under a conservative remap)

  With the wrong (pre-remap) anchor the interior column is offset by
  ``w_surf_pre - w_surf_post`` while the surface interface is overwritten with
  ``w_surf_post``, so this top-to-surface difference is corrupted by exactly
  that surface-wind change.
  """
  npt = 4
  nx = 5
  h_grid, dims = cached_quasi_uniform_grid_elem_local(nx, npt)
  model = models.homme_nonhydrostatic
  v_grid = init_vertical_grid(cam30["hybrid_a_i"],
                              cam30["hybrid_b_i"],
                              cam30["p0"],
                              model)
  nlev = len(v_grid["hybrid_a_m"])
  model_config = init_physics_config(model)
  test_config = init_baroclinic_wave_config(model_config=model_config)
  # mountain=True gives a non-zero grad_phi_surf so the kinematic surface w
  # actually depends on the (remapped) surface wind.
  model_state = init_baroclinic_wave_state(h_grid, v_grid, model_config, test_config,
                                           dims, model, mountain=True)
  dynamics = model_state["dynamics"]
  static_forcing = model_state["static_forcing"]

  # A smooth, level-dependent (non-uniform) perturbation of the layer mass
  # pushes the state off the hybrid grid so the remap is non-trivial and the
  # lowest-layer thickness -- and therefore the remapped surface wind -- changes.
  factor = 1.0 + 0.2 * np.cos(np.linspace(0.0, np.pi, nlev))
  factor = factor.reshape((1, 1, 1, nlev))
  # keep theta_v = theta_v_d_mass / d_mass unchanged by scaling both together.
  dynamics["d_mass"] = device_wrapper(np.asarray(unwrap(dynamics["d_mass"])) * factor,
                                      elem_sharding_axis=0)
  dynamics["theta_v_d_mass"] = device_wrapper(np.asarray(unwrap(dynamics["theta_v_d_mass"])) * factor,
                                              elem_sharding_axis=0)

  # Smooth vertical velocity profile, surface pinned to the kinematic BC of the
  # (pre-remap) surface wind.
  u = dynamics["horizontal_wind"]
  w_i = np.tile(np.sin(np.linspace(0.0, np.pi, nlev + 1)).reshape((1, 1, 1, nlev + 1)),
                dynamics["w_i"].shape[:3] + (1,)).astype(np.asarray(unwrap(dynamics["w_i"])).dtype)
  w_i[:, :, :, -1] = np.asarray(unwrap(
      (u[:, :, :, -1, 0] * static_forcing["grad_phi_surf"][:, :, :, 0] +
       u[:, :, :, -1, 1] * static_forcing["grad_phi_surf"][:, :, :, 1]) /
      phi_to_g(static_forcing["phi_surf"], model_config, model)))
  dynamics["w_i"] = device_wrapper(w_i, elem_sharding_axis=0)

  w_i_in = dynamics["w_i"]
  dynamics_remapped = remap_dynamics(dynamics, static_forcing, v_grid, model_config, nlev, model)

  # Guard: the remap must genuinely change the surface wind, otherwise the test
  # cannot distinguish the two anchoring conventions.
  d_u_surf = jnp.max(jnp.abs(dynamics_remapped["horizontal_wind"][:, :, :, -1, :] -
                             u[:, :, :, -1, :]))
  assert d_u_surf > 1e-3, f"remap did not perturb the surface wind (|du|={float(d_u_surf):.2e})"

  # (1) surface interface is pinned to the kinematic BC of the *remapped* wind.
  u_r = dynamics_remapped["horizontal_wind"]
  w_surf_expected = np.asarray(unwrap(
      (u_r[:, :, :, -1, 0] * static_forcing["grad_phi_surf"][:, :, :, 0] +
       u_r[:, :, :, -1, 1] * static_forcing["grad_phi_surf"][:, :, :, 1]) /
      phi_to_g(static_forcing["phi_surf"], model_config, model)))
  assert jnp.max(jnp.abs(dynamics_remapped["w_i"][:, :, :, -1] - w_surf_expected)) < 1e-8

  # (2) the surface-wind change is well above the invariant tolerance, so the
  # buggy (pre-remap) anchor would corrupt the invariant by a measurable amount.
  w_surf_change = jnp.max(jnp.abs(dynamics_remapped["w_i"][:, :, :, -1] - w_i_in[:, :, :, -1]))
  assert w_surf_change > 1e-4, f"surface w change too small to be discriminating ({float(w_surf_change):.2e})"

  # (3) the conservative remap preserves w_i[surface] - w_i[top].  Fails with the
  # pre-remap anchor by exactly the surface-wind change above.
  diff_in = w_i_in[:, :, :, -1] - w_i_in[:, :, :, 0]
  diff_out = dynamics_remapped["w_i"][:, :, :, -1] - dynamics_remapped["w_i"][:, :, :, 0]
  assert jnp.max(jnp.abs(diff_out - diff_in)) < 1e-6
