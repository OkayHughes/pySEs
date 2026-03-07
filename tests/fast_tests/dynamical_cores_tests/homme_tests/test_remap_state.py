from pysces.analytic_initialization.moist_baroclinic_wave import init_baroclinic_wave_config
from pysces.dynamical_cores.physics_config import init_physics_config
from pysces.mesh_generation.equiangular_metric import init_quasi_uniform_grid
from pysces.config import jnp, np, device_wrapper
from ....test_data.mass_coordinate_grids import cam30
from pysces.dynamical_cores.utils_3d import phi_to_g
from pysces.dynamical_cores.mass_coordinate import init_vertical_grid
from pysces.initialization import init_baroclinic_wave_state
from pysces.dynamical_cores.model_state import remap_dynamics
from pysces.model_info import models


def test_remap_state():
  npt = 4
  nx = 5
  h_grid, dims = init_quasi_uniform_grid(nx, npt)
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
