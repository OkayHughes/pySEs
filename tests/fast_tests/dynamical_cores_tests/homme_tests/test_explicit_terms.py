from pysces.mesh_generation.equiangular_metric import init_quasi_uniform_grid
from ....test_data.mass_coordinate_grids import cam30
from pysces.dynamical_cores.model_state import project_dynamics
from pysces.dynamical_cores.mass_coordinate import init_vertical_grid
from pysces.analytic_initialization.moist_baroclinic_wave import init_baroclinic_wave_config
from pysces.config import jnp, device_wrapper, np
from pysces.dynamical_cores.physics_config import init_physics_config
from pysces.dynamical_cores.homme.explicit_terms import eval_energy_quantities
from pysces.operations_2d.operators import inner_product
from pysces.operations_2d.local_assembly import project_scalar
from pysces.initialization import init_baroclinic_wave_state
from pysces.model_info import models


def test_notopo():
  npt = 4
  nx = 5
  model = models.homme_nonhydrostatic
  h_grid, dims = init_quasi_uniform_grid(nx, npt)
  v_grid = init_vertical_grid(cam30["hybrid_a_i"],
                              cam30["hybrid_b_i"],
                              cam30["p0"],
                              model)
  model_config = init_physics_config(model)

  test_config = init_baroclinic_wave_config(model_config=model_config)
  for _ in range(10):
    print("\nInitializing random atmosphere\n" + "=" * 28 + "\n")
    model_state = init_baroclinic_wave_state(h_grid, v_grid, model_config, test_config, dims, model, mountain=False)
    u_shape = model_state["dynamics"]["horizontal_wind"].shape
    model_state["dynamics"]["horizontal_wind"] += device_wrapper(np.random.normal(scale=0.1, size=u_shape))
    w_i_shape = model_state["dynamics"]["w_i"].shape
    model_state["dynamics"]["w_i"] += device_wrapper(np.random.normal(scale=0.1, size=w_i_shape))
    phi_i_shape = model_state["dynamics"]["phi_i"][:, :, :, :-1].shape
    phi_pert_upper_lev = device_wrapper(np.random.normal(scale=1.0, size=phi_i_shape), elem_sharding_axis=0)
    phi_pert_surf = jnp.zeros_like(model_state["static_forcing"]["phi_surf"])[:, :, :, np.newaxis]
    phi_pert = jnp.concatenate((phi_pert_upper_lev, phi_pert_surf), axis=-1)
    model_state["dynamics"]["phi_i"] += phi_pert
    model_state["dynamics"] = project_dynamics(model_state["dynamics"], h_grid, dims, model)
    pairs, empirical_tendencies = eval_energy_quantities(model_state["dynamics"], model_state["static_forcing"],
                                                         h_grid, v_grid, model_config, dims, model)

    def compare_quantities(a, b, label):
      a_int = inner_product(project_scalar(a, h_grid, dims), jnp.ones_like(a), h_grid)
      b_int = inner_product(project_scalar(b, h_grid, dims), jnp.ones_like(b), h_grid)
      c_int = inner_product(project_scalar(a + b, h_grid, dims), jnp.ones_like(a), h_grid)
      print(f"pair {label}: a_int: {a_int}, b_int: {b_int}, sum: {c_int}")
      #assert (np.abs(c_int) < 1e-8)
    for pair_name in ["ke_ke_1",
                      "ke_ke_2",
                      "ke_ke_3",
                      "ke_ke_4",
                      "ke_ke_5",
                      "ke_ke_6",
                      "ke_pe_1",
                      "pe_pe_1",
                      "ke_ie_1",
                      "ke_ie_2",
                      "ke_ie_3"]:
      compare_quantities(*pairs[pair_name], pair_name)
    total_energy_change = inner_product(project_scalar(empirical_tendencies["ke"] +
                                                       empirical_tendencies["ie"] +
                                                       empirical_tendencies["pe"], h_grid, dims),
                                        jnp.ones_like(empirical_tendencies["ke"]),
                                        h_grid)
    print(f"total energy change: {total_energy_change}")
    #assert (np.abs(total_energy_change) < 1e-8)
