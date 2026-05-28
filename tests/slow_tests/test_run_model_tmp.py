from ..test_data.mass_coordinate_grids import cam30
from ..context import get_figdir, plot_grid
from pyses._config import get_backend as _get_backend
from pyses.analytic_initialization.hydrostatic_solid_body import (init_solid_body_config, init_solid_body_state)
from pyses.dynamical_cores.run_dycore import init_simulator
from pyses.mesh_generation.equiangular_metric import init_quasi_uniform_grid
from pyses.mesh_generation.element_local_metric import init_stretched_grid_elem_local
from pyses.dynamical_cores.mass_coordinate import init_vertical_grid
from pyses.dynamical_cores.model_info import models, cam_se_models, homme_models
from pyses.dynamical_cores.model_config import init_default_config, hypervis_opts


_be = _get_backend()
jnp = _be.np
get_global_array = _be.get_global_array


def test_theta_steady_state():
  for model in [models.homme_hydrostatic, models.cam_se]:
    npt = 4
    nx = 31
    h_grid, dims = init_quasi_uniform_grid(nx, npt, calc_smooth_tensor=True)
    v_grid = init_vertical_grid(cam30["hybrid_a_i"],
                                cam30["hybrid_b_i"],
                                cam30["p0"],
                                model)

    total_time = (3600.0 * 4.0)
    hv_type = hypervis_opts.variable_resolution
    physics_config, diffusion_config, timestep_config = init_default_config(nx, h_grid, v_grid,
                                                                            dims, model,
                                                                            hypervis_type=hv_type,
                                                                            physics_dt=150.0)
    diffusion_config["nu_top"] = 0.0
    test_config = init_solid_body_config(model_config=physics_config,
                                         lapse=0.0,
                                         mountain_width=3  * jnp.pi / 180,
                                         )
    model_state = init_solid_body_state(h_grid, v_grid, physics_config, test_config, dims, model, mountain=True,
                                        enforce_hydrostatic=True)
    simulator = init_simulator(h_grid, v_grid,
                               physics_config,
                               diffusion_config,
                               timestep_config,
                               dims,
                               model)

    t = 0.0
    import matplotlib.pyplot as plt
    from os.path import join
    from os import makedirs
    figdir = get_figdir()
    subdir = "acid_test"
    figdir = join(figdir, subdir)
    makedirs(figdir, exist_ok=True)

    ct = 0
    for t, state in simulator(model_state):
      print(t)
      if t > total_time:
        break

      if ct%6 == 0:
        end_state = state["dynamics"]
        ps = v_grid["hybrid_a_i"][0] * v_grid["reference_surface_mass"] + jnp.sum(end_state["d_mass"], axis=-1)
        ps_begin = v_grid["hybrid_a_i"][0] * v_grid["reference_surface_mass"] + jnp.sum(model_state["dynamics"]["d_mass"], axis=-1)
        #thermo = end_state["theta_v_d_mass"][:, :, :, 12] / end_state["d_mass"][:, :, :, 12]
        levels = jnp.linspace(1e5-1, 1e5+1, 11)
        plt.figure()
        plt.tricontourf(get_global_array(h_grid["physical_coords"][:, :, :, 1], dims).flatten(),
                        get_global_array(h_grid["physical_coords"][:, :, :, 0], dims).flatten(),
                        get_global_array(ps, dims).flatten(), levels=levels)
        plt.colorbar()
        #plot_grid(h_grid, plt.gca())
        plt.savefig(f"{figdir}/ps_{model}_tstep={ct}.pdf")
        plt.figure()
        plt.tricontourf(get_global_array(h_grid["physical_coords"][:, :, :, 1], dims).flatten(),
                        get_global_array(h_grid["physical_coords"][:, :, :, 0], dims).flatten(),
                        get_global_array(ps - ps_begin, dims).flatten())
        plt.colorbar()
        #plt.savefig(f"{figdir}/ps_diff_{model}_tstep={ct}.pdf")
        plt.figure()
        #plot_grid(h_grid, plt.gca())
        plt.tricontourf(get_global_array(h_grid["physical_coords"][:, :, :, 1], dims).flatten(),
                        get_global_array(h_grid["physical_coords"][:, :, :, 0], dims).flatten(),
                        get_global_array(end_state["horizontal_wind"][:, :, :, 12, 1], dims).flatten())
        plt.colorbar()
        #plot_grid(h_grid, plt.gca())
        #plt.savefig(f"{figdir}/v_end_{model}_tstep={ct}.pdf")
        levels = jnp.linspace(-1e-2, 1e-2, 11)
        plt.figure()
        plt.tricontourf(get_global_array(h_grid["physical_coords"][:, :, :, 1], dims).flatten(),
                        get_global_array(h_grid["physical_coords"][:, :, :, 0], dims).flatten(),
                        get_global_array(jnp.mean(end_state["horizontal_wind"][:, :, :, :, 0], axis=-1), dims).flatten(), levels=levels)
        plt.colorbar()
        #plot_grid(h_grid, plt.gca())
        plt.savefig(f"{figdir}/u_end_{model}_tstep={str(ct).zfill(4)}.pdf")
        # plt.figure()
        # plt.tricontourf(get_global_array(h_grid["physical_coords"][:, :, :, 1], dims).flatten(),
        #                 get_global_array(h_grid["physical_coords"][:, :, :, 0], dims).flatten(),
        #                 get_global_array(jnp.max(end_state["w_i"], axis=-1), dims).flatten())
        # plt.colorbar()
        #plot_grid(h_grid, plt.gca())
        #plt.savefig(f"{figdir}/w_end_{model}_tstep={ct}.pdf")
        # plt.figure()
        # plt.tricontourf(get_global_array(h_grid["physical_coords"][:, :, :, 1], dims).flatten(),
        #                 get_global_array(h_grid["physical_coords"][:, :, :, 0], dims).flatten(),
        #                 get_global_array(thermo, dims).flatten())
        # plt.colorbar()
        #plot_grid(h_grid, plt.gca())
        #plt.savefig(f"{figdir}/thermo_end_{model}_tstep={ct}.pdf")
      ct += 1
