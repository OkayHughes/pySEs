from pyses.mesh_generation.equiangular_metric import init_quasi_uniform_grid
from pyses.mesh_generation.element_local_metric import (init_quasi_uniform_grid_elem_local,
                                                      init_stretched_grid_elem_local)
from pyses.operations_2d.horizontal_grid import eval_hypervis_tensor, get_global_array, smooth_tensor
from pyses.operations_2d.operators import horizontal_weak_laplacian
from pyses.operations_2d.local_assembly import project_scalar
from pyses._config import get_backend as _get_backend
import numpy as np
from ...context import test_npts
from ...reference_implementations.tensor_hypervis_ref import tensor_hypervis_ref
from scipy.special import sph_harm_y
_be = _get_backend()
jnp = _be.np
device_wrapper = _be.array


def test_hypervisc_tensor():
  for nx in [5, 6]:
    for npt in test_npts:
      grid_equi, _ = init_quasi_uniform_grid(nx, npt, wrapped=False)
      grid_elem_local, _ = init_quasi_uniform_grid_elem_local(nx, npt, wrapped=False)
      for grid in [grid_equi, grid_elem_local]:
        visc_tensor_for = tensor_hypervis_ref(grid["metric_inverse"], grid["contra_to_physical"])
        visc_tensor_operational, _ = eval_hypervis_tensor(grid["metric_inverse"], grid["contra_to_physical"])
        assert np.allclose(visc_tensor_for, visc_tensor_operational)


def test_hypervisc_tensor_algebraic():
  nx = 31
  npt = 4
  grid, dims = init_quasi_uniform_grid_elem_local(nx, npt)
  evals, evecs = jnp.linalg.eigh(grid["metric_inverse"])
  visc_tensor, _ = eval_hypervis_tensor(grid["metric_inverse"], grid["contra_to_physical"], hypervis_scaling=0.0)
  shucked_tensor = jnp.einsum("fijsr,fijmr->fijsm", visc_tensor, grid["physical_to_contra"])
  shucked_tensor = jnp.einsum("fijsm,fijns->fijmn", shucked_tensor, grid["physical_to_contra"])
  assert jnp.max(jnp.abs(get_global_array(shucked_tensor, dims) -
                         get_global_array(grid["metric_inverse"], dims))) < 1e-8

  shucked_tensor = jnp.einsum("fijnm,fijnc->fijmc", shucked_tensor, evecs)
  shucked_tensor = jnp.einsum("fijmc,fijmd->fijdc", shucked_tensor, evecs)
  diag_evals = np.zeros_like(shucked_tensor)
  diag_evals[:, :, :, 0, 0] = evals[:, :, :, 0]
  diag_evals[:, :, :, 1, 1] = evals[:, :, :, 1]
  assert jnp.max(jnp.abs(diag_evals[:dims["num_elem"], :, :, :, :] - get_global_array(shucked_tensor, dims))) < 1e-8


def test_hyperviscosity_sphere_harmonics_mobius():
  nx = 31
  radius_earth = 6371e3
  npt = 4
  nu_const = 1e15
  nu_tensor = 3.4e-8
  grid_squish, dims_squish = init_stretched_grid_elem_local(nx, npt, axis_dilation=np.array([1.0, 2.0, 1.0]))
  grid_uniform, dims = init_quasi_uniform_grid_elem_local(nx, npt)
  for grid, label in zip([grid_uniform, grid_squish], ["uniform", "squish"]):
    grid = smooth_tensor(grid, dims)

    # need to check nu_tensor = 3.4e-8 with tensor HV leads to identical scaling as nu_const = 1e15 on ne30 grid
    # evaluate that hv_conversion = ( (np-1)*dx_unit_sphere / 2 )^{hv_scaling} * rearth^4
    # gives nu_tensor = nu_const * OPERATOR_HV^{-1}

    lat = grid["physical_coords"][:, :, :, 0]
    lon = grid["physical_coords"][:, :, :, 1]
    m = 5
    wavenumber_l = 10
    Ymn = jnp.real(device_wrapper(sph_harm_y(wavenumber_l, m, lat + np.pi / 2.0, lon)))
    biharmonic_Ymn_discont = horizontal_weak_laplacian(-wavenumber_l * (wavenumber_l + 1) * Ymn, grid, a=radius_earth)
    biharmonic_Ymn = project_scalar(biharmonic_Ymn_discont, grid, dims)
    norm_const = (wavenumber_l * (wavenumber_l + 1))**2
    evals, evecs = np.linalg.eigh(grid["metric_inverse"])
    hv_scaling = 3.2
    area_ratio = grid["mass_matrix"] / grid_uniform["mass_matrix"]
    variable_resolution_coefficient = np.sqrt(area_ratio)**(hv_scaling)

    biharmonic_Ymn_discont_tensor = horizontal_weak_laplacian(-wavenumber_l * (wavenumber_l + 1) * Ymn,
                                                              grid,
                                                              a=radius_earth,
                                                              apply_tensor=True)
    biharmonic_Ymn_tensor = project_scalar(biharmonic_Ymn_discont_tensor, grid, dims)
    reference_scaling = get_global_array(nu_const * biharmonic_Ymn / norm_const, dims)
    reference_scaling_tensor = get_global_array(nu_tensor * biharmonic_Ymn_tensor /
                                                norm_const /
                                                variable_resolution_coefficient,
                                                dims)
    biharmonic_Ymn_tensor = get_global_array(biharmonic_Ymn_tensor, dims)
    mask = np.logical_and(jnp.abs(reference_scaling_tensor) > jnp.max(jnp.abs(reference_scaling_tensor)) / 2,
                          jnp.abs(reference_scaling) > jnp.max(jnp.abs(reference_scaling)) / 2)
    ratio = np.log10(np.abs(reference_scaling_tensor[mask] / reference_scaling[mask]))
    assert np.max(np.abs(ratio)) < 1.0

    # TODO [scalars]
    # * Write down what correct scaling and order of application is based on `nu` as
    # it is defined in standard SE dycores for scalars
    # * test hypothesized stability condition on earth-sized sphere
    # * Hypothesize choice of configuration that should render default hyperviscosity and
    # tensor hyperviscosity approximately equivalent on grid.
    # TODO [Vectors]
