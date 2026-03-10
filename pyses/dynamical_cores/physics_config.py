from .._config import get_backend as _get_backend
from .model_info import cam_se_models, variable_kappa_models
_be = _get_backend()
device_wrapper = _be.array


boltzmann = 1.38065e-23
avogadro = 6.02214e26

universal_R = boltzmann * avogadro

molec_weight_dry_air = 28.966
molec_weight_water_vapor = 18.016

degrees_of_freedom_igl = {1: 3,
                          2: 5,
                          3: 6}

# Not physically realistic, but achieves close Rgas, cp equivalence
typical_mass_ratios = {frozenset(["N2", "O2"]): {"O2": 0.26,
                                                 "N2": 0.74}}

gas_properties = {"N2": {"num_atoms": 2,
                         "molecular_weight": 28},
                  "O2": {"num_atoms": 2,
                         "molecular_weight": 32},
                  "Ar": {"num_atoms": 1,
                         "molecular_weight": 40}}


def cp_base(dof):
  """
  Compute the molar heat capacity at constant pressure for an ideal gas.

  Uses the equipartition theorem: ``cp = R * (1 + f/2)`` where ``f`` is the
  number of active degrees of freedom for a molecule with ``dof`` atoms.

  Parameters
  ----------
  dof : int
      Number of atoms in the molecule (1, 2, or 3); used to look up the
      degrees of freedom from ``degrees_of_freedom_igl``.

  Returns
  -------
  cp_mol : float
      Molar heat capacity at constant pressure (J mol^-1 K^-1).
  """
  return universal_R * (1.0 + degrees_of_freedom_igl[dof] / 2.0)


def init_physics_config(model,
                        Rgas=universal_R / molec_weight_dry_air,
                        radius_earth=6371e3,
                        angular_freq_earth=7.292e-5,
                        gravity=9.81,
                        p0=1e5,
                        cp=1.00464e3,
                        cp_water_vapor=1.810e3,
                        R_water_vapor=universal_R / molec_weight_water_vapor,
                        epsilon=molec_weight_water_vapor / molec_weight_dry_air,
                        dry_air_species=["N2", "O2"]):
  """
  Build the physics configuration dict for a 3-D atmospheric model.

  For CAM-SE models the dict is extended with per-species gas constants and
  heat capacities for the dry-air mixture and for water vapour.  For
  variable-kappa models the dry-air species are treated individually using
  ideal-gas theory; otherwise a single bulk dry-air entry is used.

  Parameters
  ----------
  model : model_info.models
      Model identifier; selects which sub-dicts are populated.
  Rgas : float, optional
      Specific gas constant for dry air (J kg^-1 K^-1); default derived from
      universal gas constant and the molecular weight of dry air (~287 J kg^-1 K^-1).
  radius_earth : float, optional
      Mean Earth radius in metres (default: 6371e3).
  angular_freq_earth : float, optional
      Earth's rotation rate in rad s^-1 (default: 7.292e-5).
  gravity : float, optional
      Surface gravitational acceleration in m s^-2 (default: 9.81).
  p0 : float, optional
      Reference pressure in Pa (default: 1e5).
  cp : float, optional
      Specific heat capacity of dry air at constant pressure in J kg^-1 K^-1
      (default: 1004.64).
  cp_water_vapor : float, optional
      Specific heat capacity of water vapour at constant pressure in
      J kg^-1 K^-1 (default: 1810).
  R_water_vapor : float, optional
      Specific gas constant for water vapour (J kg^-1 K^-1).
  epsilon : float, optional
      Ratio of molecular weight of water vapour to dry air (~0.622).
  dry_air_species : list[str], optional
      Species names for the dry-air mixture (default: ``["N2", "O2"]``);
      used only for variable-kappa CAM-SE models.

  Returns
  -------
  physics_config : dict
      Dict containing ``"gravity"``, ``"radius_earth"``,
      ``"angular_freq_earth"``, ``"p0"``, ``"epsilon"``, ``"Rgas"``,
      ``"cp"``, ``"moisture_species_Rgas"``, ``"moisture_species_cp"``,
      and (for CAM-SE) ``"dry_air_species_Rgas"`` and ``"dry_air_species_cp"``.
  """

  physics_config = {"gravity": device_wrapper(gravity),
                    "radius_earth": device_wrapper(radius_earth),
                    "angular_freq_earth": device_wrapper(angular_freq_earth),
                    "p0": device_wrapper(p0),
                    "epsilon": epsilon}
  physics_config["Rgas"] = device_wrapper(Rgas)
  physics_config["cp"] = device_wrapper(cp)
  if model in cam_se_models:
    if model in variable_kappa_models:
      physics_config["dry_air_species_Rgas"] = {}
      physics_config["dry_air_species_cp"] = {}
      for species in dry_air_species:
         R_species = universal_R / gas_properties[species]["molecular_weight"]
         physics_config["dry_air_species_Rgas"][species] = device_wrapper(R_species)
         cp_species = cp_base(gas_properties[species]["num_atoms"]) / gas_properties[species]["molecular_weight"]
         physics_config["dry_air_species_cp"][species] = device_wrapper(cp_species)
    else:
      physics_config["dry_air_species_Rgas"] = {"dry_air": device_wrapper(Rgas)}
      physics_config["dry_air_species_cp"] = {"dry_air": device_wrapper(cp)}

  physics_config["moisture_species_Rgas"] = {"water_vapor": device_wrapper(R_water_vapor)}
  physics_config["moisture_species_cp"] = {"water_vapor": device_wrapper(cp_water_vapor)}

  return physics_config
