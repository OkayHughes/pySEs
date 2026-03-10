from src._config import get_backend as _get_backend
import numpy as np
_be = _get_backend()
jnp = _be.np
device_wrapper = _be.array


cam30 = {"hybrid_a_i": device_wrapper(np.array([0.00225523952394724, 0.00503169186413288, 0.0101579474285245,
                                                0.0185553170740604, 0.0306691229343414, 0.0458674766123295,
                                                0.0633234828710556, 0.0807014182209969, 0.0949410423636436,
                                                0.11169321089983, 0.131401270627975, 0.154586806893349,
                                                0.181863352656364, 0.17459799349308, 0.166050657629967,
                                                0.155995160341263, 0.14416541159153, 0.130248308181763,
                                                0.113875567913055, 0.0946138575673103, 0.0753444507718086,
                                                0.0576589405536652, 0.0427346378564835, 0.0316426791250706,
                                                0.0252212174236774, 0.0191967375576496, 0.0136180268600583,
                                                0.00853108894079924, 0.00397881818935275, 0, 0])),
         "hybrid_b_i": device_wrapper(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0393548272550106,
                                                0.0856537595391273, 0.140122056007385, 0.204201176762581,
                                                0.279586911201477, 0.368274360895157, 0.47261056303978,
                                                0.576988518238068, 0.672786951065063, 0.753628432750702,
                                                0.813710987567902, 0.848494648933411, 0.881127893924713,
                                                0.911346435546875, 0.938901245594025, 0.963559806346893,
                                                0.985112190246582, 1])),
         "p0": device_wrapper(1e5)}


def vertical_grid_finite_diff(nlev, eta_top=0.2, p0=1e5, frac_isobaric=0.4):
  eta_i = jnp.linspace(eta_top, 1.0, nlev + 1)
  num_isobaric = int(frac_isobaric * nlev)
  hybrid_b_i = np.zeros_like(eta_i)
  hybrid_b_i[num_isobaric:] = np.linspace(0, 1.0, len(hybrid_b_i[num_isobaric:]))
  hybrid_a_i = eta_i - device_wrapper(hybrid_b_i)
  return {"hybrid_a_i": device_wrapper(hybrid_a_i),
          "hybrid_b_i": device_wrapper(hybrid_b_i),
          "p0": device_wrapper(p0)}
