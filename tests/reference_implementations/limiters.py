from src._config import get_backend as _get_backend
import numpy as np
_be = _get_backend()


def clip_and_sum_limiter_for(tracer_like_tend_in, mass_matrix_in, tracer_min_in, tracer_max_in, d_mass_in):
  tracer_like_tend = np.copy(tracer_like_tend_in)
  mass_matrix = np.copy(mass_matrix_in)
  tracer_max = np.copy(tracer_max_in)
  tracer_min = np.copy(tracer_min_in)
  d_mass = np.copy(d_mass_in)
  npt = tracer_like_tend.shape[0]
  nlev = tracer_like_tend.shape[2]
  x = np.zeros((npt * npt))
  c = np.zeros((npt * npt))
  v = np.zeros((npt * npt))
  tracer_like_tend_out = np.copy(tracer_like_tend)

  for k in range(nlev):
    k1 = 0
    for i in range(npt):
      for j in range(npt):
        c[k1] = mass_matrix[i, j] * d_mass[i, j, k]
        x[k1] = tracer_like_tend[i, j, k] / d_mass[i, j, k]
        k1 += 1
    sumc = np.sum(c)
    mass = np.sum(c * x)
    # this would only fail in very esoteric situations
    assert sumc > 0
    if mass < tracer_min[k] * sumc:
      tracer_min[k] = mass / sumc
    if mass > tracer_max[k] * sumc:
      tracer_max[k] = mass / sumc
    addmass = 0.0
    modified = False

    for k1 in range(npt * npt):
      if x[k1] > tracer_max[k]:
        modified = True
        addmass += (x[k1] - tracer_max[k]) * c[k1]
        x[k1] = tracer_max[k]
      elif x[k1] < tracer_min[k]:
        modified = True
        addmass += (x[k1] - tracer_min[k]) * c[k1]
        x[k1] = tracer_min[k]
    if not modified:
     continue

    if np.abs(addmass) > 0.0:
      if addmass > 0.0:
        v = tracer_max[k] - x
      else:
        v = x - tracer_min[k]
      den = np.sum(v * c)
      if den > 0:
        x += (addmass / den) * v
    k1 = 0
    for i in range(npt):
      for j in range(npt):
        tracer_like_tend_out[i, j, k] = x[k1] * d_mass[i, j, k]
        k1 += 1
  return tracer_like_tend_out


def full_limiter_for(tracer_like_tend_in,
                     mass_matrix_in,
                     tracer_min_in,
                     tracer_max_in,
                     d_mass_in,
                     tol_limiter=1e-10):
  tracer_like_tend = np.copy(tracer_like_tend_in)
  mass_matrix = np.copy(mass_matrix_in)
  tracer_max = np.copy(tracer_max_in)
  tracer_min = np.copy(tracer_min_in)
  d_mass = np.copy(d_mass_in)
  npt = tracer_like_tend.shape[0]
  nlev = tracer_like_tend.shape[2]
  x = np.zeros((npt * npt))
  c = np.zeros((npt * npt))

  tracer_like_tend_out = np.copy(tracer_like_tend)

  for k in range(nlev):

    sumc = 0.0
    mass = 0.0
    k1 = 0
    for i in range(npt):
      for j in range(npt):
        c[k1] = mass_matrix[i, j] * d_mass[i, j, k]
        x[k1] = tracer_like_tend[i, j, k] / d_mass[i, j, k]
        sumc += c[k1]
        mass += c[k1] * x[k1]
        k1 += 1

    assert sumc > 0

    # relax constraints if problem is infeasible
    if mass < tracer_min[k] * sumc:
      tracer_min[k] = mass / sumc

    if mass > tracer_max[k] * sumc:
      tracer_max[k] = mass / sumc

    for iter in range(npt * npt - 1):
      addmass = 0.0

      for k1 in range(npt * npt):
        if x[k1] > tracer_max[k]:
          addmass += (x[k1] - tracer_max[k]) * c[k1]
          x[k1] = tracer_max[k]
        elif x[k1] < tracer_min[k]:
          addmass += (x[k1] - tracer_min[k]) * c[k1]
          x[k1] = tracer_min[k]

      if np.abs(addmass) <= tol_limiter * np.abs(mass):
       break

      weightssum = 0.0
      if addmass > 0.0:
        for k1 in range(npt * npt):
          if x[k1] < tracer_max[k]:
            weightssum += c[k1]
        for k1 in range(npt * npt):
          if x[k1] < tracer_max[k]:
            x[k1] += addmass / weightssum
      else:
        for k1 in range(npt * npt):
          if x[k1] > tracer_min[k]:
            weightssum += c[k1]
        for k1 in range(npt * npt):
          if x[k1] > tracer_min[k]:
            x[k1] += addmass / weightssum
      k1 = 0
      for i in range(npt):
        for j in range(npt):
          tracer_like_tend_out[i, j, k] = x[k1] * d_mass[i, j, k]
          k1 += 1
  return tracer_like_tend_out
