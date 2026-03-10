from pyses._config import get_backend as _get_backend
import numpy as np
_be = _get_backend()


def for_loop_remap(Qdp, dp1, dp2, filter=False):
  tiny = 1e-12
  qsize = Qdp.shape[4]
  nx = Qdp.shape[1]
  nf = Qdp.shape[0]
  nlev = Qdp.shape[3]

  def nlev_fn():
    return np.zeros((nlev,))

  def nlevp_fn():
    return np.zeros((nlev + 1,))

  z1c = nlevp_fn()
  z2c = nlevp_fn()
  z2c = nlevp_fn()
  rhs = nlevp_fn()
  lower_diag = nlevp_fn()
  upper_diag = nlevp_fn()
  q_diag = nlevp_fn()
  diag = nlevp_fn()
  zv = nlevp_fn()
  Qcol = nlev_fn()
  zkr = np.zeros((nlev + 1,), dtype=np.int32)
  dy = nlev_fn()
  filter_code = np.zeros((nlev,), dtype=np.int32)
  Qdp_out = np.zeros_like(Qdp)
  qmax = 1e50
  for f in range(nf):
    for q in range(qsize):
      for i in range(nx):
        for j in range(nx):
          z1c[0] = 0
          z2c[0] = 0
          for k in range(nlev):
            z1c[k + 1] = z1c[k] + dp1[f, i, j, k]
            z2c[k + 1] = z2c[k] + dp2[f, i, j, k]

          zv[0] = 0
          for k in range(nlev):
            Qcol[k] = Qdp[f, i, j, k, q]
            zv[k + 1] = zv[k] + Qcol[k]

          zkr[:] = 99
          ilev = 1
          zkr[0] = 0
          zkr[nlev] = nlev - 1
          for k in range(1, nlev):
            for jk in range(ilev, nlev + 1):
              if z1c[jk] >= z2c[k]:
                ilev = jk
                zkr[k] = jk - 1
                break  # possible jank
          if np.any(zkr == 99):
            print(f"Qdp: {Qdp}")
            print(f"d_mass_model: {dp1}")
            print(f"d_mass_ref: {dp2}")
            print(np.cumsum(dp1, axis=-1))
            print(np.cumsum(dp2, axis=-1))
            print(zkr)
            raise ValueError("data is bad")
          zgam = (z2c - z1c[zkr]) / (z1c[zkr + 1] - z1c[zkr])
          zgam[0] = 0.0
          zgam[nlev] = 1.0

          zhdp = z1c[1:] - z1c[:-1]

          h = 1 / zhdp
          zarg = Qcol * h
          rhs[:] = 0
          lower_diag[:] = 0
          diag[:] = 0
          upper_diag[:] = 0

          rhs[0] = 3 * zarg[0]
          rhs[1:-1] = 3 * (zarg[1:] * h[1:] + zarg[:-1] * h[:-1])
          rhs[nlev] = 3 * zarg[nlev - 1]

          lower_diag[0] = 1
          lower_diag[1:-1] = h[:-1]
          lower_diag[nlev] = 1

          diag[0] = 2
          diag[1:-1] = 2 * (h[1:] + h[:-1])
          diag[nlev] = 2

          upper_diag[0] = 1
          upper_diag[1:-1] = h[1:]
          upper_diag[nlev] = 0

          q_diag[0] = -upper_diag[0] / diag[0]
          rhs[0] = rhs[0] / diag[0]

          for k in range(1, nlev + 1):
            tmp_cal = 1 / (diag[k] + lower_diag[k] * q_diag[k - 1])
            q_diag[k] = -upper_diag[k] * tmp_cal
            rhs[k] = (rhs[k] - lower_diag[k] * rhs[k - 1]) * tmp_cal
          for k in reversed(range(0, nlev)):
            rhs[k] = rhs[k] + q_diag[k] * rhs[k + 1]

          if filter:
            filter_code[:] = 0
            dy[:-1] = zarg[1:] - zarg[:-1]
            dy[nlev - 1] = dy[nlev - 2]

            dy = np.where(np.abs(dy) < tiny, 0.0, dy)

            for k in range(nlev):
              im1 = max(0, k - 1)
              im2 = max(0, k - 2)
              im3 = max(0, k - 3)
              ip1 = min(nlev - 1, k + 1)
              t1 = int(np.where((zarg[k] - rhs[k]) * (rhs[k] - zarg[im1]) >= 0, 1, 0))
              cond1 = dy[im2] * (rhs[k] - zarg[im1]) > 0
              cond2 = dy[im2] * dy[im3] > 0
              cond3 = dy[k] * dy[ip1] > 0
              cond4 = dy[im2] * dy[k] < 0
              t2 = int(np.where(cond1 and cond2 and cond3 and cond4, 1, 0))
              t3 = int(np.where(np.abs(rhs[k] - zarg[im1]) > np.abs(rhs[k] - zarg[k]), 1, 0))

              filter_code[k] = np.where(t1 + t2 > 0, 0, 1)
              rhs[k] = ((1 - filter_code[k]) * rhs[k] + filter_code[k] *
                        (t3 * zarg[k] + (1 - t3) * zarg[im1]))
              filter_code[im1] = max(filter_code[im1], filter_code[k])

            rhs = np.where(rhs > qmax, qmax, rhs)
            rhs = np.where(rhs < 0, 0, rhs)

            za0 = rhs[:-1]
            za1 = -4 * rhs[:-1] - 2 * rhs[1:] + 6 * zarg
            za2 = 3 * rhs[:-1] + 3 * rhs[1:] - 6 * zarg

            dy = rhs[1:] - rhs[:-1]
            dy = np.where(np.abs(dy) < tiny, 0, dy)

            #
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # ! Compute the 3 quadratic spline coeffients {za0, za1, za2}				   !!
            # ! knowing the quadratic spline parameters {rho_left,rho_right,zarg}		   !!
            # ! Zerroukat et.al., Q.J.R. Meteorol. Soc., Vol. 128, pp. 2801-2820 (2002).   !!
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            h = rhs[1:]

            for k in range(nlev):
              xm_d = np.where(np.abs(za2[k]) < tiny, 1.0, 2 * za2[k])
              xm = np.where(np.abs(za2[k]) < tiny, 0, -za1[k] / xm_d)
              f_xm = za0[k] + za1[k] * xm + za2[k] * xm**2

              t1 = int(np.where(np.abs(za2[k]) > tiny, 1, 0))
              t2 = int(np.where(xm <= 0 or xm >= 1, 1, 0))
              t3 = int(np.where(za2[k] > 0, 1, 0))
              t4 = int(np.where(za2[k] < 0, 1, 0))
              tm = int(np.where(t1 * ((1 - t2) + t3) == 2, 1, 0))
              tp = int(np.where(t1 * ((1 - t2) + (1 - t3) + t4) == 3, 1, 0))

              peaks = 0
              peaks = np.where(tm == 1, -1, peaks)
              peaks = np.where(tp == 1, 1, peaks)
              peaks_min = np.where(tm == 1, f_xm, np.minimum(za0[k], za0[k] + za1[k] + za2[k]))
              peaks_max = np.where(tp == 1, f_xm, np.maximum(za0[k], za0[k] + za1[k] + za2[k]))

              im1 = max(0, k - 1)
              im2 = max(0, k - 2)
              ip1 = min(nlev - 1, k + 1)
              ip2 = min(nlev - 1, k + 2)
              cond1 = dy[im2] * dy[im1] <= tiny
              cond2 = dy[ip1] * dy[ip2] <= tiny
              cond3 = dy[im1] * dy[ip1] >= tiny
              cond4 = dy[im1] * float(peaks) <= tiny

              t1 = int(np.where(cond1 or cond2 or cond2 or cond4, np.abs(peaks), 0))
              cond1 = rhs[k] >= qmax
              cond2 = rhs[k] <= 0
              cond3 = peaks_max > qmax
              cond4 = peaks_min < tiny
              filter_code[k] = np.where(cond1 or cond2 or cond3 or cond4, 1, t1 + (1 - t1) * filter_code[k])

              if (filter_code[k] > 0):
                level1 = rhs[k]
                level2 = (2 * rhs[k] + h[k]) / 3
                # level3 = 0.5 * (rhs[k] + h[k])
                level4 = (1 / 3) * rhs[k] + 2 * (1 / 3) * h[k]
                level5 = h[k]

                t1 = int(np.where(h[k] >= rhs[k], 1, 0))
                t2 = int(np.where(zarg[k] <= level1 or zarg[k] >= level5, 1, 0))
                t3 = int(np.where(zarg[k] > level1 and zarg[k] < level2, 1, 0))
                t4 = int(np.where(zarg[k] > level4 and zarg[k] < level5, 1, 0))

                lt1 = t1 * t2
                lt2 = t1 * (1 - t2 + t3)
                lt3 = t1 * (1 - t2 + 1 - t3 + t4)

                za0[k] = np.where(lt1 == 1, zarg[k], za0[k])
                za1[k] = np.where(lt1 == 1, 0, za1[k])
                za2[k] = np.where(lt1 == 1, 0, za2[k])

                za0[k] = np.where(lt2 == 2, rhs[k], za0[k])
                za1[k] = np.where(lt2 == 2, 0, za1[k])
                za2[k] = np.where(lt2 == 2, 3 * (zarg[k] - rhs[k]), za2[k])

                za0[k] = np.where(lt3 == 3, -2 * h[k] + 3 * zarg[k], za0[k])
                za1[k] = np.where(lt3 == 3, 6 * h[k] - 6 * zarg[k], za1[k])
                za2[k] = np.where(lt3 == 3, -3 * h[k] + 3 * zarg[k], za2[k])

                t2 = int(np.where(zarg[k] >= level1 or zarg[k] <= level5, 1, 0))
                t3 = int(np.where(zarg[k] < level1 and zarg[k] > level2, 1, 0))
                t4 = int(np.where(zarg[k] < level4 and zarg[k] > level5, 1, 0))

                lt1 = (1 - t1) * t2
                lt2 = (1 - t1) * (1 - t2 + t3)
                lt3 = (1 - t1) * (1 - t2 + 1 - t3 + t4)

                za0[k] = np.where(lt1 == 1, zarg[k], za0[k])
                za1[k] = np.where(lt1 == 1, 0, za1[k])
                za2[k] = np.where(lt1 == 1, 0, za2[k])

                za0[k] = np.where(lt2 == 2, rhs[k], za0[k])
                za1[k] = np.where(lt2 == 2, 0, za1[k])
                za2[k] = np.where(lt2 == 2, 3 * (zarg[k] - rhs[k]), za2[k])

                za0[k] = np.where(lt3 == 3, -2 * h[k] + 3 * zarg[k], za0[k])
                za1[k] = np.where(lt3 == 3, 6 * h[k] - 6 * zarg[k], za1[k])
                za2[k] = np.where(lt3 == 3, -3 * h[k] + 3 * zarg[k], za2[k])
          else:
            za0 = rhs[:-1]
            za1 = -4 * rhs[:-1] - 2 * rhs[1:] + 6 * zarg
            za2 = 3 * rhs[:-1] + 3 * rhs[1:] - 6 * zarg

          zv1 = 0
          for k in range(nlev):
            if (zgam[k + 1] > 1):
              raise ValueError(f'r not in [0:1] {zgam[k + 1]}')
            zv2 = zv[zkr[k + 1]] + (za0[zkr[k + 1]] * zgam[k + 1] +
                                    (za1[zkr[k + 1]] / 2) * (zgam[k + 1]**2) +
                                    (za2[zkr[k + 1]] / 3) * (zgam[k + 1]**3)) * zhdp[zkr[k + 1]]
            Qdp_out[f, i, j, k, q] = (zv2 - zv1)
            zv1, zv2 = zv2, zv1
  return Qdp_out
