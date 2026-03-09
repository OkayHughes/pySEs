from src._config import get_backend as _get_backend
_be = _get_backend()
import numpy as np


def tensor_hypervis_ref(met_inv, jac_inv, hypervis_scaling=3.2):
  visc_tensor = np.zeros((*met_inv.shape[:3], 2, 2))
  # matricies for tensor hyper-viscosity
  for f in range(met_inv.shape[0]):
    for i in range(met_inv.shape[1]):
      for j in range(met_inv.shape[2]):
        eig = np.zeros((2,))
        M = met_inv[f, i, j, :, :]
        # compute eigenvectors of metinv (probably same as computed above)
        eig[0] = (M[0, 0] + M[1, 1] +
                  np.sqrt(4.0 * M[0, 1] * M[1, 0] + (M[0, 0] - M[1, 1])**2)) / 2.0
        eig[1] = (M[0, 0] + M[1, 1] - np.sqrt(4.0 * M[0, 1] * M[1, 0] +
                  (M[0, 0] - M[1, 1])**2)) / 2.0

        # use DE to store M - Lambda, to compute eigenvectors
        DE = np.copy(M)
        DE[0, 0] = DE[0, 0] - eig[0]
        DE[1, 1] = DE[1, 1] - eig[0]

        imaxM = np.unravel_index(np.argmax(np.abs(DE)), shape=DE.shape)
        E = np.zeros((2, 2))
        if np.isclose(np.max(np.abs(DE)), 0.0):
            E[0, 0] = 1.0
            E[1, 0] = 0.0
        elif imaxM[0] == 0 and imaxM[1] == 0:
            E[1, 0] = 1.0
            E[0, 0] = -DE[1, 0] / DE[0, 0]
        elif imaxM[0] == 0 and imaxM[1] == 1:
            E[1, 0] = 1.0
            E[0, 0] = -DE[1, 1] / DE[0, 1]
        elif imaxM[0] == 1 and imaxM[1] == 0:
            E[0, 0] = 1.0
            E[1, 0] = -DE[0, 0] / DE[1, 0]
        elif imaxM[0] == 1 and imaxM[1] == 1:
            E[0, 0] = 1.0
            E[1, 0] = -DE[0, 1] / DE[1, 1]

        # the other eigenvector is orthgonal:
        E[0, 1] = -E[1, 0]
        E[1, 1] = E[0, 0]

        # normalize columns
        E[:, 0] = E[:, 0] / np.sqrt(np.sum(E[:, 0] * E[:, 0]))
        E[:, 1] = E[:, 1] / np.sqrt(np.sum(E[:, 1] * E[:, 1]))

        # OBTAINING TENSOR FOR HV:

        # Instead of the traditional scalar Laplace operator \grad \cdot \grad
        # we introduce \grad \cdot V \grad
        # where V = D E LAM LAM^* E^T D^T.
        # Recall (metric_tensor)^{-1}=(D^T D)^{-1} = E LAM E^T.
        # Here, LAM = diag( 4/((np-1)dx)^2 , 4/((np-1)dy)^2 ) = diag(  4/(dx_elem)^2, 4/(dy_elem)^2 )
        # Note that metric tensors and LAM correspondingly are quantities on a unit sphere.

        # This motivates us to use V = D E LAM LAM^* E^T D^T
        # where LAM^* = diag( nu1, nu2 ) where nu1, nu2 are HV coefficients scaled like
        # (dx)^{hv_scaling/2}, (dy)^{hv_scaling/2}.
        # (Halves in powers come from the fact that HV consists of two Laplace iterations.)

        # Originally, we took LAM^* = diag(
        #  1/(eig(1)**(hypervis_scaling/4.0d0))*(rearth**(hypervis_scaling/2.0d0))
        #  1/(eig(2)**(hypervis_scaling/4.0d0))*(rearth**(hypervis_scaling/2.0d0)) ) =
        #  = diag( lamStar1, lamStar2)
        #  \simeq ((np-1)*dx_sphere / 2 )^hv_scaling/2 = SQRT(OPERATOR_HV)
        # because 1/eig(...) \simeq (dx_on_unit_sphere)^2 .
        # Introducing the notation OPERATOR = lamStar^2 is useful for conversion formulas.

        # This leads to the following conversion formula: nu_const is nu used for traditional HV on uniform grids
        # nu_tensor = nu_const * OPERATOR_HV^{-1}, so
        # nu_tensor = nu_const *((np-1)*dx_sphere / 2 )^{ - hv_scaling} or
        # nu_tensor = nu_const *(2/( (np-1) * dx_sphere) )^{hv_scaling} .
        # dx_sphere = 2\pi *rearth/(np-1)/4/NE
        # [nu_tensor] = [meter]^{4-hp_scaling}/[sec]

        # (1) Later developments:
        # Apply tensor V only at the second Laplace iteration. Thus, LAM^* should be scaled as
        # (dx)^{hv_scaling}, (dy)^{hv_scaling},
        # see this code below:
        #          DEL(1:2,1) = (lamStar1**2) *eig(1)*DE(1:2,1)
        #          DEL(1:2,2) = (lamStar2**2) *eig(2)*DE(1:2,2)

        # (2) Later developments:
        # Bringing [nu_tensor] to 1/[sec]:
        # lamStar1 = 1/(eig(1)**(hypervis_scaling/4.0d0)) * (rearth**2.0d0)
        # lamStar2 = 1/(eig(2)**(hypervis_scaling/4.0d0)) * (rearth**2.0d0)
        # OPERATOR_HV = ( (np-1)*dx_unif_sphere / 2 )^{hv_scaling} * rearth^4
        # Conversion formula:
        # nu_tensor = nu_const * OPERATOR_HV^{-1}, so
        # nu_tensor = nu_const *( 2*rearth /((np-1)*dx))^{hv_scaling} * rearth^{-4.0}.

        # For the baseline coefficient nu=1e15 for NE30,
        # nu_tensor=7e-8 (BUT RUN TWICE AS SMALL VALUE FOR NOW) for hv_scaling=3.2
        # and
        # nu_tensor=1.3e-6 for hv_scaling=4.0.

        D = jac_inv[f, i, j, :, :]
        # matrix D*E
        DE[0, 0] = np.sum(D[0, :] * E[:, 0])
        DE[0, 1] = np.sum(D[0, :] * E[:, 1])
        DE[1, 0] = np.sum(D[1, :] * E[:, 0])
        DE[1, 1] = np.sum(D[1, :] * E[:, 1])

        lamStar1 = 1.0 / (eig[0]**(hypervis_scaling / 4.0))  # *(rearth**2.0d0)
        lamStar2 = 1.0 / (eig[1]**(hypervis_scaling / 4.0))  # *(rearth**2.0d0)

        # matrix (DE) * Lam^* * Lam , tensor HV when V is applied at each Laplace calculation
        #  DEL(1:2,1) = lamStar1*eig(1)*DE(1:2,1)
        #  DEL(1:2,2) = lamStar2*eig(2)*DE(1:2,2)

        # matrix (DE) * (Lam^*)^2 * Lam, tensor HV when V is applied only once, at the last Laplace calculation
        # will only work with hyperviscosity, not viscosity
        DEL = np.zeros((2, 2))
        DEL[0:2, 0] = (lamStar1**2) * eig[0] * DE[0:2, 0]
        DEL[0:2, 1] = (lamStar2**2) * eig[1] * DE[0:2, 1]

        # matrix (DE) * Lam^* * Lam  *E^t *D^t or (DE) * (Lam^*)^2 * Lam  *E^t *D^t
        visc_tensor[f, i, j, 0, 0] = np.sum(DEL[0, :] * DE[0, :])
        visc_tensor[f, i, j, 0, 1] = np.sum(DEL[0, :] * DE[1, :])
        visc_tensor[f, i, j, 1, 0] = np.sum(DEL[1, :] * DE[0, :])
        visc_tensor[f, i, j, 1, 1] = np.sum(DEL[1, :] * DE[1, :])

  return visc_tensor
