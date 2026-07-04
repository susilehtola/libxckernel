"""Layer-2 response algebra: the "MO picture" shared by all response codes.

The six-code survey (docs/dedup-analysis.md) shows every code hand-implements
the same three boundary operations around the XC contraction:

  S1  pack/unpack   occ x vir amplitudes <-> rotation/transition matrices
  S2  forward       amplitudes -> perturbed AO density matrix
  S4  backward      AO Fock-like matrix -> occ x vir gradient block

plus the sigma-vector template (S3) combining them with the orbital-energy
diagonal.  These are provided here with the conventions EXPLICIT -- factor
placement and index order differ between codes (PySCF folds the occupation
into the DM; Psi4 folds a sign into C_right; Dalton multiplies by 2 in
LRAO2MO), and a shared layer must parametrize rather than assume.

Conventions of this module (PySCF-compatible defaults):
  * amplitudes z are (nocc, nvir), occupied index first;
  * transition_dm(z) = occ * C_vir z^T C_occ^T   (the AO transition density
    such that rho^X(r) = occ * sum_ia z_ia phi_i phi_a);
  * project_ov(V) = C_occ^T V^T C_vir            (adjoint of transition_dm:
    project_ov(V)_ia = occ^-1 * dTr[V dm(z)]/dz_ia evaluated without occ);
  * orbital_rotation_dm implements the one-index transform [kappa, D] of the
    idempotent density for full (Z,Y) rotations, with the exponential sign
    convention explicit (see mo.py on why odd orders need it).
"""

from __future__ import annotations

import numpy as np


def transition_dm(z: np.ndarray, Co: np.ndarray, Cv: np.ndarray,
                  occ: float = 2.0) -> np.ndarray:
    """AO transition density matrix from occ x vir amplitudes.

    dm_pq = occ * sum_ia z_ia Cv_pa Co_qi  (PySCF gen_tda_operation layout).
    """
    return occ * (Cv @ z.T @ Co.T)


def project_ov(Vao: np.ndarray, Co: np.ndarray, Cv: np.ndarray) -> np.ndarray:
    """Project an AO Fock-like matrix onto the occ x vir block.

    (Co^T Vao^T Cv), the adjoint of transition_dm without the occupation
    factor -- matches PySCF's einsum('pq,qo,pv->ov', Vao, Co, Cv).
    """
    return Co.T @ Vao.T @ Cv


def orbital_rotation_dm(kappa: np.ndarray, C: np.ndarray, nocc: int,
                        occ: float = 2.0, sign: int = -1) -> np.ndarray:
    """First-order density response to C(x) = C exp(sign * kappa).

    dP = sign * occ * C [kappa, Pi] C^T with Pi the occupied projector; for
    kappa antisymmetric with only ov/vo blocks this is the familiar
    -occ (c_a c_i^T + c_i c_a^T) form of mo.py at sign=-1.
    """
    if sign not in (-1, +1):
        raise ValueError("sign must be -1 or +1")
    Pi = np.zeros((C.shape[1], C.shape[1]))
    Pi[:nocc, :nocc] = np.eye(nocc)
    comm = kappa @ Pi - Pi @ kappa
    return sign * occ * (C @ comm @ C.T)


def tda_sigma(zs: np.ndarray, e_ia: np.ndarray, Co: np.ndarray,
              Cv: np.ndarray, vresp, occ: float = 2.0) -> np.ndarray:
    """TDA sigma vectors  (A z)_ia = e_ia z_ia + project(vresp(dm(z))).

    zs: (nz, nocc, nvir) amplitude batch; vresp: callable mapping a batch of
    AO DMs (nz, nao, nao) to AO response matrices (the Coulomb/exchange/XC
    combination is the host's to assemble -- layer 1 provides the XC part).
    """
    zs = np.asarray(zs)
    dms = np.array([transition_dm(z, Co, Cv, occ) for z in zs])
    v1 = vresp(dms)
    out = np.empty_like(zs)
    for k, z in enumerate(zs):
        out[k] = e_ia * z + project_ov(v1[k], Co, Cv)
    return out
