"""AO->MO contraction helpers: MO-basis kernel and the XC orbital Hessian.

Orbital rotations are parametrized in the standard exponential form,

    C(x) = C exp(-kappa),   kappa = sum_ia x_ia K_ia,
    K_ia = e_a e_i^T - e_i e_a^T   (MO-basis antisymmetric generator),

with i occupied, a virtual, and the density matrix

    P(x) = occ * C_occ(x) C_occ(x)^T .

Because P is *quadratic* in x, the orbital Hessian of the XC energy has two
contributions -- the kernel contracted with two first-order density responses,
and the Fock matrix contracted with the second-order density response:

    H_ia,jb = d2 Exc / dx_ia dx_jb
            = sum_{uvts} g_uv,ts (dP/dx_ia)_uv (dP/dx_jb)_ts
            + sum_{uv}   F_uv    (d2P/dx_ia dx_jb)_uv .

Differentiating P(x) = occ * C U Pi U^T C^T (Pi = occupied projector) at x = 0
gives the closed forms used below:

    dP/dx_ia            = -occ (c_a c_i^T + c_i c_a^T)
    (d2P)_MO/dx_ia dx_jb =  occ [ delta_ij (e_a e_b^T + e_b e_a^T)
                                 - delta_ab (e_i e_j^T + e_j e_i^T) ]

so the Fock term reduces to occ [ delta_ij (F~ + F~^T)_ab
                                 - delta_ab (F~ + F~^T)_ij ],  F~ = C^T F C --
the familiar delta_ij F_ab - delta_ab F_ij structure of orbital Hessians.

This module only supplies the XC contribution; the host code adds the kinetic,
nuclear and Coulomb (and exact-exchange) parts of the Hessian itself.

Sign convention of kappa: codes differ between C exp(-kappa) and C exp(+kappa).
At the expansion point x = 0 the Hessian is *invariant* under kappa -> -kappa
(the kernel term is quadratic in dP/dx, and d2P/dx2 is the even second-order
term of exp(-+kappa)), so orbital_hessian deliberately takes no sign argument --
both conventions yield the identical matrix (verified numerically in
tests/hessian_validate.py).  The convention only matters for odd-order
quantities (e.g. the orbital gradient dExc/dx_ia = -+ 2 occ (C^T F C)_ai) or
away from x = 0; any future gradient helper must expose it.
"""

from __future__ import annotations

import numpy as np


def mo_transform(G: np.ndarray, C1: np.ndarray, C2: np.ndarray,
                 C3: np.ndarray, C4: np.ndarray) -> np.ndarray:
    """Transform the AO kernel g_uv,ts to MO indices: (i a | fxc | j b).

    Each C? is an (nao, n?) coefficient block for the corresponding index; e.g.
    mo_transform(G, Co, Cv, Co, Cv) yields the (ia|fxc|jb) block used in
    TDDFT/CPKS working equations.
    """
    return np.einsum("uvts,ui,va,tj,sb->iajb", G, C1, C2, C3, C4,
                     optimize=True)


def orbital_hessian(F: np.ndarray, G: np.ndarray, C: np.ndarray,
                    nocc: int, occ: float = 2.0) -> np.ndarray:
    """XC orbital Hessian H[i,a,j,b] = d2 Exc / dx_ia dx_jb at x = 0.

    Parameters
    ----------
    F : (nao, nao) XC Fock matrix dExc/dP_uv (symmetric).
    G : (nao, nao, nao, nao) AO XC kernel d2Exc/dP_uv dP_ts.
    C : (nao, nmo) MO coefficients, occupied first.
    nocc : number of occupied orbitals.
    occ : occupation per orbital (2 closed-shell, 1 per spin channel).
    """
    Co, Cv = C[:, :nocc], C[:, nocc:]

    # first-order density response  dP/dx_ia = -occ (c_a c_i^T + c_i c_a^T)
    dP = -occ * (np.einsum("ua,vi->iauv", Cv, Co)
                 + np.einsum("ui,va->iauv", Co, Cv))

    # kernel term (quadratic in dP, so its overall sign drops out)
    H = np.einsum("uvts,iauv,jbts->iajb", G, dP, dP, optimize=True)

    # Fock term from the second-order density response
    Ft = C.T @ F @ C
    Foo = Ft[:nocc, :nocc]
    Fvv = Ft[nocc:, nocc:]
    no, nv = Co.shape[1], Cv.shape[1]
    H += occ * (np.einsum("ij,ab->iajb", np.eye(no), Fvv + Fvv.T)
                - np.einsum("ab,ij->iajb", np.eye(nv), Foo + Foo.T))
    return H
