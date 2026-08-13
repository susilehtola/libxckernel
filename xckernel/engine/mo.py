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

Sign convention of kappa: codes differ between C exp(-kappa) and C exp(+kappa);
we write C(x) = C exp(sign * kappa) with sign = -+1 the caller's convention.
Only *odd* orders depend on it: the orbital gradient (linear in dP/dx) flips
sign with the convention, as does any third-order orbital derivative (odd
overall through dP dP dP and dP d2P terms) -- orbital_gradient therefore takes
an explicit ``sign``.  At the expansion point x = 0 the *Hessian* is invariant
under kappa -> -kappa (its kernel term is quadratic in dP/dx, and d2P/dx2 is
the even second-order term of exp(sign*kappa)), so orbital_hessian deliberately
takes no sign argument -- both conventions yield the identical matrix (verified
numerically in tests/hessian_validate.py).  Note the convention lives only in
this orbital-rotation layer: the AO-basis derivative tower (F_uv, g_uv,ts,
higher) never involves kappa.
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


def orbital_gradient(F: np.ndarray, C: np.ndarray, nocc: int,
                     occ: float = 2.0, sign: int = -1) -> np.ndarray:
    """XC orbital gradient g[i,a] = dExc/dx_ia at x = 0.

    With C(x) = C exp(sign * kappa) the first-order density response is
    dP/dx_ia = sign * occ (c_a c_i^T + c_i c_a^T), so

        dExc/dx_ia = sign * occ (F~ + F~^T)_ia,   F~ = C^T F C.

    ``sign`` is the caller's exponential convention (-1 for C exp(-kappa),
    +1 for C exp(+kappa)); unlike the Hessian at x = 0, this first-order
    quantity depends on it.
    """
    if sign not in (-1, +1):
        raise ValueError("sign must be -1 or +1")
    Ft = C.T @ F @ C
    return sign * occ * (Ft[:nocc, nocc:] + Ft[nocc:, :nocc].T)


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
