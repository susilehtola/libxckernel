"""Grid-field collocation from density matrices over a real basis.

This module is the runtime companion of the generated kernels: it maps
density matrices to the per-point field operands (``rho``, ``grad_rho``,
``sigma``, ``tau``, ``lapl_rho``, ``hess_rho``, ``jp``) that the kernels
take as inputs.  It also settles the complex-coefficient case: with complex
MO coefficients over a REAL basis, no new generated code is needed, because
every field is a LINEAR functional of the density matrix and the bilinear
basis kernels are either symmetric or antisymmetric.

Density-matrix convention (bra-ket): ``rho = sum_uv P_uv chi_u^* chi_v``,
i.e. ``P_uv = sum_i n_i C_ui^* C_vi``, so that matrix elements come out as
``F_uv = dE/dP_uv = <chi_u|F|chi_v>``.  This is the transpose of the
PySCF-style ``C n C^dagger`` -- the two coincide for real symmetric density
matrices, and hosts with the other convention pass ``P.T``.

For a real basis and a Hermitian ``P = S + iA`` (``S`` real symmetric,
``A`` real antisymmetric):

* the symmetric-kernel fields (rho, grad rho, tau, laplacian, Hessian)
  contract ``S = Re P`` only -- ``iA`` is annihilated by symmetry;
* the antisymmetric jp kernel of :mod:`.ingredients` contracts ``A``:
  ``j_p = Im sum_i n_i psi_i^* grad psi_i`` works out to the generated
  kernel evaluated on ``Im P``.

Both channels are therefore served by the REAL generated kernels; the
complex Fock matrix is reassembled as ``F = sym(G) - i asym(G)`` from the
real general output ``G`` of the current-family Fock kernel (for families
without jp dependence, ``asym(G) = 0`` and the Fock matrix is real).

Perturbed (response) density matrices need not be Hermitian and may be
complex; :func:`collocate` applies the literal linear collocation formulas,
so complex input simply yields complex perturbed fields, which may be fed
to the (multi)linear response kernels one real part at a time.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .basis import HESS_COMPS

__all__ = ["collocate", "collocate_pair", "hermitian_fields",
           "hermitian_fock"]


def collocate(P, chi, dchi, lapl_chi=None, hess_chi=None) -> Dict[str, np.ndarray]:
    """Literal linear collocation of the primitive fields from a matrix ``P``.

    This applies the kernel formulas of :mod:`.ingredients` exactly as
    written, with no symmetry assumptions: a general or complex ``P``
    yields general or complex fields (the response use case).  For a
    COMPLEX basis the u-side basis values enter conjugated (sesquilinear
    forms); for a real basis the conjugation is a no-op and the formulas
    are the literal bilinears.  For physical fields of a Hermitian density
    matrix over a real basis use :func:`hermitian_fields`.

    chi: (nbf, ng); dchi: (3, nbf, ng); lapl_chi: (nbf, ng) optional;
    hess_chi: (6, nbf, ng) optional, packed xx,xy,xz,yy,yz,zz.

    Returns a dict with keys ``rho``, ``grad_rho`` (3,ng), ``sigma``,
    ``tau``, ``jp`` (3,ng), and, when the inputs allow, ``lapl_rho`` and
    ``hess_rho`` (6,ng).
    """
    P = np.asarray(P)
    chi_c, dchi_c = np.conj(chi), np.conj(dchi)
    out: Dict[str, np.ndarray] = {}
    out["rho"] = np.einsum("uv,ug,vg->g", P, chi_c, chi)
    out["grad_rho"] = (np.einsum("uv,cug,vg->cg", P, dchi_c, chi)
                       + np.einsum("uv,ug,cvg->cg", P, chi_c, dchi))
    out["sigma"] = np.einsum("cg,cg->g", out["grad_rho"], out["grad_rho"])
    out["tau"] = 0.5 * np.einsum("uv,cug,cvg->g", P, dchi_c, dchi)
    out["jp"] = 0.5 * (np.einsum("uv,ug,cvg->cg", P, chi_c, dchi)
                       - np.einsum("uv,cug,vg->cg", P, dchi_c, chi))
    if lapl_chi is not None:
        out["lapl_rho"] = (np.einsum("uv,ug,vg->g", P, np.conj(lapl_chi), chi)
                           + 2.0 * np.einsum("uv,cug,cvg->g", P, dchi_c, dchi)
                           + np.einsum("uv,ug,vg->g", P, chi_c, lapl_chi))
    if hess_chi is not None:
        hess_c = np.conj(hess_chi)
        comps = []
        for k, (i, j) in enumerate(HESS_COMPS):
            comps.append(np.einsum("uv,ug,vg->g", P, hess_c[k], chi)
                         + np.einsum("uv,ug,vg->g", P, dchi_c[i], dchi[j])
                         + np.einsum("uv,ug,vg->g", P, dchi_c[j], dchi[i])
                         + np.einsum("uv,ug,vg->g", P, chi_c, hess_chi[k]))
        out["hess_rho"] = np.stack(comps)
    return out


def collocate_pair(X, psi_l, dpsi_l, psi_r, dpsi_r) -> Dict[str, np.ndarray]:
    """Perturbed fields of an orbital-pair trial matrix, without AO matrices.

    Computes the fields of the density-matrix direction
    ``D_uv = sum_ij X_ij psi^l_i(u)^* psi^r_j(v)`` directly from
    molecular-orbital values on the grid:
    ``rho^X = sum_ij X_ij psi^l_i{}^* psi^r_j`` and its gradient-,
    tau-, and jp-type companions.  The cost is a handful of rank-k
    matrix products at O(n_orb n_grid); no (nbf, nbf) matrix is formed,
    which is the natural mode for plane-wave and other matrix-free hosts
    (the AO route via :func:`collocate` gives identical fields).

    X: (nl, nr); psi_l: (nl, ng); dpsi_l: (3, nl, ng); psi_r, dpsi_r
    likewise. Returns rho, grad_rho (3,ng), sigma, tau, jp (3,ng).
    """
    X = np.asarray(X)
    Mv = X.T @ np.conj(psi_l)
    Md = np.stack([X.T @ np.conj(dpsi_l[c]) for c in range(3)])
    out: Dict[str, np.ndarray] = {}
    out["rho"] = np.einsum("jg,jg->g", Mv, psi_r)
    out["grad_rho"] = (np.einsum("cjg,jg->cg", Md, psi_r)
                       + np.einsum("jg,cjg->cg", Mv, dpsi_r))
    out["sigma"] = np.einsum("cg,cg->g", out["grad_rho"], out["grad_rho"])
    out["tau"] = 0.5 * np.einsum("cjg,cjg->g", Md, dpsi_r)
    out["jp"] = 0.5 * (np.einsum("jg,cjg->cg", Mv, dpsi_r)
                       - np.einsum("cjg,jg->cg", Md, psi_r))
    return out


def hermitian_fields(P, chi, dchi, lapl_chi=None, hess_chi=None,
                     ) -> Dict[str, np.ndarray]:
    """Physical (real) grid fields of a Hermitian density matrix ``P``.

    ``P`` may be real symmetric (the usual case) or complex Hermitian
    (complex MO coefficients over a real basis; for complex basis
    functions use :func:`collocate` directly).  The symmetric-kernel
    fields contract ``Re P``; the paramagnetic current contracts ``Im P``
    through the antisymmetric jp kernel, so all returned arrays are real.
    ``jp`` is identically zero for real ``P``.
    """
    P = np.asarray(P)
    if np.iscomplexobj(P):
        S, A = np.ascontiguousarray(P.real), np.ascontiguousarray(P.imag)
    else:
        S, A = P, None
    out = collocate(S, chi, dchi, lapl_chi, hess_chi)
    if A is not None:
        out["jp"] = collocate(A, chi, dchi)["jp"]
    return out


def hermitian_fock(G: np.ndarray, imag: bool = True) -> np.ndarray:
    """Reassemble the complex Fock matrix from a real general kernel output.

    The generated Fock kernels return ``G_uv = dE/dM_uv`` for a real general
    matrix ``M`` whose symmetric part carries the density fields and whose
    antisymmetric part carries the current.  For a complex Hermitian ``P``
    (bra-ket convention, current channel from ``Im P``) the corresponding
    Fock matrix is ``F = sym(G) - i asym(G)``; the antisymmetric part is
    nonzero only for current-dependent (cmgga) families.  With
    ``imag=False`` the real symmetric part alone is returned (sufficient
    for current-free functionals).
    """
    Gs = 0.5 * (G + G.T)
    if not imag:
        return Gs
    Ga = 0.5 * (G - G.T)
    if not np.abs(Ga).max() > 0.0:
        return Gs
    return Gs - 1j * Ga
