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


def rpa_sigma(xys: np.ndarray, e_ia: np.ndarray, Co: np.ndarray,
              Cv: np.ndarray, vresp, occ: float = 2.0) -> np.ndarray:
    """Full RPA/TDHF sigma vectors: the action of [[A, B], [-B, -A]] on (X, Y)
    for real orbitals.

    The (X, Y) pair enters through ONE AO density matrix per vector,
        dm = occ (C_v X^T C_o^T + C_o Y C_v^T),
    (the Y block is the transpose slot), so a single vresp call serves both
    blocks; the result is projected two ways:
        top    (A X + B Y)_ia = e_ia X_ia + (C_o^T V^T C_v)_ia
        bottom (B X + A Y)_ia = e_ia Y_ia + (C_o^T V   C_v)_ia
    and the bottom is negated per the [[A,B],[-B,-A]] supervector convention.

    xys: (nz, 2, nocc, nvir).  Returns the same shape.
    """
    xys = np.asarray(xys)
    dms = np.array([transition_dm(x, Co, Cv, occ)
                    + occ * (Co @ y @ Cv.T) for x, y in xys])
    v1 = vresp(dms)
    out = np.empty_like(xys)
    for k, (x, y) in enumerate(xys):
        out[k, 0] = e_ia * x + project_ov(v1[k], Co, Cv)      # A X + B Y
        out[k, 1] = -(e_ia * y + Co.T @ v1[k] @ Cv)           # -(B X + A Y)
    return out


def perturbed_dm_order(kappas, C: np.ndarray, nocc: int,
                       occ: float = 2.0, sign: int = -1) -> np.ndarray:
    """Mixed n-th order density response to C(x) = C exp(sign * sum_i x_i k_i).

    Returns  d^n P / dx_1 ... dx_n  at x = 0, which by the BCH expansion of
    e^{K} Pi e^{-K} is the permutation-symmetrized nested commutator

        sign^n / n! * occ * C [ sum_perm [k_p1, [k_p2, ... [k_pn, Pi]]] ] C^T.

    This is the universal higher-order perturbed-density builder (survey S2):
    at n=1 it reduces to orbital_rotation_dm; at n=2 it is the mathematical
    content of VeloxChem's D_bc = [k_b, D_c] + [k_c, D_b] and Dalton's
    commute_d_x chains (up to each code's normalization convention, which is
    why the factor here is the *true mixed derivative* -- adapters rescale).
    """
    from itertools import permutations
    from math import factorial
    if sign not in (-1, +1):
        raise ValueError("sign must be -1 or +1")
    kappas = list(kappas)
    n = len(kappas)
    Pi = np.zeros((C.shape[1], C.shape[1]))
    Pi[:nocc, :nocc] = np.eye(nocc)
    total = np.zeros_like(Pi)
    for perm in permutations(range(n)):
        M = Pi
        for idx in reversed(perm):
            k = kappas[idx]
            M = k @ M - M @ k
        total += M
    fac = (sign ** n) / factorial(n)
    return fac * occ * (C @ total @ C.T)


def unit_rotation(i: int, a: int, nocc: int, nmo: int) -> np.ndarray:
    """The antisymmetric unit generator K_ia = e_a e_i^T - e_i e_a^T
    (a counted within the virtual block)."""
    K = np.zeros((nmo, nmo))
    K[nocc + a, i] = 1.0
    K[i, nocc + a] = -1.0
    return K


def _set_partitions(items):
    """All partitions of a list into unordered nonempty blocks."""
    items = list(items)
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for part in _set_partitions(rest):
        yield [[first]] + part
        for i in range(len(part)):
            yield part[:i] + [[first] + part[i]] + part[i + 1:]


def response_sigma_xc(kappas, C: np.ndarray, nocc: int, fock_contract,
                      occ: float = 2.0, sign: int = -1) -> np.ndarray:
    """XC sigma vector of arbitrary response order,
    sigma_ia = d^{n+1} Exc / dt_1 ... dt_n dx_ia at 0 for
    C(x) = C exp(sign * (sum_k t_k kappa_k + sum_ia x_ia K_ia)).

    The Leibniz/Faa-di-Bruno expansion of d^n/dt [ F(P) dP/dx_ia ] gives

      sigma_ia = sum_{S subset of perturbations}
                 sum_{partitions pi of S}
                   Tr[ (g_{1+|pi|} : prod_{beta in pi} D^beta)
                       * d^{|S^c|+1} P(kappa_{S^c}, K_ia) ],

    with D^beta = perturbed_dm_order(block) and g_m the m-th order XC
    contraction.  n=1 is the linear-response gradient, n=2 quadratic (E[3]),
    n=3 cubic (E[4]), and so on -- the assembly is the same loop.

    fock_contract(Ds) -> AO matrix: the order-(len(Ds)+1) XC contraction
    g_{1+m} : D_1 : ... : D_m; an empty list yields the XC Fock matrix.
    """
    from itertools import combinations
    if sign not in (-1, +1):
        raise ValueError("sign must be -1 or +1")
    kappas = list(kappas)
    n = len(kappas)
    nmo = C.shape[1]
    nvir = nmo - nocc
    idx = tuple(range(n))

    # effective Fock matrix per perturbation subset S (Faa di Bruno over S)
    eff = {}
    for r in range(n + 1):
        for S in combinations(idx, r):
            M = None
            for part in _set_partitions(S):
                Ds = [perturbed_dm_order([kappas[j] for j in block], C, nocc,
                                         occ, sign) for block in part]
                term = fock_contract(Ds)
                M = term if M is None else M + term
            eff[S] = M

    sigma = np.zeros((nocc, nvir))
    for i in range(nocc):
        for a in range(nvir):
            K = unit_rotation(i, a, nocc, nmo)
            tot = 0.0
            for S, M in eff.items():
                rest = [kappas[j] for j in idx if j not in S] + [K]
                tot += np.sum(M * perturbed_dm_order(rest, C, nocc, occ, sign))
            sigma[i, a] = tot
    return sigma


def quadratic_sigma_xc(kappa_B: np.ndarray, kappa_C: np.ndarray,
                       C: np.ndarray, nocc: int,
                       fock0, fresp, fresp2,
                       occ: float = 2.0, sign: int = -1) -> np.ndarray:
    """XC part of the quadratic-response sigma vector,
    sigma_ia = d^3 Exc / db dc dx_ia at 0 for C(x) = C exp(sign*(b kB + c kC
    + sum x_ia K_ia)).

    Assembled purely from the derivative tower and the nested-commutator
    density builder -- the total-derivative identity

      sigma_ia = Tr[F3^{BC} dP(K)] + Tr[F^B d2P(kC,K)] + Tr[F^C d2P(kB,K)]
               + Tr[F0 d3P(kB,kC,K)],
      F3^{BC}  = g3[D^B, D^C] + g2[D^{BC}]

    reproduces the closed forms every surveyed code hand-derives (VeloxChem's
    xi/zeta, Dalton's Q3FOCK one-index chains) without deriving them: each
    trace term IS one of those contributions.

    Callables supplied by the host (layer 1 provides the XC ones):
      fock0()        -> AO XC Fock matrix at the reference density
      fresp(D)       -> order-2 contraction  g2 : D        (AO matrix)
      fresp2(D1,D2)  -> order-3 contraction  g3 : D1 : D2  (AO matrix)

    Thin wrapper over response_sigma_xc (the arbitrary-order assembly).
    """
    def fock_contract(Ds):
        if len(Ds) == 0:
            return fock0()
        if len(Ds) == 1:
            return fresp(Ds[0])
        if len(Ds) == 2:
            return fresp2(Ds[0], Ds[1])
        raise ValueError("quadratic response needs contractions up to order 3")

    return response_sigma_xc([kappa_B, kappa_C], C, nocc, fock_contract,
                             occ=occ, sign=sign)
