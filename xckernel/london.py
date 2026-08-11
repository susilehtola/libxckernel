"""Explicit magnetic-field derivatives of xc kernels with London orbitals.

London (gauge-including) atomic orbitals attach a field-dependent phase to
each basis function,

    chi_a^B(r) = exp[-(i/2c) B . (R_a x r)] chi_a(r),

so the xc Fock matrix acquires an explicit field derivative at fixed density
matrix: the GIAO xc contribution to the CPKS right-hand side b^B.  At a real
(current-free) reference, the ingredient fields are unaffected by B to first
order -- the phase factors cancel in every P-contracted, index-symmetric
bilinear -- so only the free basis-function pair is differentiated:

    dF^xc_ab / dB_s |_{B=0, P} = (i/2c) K^s_ab,

with K^s real and antisymmetric.  This module emits K^s.  All B-derivative
factors reduce to two kinds of host-computable operands:

    Rchi[a]      = R_{alpha,a} chi_alpha(r_g)          (3, nbf, ng)
    Rdchi[a][c]  = R_{alpha,a} d_c chi_alpha(r_g)      (3, 3, nbf, ng)
    Rlapl_chi[a] = R_{alpha,a} lapl chi_alpha(r_g)     (3, nbf, ng)
    rg[b]        = Cartesian grid coordinates          (3, ng)

using (R x r)_s = eps_{sab} R_a r_b and d_c (R x r)_s = eps_{sac} R_a.

The emitted kernels follow from the same symbolic machinery as everything
else: the phase derivative of each primitive bilinear Q_k is assembled
symbolically and contracted with the first functional derivatives.
"""

from __future__ import annotations

from typing import List, Tuple

import sympy as sp

from .basis import AXES, Orbital
from .functional import Functional
from .kernel import KernelIntegrand, fock

#: Levi-Civita symbol.
def _eps(i: int, j: int, k: int) -> int:
    return (i - j) * (j - k) * (k - i) // 2


def _rg(b: int) -> sp.Symbol:
    return sp.Symbol(f"rg_{AXES[b]}", real=True)


def _Rchi(lbl: str, a: int) -> sp.Symbol:
    return sp.Symbol(f"Rchi_{lbl}_{AXES[a]}", real=True)


def _Rdchi(lbl: str, a: int, c: int) -> sp.Symbol:
    return sp.Symbol(f"Rdchi_{lbl}_{AXES[a]}_{AXES[c]}", real=True)


def _Rlapl(lbl: str, a: int) -> sp.Symbol:
    return sp.Symbol(f"Rlapl_chi_{lbl}_{AXES[a]}", real=True)


def _m(lbl: str, s: int) -> sp.Expr:
    """(R_alpha x r)_s chi_alpha as an expression in the operands."""
    total = sp.Integer(0)
    for a in range(3):
        for b in range(3):
            e = _eps(s, a, b)
            if e:
                total += e * _Rchi(lbl, a) * _rg(b)
    return total


def _m_grad(lbl: str, s: int, c: int) -> sp.Expr:
    """d/dB_s of d_c chi^B (real factor): (R x r)_s d_c chi + eps_{sac} R_a chi."""
    total = sp.Integer(0)
    for a in range(3):
        for b in range(3):
            e = _eps(s, a, b)
            if e:
                total += e * _Rdchi(lbl, a, c) * _rg(b)
        e2 = _eps(s, a, c)
        if e2:
            total += e2 * _Rchi(lbl, a)
    return total


def _m_lapl(lbl: str, s: int) -> sp.Expr:
    """d/dB_s of lapl chi^B (real factor): (R x r)_s lapl chi
    + 2 sum_c eps_{sac} R_a d_c chi."""
    total = sp.Integer(0)
    for a in range(3):
        for b in range(3):
            e = _eps(s, a, b)
            if e:
                total += e * _Rlapl(lbl, a) * _rg(b)
        for c in range(3):
            e2 = _eps(s, a, c)
            if e2:
                total += 2 * e2 * _Rdchi(lbl, a, c)
    return total


def london_fock(family: str, s: int) -> KernelIntegrand:
    """Integrand of K^s: the real factor of the explicit GIAO B_s-derivative
    of the xc Fock matrix, F^{B_s}_uv = (i/2c) K^s_uv at a real reference."""
    func = Functional.of_family(family)
    u, v = Orbital.make("u"), Orbital.make("v")
    w = sp.Symbol("w", real=True, positive=True)

    # d/dB_s of the primitive bilinears (real factors; global i/2c pulled out)
    def d_rho() -> sp.Expr:
        # _m already contains the basis-function factor
        return _m("u", s) * v.val - u.val * _m("v", s)

    def d_grad(cc: int) -> sp.Expr:
        # d/dB_s [ (d_c chi_u) chi_v + chi_u (d_c chi_v) ] (real factor)
        t = _m_grad("u", s, cc) * v.val + _m("u", s) * v.grad[cc] \
            - u.grad[cc] * _m("v", s) - u.val * _m_grad("v", s, cc)
        return t

    def d_tau() -> sp.Expr:
        t = sp.Integer(0)
        for cc in range(3):
            t += _m_grad("u", s, cc) * v.grad[cc] \
                - u.grad[cc] * _m_grad("v", s, cc)
        return sp.Rational(1, 2) * t

    def d_lapl() -> sp.Expr:
        t = _m_lapl("u", s) * v.val - u.lapl * _m("v", s) \
            + _m("u", s) * v.lapl - u.val * _m_lapl("v", s)
        for cc in range(3):
            t += 2 * (_m_grad("u", s, cc) * v.grad[cc]
                      - u.grad[cc] * _m_grad("v", s, cc))
        return t

    from .ingredients import GRAD_RHO
    expr = sp.Integer(0)
    for ing in func.ingredients:
        vsym = sp.Symbol(f"v{ing.name}", real=True)
        if ing.name == "rho":
            expr += vsym * d_rho()
        elif ing.name == "sigma":
            # d sigma-seed: 2 grad_rho . d/dB_s grad-pair
            for cc in range(3):
                expr += 2 * vsym * GRAD_RHO[cc].symbol * d_grad(cc)
        elif ing.name == "tau":
            expr += vsym * d_tau()
        elif ing.name == "lapl":
            expr += vsym * d_lapl()
        else:
            raise ValueError(f"GIAO derivative not available for "
                             f"ingredient {ing.name!r}")
    expr = sp.expand(w * expr)
    return KernelIntegrand(functional=func, index_pairs=[("u", "v")],
                           expr=expr)
