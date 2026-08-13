"""Explicit cell-deformation (strain) derivatives of the XC tower: the
uniform-grid Pulay layer for plane-wave and finite-difference hosts.

Convention (Nielsen-Martin): under the deformation gradient A = 1 + eps
the cell deforms as r -> A r with the expansion coefficients and the
fractional grid held fixed, so reciprocal vectors transform as
G -> A^{-T} G, the volume as Omega -> det(A) Omega, and the plane-wave
phase G.r at a mapped grid point is invariant.  The uniform-grid weight
w = Omega/N and every field then carry closed-form EXPLICIT derivatives
at eps = 0:

* weight     dw/de_ab      = +delta_ab w          (volume element)
* rho        drho/de_ab    = -delta_ab rho        (1/Omega normalization)
* grad rho   d(d_i rho)    = -delta_ab d_i rho - delta_ib d_a rho
* laplacian  d(lapl rho)   = -delta_ab lapl - 2 hess_rho_ab
* hessian    d(hess_ij)    = -delta_ab h_ij - delta_ib h_aj - delta_jb h_ia
* tau        dtau/de_ab    = -delta_ab tau - 2 tau_tensor_ab
* j_p        d(jp_i)/de_ab = -delta_ab jp_i - delta_ib jp_a

(the -delta_ab terms are the normalization of |psi|^2-like bilinears, the
index-transfer terms the metric d'_c = d_c - eps_dc d_d).  Everything
closes over existing operands except tau, which needs the kinetic-energy
-density tensor tau_tensor_ab = 1/2 sum_i d_a psi_i d_b psi_i (trace =
tau, packed xx,xy,xz,yy,yz,zz like the density Hessian) -- the same
tensor any meta-GGA stress implementation already builds.

Derived ingredients chain through the primitives exactly as in the
response layer, e.g. dsigma/de_ab = -2 delta_ab sigma - 2 d_a rho d_b rho
-- the familiar GGA stress structure.  The seeds are the derivative with
respect to the UNSYMMETRIZED deformation gradient; hosts contracting with
a symmetric strain take the symmetric part, and the antisymmetric part
vanishing (rotational invariance) is a free validation sum rule.

Composition with the rest of the tower is monomial-wise like every other
operator here, so strain derivatives of response integrands (elastic
constants, internal strain) come from the same seeds.
"""

from __future__ import annotations

from collections import Counter

import sympy as sp

from ..inputs.basis import AXES, HESS_COMPS, HESS_INDEX
from ..inputs.functional import Functional
from ..inputs.ingredients import GRAD_RHO, HESS_RHO, JP, PRIM_BY_SYMBOL
from .deriv import LIBXC_MULTISET, VARS, libxc_symbol
from .kernel import KernelIntegrand

#: kinetic-energy-density tensor operand, (6, ng), packed like hess_rho;
#: tau_tensor_xx + tau_tensor_yy + tau_tensor_zz = tau.
TAU_TENSOR = tuple(
    sp.Symbol(f"tau_tensor_{AXES[i]}{AXES[j]}", real=True)
    for (i, j) in HESS_COMPS)

#: per-particle XC energy density (the Libxc zk array); the energy density
#: is rho * zk, which is what the volume term of the strain derivative uses.
ZK = sp.Symbol("zk", real=True)


def _d(a: int, b: int) -> int:
    return 1 if a == b else 0


def strain_primitive(prim, a: int, b: int) -> sp.Expr:
    """Explicit d(primitive)/d eps_ab at fixed coefficients and fractional
    grid, in existing operands (plus tau_tensor for tau)."""
    name = prim.name
    if name == "rho":
        return -_d(a, b) * prim.symbol
    if name.startswith("grad_rho_"):
        i = AXES.index(name[-1])
        return (-_d(a, b) * prim.symbol
                - _d(i, b) * GRAD_RHO[a].symbol)
    if name == "lapl_rho":
        return (-_d(a, b) * prim.symbol
                - 2 * HESS_RHO[HESS_INDEX[(a, b)]].symbol)
    if name == "tau":
        return (-_d(a, b) * prim.symbol
                - 2 * TAU_TENSOR[HESS_INDEX[(a, b)]])
    if name.startswith("hess_rho_"):
        comp = name[len("hess_rho_"):]
        i, j = AXES.index(comp[0]), AXES.index(comp[1])
        return (-_d(a, b) * prim.symbol
                - _d(i, b) * HESS_RHO[HESS_INDEX[(a, j)]].symbol
                - _d(j, b) * HESS_RHO[HESS_INDEX[(i, a)]].symbol)
    if name.startswith("jp_"):
        i = AXES.index(name[-1])
        return (-_d(a, b) * prim.symbol
                - _d(i, b) * JP[a].symbol)
    if name == "inv_rho":
        # closes rationally through rho, like the response layer
        return -prim.symbol**2 * (-_d(a, b) * sp.Symbol("rho", real=True))
    raise ValueError(f"strain seed of primitive {name!r} not defined")


def strain_ingredient(ing, a: int, b: int) -> sp.Expr:
    """Explicit d(ingredient)/d eps_ab, chained through its primitives."""
    E = ing.value
    total = sp.Integer(0)
    for atom in E.free_symbols:
        prim = PRIM_BY_SYMBOL.get(atom)
        if prim is not None:
            total += sp.diff(E, atom) * strain_primitive(prim, a, b)
    return sp.expand(total)


def strain_seed_fn(func: Functional, a: int, b: int):
    """Monomial-level seed for the explicit strain operator d/d eps_ab:
    fields by their closed-form seeds, Libxc derivative symbols by the
    chain rule through the functional's own ingredients, the quadrature
    weight by its volume derivative +delta_ab w."""
    from .fastpoly import from_expr
    by_name = {ing.name: ing for ing in func.ingredients}

    def seed(atom: sp.Symbol):
        prim = PRIM_BY_SYMBOL.get(atom)
        if prim is not None:
            return from_expr(strain_primitive(prim, a, b))
        if atom.name in LIBXC_MULTISET:
            ms = LIBXC_MULTISET[atom.name]
            d = sp.Integer(0)
            for Y in VARS:
                ing = by_name.get(Y)
                if ing is not None:
                    d += libxc_symbol(ms + Counter({Y: 1})) \
                        * strain_ingredient(ing, a, b)
            return from_expr(d)
        if atom.name == "w":
            return from_expr(_d(a, b) * atom)
        if atom.name == "zk":
            # d zk/d eps = sum_Y vY dY/de / rho ... never needed: zk only
            # appears in the energy path, which uses the ingredient chain
            # directly; keep zk inert under the monomial operator.
            return None
        return None  # basis data, perturbed fields, tau_tensor itself
    return seed


def strain_derivative(ki: KernelIntegrand, a: int, b: int) -> KernelIntegrand:
    """Explicit d/d eps_ab of a tower integrand (Fock, response kernel):
    the strain analog of the grid-motion class, for composition into
    higher cell derivatives."""
    from .fastpoly import from_expr, seeded_derivative, to_expr
    poly = getattr(ki, "poly", None)
    if poly is None:
        poly = from_expr(sp.expand(ki.expr))
    out = seeded_derivative(poly, strain_seed_fn(ki.functional, a, b))
    return KernelIntegrand(functional=ki.functional,
                           index_pairs=list(ki.index_pairs),
                           expr=to_expr(out))


def strain_energy_derivative(family: str, a: int, b: int) -> sp.Expr:
    """Per-point integrand of the explicit XC strain derivative (the XC
    stress contribution): the host multiplies by w = Omega/N and sums,

        dExc/de_ab|expl = sum_g w_g [ delta_ab rho zk
                                      + sum_Y vY dY/de_ab|expl ]_g.

    The delta_ab rho zk term is the volume element (dw/de folded to the
    per-point level); the density-normalization and metric terms live in
    the ingredient seeds.  LDA check: the two combine to the textbook
    delta_ab (e - vrho rho).  Operands: fields, zk, the vY arrays, and
    tau_tensor for tau families."""
    func = Functional.of_family(family)
    total = _d(a, b) * sp.Symbol("rho", real=True) * ZK
    for ing in func.ingredients:
        total += func.vsymbol(ing) * strain_ingredient(ing, a, b)
    return sp.expand(total)
