"""Explicit cell-deformation (strain) derivatives of the XC tower: the
uniform-grid Pulay layer for plane-wave and finite-difference hosts.

Convention (Nielsen-Martin): under the deformation gradient A = 1 + E
the cell deforms as r -> A r with the expansion coefficients and the
fractional grid held fixed, so reciprocal vectors transform as
G -> A^{-T} G, the volume as Omega -> det(A) Omega, and the plane-wave
phase G.r at a mapped grid point is invariant.  Everything in this
module derives from the SINGLE master transformation law that follows:
a field with k gradient-type indices transforms as

    T'_{i1..ik}(E) = (det A)^{-1} (A^{-T})_{i1 j1} .. (A^{-T})_{ik jk}
                     T_{j1..jk},

the quadrature weight (volume element) as w' = (det A) w, and 1/rho as
(det A) inv_rho.  Scalars with hidden gradient indices are traces of
tensor operands: the Laplacian of the density Hessian, tau of the
kinetic-energy-density tensor tau_tensor_ab = 1/2 sum_i d_a psi_i
d_b psi_i (trace = tau, hess packing) -- the same tensor any meta-GGA
stress implementation already builds, and the only operand beyond the
existing ones.  Perturbed fields (rho_p<k>, ...) are 1/sqrt(Omega)-
normalized bilinears like their ground-state counterparts and obey the
identical law with the label carried through.

Strain seeds AT ANY ORDER are obtained by expanding the law to the
requested order in E (A^{-1} = sum_k (-E)^k, det A^{+-1} by the
exponential-trace series) and differentiating at E = 0 -- nothing is
hand-derived.  Derived ingredients (sigma, eta, the gauge-corrected
tau) chain through their primitive expressions; Libxc derivative
symbols Taylor-expand in the deformed variables.  First derivatives
reproduce the textbook stress structures (LDA: delta_ab (e - vrho rho);
GGA: the -2 vsigma d_a rho d_b rho metric term); grid positions are
linear in E but the law is not (metric and normalization), which is
why second derivatives need the law to second order rather than a
composition of first-order seeds.  The antisymmetric part of the first
derivative vanishing (rotational invariance) is a free validation sum
rule.
"""

from __future__ import annotations

import re as _re
from collections import Counter
from functools import lru_cache

import sympy as sp

from ..inputs.basis import AXES, HESS_COMPS
from ..inputs.functional import Functional
from ..inputs.ingredients import PRIM_BY_SYMBOL
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

#: the deformation E: nine independent symbols (unsymmetrized; hosts
#: contracting with a symmetric strain take the symmetric part).
E = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"_strain_e_{i}{j}"))
_EPS_SYMS = set(E)
_E0 = {s: 0 for s in _EPS_SYMS}


def _sym(name: str) -> sp.Symbol:
    return sp.Symbol(name, real=True)


# --- the master law, expanded to a requested order in E ----------------------

def _trunc(expr: sp.Expr, order: int) -> sp.Expr:
    """Drop monomials beyond E^order.  The expansions are only trusted to
    that order, and truncating after EVERY product keeps the polynomial
    sizes bounded (unguarded, the spurious high orders explode
    combinatorially in the nine deformation symbols)."""
    keep = []
    for term in sp.Add.make_args(sp.expand(expr)):
        deg = sum(p for s, p in term.as_powers_dict().items()
                  if s in _EPS_SYMS)
        if deg <= order:
            keep.append(term)
    return sp.Add(*keep)


@lru_cache(maxsize=None)
def _cell_algebra(order: int):
    """(A^{-T}, det A, 1/det A) as polynomials in E, exact through
    E^order and truncated there."""
    ainv = sp.zeros(3, 3)
    P = sp.eye(3)
    for _k in range(order + 1):
        ainv += P
        P = -P * E
    # det(1+E)^{+-1} = exp(+-sum_k (-1)^{k+1} tr(E^k)/k), truncated
    logdet = sp.Integer(0)
    Pk = sp.eye(3)
    for k in range(1, order + 1):
        Pk = Pk * E
        logdet += sp.Rational((-1) ** (k + 1), k) * Pk.trace()
    det = invdet = sp.Integer(1)
    term_p = term_m = sp.Integer(1)
    for k in range(1, order + 1):
        term_p = _trunc(term_p * logdet / k, order)
        term_m = _trunc(term_m * (-logdet) / k, order)
        det += term_p
        invdet += term_m
    return ainv.T, _trunc(det, order), _trunc(invdet, order)


def _hess_get(prefix: str):
    def get(k, l):
        return _sym(f"{prefix}_{AXES[min(k, l)]}{AXES[max(k, l)]}")
    return get


@lru_cache(maxsize=None)
def deformed_atom(atom: sp.Symbol, order: int = 2) -> "sp.Expr | None":
    """The operand under the master law, exact through E^order; None for
    atoms the law does not touch (basis data; Libxc symbols are handled
    by their own Taylor expansion)."""
    ainvt, det, invdet = _cell_algebra(order)
    name = atom.name

    def vec(get, i):
        return _trunc(invdet * sum(ainvt[i, j] * get(j)
                                   for j in range(3)), order)

    def tensor(get, i, j):
        return _trunc(invdet * sum(ainvt[i, k] * ainvt[j, l] * get(k, l)
                                   for k in range(3) for l in range(3)),
                      order)

    def trace(get):
        return sum(tensor(get, c, c) for c in range(3))

    if name == "w" or name == "inv_rho":
        return det * atom
    m = _re.match(r"^(rho|lapl_rho|tau)(_p\d+)?$", name)
    if m:
        base, lab = m.groups()
        suffix = lab or ""
        if base == "rho":
            return invdet * atom
        if base == "lapl_rho":
            return trace(_hess_get(f"hess_rho{suffix}"))
        return trace(_hess_get(f"tau_tensor{suffix}"))
    m = _re.match(r"^(grad_rho|jp)(_p\d+)?_([xyz])$", name)
    if m:
        base, lab, ax = m.groups()
        suffix = lab or ""
        return vec(lambda j: _sym(f"{base}{suffix}_{AXES[j]}"),
                   AXES.index(ax))
    m = _re.match(r"^(hess_rho|tau_tensor)(_p\d+)?_([xyz]{2})$", name)
    if m:
        base, lab, comp = m.groups()
        suffix = lab or ""
        return tensor(_hess_get(f"{base}{suffix}"),
                      AXES.index(comp[0]), AXES.index(comp[1]))
    return None


def _deformed_variable(ing, order: int) -> sp.Expr:
    """Deformed value Y(E) of a Libxc input variable: its ingredient
    expression with every primitive replaced by the master law."""
    subs = {}
    for atom in ing.value.free_symbols:
        if PRIM_BY_SYMBOL.get(atom) is not None:
            subs[atom] = deformed_atom(atom, order)
    return _trunc(ing.value.subs(subs, simultaneous=True), order)


def _libxc_taylor(atom: sp.Symbol, func: Functional, order: int) -> sp.Expr:
    """Taylor expansion of a Libxc derivative symbol under deformation,
    through E^order: v_M(E) = sum_n 1/n! sum_{Y1..Yn} v_{M+Y1..Yn}
    dY1 .. dYn, with dY = Y(E) - Y."""
    from itertools import combinations_with_replacement
    from math import factorial

    ms = LIBXC_MULTISET[atom.name]
    by_name = {ing.name: ing for ing in func.ingredients}
    active = [Y for Y in VARS if Y in by_name]
    delta = {Y: sp.expand(_deformed_variable(by_name[Y], order)
                          - by_name[Y].value) for Y in active}
    total = atom
    for n in range(1, order + 1):
        for combo in combinations_with_replacement(active, n):
            mult = Counter(combo)
            coeff = factorial(n)
            for c in mult.values():
                coeff //= factorial(c)
            total += sp.Rational(coeff, factorial(n)) \
                * libxc_symbol(ms + mult) \
                * _trunc(sp.prod([delta[Y] for Y in combo]), order)
    return total


def deform(expr: sp.Expr, func: Functional, order: int = 2) -> sp.Expr:
    """The integrand under finite deformation, exact through E^order:
    every transformable operand by the master law, every Libxc
    derivative symbol by its Taylor expansion.  All strain derivatives
    are sp.diff of this at E = 0."""
    subs = {}
    for atom in expr.free_symbols:
        if atom in _EPS_SYMS:
            raise ValueError("expression already carries deformation symbols")
        if atom.name in LIBXC_MULTISET:
            subs[atom] = _libxc_taylor(atom, func, order)
            continue
        d = deformed_atom(atom, order)
        if d is not None:
            subs[atom] = d
    return _trunc(expr.subs(subs, simultaneous=True), order)


def strain_diff(expr: sp.Expr, func: Functional,
                *components: "tuple[int, int]") -> sp.Expr:
    """d^n/d eps_{a1 b1} .. d eps_{an bn} of an integrand at eps = 0,
    n = len(components) -- the single derivative entry point behind
    every convenience wrapper below."""
    d = deform(sp.expand(expr), func, order=len(components))
    for (a, b) in components:
        d = sp.diff(d, E[a, b])
    return sp.expand(d.subs(_E0))


# --- derived first-order seeds (for the monomial operator) -------------------

@lru_cache(maxsize=None)
def _atom_seed1(atom: sp.Symbol, a: int, b: int) -> "sp.Expr | None":
    d = deformed_atom(atom, 1)
    if d is None:
        return None
    return sp.expand(sp.diff(d, E[a, b]).subs(_E0))


def strain_primitive(prim, a: int, b: int) -> sp.Expr:
    """First-order seed of a primitive field, derived from the master
    law (kept as the named entry point the validation suite uses)."""
    return _atom_seed1(prim.symbol, a, b)


def strain_ingredient(ing, a: int, b: int) -> sp.Expr:
    """First-order seed of a Libxc input variable, derived by the chain
    rule through the deformed primitives."""
    return sp.expand(sp.diff(_deformed_variable(ing, 1),
                             E[a, b]).subs(_E0))


def strain_seed_fn(func: Functional, a: int, b: int):
    """Monomial-level seed for the explicit strain tangent d/d eps_ab at
    eps = 0: every rule differentiated out of the master law / Taylor
    expansion.  NOTE: the tangent operator composed with itself is NOT
    the second cell derivative (the law is nonlinear in E); use
    strain_diff / strain_derivative2 for higher orders."""
    from .fastpoly import from_expr
    by_name = {ing.name: ing for ing in func.ingredients}

    def seed(atom: sp.Symbol):
        if atom.name in LIBXC_MULTISET:
            ms = LIBXC_MULTISET[atom.name]
            d = sp.Integer(0)
            for Y in VARS:
                ing = by_name.get(Y)
                if ing is not None:
                    d += libxc_symbol(ms + Counter({Y: 1})) \
                        * strain_ingredient(ing, a, b)
            return from_expr(d)
        s = _atom_seed1(atom, a, b)
        if s is not None and s != 0:
            return from_expr(s)
        return None  # basis data, zk, inert atoms (or identically zero)
    return seed


def strain_derivative(ki: KernelIntegrand, a: int, b: int) -> KernelIntegrand:
    """Explicit d/d eps_ab of a tower integrand (Fock, response kernel):
    the strain tangent, monomial-wise for the big integrands."""
    from .fastpoly import from_expr, seeded_derivative, to_expr
    poly = getattr(ki, "poly", None)
    if poly is None:
        poly = from_expr(sp.expand(ki.expr))
    out = seeded_derivative(poly, strain_seed_fn(ki.functional, a, b))
    return KernelIntegrand(functional=ki.functional,
                           index_pairs=list(ki.index_pairs),
                           expr=to_expr(out))


def strain_derivative2(ki: KernelIntegrand, ab: "tuple[int, int]",
                       cd: "tuple[int, int]") -> KernelIntegrand:
    """Exact second cell derivative of a tower integrand at eps = 0
    (expression-level, via the order-2 master law)."""
    return KernelIntegrand(
        functional=ki.functional, index_pairs=list(ki.index_pairs),
        expr=strain_diff(ki.expr, ki.functional, ab, cd))


# --- the energy path ---------------------------------------------------------

def _energy_taylor(family: str, order: int) -> "tuple[sp.Expr, Functional]":
    """The per-point energy integrand det(A) * e(Y(E)) through E^order:
    the deformed volume element times the Taylor expansion of the energy
    density in the deformed variables."""
    func = Functional.of_family(family)
    from itertools import combinations_with_replacement
    from math import factorial

    by_name = {ing.name: ing for ing in func.ingredients}
    active = [Y for Y in VARS if Y in by_name]
    delta = {Y: sp.expand(_deformed_variable(by_name[Y], order)
                          - by_name[Y].value) for Y in active}
    e = sp.Symbol("rho", real=True) * ZK
    for n in range(1, order + 1):
        for combo in combinations_with_replacement(active, n):
            mult = Counter(combo)
            coeff = factorial(n)
            for c in mult.values():
                coeff //= factorial(c)
            e += sp.Rational(coeff, factorial(n)) * libxc_symbol(mult) \
                * _trunc(sp.prod([delta[Y] for Y in combo]), order)
    _, det, _ = _cell_algebra(order)
    return _trunc(det * e, order), func


def strain_energy_derivative(family: str, a: int, b: int) -> sp.Expr:
    """Per-point integrand of the explicit XC strain derivative (the XC
    stress contribution): the host multiplies by w = Omega/N and sums.
    LDA check: reduces to the textbook delta_ab (e - vrho rho).
    Operands: fields, zk, the vY arrays, and tau_tensor for tau
    families."""
    e, _ = _energy_taylor(family, 1)
    return sp.expand(sp.diff(e, E[a, b]).subs(_E0))


def strain_energy_hessian(family: str, ab: "tuple[int, int]",
                          cd: "tuple[int, int]") -> sp.Expr:
    """Per-point integrand of the explicit second strain derivative of
    Exc (the pure-strain XC elastic term); host multiplies by w and
    sums.  Operands: fields, zk, vY, v2YZ, tau_tensor, hess_rho."""
    e, _ = _energy_taylor(family, 2)
    return sp.expand(sp.diff(e, E[ab[0], ab[1]],
                             E[cd[0], cd[1]]).subs(_E0))
