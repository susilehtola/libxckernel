"""Monomial-level polynomial arithmetic for the derivative tower.

The generic path (sympy.diff over the full expression per atom, then a global
expand) is quadratic-and-worse in the term count and becomes intractable for
high-order kernels (unpolarized meta-GGA order 4 did not finish in 12 minutes).
But every integrand in this library is a plain multivariate polynomial in
opaque symbols, and the derivative of a *monomial* is trivial.  This module
represents expressions as {powers-tuple: coefficient} dictionaries and applies
the seeded total derivative term-by-term with hash-map accumulation -- exact,
and linear in (terms x atoms-per-term x seed-monomials).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import sympy as sp

#: A monomial key: sorted tuple of (symbol, integer power).
Mono = Tuple[Tuple[sp.Symbol, int], ...]
#: Polynomial: monomial key -> rational/float coefficient.
Poly = Dict[Mono, sp.Rational]


def from_expr(expr: sp.Expr) -> Poly:
    """Expand once and convert to the dict representation."""
    poly: Poly = {}
    for term in sp.Add.make_args(sp.expand(expr)):
        coeff, rest = term.as_coeff_Mul()
        powers = rest.as_powers_dict() if rest != 1 else {}
        key = tuple(sorted(((s, int(e)) for s, e in powers.items()),
                           key=lambda p: p[0].name))
        poly[key] = poly.get(key, sp.Integer(0)) + coeff
    return {k: c for k, c in poly.items() if c != 0}


def to_expr(poly: Poly) -> sp.Expr:
    terms = []
    for key, coeff in poly.items():
        t = coeff
        for s, e in key:
            t *= s ** e
        terms.append(t)
    return sp.Add(*terms) if terms else sp.Integer(0)


def _mul_mono(a: Mono, b: Mono) -> Mono:
    d: Dict[sp.Symbol, int] = dict(a)
    for s, e in b:
        d[s] = d.get(s, 0) + e
    return tuple(sorted(((s, e) for s, e in d.items() if e != 0),
                        key=lambda p: p[0].name))


def seeded_derivative(poly: Poly,
                      seed: Callable[[sp.Symbol], "Poly | None"]) -> Poly:
    """Total derivative sum_atoms (d poly / d atom) * seed(atom).

    ``seed`` maps a symbol to its derivative as a Poly, or None for
    P-independent atoms.  Seeds are cached per symbol by the caller if needed;
    here we memoize locally.
    """
    cache: Dict[sp.Symbol, "Poly | None"] = {}
    out: Poly = {}
    for key, coeff in poly.items():
        for i, (atom, exp) in enumerate(key):
            if atom not in cache:
                cache[atom] = seed(atom)
            d = cache[atom]
            if not d:
                continue
            # monomial with atom's power reduced by one
            rest = list(key)
            if exp == 1:
                rest.pop(i)
            else:
                rest[i] = (atom, exp - 1)
            base = tuple(rest)
            c0 = coeff * exp
            for dkey, dcoeff in d.items():
                nk = _mul_mono(base, dkey)
                out[nk] = out.get(nk, sp.Integer(0)) + c0 * dcoeff
    return {k: c for k, c in out.items() if c != 0}
