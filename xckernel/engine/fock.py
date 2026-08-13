"""Assembly of the exchange-correlation Fock matrix element.

    F_uv = d Exc / d P_uv
         = sum_g w_g  sum_k  (d Exc / d k)_g  (d k / d P_uv)_g

where g runs over grid points, k over the functional's ingredients, w_g is the
grid weight, (d Exc / d k) is the Libxc derivative (an opaque symbol), and
(d k / d P_uv) is the ingredient seed derived by this library.

This module returns the *per-grid-point integrand* as a SymPy expression with
free orbitals u and v: the contraction over g is a plain weighted sum that the
codegen layer turns into a loop / einsum.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from ..inputs.basis import Orbital
from ..inputs.functional import Functional


@dataclass
class FockIntegrand:
    """The summand of F_uv over grid points."""

    functional: Functional
    u: Orbital
    v: Orbital
    weight: sp.Symbol
    expr: sp.Expr  # w * sum_k vk * dk/dP_uv


def fock_integrand(family: str,
                   u_label: str = "u",
                   v_label: str = "v") -> FockIntegrand:
    """Build the per-grid integrand of F_uv for a functional family."""
    func = Functional.of_family(family)
    u = Orbital.make(u_label)
    v = Orbital.make(v_label)
    w = sp.Symbol("w", real=True, positive=True)

    integrand = sp.Integer(0)
    for ing in func.ingredients:
        integrand += func.vsymbol(ing) * ing.seed(u, v)
    integrand = sp.expand(w * integrand)

    return FockIntegrand(functional=func, u=u, v=v, weight=w, expr=integrand)
