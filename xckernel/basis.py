"""Symbolic basis-function fields on the integration grid.

A single basis function chi_i evaluated at a grid point carries, for our
purposes, a finite set of local values:

    val   = chi_i
    grad  = (d/dx, d/dy, d/dz) chi_i
    lapl  = laplacian chi_i

Everything the ingredients need (density, gradient, kinetic energy density,
laplacian of the density) is a bilinear form in two such basis functions
contracted with the density matrix P.  We therefore represent a basis function
by a small bundle of SymPy symbols tagged with an index label ('u', 'v', ...).
The label 'u'/'v' are the *free* indices of a Fock matrix element F_uv; 'a'/'b'
are the *summed* indices of the microscopic definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import sympy as sp

#: Cartesian axis labels, fixed order used throughout for gradient vectors.
AXES: Tuple[str, str, str] = ("x", "y", "z")


@dataclass(frozen=True)
class Orbital:
    """Local values of one basis function, tagged by an index ``label``.

    Attributes are SymPy symbols named so generated code is readable, e.g.
    ``chi_u``, ``dchi_u_x``, ``lapl_chi_u``.
    """

    label: str
    val: sp.Symbol
    grad: Tuple[sp.Symbol, sp.Symbol, sp.Symbol]
    lapl: sp.Symbol

    @classmethod
    def make(cls, label: str) -> "Orbital":
        val = sp.Symbol(f"chi_{label}", real=True)
        grad = tuple(sp.Symbol(f"dchi_{label}_{ax}", real=True) for ax in AXES)
        lapl = sp.Symbol(f"lapl_chi_{label}", real=True)
        return cls(label=label, val=val, grad=grad, lapl=lapl)

    @property
    def symbols(self) -> Tuple[sp.Symbol, ...]:
        """All symbols belonging to this orbital, for codegen operand lists."""
        return (self.val, *self.grad, self.lapl)


def dot(a: Tuple[sp.Expr, ...], b: Tuple[sp.Expr, ...]) -> sp.Expr:
    """Euclidean dot product of two 3-vectors of SymPy expressions."""
    return sp.Add(*(ai * bi for ai, bi in zip(a, b)))
