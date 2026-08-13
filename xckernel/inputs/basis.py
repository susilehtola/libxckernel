"""Symbolic basis-function fields on the integration grid.

A single basis function chi_i evaluated at a grid point carries, for our
purposes, a finite set of local values:

    val   = chi_i
    grad  = (d/dx, d/dy, d/dz) chi_i
    lapl  = laplacian chi_i
    hess  = the six independent second derivatives d_i d_j chi_i
            (packed xx, xy, xz, yy, yz, zz; needed by ingredients built on
            the density Hessian, e.g. the local-hybrid calibration variable
            eta = grad rho . (grad grad rho) . grad rho)

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

#: (i, j) axis pairs of the six independent symmetric-tensor components, in
#: the canonical packing xx, xy, xz, yy, yz, zz.
HESS_COMPS: Tuple[Tuple[int, int], ...] = tuple(
    (i, j) for i in range(3) for j in range(i, 3))
#: (i, j) -> packed component index, both orderings.
HESS_INDEX = {(i, j): k for k, (i, j) in enumerate(HESS_COMPS)}
HESS_INDEX.update({(j, i): k for k, (i, j) in enumerate(HESS_COMPS)})


@dataclass(frozen=True)
class Orbital:
    """Local values of one basis function, tagged by an index ``label``.

    Attributes are SymPy symbols named so generated code is readable, e.g.
    ``chi_u``, ``dchi_u_x``, ``lapl_chi_u``, ``hess_chi_u_xy``.
    """

    label: str
    val: sp.Symbol
    grad: Tuple[sp.Symbol, sp.Symbol, sp.Symbol]
    lapl: sp.Symbol
    hess: Tuple[sp.Symbol, ...]

    @classmethod
    def make(cls, label: str) -> "Orbital":
        val = sp.Symbol(f"chi_{label}", real=True)
        grad = tuple(sp.Symbol(f"dchi_{label}_{ax}", real=True) for ax in AXES)
        lapl = sp.Symbol(f"lapl_chi_{label}", real=True)
        hess = tuple(
            sp.Symbol(f"hess_chi_{label}_{AXES[i]}{AXES[j]}", real=True)
            for (i, j) in HESS_COMPS)
        return cls(label=label, val=val, grad=grad, lapl=lapl, hess=hess)

    def hess_ij(self, i: int, j: int) -> sp.Symbol:
        """Second-derivative symbol d_i d_j chi, from the packed components."""
        return self.hess[HESS_INDEX[(i, j)]]

    @property
    def symbols(self) -> Tuple[sp.Symbol, ...]:
        """All symbols belonging to this orbital, for codegen operand lists."""
        return (self.val, *self.grad, self.lapl, *self.hess)


def dot(a: Tuple[sp.Expr, ...], b: Tuple[sp.Expr, ...]) -> sp.Expr:
    """Euclidean dot product of two 3-vectors of SymPy expressions."""
    return sp.Add(*(ai * bi for ai, bi in zip(a, b)))
