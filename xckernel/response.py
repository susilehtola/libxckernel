"""Response contraction engine: n-th order XC response Fock matrices.

The survey of production codes (docs/dedup-analysis.md) fixes the driver
contract: perturbed AO density matrices in, AO Fock-like matrices out, with the
functional derivatives contracted against the perturbed densities pointwise on
the grid.  In the derivative-tower language this is the repeated density-matrix
derivative of Exc with all but one index pair *contracted* with perturbed
density matrices:

    F^{X1..Xm}_uv = sum_{t1 s1 .. tm sm} (D^{m+1} Exc)_uv,t1s1,..,tmsm
                    * D^{X1}_{t1 s1} ... D^{Xm}_{tm sm}.

The crucial structural fact: contracting an ingredient seed with a perturbed
density matrix folds into a *perturbed field on the grid*,

    sum_ts (d k / dP_ts) D^X_ts = k^X(r)     (perturbed rho, grad rho, tau, ...),

so the integrand never carries more than one free orbital pair, and the cost is
O(N^2 * n_grid) per perturbation combination -- never an N^4 tensor.  This is
exactly the structure every surveyed code hand-writes (PySCF's wv1/wv2, Psi4's
compute_Vx, Dalton's quad-fast prefactors, VeloxChem's DensityGridQuad,
ERKALE's and HelFEM's chain-rule blocks); here it is generated.

The operator D_X below is therefore the same total derivative as deriv.D_ts,
with a different seed:

* a primitive-field atom (rho, grad_rho_i, lapl_rho, tau) maps to its perturbed
  field symbol (rho_X, grad_rho_X_i, ...);
* a Libxc derivative symbol v_M bumps by one variable Y and multiplies by the
  perturbed *variable* Y^X -- where sigma^X = 2 grad rho . grad rho^X is derived
  from the perturbed gradient fields by the chain rule;
* basis data, weights, and perturbed fields of *other* perturbations are
  constants (the response form is multilinear in the D^X: higher-order
  "perturbed densities" like VeloxChem's D^BC enter as separate inputs supplied
  by the solver, not by differentiating D^B).

Perturbation labels are 'p1', 'p2', ... and become array-parameter names in the
generated code (rho_p1 (ng,), grad_rho_p1 (3,ng), tau_p1, lapl_rho_p1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import sympy as sp

from .basis import AXES
from .deriv import LIBXC_MULTISET, VARS, libxc_symbol
from .fock import fock_integrand
from .functional import Functional
from .ingredients import GRAD_RHO, PRIM_BY_SYMBOL


def pert_field(prim_name: str, label: str) -> sp.Symbol:
    """Perturbed-field symbol for a primitive field, e.g. ('rho','p1')->rho_p1,
    ('grad_rho_x','p1')->grad_rho_p1_x."""
    if prim_name == "rho":
        return sp.Symbol(f"rho_{label}", real=True)
    if prim_name.startswith("grad_rho_"):
        ax = prim_name[-1]
        return sp.Symbol(f"grad_rho_{label}_{ax}", real=True)
    if prim_name == "lapl_rho":
        return sp.Symbol(f"lapl_rho_{label}", real=True)
    if prim_name == "tau":
        return sp.Symbol(f"tau_{label}", real=True)
    if prim_name.startswith("jp_"):
        ax = prim_name[-1]
        return sp.Symbol(f"jp_{label}_{ax}", real=True)
    if prim_name.startswith("hess_rho_"):
        comp = prim_name[len("hess_rho_"):]
        return sp.Symbol(f"hess_rho_{label}_{comp}", real=True)
    raise ValueError(f"unknown primitive {prim_name!r}")


def _pert_atom(prim, label: str) -> sp.Expr:
    """Perturbation of a primitive-field atom.  Ordinary primitives map to
    their perturbed-field symbol; the rational pseudo-primitive inv_rho
    closes through rho: (1/rho)^X = -inv_rho^2 rho^X, so no independent
    perturbed operand is needed."""
    if prim.name == "inv_rho":
        return -prim.symbol**2 * pert_field("rho", label)
    return pert_field(prim.name, label)


def perturbed_ingredient(ing, label: str) -> sp.Expr:
    """Perturbed value Y^X of a Libxc input variable, generically from the
    ingredient's value expression: Y^X = sum_p (dY/d p) p^X over the primitive
    fields p it contains.  Reproduces sigma^X = 2 grad rho . grad rho^X and
    extends to mapped ingredients (gauge-corrected tau, etc.)."""
    E = ing.value
    total = sp.Integer(0)
    for atom in E.free_symbols:
        prim = PRIM_BY_SYMBOL.get(atom)
        if prim is not None:
            total += sp.diff(E, atom) * _pert_atom(prim, label)
    return sp.expand(total)


def perturbed_variable(var: str, label: str) -> sp.Expr:
    """Perturbed value Y^X of a Libxc input variable under perturbation X."""
    if var == "rho":
        return pert_field("rho", label)
    if var == "lapl":
        return pert_field("lapl_rho", label)
    if var == "tau":
        return pert_field("tau", label)
    if var == "sigma":
        # sigma = grad rho . grad rho  =>  sigma^X = 2 grad rho . grad rho^X
        total = sp.Integer(0)
        for p in GRAD_RHO:
            ax = p.name[-1]
            total += 2 * p.symbol * sp.Symbol(f"grad_rho_{label}_{ax}", real=True)
        return total
    if var == "eta":
        from .ingredients import ETA_ING
        return perturbed_ingredient(ETA_ING, label)
    raise ValueError(f"unknown libxc variable {var!r}")


def _seed_fn(func: Functional, label: str):
    """Monomial-level seed map for fastpoly.seeded_derivative."""
    from collections import Counter

    from .fastpoly import from_expr
    by_name = {ing.name: ing for ing in func.ingredients}

    def seed(atom: sp.Symbol):
        prim = PRIM_BY_SYMBOL.get(atom)
        if prim is not None:
            # ground-state field -> its perturbation (rational closure for
            # inv_rho: no independent operand, folds through rho^X)
            return from_expr(_pert_atom(prim, label))
        if atom.name in LIBXC_MULTISET:
            # Libxc derivative: bump by each active variable Y, times Y^X
            # (Y^X derived from the FUNCTIONAL's own ingredient, so mapped
            # variables like the gauge-corrected tau chain correctly)
            ms = LIBXC_MULTISET[atom.name]
            d = sp.Integer(0)
            for Y in VARS:
                ing = by_name.get(Y)
                if ing is not None:
                    d += libxc_symbol(ms + Counter({Y: 1})) \
                        * perturbed_ingredient(ing, label)
            return from_expr(d)
        return None  # basis data, weight, other perturbations' fields
    return seed


def contracted_derivative(expr: sp.Expr, func: Functional,
                          label: str) -> sp.Expr:
    """Apply D_X = sum_ts D^X_ts d/dP_ts to an integrand, with the contraction
    folded into perturbed-field symbols labelled ``label``.

    Implemented monomial-wise (fastpoly) -- the generic sympy.diff-per-atom
    path is intractable for high-order kernels."""
    from .fastpoly import from_expr, seeded_derivative, to_expr
    return to_expr(seeded_derivative(from_expr(expr), _seed_fn(func, label)))


@dataclass
class ResponseIntegrand:
    """Per-grid integrand of an m-th order response Fock matrix element with
    free orbital pair (u, v) and perturbation labels p1..pm.

    High-order integrands carry the monomial dictionary (``poly``) instead of
    a materialized SymPy expression -- building the expression costs more than
    the derivative itself.  ``expr`` is computed lazily on access."""

    functional: Functional
    labels: List[str]
    index_pairs: List[Tuple[str, str]]   # single free pair, for codegen
    _expr: "sp.Expr | None" = None
    poly: "dict | None" = None

    @property
    def expr(self) -> sp.Expr:
        if self._expr is None and self.poly is not None:
            from .fastpoly import to_expr
            self._expr = to_expr(self.poly)
        return self._expr


def response_fock(family: str, order: int = 2,
                  u: str = "u", v: str = "v") -> ResponseIntegrand:
    """Integrand of the response Fock matrix at a given derivative order.

    order=1 is the plain XC Fock (no perturbations); order=2 the fxc
    contraction with one perturbed DM (linear response); order=3 the kxc
    contraction with two perturbed DMs (quadratic response); and so on.
    """
    if order < 1:
        raise ValueError("order must be >= 1")
    from .fastpoly import from_expr, seeded_derivative, to_expr
    fi = fock_integrand(family, u, v)
    func = fi.functional
    labels = [f"p{k}" for k in range(1, order)]
    poly = from_expr(fi.expr)
    for label in labels:
        poly = seeded_derivative(poly, _seed_fn(func, label))
    return ResponseIntegrand(functional=func, labels=labels,
                             index_pairs=[(u, v)], poly=poly)
