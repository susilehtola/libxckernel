"""Geometric (nuclear-displacement) derivatives of the XC tower: the
force-aware layer.

Under a nuclear displacement X = (A, d) the molecular-grid XC integral

    E = sum_g w_g(R) e(fields(r_g; R))

depends on R three ways, and the total derivative splits into three term
classes, each generated here and dispatched separately:

* **basis class** -- the basis functions on atom A move (the fixed-grid
  convention of most Hessian codes).  Two kinds of term: the functional
  chain contracted with host-supplied fixed-grid perturbed fields
  (rho_p1, grad_rho_p1, tau_p1: the atom-restricted collocation
  derivatives of the fields, e.g. rho^X = -2 sum_{u in A} (D+D^T)_uv
  dchi_u chi_v), and the seed-derivative terms where the displacement
  hits the kernel's own free-index collocation factors, carried by the
  ATOM-MASKED operands dchi_gA (nbf, ng) = -dchi_d restricted to atom A
  and ddchi_gA (3, nbf, ng) = -d_d grad chi restricted to atom A.  (The
  minus signs of d chi/d X_A = -d chi/d r are folded INTO the masked
  operands, mirroring production practice.)

* **grid class** -- grid point g rides its parent atom: the term is
  w_g M^A_g d_d e(r_g), with M^A the parent-atom mask.  d_d e is the
  SPATIAL GRADIENT of the integrand density, an ordinary field
  expression; the host folds w*M^A into the weight operand, so a single
  generated kernel serves every atom and direction.  Direction-resolved
  operands: drho_g (ng,), dgrad_rho_g (3, ng), dtau_g (ng,) are the
  d-components of grad rho, (grad grad rho) rows, and grad tau; the
  basis factors differentiate to dchi_g / ddchi_g (unmasked analogues of
  the basis-class operands, without the sign fold).

* **weight class** -- the Becke partition weights change: the ORIGINAL
  kernel evaluated with w := dw/dX.  No new kernel is generated; the
  dispatch table simply says so.

Translational invariance ties the classes together: summed over all
atoms A (with sum_A M^A = 1 and sum_A dw/dX_A = 0) the three classes
cancel exactly -- the recommended validation for any host wiring.

The operators act on ANY tower integrand (energy, Fock, response
kernels), so higher geometric derivatives compose mechanically.
"""

from __future__ import annotations

import re as _re
from collections import Counter
from dataclasses import dataclass

import sympy as sp

from .basis import AXES
from .deriv import LIBXC_MULTISET, VARS, libxc_symbol
from .fock import fock_integrand
from .functional import Functional
from .ingredients import PRIM_BY_SYMBOL
from .kernel import KernelIntegrand
from .response import _seed_fn

_CHI_RE = _re.compile(r"^chi_(\w+)$")
_DCHI_RE = _re.compile(r"^dchi_(\w+)_([xyz])$")

# --- direction-resolved spatial-gradient operands ----------------------------

#: d-component of grad rho (the host slices its gradient array).
DRHO_G = sp.Symbol("drho_g", real=True)
#: d-row of the density Hessian, (3, ng): dgrad_rho_g[i] = d_d d_i rho.
DGRAD_RHO_G = tuple(sp.Symbol(f"dgrad_rho_g_{ax}", real=True) for ax in AXES)
#: d-component of grad tau.
DTAU_G = sp.Symbol("dtau_g", real=True)


def _spatial_field_gradient(var: str) -> sp.Expr:
    """d_d of a Libxc input variable, in direction-resolved operands."""
    from .ingredients import GRAD_RHO
    if var == "rho":
        return DRHO_G
    if var == "tau":
        return DTAU_G
    if var == "sigma":
        return 2 * sum(p.symbol * DGRAD_RHO_G[i] for i, p in enumerate(GRAD_RHO))
    raise ValueError(f"spatial gradient of {var!r} not supported yet "
                     "(lda/gga/mgga_tau families)")


def _spatial_seed(func: Functional):
    """Monomial-level seed for the spatial-gradient operator d_d."""
    from .fastpoly import from_expr
    by_name = {ing.name: ing for ing in func.ingredients}

    def seed(atom: sp.Symbol):
        name = atom.name
        prim = PRIM_BY_SYMBOL.get(atom)
        if prim is not None:
            if prim.name == "rho":
                return from_expr(DRHO_G)
            if prim.name.startswith("grad_rho_"):
                i = AXES.index(prim.name[-1])
                return from_expr(DGRAD_RHO_G[i])
            if prim.name == "tau":
                return from_expr(DTAU_G)
            raise ValueError(f"spatial gradient of primitive {prim.name!r} "
                             "not supported yet")
        if name in LIBXC_MULTISET:
            ms = LIBXC_MULTISET[name]
            total = sp.Integer(0)
            for Y in VARS:
                if by_name.get(Y) is not None:
                    total += libxc_symbol(ms + Counter({Y: 1})) \
                        * _spatial_field_gradient(Y)
            return from_expr(total)
        # basis factors: chi_u -> dchi_g_u; dchi_u_i -> ddchi_g_u_i
        m = _CHI_RE.match(name)
        if m:
            return from_expr(sp.Symbol(f"dchi_g_{m.group(1)}", real=True))
        m = _DCHI_RE.match(name)
        if m:
            return from_expr(sp.Symbol(f"ddchi_g_{m.group(1)}_{m.group(2)}",
                                       real=True))
        if name.startswith("lapl_chi"):
            raise ValueError("spatial gradient of lapl_chi needs "
                             "third-derivative collocation (not wired yet)")
        return None  # weight, perturbed fields of other labels
    return seed


def spatial_gradient(ki: KernelIntegrand) -> KernelIntegrand:
    """d_d of a tower integrand's density (w treated as the quadrature
    datum it is).  This is the grid-motion class: the host calls the
    generated kernel with weight operand w := w * M^A (parent-atom mask
    folded in), once per direction with sliced operands."""
    from .fastpoly import from_expr, seeded_derivative, to_expr
    poly = getattr(ki, "poly", None)
    if poly is None:
        poly = from_expr(sp.expand(ki.expr))
    out = seeded_derivative(poly, _spatial_seed(ki.functional))
    return KernelIntegrand(functional=ki.functional,
                           index_pairs=list(ki.index_pairs),
                           expr=to_expr(out))


# --- the basis (fixed-grid) class --------------------------------------------

def _geometric_seed(func: Functional):
    """Seed for the basis class: functional chain against fixed-grid
    perturbed fields (label p1) plus atom-masked seed-derivative operands
    (sign of d chi/d X_A folded into dchi_gA / ddchi_gA)."""
    from .fastpoly import from_expr
    resp = _seed_fn(func, "p1")

    def seed(atom: sp.Symbol):
        name = atom.name
        m = _CHI_RE.match(name)
        if m:
            return from_expr(sp.Symbol(f"dchi_gA_{m.group(1)}", real=True))
        m = _DCHI_RE.match(name)
        if m:
            return from_expr(sp.Symbol(f"ddchi_gA_{m.group(1)}_{m.group(2)}",
                                       real=True))
        if name.startswith("lapl_chi"):
            raise ValueError("geometric seed of lapl_chi needs "
                             "third-derivative collocation (not wired yet)")
        return resp(atom)
    return seed


def geometric_fock(family: str, u: str = "u", v: str = "v") -> KernelIntegrand:
    """Basis-class integrand of dF_uv/dX at fixed density and fixed grid.

    Operands: perturbed fields rho_p1 / grad_rho_p1 / tau_p1 (the
    atom-restricted fixed-grid field derivatives) and the atom-masked
    collocation derivatives dchi_gA (nbf, ng) / ddchi_gA (3, nbf, ng),
    both carrying the -d/dr sign."""
    from .fastpoly import from_expr, seeded_derivative, to_expr
    fi = fock_integrand(family, u, v)
    out = seeded_derivative(from_expr(fi.expr), _geometric_seed(fi.functional))
    return KernelIntegrand(functional=fi.functional, index_pairs=[(u, v)],
                           expr=to_expr(out))


# --- dispatch ----------------------------------------------------------------

@dataclass
class GeometricDispatch:
    """The three term classes of a force-aware geometric derivative.

    basis:  new integrand (fixed-grid convention alone stops here);
    grid:   new integrand, call with weight := w * M^A;
    weight: the ORIGINAL integrand, call with weight := dw/dX.
    """
    basis: KernelIntegrand
    grid: KernelIntegrand
    weight: KernelIntegrand


def geometric_dispatch(family: str) -> GeometricDispatch:
    """Everything a host needs for the full geometric derivative of the
    XC Fock matrix, quadrature dependence included."""
    from .kernel import fock
    fi = fock(family)
    return GeometricDispatch(basis=geometric_fock(family),
                             grid=spatial_gradient(fi),
                             weight=fi)
