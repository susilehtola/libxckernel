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
                   v_label: str = "v",
                   coords=None) -> FockIntegrand:
    """Build the per-grid integrand of F_uv for a functional family.

    ``coords`` selects the coordinate system the basis derivatives live
    in; the default is Cartesian, which reproduces the historical
    integrand exactly. In a curvilinear system the ingredient seeds carry
    the Lame factors, so the emitted Fock build needs no scale factors of
    its own.
    """
    func = Functional.of_family(family, coords)
    u = Orbital.make(u_label, coords)
    v = Orbital.make(v_label, coords)
    w = sp.Symbol("w", real=True, positive=True)

    integrand = sp.Integer(0)
    for ing in func.ingredients:
        integrand += func.vsymbol(ing) * ing.seed(u, v)
    integrand = sp.expand(w * integrand)

    return FockIntegrand(functional=func, u=u, v=v, weight=w, expr=integrand)


def vxc_channels(family: str, coords=None) -> "dict[str, sp.Expr]":
    """Per-point coefficient channels of the GROUND-STATE XC potential.

    ``fock_integrand`` gives dExc/dP_uv as one expression in the basis
    data of the pair (u, v).  A host that assembles the XC matrix does
    not want that form: it wants the coefficient that multiplies each
    ingredient SEED, so it can contract the coefficient field against
    the basis-function products it already has,

        dExc/dP_uv = sum_g w_g [ u_g chi_u chi_v
                                 + sum_i v_{i,g} d_i(chi_u chi_v)/h_i
                                 + w_g (tau seed)_uv ] .

    Those coefficients are the derivatives of the energy density with
    respect to the PRIMITIVE fields,

        u = dExc/d rho,  v_i = dExc/d (grad rho)_i,  w = dExc/d tau,

    obtained here by the chain rule through the ingredients rather than
    written out by hand: for a GGA v_i comes out as 2 vsigma (grad
    rho)_i, which is exactly the ``build_vgrad`` every host codes by
    itself, and for the gauge-corrected and Hessian families it comes
    out equally automatically.  The first-order counterpart of
    :func:`~xckernel.engine.response.fxc_channels`, and the same shape,
    so a host assembles the potential and the kernel with one routine.

    Keys are ``rho``, ``grad_<axis>`` per component of the coordinate
    system, and ``tau``/``lapl`` for the families that carry them.
    """
    from ..inputs.ingredients import prim_by_symbol_for

    func = Functional.of_family(family, coords)
    energy = sum((func.vsymbol(ing) * ing.value
                  for ing in func.ingredients), sp.Integer(0))
    out = {}
    for key, prim in _channel_primitives(energy, coords):
        out[key] = sp.expand(sp.diff(energy, prim.symbol))
    return out


#: Libxc-facing channel key for a primitive.  The common ones keep the
#: names ``fxc_channels`` uses, so a host consumes the potential and the
#: kernel through one layout; anything else (the paramagnetic current
#: density, the density Hessian, the 1/rho pseudo-primitive) keeps its
#: own primitive name.
def _channel_key(name: str) -> str:
    if name.startswith("grad_rho_"):
        return "grad_" + name[len("grad_rho_"):]
    if name == "lapl_rho":
        return "lapl"
    return name


def _channel_primitives(energy, coords):
    """The primitive fields the energy actually depends on, in a stable
    order, paired with their channel keys."""
    from ..inputs.ingredients import prim_by_symbol_for
    by_sym = prim_by_symbol_for(coords)
    seen, out = set(), []
    for sym in sorted(energy.free_symbols, key=lambda a: a.name):
        prim = by_sym.get(sym)
        if prim is None or prim.name in seen:
            continue
        seen.add(prim.name)
        out.append((_channel_key(prim.name), prim))
    return out


def check_vxc_channels(family: str, coords=None) -> sp.Expr:
    """Residual between the channel form and ``fock_integrand``.

    The two must be the same expression: the channels are only a
    regrouping of dExc/dP_uv by seed.  Returns the (expanded) difference,
    which is identically zero when they agree.
    """
    fi = fock_integrand(family, coords=coords)
    func = fi.functional
    energy = sum((func.vsymbol(ing) * ing.value
                  for ing in func.ingredients), sp.Integer(0))
    ch = vxc_channels(family, coords)
    total = sp.Integer(0)
    for key, prim in _channel_primitives(energy, coords):
        total += ch[key] * prim.seed(fi.u, fi.v)
    return sp.expand(fi.expr / fi.weight - total)
