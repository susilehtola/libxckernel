"""Spin-polarized Fock and kernel: packing, D^s operator, assembly.

Libxc packs each derivative array by component.  For a derivative with respect
to a multiset of scalar variables we need (i) the array name (v2rhosigma, ...)
and (ii) the flat component index inside it.  The array name is fixed by the
*group* multiset (rho, sigma, ...); the component index enumerates the scalar
components in Libxc's order: product over the canonically-ordered group list,
with nondecreasing component indices within any run of identical groups (so
same-group second derivatives collapse to the symmetric aa/ab/bb packing).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Tuple

import sympy as sp

from .basis import Orbital
from .deriv import libxc_deriv_name
from .spin import (COMPS, GROUPS, SCALARS, Scalar, _GROUP_RANK,
                   family_scalars, GRAD)


# --- component packing ------------------------------------------------------

def _group_list(group_ms: Counter) -> List[str]:
    """Group multiset -> canonical ordered list, e.g. {rho:1,sigma:1}->[rho,sigma]."""
    out: List[str] = []
    for g in GROUPS:
        out += [g] * group_ms.get(g, 0)
    return out


def _comp_tuples(glist: List[str]) -> List[Tuple[int, ...]]:
    """All valid component-index tuples for a canonical group list, in Libxc order."""
    ranges = [range(len(COMPS[g])) for g in glist]
    result = []
    for combo in product(*ranges):
        ok = all(not (glist[i] == glist[i + 1] and combo[i] > combo[i + 1])
                 for i in range(len(glist) - 1))
        if ok:
            result.append(combo)
    return result


def scalars_to_symbol(scalars: Tuple[Scalar, ...]) -> sp.Symbol:
    """Libxc derivative symbol '<array>_<flatcomp>' for a scalar multiset."""
    ordered = sorted(scalars, key=lambda sc: sc.key)
    group_ms = Counter(sc.group for sc in ordered)
    array = libxc_deriv_name(group_ms)                 # vrho / v2rhosigma / ...
    glist = _group_list(group_ms)
    comp_idx = tuple(COMPS[sc.group].index(sc.comp) for sc in ordered)
    flat = _comp_tuples(glist).index(comp_idx)
    return sp.Symbol(f"{array}_{flat}", real=True)


# Registry: derivative symbol name -> the scalar multiset it stands for, so the
# operator can bump it.  Built lazily as symbols are created.
_SYM_SCALARS: Dict[str, Tuple[Scalar, ...]] = {}


def _register(scalars: Tuple[Scalar, ...]) -> sp.Symbol:
    sym = scalars_to_symbol(scalars)
    _SYM_SCALARS[sym.name] = tuple(sorted(scalars, key=lambda sc: sc.key))
    return sym


# --- grad-field seed lookup (P-dependent atoms that appear in expressions) --

_GRAD_INFO = {}   # symbol -> (spin, axis)
for _s in ("a", "b"):
    for _ax in range(3):
        _GRAD_INFO[GRAD[_s][_ax]] = (_s, _ax)


def _grad_seed(spin: str, ax: int, dspin: str, u: Orbital, v: Orbital) -> sp.Expr:
    if dspin != spin:
        return sp.Integer(0)
    return u.grad[ax] * v.val + u.val * v.grad[ax]


# --- the spin-resolved directional derivative D^dspin ----------------------

def directional_derivative(expr: sp.Expr, family: str, dspin: str,
                           u_label: str, v_label: str) -> sp.Expr:
    """Apply D^dspin_{u,v} = d/dP^dspin_uv to an integrand expression."""
    u = Orbital.make(u_label)
    v = Orbital.make(v_label)
    scalars = family_scalars(family)
    result = sp.Integer(0)
    for atom in expr.free_symbols:
        # spin-resolved gradient field
        if atom in _GRAD_INFO:
            spin, ax = _GRAD_INFO[atom]
            d = _grad_seed(spin, ax, dspin, u, v)
        # Libxc derivative symbol: bump by each family scalar Y
        elif atom.name in _SYM_SCALARS:
            base = _SYM_SCALARS[atom.name]
            d = sp.Integer(0)
            for Y in scalars:
                d += _register(base + (Y,)) * Y.seed(dspin, u, v)
        else:
            d = sp.Integer(0)  # basis data / weight
        if d != 0:
            result += sp.diff(expr, atom) * d
    return sp.expand(result)


# --- assembly ---------------------------------------------------------------

# --- response contraction (perturbed-field seeds) ---------------------------

def _pert_grad(spin: str, label: str, ax: str) -> sp.Symbol:
    return sp.Symbol(f"grad_rho_{spin}_{label}_{ax}", real=True)


def pert_scalar_value(sc: Scalar, label: str) -> sp.Expr:
    """Perturbed value of a Libxc scalar variable under perturbation ``label``.

    A perturbation carries BOTH spin channels (D^{X,a}, D^{X,b}); the perturbed
    sigma components mix them: sigma_st^X = grad_s^X . grad_t + grad_s . grad_t^X
    (which reduces to 2 grad_s . grad_s^X for the same-spin components).
    """
    from .basis import AXES
    if sc.group == "rho":
        return sp.Symbol(f"rho_{sc.comp}_{label}", real=True)
    if sc.group == "lapl":
        return sp.Symbol(f"lapl_rho_{sc.comp}_{label}", real=True)
    if sc.group == "tau":
        return sp.Symbol(f"tau_{sc.comp}_{label}", real=True)
    # sigma
    from .spin import COMP_SPINS
    s1, s2 = COMP_SPINS["sigma"][sc.comp]
    total = sp.Integer(0)
    for i, ax in enumerate(AXES):
        total += _pert_grad(s1, label, ax) * GRAD[s2][i] \
            + GRAD[s1][i] * _pert_grad(s2, label, ax)
    return total


def _seed_fn_spin(family: str, label: str):
    """Monomial-level seed map for the spin engine (fastpoly)."""
    from .basis import AXES
    from .fastpoly import from_expr
    scalars = family_scalars(family)

    def seed(atom: sp.Symbol):
        if atom in _GRAD_INFO:
            spin, ax = _GRAD_INFO[atom]
            return {((_pert_grad(spin, label, AXES[ax]), 1),): sp.Integer(1)}
        if atom.name in _SYM_SCALARS:
            base = _SYM_SCALARS[atom.name]
            d = sp.Integer(0)
            for Y in scalars:
                d += _register(base + (Y,)) * pert_scalar_value(Y, label)
            return from_expr(d)
        return None  # basis data, weight, other perturbations' fields
    return seed


def contracted_derivative(expr: sp.Expr, family: str, label: str) -> sp.Expr:
    """Apply D_X = sum_s sum_ts D^{X,s}_ts d/dP^s_ts with the contraction
    folded into spin-resolved perturbed-field symbols.

    Implemented monomial-wise (fastpoly); the generic sympy.diff path is
    intractable at high order."""
    from .fastpoly import from_expr, seeded_derivative, to_expr
    return to_expr(seeded_derivative(from_expr(expr),
                                     _seed_fn_spin(family, label)))


@dataclass
class SpinIntegrand:
    family: str
    spins: List[str]                 # differentiation spin per order
    index_pairs: List[Tuple[str, str]]
    expr: sp.Expr


def fock_spin(family: str, spin: str, u: str = "u", v: str = "v") -> SpinIntegrand:
    """F^spin_uv = dExc/dP^spin_uv."""
    U, V = Orbital.make(u), Orbital.make(v)
    w = sp.Symbol("w", real=True, positive=True)
    expr = sp.Integer(0)
    for X in family_scalars(family):
        expr += _register((X,)) * X.seed(spin, U, V)
    expr = sp.expand(w * expr)
    return SpinIntegrand(family, [spin], [(u, v)], expr)


def kernel_spin(family: str, spin1: str, spin2: str,
                u="u", v="v", t="t", s="s") -> SpinIntegrand:
    """g^{spin1 spin2}_uv,ts = dExc/dP^spin1_uv dP^spin2_ts."""
    fi = fock_spin(family, spin1, u, v)
    expr = directional_derivative(fi.expr, family, spin2, t, s)
    return SpinIntegrand(family, [spin1, spin2], [(u, v), (t, s)], expr)


@dataclass
class SpinResponseIntegrand:
    """m-th order spin-resolved response Fock integrand: free pair (u,v) in
    spin channel ``spin``, contracted with perturbations p1..pm (each carrying
    both spin channels of a perturbed DM pair)."""

    family: str
    spin: str
    labels: List[str]
    index_pairs: List[Tuple[str, str]]
    expr: sp.Expr


def response_fock_spin(family: str, spin: str, order: int = 2,
                       u: str = "u", v: str = "v") -> SpinResponseIntegrand:
    """Spin-channel response Fock at a given derivative order.

    order=1: F^s_uv; order=2: the fxc contraction sum_t g^{st} : D^{X,t}
    (linear response); order=3: quadratic response; etc."""
    if order < 1:
        raise ValueError("order must be >= 1")
    from .fastpoly import from_expr, seeded_derivative, to_expr
    fi = fock_spin(family, spin, u, v)
    labels = [f"p{k}" for k in range(1, order)]
    poly = from_expr(fi.expr)
    for label in labels:
        poly = seeded_derivative(poly, _seed_fn_spin(family, label))
    return SpinResponseIntegrand(family=family, spin=spin, labels=labels,
                                 index_pairs=[(u, v)], expr=to_expr(poly))


def response_fock_st(family: str, order: int = 2,
                     parities: Tuple[int, ...] = None,
                     u: str = "u", v: str = "v") -> SpinResponseIntegrand:
    """Spin-adapted response Fock from a closed-shell reference.

    Each perturbation carries a spin parity: +1 (singlet-type, D^{X,b} =
    +D^{X,a}) or -1 (triplet-type, D^{X,b} = -D^{X,a}).  Substituting the
    parity constraint and the closed-shell ground state (beta fields equal
    alpha fields) into the alpha-channel spin response produces the
    singlet/triplet kernel combinations mechanically at any order -- e.g. at
    order 2 the familiar fxc^{aa} +/- fxc^{ab}, and at order 3 Dalton's
    spin-parity-bit algebra.  All fields and perturbed fields in the result
    are ALPHA-channel quantities; the Libxc derivative components are the
    polarized ones evaluated at the closed-shell density (rho/2, rho/2).
    """
    if parities is None:
        parities = (+1,) * (order - 1)
    if len(parities) != order - 1:
        raise ValueError("need one parity per perturbation")
    if any(p not in (-1, +1) for p in parities):
        raise ValueError("parities must be +1 (singlet) or -1 (triplet)")

    from .basis import AXES
    ri = response_fock_spin(family, "a", order, u, v)
    subs = {}
    # closed-shell ground state: beta gradient fields -> alpha fields
    for i in range(3):
        subs[GRAD["b"][i]] = GRAD["a"][i]
    # perturbation parity: beta perturbed fields -> parity * alpha fields
    for label, par in zip(ri.labels, parities):
        for base in ("rho", "lapl_rho", "tau"):
            subs[sp.Symbol(f"{base}_b_{label}", real=True)] = \
                par * sp.Symbol(f"{base}_a_{label}", real=True)
        for ax in AXES:
            subs[_pert_grad("b", label, ax)] = par * _pert_grad("a", label, ax)
    expr = sp.expand(ri.expr.subs(subs, simultaneous=True))
    return SpinResponseIntegrand(family=family, spin="a", labels=ri.labels,
                                 index_pairs=[(u, v)], expr=expr)
