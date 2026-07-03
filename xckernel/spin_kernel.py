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
