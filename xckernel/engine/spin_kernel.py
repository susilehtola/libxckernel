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

from ..inputs.basis import Orbital
from .deriv import libxc_deriv_name
from .spin import (COMPS, FIELD_PERTS, FIELD_SEEDS, GROUPS, HESS_S,
                   INV_RHO_S, JP_S, SCALARS, Scalar, _GROUP_RANK,
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


# P-dependent field atoms (gradient, current, inverse-density, Hessian
# fields) are seeded through the spin.FIELD_SEEDS registry.


# --- the spin-resolved directional derivative D^dspin ----------------------

def directional_derivative(expr: sp.Expr, family: str, dspin: str,
                           u_label: str, v_label: str) -> sp.Expr:
    """Apply D^dspin_{u,v} = d/dP^dspin_uv to an integrand expression.

    Applied monomial-wise via the fastpoly representation, like every
    other derivative operator of the library."""
    from .fastpoly import from_expr, seeded_derivative, to_expr
    u = Orbital.make(u_label)
    v = Orbital.make(v_label)
    scalars = family_scalars(family)

    def seed(atom: sp.Symbol):
        # spin-resolved P-dependent field
        if atom in FIELD_SEEDS:
            d = FIELD_SEEDS[atom](dspin, u, v)
        # Libxc derivative symbol: bump by each family scalar Y
        elif atom.name in _SYM_SCALARS:
            base = _SYM_SCALARS[atom.name]
            d = sp.Integer(0)
            for Y in scalars:
                d += _register(base + (Y,)) * Y.seed(dspin, u, v)
        else:
            return None  # basis data / weight
        return from_expr(d) if d != 0 else None

    return to_expr(seeded_derivative(from_expr(expr), seed))


# --- assembly ---------------------------------------------------------------

# --- response contraction (perturbed-field seeds) ---------------------------

def _pert_grad(spin: str, label: str, ax: str) -> sp.Symbol:
    return sp.Symbol(f"grad_rho_{spin}_{label}_{ax}", real=True)


def pert_scalar_value(sc: Scalar, label: str) -> sp.Expr:
    """Perturbed value of a Libxc scalar variable under perturbation ``label``.

    A perturbation carries BOTH spin channels (D^{X,a}, D^{X,b}); the perturbed
    sigma components mix them: sigma_st^X = grad_s^X . grad_t + grad_s . grad_t^X
    (which reduces to 2 grad_s . grad_s^X for the same-spin components).
    Composite scalars (gauge-corrected tau, eta) carry their own pert closure.
    """
    from ..inputs.basis import AXES
    if sc.pert is not None:
        return sc.pert(label)
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


def pert2_scalar_value(sc: Scalar, l1: str, l2: str) -> sp.Expr:
    """Second-order perturbed value of a polarized Libxc scalar under
    two independent perturbations: nonzero only through the scalar's
    own nonlinearity (the sigma components; composite scalars carry
    their own closure if they define one)."""
    from ..inputs.basis import AXES
    if sc.group != "sigma":
        return sp.Integer(0)
    from .spin import COMP_SPINS
    s1, s2 = COMP_SPINS["sigma"][sc.comp]
    total = sp.Integer(0)
    for ax in AXES:
        total += _pert_grad(s1, l1, ax) * _pert_grad(s2, l2, ax) \
            + _pert_grad(s2, l1, ax) * _pert_grad(s1, l2, ax)
    return total


def fxc_bilinear_spin(family: str, l1: str = "p1",
                      l2: str = "p2") -> sp.Expr:
    """Per-point integrand of the spin-polarized XC kernel bilinear
    form between two perturbations (each carrying both spin channels),

        d2 Exc / dl1 dl2 = sum_KL v2_KL K^l1 L^l2 + sum_K v_K K^{l1 l2},

    over the polarized Libxc scalar variables K, L of the family; the
    host multiplies by the quadrature weight and sums.  The polarized
    counterpart of response.fxc_bilinear."""
    from .spin import family_scalars
    scalars = family_scalars(family)
    total = sp.Integer(0)
    for K in scalars:
        for L in scalars:
            total += _register((K, L)) \
                * pert_scalar_value(K, l1) * pert_scalar_value(L, l2)
        total += _register((K,)) * pert2_scalar_value(K, l1, l2)
    return sp.expand(total)


def fxc_bilinear_st(family: str, parities: "Tuple[int, int]" = (+1, +1),
                    l1: str = "p1", l2: str = "p2") -> sp.Expr:
    """Closed-shell spin-adapted fxc bilinear: the polarized bilinear
    at the spin-compensated point, with each perturbation carrying a
    spin parity (+1 singlet-type, -1 triplet-type).  All fields and
    perturbed fields in the result are ALPHA-channel quantities, and
    the Libxc derivative components are the polarized ones evaluated
    at the closed-shell density (rho/2, rho/2).  Mixed parities
    vanish identically (checked by the validation suite); the
    (-1, -1) combination is the triplet Casida XC kernel."""
    from ..inputs.basis import AXES
    from .fastpoly import from_expr, subs_signed, to_expr

    if any(p not in (-1, +1) for p in parities):
        raise ValueError("parities must be +1 (singlet) or -1 (triplet)")
    b = fxc_bilinear_spin(family, l1, l2)
    mapping = {}
    for i in range(3):
        mapping[GRAD["b"][i]] = (GRAD["a"][i], +1)
    for base in ("rho", "lapl_rho", "tau"):
        mapping[sp.Symbol(f"{base}_b", real=True)] = \
            (sp.Symbol(f"{base}_a", real=True), +1)
    for label, par in zip((l1, l2), parities):
        for base in ("rho", "lapl_rho", "tau"):
            mapping[sp.Symbol(f"{base}_b_{label}", real=True)] = \
                (sp.Symbol(f"{base}_a_{label}", real=True), par)
        for ax in AXES:
            mapping[_pert_grad("b", label, ax)] = \
                (_pert_grad("a", label, ax), par)
    expr = to_expr(subs_signed(from_expr(b), mapping))

    # closed-shell symmetry of the polarized derivative arrays: at the
    # spin-compensated point, components related by a global a<->b
    # exchange are equal; canonicalize to the lexicographically first
    # partner so parity cancellations become identities (mixed-parity
    # bilinears vanish symbolically).
    from .spin import SCALARS
    _swap = {"a": "b", "b": "a", "aa": "bb", "bb": "aa", "ab": "ab"}
    canon = {}
    for atom in expr.free_symbols:
        scalars = _SYM_SCALARS.get(atom.name)
        if scalars is None:
            continue
        partner = scalars_to_symbol(tuple(
            SCALARS[(sc.group, _swap[sc.comp])] for sc in scalars))
        if partner.name < atom.name:
            canon[atom] = partner
    return sp.expand(expr.subs(canon, simultaneous=True))


def fxc_channels_st(family: str, parity: int = -1,
                    label: str = "p1") -> "dict[str, sp.Expr]":
    """Per-point coefficient channels of the closed-shell spin-adapted
    fxc contraction with one perturbation of the given parity
    (-1: triplet, the spin-flip Casida kernel): derivatives of the st
    bilinear with respect to the second perturbation's alpha-channel
    operands, keyed 'rho'/'grad_x'/../'tau'."""
    from ..inputs.basis import AXES
    b = fxc_bilinear_st(family, (parity, parity), l1=label, l2="_q")
    out = {"rho": sp.expand(
        sp.diff(b, sp.Symbol("rho_a__q", real=True)))}
    for ax in AXES:
        out[f"grad_{ax}"] = sp.expand(
            sp.diff(b, sp.Symbol(f"grad_rho_a__q_{ax}", real=True)))
    tau_q = sp.Symbol("tau_a__q", real=True)
    if b.has(tau_q):
        out["tau"] = sp.expand(sp.diff(b, tau_q))
    return out


def fxc_channels_spin(family: str,
                      label: str = "p1") -> "dict[str, sp.Expr]":
    """Per-point coefficient channels of the polarized fxc contraction
    with one perturbation: derivatives of the spin bilinear with
    respect to the second perturbation's operands, keyed
    'rho_a'/'rho_b', 'grad_a_x'/... and (tau families)
    'tau_a'/'tau_b'.  The polarized counterpart of
    response.fxc_channels."""
    from ..inputs.basis import AXES
    b = fxc_bilinear_spin(family, l1=label, l2="_q")
    out = {}
    for s in ("a", "b"):
        out[f"rho_{s}"] = sp.expand(
            sp.diff(b, sp.Symbol(f"rho_{s}__q", real=True)))
        for ax in AXES:
            out[f"grad_{s}_{ax}"] = sp.expand(
                sp.diff(b, sp.Symbol(f"grad_rho_{s}__q_{ax}", real=True)))
        tau_q = sp.Symbol(f"tau_{s}__q", real=True)
        if b.has(tau_q):
            out[f"tau_{s}"] = sp.expand(sp.diff(b, tau_q))
    return out


def _seed_fn_spin(family: str, label: str):
    """Monomial-level seed map for the spin engine (fastpoly)."""
    from .fastpoly import from_expr
    scalars = family_scalars(family)

    def seed(atom: sp.Symbol):
        if atom in FIELD_PERTS:
            return from_expr(FIELD_PERTS[atom](label))
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
    both spin channels of a perturbed DM pair).

    Carries the monomial dictionary (``poly``); ``expr`` materialized lazily."""

    family: str
    spin: str
    labels: List[str]
    index_pairs: List[Tuple[str, str]]
    _expr: "sp.Expr | None" = None
    poly: "dict | None" = None

    @property
    def expr(self) -> sp.Expr:
        if self._expr is None and self.poly is not None:
            from .fastpoly import to_expr
            self._expr = to_expr(self.poly)
        return self._expr


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
                                 index_pairs=[(u, v)], poly=poly)


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

    from ..inputs.basis import AXES, HESS_COMPS
    from .fastpoly import subs_signed
    ri = response_fock_spin(family, "a", order, u, v)
    mapping = {}
    # closed-shell ground state: beta fields -> alpha fields
    for i in range(3):
        mapping[GRAD["b"][i]] = (GRAD["a"][i], +1)
        mapping[JP_S["b"][i]] = (JP_S["a"][i], +1)
    mapping[INV_RHO_S["b"]] = (INV_RHO_S["a"], +1)
    for k in range(6):
        mapping[HESS_S["b"][k]] = (HESS_S["a"][k], +1)
    # perturbation parity: beta perturbed fields -> parity * alpha fields
    _h6 = [f"{AXES[i]}{AXES[j]}" for (i, j) in HESS_COMPS]
    for label, par in zip(ri.labels, parities):
        for base in ("rho", "lapl_rho", "tau"):
            mapping[sp.Symbol(f"{base}_b_{label}", real=True)] = \
                (sp.Symbol(f"{base}_a_{label}", real=True), par)
        for ax in AXES:
            mapping[_pert_grad("b", label, ax)] = \
                (_pert_grad("a", label, ax), par)
            mapping[sp.Symbol(f"jp_b_{label}_{ax}", real=True)] = \
                (sp.Symbol(f"jp_a_{label}_{ax}", real=True), par)
        for comp in _h6:
            mapping[sp.Symbol(f"hess_rho_b_{label}_{comp}", real=True)] = \
                (sp.Symbol(f"hess_rho_a_{label}_{comp}", real=True), par)
    poly = subs_signed(ri.poly, mapping)
    return SpinResponseIntegrand(family=family, spin="a", labels=ri.labels,
                                 index_pairs=[(u, v)], poly=poly)
