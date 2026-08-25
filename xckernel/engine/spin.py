"""Spin-polarized core: density-matrix derivatives with spin channels.

Open-shell DFT resolves every ingredient by spin.  The density matrix splits
into P^a and P^b; the Libxc variables split into components

    rho   -> rho_a, rho_b
    sigma -> sigma_aa, sigma_ab, sigma_bb   (grad rho_s . grad rho_t)
    lapl  -> lapl_a, lapl_b
    tau   -> tau_a, tau_b

The Fock matrix is spin-resolved, F^s_uv = dExc/dP^s_uv, and the kernel gains a
spin-pair label, g^{st}_uv,ts = dExc/dP^s_uv dP^t_ts.

Everything is still the derivative tower of deriv.py, now with a spin index on
the operator: D^s_uv = d/dP^s_uv.  A scalar Libxc variable X=(group, comp) has a
seed dX/dP^s that is nonzero only for the spin(s) that variable depends on
(encoded in its component), and differentiating a Libxc derivative symbol bumps
it by one scalar variable Y and multiplies by dY/dP^s -- exactly as before, but
now the Libxc symbols carry a packed component index matching Libxc's output
arrays (vrho[:,c], v2rhosigma[:,c], ...).
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from dataclasses import dataclass
from itertools import product
from typing import Callable, Dict, List, Tuple

import sympy as sp

from ..inputs import ingredients as _ing
from ..inputs.basis import AXES, HESS_COMPS, HESS_INDEX, Orbital, dot

SPINS: Tuple[str, str] = ("a", "b")
GROUPS: Tuple[str, ...] = ("rho", "sigma", "lapl", "tau", "eta")
_GROUP_RANK = {g: i for i, g in enumerate(GROUPS)}

#: Component labels of each variable group, in Libxc order.
COMPS: Dict[str, Tuple[str, ...]] = {
    "rho": ("a", "b"),
    "sigma": ("aa", "ab", "bb"),
    "lapl": ("a", "b"),
    "tau": ("a", "b"),
    # the density-Hessian ingredient eta is spin-pure (both gradients and the
    # Hessian carry the same spin), so it packs density-like -- matching the
    # polarized hess dimension of the Libxc local-hybrid branch.
    "eta": ("a", "b"),
}
#: Which spin(s) each component depends on (for seed selection).
COMP_SPINS: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "rho": {"a": ("a",), "b": ("b",)},
    "lapl": {"a": ("a",), "b": ("b",)},
    "tau": {"a": ("a",), "b": ("b",)},
    "sigma": {"aa": ("a", "a"), "ab": ("a", "b"), "bb": ("b", "b")},
    "eta": {"a": ("a",), "b": ("b",)},
}


# --- spin-resolved gradient field symbols (the only fields that appear) -----

def _grad_sym(spin: str, ax: str) -> sp.Symbol:
    return sp.Symbol(f"grad_rho_{spin}_{ax}", real=True)


@lru_cache(maxsize=None)
def grad_syms(coords=None) -> Dict[str, Tuple[sp.Symbol, ...]]:
    """Spin-resolved gradient field symbols named after the system's own
    axes: ``grad_rho_a_x`` in Cartesian, ``grad_rho_a_r`` for a radial
    worker, ``grad_rho_a_mu`` for prolate spheroidal.  The chain rule
    itself is unchanged -- the components are PHYSICAL ones -- so only
    the names and their number depend on the coordinate system."""
    axes = AXES if coords is None else coords.axes
    return {s: tuple(_grad_sym(s, ax) for ax in axes) for s in SPINS}


GRAD = grad_syms()

#: spin-resolved paramagnetic-current, inverse-density, and density-Hessian
#: fields (current-density and density-Hessian meta-GGA families).
JP_S = {s: tuple(sp.Symbol(f"jp_{s}_{ax}", real=True) for ax in AXES)
        for s in SPINS}
INV_RHO_S = {s: sp.Symbol(f"inv_rho_{s}", real=True) for s in SPINS}
HESS_S = {s: tuple(sp.Symbol(f"hess_rho_{s}_{AXES[i]}{AXES[j]}", real=True)
                   for (i, j) in HESS_COMPS) for s in SPINS}


# --- scalar-variable seeds  d(scalar)/dP^s at orbital pair (u, v) ----------

# The spin-resolved seeds are the SAME bilinear forms as the
# spin-summed ones, restricted to one spin channel, so they delegate to
# the single coordinate-aware implementation in inputs.ingredients
# rather than repeating it.  That is what lets a spin-polarized kernel
# be emitted for a curvilinear system at all: the metric lives in one
# place, and the two cannot drift apart.

def _seed_rho(spin: str):
    def f(s: str, u: Orbital, v: Orbital) -> sp.Expr:
        return _ing._k_rho(u, v) if s == spin else sp.Integer(0)
    return f


def _seed_lapl(spin: str):
    def f(s: str, u: Orbital, v: Orbital) -> sp.Expr:
        return _ing._k_lapl(u, v) if s == spin else sp.Integer(0)
    return f


def _seed_tau(spin: str, coords=None):
    def f(s: str, u: Orbital, v: Orbital) -> sp.Expr:
        return _ing._k_tau(u, v, coords) if s == spin else sp.Integer(0)
    return f


def _seed_grad(spin: str, ax: int, coords=None):
    kernel = _ing._k_grad(ax, coords)

    def f(s: str, u: Orbital, v: Orbital) -> sp.Expr:
        return kernel(u, v) if s == spin else sp.Integer(0)
    return f


def _sigma_value(comp: str, coords=None) -> sp.Expr:
    s1, s2 = COMP_SPINS["sigma"][comp]
    g = grad_syms(coords)
    return dot(g[s1], g[s2])


def _seed_sigma(comp: str, coords=None):
    # chain rule through the spin-resolved gradient fields
    value = _sigma_value(comp, coords)
    g = grad_syms(coords)
    grad_seed_of = {}  # symbol -> (spin, axis)
    for spin in SPINS:
        for ax in range(len(g[spin])):
            grad_seed_of[g[spin][ax]] = (spin, ax)

    def f(s: str, u: Orbital, v: Orbital) -> sp.Expr:
        total = sp.Integer(0)
        for gsym, (spin, ax) in grad_seed_of.items():
            total += sp.diff(value, gsym) \
                * _seed_grad(spin, ax, coords)(s, u, v)
        return sp.expand(total)
    return f


@dataclass(frozen=True)
class Scalar:
    """A single Libxc scalar variable, e.g. (rho, a) or (sigma, ab)."""

    group: str
    comp: str
    seed: Callable[[str, Orbital, Orbital], sp.Expr]
    #: perturbed value under a response label (None -> group-default form)
    pert: "Callable[[str], sp.Expr] | None" = None

    @property
    def key(self) -> Tuple[int, int]:
        return (_GROUP_RANK[self.group], COMPS[self.group].index(self.comp))


def _make_scalars() -> Dict[Tuple[str, str], Scalar]:
    out: Dict[Tuple[str, str], Scalar] = {}
    for comp in COMPS["rho"]:
        out[("rho", comp)] = Scalar("rho", comp, _seed_rho(comp))
    for comp in COMPS["sigma"]:
        out[("sigma", comp)] = Scalar("sigma", comp, _seed_sigma(comp))
    for comp in COMPS["lapl"]:
        out[("lapl", comp)] = Scalar("lapl", comp, _seed_lapl(comp))
    for comp in COMPS["tau"]:
        out[("tau", comp)] = Scalar("tau", comp, _seed_tau(comp))
    return out


def _seed_jp_field(spin: str, ax: int):
    def f(s: str, u: Orbital, v: Orbital) -> sp.Expr:
        if s != spin:
            return sp.Integer(0)
        return sp.Rational(1, 2) * (u.val * v.grad[ax] - u.grad[ax] * v.val)
    return f


def _seed_hess_field(spin: str, k: int):
    i, j = HESS_COMPS[k]

    def f(s: str, u: Orbital, v: Orbital) -> sp.Expr:
        if s != spin:
            return sp.Integer(0)
        return (u.hess_ij(i, j) * v.val + u.grad[i] * v.grad[j]
                + u.grad[j] * v.grad[i] + u.val * v.hess_ij(i, j))
    return f


def _seed_inv_rho_field(spin: str):
    def f(s: str, u: Orbital, v: Orbital) -> sp.Expr:
        if s != spin:
            return sp.Integer(0)
        return -INV_RHO_S[spin]**2 * u.val * v.val
    return f


#: every spin-resolved P-dependent FIELD symbol -> its seed d(field)/dP^s_uv.
FIELD_SEEDS: Dict[sp.Symbol, Callable[[str, Orbital, Orbital], sp.Expr]] = {}
for _s in SPINS:
    for _ax in range(3):
        FIELD_SEEDS[GRAD[_s][_ax]] = _seed_grad(_s, _ax)
        FIELD_SEEDS[JP_S[_s][_ax]] = _seed_jp_field(_s, _ax)
    FIELD_SEEDS[INV_RHO_S[_s]] = _seed_inv_rho_field(_s)
    for _k in range(6):
        FIELD_SEEDS[HESS_S[_s][_k]] = _seed_hess_field(_s, _k)


#: perturbed-field counterparts: field symbol -> callable(label) giving the
#: field's value under perturbation `label` (contracted-response engine).
def _pert_field_value(sym_name: str) -> Callable[[str], sp.Expr]:
    def f(label: str) -> sp.Expr:
        # grad_rho_a_x -> grad_rho_a_<label>_x, hess_rho_a_xy analogous
        stem, comp = sym_name.rsplit("_", 1)
        return sp.Symbol(f"{stem}_{label}_{comp}", real=True)
    return f


FIELD_PERTS: Dict[sp.Symbol, Callable[[str], sp.Expr]] = {}
for _s in SPINS:
    for _ax in range(3):
        FIELD_PERTS[GRAD[_s][_ax]] = _pert_field_value(GRAD[_s][_ax].name)
        FIELD_PERTS[JP_S[_s][_ax]] = _pert_field_value(JP_S[_s][_ax].name)
    for _k in range(6):
        FIELD_PERTS[HESS_S[_s][_k]] = _pert_field_value(HESS_S[_s][_k].name)


def _pert_inv_rho(spin: str):
    def f(label: str) -> sp.Expr:
        return -INV_RHO_S[spin]**2 * sp.Symbol(f"rho_{spin}_{label}",
                                               real=True)
    return f


for _s in SPINS:
    FIELD_PERTS[INV_RHO_S[_s]] = _pert_inv_rho(_s)


# --- composite scalars: gauge-corrected tau and the density-Hessian eta -----

def _seed_ctau(spin: str):
    """tau~_s = tau_s - |j_s|^2 / (2 rho_s): seed chains through tau, jp,
    and inv_rho (d inv_rho = -inv_rho^2 d rho)."""
    base_tau = _seed_tau(spin)

    def f(s: str, u: Orbital, v: Orbital) -> sp.Expr:
        if s != spin:
            return sp.Integer(0)
        inv = INV_RHO_S[spin]
        jj = sum(j**2 for j in JP_S[spin])
        d = base_tau(s, u, v)
        d += sp.Rational(1, 2) * inv**2 * jj * u.val * v.val
        for ax in range(3):
            d -= inv * JP_S[spin][ax] *                 sp.Rational(1, 2) * (u.val * v.grad[ax] - u.grad[ax] * v.val)
        return sp.expand(d)
    return f


def _pert_ctau(spin: str):
    def f(label: str) -> sp.Expr:
        inv = INV_RHO_S[spin]
        expr = sp.Symbol(f"tau_{spin}_{label}", real=True)
        expr += sp.Rational(1, 2) * inv**2 * sum(j**2 for j in JP_S[spin])             * sp.Symbol(f"rho_{spin}_{label}", real=True)
        for ax, axname in enumerate(AXES):
            expr -= inv * JP_S[spin][ax] *                 sp.Symbol(f"jp_{spin}_{label}_{axname}", real=True)
        return sp.expand(expr)
    return f


def _eta_value(spin: str) -> sp.Expr:
    """eta_s = grad rho_s . (grad grad^T rho_s) . grad rho_s (spin-pure)."""
    g = GRAD[spin]
    total = sp.Integer(0)
    for i in range(3):
        for j in range(3):
            total += g[i] * HESS_S[spin][HESS_INDEX[(i, j)]] * g[j]
    return sp.expand(total)


def _seed_eta(spin: str):
    value = _eta_value(spin)
    syms = list(GRAD[spin]) + list(HESS_S[spin])

    def f(s: str, u: Orbital, v: Orbital) -> sp.Expr:
        total = sp.Integer(0)
        for sym in syms:
            total += sp.diff(value, sym) * FIELD_SEEDS[sym](s, u, v)
        return sp.expand(total)
    return f


def _pert_eta(spin: str):
    value = _eta_value(spin)
    syms = list(GRAD[spin]) + list(HESS_S[spin])

    def f(label: str) -> sp.Expr:
        total = sp.Integer(0)
        for sym in syms:
            total += sp.diff(value, sym) * FIELD_PERTS[sym](label)
        return sp.expand(total)
    return f


SCALARS = _make_scalars()
for _c in COMPS["eta"]:
    SCALARS[("eta", _c)] = Scalar("eta", _c, _seed_eta(_c), _pert_eta(_c))

#: per-family scalar overrides: cmgga_tau evaluates the Libxc tau slot at the
#: gauge-corrected tau~, so the tau scalars carry the ctau seed/pert instead.
FAMILY_SCALAR_OVERRIDES: Dict[str, Dict[Tuple[str, str], Scalar]] = {
    "cmgga_tau": {
        ("tau", c): Scalar("tau", c, _seed_ctau(c), _pert_ctau(c))
        for c in COMPS["tau"]
    },
}

FAMILY_GROUPS: Dict[str, List[str]] = {
    "lda": ["rho"],
    "gga": ["rho", "sigma"],
    "mgga_tau": ["rho", "sigma", "tau"],
    "mgga_lapl": ["rho", "sigma", "lapl"],
    "mgga": ["rho", "sigma", "lapl", "tau"],
    "cmgga_tau": ["rho", "sigma", "tau"],
    "hmgga": ["rho", "sigma", "lapl", "tau", "eta"],
}


@lru_cache(maxsize=None)
def _scalars_for(coords) -> Dict[Tuple[str, str], Scalar]:
    """Scalar table for one coordinate system.

    Only the seeds carry the metric, and only rho/sigma/tau have seeds
    that a curvilinear system can express -- ``check_coordinates``
    refuses the families that need the others -- so a non-Cartesian
    table is built over those three groups alone.
    """
    if coords is None or coords.is_cartesian:
        return SCALARS
    out: Dict[Tuple[str, str], Scalar] = {}
    for comp in COMPS["rho"]:
        out[("rho", comp)] = Scalar("rho", comp, _seed_rho(comp))
    for comp in COMPS["sigma"]:
        out[("sigma", comp)] = Scalar("sigma", comp,
                                      _seed_sigma(comp, coords))
    for comp in COMPS["tau"]:
        out[("tau", comp)] = Scalar("tau", comp, _seed_tau(comp, coords))
    return out


def family_scalars(family: str, coords=None) -> List[Scalar]:
    _ing.check_coordinates(family, coords)
    table = _scalars_for(coords)
    over = FAMILY_SCALAR_OVERRIDES.get(family, {})
    return [over.get((g, c), table[(g, c)])
            for g in FAMILY_GROUPS[family] for c in COMPS[g]]
