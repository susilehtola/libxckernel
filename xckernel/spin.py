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
from dataclasses import dataclass
from itertools import product
from typing import Callable, Dict, List, Tuple

import sympy as sp

from .basis import AXES, Orbital, dot

SPINS: Tuple[str, str] = ("a", "b")
GROUPS: Tuple[str, ...] = ("rho", "sigma", "lapl", "tau")
_GROUP_RANK = {g: i for i, g in enumerate(GROUPS)}

#: Component labels of each variable group, in Libxc order.
COMPS: Dict[str, Tuple[str, ...]] = {
    "rho": ("a", "b"),
    "sigma": ("aa", "ab", "bb"),
    "lapl": ("a", "b"),
    "tau": ("a", "b"),
}
#: Which spin(s) each component depends on (for seed selection).
COMP_SPINS: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "rho": {"a": ("a",), "b": ("b",)},
    "lapl": {"a": ("a",), "b": ("b",)},
    "tau": {"a": ("a",), "b": ("b",)},
    "sigma": {"aa": ("a", "a"), "ab": ("a", "b"), "bb": ("b", "b")},
}


# --- spin-resolved gradient field symbols (the only fields that appear) -----

def _grad_sym(spin: str, ax: str) -> sp.Symbol:
    return sp.Symbol(f"grad_rho_{spin}_{ax}", real=True)


GRAD = {s: tuple(_grad_sym(s, ax) for ax in AXES) for s in SPINS}


# --- scalar-variable seeds  d(scalar)/dP^s at orbital pair (u, v) ----------

def _seed_rho(spin: str):
    def f(s: str, u: Orbital, v: Orbital) -> sp.Expr:
        return u.val * v.val if s == spin else sp.Integer(0)
    return f


def _seed_lapl(spin: str):
    def f(s: str, u: Orbital, v: Orbital) -> sp.Expr:
        if s != spin:
            return sp.Integer(0)
        return u.lapl * v.val + 2 * dot(u.grad, v.grad) + u.val * v.lapl
    return f


def _seed_tau(spin: str):
    def f(s: str, u: Orbital, v: Orbital) -> sp.Expr:
        return sp.Rational(1, 2) * dot(u.grad, v.grad) if s == spin \
            else sp.Integer(0)
    return f


def _seed_grad(spin: str, ax: int):
    def f(s: str, u: Orbital, v: Orbital) -> sp.Expr:
        if s != spin:
            return sp.Integer(0)
        return u.grad[ax] * v.val + u.val * v.grad[ax]
    return f


def _sigma_value(comp: str) -> sp.Expr:
    s1, s2 = COMP_SPINS["sigma"][comp]
    return dot(GRAD[s1], GRAD[s2])


def _seed_sigma(comp: str):
    # chain rule through the spin-resolved gradient fields
    value = _sigma_value(comp)
    grad_seed_of = {}  # symbol -> (spin, axis)
    for spin in SPINS:
        for ax in range(3):
            grad_seed_of[GRAD[spin][ax]] = (spin, ax)

    def f(s: str, u: Orbital, v: Orbital) -> sp.Expr:
        total = sp.Integer(0)
        for gsym, (spin, ax) in grad_seed_of.items():
            total += sp.diff(value, gsym) * _seed_grad(spin, ax)(s, u, v)
        return sp.expand(total)
    return f


@dataclass(frozen=True)
class Scalar:
    """A single Libxc scalar variable, e.g. (rho, a) or (sigma, ab)."""

    group: str
    comp: str
    seed: Callable[[str, Orbital, Orbital], sp.Expr]

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


SCALARS = _make_scalars()

FAMILY_GROUPS: Dict[str, List[str]] = {
    "lda": ["rho"],
    "gga": ["rho", "sigma"],
    "mgga_tau": ["rho", "sigma", "tau"],
    "mgga": ["rho", "sigma", "lapl", "tau"],
}


def family_scalars(family: str) -> List[Scalar]:
    return [SCALARS[(g, c)] for g in FAMILY_GROUPS[family] for c in COMPS[g]]
