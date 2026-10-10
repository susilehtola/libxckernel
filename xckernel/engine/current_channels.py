"""Coefficient channels of the current-corrected tau-meta-GGA for hosts with
COMPLEX basis functions and REAL symmetric density matrices (HelFEM).

The catalog's ``cmgga_tau`` family assumes the opposite convention: real
basis functions and a general density matrix, whose imaginary part
carries the paramagnetic current.  There a real symmetric perturbation
leaves j_p unchanged, and the real response kernels drop the current
terms.  In HelFEM the basis functions carry the phase,
chi_u = f_u exp(i m_u phi) with f_u real, and the density matrices are
real symmetric, so the current lives in the basis:

    rho = sum_uv P_uv Re(chi_u^* chi_v)
    j_p = sum_uv P_uv Im(chi_u^* grad chi_v)
    tau = 1/2 sum_uv P_uv Re(grad chi_u^* . grad chi_v)

and a real symmetric perturbation moves j_p.  The derivative of j_p with
respect to P_uv, the seed of the current channel, is the real bilinear
Im(chi_u^* grad chi_v), symmetrized over (u, v).

The channels themselves are derivatives of the energy density with
respect to the primitive fields, and do not depend on the convention:

    u = de/d rho,  v_i = de/d (grad rho)_i,  w_tau = de/d tau,
    y_c = de/d (j_p)_c,

with e = f(rho, sigma, tau~), tau~ = tau - |j_p|^2 / (2 rho) (per spin:
tau~_s = tau_s - |j_s|^2 / (2 rho_s)), and Libxc's derivatives of f taken
at tau~.  The fxc channels are their first-order responses to a
perturbation with rho_p1, grad_rho_p1, tau_p1 and j_p1.  This module
derives both from that definition alone, with the gradient and current
components as separate axis lists, so a worker whose current has
components its gradient lacks (the pure-m diatomic worker: j_p is
purely azimuthal, grad rho has no azimuthal component) says so in its
signature.  The axes are physical (orthonormal-frame) components, so the
channels follow from sums of squares, whatever the coordinate system.
"""

from __future__ import annotations

from itertools import combinations_with_replacement
from typing import Dict, Sequence, Tuple

import sympy as sp

#: Libxc's variable kinds of a tau-meta-GGA and their number of
#: polarized components (rho: a, b; sigma: aa, ab, bb; tau: a, b)
_KINDS = ("rho", "sigma", "tau")
_NCOMP = {"rho": 2, "sigma": 3, "tau": 2}


def _sym(name: str) -> sp.Symbol:
    return sp.Symbol(name, real=True)


def _pair_index(i: int, j: int, n: int) -> int:
    """Libxc's packed index of the symmetric pair (i, j) of n components
    (upper triangle, row major): (0,0) 0, (0,1) 1, ..., (n-1,n-1)."""
    i, j = min(i, j), max(i, j)
    return i * n - i * (i - 1) // 2 + (j - i)


def _v1(kind: str, idx, polarized: bool) -> sp.Symbol:
    return _sym(f"v{kind}" + (f"_{idx}" if polarized else ""))


def _v2(k1: str, i1, k2: str, i2, polarized: bool) -> sp.Symbol:
    """Libxc's second derivative d2f/d(k1_i1) d(k2_i2), flat packing."""
    a, b = _KINDS.index(k1), _KINDS.index(k2)
    if a > b:
        k1, i1, k2, i2 = k2, i2, k1, i1
    name = f"v2{k1}2" if k1 == k2 else f"v2{k1}{k2}"
    if not polarized:
        return _sym(name)
    if k1 == k2:
        idx = _pair_index(i1, i2, _NCOMP[k1])
    else:
        idx = i1 * _NCOMP[k2] + i2
    return _sym(f"{name}_{idx}")


def _fields(spins, gaxes, jaxes, label: str = ""):
    """Primitive field symbols, per spin tag ('' unpolarized)."""
    sfx = f"_{label}" if label else ""
    out = {}
    for s in spins:
        t = f"_{s}" if s else ""
        out[s] = {
            "rho": _sym(f"rho{t}{sfx}"),
            "tau": _sym(f"tau{t}{sfx}"),
            "grad": [_sym(f"grad_rho{t}{sfx}_{ax}") for ax in gaxes],
            "jp": [_sym(f"jp{t}{sfx}_{ax}") for ax in jaxes],
        }
    return out


def _ingredients(F, polarized: bool):
    """[(kind, Libxc index, value)] of f's arguments in the primitives."""
    spins = ("a", "b") if polarized else ("",)
    ing = []
    for i, s in enumerate(spins):
        ing.append(("rho", i, F[s]["rho"]))
    pairs = [("a", "a"), ("a", "b"), ("b", "b")] if polarized else [("", "")]
    for i, (s, t) in enumerate(pairs):
        ing.append(("sigma", i, sum(g * h for g, h in
                                    zip(F[s]["grad"], F[t]["grad"]))))
    for i, s in enumerate(spins):
        j2 = sum(c * c for c in F[s]["jp"])
        ing.append(("tau", i, F[s]["tau"] - j2 / (2 * F[s]["rho"])))
    return ing


def _primitives(F, spins):
    """[(channel key, symbol)]: rho, grad_<ax>, tau, jp_<ax> per spin."""
    out = []
    for s in spins:
        t = f"_{s}" if s else ""
        out.append((f"rho{t}", F[s]["rho"]))
        for sym in F[s]["grad"]:
            ax = sym.name.rsplit("_", 1)[1]
            out.append((f"grad{t}_{ax}", sym))
        out.append((f"tau{t}", F[s]["tau"]))
        for sym in F[s]["jp"]:
            ax = sym.name.rsplit("_", 1)[1]
            out.append((f"jp{t}_{ax}", sym))
    return out


def vxc_current_channels(gaxes: Sequence[str], jaxes: Sequence[str],
                         polarized: bool = False) -> Dict[str, sp.Expr]:
    """Ground-state channels de/dX of the current-corrected tau-meta-GGA:
    keys rho, grad_<ax>, tau, jp_<ax> (with _a/_b when polarized).

    The rho channel includes the tau~ term, vtau |j|^2 / (2 rho^2); the
    current channel is -vtau j / rho; grad and tau are those of the
    tau-meta-GGA."""
    spins = ("a", "b") if polarized else ("",)
    F = _fields(spins, gaxes, jaxes)
    ing = _ingredients(F, polarized)
    out = {}
    for key, x in _primitives(F, spins):
        out[key] = sp.expand(sum(
            (_v1(k, i, polarized) * sp.diff(val, x) for k, i, val in ing),
            sp.Integer(0)))
    return out


def fxc_current_channels(gaxes: Sequence[str], jaxes: Sequence[str],
                         polarized: bool = False,
                         label: str = "p1") -> Dict[str, sp.Expr]:
    """First-order response of the ground-state channels to a perturbation
    with fields <prim>_<label> (rho_p1, grad_rho_p1_<ax>, tau_p1,
    jp_p1_<ax>; per spin when polarized):

        d(de/dX) = sum_Y d2e/dX dY  Y_p1,
        d2e/dX dY = sum_IJ f_IJ dI/dX dJ/dY + sum_I f_I d2I/dX dY,

    with I, J over f's arguments (rho, sigma, tau~) and f_I, f_IJ Libxc's
    derivatives in their flat packing.  The second term carries the
    nonlinearities of sigma and tau~, among them the rho and j cross
    terms of tau~."""
    spins = ("a", "b") if polarized else ("",)
    F = _fields(spins, gaxes, jaxes)
    P = _fields(spins, gaxes, jaxes, label)
    ing = _ingredients(F, polarized)
    prims = _primitives(F, spins)
    pert = dict(zip([x for _, x in prims], [y for _, y in
                                            _primitives(P, spins)]))
    # the first derivatives dI/dX, and the perturbed ingredients dI
    dI = [{x: sp.diff(val, x) for _, x in prims} for _, _, val in ing]
    pI = [sp.expand(sum((d[x] * pert[x] for _, x in prims), sp.Integer(0)))
          for d in dI]
    out = {}
    for key, x in prims:
        total = sp.Integer(0)
        for a, (ka, ia, va) in enumerate(ing):
            if dI[a][x] == 0:
                continue
            for b, (kb, ib, _) in enumerate(ing):
                if pI[b] != 0:
                    total += (_v2(ka, ia, kb, ib, polarized)
                              * dI[a][x] * pI[b])
        for a, (ka, ia, va) in enumerate(ing):
            d2 = sum((sp.diff(va, x, y) * pert[y] for _, y in prims),
                     sp.Integer(0))
            total += _v1(ka, ia, polarized) * d2
        out[key] = sp.expand(total)
    return out


def libxc_derivative_names(polarized: bool) -> Tuple[Tuple, Tuple]:
    """((kind, index) of the first derivatives, ((k1, i1), (k2, i2)) of the
    second), in Libxc order, for validation hosts that fill the arrays."""
    if not polarized:
        first = tuple((k, None) for k in _KINDS)
        second = tuple(((a, None), (b, None))
                       for a, b in combinations_with_replacement(_KINDS, 2))
        return first, second
    first = tuple((k, i) for k in _KINDS for i in range(_NCOMP[k]))
    second = []
    for a, b in combinations_with_replacement(_KINDS, 2):
        for i in range(_NCOMP[a]):
            for j in range(_NCOMP[b]):
                if a == b and j < i:
                    continue
                second.append(((a, i), (b, j)))
    return first, tuple(second)
