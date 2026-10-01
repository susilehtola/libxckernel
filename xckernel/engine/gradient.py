"""The XC nuclear gradient in the forms the C catalog ships: per-function
rows for the basis class and per-point spatial gradients for the grid
class.  The term classes are those of geometric.py:

    dE/dX_{A,d} = basis + grid + weight

* **basis class** (``energy_gradient_rows``) -- atom A's basis functions
  move.  Every field is a density-matrix contraction of a symmetric
  bilinear form in the collocation, field = sum_uv D_uv B(chi_u, chi_v),
  so displacing the functions on A gives

      dE/dX_{A,d} = sum_{u in A} g_{d,u},
      g_{d,u} = sum_v D_uv [F_uv with chi_u -> d chi_u/dX_d]
              + sum_v D_vu [F_vu with chi_u -> d chi_u/dX_d],

  F the XC Fock integrand.  The undisplaced side contracts with D into
  the density-contracted collocation rows of geometric_hessian:
  U0_u = (D chi)_u, Ui_u = (D d_i chi)_u and UL_u = (D lapl chi)_u; the
  displaced side differentiates the collocation, chi -> d_d chi,
  d_i chi -> d_d d_i chi (hess_chi), lapl chi -> d_d lapl chi
  (dlapl_chi).  d chi_u/dX_{A,d} = -d_d chi_u: the minus sign is folded
  into the generated integrand, so the operands are the PLAIN collocation
  derivatives and the rows sum to +dE/dX (the host adds them over the
  functions on each atom).  Unrestricted: g^s from F^s and the rows of
  D^s; the gradient is the sum over the two channels.

* **grid class** (``energy_grid_gradient``) -- grid point g rides its
  parent atom: sum_{g in A} w_g d_d e(r_g), with
  d_d e = sum_k v_k d_d field_k in the field gradients grad_rho,
  hess_rho, grad_tau and grad_lapl_rho.  The host reduces the per-point
  values over each atom's points.

* **weight class** -- the original energy density with w := dw/dX; no
  kernel beyond the order-0 one.

Both generated classes require a symmetric density matrix (D_uv = D_vu),
as for any real-orbital density.
"""

from __future__ import annotations

import re
from typing import Tuple

import sympy as sp

from ..inputs.basis import AXES, HESS_COMPS
from .kernel import KernelIntegrand

#: families with nuclear-gradient integrands
GRADIENT_FAMILIES = ("lda", "gga", "mgga_tau", "mgga_lapl", "mgga", "hmgga")

_BASIS_SYM = re.compile(
    r"^(chi|dchi|lapl_chi|hess_chi)_(\w+?)(?:_([xyz]{1,2}))?$")
_NAXES = {"chi": 0, "dchi": 1, "lapl_chi": 0, "hess_chi": 2}


def _hess_comp(i: int, j: int) -> str:
    """Packed symmetric-tensor component name for axes (i, j)."""
    a, b = sorted((i, j))
    comp = f"{AXES[a]}{AXES[b]}"
    assert (a, b) in HESS_COMPS
    return comp


def _split_basis(sym: sp.Symbol, labels: Tuple[str, str]):
    """(slot, kind, axis) of a basis factor, or None for any other symbol."""
    m = _BASIS_SYM.match(sym.name)
    if m is None:
        return None
    kind, lbl, ax = m.groups()
    if len(ax or "") != _NAXES[kind] or lbl not in labels:
        return None
    return labels.index(lbl), kind, ax


def _seeded(kind: str, axes, d: int) -> sp.Symbol:
    """d_d of a basis factor, as a symbol on the displaced (u) side."""
    if kind == "chi":
        return sp.Symbol(f"dchi_u_{AXES[d]}", real=True)
    if kind == "dchi":
        return sp.Symbol(f"hess_chi_u_{_hess_comp(d, AXES.index(axes))}",
                         real=True)
    if kind == "hess_chi":
        # third-derivative collocation (a chi-tower component)
        return sp.Symbol(f"tchi_u_{''.join(sorted(AXES[d] + axes))}",
                         real=True)
    return sp.Symbol(f"dlapl_chi_u_{AXES[d]}", real=True)


def _row(kind: str, axes) -> sp.Symbol:
    """The D-contracted collocation row of a basis factor (v side)."""
    if kind == "chi":
        return sp.Symbol("U0_v", real=True)
    if kind == "dchi":
        return sp.Symbol(f"U{AXES.index(axes) + 1}_v", real=True)
    if kind == "hess_chi":
        return sp.Symbol(f"Uh_v_{axes}", real=True)
    return sp.Symbol("UL_v", real=True)


def _fock_expr(family: str, spin: str):
    if spin == "r":
        from .fock import fock_integrand
        fi = fock_integrand(family, "u", "v")
        return fi.functional, fi.expr
    from ..inputs.functional import Functional
    from .spin_kernel import fock_spin
    return Functional.of_family(family), \
        fock_spin(family, spin[1], "u", "v").expr


def energy_gradient_rows(family: str, spin: str, d: int) -> KernelIntegrand:
    """Basis class of the XC energy gradient, direction d, as per-function
    rows: the integrand of g_{d,u} (+dE/dX contributions) with the
    displaced collocation on label u and the density-contracted rows on
    label v (both evaluated at the same function in the row-wise
    contraction)."""
    if family not in GRADIENT_FAMILIES:
        raise ValueError(f"no gradient rows for family {family!r}")
    func, expr = _fock_expr(family, spin)
    labels = ("u", "v")
    total = sp.Integer(0)
    for term in sp.Add.make_args(sp.expand(expr)):
        parts = [None, None]
        rest = sp.Integer(1)
        for base, e in term.as_powers_dict().items():
            info = _split_basis(base, labels) if base.is_Symbol else None
            if info is None:
                rest *= base ** e
                continue
            if e != 1 or parts[info[0]] is not None:
                raise ValueError(f"Fock term not bilinear in the basis: {term}")
            parts[info[0]] = info[1:]
        if parts[0] is None or parts[1] is None:
            raise ValueError(f"Fock term without both basis factors: {term}")
        (ku, au), (kv, av) = parts
        # displace the first slot, contract the second -- and vice versa
        total += rest * (_seeded(ku, au, d) * _row(kv, av)
                         + _row(ku, au) * _seeded(kv, av, d))
    # d chi/dX_{A,d} = -d_d chi
    return KernelIntegrand(functional=func, index_pairs=[("u", "v")],
                           expr=sp.expand(-total))


# --- grid class ----------------------------------------------------------------

def _grad_scalar(name: str, d: int) -> sp.Symbol:
    return sp.Symbol(f"{name}_{AXES[d]}", real=True)


def _hess(prefix: str, i: int, j: int) -> sp.Symbol:
    return sp.Symbol(f"{prefix}_{_hess_comp(i, j)}", real=True)


def _d3(prefix: str, i: int, j: int, k: int) -> sp.Symbol:
    """Third derivative of the density, d_i d_j d_k rho."""
    return sp.Symbol(f"{prefix}_{''.join(sorted(AXES[i] + AXES[j] + AXES[k]))}",
                     real=True)


def _deta(s: str, d: int) -> sp.Expr:
    """d_d eta, eta = grad rho . (grad grad rho) . grad rho, channel suffix
    s ('' or '_a'/'_b'): 2 rho_id rho_ij rho_j + rho_i rho_ijd rho_j."""
    g = [_grad_scalar(f"grad_rho{s}", i) for i in range(3)]
    return sum(2 * _hess(f"hess_rho{s}", i, d) * _hess(f"hess_rho{s}", i, j)
               * g[j] + g[i] * _d3(f"d3rho{s}", i, j, d) * g[j]
               for i in range(3) for j in range(3))


def energy_grid_gradient(family: str, spin: str, d: int) -> sp.Expr:
    """Grid class of the XC energy gradient, direction d: the per-point
    w * d_d e(r) with e the XC energy density per volume.

    spin 'r': unpolarized fields; spin 'u': the polarized energy density
    (both channels -- the grid class has no free basis index to split)."""
    if family not in GRADIENT_FAMILIES:
        raise ValueError(f"no grid gradient for family {family!r}")
    w = sp.Symbol("w", real=True, positive=True)
    if spin == "r":
        from ..inputs.functional import Functional
        func = Functional.of_family(family)
        grad = [_grad_scalar("grad_rho", i) for i in range(3)]
        dfield = {
            "rho": grad[d],
            "sigma": 2 * sum(grad[i] * _hess("hess_rho", i, d)
                             for i in range(3)),
            "tau": _grad_scalar("grad_tau", d),
            "lapl": _grad_scalar("grad_lapl_rho", d),
            "eta": _deta("", d),
        }
        total = sum(func.vsymbol(ing) * dfield[ing.name]
                    for ing in func.ingredients)
        return sp.expand(w * total)
    if spin != "u":
        raise ValueError(f"grid gradient spin must be 'r' or 'u', got {spin!r}")
    from .spin import COMP_SPINS, family_scalars
    from .spin_kernel import _register
    grad = {s: [_grad_scalar(f"grad_rho_{s}", i) for i in range(3)]
            for s in "ab"}
    total = sp.Integer(0)
    for K in family_scalars(family):
        if K.group == "rho":
            df = grad[K.comp][d]
        elif K.group == "sigma":
            s1, s2 = COMP_SPINS["sigma"][K.comp]
            df = sum(grad[s1][i] * _hess(f"hess_rho_{s2}", i, d)
                     + grad[s2][i] * _hess(f"hess_rho_{s1}", i, d)
                     for i in range(3))
        elif K.group == "tau":
            df = _grad_scalar(f"grad_tau_{K.comp}", d)
        elif K.group == "lapl":
            df = _grad_scalar(f"grad_lapl_rho_{K.comp}", d)
        elif K.group == "eta":
            df = _deta(f"_{K.comp}", d)
        else:
            raise ValueError(f"no spatial gradient for {K.group!r}")
        total += _register((K,)) * df
    return sp.expand(w * total)
