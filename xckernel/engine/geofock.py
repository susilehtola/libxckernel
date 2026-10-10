"""Nuclear derivatives of the XC Fock matrix, dF/dX_{A,d}, in the forms
the C catalog ships (CPHF right-hand sides, nuclear Hessians).  The term
classes are those of geometric.py:

* **basis class** (``fock_basis_class``) -- atom A's basis functions
  move, the grid is fixed.  Three contributions:

    dF_uv/dX = [u on A] F_uv(chi_u -> d chi_u/dX)
             + [v on A] F_uv(chi_v -> d chi_v/dX)
             + sum_k (dF_uv/dfield_k) field_k^X,

  the last being the linear-response (o2) contraction with the perturbed
  fields of the displacement.  Those fields are reductions over A's
  functions, field_k^X(g) = sum_{u on A} [B_k(d chi_u, (D chi)_u) +
  B_k((D chi)_u, d chi_u)] for the field's bilinear form B_k
  (``perturbed_field_terms``), which the kernel evaluates itself from
  the collocation tower, the D-contracted tower and the atom mask.
  d chi/dX_{A,d} = -d_d chi: the sign is folded in.

* **grid class** (``fock_grid_class``) -- grid points ride their parent
  atom: d_d of the Fock integrand (``spatial_derivative``), called with
  w := w M^A.

* **weight class** -- the order-1 kernel with w := dw/dX.

Density matrices are symmetric, except for the complex-orbital
cmgga_tau family (gradient.GENERAL_DM_FAMILIES), whose general M makes
the two index slots contract differently: the perturbed fields then read
the M^T-contracted tower DTchi next to Dchi.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import List, Tuple

import sympy as sp

from ..inputs.basis import AXES, Orbital
from .gradient import (_d3, _deta, _dtau_tilde, _grad_scalar, _hess,
                       _hess_comp)
from .kernel import KernelIntegrand

#: families with dF/dX entries
FOCK_DERIV_FAMILIES = ("lda", "gga", "mgga_tau", "mgga_lapl", "mgga",
                       "hmgga", "cmgga_tau")

_SORT = lambda s: "".join(sorted(s))
_BASIS = re.compile(r"^(chi|dchi|lapl_chi|hess_chi)_(u|v)(?:_([xyz]{1,2}))?$")


def _basis_parts(sym: sp.Symbol):
    m = _BASIS.match(sym.name)
    if not m:
        return None
    kind, lbl, ax = m.groups()
    if len(ax or "") != {"chi": 0, "dchi": 1, "lapl_chi": 0,
                         "hess_chi": 2}[kind]:
        return None
    return lbl, kind, ax or ""


def _lapl_combo(prefix: str = "") -> List[Tuple[str, int]]:
    return [(_SORT(prefix + a + a), 1) for a in AXES]


def _tower(kind: str, ax: str, d: str = "") -> List[Tuple[str, int]]:
    """A basis factor (optionally differentiated along d) as a combination
    of tower components."""
    if kind == "lapl_chi":
        return _lapl_combo(d)
    return [(_SORT(d + ax), 1)]


# --- the perturbed fields of a displacement --------------------------------------

def _primitive_for(name: str):
    """(primitive, channel) of a perturbed-field symbol: rho_a_p1 ->
    (rho, 'a'), grad_rho_p1_x -> (grad_rho_x, '')."""
    from ..inputs.ingredients import PRIMITIVES
    m = re.fullmatch(r"(grad_rho|hess_rho|jp|rho|lapl_rho|tau)"
                     r"(?:_([ab]))?_p1(?:_([xyz]{1,2}))?", name)
    if not m:
        raise ValueError(f"not a perturbed field: {name}")
    base, ch, ax = m.groups()
    return PRIMITIVES[f"{base}_{ax}" if ax else base], ch or ""


def perturbed_field_terms(name: str, d: int, general: bool = False):
    """The perturbed field ``name`` of the displacement of atom A along d,
    as a masked reduction over A's functions:

        field^X(g) = sum_u mask_u sum_t c_t chi_{a_t}(u,g) R_t(u,g),

    returned as (channel, [(c_t, a_t, b_t)]) with a_t, b_t tower axis
    strings and R the D-contracted tower Dchi of that channel's density
    matrix. With ``general`` (non-symmetric M), displacing the second
    index slot contracts the first with M^T: those terms read DTchi and
    are flagged by a 'T' prefix on b_t."""
    prim, ch = _primitive_for(name)
    expr = sp.expand(prim.kernel(Orbital.make("u"), Orbital.make("v")))
    acc: Counter = Counter()
    for term in sp.Add.make_args(expr):
        coeff, rest = term.as_coeff_Mul()
        parts = {}
        for base, e in rest.as_powers_dict().items():
            info = _basis_parts(base)
            if info is None or e != 1:
                raise ValueError(f"field kernel not bilinear: {term}")
            parts[info[0]] = info[1:]
        (ku, au), (kv, av) = parts["u"], parts["v"]
        dd = AXES[d]
        # displace either slot (d chi/dX = -d_d chi), contract the other:
        # the first slot's partner with M (Dchi), the second's with M^T
        for (ks, as_), (kr, ar), t in (((ku, au), (kv, av), ""),
                                       ((kv, av), (ku, au),
                                        "T" if general else "")):
            for sa, wa in _tower(ks, as_, dd):
                for sb, wb in _tower(kr, ar):
                    acc[(sa, t + sb)] += -coeff * wa * wb
    return ch, [(c, a, b) for (a, b), c in acc.items() if c != 0]


def mo_field_terms(name: str, sign: int = +1):
    """The perturbed field ``name`` of a trial vector X in the occupied x
    virtual space, P = C_occ X C_vir^T + sign C_vir X^T C_occ^T (sign -1:
    the antisymmetric, imaginary perturbation of a general density
    matrix, under which the density-like fields cancel and the
    paramagnetic current remains):

        field(g) = sum_t c_t sum_i phi_o[a_t](i,g) Z[b_t](i,g),
        Z[b](i,g) = sum_a X(i,a) phi_v[b](a,g),

    returned as (channel, [(c_t, a_t, b_t)]) with a_t, b_t tower axis
    strings of the occupied and virtual MO collocation."""
    prim, ch = _primitive_for(name)
    expr = sp.expand(prim.kernel(Orbital.make("u"), Orbital.make("v")))
    acc: Counter = Counter()
    for term in sp.Add.make_args(expr):
        coeff, rest = term.as_coeff_Mul()
        parts = {}
        for base, e in rest.as_powers_dict().items():
            info = _basis_parts(base)
            if info is None or e != 1:
                raise ValueError(f"field kernel not bilinear: {term}")
            parts[info[0]] = info[1:]
        (ku, au), (kv, av) = parts["u"], parts["v"]
        # B(phi_i, phi_a) and B(phi_a, phi_i): the occupied orbital in either
        # slot, the virtual one contracted with X
        for sg, (ko, ao), (kw, aw) in ((1, (ku, au), (kv, av)),
                                       (sign, (kv, av), (ku, au))):
            for so, wo in _tower(ko, ao):
                for sv, wv in _tower(kw, aw):
                    acc[(so, sv)] += sg * coeff * wo * wv
    return ch, [(c, a, b) for (a, b), c in acc.items() if c != 0]


# --- the basis class -----------------------------------------------------------

def _masked(kind: str, ax: str, lbl: str, d: int) -> sp.Symbol:
    """The atom-masked derivative d_d of a basis factor (label lbl)."""
    dd = AXES[d]
    if kind == "chi":
        return sp.Symbol(f"Mdchi_{lbl}_{dd}", real=True)
    if kind == "dchi":
        return sp.Symbol(f"Mhess_chi_{lbl}_{_SORT(dd + ax)}", real=True)
    if kind == "hess_chi":
        return sp.Symbol(f"Mtchi_{lbl}_{_SORT(dd + ax)}", real=True)
    return sp.Symbol(f"Mdlapl_chi_{lbl}_{dd}", real=True)


def _fock(family: str, spin: str):
    if spin == "r":
        from .fock import fock_integrand
        return fock_integrand(family, "u", "v").expr
    from .spin_kernel import fock_spin
    return fock_spin(family, spin[1], "u", "v").expr


def _response(family: str, spin: str):
    if spin == "r":
        from .response import response_fock
        return response_fock(family, 2).expr
    from .spin_kernel import response_fock_spin
    return response_fock_spin(family, spin[1], 2).expr


def fock_basis_class(family: str, spin: str, d: int) -> KernelIntegrand:
    """Basis class of dF_uv/dX_{A,d}: the masked seed terms plus the o2
    contraction with the perturbed fields (symbols *_p1, evaluated by the
    kernel from perturbed_field_terms)."""
    if family not in FOCK_DERIV_FAMILIES:
        raise ValueError(f"no dF/dX for family {family!r}")
    from ..inputs.functional import Functional
    seeds = sp.Integer(0)
    for term in sp.Add.make_args(sp.expand(_fock(family, spin))):
        for base, e in term.as_powers_dict().items():
            info = _basis_parts(base) if base.is_Symbol else None
            if info is None:
                continue
            lbl, kind, ax = info
            seeds += term.subs(base, _masked(kind, ax, lbl, d))
    expr = sp.expand(_response(family, spin) - seeds)
    return KernelIntegrand(functional=Functional.of_family(family),
                           index_pairs=[("u", "v")], expr=expr)


# --- the grid class ------------------------------------------------------------

def _field_gradient(name: str, d: int) -> sp.Expr:
    """d_d of a per-point field operand of the Fock integrand."""
    dd = AXES[d]
    m = re.fullmatch(r"grad_rho(_[ab])?_([xyz])", name)
    if m:
        return _hess(f"hess_rho{m.group(1) or ''}", AXES.index(m.group(2)), d)
    m = re.fullmatch(r"hess_rho(_[ab])?_([xyz]{2})", name)
    if m:
        i, j = (AXES.index(c) for c in m.group(2))
        return _d3(f"d3rho{m.group(1) or ''}", i, j, d)
    m = re.fullmatch(r"jp(_[ab])?_([xyz])", name)
    if m:
        return sp.Symbol(f"jpgrad{m.group(1) or ''}_{m.group(2)}_{dd}",
                         real=True)
    m = re.fullmatch(r"inv_rho(_[ab])?", name)
    if m:
        s = m.group(1) or ""
        return -sp.Symbol(name, real=True) ** 2 * _grad_scalar(f"grad_rho{s}", d)
    raise ValueError(f"no spatial gradient for operand {name!r}")


def _variable_gradient(family: str, var: str, s: str, d: int) -> sp.Expr:
    """d_d of a Libxc input variable; s the channel suffix ('', '_a', ...)
    and, for sigma, the pair of channels."""
    if var == "rho":
        return _grad_scalar(f"grad_rho{s}", d)
    if var == "tau":
        if family == "cmgga_tau":
            return _dtau_tilde(s, d)
        return _grad_scalar(f"grad_tau{s}", d)
    if var == "lapl":
        return _grad_scalar(f"grad_lapl_rho{s}", d)
    if var == "eta":
        return _deta(s, d)
    raise ValueError(var)


def _sigma_gradient(s1: str, s2: str, d: int) -> sp.Expr:
    g = lambda s, i: _grad_scalar(f"grad_rho{s}", i)
    return sum(g(s1, i) * _hess(f"hess_rho{s2}", i, d)
               + g(s2, i) * _hess(f"hess_rho{s1}", i, d) for i in range(3))


def spatial_derivative(expr: sp.Expr, family: str, spin: str,
                       d: int) -> sp.Expr:
    """d_d of a per-point integrand (the weight w held fixed): basis
    factors differentiate along the tower, fields to their spatial
    gradients, functional derivatives by the chain rule."""
    from .fastpoly import from_expr, seeded_derivative, to_expr
    dd = AXES[d]
    if spin == "r":
        from .deriv import LIBXC_MULTISET, libxc_symbol
        from ..inputs.functional import Functional
        names = [i.name for i in Functional.of_family(family).ingredients]

        def vseed(atom):
            ms = LIBXC_MULTISET.get(atom.name)
            if ms is None:
                return None
            tot = sp.Integer(0)
            for Y in names:
                dY = (2 * sum(_grad_scalar("grad_rho", i)
                              * _hess("hess_rho", i, d) for i in range(3))
                      if Y == "sigma" else
                      _variable_gradient(family, Y, "", d))
                tot += libxc_symbol(ms + Counter({Y: 1})) * dY
            return tot
    else:
        from .spin import COMP_SPINS, family_scalars
        from .spin_kernel import _SYM_SCALARS, _register
        scalars = family_scalars(family)

        def vseed(atom):
            base = _SYM_SCALARS.get(atom.name)
            if base is None:
                return None
            tot = sp.Integer(0)
            for K in scalars:
                if K.group == "sigma":
                    s1, s2 = COMP_SPINS["sigma"][K.comp]
                    dY = _sigma_gradient(f"_{s1}", f"_{s2}", d)
                else:
                    dY = _variable_gradient(family, K.group, f"_{K.comp}", d)
                tot += _register(base + (K,)) * dY
            return tot

    def seed(atom: sp.Symbol):
        name = atom.name
        info = _basis_parts(atom)
        if info is not None:
            lbl, kind, ax = info
            if kind == "chi":
                return from_expr(sp.Symbol(f"dchi_{lbl}_{dd}", real=True))
            if kind == "dchi":
                return from_expr(sp.Symbol(
                    f"hess_chi_{lbl}_{_hess_comp(d, AXES.index(ax))}",
                    real=True))
            if kind == "hess_chi":
                return from_expr(sp.Symbol(f"tchi_{lbl}_{_SORT(dd + ax)}",
                                           real=True))
            return from_expr(sp.Symbol(f"dlapl_chi_{lbl}_{dd}", real=True))
        if name == "w":
            return None
        v = vseed(atom)
        if v is not None:
            return from_expr(sp.expand(v))
        return from_expr(_field_gradient(name, d))

    return sp.expand(to_expr(seeded_derivative(from_expr(sp.expand(expr)),
                                               seed)))


def fock_grid_class(family: str, spin: str, d: int) -> KernelIntegrand:
    """Grid class of dF_uv/dX_{A,d}: d_d of the Fock integrand, to be
    called with w := w M^A."""
    if family not in FOCK_DERIV_FAMILIES:
        raise ValueError(f"no dF/dX for family {family!r}")
    from ..inputs.functional import Functional
    return KernelIntegrand(functional=Functional.of_family(family),
                           index_pairs=[("u", "v")],
                           expr=spatial_derivative(_fock(family, spin),
                                                   family, spin, d))
