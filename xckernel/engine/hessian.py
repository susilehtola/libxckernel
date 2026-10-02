"""The explicit XC nuclear Hessian (second derivative of the XC energy at
fixed density, quadrature dependence included) in the forms the C
catalog ships.

With grid points riding their parent atoms and weights w(R),

    d2E/dX_{A,d} dY_{B,e} = sum_g [ w^{AB} e + w^A eps^B + w^B eps^A
                                    + w d2e/dX_{A,d} dY_{B,e} ],

eps^B(g) the first derivative of the energy density e at point g for
atom B (basis motion plus, for B the parent of g, the point's motion).
The last term splits into the classes generated here, each from an
integrand that is already validated:

* **BB** (basis-basis): the second basis displacement (atom B,
  direction e) of the g1 rows of the energy gradient (atom A, direction
  d, summed by the host over A's functions). The displaced collocation of
  the row differentiates again, masked to B's functions; the D-contracted
  row differentiates into -D (mask_B o d_e chi); the functional derivatives
  chain through the perturbed fields of B's displacement (the response
  seeds; the kernel forms those fields itself, as for f1).
* **BG** (basis of A, grid of B): the spatial derivative of the g1 rows,
  called with w := w M^B; GB is its transpose.
* **GG** (grid-grid, A = B = the parent): the spatial derivative of the
  grid-class integrand w d_d e, per point.
* **eps** (per point, basis part): the g1 rows without w, reduced over
  B's functions; the grid part is gg with w = 1 at B's points.

The weight terms w^{AB} e and w^A eps^B are the host's (it owns dw and
d2w). Every basis and field symbol is a derivative-tower component, so
each derivative appends an axis.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import List, Tuple

import sympy as sp

from ..inputs.basis import AXES
from .kernel import KernelIntegrand

#: families with nuclear-Hessian entries
HESSIAN_FAMILIES = ("lda", "gga", "mgga_tau", "mgga_lapl", "mgga", "hmgga",
                    "cmgga_tau")

_S = lambda s: "".join(sorted(s))


def _sym(name: str) -> sp.Symbol:
    return sp.Symbol(name, real=True)


def _lapl(prefix: str = "") -> List[str]:
    return [_S(prefix + a + a) for a in AXES]


# --- the g1-row vocabulary as tower components -----------------------------------

def _seeded_axes(name: str):
    """Tower axes of a displaced-collocation symbol of the g1 rows (label
    u), as a list (a Laplacian expands to three), or None."""
    m = re.fullmatch(r"dchi_u_([xyz])", name)
    if m:
        return [m.group(1)]
    m = re.fullmatch(r"hess_chi_u_([xyz]{2})", name)
    if m:
        return [m.group(1)]
    m = re.fullmatch(r"tchi_u_([xyz]{3})", name)
    if m:
        return [m.group(1)]
    m = re.fullmatch(r"dlapl_chi_u_([xyz])", name)
    if m:
        return _lapl(m.group(1))
    m = re.fullmatch(r"T_u_([xyz]+)", name)
    if m:
        return [m.group(1)]
    return None


def _row_axes(name: str):
    """(transposed?, tower axes list) of a D-contracted row symbol (label
    v), or None."""
    m = re.fullmatch(r"U(T?)(0|[123]|L)_v", name)
    if m:
        t, k = m.groups()
        if k == "0":
            return bool(t), [""]
        if k == "L":
            return bool(t), _lapl()
        return bool(t), [AXES[int(k) - 1]]
    m = re.fullmatch(r"U(T?)h_v_([xyz]{2})", name)
    if m:
        return bool(m.group(1)), [m.group(2)]
    m = re.fullmatch(r"R(T?)_v_([xyz]*)", name)
    if m:
        return bool(m.group(1)), [m.group(2)]
    return None


# --- the response seeds of a basis displacement ---------------------------------------

def _response_seed(family: str, spin: str):
    from ..inputs.functional import Functional
    if spin == "r":
        from .response import _seed_fn
        return _seed_fn(Functional.of_family(family), "p1")
    from .spin_kernel import _seed_fn_spin
    return _seed_fn_spin(family, "p1")


def hessian_bb(family: str, spin: str, d: int, e: int) -> KernelIntegrand:
    """BB class, rows for (A, d) x (B, e): the B_e displacement of the g1
    rows. Symbols: MT_u_<axes> (B-masked collocation), RM_v_<axes>
    (= D (mask_B o d^axes chi), RMT_ with D^T), the unmasked g1-row
    symbols, and the *_p1 perturbed fields of B's displacement."""
    from .fastpoly import from_expr, seeded_derivative, to_expr
    from .gradient import energy_gradient_rows
    ki = energy_gradient_rows(family, spin, d)
    resp = _response_seed(family, spin)
    de = AXES[e]

    def seed(atom: sp.Symbol):
        name = atom.name
        ax = _seeded_axes(name)
        if ax is not None:
            # the row's own function, displaced again if it is on B
            return from_expr(-sum(_sym(f"MT_u_{_S(a + de)}") for a in ax))
        r = _row_axes(name)
        if r is not None:
            t, axs = r
            # the contracted partner functions on B move
            return from_expr(-sum(_sym(f"RM{'T' if t else ''}_v_{_S(a + de)}")
                                  for a in axs))
        return resp(atom)

    out = seeded_derivative(from_expr(sp.expand(ki.expr)), seed)
    return KernelIntegrand(functional=ki.functional, index_pairs=[("u", "v")],
                           expr=sp.expand(to_expr(out)))


# --- spatial derivatives -----------------------------------------------------------

def _tower_field(base: str, s: str, axes: str) -> sp.Symbol:
    """A field-tower component by its ABI name: rho_a_xy, tau_x, jpx_b_z."""
    parts = [base] + ([s] if s else []) + ([_S(axes)] if axes else [])
    return _sym("_".join(parts))


def _field_spatial(name: str, d: str):
    """d_d of a per-point field operand, or None when the symbol is not a
    field. Handles the engine's names and the tower names alike."""
    S = r"(?:_([ab]))?"
    m = re.fullmatch(r"grad_rho" + S + r"_([xyz])", name)
    if m:
        return _tower_field("rho", m.group(1), m.group(2) + d)
    m = re.fullmatch(r"hess_rho" + S + r"_([xyz]{2})", name)
    if m:
        return _tower_field("rho", m.group(1), m.group(2) + d)
    m = re.fullmatch(r"d3rho" + S + r"_([xyz]{3})", name)
    if m:
        return _tower_field("rho", m.group(1), m.group(2) + d)
    m = re.fullmatch(r"grad_tau" + S + r"_([xyz])", name)
    if m:
        return _tower_field("tau", m.group(1), m.group(2) + d)
    m = re.fullmatch(r"grad_lapl_rho" + S + r"_([xyz])", name)
    if m:
        return sum(_tower_field("rho", m.group(1), a + d)
                   for a in _lapl(m.group(2)))
    m = re.fullmatch(r"jp" + S + r"_([xyz])", name)
    if m:
        return _tower_field(f"jp{m.group(2)}", m.group(1), d)
    m = re.fullmatch(r"jpgrad" + S + r"_([xyz])_([xyz])", name)
    if m:
        return _tower_field(f"jp{m.group(2)}", m.group(1), m.group(3) + d)
    m = re.fullmatch(r"inv_rho" + S, name)
    if m:
        return -_sym(name) ** 2 * _tower_field("rho", m.group(1), d)
    # tower names (from an earlier spatial derivative)
    m = re.fullmatch(r"(rho|tau|jp[xyz])" + S + r"_([xyz]+)", name)
    if m:
        return _tower_field(m.group(1), m.group(2), m.group(3) + d)
    return None


def _variable_spatial(family: str, group: str, s: str, d: str) -> sp.Expr:
    """d_d of a Libxc variable in tower names; s = '' or '_a'/'_b' for
    single-channel variables."""
    ch = s[1:] if s else ""
    g = lambda i: _tower_field("rho", ch, AXES[i])
    if group == "rho":
        return _tower_field("rho", ch, d)
    if group == "tau":
        if family == "cmgga_tau":
            jp = [_sym(f"jp{s}_{a}") for a in AXES]
            djp = [_tower_field(f"jp{a}", ch, d) for a in AXES]
            inv = _sym(f"inv_rho{s}")
            return (_tower_field("tau", ch, d)
                    - inv * sum(j * dj for j, dj in zip(jp, djp))
                    + sp.Rational(1, 2) * inv ** 2
                    * sum(j * j for j in jp) * _tower_field("rho", ch, d))
        return _tower_field("tau", ch, d)
    if group == "lapl":
        return sum(_tower_field("rho", ch, a + d) for a in _lapl())
    if group == "eta":
        h = lambda i, j: _tower_field("rho", ch, AXES[i] + AXES[j])
        t = lambda i, j: _tower_field("rho", ch, AXES[i] + AXES[j] + d)
        di = AXES.index(d)
        return sum(2 * h(i, di) * h(i, j) * g(j) + g(i) * t(i, j) * g(j)
                   for i in range(3) for j in range(3))
    raise ValueError(group)


def _sigma_spatial(c1: str, c2: str, d: str) -> sp.Expr:
    g = lambda c, i: _tower_field("rho", c, AXES[i])
    h = lambda c, i: _tower_field("rho", c, AXES[i] + d)
    return sum(g(c1, i) * h(c2, i) + g(c2, i) * h(c1, i) for i in range(3))


def spatial(expr: sp.Expr, family: str, spin: str, d: int) -> sp.Expr:
    """d_d of a per-point integrand, w held fixed: displaced collocation
    and contracted rows append the axis (unmasked), fields go to their
    spatial gradients, functional derivatives by the chain rule."""
    from .fastpoly import from_expr, seeded_derivative, to_expr
    dd = AXES[d]
    if spin == "r":
        from ..inputs.functional import Functional
        from .deriv import LIBXC_MULTISET, libxc_symbol
        groups = [i.name for i in Functional.of_family(family).ingredients]

        def vseed(atom):
            ms = LIBXC_MULTISET.get(atom.name)
            if ms is None:
                return None
            tot = sp.Integer(0)
            for Y in groups:
                dY = (2 * sum(_tower_field("rho", "", AXES[i])
                              * _tower_field("rho", "", AXES[i] + dd)
                              for i in range(3)) if Y == "sigma"
                      else _variable_spatial(family, Y, "", dd))
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
                    c1, c2 = COMP_SPINS["sigma"][K.comp]
                    dY = _sigma_spatial(c1, c2, dd)
                else:
                    dY = _variable_spatial(family, K.group, f"_{K.comp}", dd)
                tot += _register(base + (K,)) * dY
            return tot

    def seed(atom: sp.Symbol):
        name = atom.name
        if name == "w":
            return None
        ax = _seeded_axes(name)
        if ax is not None:
            return from_expr(sum(_sym(f"T_u_{_S(a + dd)}") for a in ax))
        r = _row_axes(name)
        if r is not None:
            t, axs = r
            return from_expr(sum(_sym(f"R{'T' if t else ''}_v_{_S(a + dd)}")
                                 for a in axs))
        v = vseed(atom)
        if v is not None:
            return from_expr(sp.expand(v))
        f = _field_spatial(name, dd)
        if f is not None:
            return from_expr(sp.expand(f))
        raise ValueError(f"no spatial derivative for {name!r}")

    return sp.expand(to_expr(seeded_derivative(from_expr(sp.expand(expr)),
                                               seed)))


def hessian_bg(family: str, spin: str, d: int, e: int) -> KernelIntegrand:
    """BG class, rows for (A basis, d) x (B grid, e): d_e of the g1 rows
    (host passes w := w M^B)."""
    from .gradient import energy_gradient_rows
    ki = energy_gradient_rows(family, spin, d)
    return KernelIntegrand(functional=ki.functional, index_pairs=[("u", "v")],
                           expr=spatial(ki.expr, family, spin, e))


def hessian_gg(family: str, spin: str, d: int, e: int) -> sp.Expr:
    """GG class, per point: w d_d d_e e (spin 'r' or 'u')."""
    from .gradient import energy_grid_gradient
    return spatial(energy_grid_gradient(family, spin, d), family, spin, e)


def energy_eps_rows(family: str, spin: str, e: int) -> KernelIntegrand:
    """The basis part of eps^B per point, as g1 rows without the weight
    (the kernel reduces them over B's functions)."""
    from .gradient import energy_gradient_rows
    ki = energy_gradient_rows(family, spin, e)
    w = [x for x in ki.expr.free_symbols if x.name == "w"]
    return KernelIntegrand(functional=ki.functional, index_pairs=[("u", "v")],
                           expr=sp.expand(ki.expr.subs(w[0], 1)))
