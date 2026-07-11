"""Geometric (nuclear-displacement) derivatives of the XC tower: the
force-aware layer.

Under a nuclear displacement X = (A, d) the molecular-grid XC integral

    E = sum_g w_g(R) e(fields(r_g; R))

depends on R three ways, and the total derivative splits into three term
classes, each generated here and dispatched separately:

* **basis class** -- the basis functions on atom A move (the fixed-grid
  convention of most Hessian codes).  Two kinds of term: the functional
  chain contracted with host-supplied fixed-grid perturbed fields
  (rho_p1, grad_rho_p1, tau_p1: the atom-restricted collocation
  derivatives of the fields, e.g. rho^X = -2 sum_{u in A} (D+D^T)_uv
  dchi_u chi_v), and the seed-derivative terms where the displacement
  hits the kernel's own free-index collocation factors, carried by the
  ATOM-MASKED operands dchi_gA (nbf, ng) = -dchi_d restricted to atom A
  and ddchi_gA (3, nbf, ng) = -d_d grad chi restricted to atom A.  (The
  minus signs of d chi/d X_A = -d chi/d r are folded INTO the masked
  operands, mirroring production practice.)

* **grid class** -- grid point g rides its parent atom: the term is
  w_g M^A_g d_d e(r_g), with M^A the parent-atom mask.  d_d e is the
  SPATIAL GRADIENT of the integrand density, an ordinary field
  expression; the host folds w*M^A into the weight operand, so a single
  generated kernel serves every atom and direction.  Direction-resolved
  operands: drho_g (ng,), dgrad_rho_g (3, ng), dtau_g (ng,) are the
  d-components of grad rho, (grad grad rho) rows, and grad tau; the
  basis factors differentiate to dchi_g / ddchi_g (unmasked analogues of
  the basis-class operands, without the sign fold).

* **weight class** -- the Becke partition weights change: the ORIGINAL
  kernel evaluated with w := dw/dX.  No new kernel is generated; the
  dispatch table simply says so.

Translational invariance ties the classes together: summed over all
atoms A (with sum_A M^A = 1 and sum_A dw/dX_A = 0) the three classes
cancel exactly -- the recommended validation for any host wiring.

The operators act on ANY tower integrand (energy, Fock, response
kernels), so higher geometric derivatives compose mechanically.
"""

from __future__ import annotations

import re as _re
from collections import Counter
from dataclasses import dataclass

import sympy as sp

from .basis import AXES
from .deriv import LIBXC_MULTISET, VARS, libxc_symbol
from .fock import fock_integrand
from .functional import Functional
from .ingredients import PRIM_BY_SYMBOL
from .kernel import KernelIntegrand
from .response import _seed_fn

_CHI_RE = _re.compile(r"^chi_(\w+)$")
_DCHI_RE = _re.compile(r"^dchi_(\w+)_([xyz])$")

# --- direction-resolved spatial-gradient operands ----------------------------

#: d-component of grad rho (the host slices its gradient array).
DRHO_G = sp.Symbol("drho_g", real=True)
#: d-row of the density Hessian, (3, ng): dgrad_rho_g[i] = d_d d_i rho.
DGRAD_RHO_G = tuple(sp.Symbol(f"dgrad_rho_g_{ax}", real=True) for ax in AXES)
#: d-component of grad tau.
DTAU_G = sp.Symbol("dtau_g", real=True)


def _spatial_field_gradient(var: str) -> sp.Expr:
    """d_d of a Libxc input variable, in direction-resolved operands."""
    from .ingredients import GRAD_RHO
    if var == "rho":
        return DRHO_G
    if var == "tau":
        return DTAU_G
    if var == "sigma":
        return 2 * sum(p.symbol * DGRAD_RHO_G[i] for i, p in enumerate(GRAD_RHO))
    raise ValueError(f"spatial gradient of {var!r} not supported yet "
                     "(lda/gga/mgga_tau families)")


def _spatial_seed(func: Functional):
    """Monomial-level seed for the spatial-gradient operator d_d."""
    from .fastpoly import from_expr
    by_name = {ing.name: ing for ing in func.ingredients}

    def seed(atom: sp.Symbol):
        name = atom.name
        prim = PRIM_BY_SYMBOL.get(atom)
        if prim is not None:
            if prim.name == "rho":
                return from_expr(DRHO_G)
            if prim.name.startswith("grad_rho_"):
                i = AXES.index(prim.name[-1])
                return from_expr(DGRAD_RHO_G[i])
            if prim.name == "tau":
                return from_expr(DTAU_G)
            raise ValueError(f"spatial gradient of primitive {prim.name!r} "
                             "not supported yet")
        if name in LIBXC_MULTISET:
            ms = LIBXC_MULTISET[name]
            total = sp.Integer(0)
            for Y in VARS:
                if by_name.get(Y) is not None:
                    total += libxc_symbol(ms + Counter({Y: 1})) \
                        * _spatial_field_gradient(Y)
            return from_expr(total)
        # basis factors: chi_u -> dchi_g_u; dchi_u_i -> ddchi_g_u_i
        m = _CHI_RE.match(name)
        if m:
            return from_expr(sp.Symbol(f"dchi_g_{m.group(1)}", real=True))
        m = _DCHI_RE.match(name)
        if m:
            return from_expr(sp.Symbol(f"ddchi_g_{m.group(1)}_{m.group(2)}",
                                       real=True))
        if name.startswith("lapl_chi"):
            raise ValueError("spatial gradient of lapl_chi needs "
                             "third-derivative collocation (not wired yet)")
        return None  # weight, perturbed fields of other labels
    return seed


def spatial_gradient(ki: KernelIntegrand) -> KernelIntegrand:
    """d_d of a tower integrand's density (w treated as the quadrature
    datum it is).  This is the grid-motion class: the host calls the
    generated kernel with weight operand w := w * M^A (parent-atom mask
    folded in), once per direction with sliced operands."""
    from .fastpoly import from_expr, seeded_derivative, to_expr
    poly = getattr(ki, "poly", None)
    if poly is None:
        poly = from_expr(sp.expand(ki.expr))
    out = seeded_derivative(poly, _spatial_seed(ki.functional))
    return KernelIntegrand(functional=ki.functional,
                           index_pairs=list(ki.index_pairs),
                           expr=to_expr(out))


def spatial_energy_gradient(family: str) -> sp.Expr:
    """d_d of the XC energy density e(fields(r)): the per-point scalar
    sum_k v_k d_d field_k, in direction-resolved operands. This is the
    grid-motion class of the ENERGY derivative (the XC gradient): the host
    contracts it with w * M^A over the parent atom's points."""
    func = Functional.of_family(family)
    total = sp.Integer(0)
    for ing in func.ingredients:
        total += func.vsymbol(ing) * _spatial_field_gradient(ing.name)
    return sp.expand(total)


# --- spin-polarized spatial gradient (grid-motion class) ----------------------

def _spatial_field_gradient_spin(K) -> sp.Expr:
    """d_d of a polarized Libxc scalar variable, in per-channel
    direction-resolved operands (drho_a_g, dgrad_rho_a_g_i, dtau_a_g and
    the beta twins)."""
    from .spin import COMP_SPINS, GRAD
    if K.group == "rho":
        return sp.Symbol(f"drho_{K.comp}_g", real=True)
    if K.group == "tau":
        return sp.Symbol(f"dtau_{K.comp}_g", real=True)
    if K.group == "sigma":
        s1, s2 = COMP_SPINS["sigma"][K.comp]
        dg = {s: [sp.Symbol(f"dgrad_rho_{s}_g_{ax}", real=True) for ax in AXES]
              for s in (s1, s2)}
        return sum(GRAD[s1][i] * dg[s2][i] + GRAD[s2][i] * dg[s1][i]
                   for i in range(3))
    raise ValueError(f"spatial gradient of {K.group!r} not supported yet "
                     "(lda/gga/mgga_tau families)")


def spatial_energy_gradient_spin(family: str) -> sp.Expr:
    """d_d of the polarized XC energy density: the per-point scalar
    sum_K v_K d_d field_K over the polarized scalar variables. The
    grid-motion class of the UKS XC gradient."""
    from .spin import family_scalars
    from .spin_kernel import _register
    total = sp.Integer(0)
    for K in family_scalars(family):
        total += _register((K,)) * _spatial_field_gradient_spin(K)
    return sp.expand(total)


# --- the basis (fixed-grid) class --------------------------------------------

def _geometric_seed(func: Functional):
    """Seed for the basis class: functional chain against fixed-grid
    perturbed fields (label p1) plus atom-masked seed-derivative operands
    (sign of d chi/d X_A folded into dchi_gA / ddchi_gA)."""
    from .fastpoly import from_expr
    resp = _seed_fn(func, "p1")

    def seed(atom: sp.Symbol):
        name = atom.name
        m = _CHI_RE.match(name)
        if m:
            return from_expr(sp.Symbol(f"dchi_gA_{m.group(1)}", real=True))
        m = _DCHI_RE.match(name)
        if m:
            return from_expr(sp.Symbol(f"ddchi_gA_{m.group(1)}_{m.group(2)}",
                                       real=True))
        if name.startswith("lapl_chi"):
            raise ValueError("geometric seed of lapl_chi needs "
                             "third-derivative collocation (not wired yet)")
        return resp(atom)
    return seed


def geometric_fock(family: str, u: str = "u", v: str = "v") -> KernelIntegrand:
    """Basis-class integrand of dF_uv/dX at fixed density and fixed grid.

    Operands: perturbed fields rho_p1 / grad_rho_p1 / tau_p1 (the
    atom-restricted fixed-grid field derivatives) and the atom-masked
    collocation derivatives dchi_gA (nbf, ng) / ddchi_gA (3, nbf, ng),
    both carrying the -d/dr sign."""
    from .fastpoly import from_expr, seeded_derivative, to_expr
    fi = fock_integrand(family, u, v)
    out = seeded_derivative(from_expr(fi.expr), _geometric_seed(fi.functional))
    return KernelIntegrand(functional=fi.functional, index_pairs=[(u, v)],
                           expr=to_expr(out))


# --- dispatch ----------------------------------------------------------------

@dataclass
class GeometricDispatch:
    """The three term classes of a force-aware geometric derivative.

    basis:  new integrand (fixed-grid convention alone stops here);
    grid:   new integrand, call with weight := w * M^A;
    weight: the ORIGINAL integrand, call with weight := dw/dX.
    """
    basis: KernelIntegrand
    grid: KernelIntegrand
    weight: KernelIntegrand


def geometric_dispatch(family: str) -> GeometricDispatch:
    """Everything a host needs for the full geometric derivative of the
    XC Fock matrix, quadrature dependence included."""
    from .kernel import fock
    fi = fock(family)
    return GeometricDispatch(basis=geometric_fock(family),
                             grid=spatial_gradient(fi),
                             weight=fi)

# --- the second-order geometric operator (fixed-grid basis class) -------------

@dataclass
class GeometricHessian:
    """The explicit fixed-grid XC Hessian as a function-pair kernel:

        H^{(A,x),(B,y)} = sum_{u in A, v in B} pair_{uv} + delta_AB sum_{u in A} same_u

    ``pair`` carries two free labels (u for the X side, v for the Y side);
    ``same`` one. Operand vocabulary (all per function and grid point):
    U0_u = (phi D)_u and Ui_u = (dphi_i D)_u density-contracted collocation
    rows; dchi_gA_u / ddchi_gA_u_i (X side) and dchi_gB_v / ddchi_gB_v_i
    (Y side) masked displacement collocations CARRYING the -d/dr sign;
    d2chi_g2_u / d3chi_g2_u_i the double-displacement collocations (two
    signs cancel: they carry +d^2/dxdy); D_u_v the local density-matrix
    pair factor.
    """
    family: str
    pair: sp.Expr
    same: sp.Expr
    #: backend-facing structure: the i=0 instances of the per-function
    #: rows and seed members (i-generic across Cartesian components),
    #: WITHOUT the quadrature weight. Emitters bind the component index
    #: and fold class symmetry factors (symmetric classes enter
    #: accumulate-plus-transpose hosts at half weight).
    hints: dict


def _geo_rows(label: str, side: str, func: Functional):
    """Per-function field-derivative rows F_k(u) for one displacement:
    field^{(A,x)} = sum_{u in A} F_k(u), plus the gradient rows G_i."""
    U0 = sp.Symbol(f"U0_{label}", real=True)
    Ui = [sp.Symbol(f"U{i + 1}_{label}", real=True) for i in range(3)]
    dg = sp.Symbol(f"dchi_g{side}_{label}", real=True)
    ddg = [sp.Symbol(f"ddchi_g{side}_{label}_{ax}", real=True) for ax in AXES]
    from .ingredients import GRAD_RHO
    G = [2 * (U0 * ddg[i] + Ui[i] * dg) for i in range(3)]
    rows = {"rho": 2 * U0 * dg,
            "sigma": 2 * sum(GRAD_RHO[i].symbol * G[i] for i in range(3)),
            "tau": sum(Ui[i] * ddg[i] for i in range(3))}
    return {k: rows[k] for k in rows if k in {i.name for i in func.ingredients}}, G, (U0, Ui, dg, ddg)


def geometric_hessian(family: str, u: str = "u", v: str = "v") -> GeometricHessian:
    """Second geometric derivative of the XC energy integrand at fixed
    density and fixed grid (the basis class): the explicit term of the
    analytic nuclear Hessian."""
    from collections import Counter

    from .deriv import libxc_symbol
    from .ingredients import GRAD_RHO
    func = Functional.of_family(family)
    names = [ing.name for ing in func.ingredients]
    if any(n not in ("rho", "sigma", "tau") for n in names):
        raise ValueError("geometric_hessian supports lda/gga/mgga_tau")
    w = sp.Symbol("w", real=True, positive=True)
    D = sp.Symbol(f"D_{u}_{v}", real=True)

    FA, GA, (U0u, Uiu, dgu, ddgu) = _geo_rows(u, "A", func)
    FB, GB, (U0v, Uiv, dgv, ddgv) = _geo_rows(v, "B", func)

    # field x field through the second functional derivatives
    pair = sp.Integer(0)
    for k in names:
        for l in names:
            pair += libxc_symbol(Counter({k: 1}) + Counter({l: 1})) * FA[k] * FB[l]
    # the vsigma gradient cross term of sigma^{XY}
    if "sigma" in names:
        pair += func.vsymbol([i for i in func.ingredients if i.name == "sigma"][0])             * 2 * sum(GA[i] * GB[i] for i in range(3))

    # potential times the two-center seed second derivatives
    vs = {ing.name: func.vsymbol(ing) for ing in func.ingredients}
    pair += vs["rho"] * 2 * D * dgu * dgv
    if "sigma" in names:
        pair += vs["sigma"] * 2 * sum(
            GRAD_RHO[i].symbol * 2 * D * (ddgu[i] * dgv + dgu * ddgv[i])
            for i in range(3))
    if "tau" in names:
        pair += vs["tau"] * D * sum(ddgu[i] * ddgv[i] for i in range(3))

    # potential times the one-center seed second derivatives
    d2 = sp.Symbol(f"d2chi_g2_{u}", real=True)
    d3 = [sp.Symbol(f"d3chi_g2_{u}_{ax}", real=True) for ax in AXES]
    same = vs["rho"] * 2 * U0u * d2
    if "sigma" in names:
        same += vs["sigma"] * 2 * sum(
            GRAD_RHO[i].symbol * 2 * (U0u * d3[i] + Uiu[i] * d2) for i in range(3))
    if "tau" in names:
        same += vs["tau"] * sum(Uiu[i] * d3[i] for i in range(3))

    hints = {"F_rho": FA["rho"], "G_i": GA[0],
             "seed_pair_rho": vs["rho"] * 2 * D * dgu * dgv,
             "seed_same_rho": vs["rho"] * 2 * U0u * d2}
    if "sigma" in names:
        hints["F_sigma_i"] = 2 * GRAD_RHO[0].symbol * sp.Symbol("G_i", real=True)
        # the left member of the transpose pair; the mirror comes from
        # the host's accumulate-plus-transpose
        hints["seed_pair_sigma_i"] = vs["sigma"] * 2 * GRAD_RHO[0].symbol \
            * 2 * D * ddgu[0] * dgv
        hints["seed_same_sigma_i"] = vs["sigma"] * 2 * GRAD_RHO[0].symbol \
            * 2 * (U0u * d3[0] + Uiu[0] * d2)
    if "tau" in names:
        hints["F_tau_i"] = Uiu[0] * ddgu[0]
        hints["seed_pair_tau_i"] = vs["tau"] * D * ddgu[0] * ddgv[0]
        hints["seed_same_tau_i"] = vs["tau"] * Uiu[0] * d3[0]

    return GeometricHessian(family=family, pair=sp.expand(w * pair),
                            same=sp.expand(w * same), hints=hints)

# --- spin-polarized geometric layer -------------------------------------------

def _geometric_seed_spin(family: str, label: str = "p1"):
    """Spin analogue of _geometric_seed: basis factors go to the
    atom-masked displacement collocations (collocations carry no spin);
    gradient fields and polarized Libxc symbols chain to the fixed-grid
    perturbed fields of BOTH spin channels (a displacement perturbs both,
    like any one-electron perturbation)."""
    from .fastpoly import from_expr
    from .spin_kernel import _seed_fn_spin
    resp = _seed_fn_spin(family, label)

    def seed(atom: sp.Symbol):
        name = atom.name
        m = _CHI_RE.match(name)
        if m:
            return from_expr(sp.Symbol(f"dchi_gA_{m.group(1)}", real=True))
        m = _DCHI_RE.match(name)
        if m:
            return from_expr(sp.Symbol(f"ddchi_gA_{m.group(1)}_{m.group(2)}",
                                       real=True))
        if name.startswith("lapl_chi"):
            raise ValueError("geometric seed of lapl_chi needs "
                             "third-derivative collocation (not wired yet)")
        return resp(atom)
    return seed


def geometric_fock_spin(family: str, spin: str, u: str = "u", v: str = "v"):
    """Basis-class integrand of dF^spin_uv/dX at fixed density and fixed
    grid: the spin-resolved analogue of geometric_fock. Operands are the
    per-channel fixed-grid perturbed fields (rho_a_p1, rho_b_p1,
    grad_rho_{a,b}_p1_i, tau_{a,b}_p1) and the SHARED atom-masked
    collocation derivatives dchi_gA / ddchi_gA (sign folded)."""
    from .fastpoly import from_expr, seeded_derivative, to_expr
    from .spin_kernel import SpinIntegrand, fock_spin
    fi = fock_spin(family, spin, u, v)
    out = seeded_derivative(from_expr(fi.expr),
                            _geometric_seed_spin(family, "p1"))
    return SpinIntegrand(family=family, spins=[spin], index_pairs=[(u, v)],
                         expr=to_expr(out))


def _geo_rows_spin(label: str, side: str, family: str):
    """Per-function field-derivative rows F_K(u) for one displacement,
    per polarized scalar K. U rows are per spin channel (U0a_u = (phi
    D^a)_u, ...); the displacement collocations carry no spin."""
    from .spin import GRAD, SPINS, family_scalars
    U0 = {s: sp.Symbol(f"U0{s}_{label}", real=True) for s in SPINS}
    Ui = {s: [sp.Symbol(f"U{i + 1}{s}_{label}", real=True) for i in range(3)]
          for s in SPINS}
    dg = sp.Symbol(f"dchi_g{side}_{label}", real=True)
    ddg = [sp.Symbol(f"ddchi_g{side}_{label}_{ax}", real=True) for ax in AXES]
    G = {s: [2 * (U0[s] * ddg[i] + Ui[s][i] * dg) for i in range(3)]
         for s in SPINS}
    rows = {}
    for K in family_scalars(family):
        if K.group == "rho":
            rows[K] = 2 * U0[K.comp] * dg
        elif K.group == "tau":
            rows[K] = sum(Ui[K.comp][i] * ddg[i] for i in range(3))
        elif K.group == "sigma":
            s1, s2 = K.comp
            rows[K] = sum(GRAD[s1][i] * G[s2][i] + GRAD[s2][i] * G[s1][i]
                          for i in range(3)) if s1 != s2 else                 2 * sum(GRAD[s1][i] * G[s1][i] for i in range(3))
        else:
            raise ValueError(f"spin geometric rows: unsupported group {K.group}")
    return rows, G, (U0, Ui, dg, ddg)


def geometric_hessian_spin(family: str, u: str = "u",
                           v: str = "v") -> GeometricHessian:
    """Spin-polarized second geometric derivative of the XC energy
    integrand at fixed density and fixed grid. Same pair/same structure
    as geometric_hessian; the D_u_v pair factor and the U rows split per
    channel (D_a_u_v, D_b_u_v; U0a_u, ...), and the functional
    derivatives are the polarized component symbols of spin_kernel."""
    from .spin import GRAD, SPINS, family_scalars
    from .spin_kernel import _register
    scalars = family_scalars(family)
    if any(K.group not in ("rho", "sigma", "tau") for K in scalars):
        raise ValueError("geometric_hessian_spin supports lda/gga/mgga_tau")
    w = sp.Symbol("w", real=True, positive=True)
    D = {s: sp.Symbol(f"D_{s}_{u}_{v}", real=True) for s in SPINS}

    FA, GA, (U0u, Uiu, dgu, ddgu) = _geo_rows_spin(u, "A", family)
    FB, GB, (U0v, Uiv, dgv, ddgv) = _geo_rows_spin(v, "B", family)

    # field x field through the second functional derivatives
    pair = sp.Integer(0)
    for K in scalars:
        for L in scalars:
            pair += _register((K, L)) * FA[K] * FB[L]

    # the vsigma gradient cross terms of sigma_st^{XY}
    vs = {K: _register((K,)) for K in scalars}
    for K in scalars:
        if K.group != "sigma":
            continue
        s1, s2 = K.comp
        if s1 == s2:
            pair += vs[K] * 2 * sum(GA[s1][i] * GB[s1][i] for i in range(3))
        else:
            pair += vs[K] * sum(GA[s1][i] * GB[s2][i] + GA[s2][i] * GB[s1][i]
                                for i in range(3))

    # potential times the two-center seed second derivatives
    for K in scalars:
        if K.group == "rho":
            pair += vs[K] * 2 * D[K.comp] * dgu * dgv
        elif K.group == "tau":
            pair += vs[K] * D[K.comp] * sum(ddgu[i] * ddgv[i]
                                            for i in range(3))
        else:
            s1, s2 = K.comp
            # for s1 == s2 the two sums coincide and their total IS the
            # required 2 grad_s . (grad rho_s)^{XY} seed
            cross = sum(GRAD[s2][i] * 2 * D[s1] * (ddgu[i] * dgv + dgu * ddgv[i])
                        for i in range(3)) \
                + sum(GRAD[s1][i] * 2 * D[s2] * (ddgu[i] * dgv + dgu * ddgv[i])
                      for i in range(3))
            pair += vs[K] * cross

    # potential times the one-center seed second derivatives
    d2 = sp.Symbol(f"d2chi_g2_{u}", real=True)
    d3 = [sp.Symbol(f"d3chi_g2_{u}_{ax}", real=True) for ax in AXES]
    same = sp.Integer(0)
    for K in scalars:
        if K.group == "rho":
            same += vs[K] * 2 * U0u[K.comp] * d2
        elif K.group == "tau":
            same += vs[K] * sum(Uiu[K.comp][i] * d3[i] for i in range(3))
        else:
            s1, s2 = K.comp
            same += vs[K] * sum(
                GRAD[s2][i] * 2 * (U0u[s1] * d3[i] + Uiu[s1][i] * d2)
                + GRAD[s1][i] * 2 * (U0u[s2] * d3[i] + Uiu[s2][i] * d2)
                for i in range(3))

    # emit-ready structure: i = 0 instances of rows and seed members,
    # weight-free, grouped the way a two-channel emitter consumes them
    # (per D/U channel for the seeds, per right G channel for the vsigma
    # gradient cross). G_<s>_i are placeholders for the materialized
    # per-point G rows.
    Gph = {s: sp.Symbol(f"G_{s}_i", real=True) for s in SPINS}
    F_i0 = {}
    for K in scalars:
        if K.group == "rho":
            F_i0[K] = FA[K]
        elif K.group == "tau":
            F_i0[K] = Uiu[K.comp][0] * ddgu[0]
        else:
            s1, s2 = K.comp
            F_i0[K] = (2 * GRAD[s1][0] * Gph[s1] if s1 == s2 else
                       GRAD[s1][0] * Gph[s2] + GRAD[s2][0] * Gph[s1])
    cIp_left = {s: sp.Integer(0) for s in SPINS}     # right G channel -> left
    sp_sig = {s: sp.Integer(0) for s in SPINS}       # D channel -> pair seed
    ss_sig = {s: sp.Integer(0) for s in SPINS}       # U channel -> same seed
    for K in scalars:
        if K.group != "sigma":
            continue
        s1, s2 = K.comp
        if s1 == s2:
            cIp_left[s1] += vs[K] * 2 * Gph[s1]
            sp_sig[s1] += vs[K] * 4 * GRAD[s1][0] * D[s1] * ddgu[0] * dgv
            ss_sig[s1] += vs[K] * 4 * GRAD[s1][0] * (U0u[s1] * d3[0]
                                                     + Uiu[s1][0] * d2)
        else:
            cIp_left[s1] += vs[K] * Gph[s2]
            cIp_left[s2] += vs[K] * Gph[s1]
            sp_sig[s1] += vs[K] * 2 * GRAD[s2][0] * D[s1] * ddgu[0] * dgv
            sp_sig[s2] += vs[K] * 2 * GRAD[s1][0] * D[s2] * ddgu[0] * dgv
            ss_sig[s1] += vs[K] * 2 * GRAD[s2][0] * (U0u[s1] * d3[0]
                                                     + Uiu[s1][0] * d2)
            ss_sig[s2] += vs[K] * 2 * GRAD[s1][0] * (U0u[s2] * d3[0]
                                                     + Uiu[s2][0] * d2)
    sp_tau, ss_tau = {}, {}
    for K in scalars:
        if K.group == "tau":
            sp_tau[K.comp] = vs[K] * D[K.comp] * ddgu[0] * ddgv[0]
            ss_tau[K.comp] = vs[K] * Uiu[K.comp][0] * d3[0]

    hints = {"scalars": scalars, "vs": vs,
             "G_i": {s: GA[s][0] for s in SPINS},
             "F_i0": F_i0,
             "classIp_left": cIp_left,
             "seed_pair_sigma_i": sp_sig, "seed_pair_tau_i": sp_tau,
             "seed_same_sigma_i": ss_sig, "seed_same_tau_i": ss_tau}
    return GeometricHessian(family=family, pair=sp.expand(w * pair),
                            same=sp.expand(w * same), hints=hints)

