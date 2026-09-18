"""Emit the fixed-grid nuclear-Hessian kernels for GauXC as C++.

GauXC has no nuclear Hessian at all: its public integrator stops at
``eval_exc_grad``.  This writer supplies the exchange-correlation half of
one, in the form GauXC's local work drivers already consume -- per
grid-point inline functions over the collocation rows it already builds.

The explicit (fixed-grid, fixed-weight) Hessian of ``engine.geometric``
is

    H^{(A,x),(B,y)} = sum_{u in A, v in B} pair_uv + delta_AB sum_{u in A} same_u

and ``pair`` splits into two structurally different halves, which a host
must assemble differently:

* the monomials WITHOUT the density-matrix pair factor ``D_u_v``
  factorise into a u-side and a v-side, so they collapse to an OUTER
  PRODUCT of per-atom-per-direction rows -- a rank update over the
  3N x 3N Hessian rather than a nested loop over basis-function pairs.
  Summed over ``u in A`` the rows ARE the nuclear-perturbed fields
  rho^{A,x}, grad rho^{A,x}, tau^{A,x}.
* the monomials WITH ``D_u_v`` cannot factorise; they are the Pulay
  terms and need the function pair.

The first half is, term for term,

    sum_{k<=l} v2_kl [F_k^A F_l^B + (k/=l) F_l^A F_k^B]      (1)
  + vsigma * 2 sum_i G_i^A G_i^B                             (2)

i.e. the ordinary fxc quadratic form on the two perturbed field sets,
PLUS a term (2) that a host reusing its fxc contraction would silently
drop.  (2) exists because sigma = |grad rho|^2 is QUADRATIC in grad rho,
so d^2 sigma / dA dB survives even for A /= B; tau, being bilinear in
grad chi, has no counterpart.  It is ~10% of the GGA pair term, which is
small enough to look like a plausible Hessian and large enough to be
wrong.  ``_check_pair_decomposition`` proves the identity against
``geometric_hessian`` symbolically at import-test time rather than
asserting it.

Operand mapping to GauXC (restricted):

    U0_u        <-> xmat            (X = xmat_fac * P * B)
    Ui_u        <-> xmat_x/y/z      (X_c = xmat_fac * P * d_c B)
    D_u_v       <-> xmat_fac * P    (the gathered local block)
    dchi_gA_u   <-> -dbasis_<x>     (sign folded in, masked to atom A)
    ddchi_gA_u_i<-> -d2basis_<x><i> (ditto)
    d2chi_g2_u  <-> +d2basis_<x><y> (two displacement signs cancel)
    d3chi_g2_u_i<-> +d3basis_<x><y><i>

Reproduce with: python -m xckernel.emitters.gauxcwriter --emit <file>
"""

from __future__ import annotations

import functools

import sympy as sp

from . import fieldkernel
from .fieldkernel import ExplicitLayout, FieldKernel
from ..engine.geometric import geometric_hessian
from ..inputs.basis import AXES

#: families GauXC's host driver can carry today. The laplacian meta-GGAs
#: are absent deliberately: a laplacian Hessian needs FOURTH collocation
#: derivatives, and gau2grid stops at der3.
FAMILIES = ("lda", "gga", "mgga_tau")

#: Libxc second-derivative name for an (ingredient, ingredient) pair.
def _v2(k: str, l: str) -> str:
    return f"v2{k}2" if k == l else f"v2{k}{l}"


def _ingredients(family: str) -> list:
    return {"lda": ["rho"],
            "gga": ["rho", "sigma"],
            "mgga_tau": ["rho", "sigma", "tau"]}[family]


def _rows(family: str, side: str):
    """Per-function field rows for one displacement side.

    ``side`` is "A" (u index) or "B" (v index); the row symbols mirror
    geometric_hessian's own operand names so the decomposition check
    below compares like with like.
    """
    lab, oth = ("u", "A") if side == "A" else ("v", "B")
    S = lambda n: sp.Symbol(n, real=True)
    U0 = S(f"U0_{lab}")
    Ui = [S(f"U{i + 1}_{lab}") for i in range(3)]
    dchi = S(f"dchi_g{oth}_{lab}")
    ddchi = [S(f"ddchi_g{oth}_{lab}_{a}") for a in AXES]
    grad_rho = [S(f"grad_rho_{a}") for a in AXES]

    G = [2 * (U0 * ddchi[i] + Ui[i] * dchi) for i in range(3)]
    rows = {
        "rho": 2 * U0 * dchi,
        "sigma": 2 * sum(grad_rho[i] * G[i] for i in range(3)),
        "tau": sum(Ui[i] * ddchi[i] for i in range(3)),
    }
    return rows, G


def pair_expression(family: str):
    """The non-Pulay half of ``pair``, in terms of the SUMMED rows.

    Returned in row symbols (F_rho_A, F_sigma_B, G_A_x, ...) because the
    host sums the per-function rows over the shells of an atom before
    contracting; that sum is what makes this a rank update.
    """
    ings = _ingredients(family)
    S = lambda n: sp.Symbol(n, real=True)
    FA = {k: S(f"F_{k}_A") for k in ings}
    FB = {k: S(f"F_{k}_B") for k in ings}
    GA = [S(f"G_A_{a}") for a in AXES]
    GB = [S(f"G_B_{a}") for a in AXES]

    out = 0
    for i, k in enumerate(ings):
        for j, l in enumerate(ings):
            if j < i:
                continue
            c = S(_v2(k, l))
            out += c * FA[k] * FB[l] if k == l else c * (FA[k] * FB[l] + FA[l] * FB[k])
    if "sigma" in ings:
        out += S("vsigma") * 2 * sum(GA[i] * GB[i] for i in range(3))
    return out


def _check_pair_decomposition(family: str):
    """Prove the factorised form against geometric_hessian itself.

    The point of the writer is that the host never sees the unfactorised
    expression, so the factorisation is the one step nothing downstream
    can check. Do it here, symbolically, for every family emitted.
    """
    gh = geometric_hessian(family)
    ex = sp.expand(gh.pair)
    D = sp.Symbol("D_u_v", real=True)
    terms = ex.args if ex.is_Add else (ex,)
    noD = sum(t for t in terms if not t.has(D))

    rowsA, GA = _rows(family, "A")
    rowsB, GB = _rows(family, "B")
    subs = {}
    for k in _ingredients(family):
        subs[sp.Symbol(f"F_{k}_A", real=True)] = rowsA[k]
        subs[sp.Symbol(f"F_{k}_B", real=True)] = rowsB[k]
    for i, a in enumerate(AXES):
        subs[sp.Symbol(f"G_A_{a}", real=True)] = GA[i]
        subs[sp.Symbol(f"G_B_{a}", real=True)] = GB[i]

    # Align by NAME: geometric_hessian's Libxc symbols carry no
    # assumptions while the row symbols are real=True, and sympy treats
    # same-named symbols with different assumptions as distinct. Comparing
    # without this yields a spurious residual.
    pool = {sym.name: sym for sym in noD.free_symbols}
    def align(e):
        return e.subs({sym: pool[sym.name] for sym in e.free_symbols
                       if sym.name in pool and pool[sym.name] is not sym},
                      simultaneous=True)

    cand = align(pair_expression(family).subs(subs))
    w = pool.get("w", sp.Symbol("w", real=True))
    resid = sp.simplify(sp.expand(noD / w - cand))
    if resid != 0:
        raise AssertionError(
            f"{family}: factorised pair term does not reproduce "
            f"geometric_hessian ({sp.count_ops(resid)} ops residual)")


def spec_rows(family: str) -> FieldKernel:
    """Per-function rows the host sums over the shells of one atom."""
    rows, G = _rows(family, "A")
    ings = _ingredients(family)
    exprs, targets = {}, []
    for k in ings:
        exprs[f"F_{k}"] = rows[k]
        targets.append(f"F_{k}")
    for i, a in enumerate(AXES):
        exprs[f"G_{a}"] = G[i]
        targets.append(f"G_{a}")
    return FieldKernel(
        name=f"xck_gauxc_hess_rows_{family}",
        exprs=exprs,
        layout=ExplicitLayout(targets=targets, ret="None"),
        doc=(f"{family}: per-function nuclear-displacement field rows.",
             "Sum over the basis functions of one atom to get the perturbed",
             "fields rho^(A,x), grad rho^(A,x) and tau^(A,x) at this point.",
             "dchi/ddchi carry the -d/dr sign and the atom mask.")
    )


def spec_pair(family: str) -> FieldKernel:
    """Outer-product half of the pair term, from the summed rows."""
    e = pair_expression(family)
    return FieldKernel(
        name=f"xck_gauxc_hess_pair_{family}",
        exprs={"h": e},
        layout=ExplicitLayout(targets=["h"], ret="None"),
        doc=(f"{family}: non-Pulay pair term, per grid point.",
             "The fxc quadratic form on the two perturbed field sets, PLUS",
             "the sigma-curvature term 2 vsigma G^A . G^B -- sigma is",
             "quadratic in grad rho, so d2 sigma / dA dB survives for A /= B.",
             "Multiply by the quadrature weight at the call site.")
    )


def _seed_specs(family: str):
    """The Pulay (D_u_v) and same-atom members, COMPLETE rather than
    i-generic.

    ``geometric_hessian``'s hints expose one Cartesian instance of each
    seed and leave the host to call it per component and sum. Emitting
    the assembled term instead keeps that bookkeeping on this side of
    the interface, where it is checked; a host that mis-orders the
    components would otherwise get a plausible, wrong Hessian. ``D_u_v``
    appears linearly in every Pulay monomial, so it factors out and the
    host multiplies by its own density-matrix element.
    """
    gh = geometric_hessian(family)
    D = sp.Symbol("D_u_v", real=True)

    ex = sp.expand(gh.pair)
    terms = ex.args if ex.is_Add else (ex,)
    withD = sum(t for t in terms if t.has(D))
    pulay = sp.simplify(withD / D) if withD != 0 else sp.Integer(0)
    if pulay.has(D):
        raise AssertionError(f"{family}: D_u_v is not linear in the Pulay term")

    out = []
    if pulay != 0:
        out.append(FieldKernel(
            name=f"xck_gauxc_hess_pulay_{family}",
            exprs={"s": pulay},
            layout=ExplicitLayout(targets=["s"], ret="None"),
            doc=(f"{family}: Pulay term per basis-function PAIR, with the",
                 "density-matrix factor D_uv DIVIDED OUT -- multiply by it,",
                 "and by the quadrature weight, at the call site.",
                 "Cannot factorise into an outer product; this is the only",
                 "piece that needs the function pair.")
        ))

    same = sp.expand(gh.same)
    if same != 0:
        out.append(FieldKernel(
            name=f"xck_gauxc_hess_same_{family}",
            exprs={"s": same},
            layout=ExplicitLayout(targets=["s"], ret="None"),
            doc=(f"{family}: same-atom (delta_AB) term, per basis function.",
                 "Complete over Cartesian components; multiply by the",
                 "quadrature weight at the call site.")
        ))
    return out


# ---------------------------------------------------------------------------
# Generated CALL SITES
#
# The kernels take positional doubles in the order sympy discovers their
# operands -- sorted by libxckernel's own names. A host that writes the
# calls by hand has to reproduce that order from a different vocabulary:
# GauXC calls sigma "gamma", and "gamma" and "sigma" sort differently. The
# meta-GGA pair call was written in GauXC's order and mis-bound five of its
# seven Libxc derivatives, a Hessian wrong by ~1e5 that the compiler could
# not see (every argument is a double) and only a finite-difference check
# caught.
#
# So the call sites are generated too. The argument ORDER comes from each
# kernel's own signature; each argument is bound by NAME through the table
# below. An operand with no binding stops generation rather than being
# bound to the wrong array. The per-atom row layout -- which slot holds
# rho, sigma, tau and the gradient rows -- is emitted for the writes and
# the reads alike, so the two cannot disagree either.
# ---------------------------------------------------------------------------

#: Per-grid-point quantities, by GauXC's own array names. THE one place
#: that knows GauXC's vocabulary.
_POINT = {
    "vrho": "vrho[ip]", "vsigma": "vgamma[ip]", "vtau": "vtau[ip]",
    "v2rho2": "v2rho2[ip]", "v2rhosigma": "v2rhogamma[ip]",
    "v2rhotau": "v2rhotau[ip]", "v2sigma2": "v2gamma2[ip]",
    "v2sigmatau": "v2gammatau[ip]", "v2tau2": "v2tau2[ip]",
    "w": "weights[ip]",
    "grad_rho_x": "dden_x[ip]", "grad_rho_y": "dden_y[ip]",
    "grad_rho_z": "dden_z[ip]",
}


def row_layout(family: str) -> dict:
    """Slot of each per-atom row: the fields first, then G_x, G_y, G_z.

    The host allocates ``nfield + 3`` slots with ``nfield`` the number of
    ingredients; this is the same rule, stated once.
    """
    ings = _ingredients(family)
    lay = {f"F_{k}": i for i, k in enumerate(ings)}
    for c, a in enumerate(AXES):
        lay[f"G_{a}"] = len(ings) + c
    return lay


def _bindings(kind: str, family: str) -> dict:
    """Operand/target -> GauXC expression for one call site."""
    b = dict(_POINT)
    if kind == "rows":
        b.update({"U0_u": "U0", "U1_u": "U1", "U2_u": "U2", "U3_u": "U3",
                  "dchi_gA_u": "dchi"})
        for c, a in enumerate(AXES):
            b[f"ddchi_gA_u_{a}"] = f"ddchi[{c}]"
        b.update({"F_rho": "F_rho", "F_sigma": "F_sigma", "F_tau": "F_tau",
                  "G_x": "Gx", "G_y": "Gy", "G_z": "Gz"})
    elif kind == "pair":
        lay = row_layout(family)
        for k in _ingredients(family):
            b[f"F_{k}_A"] = f"ROW(a,dx,{lay[f'F_{k}']},ip)"
            b[f"F_{k}_B"] = f"ROW(b,dy,{lay[f'F_{k}']},ip)"
        for a in AXES:
            b[f"G_A_{a}"] = f"ROW(a,dx,{lay[f'G_{a}']},ip)"
            b[f"G_B_{a}"] = f"ROW(b,dy,{lay[f'G_{a}']},ip)"
        b["h"] = "h"
    elif kind == "pulay":
        b.update({"dchi_gA_u": "dAu", "dchi_gB_v": "dBv"})
        for c, a in enumerate(AXES):
            b[f"ddchi_gA_u_{a}"] = f"ddA[{c}]"
            b[f"ddchi_gB_v_{a}"] = f"ddB[{c}]"
        b["s"] = "sv"
    elif kind == "pulayW":
        for a in range(4):
            for c in range(4):
                b[f"W{a}{c}"] = f"W[{4*a+c}]"
    elif kind == "egrad":
        lay = row_layout(family)
        for k in _ingredients(family):
            b[f"F_{k}_A"] = f"ROW(a,d,{lay[f'F_{k}']},ip)"
        b["de"] = "de"
    elif kind == "mu":
        b = {}
        for c, a in enumerate(AXES):
            b[f"R_D_{a}"] = f"RD[{c}]"
            b[f"R_E_{a}"] = f"RE[{c}]"
            b[f"r_{a}"] = f"rg[{c}]"
        b["mu"] = "mu"
        for i in range(6):
            b[f"dmu_{i}"] = f"dmu[{i}]"
            for j in range(6):
                b[f"d2mu_{i}_{j}"] = f"d2mu[{6 * i + j}]"
    elif kind.startswith("cell_"):
        b = {"mu": "mu", "s": "s", "t": "t", "u": "u"}
    elif kind == "same":
        b.update({"U0_u": "xmat[k]", "U1_u": "xmat_x[k]",
                  "U2_u": "xmat_y[k]", "U3_u": "xmat_z[k]",
                  "d2chi_g2_u": "d2c"})
        for c, a in enumerate(AXES):
            b[f"d3chi_g2_u_{a}"] = f"d3c[{c}]"
        b["s"] = "sv"
    return b


def _spec_for(kind: str, family: str) -> FieldKernel:
    if kind == "rows":
        return spec_rows(family)
    if kind == "pair":
        return spec_pair(family)
    if kind == "pulayW":
        return spec_pulay_weights(family)
    if kind == "egrad":
        return spec_egrad(family)
    if kind == "mu":
        return spec_mu_derivs()
    if kind.startswith("cell_"):
        return spec_cell(kind[5:])
    for s in _seed_specs(family):
        if s.name == f"xck_gauxc_hess_{kind}_{family}":
            return s
    raise KeyError(f"{kind}/{family}")


def _bound_call(kind: str, family: str) -> str:
    """One kernel call, arguments in the kernel's OWN order, bound by name."""
    spec = _spec_for(kind, family)
    b = _bindings(kind, family)
    ins = spec.operands()
    outs = [t for t, _ in spec.layout.assignments(spec.exprs)]
    missing = [n for n in ins + outs if n not in b]
    if missing:
        raise KeyError(f"{spec.name}: no GauXC binding for {missing}")
    args = [b[n] for n in ins + outs]
    # A binding table with a copy-paste slip would bind two operands to
    # one array. Refuse that too.
    dup = sorted({a for a in args if args.count(a) > 1})
    if dup:
        raise AssertionError(f"{spec.name}: operands share a binding: {dup}")
    return f"xckernel::{spec.name}( " + ", ".join(args) + " );"


def _dispatch(body_for) -> list:
    """if(mgga) / else if(gga) / else(lda), highest rung first, as the
    hand-written dispatch it replaces was ordered."""
    L = []
    for i, (cond, fam) in enumerate((("is_mgga", "mgga_tau"),
                                     ("is_gga", "gga"), (None, "lda"))):
        head = (f"if( {cond} ) {{" if i == 0 else
                f"}} else if( {cond} ) {{" if cond else "} else {")
        L.append(head)
        L.extend("  " + line for line in body_for(fam))
    L.append("}")
    return L


_BEGIN = ("// ==> BEGIN GENERATED CODE [xckernel gauxcwriter: {what}] <==\n"
          "// Arguments are bound BY NAME from each kernel's own signature;\n"
          "// regenerate rather than edit: python -m xckernel.emitters.gauxcwriter --emit-dir <dir>")
_END = "// ==> END GENERATED CODE <=="


def emit_call_sites() -> dict:
    """Every generated call site, keyed by include-file stem."""
    out = {}

    def rows_body(fam):
        lay = row_layout(fam)
        body = [_bound_call("rows", fam)]
        for name in sorted(lay, key=lay.get):
            host = _bindings("rows", fam)[name]
            body.append(f"ROW(a,d,{lay[name]},ip) += {host};")
        return body

    out["gauxc_hess_call_rows"] = rows_body
    out["gauxc_hess_call_pair"] = lambda fam: [_bound_call("pair", fam)]
    out["gauxc_hess_call_pulayW"] = lambda fam: [_bound_call("pulayW", fam)]
    out["gauxc_hess_call_same"] = lambda fam: [_bound_call("same", fam)]

    out["gauxc_hess_call_egrad"] = lambda fam: [_bound_call("egrad", fam)]

    texts = {}
    for stem, fn in out.items():
        what = stem.replace("gauxc_hess_call_", "") + " call site"
        texts[stem] = "\n".join([_BEGIN.format(what=what)] + _dispatch(fn)
                                 + [_END]) + "\n"

    # weight class: no functional rung, the partition scheme decides
    texts["gauxc_hess_call_mu"] = "\n".join(
        [_BEGIN.format(what="mu call site"), _bound_call("mu", None), _END]) + "\n"
    texts["gauxc_hess_call_cell"] = "\n".join(
        [_BEGIN.format(what="cell-function call site"),
         "if( is_becke ) {", "  " + _bound_call("cell_becke", None),
         "} else {", "  " + _bound_call("cell_ssf", None), "}", _END]) + "\n"
    return texts


def write_include_files(directory: str) -> list:
    import os
    os.makedirs(directory, exist_ok=True)
    written = []
    with open(os.path.join(directory, "gauxc_hess_kernel.hpp"), "w") as f:
        f.write(emit_header())
    written.append("gauxc_hess_kernel.hpp")
    for stem, text in sorted(emit_call_sites().items()):
        with open(os.path.join(directory, f"{stem}.inc"), "w") as f:
            f.write(text)
        written.append(f"{stem}.inc")
    return written


#: The row basis of the Pulay term on each side: the displaced function
#: itself (slot 0) and its three displaced gradient components (slots 1-3).
def _pulay_rows(side: str):
    lab, oth = ("u", "A") if side == "A" else ("v", "B")
    S = lambda n: sp.Symbol(n, real=True)
    return [S(f"dchi_g{oth}_{lab}")] + [S(f"ddchi_g{oth}_{lab}_{a}") for a in AXES]


@functools.lru_cache(maxsize=None)
def pulay_weights(family: str):
    """The Pulay term as a 4 x 4 POINT-WEIGHT matrix W_ab(g).

    Every Pulay monomial is exactly one u-row times one v-row times a
    per-point factor, so the term is bilinear in the two row sets:

        pulay_uv(g) = sum_ab alpha_a(u,g) W_ab(g) beta_b(v,g),

    with alpha = (dchi_gA, ddchi_gA_x, ddchi_gA_y, ddchi_gA_z) and beta the
    same on the v side. A host can then form
    T_uv = sum_a alpha_a diag(W_a.) beta^T as matrix products instead of
    calling a kernel for every function pair at every point -- the
    difference between BLAS-3 and nbe^2 npts scalar calls. Bilinearity
    makes the extraction exact: W_ab = d^2 pulay / d alpha_a d beta_b.
    """
    spec = [s for s in _seed_specs(family) if s.name.endswith(f"pulay_{family}")][0]
    e = sp.expand(spec.exprs["s"])
    pool = {sym.name: sym for sym in e.free_symbols}
    al = [pool.get(r.name, r) for r in _pulay_rows("A")]
    be = [pool.get(r.name, r) for r in _pulay_rows("B")]
    W = [[sp.expand(sp.diff(e, al[a], be[b])) for b in range(4)] for a in range(4)]

    # Prove it: the weights must rebuild the term exactly, and must not
    # still depend on either row set (bilinearity).
    rebuilt = sum(al[a] * W[a][b] * be[b] for a in range(4) for b in range(4))
    if sp.expand(rebuilt - e) != 0:
        raise AssertionError(f"{family}: Pulay term is not bilinear in the rows")
    rows = set(al) | set(be)
    for a in range(4):
        for b in range(4):
            if W[a][b].free_symbols & rows:
                raise AssertionError(f"{family}: W_{a}{b} still depends on a row")
    return W


def spec_pulay_weights(family: str) -> FieldKernel:
    """Emit the NONZERO W_ab(g) only; the host zero-fills a 4 x 4 array
    per point, so the rung decides which entries get written."""
    W = pulay_weights(family)
    exprs, targets = {}, []
    for a in range(4):
        for b in range(4):
            if W[a][b] != 0:
                exprs[f"W{a}{b}"] = W[a][b]
                targets.append(f"W{a}{b}")
    return FieldKernel(
        name=f"xck_gauxc_hess_pulayW_{family}",
        exprs=exprs,
        layout=ExplicitLayout(targets=targets, ret="None"),
        doc=(f"{family}: Pulay point-weight matrix W_ab, weight included.",
             "Slots: 0 = displaced function, 1-3 = its displaced gradient.",
             "pulay_uv = sum_ab alpha_a(u) W_ab beta_b(v): contract the rows",
             "as matrix products rather than per function pair.")
    )


# ---------------------------------------------------------------------------
# WEIGHT class: the pieces of the partition-weight Hessian
#
# w = q P_C / Z with P_D = prod_{E /= D} s(mu_DE), mu_DE = (r_D - r_E)/R_DE.
# The host works with log-derivatives, so what it needs per atom pair is
# mu and its gradient and Hessian in the six coordinates of D and E (the
# point held fixed -- the parent atom's coordinates are recovered by
# translational invariance), and per cell function s together with
# t = s'/s and u = s''/s. Both are generated here.
#
# t and u come from the FACTORED ln s, never from 1 - g: the latter
# cancels catastrophically as mu -> 1, exactly where t diverges.
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def spec_mu_derivs() -> FieldKernel:
    """mu_DE and its first and second derivatives in (R_D, R_E), the point
    r held fixed. Coordinates are ordered D_x, D_y, D_z, E_x, E_y, E_z;
    the Hessian is emitted in full (row-major 6 x 6)."""
    S = lambda n: sp.Symbol(n, real=True)
    D = [S(f"R_D_{a}") for a in AXES]
    E = [S(f"R_E_{a}") for a in AXES]
    r = [S(f"r_{a}") for a in AXES]
    dist = lambda u, v: sp.sqrt(sum((ui - vi)**2 for ui, vi in zip(u, v)))
    mu = (dist(r, D) - dist(r, E)) / dist(D, E)
    X = D + E
    exprs = {"mu": mu}
    targets = ["mu"]
    for i in range(6):
        exprs[f"dmu_{i}"] = sp.diff(mu, X[i])
        targets.append(f"dmu_{i}")
    for i in range(6):
        for j in range(6):
            exprs[f"d2mu_{i}_{j}"] = sp.diff(mu, X[i], X[j])
            targets.append(f"d2mu_{i}_{j}")
    return FieldKernel(
        name="xck_gauxc_weight_mu", exprs=exprs,
        layout=ExplicitLayout(targets=targets, ret="None"),
        doc=("Becke/SSF confocal coordinate mu_DE = (|r-R_D| - |r-R_E|)/|R_D-R_E|,",
             "its gradient and full Hessian in (R_D, R_E) at fixed r.")
    )


def _cell_log(kind: str):
    """(ln s up to a constant, s) in the variable mu, in factored form."""
    m = sp.Symbol("mu", real=True)
    if kind == "becke":
        h = lambda q: sp.Rational(3, 2) * q - q**3 / 2
        p1 = h(m)
        p2 = h(p1)
        lns = (8 * sp.log(1 - m) + 4 * sp.log(2 + m)
               + 2 * sp.log(2 + p1) + sp.log(2 + p2))
        s = (1 - m)**8 * (2 + m)**4 * (2 + p1)**2 * (2 + p2) / 256
    elif kind == "ssf":
        a = sp.Rational(64, 100)          # integrator::magic_ssf_factor
        z = m / a
        q = 5 * z**3 + 20 * z**2 + 29 * z + 16
        lns = 4 * sp.log(1 - z) + sp.log(q)
        s = (1 - z)**4 * q / 32
    else:
        raise KeyError(kind)
    return m, lns, s


@functools.lru_cache(maxsize=None)
def cell_functions(kind: str):
    """s, t = s'/s, u = s''/s, proven against the textbook definitions."""
    m, lns, s = _cell_log(kind)
    t = sp.diff(lns, m)
    u = sp.diff(lns, m, 2) + t**2
    # the textbook cell functions, as the weights code evaluates them
    if kind == "becke":
        h = lambda q: sp.Rational(3, 2) * q - q**3 / 2
        ref = (1 - h(h(h(m)))) / 2
    else:
        z = m / sp.Rational(64, 100)
        ref = (1 - (35 * (z - z**3) + 21 * z**5 - 5 * z**7) / 16) / 2
    if sp.expand(s - ref) != 0:
        raise AssertionError(f"{kind}: factored s differs from the definition")
    for num, name in ((sp.diff(ref, m), "t"), (sp.diff(ref, m, 2), "u")):
        mine = t if name == "t" else u
        if sp.cancel(mine * ref - num) != 0:
            raise AssertionError(f"{kind}: {name} is not the {name}-ratio")
    return s, t, u


def spec_cell(kind: str) -> FieldKernel:
    s, t, u = cell_functions(kind)
    return FieldKernel(
        name=f"xck_gauxc_weight_cell_{kind}",
        exprs={"s": s, "t": t, "u": u},
        layout=ExplicitLayout(targets=["s", "t", "u"], ret="None"),
        doc=(f"{kind} cell function s(mu) with t = s'/s and u = s''/s, from",
             "the factored ln s. Valid strictly inside the switching region",
             "(|mu| < 1 Becke, |mu| < a SSF); the host handles the outside.")
    )


def spec_egrad(family: str) -> FieldKernel:
    """Basis-class first derivative of the energy DENSITY at one point,
    for one atom and direction: the chain rule through the same per-atom
    rows F_k the Hessian's outer product uses. The weight-class cross
    term pairs it with the weight gradient."""
    S = lambda n: sp.Symbol(n, real=True)
    de = sum(S(f"v{k}") * S(f"F_{k}_A") for k in _ingredients(family))
    return FieldKernel(
        name=f"xck_gauxc_hess_egrad_{family}",
        exprs={"de": de},
        layout=ExplicitLayout(targets=["de"], ret="None"),
        doc=(f"{family}: d e(r_g) / d R_A, e = the unweighted energy density,",
             "basis class only (points fixed). No quadrature weight.")
    )


def specs():
    out = []
    for fam in FAMILIES:
        _check_pair_decomposition(fam)
        out.append(spec_rows(fam))
        out.append(spec_pair(fam))
        out.extend(_seed_specs(fam))
        out.append(spec_pulay_weights(fam))
        out.append(spec_egrad(fam))
    out.append(spec_mu_derivs())
    out.extend(spec_cell(k) for k in ("becke", "ssf"))
    return out


def emit_header(cse: bool = True) -> str:
    body = [fieldkernel.emit_cxx(s, cse=cse, drop_zero=True) for s in specs()]
    pre = "\n".join([
        "// Machine-generated by xckernel; do not edit.",
        "// Reproduce with: python -m xckernel.emitters.gauxcwriter --emit <file>",
        "// Copyright (c) 2026 Susi Lehtola.",
        "#pragma once",
        "#include <cmath>",
        "",
        "namespace GauXC {",
        "namespace xckernel {",
        "",
    ])
    tail = "\n".join(["", "} // namespace xckernel", "} // namespace GauXC", ""])
    return pre + "\n\n".join(body) + tail


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--emit", metavar="FILE",
                   help="write only the kernel header")
    p.add_argument("--emit-dir", metavar="DIR",
                   help="write the kernel header AND the generated call sites")
    p.add_argument("--no-cse", action="store_true")
    a = p.parse_args(argv)
    if a.emit_dir:
        for f in write_include_files(a.emit_dir):
            print(f"wrote {a.emit_dir}/{f}")
        return
    src = emit_header(cse=not a.no_cse)
    if a.emit:
        with open(a.emit, "w") as f:
            f.write(src)
        print(f"wrote {a.emit}")
    else:
        print(src)


if __name__ == "__main__":
    main()
