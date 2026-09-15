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
    """The Pulay (D_u_v) and same-atom members, straight from the hints."""
    gh = geometric_hessian(family)
    h = gh.hints
    out = []
    for kind in ("pair", "same"):
        exprs, targets = {}, []
        for k in _ingredients(family):
            key = f"seed_{kind}_{k}" if k == "rho" else f"seed_{kind}_{k}_i"
            if key not in h:
                continue
            exprs[f"s_{k}"] = h[key]
            targets.append(f"s_{k}")
        if not exprs:
            continue
        out.append(FieldKernel(
            name=f"xck_gauxc_hess_seed_{kind}_{family}",
            exprs=exprs,
            layout=ExplicitLayout(targets=targets, ret="None"),
            doc=((f"{family}: Pulay term, per basis-function PAIR (u in A, v in B).",
                  "Carries the density-matrix pair factor, so it cannot",
                  "factorise into an outer product.")
                 if kind == "pair" else
                 (f"{family}: same-atom (delta_AB) term, per basis function.",
                  "The sigma/tau members are i-generic: call once per",
                  "Cartesian component and sum."))
        ))
    return out


def specs():
    out = []
    for fam in FAMILIES:
        _check_pair_decomposition(fam)
        out.append(spec_rows(fam))
        out.append(spec_pair(fam))
        out.extend(_seed_specs(fam))
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
    p.add_argument("--emit", metavar="FILE")
    p.add_argument("--no-cse", action="store_true")
    a = p.parse_args(argv)
    src = emit_header(cse=not a.no_cse)
    if a.emit:
        with open(a.emit, "w") as f:
            f.write(src)
        print(f"wrote {a.emit}")
    else:
        print(src)


if __name__ == "__main__":
    main()
