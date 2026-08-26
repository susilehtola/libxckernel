"""Validate the generated open-shell VeloxChem contractions.

VeloxChem has no open-shell kxc or lxc on any rung (XCIntegrator throws
"Not implemented for open-shell"), so there is no reference
implementation to compare against directly.  There is, however, a
reference for the SPIN-COMPENSATED LIMIT: their hand-written closed-shell
routines, which are in production and trusted.

Setting rho_a = rho_b in the generated open-shell expression must
reproduce their folded coefficient combinations exactly -- the binomial
patterns that appear in their source as

    rr   = (v2rho2_aa + v2rho2_ab)
    rrr  = (v3rho3_aaa + 2 v3rho3_aab + v3rho3_abb)
    rrrr = (v4rho4_aaaa + 3 v4rho4_aaab + 3 v4rho4_aabb + v4rho4_abbb)

transcribed here from XCIntegratorForLDA.cpp.  Those combinations are
only valid when the two spin densities coincide; the generated
expression carries the spin sums explicitly and so remains valid when
they do not, which is the whole point of the contribution.

Run with: python -m xckernel.tests.vlx_openshell_validate
"""

from __future__ import annotations

import sympy as sp

from ..engine.spin_kernel import response_fock_spin

#: The closed-shell coefficient combinations VeloxChem folds by hand,
#: transcribed verbatim from src/dft_func/XCIntegratorForLDA.cpp.
#: order -> (libxc array, {flat component index: multiplicity})
VELOXCHEM_CLOSED_SHELL = {
    2: ("v2rho2", {0: 1, 1: 1}),                    # rr
    3: ("v3rho3", {0: 1, 1: 2, 2: 1}),              # rrr
    4: ("v4rho4", {0: 1, 1: 3, 2: 3, 3: 1}),        # rrrr
}


def _generated_alpha(order: int) -> sp.Expr:
    """Our open-shell alpha-channel coefficient, weight and pair divided out."""
    ri = response_fock_spin("lda", "a", order)
    cu = sp.Symbol("chi_u", real=True)
    cv = sp.Symbol("chi_v", real=True)
    w = sp.Symbol("w", real=True, positive=True)
    return sp.expand(ri.expr / (cu * cv * w))


def check_closed_shell_limit(order: int):
    """rho_a = rho_b must collapse the spin sums onto their folding."""
    expr = _generated_alpha(order)
    labels = [f"p{k}" for k in range(1, order)]

    # collapse both spin channels of every perturbation onto one density
    sub = {}
    dens = []
    for lab in labels:
        r = sp.Symbol(f"r_{lab}", real=True)
        dens.append(r)
        for s in ("a", "b"):
            sub[sp.Symbol(f"rho_{s}_{lab}", real=True)] = r
    got = sp.expand(expr.subs(sub))

    array, folding = VELOXCHEM_CLOSED_SHELL[order]
    want = sp.expand(
        sum(mult * sp.Symbol(f"{array}_{idx}", real=True)
            for idx, mult in folding.items())
        * sp.prod(dens))

    ok = sp.simplify(got - want) == 0
    return (f"LDA order {order}: closed-shell limit matches VeloxChem's "
            f"hand-folded coefficients", ok, got if not ok else None)


def check_spin_sum_completeness(order: int):
    """Every spin combination of the perturbed densities must appear.

    A dropped spin channel would still pass the closed-shell check, since
    collapsing rho_a = rho_b hides which channel a term came from.  There
    are 2^(order-1) sign patterns and each must be present exactly once
    per derivative component that can reach it.
    """
    expr = _generated_alpha(order)
    labels = [f"p{k}" for k in range(1, order)]
    seen = set()
    for mono in sp.Add.make_args(expr):
        # pair each spin with ITS perturbation label: collecting spins in
        # factor order conflates rho_b_p1*rho_a_p2 with rho_a_p1*rho_b_p2
        by_label = {}
        for fac in mono.as_ordered_factors():
            base, exp = fac.as_base_exp()
            name = getattr(base, "name", "")
            if name.startswith("rho_"):
                _, spin, lab = name.split("_")
                for _ in range(int(exp)):
                    by_label.setdefault(lab, []).append(spin)
        seen.add(tuple(by_label[lab].pop() for lab in labels))
    want = 2 ** len(labels)
    return (f"LDA order {order}: all {want} spin combinations present",
            len(seen) == want, None if len(seen) == want else sorted(seen))


def main():
    checks = []
    for order in (2, 3, 4):
        checks.append(check_closed_shell_limit(order))
        checks.append(check_spin_sum_completeness(order))
    bad = [c for c in checks if not c[1]]
    for label, ok, detail in checks:
        print(f"  {'ok  ' if ok else 'FAIL'} {label}")
        if detail is not None:
            print(f"        got: {detail}")
    tag = "OK " if not bad else "FAIL"
    print(f"[{tag}] vlx_openshell_validate: {len(checks)} checks, "
          f"{len(bad)} failures")
    raise SystemExit(0 if not bad else 1)


if __name__ == "__main__":
    main()
