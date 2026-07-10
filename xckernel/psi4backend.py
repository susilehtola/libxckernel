"""Psi4 emitter backend: digest the pattern-collapsed IR and emit the
contraction code of Psi4's libfock routines in Psi4's native idiom.

The interfaces-are-generated contract: the physics (which monomials, which
coefficients, which functional-derivative arrays) lives in the symbolic
tower and its collapsed einsum IR; this backend only rewrites operands
into Psi4's naming and scaling conventions and prints the loop shapes that
libfock uses. The emitted region is marked and reproducible; the
surrounding plumbing (collocation, perturbed densities, GEMMs against
scratch, scatter) stays host code.

Conventions encoded here (established and validated against Psi4 in the
mgga-vx work):

* Psi4's restricted Vx equals the unpolarized response kernel with
  perturbed fields built from (Dk + Dk^T)/2 and Libxc's unpolarized
  arrays; every TAU index of a Psi4 derivative array carries a factor
  1/2 relative to Libxc (V_TAU_A = vtau/2, V_RHO_A_TAU_A = v2rhotau/2,
  V_TAU_A_TAU_A = v2tau2/4, ...).
* Assembly: the (phi, phi) pattern enters the right-factor accumulator T
  at HALF weight (the trailing adjoint completion doubles it); the mixed
  (dphi_i, phi)/(phi, dphi_i) transpose pair enters once at full weight;
  the diagonal (dphi_i, dphi_i) patterns are symmetric on their own and
  are contracted after the adjoint completion at full weight.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .codegen import CollapsedKernel, collapse

#: our operand name -> (psi4 expression at point P, numeric factor)
PSI4_RV_VX_OPERANDS: Dict[str, Tuple[str, float]] = {
    "w": ("w[P]", 1.0),
    # ground-state fields
    "grad_rho_x": ("rho_x[P]", 1.0),
    "grad_rho_y": ("rho_y[P]", 1.0),
    "grad_rho_z": ("rho_z[P]", 1.0),
    # perturbed fields (rho_k conventions: rho_k = rho^X, rho_k_i = (grad rho)^X_i,
    # tau_k = tau^X, all for the symmetrized perturbed density)
    "rho_p1": ("rho_k[P]", 1.0),
    "grad_rho_p1_x": ("rho_k_x[P]", 1.0),
    "grad_rho_p1_y": ("rho_k_y[P]", 1.0),
    "grad_rho_p1_z": ("rho_k_z[P]", 1.0),
    "tau_p1": ("tau_k[P]", 1.0),
    # functional derivative arrays: Libxc name -> Psi4 vals[] pointer,
    # with the 1/2-per-TAU-index scaling folded into the factor
    "vrho": ("v_rho[P]", 1.0),
    "vsigma": ("v_gamma[P]", 1.0),
    "vtau": ("v_tau[P]", 2.0),
    "v2rho2": ("v2_rho2[P]", 1.0),
    "v2rhosigma": ("v2_rho_gamma[P]", 1.0),
    "v2sigma2": ("v2_gamma_gamma[P]", 1.0),
    "v2rhotau": ("v2_rho_tau[P]", 2.0),
    "v2sigmatau": ("v2_gamma_tau[P]", 2.0),
    "v2tau2": ("v2_tau_tau[P]", 4.0),
}

#: our operand name -> minimum ansatz that provides it
_ANSATZ_REQ = {
    "grad_rho": 1, "vsigma": 1, "v2rhosigma": 1, "v2sigma2": 1,
    "tau": 2, "vtau": 2, "v2rhotau": 2, "v2sigmatau": 2, "v2tau2": 2,
}

_BASIS_CODE = {"chi": "phi", "dchi[0]": "phi_x", "dchi[1]": "phi_y",
               "dchi[2]": "phi_z"}


def _ansatz_of(name: str) -> int:
    if "tau" in name:
        return 2
    if "sigma" in name or name.startswith("grad_rho"):
        return 1
    return 0


def _transform_monomials(monos, operands):
    """IR monomials -> (guard ansatz, coefficient, [psi4 exprs]) with the
    convention factors folded into the coefficient."""
    out = []
    for coeff, factors in monos:
        c = float(coeff)
        exprs: List[str] = []
        guard = 0
        for name, e in factors:
            if name not in operands:
                raise ValueError(f"psi4backend: no operand mapping for {name!r}")
            expr, fac = operands[name]
            c *= fac ** e
            exprs.extend([expr] * e)
            guard = max(guard, _ansatz_of(name))
        out.append((guard, c, sorted(exprs)))
    return out


def _cxx_sum(monos_t, weight: float, indent: str) -> List[str]:
    """Emit `c = ...;` accumulation statements grouped by ansatz guard."""
    by_guard: Dict[int, List[str]] = {}
    for guard, c, exprs in monos_t:
        cw = weight * c
        pre = [] if cw == 1.0 else [f"{cw:.17g}"]
        term = " * ".join(pre + exprs)
        by_guard.setdefault(guard, []).append(term)
    lines = []
    for guard in sorted(by_guard):
        terms = by_guard[guard]
        if guard > 0:
            lines.append(f"{indent}if (ansatz >= {guard}) {{")
            for t in terms:
                lines.append(f"{indent}    c += {t};")
            lines.append(f"{indent}}}")
        else:
            for t in terms:
                lines.append(f"{indent}c += {t};")
    return lines


def emit_rv_vx_contraction(family: str = "mgga_tau") -> str:
    """The generated contraction region of RV::compute_Vx_full: per-point
    coefficient assembly into the right-factor accumulator T, the
    completion GEMM + adjoint, and the diagonal (dphi_i, dphi_i) block.

    Drop-in for the region between the perturbed-field construction and
    the unpacking, using the surrounding function's variable names."""
    from .response import response_fock
    ck = collapse(response_fock(family, 2))
    (u_lbl, v_lbl) = (ck.u_lbl, ck.v_lbl)

    # organize patterns
    pat = {(u, v): monos for (u, v, monos) in ck.patterns}
    mixed = {}   # i -> monomials (merged transpose pair)
    diag = {}    # i -> monomials
    for (u, v), monos in pat.items():
        if u == "chi" and v == "chi":
            continue
        if u == "chi" and v.startswith("dchi["):
            i = int(v[5])
            mixed[i] = monos
        elif v == "chi" and u.startswith("dchi["):
            i = int(u[5])
            # transpose partner of the (chi, dchi) pattern: coefficients are
            # identical by construction; the adjoint completion supplies it.
            continue
        elif u == v and u.startswith("dchi["):
            diag[int(u[5])] = monos
        else:
            raise ValueError(f"psi4backend: unsupported pattern {(u, v)}")

    ops = PSI4_RV_VX_OPERANDS
    L: List[str] = []
    A = L.append
    A("            // ==> BEGIN GENERATED CODE"
      " [xckernel psi4backend: response_fock(%s, order=2), restricted] <==" % family)
    A("            // Reproduce with: python -m xckernel.psi4backend")
    A("            // Physics source: the symbolic derivative tower; every")
    A("            // coefficient below is IR output, not hand-derived.")
    A("            for (int P = 0; P < npoints; P++) {")
    A("                std::fill(Tp[P], Tp[P] + nlocal, 0.0);")
    A("                // Do a simple screen: ignore contributions where rho is too small.")
    A("                if (rho_a[P] < v2_rho_cutoff_) continue;")
    A("                double c;")
    A("                // (phi, phi) pattern at half weight (adjoint completion doubles)")
    A("                c = 0.0;")
    L.extend(_cxx_sum(_transform_monomials(pat[("chi", "chi")], ops), 0.5, "                "))
    A("                C_DAXPY(nlocal, c, phi[P], 1, Tp[P], 1);")
    for i, nm in enumerate(("phi_x", "phi_y", "phi_z")):
        A(f"                // (phi, dphi_{'xyz'[i]}) + transpose at full weight")
        A("                c = 0.0;")
        L.extend(_cxx_sum(_transform_monomials(mixed[i], ops), 1.0, "                "))
        A(f"                C_DAXPY(nlocal, c, {nm}[P], 1, Tp[P], 1);")
    A("            }")
    A("")
    A("            // ===> Contract T against phi, and complete with the adjoint <===")
    A("            C_DGEMM('T', 'N', nlocal, nlocal, npoints, 1.0, phi[0], coll_funcs, Tp[0], max_functions, 0.0,")
    A("                    Vx_localp[0], max_functions);")
    A("            for (int m = 0; m < nlocal; m++) {")
    A("                for (int n = 0; n <= m; n++) {")
    A("                    Vx_localp[m][n] = Vx_localp[n][m] = Vx_localp[m][n] + Vx_localp[n][m];")
    A("                }")
    A("            }")
    A("")
    if diag:
        guards = {g for i in diag for (g, c, e) in _transform_monomials(diag[i], ops)}
        outer = min(guards)
        A(f"            // (dphi_i, dphi_i) diagonal patterns: symmetric on their own,")
        A(f"            // contracted after the adjoint completion at full weight")
        A(f"            if (ansatz >= {outer}) {{")
        A("                double** phi_i[3] = {phi_x, phi_y, phi_z};")
        A("                for (int i = 0; i < 3; i++) {")
        A("                    for (int P = 0; P < npoints; P++) {")
        A("                        std::fill(Tp[P], Tp[P] + nlocal, 0.0);")
        A("                        if (rho_a[P] < v2_rho_cutoff_) continue;")
        A("                        double c;")
        # all three diagonal patterns carry identical monomials; assert and emit once
        m0 = _transform_monomials(diag[0], ops)
        for i in (1, 2):
            mi = _transform_monomials(diag[i], ops)
            if sorted((g, c, tuple(e)) for g, c, e in mi) != \
                    sorted((g, c, tuple(e)) for g, c, e in m0):
                raise ValueError("psi4backend: anisotropic diagonal patterns")
        A("                        c = 0.0;")
        L.extend(_cxx_sum(m0, 1.0, "                        "))
        A("                        C_DAXPY(nlocal, c, phi_i[i][P], 1, Tp[P], 1);")
        A("                    }")
        A("                    C_DGEMM('T', 'N', nlocal, nlocal, npoints, 1.0, phi_i[i][0], coll_funcs, Tp[0],")
        A("                            max_functions, 1.0, Vx_localp[0], max_functions);")
        A("                }")
        A("            }")
    A("            // ==> END GENERATED CODE <==")
    return "\n".join(L)


if __name__ == "__main__":
    print(emit_rv_vx_contraction())
