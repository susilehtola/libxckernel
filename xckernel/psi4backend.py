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
from .compact import VECTOR_GROUPS, contract_dots, hoist_common

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


def _cxx_sum(monos_t, weight: float, indent: str, var: str = "c") -> List[str]:
    """Emit `<var> += ...;` accumulation statements grouped by ansatz guard."""
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
                lines.append(f"{indent}    {var} += {t};")
            lines.append(f"{indent}}}")
        else:
            for t in terms:
                lines.append(f"{indent}{var} += {t};")
    return lines


def _compact_patterns(phiphi, mixed, diag):
    """Run the IR compaction passes; returns compacted monomials plus the
    dot / hoist definition tables."""
    dots = {}
    phiphi, dd = contract_dots([(float(c), tuple(f)) for c, f in phiphi])
    dots.update(dd)
    mixed_c = {}
    for i in range(3):
        mixed_c[i], dd = contract_dots([(float(c), tuple(f)) for c, f in mixed[i]])
        dots.update(dd)
    hoists, mixed_c = hoist_common(mixed_c)
    diag_c = {}
    for i in diag:
        diag_c[i], dd = contract_dots([(float(c), tuple(f)) for c, f in diag[i]])
        dots.update(dd)
    return phiphi, mixed_c, diag_c, dots, hoists


def _emit_intermediates(dots, hoists, ops, indent) -> Tuple[List[str], Dict[str, Tuple[str, float]]]:
    """Per-point intermediate definitions (dot products, hoisted sums),
    ansatz-guarded; extends the operand map with the new names."""
    ops = dict(ops)
    lines: List[str] = []
    for name in sorted(dots):
        g1, g2 = dots[name]
        comps = [f"{ops[a][0]} * {ops[b][0]}"
                 for a, b in zip(VECTOR_GROUPS[g1], VECTOR_GROUPS[g2])]
        guard = max(_ansatz_of(a) for a in VECTOR_GROUPS[g1] + VECTOR_GROUPS[g2])
        if guard > 0:
            lines.append(f"{indent}double {name} = 0.0;")
            lines.append(f"{indent}if (ansatz >= {guard}) {name} = "
                         + " + ".join(comps) + ";")
        else:
            lines.append(f"{indent}const double {name} = " + " + ".join(comps) + ";")
        ops[name] = (name, 1.0)
    for name in sorted(hoists):
        g, rem = hoists[name]
        monos_t = _transform_monomials(rem, ops)
        lines.append(f"{indent}double {name} = 0.0;")
        by_guard: Dict[int, List[str]] = {}
        for guard, c, exprs in monos_t:
            pre = [] if c == 1.0 else [f"{c:.17g}"]
            by_guard.setdefault(guard, []).append(" * ".join(pre + exprs) or "1.0")
        for guard in sorted(by_guard):
            terms = by_guard[guard]
            if guard > 0:
                lines.append(f"{indent}if (ansatz >= {guard}) {{")
                for t in terms:
                    lines.append(f"{indent}    {name} += {t};")
                lines.append(f"{indent}}}")
            else:
                for t in terms:
                    lines.append(f"{indent}{name} += {t};")
        ops[name] = (name, 1.0)
    return lines, ops


def emit_rv_vx_contraction(family: str = "mgga_tau") -> str:
    """The generated contraction region of RV::compute_Vx_full: per-point
    intermediates (dot products, hoisted common factors) and coefficient
    assembly into the right-factor accumulator T, the completion GEMM +
    adjoint, and the diagonal (dphi_i, dphi_i) block."""
    from .response import response_fock
    ck = collapse(response_fock(family, 2))
    phiphi, mixed, diag = _split_patterns(ck)
    phiphi, mixed, diag, dots, hoists = _compact_patterns(phiphi, mixed, diag)
    ind = "                "
    defs, ops = _emit_intermediates(dots, hoists, PSI4_RV_VX_OPERANDS, ind)

    L: List[str] = []
    A = L.append
    A("            // ==> BEGIN GENERATED CODE"
      " [xckernel psi4backend: response_fock(%s, order=2), restricted] <==" % family)
    A("            // Reproduce with: python -m xckernel.psi4backend")
    A("            // Physics source: the symbolic derivative tower; the")
    A("            // intermediates below are IR compaction output (dot")
    A("            // contraction + common-factor hoisting), not hand-derived.")
    A("            for (int P = 0; P < npoints; P++) {")
    A("                std::fill(Tp[P], Tp[P] + nlocal, 0.0);")
    A("                // Do a simple screen: ignore contributions where rho is too small.")
    A("                if (rho_a[P] < v2_rho_cutoff_) continue;")
    L.extend(defs)
    A("                double c;")
    A("                // (phi, phi) pattern at half weight (adjoint completion doubles)")
    A("                c = 0.0;")
    L.extend(_cxx_sum(_transform_monomials(phiphi, ops), 0.5, ind))
    A("                C_DAXPY(nlocal, c, phi[P], 1, Tp[P], 1);")
    for i, nm in enumerate(("phi_x", "phi_y", "phi_z")):
        # the gradient collocation itself needs ansatz >= 1: guard the
        # whole statement group (an alpha of zero does not excuse
        # dereferencing an uninitialized pointer in the DAXPY arguments)
        A(f"                // (phi, dphi_{'xyz'[i]}) + transpose at full weight")
        A("                if (ansatz >= 1) {")
        A("                    c = 0.0;")
        L.extend(_cxx_sum(_transform_monomials(mixed[i], ops), 1.0, ind + "    "))
        A(f"                    C_DAXPY(nlocal, c, {nm}[P], 1, Tp[P], 1);")
        A("                }")
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
        m0 = _transform_monomials(diag[0], ops)
        for i in (1, 2):
            mi = _transform_monomials(diag[i], ops)
            if sorted((g, c, tuple(e)) for g, c, e in mi) != \
                    sorted((g, c, tuple(e)) for g, c, e in m0):
                raise ValueError("psi4backend: anisotropic diagonal patterns")
        used = {n for c, f in diag[0] for n, e in f if n in dots}
        ind2 = "                        "
        defs2, ops2 = _emit_intermediates({k: dots[k] for k in used}, {},
                                          PSI4_RV_VX_OPERANDS, ind2)
        guards = {g for (g, c, e) in _transform_monomials(diag[0], ops2)}
        A(f"            // (dphi_i, dphi_i) diagonal patterns: symmetric on their own,")
        A(f"            // contracted after the adjoint completion at full weight")
        A(f"            if (ansatz >= {min(guards)}) {{")
        A("                double** phi_i[3] = {phi_x, phi_y, phi_z};")
        A("                for (int i = 0; i < 3; i++) {")
        A("                    for (int P = 0; P < npoints; P++) {")
        A("                        std::fill(Tp[P], Tp[P] + nlocal, 0.0);")
        A("                        if (rho_a[P] < v2_rho_cutoff_) continue;")
        L.extend(defs2)
        A("                        double c;")
        A("                        c = 0.0;")
        L.extend(_cxx_sum(_transform_monomials(diag[0], ops2), 1.0, ind2))
        A("                        C_DAXPY(nlocal, c, phi_i[i][P], 1, Tp[P], 1);")
        A("                    }")
        A("                    C_DGEMM('T', 'N', nlocal, nlocal, npoints, 1.0, phi_i[i][0], coll_funcs, Tp[0],")
        A("                            max_functions, 1.0, Vx_localp[0], max_functions);")
        A("                }")
        A("            }")
    A("            // ==> END GENERATED CODE <==")
    return "\n".join(L)

# --- unrestricted (UV) conventions -------------------------------------------

def _uv_operands() -> Dict[str, Tuple[str, float]]:
    """Polarized operand map for UV::compute_Vx: Libxc component names to
    Psi4's spin-resolved arrays, with the 1/2-per-TAU-index scalings."""
    ops: Dict[str, Tuple[str, float]] = {"w": ("w[P]", 1.0)}
    for s_, tag in (("a", "a"), ("b", "b")):
        ops[f"rho_{s_}_p1"] = (f"rho_{tag}k[P]", 1.0)
        ops[f"tau_{s_}_p1"] = (f"tau_{tag}k[P]", 1.0)
        for ax in "xyz":
            ops[f"grad_rho_{s_}_p1_{ax}"] = (f"rho_{tag}k_{ax}[P]", 1.0)
            ops[f"grad_rho_{s_}_{ax}"] = (f"rho_{tag}{ax}[P]", 1.0)
    # first derivatives
    for i, g in enumerate(("aa", "ab", "bb")):
        ops[f"vsigma_{i}"] = (f"v_gamma_{g}[P]", 1.0)
    for i, t in enumerate(("a", "b")):
        ops[f"vtau_{i}"] = (f"v_tau_{t}[P]", 2.0)
        ops[f"vrho_{i}"] = (f"v_rho_{t}[P]", 1.0)
    # second derivatives, Libxc component packing
    for i, rr in enumerate(("aa", "ab", "bb")):
        ops[f"v2rho2_{i}"] = (f"v2_rho2_{rr}[P]", 1.0)
        ops[f"v2tau2_{i}"] = (f"v2_tau_{rr[0]}_tau_{rr[1]}[P]", 4.0)
    k = 0
    for r in ("a", "b"):
        for g in ("aa", "ab", "bb"):
            ops[f"v2rhosigma_{k}"] = (f"v2_rho_{r}_gamma_{g}[P]", 1.0)
            k += 1
    k = 0
    for gi in range(3):
        for gj in range(gi, 3):
            g1 = ("aa", "ab", "bb")[gi]
            g2 = ("aa", "ab", "bb")[gj]
            ops[f"v2sigma2_{k}"] = (f"v2_gamma_{g1}_gamma_{g2}[P]", 1.0)
            k += 1
    k = 0
    for r in ("a", "b"):
        for t in ("a", "b"):
            ops[f"v2rhotau_{k}"] = (f"v2_rho_{r}_tau_{t}[P]", 2.0)
            k += 1
    k = 0
    for g in ("aa", "ab", "bb"):
        for t in ("a", "b"):
            ops[f"v2sigmatau_{k}"] = (f"v2_gamma_{g}_tau_{t}[P]", 2.0)
            k += 1
    return ops


def _split_patterns(ck):
    """Organize a collapsed kernel into (phiphi, mixed{i}, diag{i}),
    verifying the transpose-pair and isotropy assumptions."""
    pat = {(u, v): monos for (u, v, monos) in ck.patterns}
    mixed, diag = {}, {}
    phiphi = pat[("chi", "chi")]
    for (u, v), monos in pat.items():
        if u == "chi" and v == "chi":
            continue
        if u == "chi" and v.startswith("dchi["):
            mixed[int(v[5])] = monos
        elif v == "chi" and u.startswith("dchi["):
            i = int(u[5])
            if sorted(monos) != sorted(pat[("chi", f"dchi[{i}]")]):
                raise ValueError("psi4backend: asymmetric mixed pattern")
        elif u == v and u.startswith("dchi["):
            diag[int(u[5])] = monos
        else:
            raise ValueError(f"psi4backend: unsupported pattern {(u, v)}")
    return phiphi, mixed, diag


def emit_uv_vx_contraction(family: str = "mgga_tau") -> str:
    """The generated contraction region of UV::compute_Vx: per-point
    intermediates and both spin coefficient assemblies fused in one
    point loop, per-spin completion GEMMs + adjoint, and the diagonal
    (dphi_i, dphi_i) blocks."""
    from .spin_kernel import response_fock_spin
    base_ops = _uv_operands()
    chan = {}
    dots_all, hoists_all = {}, {}
    for s_ in ("a", "b"):
        ck = collapse(response_fock_spin(family, s_, 2))
        phiphi, mixed, diag = _split_patterns(ck)
        phiphi, mixed, diag, dots, hoists = _compact_patterns(phiphi, mixed, diag)
        # keep hoist names distinct per spin channel
        ren = {name: f"{name}_{s_}" for name in hoists}
        hoists = {ren[k]: v for k, v in hoists.items()}
        for i in range(3):
            mixed[i] = [(c, tuple(sorted((ren.get(n, n), e) for n, e in f)))
                        for c, f in mixed[i]]
        chan[s_] = (phiphi, mixed, diag)
        dots_all.update(dots)
        hoists_all.update(hoists)
    ind = "                "
    defs, ops = _emit_intermediates(dots_all, hoists_all, base_ops, ind)

    L: List[str] = []
    A = L.append
    A("            // ==> BEGIN GENERATED CODE"
      " [xckernel psi4backend: response_fock_spin(%s, order=2)] <==" % family)
    A("            // Reproduce with: python -m xckernel.psi4backend --uv")
    A("            // Physics source: the symbolic derivative tower; the")
    A("            // intermediates below are IR compaction output (dot")
    A("            // contraction + common-factor hoisting), not hand-derived.")
    A("            for (int P = 0; P < npoints; P++) {")
    A("                std::fill(Tap[P], Tap[P] + nlocal, 0.0);")
    A("                std::fill(Tbp[P], Tbp[P] + nlocal, 0.0);")
    A("                if (rho_a[P] + rho_b[P] < v2_rho_cutoff_) continue;")
    L.extend(defs)
    A("                double c;")
    for s_, T in (("a", "Tap"), ("b", "Tbp")):
        phiphi, mixed, diag = chan[s_]
        A(f"                // spin {s_}: (phi, phi) pattern at half weight")
        A("                c = 0.0;")
        L.extend(_cxx_sum(_transform_monomials(phiphi, ops), 0.5, ind))
        A(f"                C_DAXPY(nlocal, c, phi[P], 1, {T}[P], 1);")
        for i, nm in enumerate(("phi_x", "phi_y", "phi_z")):
            # guard the whole group: phi_x is uninitialized below ansatz 1
            A(f"                // spin {s_}: (phi, dphi_{'xyz'[i]}) + transpose at full weight")
            A("                if (ansatz >= 1) {")
            A("                    c = 0.0;")
            L.extend(_cxx_sum(_transform_monomials(mixed[i], ops), 1.0, ind + "    "))
            A(f"                    C_DAXPY(nlocal, c, {nm}[P], 1, {T}[P], 1);")
            A("                }")
    A("            }")
    A("")
    A("            // ===> Contract Ta and Tb against phi, and complete with the adjoint <===")
    for T, V in (("Tap", "Vax_localp"), ("Tbp", "Vbx_localp")):
        A(f"            C_DGEMM('T', 'N', nlocal, nlocal, npoints, 1.0, phi[0], coll_funcs, {T}[0], max_functions, 0.0,")
        A(f"                    {V}[0], max_functions);")
    A("            for (int m = 0; m < nlocal; m++) {")
    A("                for (int n = 0; n <= m; n++) {")
    A("                    Vax_localp[m][n] = Vax_localp[n][m] = Vax_localp[m][n] + Vax_localp[n][m];")
    A("                    Vbx_localp[m][n] = Vbx_localp[n][m] = Vbx_localp[m][n] + Vbx_localp[n][m];")
    A("                }")
    A("            }")
    A("")
    diag_a = chan["a"][2]
    if diag_a:
        for s_ in ("a", "b"):
            m0 = _transform_monomials(chan[s_][2][0], ops)
            for i in (1, 2):
                mi = _transform_monomials(chan[s_][2][i], ops)
                if sorted((g, c, tuple(e)) for g, c, e in mi) != \
                        sorted((g, c, tuple(e)) for g, c, e in m0):
                    raise ValueError("psi4backend: anisotropic diagonal patterns")
        used = {n for s_ in ("a", "b") for c, f in chan[s_][2][0]
                for n, e in f if n in dots_all}
        ind2 = "                        "
        defs2, ops2 = _emit_intermediates({k: dots_all[k] for k in used}, {},
                                          base_ops, ind2)
        guards = {g for s_ in ("a", "b")
                  for (g, c, e) in _transform_monomials(chan[s_][2][0], ops2)}
        A(f"            // (dphi_i, dphi_i) diagonal patterns: symmetric on their own,")
        A(f"            // contracted after the adjoint completion at full weight")
        A(f"            if (ansatz >= {min(guards)}) {{")
        A("                double** phi_i[3] = {phi_x, phi_y, phi_z};")
        A("                for (int i = 0; i < 3; i++) {")
        A("                    for (int P = 0; P < npoints; P++) {")
        A("                        std::fill(Tap[P], Tap[P] + nlocal, 0.0);")
        A("                        std::fill(Tbp[P], Tbp[P] + nlocal, 0.0);")
        A("                        if (rho_a[P] + rho_b[P] < v2_rho_cutoff_) continue;")
        L.extend(defs2)
        A("                        double c;")
        for s_, T in (("a", "Tap"), ("b", "Tbp")):
            A("                        c = 0.0;")
            L.extend(_cxx_sum(_transform_monomials(chan[s_][2][0], ops2), 1.0, ind2))
            A(f"                        C_DAXPY(nlocal, c, phi_i[i][P], 1, {T}[P], 1);")
        A("                    }")
        for T, V in (("Tap", "Vax_localp"), ("Tbp", "Vbx_localp")):
            A(f"                    C_DGEMM('T', 'N', nlocal, nlocal, npoints, 1.0, phi_i[i][0], coll_funcs, {T}[0],")
            A(f"                            max_functions, 1.0, {V}[0], max_functions);")
        A("                }")
        A("            }")
    A("            // ==> END GENERATED CODE <==")
    return "\n".join(L)


# --- restricted Fock derivatives (dV/dR at fixed density) ---------------------

#: operand map for RV::compute_fock_derivatives (per-point locals for the
#: atom-restricted fixed-grid perturbed fields)
PSI4_RV_FX_OPERANDS: Dict[str, Tuple[str, float]] = {
    "w": ("w[P]", 1.0),
    "grad_rho_x": ("rho_g[0][P]", 1.0),
    "grad_rho_y": ("rho_g[1][P]", 1.0),
    "grad_rho_z": ("rho_g[2][P]", 1.0),
    "rho_p1": ("rho_k", 1.0),
    "grad_rho_p1_x": ("grad_k[0]", 1.0),
    "grad_rho_p1_y": ("grad_k[1]", 1.0),
    "grad_rho_p1_z": ("grad_k[2]", 1.0),
    "tau_p1": ("tau_k", 1.0),
    "vrho": ("v_rho_a[P]", 1.0),
    "vsigma": ("v_gamma[P]", 1.0),
    "vtau": ("v_tau[P]", 2.0),
    "v2rho2": ("v_rho_aa[P]", 1.0),
    "v2rhosigma": ("v2_rho_gamma[P]", 1.0),
    "v2sigma2": ("v2_gamma_gamma[P]", 1.0),
    "v2rhotau": ("v2_rho_tau[P]", 2.0),
    "v2sigmatau": ("v2_gamma_tau[P]", 2.0),
    "v2tau2": ("v2_tau_tau[P]", 4.0),
}

#: masked (atom-restricted) seed collocation slices; the -d/dr sign of the
#: geometric operands is folded into the emission factor below
_FX_MASKED = {
    "dchi_gA": "&phi_i[x][P][off]",
    "ddchi_gA[0]": "&phi_hess[hess_addr[x][0]][P][off]",
    "ddchi_gA[1]": "&phi_hess[hess_addr[x][1]][P][off]",
    "ddchi_gA[2]": "&phi_hess[hess_addr[x][2]][P][off]",
}


def emit_rv_fx_contraction(family: str = "mgga_tau") -> str:
    """The generated coefficient region of RV::compute_fock_derivatives,
    placed inside the per-(atom, direction) point loop after the plumbing
    computed the fixed-grid perturbed fields rho_k / grad_k[i] / tau_k.

    Weight rules for this routine (the accumulation visits every function
    pair from both sides, so ACC + ACC^T is applied twice): symmetric
    patterns at QUARTER weight, transpose pairs at HALF weight."""
    from .geometric import geometric_fock
    ck = collapse(geometric_fock(family))

    field_phiphi = []
    field_mixed: Dict[int, list] = {}
    field_diag: Dict[int, list] = {}
    seeds: List[Tuple[str, str, list]] = []   # (masked code, right code, monos)
    pat = {(u, v): m for (u, v, m) in ck.patterns}
    for (u, v), monos in pat.items():
        u_masked = u in _FX_MASKED
        v_masked = v in _FX_MASKED
        if u_masked and not v_masked:
            seeds.append((u, v, monos))
        elif v_masked and not u_masked:
            # transpose partner of a masked-left pattern; assert and skip
            if sorted(monos) != sorted(pat[(v, u)]):
                raise ValueError("psi4backend: asymmetric seed pattern")
        elif u_masked and v_masked:
            raise ValueError("psi4backend: doubly-masked pattern")
        elif u == "chi" and v == "chi":
            field_phiphi = monos
        elif u == "chi" and v.startswith("dchi["):
            field_mixed[int(v[5])] = monos
        elif v == "chi" and u.startswith("dchi["):
            if sorted(monos) != sorted(pat[("chi", u)]):
                raise ValueError("psi4backend: asymmetric mixed pattern")
        elif u == v and u.startswith("dchi["):
            field_diag[int(u[5])] = monos
        else:
            raise ValueError(f"psi4backend: unsupported pattern {(u, v)}")

    field_phiphi, field_mixed, field_diag, dots, hoists = \
        _compact_patterns(field_phiphi, field_mixed, field_diag)
    ind = "                    "
    defs, ops = _emit_intermediates(dots, hoists, PSI4_RV_FX_OPERANDS, ind)

    L: List[str] = []
    A = L.append
    A("                    // ==> BEGIN GENERATED CODE"
      " [xckernel psi4backend: geometric_fock(%s), restricted] <==" % family)
    A("                    // Reproduce with: python -m xckernel.psi4backend --fx")
    A("                    // Physics source: the geometric derivative of the")
    A("                    // symbolic tower (basis class, fixed grid); the")
    A("                    // intermediates are IR compaction output.")
    L.extend(defs)
    A("                    double c;")
    A("                    // field (phi, phi) pattern at quarter weight")
    A("                    c = 0.0;")
    L.extend(_cxx_sum(_transform_monomials(field_phiphi, ops), 0.25, ind))
    A("                    C_DAXPY(nlocal, c, phi[P], 1, T0p[P], 1);")
    for i in range(3):
        A(f"                    // field (dphi_{'xyz'[i]}, phi) + transpose at half weight")
        A("                    c = 0.0;")
        L.extend(_cxx_sum(_transform_monomials(field_mixed[i], ops), 0.5, ind))
        A(f"                    C_DAXPY(nlocal, c, phi_i[{i}][P], 1, T0p[P], 1);")
    for i in range(3):
        # Tip buffers and second-derivative collocation exist only for
        # ansatz >= 1: guard the whole statement group.
        monos_t = _transform_monomials(field_diag[i], ops)
        g = max(1, min(gg for gg, c, e in monos_t))
        A(f"                    // field (dphi_{'xyz'[i]}, dphi_{'xyz'[i]}) at quarter weight")
        A(f"                    if (ansatz >= {g}) {{")
        A("                        c = 0.0;")
        L.extend(_cxx_sum(monos_t, 0.25, ind + "    "))
        A(f"                        C_DAXPY(nlocal, c, phi_i[{i}][P], 1, Tip[{i}][P], 1);")
        A("                    }")
    A("                    // atom-restricted seed patterns at half weight;")
    A("                    // each masked factor carries the -d/dr sign")
    for (mask, right, monos) in sorted(seeds):
        monos_c, dd = contract_dots([(float(c), tuple(f)) for c, f in monos])
        sdefs, ops2 = _emit_intermediates(dd, {}, ops, ind)
        L.extend(sdefs)
        monos_t = _transform_monomials(monos_c, ops2)
        req = 0
        if mask.startswith("ddchi_gA"):
            req = 1
        if right.startswith("dchi["):
            req = max(req, 1)   # Tip buffers
        g = max(req, min(gg for gg, c, e in monos_t))
        A(f"                    // seed ({mask}, {right})")
        left = _FX_MASKED[mask]
        if right == "chi":
            daxpy = f"C_DAXPY(nfuncs, c, {left}, 1, &T0p[P][off], 1);"
        elif right.startswith("dchi["):
            i = int(right[5])
            daxpy = f"C_DAXPY(nfuncs, c, {left}, 1, &Tip[{i}][P][off], 1);"
        else:
            raise ValueError(f"psi4backend: seed right factor {right!r}")
        if g > 0:
            A(f"                    if (ansatz >= {g}) {{")
            A("                        c = 0.0;")
            L.extend(_cxx_sum(monos_t, -0.5, ind + "    "))
            A(f"                        {daxpy}")
            A("                    }")
        else:
            A("                    c = 0.0;")
            L.extend(_cxx_sum(monos_t, -0.5, ind))
            A(f"                    {daxpy}")
    A("                    // ==> END GENERATED CODE <==")
    return "\n".join(L)


# --- grid-motion class of the XC gradient -------------------------------------

PSI4_RV_GRIDMOTION_OPERANDS: Dict[str, Tuple[str, float]] = {
    "drho_g": ("drho_g", 1.0),
    "dgrad_rho_g_x": ("dgrad_g[0]", 1.0),
    "dgrad_rho_g_y": ("dgrad_g[1]", 1.0),
    "dgrad_rho_g_z": ("dgrad_g[2]", 1.0),
    "dtau_g": ("dtau_g", 1.0),
    "grad_rho_x": ("rho_g[0][P]", 1.0),
    "grad_rho_y": ("rho_g[1][P]", 1.0),
    "grad_rho_z": ("rho_g[2][P]", 1.0),
    "vrho": ("v_rho[P]", 1.0),
    "vsigma": ("v_gamma[P]", 1.0),
    "vtau": ("v_tau[P]", 2.0),
}


def emit_rv_gradient_gridmotion(family: str = "mgga_tau") -> str:
    """The generated scalar d_d e(r) of the gradient's grid-motion class:
    per (point, direction) after the plumbing computed drho_g, the
    density-Hessian row dgrad_g[3], and dtau_g."""
    from .fastpoly import from_expr
    from .geometric import spatial_energy_gradient
    expr = spatial_energy_gradient(family)
    monos = [(float(c), tuple(sorted((sym.name, e) for sym, e in key)))
             for key, c in from_expr(expr).items()]
    monos, dots = contract_dots(monos)
    ind = "                    "
    defs, ops = _emit_intermediates(dots, {}, PSI4_RV_GRIDMOTION_OPERANDS, ind)
    L: List[str] = []
    A = L.append
    A("                    // ==> BEGIN GENERATED CODE"
      " [xckernel psi4backend: spatial_energy_gradient(%s)] <==" % family)
    A("                    // Reproduce with: python -m xckernel.psi4backend --gridmotion")
    L.extend(defs)
    A("                    double de = 0.0;")
    L.extend(_cxx_sum(_transform_monomials(monos, ops), 1.0, ind, var="de"))
    A("                    // ==> END GENERATED CODE <==")
    return "\n".join(L)


# --- RV::compute_hessian: the explicit fixed-grid term ------------------------

#: operand map for the Hessian pair kernel, per-function context
#: (P = point, ml/nl = function, xd/yd = displacement direction, i =
#: Cartesian component). Factors fold the Psi4 conventions: U rows are
#: built from D_alpha (x2 for the total density), masked displacement
#: collocations carry -d/dr against raw PHI_X arrays (x-1), the pair
#: factor is D_alpha (x2), and tau-index libxc derivatives are stored
#: halved per tau index. Text None = right-side/pair factor folded into
#: the coefficient only (the host GEMM supplies the raw array).
PSI4_RV_HESS_OPERANDS = {
    "U0_u": ("U0[P][ml]", 2.0),
    "U1_u": ("Uip[i][P][ml]", 2.0),
    "dchi_gA_u": ("phi_i[xd][P][ml]", -1.0),
    "ddchi_gA_u_x": ("phi_hess[hess_addr[xd][i]][P][ml]", -1.0),
    "dchi_gB_v": (None, -1.0),
    "ddchi_gB_v_x": (None, -1.0),
    "D_u_v": (None, 2.0),
    "d2chi_g2_u": ("phi_hess[hess_addr[xd][yd]][P][ml]", 1.0),
    "d3chi_g2_u_x": ("phi_3[t3_addr[xd][yd][i]][P][ml]", 1.0),
    "grad_rho_x": ("rho_g[i][P]", 1.0),
    "vrho": ("v_rho_a[P]", 1.0),
    "vsigma": ("v_gamma[P]", 1.0),
    "vtau": ("v_tau[P]", 2.0),
    "G_i": ("g", 1.0),
}

#: second-derivative arrays in the class-I field x field contraction
PSI4_RV_HESS_FXX = {
    "v2rho2": ("v_rho_aa[P]", 1.0),
    "v2rhosigma": ("v2_rho_gamma[P]", 1.0),
    "v2sigma2": ("v2_gamma_gamma[P]", 1.0),
    "v2rhotau": ("v2_rho_tau[P]", 2.0),
    "v2sigmatau": ("v2_gamma_tau[P]", 2.0),
    "v2tau2": ("v2_tau_tau[P]", 4.0),
}


def _hx(expr, scale: float = 1.0, weight: str = "") -> str:
    """Transform a hint expression to a C sum with the Hessian operand
    map: per-monomial coefficient x mapped factors, optionally times the
    quadrature weight."""
    from .fastpoly import from_expr
    terms = []
    for key, coeff in sorted(from_expr(expr).items(),
                             key=lambda kv: str(kv[0])):
        c = float(coeff) * scale
        facs = []
        for sym, e in key:
            text, fac = PSI4_RV_HESS_OPERANDS[sym.name]
            c *= fac ** e
            if text is not None:
                facs.extend([text] * e)
        if weight:
            facs.insert(0, weight)
        body = " * ".join(facs)
        terms.append(f"{c:+.1f} * {body}" if body else f"{c:+.1f}")
    out = " ".join(terms)
    return out[1:] if out.startswith("+") else out


def emit_rv_hessian(family: str = "mgga_tau") -> dict:
    """Emit the four IR-dense regions of RV::compute_hessian's explicit
    fixed-grid term (the GEMM/scatter orchestration is host plumbing):
    the per-function field-derivative row fill, the class-I coefficient
    assembly, the class-II seed coefficients, and the class-III
    one-center body."""
    from collections import Counter

    from .deriv import libxc_symbol
    from .geometric import geometric_hessian
    gh = geometric_hessian(family)
    h = gh.hints
    mark = lambda what: (f"// ==> BEGIN GENERATED CODE [xckernel psi4backend: "
                         f"geometric_hessian({family}) {what}, restricted] <==")
    END = "// ==> END GENERATED CODE <=="
    regions = {}

    # R1: field-derivative row fill (bound in the host's xd/P/ml loops)
    ind = " " * 24
    r1 = [mark("rows")]
    r1.append(f"double frho = {_hx(h['F_rho'])};")
    r1.append("double fsig = 0.0, ftau = 0.0;")
    r1.append("for (int i = 0; i < 3; i++) {")
    r1.append(f"    double g = {_hx(h['G_i'])};")
    r1.append("    Gp[3 * xd + i][P][ml] = live ? g : 0.0;")
    r1.append(f"    fsig += {_hx(h['F_sigma_i'])};")
    r1.append(f"    if (is_meta) ftau += {_hx(h['F_tau_i'])};")
    r1.append("}")
    r1.append("F_rho[xd][P][ml] = live ? frho : 0.0;")
    r1.append("F_sig[xd][P][ml] = live ? fsig : 0.0;")
    r1.append("F_tau[xd][P][ml] = live ? ftau : 0.0;")
    r1.append(END)
    regions["rows"] = "\n".join(ind + line for line in r1)

    # R2: class I -- field x field through the second functional
    # derivatives, at half weight (rho-rho lives in the host LSDA block)
    names = ["rho", "sigma", "tau"]
    var = {"rho": "lr", "sigma": "ls", "tau": "lt"}
    row = {"rho": "F_rho[xd][P][ml]", "sigma": "F_sig[xd][P][ml]",
           "tau": "F_tau[xd][P][ml]"}
    r2 = [mark("class I")]
    r2.append("double wP = 0.5 * w[P];")
    for k in names:
        base, meta = [], []
        for l in names:
            if {k, l} == {"rho"}:
                continue
            text, fac = PSI4_RV_HESS_FXX[libxc_symbol(
                Counter({k: 1}) + Counter({l: 1})).name]
            term = (f"{fac:.1f} * {text} * {row[l]}" if fac != 1.0
                    else f"{text} * {row[l]}")
            (meta if "tau" in (k, l) else base).append(term)
        lhs = f"double {var[k]}"
        if base:
            r2.append(f"{lhs} = {' + '.join(base)};")
            for t in meta:
                r2.append(f"if (is_meta) {var[k]} += {t};")
        else:
            r2.append(f"{lhs} = is_meta ? {' + '.join(meta)} : 0.0;")
    r2.append("WL[P][ml] = wP * lr;")
    r2.append("WR[P][ml] = wP * ls;")
    r2.append("Tp[P][ml] = wP * lt;")
    r2.append(END)
    regions["classI"] = "\n".join(ind + line for line in r2)

    # R3: class II -- seed two-center coefficients. sigma is the left
    # member of a transpose pair (mirror from accumulate-plus-transpose,
    # full weight); tau is symmetric (half weight). All scalar factors
    # and both masked-collocation signs fold into the coefficient; the
    # host GEMMs against raw collocations and the raw D pair factor.
    r3s = [mark("class II sigma"),
           f"acc += {_hx(h['seed_pair_sigma_i'], weight='w[P]')};",
           END]
    regions["classII_sigma"] = "\n".join(" " * 32 + line for line in r3s)
    r3t = [mark("class II tau"),
           "WL[P][ml] = live ? "
           f"{_hx(h['seed_pair_tau_i'], scale=0.5, weight='w[P]')} : 0.0;",
           END]
    regions["classII_tau"] = "\n".join(" " * 32 + line for line in r3t)

    # R4: class III -- seed one-center body (both displacements on one
    # function; third-derivative collocation), at half weight
    r4 = [mark("class III")]
    r4.append("for (int i = 0; i < 3; i++) {")
    r4.append(f"    t += {_hx(h['seed_same_sigma_i'], scale=0.5, weight='w[P]')};")
    r4.append(f"    if (is_meta) t += {_hx(h['seed_same_tau_i'], scale=0.5, weight='w[P]')};")
    r4.append("}")
    r4.append(END)
    regions["classIII"] = "\n".join(" " * 28 + line for line in r4)
    return regions

# --- UV::compute_fock_derivatives ---------------------------------------------

def _uv_fx_operands() -> Dict[str, Tuple[str, float]]:
    """Operand map for the UV Fx coefficient region: the polarized Vx map
    with the perturbed fields rebound to the per-point plumbing locals
    (rho_ak, grad_ak[i], tau_ak, ...) and the gradient fields to the
    rho_ag/rho_bg pointer arrays of compute_fock_derivatives."""
    ops = dict(_uv_operands())
    for s_ in ("a", "b"):
        ops[f"rho_{s_}_p1"] = (f"rho_{s_}k", 1.0)
        ops[f"tau_{s_}_p1"] = (f"tau_{s_}k", 1.0)
        for i, ax in enumerate("xyz"):
            ops[f"grad_rho_{s_}_p1_{ax}"] = (f"grad_{s_}k[{i}]", 1.0)
            ops[f"grad_rho_{s_}_{ax}"] = (f"rho_{s_}g[{i}][P]", 1.0)
    return ops


def _fx_split(ck):
    """Split a collapsed geometric-Fock kernel into field patterns
    (phiphi, mixed{i}, diag{i}) and masked seed patterns, verifying the
    transpose-pair assumptions (shared by the RV and UV emitters)."""
    field_phiphi = []
    field_mixed: Dict[int, list] = {}
    field_diag: Dict[int, list] = {}
    seeds: List[Tuple[str, str, list]] = []
    pat = {(u, v): m for (u, v, m) in ck.patterns}
    for (u, v), monos in pat.items():
        u_masked = u in _FX_MASKED
        v_masked = v in _FX_MASKED
        if u_masked and not v_masked:
            seeds.append((u, v, monos))
        elif v_masked and not u_masked:
            if sorted(monos) != sorted(pat[(v, u)]):
                raise ValueError("psi4backend: asymmetric seed pattern")
        elif u_masked and v_masked:
            raise ValueError("psi4backend: doubly-masked pattern")
        elif u == "chi" and v == "chi":
            field_phiphi = monos
        elif u == "chi" and v.startswith("dchi["):
            field_mixed[int(v[5])] = monos
        elif v == "chi" and u.startswith("dchi["):
            if sorted(monos) != sorted(pat[("chi", u)]):
                raise ValueError("psi4backend: asymmetric mixed pattern")
        elif u == v and u.startswith("dchi["):
            field_diag[int(u[5])] = monos
        else:
            raise ValueError(f"psi4backend: unsupported pattern {(u, v)}")
    return field_phiphi, field_mixed, field_diag, seeds


def emit_uv_fx_contraction(family: str = "mgga_tau") -> str:
    """The generated coefficient region of UV::compute_fock_derivatives:
    both spin channels fused in one per-point block, with the RV weight
    rules (the accumulation visits every function pair from both sides:
    symmetric patterns at QUARTER weight, transpose pairs at HALF, seeds
    at -1/2 with the -d/dr sign folded into the emission factor)."""
    from .geometric import geometric_fock_spin
    base_ops = _uv_fx_operands()
    ind = "                    "

    chan = {}
    dots_all, hoists_all = {}, {}
    for s_ in ("a", "b"):
        ck = collapse(geometric_fock_spin(family, s_))
        phiphi, mixed, diag, seeds = _fx_split(ck)
        phiphi, mixed, diag, dots, hoists = _compact_patterns(phiphi, mixed, diag)
        ren = {name: f"{name}_{s_}" for name in hoists}
        hoists = {ren[k]: v for k, v in hoists.items()}
        for i in range(3):
            mixed[i] = [(c, tuple(sorted((ren.get(n, n), e) for n, e in f)))
                        for c, f in mixed[i]]
        chan[s_] = (phiphi, mixed, diag, seeds)
        dots_all.update(dots)
        hoists_all.update(hoists)

    defs, ops = _emit_intermediates(dots_all, hoists_all, base_ops, ind)

    L: List[str] = []
    A = L.append
    A(ind + "// ==> BEGIN GENERATED CODE"
      " [xckernel psi4backend: geometric_fock_spin(%s)] <==" % family)
    A(ind + "// Reproduce with: python -m xckernel.psi4backend --uvfx")
    A(ind + "// Physics source: the spin geometric derivative of the")
    A(ind + "// symbolic tower (basis class, fixed grid); the")
    A(ind + "// intermediates are IR compaction output.")
    L.extend(defs)
    A(ind + "double c;")
    for s_, T0, Ti in (("a", "T0ap", "Tiap"), ("b", "T0bp", "Tibp")):
        phiphi, mixed, diag, seeds = chan[s_]
        A(ind + f"// spin {s_}: field (phi, phi) pattern at quarter weight")
        A(ind + "c = 0.0;")
        L.extend(_cxx_sum(_transform_monomials(phiphi, ops), 0.25, ind))
        A(ind + f"C_DAXPY(nlocal, c, phi[P], 1, {T0}[P], 1);")
        for i in range(3):
            A(ind + f"// spin {s_}: field (dphi_{'xyz'[i]}, phi) + transpose at half weight")
            A(ind + "if (ansatz >= 1) {")
            A(ind + "    c = 0.0;")
            L.extend(_cxx_sum(_transform_monomials(mixed[i], ops), 0.5, ind + "    "))
            A(ind + f"    C_DAXPY(nlocal, c, phi_i[{i}][P], 1, {T0}[P], 1);")
            A(ind + "}")
        for i in range(3):
            monos_t = _transform_monomials(diag[i], ops)
            g = max(1, min(gg for gg, c, e in monos_t))
            A(ind + f"// spin {s_}: field (dphi_{'xyz'[i]}, dphi_{'xyz'[i]}) at quarter weight")
            A(ind + f"if (ansatz >= {g}) {{")
            A(ind + "    c = 0.0;")
            L.extend(_cxx_sum(monos_t, 0.25, ind + "    "))
            A(ind + f"    C_DAXPY(nlocal, c, phi_i[{i}][P], 1, {Ti}[{i}][P], 1);")
            A(ind + "}")
        A(ind + f"// spin {s_}: atom-restricted seed patterns at half weight;")
        A(ind + "// each masked factor carries the -d/dr sign")
        for (mask, right, monos) in sorted(seeds):
            monos_c, dd = contract_dots([(float(c), tuple(f)) for c, f in monos])
            sdefs, ops2 = _emit_intermediates(dd, {}, ops, ind)
            L.extend(sdefs)
            monos_t = _transform_monomials(monos_c, ops2)
            req = 1 if mask.startswith("ddchi_gA") else 0
            if right.startswith("dchi["):
                req = max(req, 1)
            g = max(req, min(gg for gg, c, e in monos_t))
            A(ind + f"// spin {s_}: seed ({mask}, {right})")
            left = _FX_MASKED[mask]
            if right == "chi":
                daxpy = f"C_DAXPY(nfuncs, c, {left}, 1, &{T0}[P][off], 1);"
            elif right.startswith("dchi["):
                i = int(right[5])
                daxpy = f"C_DAXPY(nfuncs, c, {left}, 1, &{Ti}[{i}][P][off], 1);"
            else:
                raise ValueError(f"psi4backend: seed right factor {right!r}")
            if g > 0:
                A(ind + f"if (ansatz >= {g}) {{")
                A(ind + "    c = 0.0;")
                L.extend(_cxx_sum(monos_t, -0.5, ind + "    "))
                A(ind + f"    {daxpy}")
                A(ind + "}")
            else:
                A(ind + "c = 0.0;")
                L.extend(_cxx_sum(monos_t, -0.5, ind))
                A(ind + f"    {daxpy}".replace("    C_", "C_"))
    A(ind + "// ==> END GENERATED CODE <==")
    return "\n".join(L)

# --- UV::compute_hessian: the explicit fixed-grid term ------------------------

#: operand map for the polarized Hessian pair kernel (per-function
#: context: P, ml/nl, xd/yd, component i). UKS is pure math throughout:
#: the U rows and D pair factors are the actual spin matrices (factor 1);
#: masked displacement collocations carry -d/dr; double-displacement
#: collocations carry +d2/dxdy; tau-index libxc arrays are stored halved.
def _uv_hess_operands() -> Dict[str, Tuple[str, float]]:
    ops = dict(_uv_operands())
    ops["w"] = (None, 1.0)      # the emitter places w[P] itself
    for s_ in ("a", "b"):
        ops[f"U0{s_}_u"] = (f"U0{s_}[P][ml]", 1.0)
        ops[f"U1{s_}_u"] = (f"Ui{s_}[i][P][ml]", 1.0)
        ops[f"G_{s_}_i"] = (f"g_{s_}", 1.0)
        # hints carry i = 0 instances: the x components bind to the
        # host's Cartesian loop index i
        ops[f"grad_rho_{s_}_x"] = (f"rho_{s_}g[i][P]", 1.0)
        del ops[f"grad_rho_{s_}_y"], ops[f"grad_rho_{s_}_z"]
        ops[f"D_{s_}_u_v"] = (None, 1.0)
    ops["dchi_gA_u"] = ("phi_i[xd][P][ml]", -1.0)
    ops["ddchi_gA_u_x"] = ("phi_hess[hess_addr[xd][i]][P][ml]", -1.0)
    ops["dchi_gB_v"] = (None, -1.0)
    ops["ddchi_gB_v_x"] = (None, -1.0)
    ops["d2chi_g2_u"] = ("phi_hess[hess_addr[xd][yd]][P][ml]", 1.0)
    ops["d3chi_g2_u_x"] = ("phi_3[t3_addr[xd][yd][i]][P][ml]", 1.0)
    return ops


def _hx_uv(expr, ops, scale: float = 1.0, weight: str = "w[P]") -> str:
    """Transform an expression to a C sum with a Hessian operand map."""
    from .fastpoly import from_expr
    terms = []
    for key, coeff in sorted(from_expr(sp_expand(expr)).items(),
                             key=lambda kv: str(kv[0])):
        c = float(coeff) * scale
        facs = []
        for sym, e in key:
            text, fac = ops[sym.name]
            c *= fac ** e
            if text is not None:
                facs.extend([text] * e)
        if weight:
            facs.insert(0, weight)
        body = " * ".join(facs)
        terms.append(f"{c:+.1f} * {body}" if body else f"{c:+.1f}")
    out = " ".join(terms)
    return out[1:] if out.startswith("+") else out


def sp_expand(e):
    import sympy
    return sympy.expand(e)


#: per-scalar row array names and ansatz guards for the UV Hessian
_UV_ROW = {("rho", "a"): ("F_ra", 0), ("rho", "b"): ("F_rb", 0),
           ("sigma", "aa"): ("F_saa", 1), ("sigma", "ab"): ("F_sab", 1),
           ("sigma", "bb"): ("F_sbb", 1),
           ("tau", "a"): ("F_ta", 2), ("tau", "b"): ("F_tb", 2)}


def emit_uv_hessian(family: str = "mgga_tau") -> dict:
    """Emit the generated regions of UV::compute_hessian's explicit
    fixed-grid GGA/meta term. UV convention: the host ends with
    hermitivitize() alone (no scale(2)), so symmetric classes enter at
    FULL weight and transpose-pair members TWICE."""
    from .geometric import geometric_hessian_spin
    from .spin_kernel import _register
    gh = geometric_hessian_spin(family)
    h = gh.hints
    scalars = h["scalars"]
    ops = _uv_hess_operands()
    mark = lambda what: (f"// ==> BEGIN GENERATED CODE [xckernel psi4backend: "
                         f"geometric_hessian_spin({family}) {what}] <==")
    END = "// ==> END GENERATED CODE <=="
    regions = {}

    # R1: field-derivative row fill (bound in the host's xd/P/ml loops)
    ind = " " * 24
    r1 = [mark("rows")]
    r1.append(f"double f_ra = {_hx_uv(h['F_i0'][scalars[0]], ops, weight='')};")
    r1.append(f"double f_rb = {_hx_uv(h['F_i0'][scalars[1]], ops, weight='')};")
    r1.append("double f_saa = 0.0, f_sab = 0.0, f_sbb = 0.0;")
    r1.append("double f_ta = 0.0, f_tb = 0.0;")
    r1.append("for (int i = 0; i < 3; i++) {")
    r1.append(f"    double g_a = {_hx_uv(h['G_i']['a'], ops, weight='')};")
    r1.append(f"    double g_b = {_hx_uv(h['G_i']['b'], ops, weight='')};")
    r1.append("    Ga[3 * xd + i][P][ml] = live ? g_a : 0.0;")
    r1.append("    Gb[3 * xd + i][P][ml] = live ? g_b : 0.0;")
    by_kc = {(K.group, K.comp): K for K in scalars}
    for comp in ("aa", "ab", "bb"):
        K = by_kc.get(("sigma", comp))
        if K is not None:
            r1.append(f"    f_s{comp} += {_hx_uv(h['F_i0'][K], ops, weight='')};")
    for comp in ("a", "b"):
        K = by_kc.get(("tau", comp))
        if K is not None:
            r1.append(f"    if (is_meta) f_t{comp} += "
                      f"{_hx_uv(h['F_i0'][K], ops, weight='')};")
    r1.append("}")
    for K in scalars:
        name, _g = _UV_ROW[(K.group, K.comp)]
        local = {"rho": {"a": "f_ra", "b": "f_rb"},
                 "sigma": {"aa": "f_saa", "ab": "f_sab", "bb": "f_sbb"},
                 "tau": {"a": "f_ta", "b": "f_tb"}}[K.group][K.comp]
        r1.append(f"{name}[xd][P][ml] = live ? {local} : 0.0;")
    regions["rows"] = "\n".join(ind + line for line in r1)

    # R2: class I (field x field, full weight, rho-rho pairs excluded --
    # the LSDA block owns them) + the vsigma gradient cross
    ind = " " * 16
    r2 = [mark("class I")]
    for L in scalars:
        Lname, Lg = _UV_ROW[(L.group, L.comp)]
        base, guarded = [], {}
        for K in scalars:
            if K.group == "rho" and L.group == "rho":
                continue
            fname = _register((K, L)).name
            text, fac = ops[fname]
            Kname, Kg = _UV_ROW[(K.group, K.comp)]
            term = (f"{fac:.1f} * {text} * {Kname}[xd][P][ml]" if fac != 1.0
                    else f"{text} * {Kname}[xd][P][ml]")
            g = Kg
            if g <= Lg:
                base.append(term)
            else:
                guarded.setdefault(g, []).append(term)
        if not base and not guarded:
            continue
        body = ["for (int P = 0; P < npoints; P++) {",
                "    for (int ml = 0; ml < nlocal; ml++) {",
                f"        double l = {' + '.join(base) if base else '0.0'};"]
        for g in sorted(guarded):
            body.append(f"        if (ansatz >= {g}) l += "
                        f"{' + '.join(guarded[g])};")
        body += ["        WL[P][ml] = w[P] * l;",
                 "    }",
                 "}",
                 "for (int yd = 0; yd < 3; yd++) {",
                 f"    C_DGEMM('T', 'N', nlocal, nlocal, npoints, 1.0, WL[0], max_functions, {Lname}[yd][0],",
                 "            max_functions, 1.0, pH[xd][yd][0], max_functions);",
                 "}"]
        r2.append(f"// class I, right row {Lname}")
        if Lg > 0:
            r2.append(f"if (ansatz >= {Lg}) {{")
            r2.extend("    " + b for b in body)
            r2.append("}")
        else:
            r2.extend(body)
    # vsigma gradient cross: left factors per right G channel
    r2.append("// vsigma gradient cross, per right G channel")
    r2.append("if (ansatz >= 1) {")
    r2.append("    for (int i = 0; i < 3; i++) {")
    r2.append("        for (int P = 0; P < npoints; P++) {")
    r2.append("            for (int ml = 0; ml < nlocal; ml++) {")
    gops = dict(ops)
    gops["G_a_i"] = ("Ga[3 * xd + i][P][ml]", 1.0)
    gops["G_b_i"] = ("Gb[3 * xd + i][P][ml]", 1.0)
    r2.append(f"                WL[P][ml] = {_hx_uv(h['classIp_left']['a'], gops)};")
    r2.append(f"                WR[P][ml] = {_hx_uv(h['classIp_left']['b'], gops)};")
    r2.append("            }")
    r2.append("        }")
    r2.append("        for (int yd = 0; yd < 3; yd++) {")
    r2.append("            C_DGEMM('T', 'N', nlocal, nlocal, npoints, 1.0, WL[0], max_functions, Ga[3 * yd + i][0],")
    r2.append("                    max_functions, 1.0, pH[xd][yd][0], max_functions);")
    r2.append("            C_DGEMM('T', 'N', nlocal, nlocal, npoints, 1.0, WR[0], max_functions, Gb[3 * yd + i][0],")
    r2.append("                    max_functions, 1.0, pH[xd][yd][0], max_functions);")
    r2.append("        }")
    r2.append("    }")
    r2.append("}")
    r2.append(END)
    regions["classI"] = "\n".join(ind + line for line in r2)

    # R3: class II -- two-center seeds. sigma members enter TWICE
    # (transpose pair under hermitivitize-only); tau symmetric at full.
    r3 = [mark("class II")]
    r3.append("if (ansatz >= 1) {")
    r3.append("    for (int P = 0; P < npoints; P++) {")
    r3.append("        bool live = std::fabs(rho_a[P]) + std::fabs(rho_b[P]) > v2_rho_cutoff_;")
    r3.append("        for (int ml = 0; ml < nlocal; ml++) {")
    r3.append("            double acc_a = 0.0, acc_b = 0.0;")
    r3.append("            if (live) {")
    r3.append("                for (int i = 0; i < 3; i++) {")
    r3.append(f"                    acc_a += {_hx_uv(h['seed_pair_sigma_i']['a'], ops, scale=2.0)};")
    r3.append(f"                    acc_b += {_hx_uv(h['seed_pair_sigma_i']['b'], ops, scale=2.0)};")
    r3.append("                }")
    r3.append("            }")
    r3.append("            WL[P][ml] = acc_a;")
    r3.append("            WR[P][ml] = acc_b;")
    r3.append("        }")
    r3.append("    }")
    r3.append("    for (int yd = 0; yd < 3; yd++) {")
    r3.append("        C_DGEMM('T', 'N', nlocal, nlocal, npoints, 1.0, WL[0], max_functions, phi_i[yd][0], coll_funcs,")
    r3.append("                0.0, WT[0], max_functions);")
    r3.append("        for (int ml = 0; ml < nlocal; ml++)")
    r3.append("            for (int nl = 0; nl < nlocal; nl++) pH[xd][yd][ml][nl] += WT[ml][nl] * Dap[ml][nl];")
    r3.append("        C_DGEMM('T', 'N', nlocal, nlocal, npoints, 1.0, WR[0], max_functions, phi_i[yd][0], coll_funcs,")
    r3.append("                0.0, WT[0], max_functions);")
    r3.append("        for (int ml = 0; ml < nlocal; ml++)")
    r3.append("            for (int nl = 0; nl < nlocal; nl++) pH[xd][yd][ml][nl] += WT[ml][nl] * Dbp[ml][nl];")
    r3.append("    }")
    r3.append("}")
    r3.append("// tau two-center seed (symmetric, full weight)")
    r3.append("if (is_meta) {")
    r3.append("    for (int i = 0; i < 3; i++) {")
    r3.append("        for (int P = 0; P < npoints; P++) {")
    r3.append("            bool live = std::fabs(rho_a[P]) + std::fabs(rho_b[P]) > v2_rho_cutoff_;")
    r3.append("            for (int ml = 0; ml < nlocal; ml++) {")
    r3.append(f"                WL[P][ml] = live ? {_hx_uv(h['seed_pair_tau_i']['a'], ops)} : 0.0;")
    r3.append(f"                WR[P][ml] = live ? {_hx_uv(h['seed_pair_tau_i']['b'], ops)} : 0.0;")
    r3.append("            }")
    r3.append("        }")
    r3.append("        for (int yd = 0; yd < 3; yd++) {")
    r3.append("            C_DGEMM('T', 'N', nlocal, nlocal, npoints, 1.0, WL[0], max_functions,")
    r3.append("                    phi_hess[hess_addr[yd][i]][0], coll_funcs, 0.0, WT[0], max_functions);")
    r3.append("            for (int ml = 0; ml < nlocal; ml++)")
    r3.append("                for (int nl = 0; nl < nlocal; nl++) pH[xd][yd][ml][nl] += WT[ml][nl] * Dap[ml][nl];")
    r3.append("            C_DGEMM('T', 'N', nlocal, nlocal, npoints, 1.0, WR[0], max_functions,")
    r3.append("                    phi_hess[hess_addr[yd][i]][0], coll_funcs, 0.0, WT[0], max_functions);")
    r3.append("            for (int ml = 0; ml < nlocal; ml++)")
    r3.append("                for (int nl = 0; nl < nlocal; nl++) pH[xd][yd][ml][nl] += WT[ml][nl] * Dbp[ml][nl];")
    r3.append("        }")
    r3.append("    }")
    r3.append("}")
    r3.append(END)
    regions["classII"] = "\n".join(ind + line for line in r3)

    # R4: class III -- one-center seeds at full weight, scattered to both
    # (xd, yd) and (yd, xd) of the (A, A) block (hermitivitize averages)
    ind4 = " " * 28
    r4 = [mark("class III")]
    r4.append("for (int i = 0; i < 3; i++) {")
    r4.append(f"    t += {_hx_uv(h['seed_same_sigma_i']['a'], ops)};")
    r4.append(f"    t += {_hx_uv(h['seed_same_sigma_i']['b'], ops)};")
    if h["seed_same_tau_i"]:
        r4.append(f"    if (is_meta) t += {_hx_uv(h['seed_same_tau_i']['a'], ops)}"
                  f" + {_hx_uv(h['seed_same_tau_i']['b'], ops)};")
    r4.append("}")
    r4.append(END)
    regions["classIII"] = "\n".join(ind4 + line for line in r4)
    return regions



if __name__ == "__main__":
    import sys
    if "--uv" in sys.argv:
        print(emit_uv_vx_contraction())
    elif "--fx" in sys.argv:
        print(emit_rv_fx_contraction())
    elif "--gridmotion" in sys.argv:
        print(emit_rv_gradient_gridmotion())
    elif "--uvfx" in sys.argv:
        print(emit_uv_fx_contraction())
    elif "--uvhessian" in sys.argv:
        for _name, _region in emit_uv_hessian().items():
            print(f"### {_name}")
            print(_region)
    elif "--hessian" in sys.argv:
        for _name, _region in emit_rv_hessian().items():
            print(f"### {_name}")
            print(_region)
    else:
        print(emit_rv_vx_contraction())