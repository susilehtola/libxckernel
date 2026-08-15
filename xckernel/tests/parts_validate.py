"""Validate the split-storage (separate Re/Im) complex emissions.

Split storage is the recommended layout for complex quantities: the host
passes separate real and imaginary collocation arrays, and the emitted
kernel accumulates the requested part of the output matrix in real
arithmetic.  Each basis-pair pattern then costs two real matrix products,
against the four real products' worth of work inside one complex product --
a 50% saving whenever only one part is physically needed, and a structural
zero-elimination when an operand is known purely real or imaginary.

Checks:

  1. part='re'/'im'/'both' against the interleaved-complex sesquilinear
     path, Fock and o2 response, plain and batch: machine precision
  2. zero-part elimination: a REAL basis declared via input_parts drops the
     imaginary arrays from the signature, empties the im-part function,
     and reduces the re part to the plain real emission
  3. the shared product-decomposition table (codegen.part_product) against
     explicit complex arithmetic, and the VeloxChem writer's helper mapping
     and emitted branches against the same table
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from ..emitters.codegen import (compile_function, generate_collapsed,
                                part_product)
from ..engine.deriv import LIBXC_MULTISET
from ..engine.kernel import fock
from ..engine.response import response_fock
from ..inputs.fields import collocate

# --- the explicit test functional (same as complex_validate) ----------------

_r, _s, _t = sp.symbols("rho sigma tau", positive=True)
_F = (_r**2 + sp.Rational(3, 10) * _s * _r + sp.Rational(1, 5) * _t**2
      + sp.Rational(1, 10) * _r * _t + sp.Rational(1, 20) * _s * _t)
_VAR = {"rho": _r, "sigma": _s, "tau": _t}


def _deriv_arrays(names, rho, sigma, tau):
    out = {}
    for name in names:
        e = _F
        for var, cnt in LIBXC_MULTISET[name].items():
            e = sp.diff(e, _VAR[var], cnt)
        out[name] = sp.lambdify((_r, _s, _t), e, "numpy")(
            rho, sigma, tau) * np.ones_like(rho)
    return out


def _call(fn, gen, args):
    sig = gen.source.split("(", 1)[1].split(")", 1)[0]
    return fn(*[args[p.strip()] for p in sig.split(",")])


def main():
    rng = np.random.default_rng(37)
    nbf, ng = 5, 24
    chir = rng.uniform(0.5, 1.5, (nbf, ng))
    dchir = rng.standard_normal((3, nbf, ng))
    w = rng.uniform(0.1, 1.0, ng)
    chix = chir + 0.3j * rng.standard_normal((nbf, ng))
    dchix = dchir + 0.3j * rng.standard_normal((3, nbf, ng))

    C = (np.eye(nbf) + 0.25 * rng.standard_normal((nbf, nbf))
         + 0.25j * rng.standard_normal((nbf, nbf)))[:, :3]
    P = C.conj() @ C.T

    f = collocate(P, chix, dchix)
    rho, grad = f["rho"].real, f["grad_rho"].real
    sig, tau = f["sigma"].real, f["tau"].real

    tested = failures = 0

    def check(label, err, tol):
        nonlocal tested, failures
        tested += 1
        ok = err < tol
        if not ok:
            failures += 1
        print(f"  [{'OK' if ok else 'FAIL'}] {label}: err {err:.2e}")

    # 1. split emissions vs the complex sesquilinear path ---------------------
    ki = fock("mgga_tau")
    genC = generate_collapsed(ki, "fc", sesquilinear=True)
    lx = _deriv_arrays(genC.libxc_args, rho, sig, tau)
    argsC = {"w": w, "chi": chix, "chi_c": chix.conj(),
             "dchi": dchix, "dchi_c": dchix.conj(), "grad_rho": grad, **lx}
    FC = _call(compile_function(genC), genC, argsC)

    argsS = {"w": w, "chi_re": chix.real, "chi_im": chix.imag,
             "dchi_re": dchix.real, "dchi_im": dchix.imag,
             "grad_rho": grad, **lx}
    genR = generate_collapsed(ki, "fr", sesquilinear=True, part="re")
    genI = generate_collapsed(ki, "fi", sesquilinear=True, part="im")
    genB = generate_collapsed(ki, "fb", sesquilinear=True, part="both")
    FR = _call(compile_function(genR), genR, argsS)
    FI = _call(compile_function(genI), genI, argsS)
    FBr, FBi = _call(compile_function(genB), genB, argsS)
    scale = float(np.abs(FC).max())
    check("Fock re part vs complex", float(np.abs(FC.real - FR).max()) / scale,
          1e-14)
    check("Fock im part vs complex", float(np.abs(FC.imag - FI).max()) / scale,
          1e-14)
    check("Fock both parts vs complex",
          max(float(np.abs(FC.real - FBr).max()),
              float(np.abs(FC.imag - FBi).max())) / scale, 1e-14)
    for g in (genR, genI, genB):
        assert "dtype=complex" not in g.source and "_c" not in g.source, \
            "split emission must be real-arithmetic-only"

    # o2 response, plain and batch -------------------------------------------
    ki2 = response_fock("mgga_tau", 2)
    gen2C = generate_collapsed(ki2, "kc2", sesquilinear=True)
    lx2 = _deriv_arrays(gen2C.libxc_args, rho, sig, tau)
    dP = rng.standard_normal((nbf, nbf)) + 1j * rng.standard_normal((nbf, nbf))
    dP = 0.5 * (dP + dP.conj().T)
    pf = collocate(dP, chix, dchix)
    pert = {"rho_p1": pf["rho"].real, "grad_rho_p1": pf["grad_rho"].real,
            "tau_p1": pf["tau"].real}
    F2C = _call(compile_function(gen2C), gen2C, {**argsC, **pert, **lx2})
    gen2B = generate_collapsed(ki2, "kb2", sesquilinear=True, part="both")
    F2Br, F2Bi = _call(compile_function(gen2B), gen2B,
                       {**argsS, **pert, **lx2})
    scale2 = float(np.abs(F2C).max())
    check("o2 response both parts vs complex",
          max(float(np.abs(F2C.real - F2Br).max()),
              float(np.abs(F2C.imag - F2Bi).max())) / scale2, 1e-14)

    nx = 2
    pertx = {"rho_p1": np.stack([pert["rho_p1"]] * nx),
             "grad_rho_p1": np.stack([pert["grad_rho_p1"]] * nx),
             "tau_p1": np.stack([pert["tau_p1"]] * nx)}
    pertx["rho_p1"][1] *= 0.5
    pertx["grad_rho_p1"][1] *= 0.5
    pertx["tau_p1"][1] *= 0.5
    gen2Cx = generate_collapsed(ki2, "kcx", batch=True, sesquilinear=True)
    F2Cx = _call(compile_function(gen2Cx), gen2Cx, {**argsC, **pertx, **lx2})
    gen2Ix = generate_collapsed(ki2, "kix", batch=True, sesquilinear=True,
                                part="im")
    F2Ix = _call(compile_function(gen2Ix), gen2Ix, {**argsS, **pertx, **lx2})
    check("batch o2 im part vs complex",
          float(np.abs(F2Cx.imag - F2Ix).max()) / scale2, 1e-14)

    # 2. zero-part elimination for a real basis -------------------------------
    real_parts = {"chi": "re", "dchi": "re"}
    genR0 = generate_collapsed(ki, "fr0", sesquilinear=True, part="re",
                               input_parts=real_parts)
    genI0 = generate_collapsed(ki, "fi0", sesquilinear=True, part="im",
                               input_parts=real_parts)
    assert "chi_im" not in genR0.source and "dchi_im" not in genR0.source, \
        "imaginary arrays must be eliminated for a real basis"
    assert "@" not in genI0.source, \
        "im part of a real-basis Fock must contain no matrix products"
    fr0 = collocate(np.real(P), chir, dchir)
    lx0 = _deriv_arrays(genC.libxc_args, fr0["rho"].real, fr0["sigma"].real,
                        fr0["tau"].real)
    args0 = {"w": w, "chi_re": chir, "dchi_re": dchir,
             "grad_rho": fr0["grad_rho"].real, **lx0}
    F0 = _call(compile_function(genR0), genR0, args0)
    Z0 = _call(compile_function(genI0), genI0, args0)
    genP = generate_collapsed(ki, "fp")
    FP = _call(compile_function(genP), genP,
               {"w": w, "chi": chir, "dchi": dchir,
                "grad_rho": fr0["grad_rho"].real, **lx0})
    check("real-basis re part vs plain real emission",
          float(np.abs(F0 - FP).max()) / float(np.abs(FP).max()), 1e-14)
    check("real-basis im part identically zero",
          float(np.abs(Z0).max()), 0.5)
    assert Z0.shape == F0.shape

    # 3. the decomposition table and the VeloxChem writer ---------------------
    a = rng.standard_normal(8) + 1j * rng.standard_normal(8)
    b = rng.standard_normal(8) + 1j * rng.standard_normal(8)
    pieces = {"re": np.real, "im": np.imag}
    err = 0.0
    for conj_l, prod in ((False, a * b), (True, a.conj() * b)):
        for p in ("re", "im"):
            got = sum(s * pieces[lp](a) * pieces[rp](b)
                      for s, lp, rp in part_product(p, conj_l))
            err = max(err, float(np.abs(pieces[p](prod) - got).max()))
    check("part_product table vs complex arithmetic", err, 1e-15)

    from ..emitters.vlxwriter import emit_branch, prod_expansion
    check("vlx prod2_r realizes the table row",
          0.0 if prod_expansion("r") == "a_r*b_r - a_i*b_i" else 1.0, 0.5)
    check("vlx prod2_i realizes the table row",
          0.0 if prod_expansion("i") == "a_r*b_i + a_i*b_r" else 1.0, 0.5)
    branch = emit_branch("QRF", "gga")
    expected = (
        "gam0_r[i] += prod2_r(rhoB_r[i],rhoB_i[i],rhoC_r[i],rhoC_i[i])",
        "gam0_i[i] += prod2_i(rhoB_r[i],rhoB_i[i],rhoC_r[i],rhoC_i[i])",
        "gam0_x_r[i] += 2.0 * prod2_r(gradB_x_r[i],gradB_x_i[i],"
        "rhoC_r[i],rhoC_i[i])",
        "gam0_xy_i[i] += prod2_i(gradB_x_r[i],gradB_x_i[i],"
        "gradC_y_r[i],gradC_y_i[i])",
    )
    miss = sum(1 for e in expected if e not in branch)
    check("vlx emitted branch lines unchanged", float(miss), 0.5)

    status = "OK " if failures == 0 else "FAIL"
    print(f"[{status}] parts_validate: {tested} checks, {failures} failures")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
