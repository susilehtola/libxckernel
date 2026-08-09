"""Validate the two-sided (matrix-free / sigma-vector) emission mode.

With ``two_sided=True``, the free pair of a generated kernel contracts
independent left/right collocation arrays. Seeding the sides with occupied
and virtual molecular-orbital values on the grid yields the sigma-vector
block directly, with no (nbf, nbf) matrix ever materialized. Checks:

  1. Fock kernel: two-sided emission with MO values == C_occ^T F_AO C_virt
  2. collocate_pair == collocate of the corresponding AO trial matrix
  3. o2 response: two-sided emission with pair-collocated perturbed fields
     == MO block of the AO response Fock matrix

All comparisons are at machine precision: the two routes evaluate the same
collapsed expressions in different contraction orders.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from ..codegen import compile_function, generate_collapsed
from ..deriv import LIBXC_MULTISET
from ..fields import collocate, collocate_pair
from ..kernel import fock
from ..response import response_fock

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
    rng = np.random.default_rng(23)
    nbf, ng, no, nv = 6, 30, 2, 3
    chi = rng.uniform(0.5, 1.5, (nbf, ng))
    dchi = rng.standard_normal((3, nbf, ng))
    w = rng.uniform(0.1, 1.0, ng)
    P = 0.1 * rng.standard_normal((nbf, nbf))
    P = P + P.T + np.eye(nbf)
    Co = rng.standard_normal((nbf, no))
    Cv = rng.standard_normal((nbf, nv))

    psi_o = np.einsum("ui,ug->ig", Co, chi)
    dpsi_o = np.einsum("ui,cug->cig", Co, dchi)
    psi_v = np.einsum("ua,ug->ag", Cv, chi)
    dpsi_v = np.einsum("ua,cug->cag", Cv, dchi)

    f = collocate(P, chi, dchi)
    rho, grad, sigma, tau = (f["rho"], f["grad_rho"], f["sigma"], f["tau"])

    tested = failures = 0

    def check(label, err, tol):
        nonlocal tested, failures
        tested += 1
        ok = err < tol
        if not ok:
            failures += 1
        print(f"  [{'OK' if ok else 'FAIL'}] {label}: err {err:.2e}")

    # 1. Fock kernel, AO route vs two-sided MO route -------------------------
    ki = fock("mgga_tau")
    gen = generate_collapsed(ki, "aofock")
    fn = compile_function(gen)
    gen2s = generate_collapsed(ki, "mofock", two_sided=True)
    fn2s = compile_function(gen2s)
    lx = _deriv_arrays(gen.libxc_args, rho, sigma, tau)
    F_ao = _call(fn, gen, {"w": w, "chi": chi, "dchi": dchi,
                           "grad_rho": grad, **lx})
    ref = Co.T @ F_ao @ Cv
    F_mo = _call(fn2s, gen2s, {"w": w, "chi_l": psi_o, "chi_r": psi_v,
                               "dchi_l": dpsi_o, "dchi_r": dpsi_v,
                               "grad_rho": grad, **lx})
    assert F_mo.shape == (no, nv)
    check("two-sided Fock == MO block of AO Fock",
          float(np.abs(F_mo - ref).max() / np.abs(ref).max()), 1e-13)

    # 2. collocate_pair == collocate of the AO trial matrix ------------------
    X = rng.standard_normal((no, nv))
    D = Co @ X @ Cv.T
    fd_ao = collocate(D, chi, dchi)
    fd_mo = collocate_pair(X, psi_o, dpsi_o, psi_v, dpsi_v)
    err = max(float(np.abs(fd_ao[k] - fd_mo[k]).max())
              for k in ("rho", "grad_rho", "tau", "jp"))
    check("collocate_pair == AO collocate", err, 1e-12)

    # 3. o2 response, AO route vs two-sided route ----------------------------
    ki2 = response_fock("mgga_tau", 2)
    genr = generate_collapsed(ki2, "aok2")
    fnr = compile_function(genr)
    genr2s = generate_collapsed(ki2, "mok2", two_sided=True)
    fnr2s = compile_function(genr2s)
    lx2 = _deriv_arrays(genr.libxc_args, rho, sigma, tau)
    pert = {"rho_p1": fd_mo["rho"], "grad_rho_p1": fd_mo["grad_rho"],
            "tau_p1": fd_mo["tau"]}
    K_ao = _call(fnr, genr, {"w": w, "chi": chi, "dchi": dchi,
                             "grad_rho": grad, **pert, **lx2})
    ref = Co.T @ K_ao @ Cv
    K_mo = _call(fnr2s, genr2s, {"w": w, "chi_l": psi_o, "chi_r": psi_v,
                                 "dchi_l": dpsi_o, "dchi_r": dpsi_v,
                                 "grad_rho": grad, **pert, **lx2})
    check("two-sided o2 sigma == MO block of AO response",
          float(np.abs(K_mo - ref).max() / np.abs(ref).max()), 1e-13)

    status = "OK " if failures == 0 else "FAIL"
    print(f"[{status}] twosided_validate: {tested} checks, {failures} "
          "failures")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
