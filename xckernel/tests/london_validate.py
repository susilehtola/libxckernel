"""Validate the explicit GIAO/London magnetic-field derivative kernels.

The reference is the sesquilinear (complex-basis) Fock emission evaluated
with numerically phased collocations chi^B = exp[-(i/2c) (B x R_a) . r] chi
at a fixed, real, symmetric density matrix: the finite difference dF/dB_s
must equal (i/2c) K^s with K^s from london_fock. The ingredient-field
derivatives vanish at a real reference, so the comparison is exact up to
finite-difference error.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from ..codegen import compile_function, generate_collapsed
from ..deriv import LIBXC_MULTISET
from ..kernel import fock
from ..london import london_fock

C_AU = 137.035999084

# explicit test functional f(rho, sigma, lapl, tau) with all couplings
_r, _s, _l, _t = sp.symbols("rho sigma lapl tau", positive=True)
_F = (_r**2 + sp.Rational(3, 10) * _s * _r + sp.Rational(1, 5) * _t**2
      + sp.Rational(1, 10) * _r * _t + sp.Rational(1, 20) * _s * _t
      + sp.Rational(1, 8) * _l**2 + sp.Rational(1, 12) * _l * _r)
_VAR = {"rho": _r, "sigma": _s, "lapl": _l, "tau": _t}
_FAM_VARS = {"lda": ["rho"], "gga": ["rho", "sigma"],
             "mgga_tau": ["rho", "sigma", "tau"],
             "mgga": ["rho", "sigma", "lapl", "tau"]}


def _deriv_arrays(names, vals):
    out = {}
    for name in names:
        ms = LIBXC_MULTISET[name]
        e = _F
        for var, cnt in ms.items():
            e = sp.diff(e, _VAR[var], cnt)
        out[name] = sp.lambdify((_r, _s, _l, _t), e, "numpy")(
            *vals) * np.ones_like(vals[0])
    return out


def _phased(B, R, rg, chi, dchi, lapl_chi):
    """London-phased collocations at field B (3,)."""
    BxR = np.cross(B, R)                       # (nbf, 3)
    theta = -(1.0 / (2 * C_AU)) * np.einsum("ua,ag->ug", BxR, rg)
    ph = np.exp(1j * theta)                    # (nbf, ng)
    gtheta = -(1.0 / (2 * C_AU)) * BxR         # (nbf, 3), constant per fn
    chiB = ph * chi
    dchiB = np.array([ph * (dchi[c] + 1j * gtheta[:, c][:, None] * chi)
                      for c in range(3)])
    laplB = ph * (lapl_chi
                  + 2j * np.einsum("uc,cug->ug", gtheta, dchi)
                  - (gtheta**2).sum(1)[:, None] * chi)
    return chiB, dchiB, laplB


def _fields(P, chiB, dchiB, laplB):
    rho = np.einsum("uv,ug,vg->g", P, chiB.conj(), chiB).real
    grad = (np.einsum("uv,cug,vg->cg", P, dchiB.conj(), chiB)
            + np.einsum("uv,ug,cvg->cg", P, chiB, dchiB.conj()).conj()).real
    # grad rho = sum P [ (d chi)* chi + chi* d chi ] = 2 Re sum P (dchi)* chi
    grad = 2 * np.einsum("uv,cug,vg->cg", P, dchiB.conj(), chiB).real
    sigma = np.einsum("cg,cg->g", grad, grad)
    tau = 0.5 * np.einsum("uv,cug,cvg->g", P, dchiB.conj(), dchiB).real
    lapl = (2 * np.einsum("uv,ug,vg->g", P, laplB.conj(), chiB).real
            + 2 * np.einsum("uv,cug,cvg->g", P, dchiB.conj(), dchiB).real)
    return rho, grad, sigma, tau, lapl


def _sesqui_fock(fam, gen, fn, P, B, R, rg, chi, dchi, lapl_chi, w):
    chiB, dchiB, laplB = _phased(B, R, rg, chi, dchi, lapl_chi)
    rho, grad, sigma, tau, lapl = _fields(P, chiB, dchiB, laplB)
    lx = _deriv_arrays(gen.libxc_args, (rho, sigma, lapl, tau))
    args = {"w": w, "chi": chiB, "chi_c": chiB.conj(),
            "dchi": dchiB, "dchi_c": dchiB.conj(),
            "lapl_chi": laplB, "lapl_chi_c": laplB.conj(),
            "grad_rho": grad, **lx}
    sig = gen.source.split("(", 1)[1].split(")", 1)[0]
    return fn(*[args[p.strip()] for p in sig.split(",")])


def main():
    rng = np.random.default_rng(23)
    nbf, ng = 5, 24
    chi = rng.uniform(0.5, 1.5, (nbf, ng))
    dchi = rng.standard_normal((3, nbf, ng))
    lapl_chi = rng.standard_normal((nbf, ng))
    R = rng.standard_normal((nbf, 3))
    rg = rng.standard_normal((3, ng))
    w = rng.uniform(0.1, 1.0, ng)
    P = 0.15 * rng.standard_normal((nbf, nbf)) + np.eye(nbf)
    P = 0.5 * (P + P.T)

    Rchi = np.einsum("ua,ug->aug", R, chi)
    Rdchi = np.einsum("ua,cug->acug", R, dchi)
    Rlapl = np.einsum("ua,ug->aug", R, lapl_chi)

    tested = failures = 0
    for fam in ("lda", "gga", "mgga_tau", "mgga"):
        ki = fock(fam)
        gen = generate_collapsed(ki, "sq", sesquilinear=True)
        fn = compile_function(gen)

        for s in range(3):
            kig = london_fock(fam, s)
            geng = generate_collapsed(kig, "Kg")
            fng = compile_function(geng)
            chiB0, dchiB0, laplB0 = _phased(np.zeros(3), R, rg, chi, dchi,
                                            lapl_chi)
            rho, grad, sigma, tau, lapl = _fields(P, chiB0, dchiB0, laplB0)
            lx = _deriv_arrays(geng.libxc_args, (rho, sigma, lapl, tau))
            args = {"w": w, "chi": chi, "dchi": dchi, "lapl_chi": lapl_chi,
                    "Rchi": Rchi, "Rdchi": Rdchi, "Rlapl_chi": Rlapl,
                    "rg": rg, "grad_rho": grad, **lx}
            sig = geng.source.split("(", 1)[1].split(")", 1)[0]
            K = fng(*[args[p.strip()] for p in sig.split(",")])
            F_an = 1j / (2 * C_AU) * K

            h = 1e-4
            B = np.zeros(3)

            def F_at(x):
                Bv = B.copy()
                Bv[s] = x
                return _sesqui_fock(fam, gen, fn, P, Bv, R, rg, chi, dchi,
                                    lapl_chi, w)
            d1 = (F_at(h) - F_at(-h)) / (2 * h)
            d2 = (F_at(2 * h) - F_at(-2 * h)) / (4 * h)
            fd = (4 * d1 - d2) / 3
            err = np.abs(F_an - fd).max() / max(np.abs(fd).max(), 1e-14)
            tested += 1
            ok = err < 1e-7
            if not ok:
                failures += 1
            print(f"  [{'OK' if ok else 'FAIL'}] {fam:9s} B_{'xyz'[s]}: "
                  f"max rel {err:.2e}")

    status = "OK " if failures == 0 else "FAIL"
    print(f"[{status}] london_validate: {tested} checks, {failures} failures")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
