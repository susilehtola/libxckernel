"""Validate the XC orbital Hessian against finite differences of Exc(kappa).

We fabricate a grid (as in validate.py), draw a random orthogonal C, occupy the
first nocc orbitals with occupation `occ`, and define

    Exc(x) = Exc[P(x)],   P(x) = occ * C_occ(x) C_occ(x)^T,
    C(x) = C exp(-kappa(x)),  kappa_ai = x_ia, kappa_ia = -x_ia.

The analytic Hessian is assembled by mo.orbital_hessian from the XC Fock matrix
and the AO XC kernel, both evaluated with the *generated einsum code* -- so this
test also exercises codegen at both derivative orders.  The reference is a mixed
central finite difference of Exc(x) at x = 0.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.linalg import expm

from ..emitters.codegen import compile_function, generate
from ..engine.kernel import fock, xc_kernel
from ..engine.mo import orbital_gradient, orbital_hessian
from .validate import (FAMILY_FUNCTIONAL, Grid, ingredients_from_P,
                       libxc_eval, exc_of_P, make_grid)


def _call(gen, fn, g: Grid, ing, out):
    """Call a generated function with fabricated-grid operands."""
    # generated code wants dchi as (3, nao, ng); the fabricated grid stores
    # (nao, 3, ng)
    args = [g.w, g.chi, np.transpose(g.dchi, (1, 0, 2))]
    if gen.uses_lapl_chi:
        args.append(g.lapl_chi)
    if gen.uses_grad_rho:
        args.append(ing["_grad"])
    args += [out[name].reshape(-1) for name in gen.libxc_args]
    return fn(*args)


def _orbitals(nbf: int, nocc: int, occ: float, seed: int):
    rng = np.random.default_rng(seed)
    C, _ = np.linalg.qr(rng.standard_normal((nbf, nbf)))
    P0 = occ * C[:, :nocc] @ C[:, :nocc].T
    return C, P0


def exc_of_x(name, family, g: Grid, C, nocc, occ, x, sign=-1) -> float:
    """Exc under the rotation C(x) = C exp(sign * kappa)."""
    nbf = C.shape[0]
    kappa = np.zeros((nbf, nbf))
    kappa[nocc:, :nocc] = x.T          # kappa_ai = x_ia
    kappa[:nocc, nocc:] = -x           # kappa_ia = -x_ia
    Cx = C @ expm(sign * kappa)
    P = occ * Cx[:, :nocc] @ Cx[:, :nocc].T
    return exc_of_P(name, family, g, P)


def check(family: str, nocc=2, occ=2.0, h=2e-4):
    name = FAMILY_FUNCTIONAL[family]
    g = make_grid(nbf=4, npts=150, seed=5)
    nbf = g.chi.shape[0]
    C, P0 = _orbitals(nbf, nocc, occ, seed=6)
    nvir = nbf - nocc

    # analytic: F and G from generated einsum code at P0
    ing = ingredients_from_P(g, P0)
    out = libxc_eval(name, family, ing, do_fxc=True)
    gF = generate(fock(family), "F_fn")
    gG = generate(xc_kernel(family), "G_fn")
    F = _call(gF, compile_function(gF), g, ing, out)
    G = _call(gG, compile_function(gG), g, ing, out)
    Ha = orbital_hessian(F, G, C, nocc, occ)

    # reference: mixed central differences of Exc(x)
    def E(x):
        return exc_of_x(name, family, g, C, nocc, occ, x)

    Hd = np.zeros((nocc, nvir, nocc, nvir))
    for i in range(nocc):
        for a in range(nvir):
            for j in range(nocc):
                for b in range(nvir):
                    def Exy(s1, s2):
                        x = np.zeros((nocc, nvir))
                        x[i, a] += s1 * h
                        x[j, b] += s2 * h
                        return E(x)
                    Hd[i, a, j, b] = (Exy(+1, +1) - Exy(+1, -1)
                                      - Exy(-1, +1) + Exy(-1, -1)) / (4 * h * h)

    err = np.max(np.abs(Ha - Hd))
    scale = np.max(np.abs(Hd)) or 1.0
    return err, err / scale


def check_gradient(family: str, sign: int, nocc=2, occ=2.0, h=1e-6):
    """Analytic orbital gradient vs FD of Exc(kappa), per sign convention."""
    name = FAMILY_FUNCTIONAL[family]
    g = make_grid(nbf=4, npts=150, seed=5)
    nbf = g.chi.shape[0]
    C, P0 = _orbitals(nbf, nocc, occ, seed=6)
    nvir = nbf - nocc

    ing = ingredients_from_P(g, P0)
    out = libxc_eval(name, family, ing)
    gF = generate(fock(family), "F_fn")
    F = _call(gF, compile_function(gF), g, ing, out)
    Ga = orbital_gradient(F, C, nocc, occ, sign=sign)

    Gd = np.zeros((nocc, nvir))
    for i in range(nocc):
        for a in range(nvir):
            def E(s):
                x = np.zeros((nocc, nvir))
                x[i, a] = s * h
                return exc_of_x(name, family, g, C, nocc, occ, x, sign=sign)
            Gd[i, a] = (E(+1) - E(-1)) / (2 * h)

    err = np.max(np.abs(Ga - Gd))
    scale = np.max(np.abs(Gd)) or 1.0
    return err, err / scale


if __name__ == "__main__":
    print("XC orbital gradient  dExc/dx_ia  vs FD, both kappa conventions")
    for fam in ("lda", "gga", "mgga_tau", "mgga"):
        for sign in (-1, +1):
            err, rel = check_gradient(fam, sign)
            print(f"  [{'OK ' if rel < 1e-6 else 'FAIL'}] {fam:9s} "
                  f"sign={sign:+d} {FAMILY_FUNCTIONAL[fam]:14s} "
                  f"abs={err:.3e} rel={rel:.3e}")

    print("XC orbital Hessian  H_ia,jb = d2Exc/dx_ia dx_jb  vs FD of Exc(kappa)")
    for fam in ("lda", "gga", "mgga_tau", "mgga"):
        err, rel = check(fam)
        print(f"  [{'OK ' if rel < 1e-4 else 'FAIL'}] {fam:9s} "
              f"{FAMILY_FUNCTIONAL[fam]:14s} abs={err:.3e} rel={rel:.3e}")
