"""Quadratic response end-to-end: the XC sigma vector sigma_ia =
d^3 Exc / db dc dx_ia, assembled by algebra.quadratic_sigma_xc from layer-1
contractions (orders 1-3) and layer-2 nested-commutator densities, validated
against pure finite differences of the energy under exp(sign*(b kB + c kC +
y K_ia)) rotations.  Both sign conventions (odd order => sign matters).
"""

from __future__ import annotations

import numpy as np
from pylibxc import LibXCFunctional
from scipy.linalg import expm

from ..algebra import quadratic_sigma_xc, unit_rotation
from ..codegen import compile_function, generate
from ..kernel import fock
from ..response import response_fock

NAME = {"lda": "LDA_X", "gga": "GGA_X_PBE"}
VARS = {"lda": ["rho"], "gga": ["rho", "sigma"]}


def _make_system(seed=9, nbf=4, npts=150, nocc=2, occ=2.0):
    rng = np.random.default_rng(seed)
    w = rng.uniform(0.1, 1.0, npts)
    chi = rng.standard_normal((nbf, npts))
    dchi = rng.standard_normal((3, nbf, npts))
    C, _ = np.linalg.qr(rng.standard_normal((nbf, nbf)))
    P0 = occ * C[:, :nocc] @ C[:, :nocc].T

    def antisym():
        a = rng.standard_normal((nbf, nbf))
        return a - a.T

    return w, chi, dchi, C, P0, antisym(), antisym()


def check(family: str, sign: int, h: float = 1.25e-3):
    nocc, occ = 2, 2.0
    w, chi, dchi, C, P0, kB, kC = _make_system(nocc=nocc, occ=occ)
    func = LibXCFunctional(NAME[family], "unpolarized")

    def fields(P):
        rho = np.einsum("uv,ug,vg->g", P, chi, chi)
        grad = np.einsum("uv,iug,vg->ig", P, dchi, chi) \
            + np.einsum("uv,ug,ivg->ig", P, chi, dchi)
        return rho, grad

    def libxc_at(P, **kw):
        rho, grad = fields(P)
        inp = {"rho": rho}
        if "sigma" in VARS[family]:
            inp["sigma"] = np.einsum("ig,ig->g", grad, grad)
        out = func.compute(inp, **kw)
        return {k: v.reshape(-1) for k, v in out.items() if v is not None}

    # ---- layer-1 generated contractions at the reference density ----
    lib0 = libxc_at(P0, do_fxc=True, do_kxc=True)
    rho0, grad0 = fields(P0)
    base = {"w": w, "chi": chi, "dchi": dchi, "grad_rho": grad0}

    gens = {n: generate(response_fock(family, n) if n > 1 else fock(family),
                        f"g{n}") for n in (1, 2, 3)}
    fns = {n: compile_function(g) for n, g in gens.items()}

    def call(n, perts):
        gen = gens[n]
        args = [w, chi, dchi]
        if gen.uses_grad_rho:
            args.append(grad0)
        args += [perts[lbl]["grad"] for lbl in (gen.pert_grads or [])]
        for name in gen.pert_scalars or []:
            field, lbl = name.rsplit("_", 1)
            args.append(perts[lbl][field])
        args += [lib0[name] for name in gen.libxc_args]
        return fns[n](*args)

    def pert_of(D):
        r, g = fields(D)
        return {"rho": r, "grad": g}

    fock0 = lambda: call(1, {})
    fresp = lambda D: call(2, {"p1": pert_of(D)})
    fresp2 = lambda D1, D2: call(3, {"p1": pert_of(D1), "p2": pert_of(D2)})

    sigma = quadratic_sigma_xc(kB, kC, C, nocc, fock0, fresp, fresp2,
                               occ=occ, sign=sign)

    # ---- FD reference: third mixed derivative of the energy ----
    def exc(P):
        rho = fields(P)[0]
        return float(np.sum(w * rho * libxc_at(P, do_exc=True)["zk"]))

    def E(b, c, y, K):
        X = sign * (b * kB + c * kC + y * K)
        Cx = C @ expm(X)
        return exc(occ * Cx[:, :nocc] @ Cx[:, :nocc].T)

    def fd_at(step, K):
        tot = 0.0
        for s1 in (+1, -1):
            for s2 in (+1, -1):
                for s3 in (+1, -1):
                    tot += s1 * s2 * s3 * E(s1 * step, s2 * step, s3 * step, K)
        return tot / (8 * step ** 3)

    nvir = C.shape[1] - nocc
    ref = np.zeros((nocc, nvir))
    for i in range(nocc):
        for a in range(nvir):
            K = unit_rotation(i, a, nocc, C.shape[1])
            # Richardson extrapolation removes the O(h^2) truncation term
            ref[i, a] = (4 * fd_at(h / 2, K) - fd_at(h, K)) / 3

    err = np.max(np.abs(sigma - ref))
    scale = np.max(np.abs(ref)) or 1.0
    return err, err / scale


if __name__ == "__main__":
    print("Quadratic-response XC sigma  d3Exc/db dc dx_ia  vs FD of Exc")
    for fam in ("lda", "gga"):
        for sign in (-1, +1):
            err, rel = check(fam, sign)
            print(f"  [{'OK ' if rel < 1e-4 else 'FAIL'}] {fam:4s} "
                  f"sign={sign:+d} abs={err:.3e} rel={rel:.3e}")
