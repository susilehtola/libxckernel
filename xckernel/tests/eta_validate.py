"""Validate the local-hybrid calibration-function family (hmgga): the
gradient-projected density Hessian

    eta = grad rho^T . (grad grad^T rho) . grad rho

[Z_{sigma,sigmasigma} of Maier et al., Phys. Chem. Chem. Phys. 18, 21133
(2016), Eqs. (22)-(23); eta in the notation of Schattenberg & Kaupp,
J. Phys. Chem. A 125, 2697 (2021), Eq. (10)] as a functional variable
alongside the full meta-GGA set (rho, sigma, lapl, tau).

eta is CUBIC in P-linear primitives (two gradient factors and one
density-Hessian factor), one factor deeper than sigma, so its derivative
tower is the richest in the library; the density-Hessian primitives also
pull in the second-derivative basis collocation hess_chi.  No functional
library computes eta derivatives, so validation is self-contained: an
explicit analytic functional f(rho, sigma, lapl, tau, eta) stands in for
Libxc (all derivative arrays exact via SymPy) and the generated kernels are
checked against Richardson-extrapolated finite differences in the density
matrix M:

  1. Fock          F_uv = dExc/dM_uv
  2. o2 response   F^X = d/dh F(M + h X)   for a general direction X
  3. o3 response   F^{X,Y} = d/dh F^X(M + h Y)  (the cubic-eta second seed)
  4. eta-independent functional reduces to the plain mgga family
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from ..codegen import compile_function, generate_collapsed
from ..deriv import LIBXC_MULTISET

# --- the explicit test functional -------------------------------------------

_r, _s, _l, _t, _e = sp.symbols("rho sigma lapl tau eta", real=True)
_F = (_r**2 + sp.Rational(3, 10) * _s * _r + sp.Rational(1, 5) * _t**2
      + sp.Rational(1, 10) * _r * _t + sp.Rational(1, 20) * _s * _t
      + sp.Rational(1, 30) * _r * _l + sp.Rational(1, 50) * _l**2
      + sp.Rational(1, 40) * _e * _r + sp.Rational(1, 60) * _e * _s
      + sp.Rational(1, 70) * _e * _t + sp.Rational(1, 80) * _e * _l
      + sp.Rational(1, 100) * _e**2)
_VAR = {"rho": _r, "sigma": _s, "lapl": _l, "tau": _t, "eta": _e}
_ARGS = (_r, _s, _l, _t, _e)

#: packed symmetric-tensor components, canonical order.
_H6 = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


def _deriv_arrays(names, F, vals):
    """Evaluate d^n f / d vars^n for every requested derivative-array name."""
    out = {}
    for name in names:
        ms = LIBXC_MULTISET[name]
        expr = F
        for var, cnt in ms.items():
            expr = sp.diff(expr, _VAR[var], cnt)
        out[name] = sp.lambdify(_ARGS, expr, "numpy")(*vals) \
            * np.ones_like(vals[0])
    return out


# --- fields from a density matrix -------------------------------------------

def _fields(M, chi, dchi, lapl_chi, hess_chi):
    rho = np.einsum("uv,ug,vg->g", M, chi, chi)
    grad = np.einsum("uv,cug,vg->cg", M, dchi, chi) \
        + np.einsum("uv,ug,cvg->cg", M, chi, dchi)
    sigma = np.einsum("cg,cg->g", grad, grad)
    lapl = (np.einsum("uv,ug,vg->g", M, lapl_chi, chi)
            + 2.0 * np.einsum("uv,cug,cvg->g", M, dchi, dchi)
            + np.einsum("uv,ug,vg->g", M, chi, lapl_chi))
    tau = 0.5 * np.einsum("uv,cug,cvg->g", M, dchi, dchi)
    hess = np.stack([
        np.einsum("uv,ug,vg->g", M, hess_chi[k], chi)
        + np.einsum("uv,ug,vg->g", M, dchi[i], dchi[j])
        + np.einsum("uv,ug,vg->g", M, dchi[j], dchi[i])
        + np.einsum("uv,ug,vg->g", M, chi, hess_chi[k])
        for k, (i, j) in enumerate(_H6)])
    eta = sum((1.0 if i == j else 2.0) * grad[i] * grad[j] * hess[k]
              for k, (i, j) in enumerate(_H6))
    return rho, grad, sigma, lapl, tau, hess, eta


def _exc(M, basis, w):
    rho, grad, sigma, lapl, tau, hess, eta = _fields(M, *basis)
    f = sp.lambdify(_ARGS, _F, "numpy")(rho, sigma, lapl, tau, eta)
    return float(np.sum(w * f))


def _call(fn, gen, M, basis, w, perts=()):
    chi, dchi, lapl_chi, hess_chi = basis
    rho, grad, sigma, lapl, tau, hess, eta = _fields(M, *basis)
    lx = _deriv_arrays(gen.libxc_args, _F, (rho, sigma, lapl, tau, eta))
    args = {"w": w, "chi": chi, "dchi": dchi, "lapl_chi": lapl_chi,
            "hess_chi": hess_chi, "grad_rho": grad, "hess_rho": hess, **lx}
    for n, X in enumerate(perts, start=1):
        prho, pgrad, _, plapl, ptau, phess, _ = _fields(X, *basis)
        args.update({f"rho_p{n}": prho, f"grad_rho_p{n}": pgrad,
                     f"lapl_rho_p{n}": plapl, f"tau_p{n}": ptau,
                     f"hess_rho_p{n}": phess})
    sig = gen.source.split("(", 1)[1].split(")", 1)[0]
    return fn(*[args[p.strip()] for p in sig.split(",")])


def main():
    rng = np.random.default_rng(11)
    nbf, ng = 4, 24
    chi = rng.uniform(0.5, 1.5, (nbf, ng))
    dchi = rng.standard_normal((3, nbf, ng))
    lapl_chi = rng.standard_normal((nbf, ng))
    hess_chi = rng.standard_normal((6, nbf, ng))
    basis = (chi, dchi, lapl_chi, hess_chi)
    w = rng.uniform(0.1, 1.0, ng)
    M = 0.1 * rng.standard_normal((nbf, nbf)) + np.eye(nbf)
    M = 0.5 * (M + M.T)

    tested = failures = 0

    def check(label, err, tol):
        nonlocal tested, failures
        tested += 1
        ok = err < tol
        if not ok:
            failures += 1
        print(f"  [{'OK' if ok else 'FAIL'}] {label}: max rel {err:.2e}")

    from ..kernel import fock
    from ..response import response_fock

    # 1. Fock vs FD in single matrix entries
    gen1 = generate_collapsed(fock("hmgga"), "hfock")
    fn1 = compile_function(gen1)
    F = _call(fn1, gen1, M, basis, w)
    worst = 0.0
    for (u, v) in [(0, 0), (0, 1), (2, 3), (3, 1)]:
        E = np.zeros_like(M)
        E[u, v] = 1.0

        def d(h):
            return (_exc(M + h * E, basis, w)
                    - _exc(M - h * E, basis, w)) / (2 * h)
        fd = (4 * d(5e-4) - d(1e-3)) / 3
        worst = max(worst, abs(F[u, v] - fd) / max(abs(fd), 1e-14))
    check("fock dExc/dM vs FD (4 entries)", worst, 1e-7)

    # 2. o2 response contraction vs FD of the Fock matrix
    gen2 = generate_collapsed(response_fock("hmgga", 2), "hk2")
    fn2 = compile_function(gen2)
    X = rng.standard_normal((nbf, nbf))
    K2 = _call(fn2, gen2, M, basis, w, perts=(X,))

    def dF(h):
        return (_call(fn1, gen1, M + h * X, basis, w)
                - _call(fn1, gen1, M - h * X, basis, w)) / (2 * h)
    ref = (4 * dF(5e-4) - dF(1e-3)) / 3
    check("o2 response vs FD(Fock)",
          float(np.abs(K2 - ref).max() / np.abs(ref).max()), 1e-7)

    # 3. o3 response contraction vs FD of the o2 contraction
    gen3 = generate_collapsed(response_fock("hmgga", 3), "hk3")
    fn3 = compile_function(gen3)
    Y = rng.standard_normal((nbf, nbf))
    K3 = _call(fn3, gen3, M, basis, w, perts=(X, Y))

    def dK2(h):
        return (_call(fn2, gen2, M + h * Y, basis, w, perts=(X,))
                - _call(fn2, gen2, M - h * Y, basis, w, perts=(X,))) / (2 * h)
    ref = (4 * dK2(5e-4) - dK2(1e-3)) / 3
    check("o3 response vs FD(o2)",
          float(np.abs(K3 - ref).max() / np.abs(ref).max()), 1e-6)

    # 4. eta-independent functional reduces to the plain mgga family
    F_noeta = _F.subs(_e, 0)
    gen_m = generate_collapsed(fock("mgga"), "mfock")
    fn_m = compile_function(gen_m)
    rho, grad, sigma, lapl, tau, hess, eta = _fields(M, *basis)
    lx = _deriv_arrays(gen_m.libxc_args, F_noeta, (rho, sigma, lapl, tau, eta))
    args_m = {"w": w, "chi": chi, "dchi": dchi, "lapl_chi": lapl_chi,
              "grad_rho": grad, **lx}
    sig_m = gen_m.source.split("(", 1)[1].split(")", 1)[0]
    Fm = fn_m(*[args_m[p.strip()] for p in sig_m.split(",")])
    lxh = _deriv_arrays(gen1.libxc_args, F_noeta, (rho, sigma, lapl, tau, eta))
    args_h = {"w": w, "chi": chi, "dchi": dchi, "lapl_chi": lapl_chi,
              "hess_chi": hess_chi, "grad_rho": grad, "hess_rho": hess, **lxh}
    sig_h = gen1.source.split("(", 1)[1].split(")", 1)[0]
    Fh = fn1(*[args_h[p.strip()] for p in sig_h.split(",")])
    check("eta-independent -> mgga",
          float(np.abs(Fh - Fm).max() / np.abs(Fm).max()), 1e-13)

    status = "OK " if failures == 0 else "FAIL"
    print(f"[{status}] eta_validate: {tested} checks, {failures} failures")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
