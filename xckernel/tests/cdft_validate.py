"""Validate the current-density-DFT family (cmgga_tau): the paramagnetic
current density j_p as an ingredient, with the functional evaluated at the
gauge-corrected tau~ = tau - j_p^2/(2 rho).

No production code computes these kernels, so validation is self-contained:
an explicit analytic functional f(rho, sigma, tau~) stands in for Libxc (all
derivative arrays exact via SymPy), the density matrix M is a GENERAL real
matrix (symmetric part feeds rho/sigma/tau, antisymmetric part carries the
current), and the generated kernels are checked against Richardson-
extrapolated finite differences in M:

  1. Fock          F_uv = dExc/dM_uv      (both symmetric and antisymmetric
                                           displacements -> the jp seed path)
  2. o2 response   F^X = d/dh F(M + h X)  for a general direction X
  3. jp = 0 consistency with the plain mgga_tau family
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from ..emitters.codegen import compile_function, generate_collapsed
from ..engine.deriv import LIBXC_MULTISET

# --- the explicit test functional -------------------------------------------

_r, _s, _t = sp.symbols("rho sigma tau", positive=True)
_F = (_r**2 + sp.Rational(3, 10) * _s * _r + sp.Rational(1, 5) * _t**2
      + sp.Rational(1, 10) * _r * _t + sp.Rational(1, 20) * _s * _t)
_VAR = {"rho": _r, "sigma": _s, "tau": _t}


def _deriv_arrays(names, rho, sigma, taut):
    """Evaluate d^n f / d vars^n for every requested Libxc array name."""
    out = {}
    for name in names:
        ms = LIBXC_MULTISET[name]
        e = _F
        for var, cnt in ms.items():
            e = sp.diff(e, _VAR[var], cnt)
        out[name] = sp.lambdify((_r, _s, _t), e, "numpy")(
            rho, sigma, taut) * np.ones_like(rho)
    return out


# --- fields from a general density matrix -----------------------------------

def _fields(M, chi, dchi):
    rho = np.einsum("uv,ug,vg->g", M, chi, chi)
    grad = np.einsum("uv,cug,vg->cg", M, dchi, chi) \
        + np.einsum("uv,ug,cvg->cg", M, chi, dchi)
    tau = 0.5 * np.einsum("uv,cug,cvg->g", M, dchi, dchi)
    jp = 0.5 * (np.einsum("uv,ug,cvg->cg", M, chi, dchi)
                - np.einsum("uv,cug,vg->cg", M, dchi, chi))
    sigma = np.einsum("cg,cg->g", grad, grad)
    return rho, grad, sigma, tau, jp


def _exc(M, chi, dchi, w):
    rho, grad, sigma, tau, jp = _fields(M, chi, dchi)
    taut = tau - 0.5 * np.einsum("cg,cg->g", jp, jp) / rho
    f = sp.lambdify((_r, _s, _t), _F, "numpy")(rho, sigma, taut)
    return float(np.sum(w * f))


def _call_fock(fn, gen, M, chi, dchi, w, pert=None):
    rho, grad, sigma, tau, jp = _fields(M, chi, dchi)
    inv_rho = 1.0 / rho
    taut = tau - 0.5 * np.einsum("cg,cg->g", jp, jp) * inv_rho
    lx = _deriv_arrays(gen.libxc_args, rho, sigma, taut)
    args = {"w": w, "chi": chi, "dchi": dchi, "grad_rho": grad,
            "jp": jp, "inv_rho": inv_rho, **lx}
    if pert is not None:
        X = pert
        prho, pgrad, _, ptau, pjp = _fields(X, chi, dchi)
        args.update({"rho_p1": prho, "grad_rho_p1": pgrad,
                     "tau_p1": ptau, "jp_p1": pjp})
    sig = gen.source.split("(", 1)[1].split(")", 1)[0]
    return fn(*[args[p.strip()] for p in sig.split(",")])


def main():
    rng = np.random.default_rng(7)
    nbf, ng = 4, 24
    chi = rng.uniform(0.5, 1.5, (nbf, ng))
    dchi = rng.standard_normal((3, nbf, ng))
    w = rng.uniform(0.1, 1.0, ng)
    M = 0.15 * rng.standard_normal((nbf, nbf)) + np.eye(nbf)
    rho0 = _fields(M, chi, dchi)[0]
    assert rho0.min() > 0.05, "test density not positive"

    tested = failures = 0

    def check(label, got, ref, tol):
        nonlocal tested, failures
        tested += 1
        rel = abs(got - ref) / max(abs(ref), 1e-14)
        ok = rel < tol
        if not ok:
            failures += 1
        print(f"  [{'OK' if ok else 'FAIL'}] {label}: "
              f"got {got:+.10e} ref {ref:+.10e} rel {rel:.2e}")

    # 1. Fock vs FD in single matrix entries (symmetric AND antisymmetric
    #    information: off-diagonal M_uv displacements are general)
    from ..engine.kernel import fock
    ki = fock("cmgga_tau")
    gen = generate_collapsed(ki, "cfock")
    fn = compile_function(gen)
    F = _call_fock(fn, gen, M, chi, dchi, w)
    for (u, v) in [(0, 0), (0, 1), (1, 0), (2, 3), (3, 2)]:
        E = np.zeros_like(M)
        E[u, v] = 1.0

        def d(h):
            return (_exc(M + h * E, chi, dchi, w)
                    - _exc(M - h * E, chi, dchi, w)) / (2 * h)
        fd = (4 * d(5e-4) - d(1e-3)) / 3
        check(f"fock dExc/dM[{u},{v}]", F[u, v], fd, 1e-7)

    # explicitly: the current path is live (jp-dependent terms nonzero)
    Fj = _call_fock(fn, gen, M, chi, dchi, w)
    A = M - M.T
    check("antisymmetric channel nonzero",
          float(np.abs(0.5 * (Fj - Fj.T)).max()) > 1e-8, True, 1e-14)

    # 2. o2 response contraction vs FD of the Fock matrix
    from ..engine.response import response_fock
    ki2 = response_fock("cmgga_tau", 2)
    gen2 = generate_collapsed(ki2, "ck2")
    fn2 = compile_function(gen2)
    X = rng.standard_normal((nbf, nbf))
    K = _call_fock(fn2, gen2, M, chi, dchi, w, pert=X)

    def dF(h):
        return (_call_fock(fn, gen, M + h * X, chi, dchi, w)
                - _call_fock(fn, gen, M - h * X, chi, dchi, w)) / (2 * h)
    ref = (4 * dF(5e-4) - dF(1e-3)) / 3
    err = float(np.abs(K - ref).max() / np.abs(ref).max())
    tested += 1
    ok = err < 1e-7
    if not ok:
        failures += 1
    print(f"  [{'OK' if ok else 'FAIL'}] o2 response vs FD(Fock): "
          f"max rel {err:.2e}")

    # 3. jp = 0 reduces to plain mgga_tau
    Msym = 0.5 * (M + M.T)
    ki_m = fock("mgga_tau")
    gen_m = generate_collapsed(ki_m, "mfock")
    fn_m = compile_function(gen_m)
    rho, grad, sigma, tau, jp = _fields(Msym, chi, dchi)
    lx = _deriv_arrays(gen_m.libxc_args, rho, sigma, tau)
    args_m = {"w": w, "chi": chi, "dchi": dchi, "grad_rho": grad, **lx}
    sig_m = gen_m.source.split("(", 1)[1].split(")", 1)[0]
    Fm = fn_m(*[args_m[p.strip()] for p in sig_m.split(",")])
    Fc = _call_fock(fn, gen, Msym, chi, dchi, w)
    err = float(np.abs(Fc - Fm).max() / np.abs(Fm).max())
    tested += 1
    ok = err < 1e-13
    if not ok:
        failures += 1
    print(f"  [{'OK' if ok else 'FAIL'}] jp=0 -> mgga_tau: max rel {err:.2e}")

    status = "OK " if failures == 0 else "FAIL"
    print(f"[{status}] cdft_validate: {tested} checks, {failures} failures")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
