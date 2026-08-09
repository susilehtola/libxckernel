"""Validate the spin-resolved current-density (cmgga_tau) and density-Hessian
(hmgga) families: ua/ub Fock and response kernels, and the closed-shell
spin-adapted (st) parity kernels.

No production code computes these kernels, so validation is self-contained,
following cdft_validate: an explicit analytic polarized functional
f(rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb, [lapl_a, lapl_b], tau_a, tau_b,
[eta_a, eta_b]) stands in for Libxc, with all spin-component derivative
arrays exact via SymPy; the density matrices (M^a, M^b) are GENERAL real
matrices (the antisymmetric parts carry the spin-resolved currents), and the
generated kernels are checked against Richardson-extrapolated finite
differences:

  1. ua/ub o1 Fock         F^s_uv = dExc/dM^s_uv
  2. ua o2 response        F^{a,X} = d/dh F^a(M^a + h X^a, M^b + h X^b)
  3. st o2 parities        closed shell, X^b = +/- X^a
  4. ua o3 response        d/dh of the o2 contraction
  5. spin-pure consistency: jp = 0 reduces cmgga_tau to mgga_tau spin
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import sympy as sp

from ..basis import HESS_COMPS
from ..codegen import compile_function, generate_collapsed
from ..spin import COMPS
from ..spin_kernel import (_SYM_SCALARS, fock_spin, response_fock_spin,
                           response_fock_st)

SPINS = ("a", "b")
_H6 = [f"{'xyz'[i]}{'xyz'[j]}" for (i, j) in HESS_COMPS]

# --- the explicit polarized test functionals --------------------------------

_V = {("rho", "a"): sp.Symbol("ra"), ("rho", "b"): sp.Symbol("rb"),
      ("sigma", "aa"): sp.Symbol("saa"), ("sigma", "ab"): sp.Symbol("sab"),
      ("sigma", "bb"): sp.Symbol("sbb"),
      ("lapl", "a"): sp.Symbol("la"), ("lapl", "b"): sp.Symbol("lb"),
      ("tau", "a"): sp.Symbol("ta"), ("tau", "b"): sp.Symbol("tb"),
      ("eta", "a"): sp.Symbol("ea"), ("eta", "b"): sp.Symbol("eb")}

_ra, _rb = _V[("rho", "a")], _V[("rho", "b")]
_saa, _sab, _sbb = (_V[("sigma", c)] for c in ("aa", "ab", "bb"))
_la, _lb = _V[("lapl", "a")], _V[("lapl", "b")]
_ta, _tb = _V[("tau", "a")], _V[("tau", "b")]
_ea, _eb = _V[("eta", "a")], _V[("eta", "b")]

_F_C = (_ra**2 + _rb**2 + sp.Rational(1, 2) * _ra * _rb
        + sp.Rational(3, 10) * (_saa * _ra + _sbb * _rb)
        + sp.Rational(1, 10) * _sab * (_ra + _rb)
        + sp.Rational(1, 5) * (_ta**2 + _tb**2)
        + sp.Rational(1, 10) * (_ra * _tb + _rb * _ta)
        + sp.Rational(1, 20) * (_saa * _tb + _sbb * _ta))

_F_H = (_F_C + sp.Rational(1, 8) * (_la**2 + _lb**2)
        + sp.Rational(1, 12) * (_la * _rb + _lb * _ra)
        + sp.Rational(1, 15) * (_ea * _ra + _eb * _rb)
        + sp.Rational(1, 25) * (_ea * _tb + _eb * _ta)
        + sp.Rational(1, 30) * (_ea * _eb))

_FUNC = {"cmgga_tau": _F_C, "hmgga": _F_H, "mgga_tau": _F_C}


def _deriv_arrays(family: str, names, vals: Dict[sp.Symbol, np.ndarray]):
    """Exact '<array>_<comp>' arrays: differentiate f wrt the scalar multiset
    that spin_kernel registered for each derivative symbol name."""
    f = _FUNC[family]
    syms = list(vals.keys())
    ng = len(next(iter(vals.values())))
    out = {}
    for name in names:
        scalars = _SYM_SCALARS[name]
        e = f
        for sc in scalars:
            e = sp.diff(e, _V[(sc.group, sc.comp)])
        out[name] = sp.lambdify(syms, e, "numpy")(*vals.values()) \
            * np.ones(ng)
    return out


# --- fields from general per-spin density matrices --------------------------

def _fields(M, chi, dchi, lapl_chi, hess_chi):
    rho = np.einsum("uv,ug,vg->g", M, chi, chi)
    grad = np.einsum("uv,cug,vg->cg", M, dchi, chi) \
        + np.einsum("uv,ug,cvg->cg", M, chi, dchi)
    lapl = np.einsum("uv,ug,vg->g", M, lapl_chi, chi) \
        + 2.0 * np.einsum("uv,cug,cvg->g", M, dchi, dchi) \
        + np.einsum("uv,ug,vg->g", M, chi, lapl_chi)
    tau = 0.5 * np.einsum("uv,cug,cvg->g", M, dchi, dchi)
    jp = 0.5 * (np.einsum("uv,ug,cvg->cg", M, chi, dchi)
                - np.einsum("uv,cug,vg->cg", M, dchi, chi))
    hess = np.empty((6, chi.shape[1]))
    for k, (i, j) in enumerate(HESS_COMPS):
        hess[k] = np.einsum("uv,ug,vg->g", M, hess_chi[k], chi) \
            + np.einsum("uv,ug,vg->g", M, dchi[i], dchi[j]) \
            + np.einsum("uv,ug,vg->g", M, dchi[j], dchi[i]) \
            + np.einsum("uv,ug,vg->g", M, chi, hess_chi[k])
    return {"rho": rho, "grad": grad, "lapl": lapl, "tau": tau,
            "jp": jp, "hess": hess}


def _eta_of(f):
    """eta = grad . H . grad from a per-spin field dict (spin-pure)."""
    g, h = f["grad"], f["hess"]
    total = np.zeros_like(f["rho"])
    for k, (i, j) in enumerate(HESS_COMPS):
        wgt = 1.0 if i == j else 2.0
        total += wgt * g[i] * h[k] * g[j]
    return total


def _scalar_vals(family: str, fa, fb) -> Dict[sp.Symbol, np.ndarray]:
    vals = {_ra: fa["rho"], _rb: fb["rho"],
            _saa: np.einsum("cg,cg->g", fa["grad"], fa["grad"]),
            _sab: np.einsum("cg,cg->g", fa["grad"], fb["grad"]),
            _sbb: np.einsum("cg,cg->g", fb["grad"], fb["grad"])}
    if family == "cmgga_tau":
        for sym, f in ((_ta, fa), (_tb, fb)):
            vals[sym] = f["tau"] - 0.5 * np.einsum(
                "cg,cg->g", f["jp"], f["jp"]) / f["rho"]
    else:
        vals[_ta], vals[_tb] = fa["tau"], fb["tau"]
    if family == "hmgga":
        vals[_la], vals[_lb] = fa["lapl"], fb["lapl"]
        vals[_ea], vals[_eb] = _eta_of(fa), _eta_of(fb)
    return vals


def _exc(family, Ma, Mb, basis, w):
    fa = _fields(Ma, *basis)
    fb = _fields(Mb, *basis)
    vals = _scalar_vals(family, fa, fb)
    f = sp.lambdify(list(vals.keys()), _FUNC[family], "numpy")(*vals.values())
    return float(np.sum(w * (f * np.ones_like(fa["rho"]))))


# --- generated-kernel caller -------------------------------------------------

def _call(fn, gen, family, Ma, Mb, basis, w, perts=(), closed_shell=False):
    """perts: list of (Xa, Xb) per label p1, p2, ..."""
    chi, dchi, lapl_chi, hess_chi = basis
    fa, fb = _fields(Ma, *basis), _fields(Mb, *basis)
    vals = _scalar_vals(family, fa, fb)
    args = {"w": w, "chi": chi, "dchi": dchi, "lapl_chi": lapl_chi,
            "hess_chi": hess_chi}
    for s, f in (("a", fa), ("b", fb)):
        args[f"grad_rho_{s}"] = f["grad"]
        args[f"jp_{s}"] = f["jp"]
        args[f"inv_rho_{s}"] = 1.0 / f["rho"]
        args[f"hess_rho_{s}"] = f["hess"]
    for k, (Xa, Xb) in enumerate(perts, start=1):
        for s, X in (("a", Xa), ("b", Xb)):
            p = _fields(X, *basis)
            args[f"rho_{s}_p{k}"] = p["rho"]
            args[f"grad_rho_{s}_p{k}"] = p["grad"]
            args[f"lapl_rho_{s}_p{k}"] = p["lapl"]
            args[f"tau_{s}_p{k}"] = p["tau"]
            args[f"jp_{s}_p{k}"] = p["jp"]
            args[f"hess_rho_{s}_p{k}"] = p["hess"]
    args.update(_deriv_arrays(family, gen.libxc_args, vals))
    sig = gen.source.split("(", 1)[1].split(")", 1)[0]
    return fn(*[args[p.strip()] for p in sig.split(",")])


def main():
    rng = np.random.default_rng(11)
    nbf, ng = 4, 20
    chi = rng.uniform(0.5, 1.5, (nbf, ng))
    dchi = rng.standard_normal((3, nbf, ng))
    hess_chi = rng.standard_normal((6, nbf, ng))
    lapl_chi = hess_chi[0] + hess_chi[3] + hess_chi[5]
    basis = (chi, dchi, lapl_chi, hess_chi)
    w = rng.uniform(0.1, 1.0, ng)
    Ma = 0.15 * rng.standard_normal((nbf, nbf)) + np.eye(nbf)
    Mb = 0.15 * rng.standard_normal((nbf, nbf)) + 0.9 * np.eye(nbf)

    tested = failures = 0

    def check(label, err, tol):
        nonlocal tested, failures
        tested += 1
        ok = err < tol
        if not ok:
            failures += 1
        print(f"  [{'OK' if ok else 'FAIL'}] {label}: max rel {err:.2e}")

    for family in ("cmgga_tau", "hmgga"):
        print(f"== {family} ==")

        # 1. ua/ub Fock vs FD of the energy
        focks = {}
        for s in SPINS:
            ki = fock_spin(family, s)
            gen = generate_collapsed(ki, f"f_{s}")
            focks[s] = (gen, compile_function(gen))
        for s, Ms in (("a", Ma), ("b", Mb)):
            gen, fn = focks[s]
            F = _call(fn, gen, family, Ma, Mb, basis, w)
            errs = []
            for (u, v) in [(0, 0), (0, 1), (1, 0), (2, 3)]:
                E = np.zeros((nbf, nbf))
                E[u, v] = 1.0

                def d(h):
                    def base(t):
                        return Ma if t == "a" else Mb
                    Pp = [base(t) + h * E if t == s else base(t)
                          for t in SPINS]
                    Pm = [base(t) - h * E if t == s else base(t)
                          for t in SPINS]
                    return (_exc(family, *Pp, basis, w)
                            - _exc(family, *Pm, basis, w)) / (2 * h)
                fd = (4 * d(5e-4) - d(1e-3)) / 3
                errs.append(abs(F[u, v] - fd) / max(abs(fd), 1e-12))
            check(f"o1 F^{s} vs FD(Exc)", max(errs), 1e-6)

        # 2. ua o2 vs FD of the alpha Fock (general two-channel direction)
        ki2 = response_fock_spin(family, "a", 2)
        gen2 = generate_collapsed(ki2, "k2")
        fn2 = compile_function(gen2)
        Xa = rng.standard_normal((nbf, nbf))
        Xb = rng.standard_normal((nbf, nbf))
        K = _call(fn2, gen2, family, Ma, Mb, basis, w, perts=[(Xa, Xb)])
        gen1, fn1 = focks["a"]

        def dF(h):
            return (_call(fn1, gen1, family, Ma + h * Xa, Mb + h * Xb,
                          basis, w)
                    - _call(fn1, gen1, family, Ma - h * Xa, Mb - h * Xb,
                            basis, w)) / (2 * h)
        ref = (4 * dF(5e-4) - dF(1e-3)) / 3
        check("o2 ua vs FD(F^a)", float(np.abs(K - ref).max()
                                        / np.abs(ref).max()), 1e-6)

        # 3. st o2 parities at the closed shell
        for par, pname in ((+1, "p"), (-1, "m")):
            ki_st = response_fock_st(family, 2, (par,))
            gen_st = generate_collapsed(ki_st, f"st_{pname}")
            fn_st = compile_function(gen_st)
            Kst = _call(fn_st, gen_st, family, Ma, Ma, basis, w,
                        perts=[(Xa, par * Xa)])

            def dFst(h):
                return (_call(fn1, gen1, family, Ma + h * Xa,
                              Ma + h * par * Xa, basis, w)
                        - _call(fn1, gen1, family, Ma - h * Xa,
                                Ma - h * par * Xa, basis, w)) / (2 * h)
            refst = (4 * dFst(5e-4) - dFst(1e-3)) / 3
            check(f"st o2 parity {par:+d} vs FD(F^a)",
                  float(np.abs(Kst - refst).max() / np.abs(refst).max()),
                  1e-6)

        # 4. ua o3 vs FD of the o2 contraction
        ki3 = response_fock_spin(family, "a", 3)
        gen3 = generate_collapsed(ki3, "k3")
        fn3 = compile_function(gen3)
        Ya = rng.standard_normal((nbf, nbf))
        Yb = rng.standard_normal((nbf, nbf))
        K3 = _call(fn3, gen3, family, Ma, Mb, basis, w,
                   perts=[(Xa, Xb), (Ya, Yb)])

        def dK2(h):
            return (_call(fn2, gen2, family, Ma + h * Ya, Mb + h * Yb,
                          basis, w, perts=[(Xa, Xb)])
                    - _call(fn2, gen2, family, Ma - h * Ya, Mb - h * Yb,
                            basis, w, perts=[(Xa, Xb)])) / (2 * h)
        ref3 = (4 * dK2(5e-4) - dK2(1e-3)) / 3
        check("o3 ua vs FD(o2)", float(np.abs(K3 - ref3).max()
                                       / np.abs(ref3).max()), 1e-5)

    # 5. jp = 0: spin cmgga_tau Fock reduces to spin mgga_tau
    Msa, Msb = 0.5 * (Ma + Ma.T), 0.5 * (Mb + Mb.T)
    for s in SPINS:
        outs = {}
        for fam in ("cmgga_tau", "mgga_tau"):
            ki = fock_spin(fam, s)
            gen = generate_collapsed(ki, "f")
            fn = compile_function(gen)
            outs[fam] = _call(fn, gen, fam, Msa, Msb, basis, w)
        err = float(np.abs(outs["cmgga_tau"] - outs["mgga_tau"]).max()
                    / np.abs(outs["mgga_tau"]).max())
        check(f"jp=0 -> mgga_tau (channel {s})", err, 1e-13)

    status = "OK " if failures == 0 else "FAIL"
    print(f"[{status}] spin_ext_validate: {tested} checks, "
          f"{failures} failures")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
