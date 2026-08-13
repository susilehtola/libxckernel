"""Validate the spin-polarized geometric layer against finite
differences with grid and weights FROZEN while basis centers move:

* first order  -- geometric_fock_spin (the basis class of dF^s/dX)
  against FD of the spin-resolved Fock matrix, and
* second order -- geometric_hessian_spin (pair/same function-pair
  kernel) against mixed FD of the energy.

Explicit s-Gaussian collocation (analytic through third derivatives),
explicit polarized polynomial functional coupling all channels."""

from __future__ import annotations

import numpy as np
import sympy as sp

from ..engine.geometric import geometric_fock_spin, geometric_hessian_spin
from ..engine.fastpoly import from_expr
from ..engine.spin_kernel import _SYM_SCALARS
from .geometric2_validate import ALPHA, BF_ATOM, colloc

# --- explicit polarized functional --------------------------------------------

_V = {("rho", "a"): sp.Symbol("Ra", positive=True),
      ("rho", "b"): sp.Symbol("Rb", positive=True),
      ("sigma", "aa"): sp.Symbol("Saa", positive=True),
      ("sigma", "ab"): sp.Symbol("Sab", real=True),
      ("sigma", "bb"): sp.Symbol("Sbb", positive=True),
      ("tau", "a"): sp.Symbol("Ta", positive=True),
      ("tau", "b"): sp.Symbol("Tb", positive=True)}
_Ra, _Rb = _V[("rho", "a")], _V[("rho", "b")]
_Saa, _Sab, _Sbb = (_V[("sigma", c)] for c in ("aa", "ab", "bb"))
_Ta, _Tb = _V[("tau", "a")], _V[("tau", "b")]
_F = (_Ra**2 + sp.Rational(4, 5) * _Rb**2 + sp.Rational(3, 10) * _Saa * _Rb
      + sp.Rational(1, 5) * _Sab * (_Ra + _Rb) + sp.Rational(3, 20) * _Sbb * _Ra
      + sp.Rational(1, 5) * _Ta**2 + sp.Rational(1, 10) * _Tb * _Ra
      + sp.Rational(1, 20) * _Sab * _Tb + sp.Rational(1, 10) * _Ta * _Rb)
_ARGS = tuple(_V.values())


def _spin_fields(Da, Db, chi, dchi):
    D = {"a": Da, "b": Db}
    rho = {s: np.einsum("uv,ug,vg->g", D[s], chi, chi) for s in "ab"}
    grad = {s: 2.0 * np.einsum("uv,cug,vg->cg", D[s], dchi, chi) for s in "ab"}
    tau = {s: 0.5 * np.einsum("uv,cug,cvg->g", D[s], dchi, dchi) for s in "ab"}
    vals = (rho["a"], rho["b"],
            np.einsum("cg,cg->g", grad["a"], grad["a"]),
            np.einsum("cg,cg->g", grad["a"], grad["b"]),
            np.einsum("cg,cg->g", grad["b"], grad["b"]),
            tau["a"], tau["b"])
    return rho, grad, tau, vals


def _deriv_arrays(names, vals):
    """Polarized derivative arrays by symbol name via the spin_kernel
    registry (name -> scalar multiset -> differentiate _F)."""
    out = {}
    for name in names:
        scs = _SYM_SCALARS.get(name)
        if scs is None:
            continue
        e = _F
        for sc in scs:
            e = sp.diff(e, _V[(sc.group, sc.comp)])
        out[name] = sp.lambdify(_ARGS, e, "numpy")(*vals) * np.ones_like(vals[0])
    return out


def energy(Da, Db, centers, pts, w):
    chi, dchi, _, _ = colloc(centers, pts)
    _, _, _, vals = _spin_fields(Da, Db, chi, dchi)
    return float(np.sum(w * sp.lambdify(_ARGS, _F, "numpy")(*vals)))


def fock_spin_matrix(spin, Da, Db, centers, pts, w):
    """F^s_uv = dE/dP^s_uv, assembled directly (the reference)."""
    chi, dchi, _, _ = colloc(centers, pts)
    _, grad, _, vals = _spin_fields(Da, Db, chi, dchi)
    d = {name: sp.lambdify(_ARGS, sp.diff(_F, v), "numpy")(*vals)
         * np.ones_like(vals[0]) for name, v in
         [("ra", _Ra), ("rb", _Rb), ("saa", _Saa), ("sab", _Sab),
          ("sbb", _Sbb), ("ta", _Ta), ("tb", _Tb)]}
    o = "ab".replace(spin, "")
    vr = d["ra"] if spin == "a" else d["rb"]
    vss = d["saa"] if spin == "a" else d["sbb"]
    vt = d["ta"] if spin == "a" else d["tb"]
    F = np.einsum("g,ug,vg->uv", w * vr, chi, chi)
    cvec = 2.0 * vss * grad[spin] + d["sab"] * grad[o]     # (3, ng)
    T = np.einsum("cg,cug,vg->uv", w * cvec, dchi, chi)
    F += T + T.T
    F += 0.5 * np.einsum("g,cug,cvg->uv", w * vt, dchi, dchi)
    return F


def _sub(name):
    if name.startswith("D_") and name.endswith("_u_v"):
        return "uv"
    if "_u_" in name or name.endswith("_u"):
        return "ug"
    if "_v_" in name or name.endswith("_v"):
        return "vg"
    return "g"


def evaluate(expr, env, out):
    total = None
    for key, coeff in from_expr(expr).items():
        subs, arrays = [], []
        for sym, e in key:
            for _ in range(e):
                subs.append(_sub(sym.name))
                arrays.append(env[sym.name])
        term = float(coeff) * np.einsum(",".join(subs) + "->" + out, *arrays)
        total = term if total is None else total + term
    return total


def main():
    rng = np.random.default_rng(11)
    natom, nbf, ng = 3, len(ALPHA), 40
    centers = rng.uniform(-0.6, 0.6, (natom, 3))
    pts = rng.uniform(-1.2, 1.2, (ng, 3))
    w = rng.uniform(0.1, 1.0, ng)
    Da = 0.1 * rng.standard_normal((nbf, nbf)) + np.eye(nbf)
    Da = 0.5 * (Da + Da.T)
    Db = 0.1 * rng.standard_normal((nbf, nbf)) + 0.8 * np.eye(nbf)
    Db = 0.5 * (Db + Db.T)
    D = {"a": Da, "b": Db}

    chi, dchi, d2chi, d3chi = colloc(centers, pts)
    rho, grad, tau, vals = _spin_fields(Da, Db, chi, dchi)

    tested = failures = 0

    # ---- first order: geometric_fock_spin vs FD of F^s ----------------------
    for spin in "ab":
        gf = geometric_fock_spin("mgga_tau", spin)
        expr = sp.expand(gf.expr)
        names = {s.name for s in expr.free_symbols}
        env = {"w": w, **_deriv_arrays(names, vals)}
        for s in "ab":
            for i, ax in enumerate("xyz"):
                env[f"grad_rho_{s}_{ax}"] = grad[s][i]
        env["chi_u"] = env["chi_v"] = chi
        for i, ax in enumerate("xyz"):
            env[f"dchi_u_{ax}"] = env[f"dchi_v_{ax}"] = dchi[i]

        A, x = 1, 2
        mA = (BF_ATOM == A).astype(float)[:, None]
        # masked displacement collocations (carry the -d/dr sign)
        env["dchi_gA_u"] = env["dchi_gA_v"] = -dchi[x] * mA
        for i, ax in enumerate("xyz"):
            env[f"ddchi_gA_u_{ax}"] = env[f"ddchi_gA_v_{ax}"] = -d2chi[x, i] * mA
        # fixed-grid perturbed fields of both channels
        for s in "ab":
            U0s, Uis = D[s] @ chi, [D[s] @ dchi[i] for i in range(3)]
            env[f"rho_{s}_p1"] = -2.0 * ((U0s * dchi[x]) * mA).sum(axis=0)
            for i, ax in enumerate("xyz"):
                env[f"grad_rho_{s}_p1_{ax}"] = -2.0 * (
                    (U0s * d2chi[x, i] + Uis[i] * dchi[x]) * mA).sum(axis=0)
            env[f"tau_{s}_p1"] = -1.0 * sum(
                (Uis[i] * d2chi[x, i] * mA).sum(axis=0) for i in range(3))

        an = evaluate(expr, env, "uv")

        h = 2e-3
        def dF(hh):
            cp, cm = centers.copy(), centers.copy()
            cp[A, x] += hh
            cm[A, x] -= hh
            return (fock_spin_matrix(spin, Da, Db, cp, pts, w)
                    - fock_spin_matrix(spin, Da, Db, cm, pts, w)) / (2 * hh)
        ref = (4 * dF(h / 2) - dF(h)) / 3
        rel = np.abs(an - ref).max() / max(np.abs(ref).max(), 1e-12)
        ok = rel < 1e-8
        tested += 1
        failures += not ok
        print(f"  [{'OK' if ok else 'FAIL'}] dF^{spin}/dX basis class: "
              f"max rel {rel:.2e}")

    # ---- second order: geometric_hessian_spin vs mixed FD of E --------------
    gh = geometric_hessian_spin("mgga_tau")
    names = {s.name for s in (gh.pair.free_symbols | gh.same.free_symbols)}
    base_env = {"w": w, "D_a_u_v": Da, "D_b_u_v": Db,
                **_deriv_arrays(names, vals)}
    for s in "ab":
        for i, ax in enumerate("xyz"):
            base_env[f"grad_rho_{s}_{ax}"] = grad[s][i]
        base_env[f"U0{s}_u"] = base_env[f"U0{s}_v"] = D[s] @ chi
        for i in range(3):
            base_env[f"U{i+1}{s}_u"] = base_env[f"U{i+1}{s}_v"] = D[s] @ dchi[i]

    def H_analytic(A, x, B, y):
        env = dict(base_env)
        mA = (BF_ATOM == A).astype(float)[:, None]
        mB = (BF_ATOM == B).astype(float)[:, None]
        env["dchi_gA_u"] = -dchi[x] * mA
        env["dchi_gB_v"] = -dchi[y] * mB
        for i, ax in enumerate("xyz"):
            env[f"ddchi_gA_u_{ax}"] = -d2chi[x, i] * mA
            env[f"ddchi_gB_v_{ax}"] = -d2chi[y, i] * mB
            env[f"d3chi_g2_u_{ax}"] = d3chi[x, y, i] * mA
        env["d2chi_g2_u"] = d2chi[x, y] * mA
        Hp = evaluate(gh.pair, env, "uv")
        val = Hp[np.ix_(BF_ATOM == A, BF_ATOM == B)].sum()
        if A == B:
            val += evaluate(gh.same, env, "u")[BF_ATOM == A].sum()
        return val

    h = 2e-3
    for (A, x, B, y) in [(0, 2, 0, 2), (0, 0, 0, 1), (0, 2, 1, 1),
                         (1, 0, 2, 2), (2, 1, 2, 1)]:
        an = H_analytic(A, x, B, y)

        def E(sa, sb, hh):
            c = centers.copy()
            c[A, x] += sa * hh
            c[B, y] += sb * hh
            return energy(Da, Db, c, pts, w)

        def fd(hh):
            if (A, x) == (B, y):
                return (E(1, 0, hh) - 2 * E(0, 0, hh) + E(-1, 0, hh)) / hh**2
            return (E(1, 1, hh) - E(1, -1, hh) - E(-1, 1, hh)
                    + E(-1, -1, hh)) / (4 * hh**2)

        ref = (4 * fd(h / 2) - fd(h)) / 3
        rel = abs(an - ref) / max(abs(ref), 1e-12)
        ok = rel < 1e-7
        tested += 1
        failures += not ok
        print(f"  [{'OK' if ok else 'FAIL'}] H^spin({A},{'xyz'[x]};{B},{'xyz'[y]}): "
              f"analytic {an:+.8e}  fd {ref:+.8e}  rel {rel:.2e}")

    status = "OK " if failures == 0 else "FAIL"
    print(f"[{status}] geometric_spin_validate: {tested} checks, "
          f"{failures} failures")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
