"""Validate the second-order geometric operator (geometric_hessian): the
explicit fixed-grid XC Hessian as a function-pair kernel, against mixed
finite differences of the energy with grid and weights FROZEN while the
basis centers move.

Explicit s-Gaussian collocation (derivatives analytic through third
order), explicit polynomial functional (derivative arrays exact)."""

from __future__ import annotations

import numpy as np
import sympy as sp

from ..engine.deriv import LIBXC_MULTISET
from ..engine.fastpoly import from_expr
from ..engine.geometric import geometric_hessian

# --- explicit functional (mgga_tau) -------------------------------------------

_r, _s, _t = sp.symbols("rho sigma tau", positive=True)
_F = (_r**2 + sp.Rational(3, 10) * _s * _r + sp.Rational(1, 5) * _t**2
      + sp.Rational(1, 10) * _r * _t + sp.Rational(1, 20) * _s * _t)
_VAR = {"rho": _r, "sigma": _s, "tau": _t}


def _deriv_arrays(names, rho, sigma, tau):
    out = {}
    for name in names:
        ms = LIBXC_MULTISET[name]
        e = _F
        for var, cnt in ms.items():
            e = sp.diff(e, _VAR[var], cnt)
        out[name] = sp.lambdify((_r, _s, _t), e, "numpy")(
            rho, sigma, tau) * np.ones_like(rho)
    return out


# --- explicit Gaussian collocation through third derivatives ------------------

ALPHA = np.array([0.9, 1.4, 0.7, 1.1, 0.8])
BF_ATOM = np.array([0, 0, 1, 1, 2])


def colloc(centers, pts):
    d = pts[None, :, :] - centers[BF_ATOM][:, None, :]        # (nbf, ng, 3)
    al = ALPHA[:, None]
    chi = np.exp(-al * np.einsum("ugc,ugc->ug", d, d))
    dchi = np.transpose(-2.0 * al[..., None] * d * chi[..., None], (2, 0, 1))
    eye = np.eye(3)
    d2 = 4.0 * al[..., None, None] ** 2 * d[:, :, :, None] * d[:, :, None, :] \
        - 2.0 * al[..., None, None] * eye
    d2chi = np.transpose(d2 * chi[..., None, None], (2, 3, 0, 1))
    # third derivatives: [4 a^2 (delta_ik d_j + delta_jk d_i + delta_ij d_k)
    #                     - 8 a^3 d_i d_j d_k] chi
    d3 = (4.0 * al[..., None, None, None] ** 2
          * (eye[None, None, :, :, None] * d[:, :, None, None, :]
             + eye[None, None, :, None, :] * d[:, :, None, :, None]
             + eye[None, None, None, :, :] * d[:, :, :, None, None])
          - 8.0 * al[..., None, None, None] ** 3
          * d[:, :, :, None, None] * d[:, :, None, :, None] * d[:, :, None, None, :])
    d3chi = np.transpose(d3 * chi[..., None, None, None], (2, 3, 4, 0, 1))
    return chi, dchi, d2chi, d3chi


def fields(D, chi, dchi):
    rho = np.einsum("uv,ug,vg->g", D, chi, chi)
    grad = 2.0 * np.einsum("uv,cug,vg->cg", D, dchi, chi)
    sigma = np.einsum("cg,cg->g", grad, grad)
    tau = 0.5 * np.einsum("uv,cug,cvg->g", D, dchi, dchi)
    return rho, grad, sigma, tau


def energy(D, centers, pts, w):
    chi, dchi, _, _ = colloc(centers, pts)
    rho, grad, sigma, tau = fields(D, chi, dchi)
    return float(np.sum(w * sp.lambdify((_r, _s, _t), _F, "numpy")(rho, sigma, tau)))


def _sub(name):
    if name == "D_u_v":
        return "uv", None
    if "_u_" in name or name.endswith("_u"):
        return "ug", None
    if "_v_" in name or name.endswith("_v"):
        return "vg", None
    return "g", None


def evaluate(expr, env, out):
    """Evaluate a monomial expression to an (nbf,nbf) pair matrix or an
    (nbf,) same-atom vector."""
    total = None
    for key, coeff in from_expr(expr).items():
        subs, arrays = [], []
        for sym, e in key:
            for _ in range(e):
                sub, _n = _sub(sym.name)
                subs.append(sub)
                arrays.append(env[sym.name])
        term = float(coeff) * np.einsum(",".join(subs) + "->" + out, *arrays)
        total = term if total is None else total + term
    return total


def main():
    rng = np.random.default_rng(5)
    natom, nbf, ng = 3, len(ALPHA), 40
    centers = rng.uniform(-0.6, 0.6, (natom, 3))
    pts = rng.uniform(-1.2, 1.2, (ng, 3))
    w = rng.uniform(0.1, 1.0, ng)
    D = 0.1 * rng.standard_normal((nbf, nbf)) + np.eye(nbf)
    D = 0.5 * (D + D.T)

    gh = geometric_hessian("mgga_tau")
    chi, dchi, d2chi, d3chi = colloc(centers, pts)
    rho, grad, sigma, tau = fields(D, chi, dchi)
    names = {s.name for s in (gh.pair.free_symbols | gh.same.free_symbols)}
    lx = _deriv_arrays([n for n in names if n in LIBXC_MULTISET], rho, sigma, tau)

    base_env = {"w": w, "D_u_v": D, **lx}
    for i, ax in enumerate("xyz"):
        base_env[f"grad_rho_{ax}"] = grad[i]
    U0 = D @ chi.reshape(nbf, ng)
    base_env["U0_u"] = base_env["U0_v"] = U0
    for i in range(3):
        Ui = D @ dchi[i]
        base_env[f"U{i+1}_u"] = base_env[f"U{i+1}_v"] = Ui

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
            Sv = evaluate(gh.same, env, "u")
            val += Sv[BF_ATOM == A].sum()
        return val

    tested = failures = 0
    h = 2e-3
    for (A, x, B, y) in [(0, 2, 0, 2), (0, 0, 0, 1), (0, 2, 1, 1),
                         (1, 0, 2, 2), (2, 1, 2, 1)]:
        an = H_analytic(A, x, B, y)

        def E(sa, sb, hh):
            c = centers.copy()
            c[A, x] += sa * hh
            c[B, y] += sb * hh
            return energy(D, c, pts, w)

        def fd(hh):
            if (A, x) == (B, y):
                return (E(1, 0, hh) - 2 * E(0, 0, hh) + E(-1, 0, hh)) / hh**2
            return (E(1, 1, hh) - E(1, -1, hh) - E(-1, 1, hh) + E(-1, -1, hh)) / (4 * hh**2)

        ref = (4 * fd(h / 2) - fd(h)) / 3
        rel = abs(an - ref) / max(abs(ref), 1e-12)
        ok = rel < 1e-7
        tested += 1
        failures += not ok
        print(f"  [{'OK' if ok else 'FAIL'}] H({A},{'xyz'[x]};{B},{'xyz'[y]}): "
              f"analytic {an:+.8e}  fd {ref:+.8e}  rel {rel:.2e}")

    status = "OK " if failures == 0 else "FAIL"
    print(f"[{status}] geometric2_validate: {tested} checks, {failures} failures")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
