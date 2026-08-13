"""Validate the force-aware geometric layer (geometric.py): the three
term classes of d(Fock)/dX against finite differences with each source of
R-dependence isolated, on a synthetic set-up where everything is exact:

  1. basis class:  move atom A's basis centers, grid + weights FIXED
  2. grid class:   move atom A's grid points, centers + weights FIXED
  3. translational sum rule: basis + grid classes summed over atoms
     cancel exactly (rigid translation changes nothing; the synthetic
     weights are translation-invariant)

Basis functions are explicit s-Gaussians chi_u(r) = exp(-a|r - C_u|^2)
so every derivative is analytic; an explicit polynomial functional
stands in for Libxc (derivative arrays exact via SymPy).
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from ..emitters.codegen import compile_function, generate_collapsed
from ..engine.deriv import LIBXC_MULTISET
from ..engine.geometric import geometric_fock, spatial_gradient
from ..engine.kernel import fock

# --- the explicit test functional (mgga_tau family) ---------------------------

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


# --- explicit Gaussian collocation --------------------------------------------

ALPHA = np.array([0.9, 1.4, 0.7, 1.1, 0.8])
BF_ATOM = np.array([0, 0, 1, 1, 2])          # which atom carries each function


def colloc(centers, pts):
    """chi (nbf,ng), dchi (3,nbf,ng), d2chi (3,3,nbf,ng) for s-Gaussians."""
    d = pts[None, :, :] - centers[BF_ATOM][:, None, :]        # (nbf, ng, 3)
    chi = np.exp(-ALPHA[:, None] * np.einsum("ugc,ugc->ug", d, d))
    dchi = -2.0 * ALPHA[:, None, None] * d * chi[:, :, None]  # (nbf, ng, 3)
    dchi = np.transpose(dchi, (2, 0, 1))                      # (3, nbf, ng)
    d2 = (4.0 * ALPHA[:, None, None, None] ** 2
          * d[:, :, :, None] * d[:, :, None, :])              # (nbf,ng,3,3)
    d2 -= 2.0 * ALPHA[:, None, None, None] * np.eye(3)
    d2chi = np.transpose(d2 * chi[:, :, None, None], (2, 3, 0, 1))  # (3,3,nbf,ng)
    return chi, dchi, d2chi


def fields(D, chi, dchi):
    rho = np.einsum("uv,ug,vg->g", D, chi, chi)
    grad = 2.0 * np.einsum("uv,cug,vg->cg", D, dchi, chi)
    sigma = np.einsum("cg,cg->g", grad, grad)
    tau = 0.5 * np.einsum("uv,cug,cvg->g", D, dchi, dchi)
    return rho, grad, sigma, tau


def call(fn, gen, args):
    sig = gen.source.split("(", 1)[1].split(")", 1)[0]
    return fn(*[args[p.strip()] for p in sig.split(",")])


def main():
    rng = np.random.default_rng(3)
    natom, nbf, ng = 3, len(ALPHA), 48
    centers = rng.uniform(-0.6, 0.6, (natom, 3))
    pts = rng.uniform(-1.2, 1.2, (ng, 3))
    parent = np.arange(ng) % natom
    w = rng.uniform(0.1, 1.0, ng)
    D = 0.1 * rng.standard_normal((nbf, nbf)) + np.eye(nbf)
    D = 0.5 * (D + D.T)

    gen_f = generate_collapsed(fock("mgga_tau"), "fk")
    fn_f = compile_function(gen_f)
    gen_b = generate_collapsed(geometric_fock("mgga_tau"), "gb")
    fn_b = compile_function(gen_b)
    gen_g = generate_collapsed(spatial_gradient(fock("mgga_tau")), "gg")
    fn_g = compile_function(gen_g)

    def fock_at(cen, p):
        chi, dchi, d2chi = colloc(cen, p)
        rho, grad, sigma, tau = fields(D, chi, dchi)
        lx = _deriv_arrays(gen_f.libxc_args, rho, sigma, tau)
        return call(fn_f, gen_f, {"w": w, "chi": chi, "dchi": dchi,
                                  "grad_rho": grad, **lx})

    chi, dchi, d2chi = colloc(centers, pts)
    rho, grad, sigma, tau = fields(D, chi, dchi)

    tested = failures = 0

    def check(label, got, ref, tol):
        nonlocal tested, failures
        tested += 1
        scale = max(np.abs(ref).max(), np.abs(got).max(), 1e-14)
        rel = np.abs(got - ref).max() / scale
        ok = rel < tol
        if not ok:
            failures += 1
        print(f"  [{'OK' if ok else 'FAIL'}] {label}: max rel {rel:.2e}")
        return got

    h = 2e-5
    for A in range(natom):
        for d in range(3):
            # ===== basis class =====
            mask = (BF_ATOM == A).astype(float)
            dchi_gA = -dchi[d] * mask[:, None]
            ddchi_gA = -d2chi[d] * mask[None, :, None]
            # fixed-grid perturbed fields (atom-restricted)
            rho_p1 = -2.0 * np.einsum("uv,ug,vg->g", D, dchi[d] * mask[:, None], chi)
            grad_p1 = -2.0 * (np.einsum("uv,iug,vg->ig", D, d2chi[d] * mask[None, :, None], chi)
                              + np.einsum("uv,ug,ivg->ig", D, dchi[d] * mask[:, None], dchi))
            tau_p1 = -1.0 * np.einsum("uv,iug,ivg->g", D, d2chi[d] * mask[None, :, None], dchi)
            lx = _deriv_arrays(gen_b.libxc_args, rho, sigma, tau)
            Fb = call(fn_b, gen_b, {"w": w, "chi": chi, "dchi": dchi,
                                    "dchi_gA": dchi_gA, "ddchi_gA": ddchi_gA,
                                    "grad_rho": grad, "rho_p1": rho_p1,
                                    "grad_rho_p1": grad_p1, "tau_p1": tau_p1, **lx})
            cp, cm = centers.copy(), centers.copy()
            cp[A, d] += h
            cm[A, d] -= h
            fd = (fock_at(cp, pts) - fock_at(cm, pts)) / (2 * h)
            check(f"basis class A={A} d={d}", Fb, fd, 5e-9)

            # ===== grid class =====
            Mw = w * (parent == A)
            dgrad = 2.0 * (np.einsum("uv,iug,vg->ig", D, d2chi[d], chi)
                           + np.einsum("uv,ug,ivg->ig", D, dchi[d], dchi))
            dtau = np.einsum("uv,iug,ivg->g", D, d2chi[d], dchi)
            lxg = _deriv_arrays(gen_g.libxc_args, rho, sigma, tau)
            Fg = call(fn_g, gen_g, {"w": Mw, "chi": chi, "dchi": dchi,
                                    "dchi_g": dchi[d], "ddchi_g": d2chi[d],
                                    "grad_rho": grad, "drho_g": grad[d],
                                    "dgrad_rho_g": dgrad, "dtau_g": dtau, **lxg})
            pp, pm = pts.copy(), pts.copy()
            pp[parent == A, d] += h
            pm[parent == A, d] -= h
            fd = (fock_at(centers, pp) - fock_at(centers, pm)) / (2 * h)
            check(f"grid class  A={A} d={d}", Fg, fd, 5e-9)
    # ===== translational sum rule (x direction) =====
    bs = np.zeros((nbf, nbf))
    gs = np.zeros((nbf, nbf))
    for A in range(natom):
        mask = (BF_ATOM == A).astype(float)
        rho_p1 = -2.0 * np.einsum("uv,ug,vg->g", D, dchi[0] * mask[:, None], chi)
        grad_p1 = -2.0 * (np.einsum("uv,iug,vg->ig", D, d2chi[0] * mask[None, :, None], chi)
                          + np.einsum("uv,ug,ivg->ig", D, dchi[0] * mask[:, None], dchi))
        tau_p1 = -1.0 * np.einsum("uv,iug,ivg->g", D, d2chi[0] * mask[None, :, None], dchi)
        lx = _deriv_arrays(gen_b.libxc_args, rho, sigma, tau)
        bs += call(fn_b, gen_b, {"w": w, "chi": chi, "dchi": dchi,
                                 "dchi_gA": -dchi[0] * mask[:, None],
                                 "ddchi_gA": -d2chi[0] * mask[None, :, None],
                                 "grad_rho": grad, "rho_p1": rho_p1,
                                 "grad_rho_p1": grad_p1, "tau_p1": tau_p1, **lx})
        dgrad = 2.0 * (np.einsum("uv,iug,vg->ig", D, d2chi[0], chi)
                       + np.einsum("uv,ug,ivg->ig", D, dchi[0], dchi))
        dtau = np.einsum("uv,iug,ivg->g", D, d2chi[0], dchi)
        lxg = _deriv_arrays(gen_g.libxc_args, rho, sigma, tau)
        gs += call(fn_g, gen_g, {"w": w * (parent == A), "chi": chi,
                                 "dchi": dchi, "dchi_g": dchi[0],
                                 "ddchi_g": d2chi[0], "grad_rho": grad,
                                 "drho_g": grad[0], "dgrad_rho_g": dgrad,
                                 "dtau_g": dtau, **lxg})
    scale = max(np.abs(bs).max(), np.abs(gs).max())
    rel = np.abs(bs + gs).max() / scale
    tested += 1
    ok = rel < 1e-13
    if not ok:
        failures += 1
    print(f"  [{'OK' if ok else 'FAIL'}] translational sum rule: "
          f"max |basis + grid| / scale = {rel:.2e}")

    status = "OK " if failures == 0 else "FAIL"
    print(f"[{status}] geometric_validate: {tested} checks, {failures} failures")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
