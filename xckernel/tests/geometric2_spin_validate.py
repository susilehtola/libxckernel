"""Validate the spin-polarized second-order spatial operators against
finite differences over grid-point motion (basis centers, densities, and
weights all frozen; the evaluation point moves):

* spatial_energy_hessian_spin -- d_g d_h of the polarized energy density
  against mixed FD of e(r) per point, and
* spatial_row_gradient_spin -- d_h of the polarized basis-class energy
  rows sum_K v_K F_K(u) against FD of the row itself.

Explicit s-Gaussian collocation (analytic through third derivatives),
explicit polarized polynomial functional coupling all channels."""

from __future__ import annotations

import numpy as np
import sympy as sp

from ..engine.geometric import (_geo_rows_spin, spatial_energy_hessian_spin,
                         spatial_gradient_spin, spatial_row_gradient_spin)
from ..engine.spin import family_scalars
from ..engine.spin_kernel import _register, fock_spin
from .geometric2_validate import ALPHA, colloc
from .geometric_spin_validate import (_ARGS, _F, _deriv_arrays, _spin_fields,
                                      evaluate)

_EDENS = sp.lambdify(_ARGS, _F, "numpy")


def _spatial_env(D, chi, dchi, d2chi, d3chi, slot, d):
    """Per-channel spatial field derivatives in direction d, bound to the
    slot's operand names (drho_a_<slot>, dgrad_rho_a_<slot>_i, ...)."""
    env = {}
    for s in "ab":
        Ds = D[s]
        env[f"drho_{s}_{slot}"] = 2.0 * np.einsum(
            "uv,ug,vg->g", Ds, dchi[d], chi)
        for i, ax in enumerate("xyz"):
            env[f"dgrad_rho_{s}_{slot}_{ax}"] = 2.0 * (
                np.einsum("uv,ug,vg->g", Ds, d2chi[d, i], chi)
                + np.einsum("uv,ug,vg->g", Ds, dchi[i], dchi[d]))
        env[f"dtau_{s}_{slot}"] = sum(
            np.einsum("uv,ug,vg->g", Ds, dchi[i], d2chi[d, i])
            for i in range(3))
    return env


def _pair_env(D, chi, dchi, d2chi, d3chi, xd, yd):
    """The gh pair operands: d_g d_h of the per-channel fields."""
    env = {}
    for s in "ab":
        Ds = D[s]
        env[f"d2rho_{s}_gh"] = 2.0 * (
            np.einsum("uv,ug,vg->g", Ds, d2chi[xd, yd], chi)
            + np.einsum("uv,ug,vg->g", Ds, dchi[xd], dchi[yd]))
        for i, ax in enumerate("xyz"):
            env[f"d2grad_rho_{s}_gh_{ax}"] = 2.0 * (
                np.einsum("uv,ug,vg->g", Ds, d3chi[xd, yd, i], chi)
                + np.einsum("uv,ug,vg->g", Ds, d2chi[xd, i], dchi[yd])
                + np.einsum("uv,ug,vg->g", Ds, d2chi[yd, i], dchi[xd])
                + np.einsum("uv,ug,vg->g", Ds, dchi[i], d2chi[xd, yd]))
        env[f"d2tau_{s}_gh"] = sum(
            np.einsum("uv,ug,vg->g", Ds, d2chi[xd, i], d2chi[yd, i])
            + np.einsum("uv,ug,vg->g", Ds, dchi[i], d3chi[xd, yd, i])
            for i in range(3))
    return env


def main():
    rng = np.random.default_rng(23)
    natom, nbf, ng = 3, len(ALPHA), 40
    centers = rng.uniform(-0.6, 0.6, (natom, 3))
    pts = rng.uniform(-1.2, 1.2, (ng, 3))
    Da = 0.1 * rng.standard_normal((nbf, nbf)) + np.eye(nbf)
    Da = 0.5 * (Da + Da.T)
    Db = 0.1 * rng.standard_normal((nbf, nbf)) + 0.8 * np.eye(nbf)
    Db = 0.5 * (Db + Db.T)
    D = {"a": Da, "b": Db}

    chi, dchi, d2chi, d3chi = colloc(centers, pts)
    rho, grad, tau, vals = _spin_fields(Da, Db, chi, dchi)

    tested = failures = 0

    # ---- d2e: spatial_energy_hessian_spin vs mixed FD of e(r) ---------------
    expr = spatial_energy_hessian_spin("mgga_tau")
    names = {s.name for s in expr.free_symbols}

    def edens(pts_):
        chi_, dchi_, _, _ = colloc(centers, pts_)
        _, _, _, vals_ = _spin_fields(Da, Db, chi_, dchi_)
        return _EDENS(*vals_)

    h = 2e-3
    for (xd, yd) in [(2, 2), (0, 1), (1, 2)]:
        env = {**_deriv_arrays(names, vals),
               **_spatial_env(D, chi, dchi, d2chi, d3chi, "g", xd),
               **_spatial_env(D, chi, dchi, d2chi, d3chi, "h", yd),
               **_pair_env(D, chi, dchi, d2chi, d3chi, xd, yd)}
        for s in "ab":
            for i, ax in enumerate("xyz"):
                env[f"grad_rho_{s}_{ax}"] = grad[s][i]
        an = evaluate(expr, env, "g")

        def fd(hh):
            ex = np.zeros(3)
            ey = np.zeros(3)
            ex[xd] = hh
            ey[yd] = hh
            if xd == yd:
                return (edens(pts + ex) - 2 * edens(pts)
                        + edens(pts - ex)) / hh**2
            return (edens(pts + ex + ey) - edens(pts + ex - ey)
                    - edens(pts - ex + ey) + edens(pts - ex - ey)) / (4 * hh**2)

        ref = (4 * fd(h / 2) - fd(h)) / 3
        rel = np.abs(an - ref).max() / max(np.abs(ref).max(), 1e-12)
        ok = rel < 1e-7
        tested += 1
        failures += not ok
        print(f"  [{'OK' if ok else 'FAIL'}] d2e({'xyz'[xd]},{'xyz'[yd]}): "
              f"max rel {rel:.2e}")

    # ---- mb: spatial_row_gradient_spin vs FD of the row ----------------------
    scalars = family_scalars("mgga_tau")
    rows, _, _ = _geo_rows_spin("u", "A", "mgga_tau")
    base = sp.expand(sum(_register((K,)) * rows[K] for K in scalars))
    base_names = {s.name for s in base.free_symbols}
    mb_expr = spatial_row_gradient_spin("mgga_tau")
    mb_names = {s.name for s in mb_expr.free_symbols}

    def row_at(pts_, xd):
        chi_, dchi_, d2chi_, _ = colloc(centers, pts_)
        _, grad_, _, vals_ = _spin_fields(Da, Db, chi_, dchi_)
        env = _deriv_arrays(base_names, vals_)
        for s in "ab":
            for i, ax in enumerate("xyz"):
                env[f"grad_rho_{s}_{ax}"] = grad_[s][i]
            env[f"U0{s}_u"] = D[s] @ chi_
            for i in range(3):
                env[f"U{i+1}{s}_u"] = D[s] @ dchi_[i]
        env["dchi_gA_u"] = -dchi_[xd]
        for i, ax in enumerate("xyz"):
            env[f"ddchi_gA_u_{ax}"] = -d2chi_[xd, i]
        return evaluate(base, env, "ug")

    for (xd, yd) in [(2, 2), (0, 1)]:
        env = _deriv_arrays(mb_names, vals)
        for s in "ab":
            for i, ax in enumerate("xyz"):
                env[f"grad_rho_{s}_{ax}"] = grad[s][i]
            env[f"U0{s}_u"] = D[s] @ chi
            for i in range(3):
                env[f"U{i+1}{s}_u"] = D[s] @ dchi[i]
            env[f"Uh0{s}_u"] = D[s] @ dchi[yd]
            for i, ax in enumerate("xyz"):
                env[f"Uhess{s}_u_{ax}"] = D[s] @ d2chi[yd, i]
        env["dchi_gA_u"] = -dchi[xd]
        env["ddchi_ghA_u"] = -d2chi[xd, yd]
        for i, ax in enumerate("xyz"):
            env[f"ddchi_gA_u_{ax}"] = -d2chi[xd, i]
            env[f"dddchi_ghA_u_{ax}"] = -d3chi[xd, yd, i]
        env.update(_spatial_env(D, chi, dchi, d2chi, d3chi, "h", yd))
        an = evaluate(mb_expr, env, "ug")

        def dR(hh):
            ey = np.zeros(3)
            ey[yd] = hh
            return (row_at(pts + ey, xd) - row_at(pts - ey, xd)) / (2 * hh)

        ref = (4 * dR(h / 2) - dR(h)) / 3
        rel = np.abs(an - ref).max() / max(np.abs(ref).max(), 1e-12)
        ok = rel < 1e-8
        tested += 1
        failures += not ok
        print(f"  [{'OK' if ok else 'FAIL'}] mb(x={'xyz'[xd]},h={'xyz'[yd]}): "
              f"max rel {rel:.2e}")

    # ---- spatial_gradient_spin: d_d of the spin Fock integrand vs FD --------
    def fock_env(pts_, extra=None):
        chi_, dchi_, d2chi_, _ = colloc(centers, pts_)
        _, grad_, _, vals_ = _spin_fields(Da, Db, chi_, dchi_)
        env = {"w": np.ones(pts_.shape[0])}
        env.update(_deriv_arrays(all_names, vals_))
        for s in "ab":
            for i, ax in enumerate("xyz"):
                env[f"grad_rho_{s}_{ax}"] = grad_[s][i]
        env["chi_u"] = env["chi_v"] = chi_
        for i, ax in enumerate("xyz"):
            env[f"dchi_u_{ax}"] = env[f"dchi_v_{ax}"] = dchi_[i]
        return env, chi_, dchi_, d2chi_

    for spin in "ab":
        fi = fock_spin("mgga_tau", spin)
        sg = spatial_gradient_spin("mgga_tau", spin)
        all_names = {s.name for s in (fi.expr.free_symbols
                                      | sg.expr.free_symbols)}
        xd = 1 if spin == "a" else 2
        env, chi_c, dchi_c, d2chi_c = fock_env(pts)
        for lbl in ("u", "v"):
            env[f"dchi_g_{lbl}"] = dchi_c[xd]
            for i, ax in enumerate("xyz"):
                env[f"ddchi_g_{lbl}_{ax}"] = d2chi_c[xd, i]
        env.update(_spatial_env(D, chi_c, dchi_c, d2chi_c, None, "g", xd))
        an = evaluate(sg.expr, env, "uvg")

        def dF(hh):
            ex = np.zeros(3)
            ex[xd] = hh
            ep, _, _, _ = fock_env(pts + ex)
            em, _, _, _ = fock_env(pts - ex)
            return (evaluate(fi.expr, ep, "uvg")
                    - evaluate(fi.expr, em, "uvg")) / (2 * hh)

        ref = (4 * dF(h / 2) - dF(h)) / 3
        rel = np.abs(an - ref).max() / max(np.abs(ref).max(), 1e-12)
        ok = rel < 1e-8
        tested += 1
        failures += not ok
        print(f"  [{'OK' if ok else 'FAIL'}] d_{'xyz'[xd]} f^{spin}_uv "
              f"(spatial_gradient_spin): max rel {rel:.2e}")

    status = "OK " if failures == 0 else "FAIL"
    print(f"[{status}] geometric2_spin_validate: {tested} checks, "
          f"{failures} failures")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
