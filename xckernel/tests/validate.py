"""Numerical validation of the AD Fock matrix against finite differences.

We fabricate a small LCAO system: random grid values of each basis function and
its derivatives at ``npts`` grid points, random grid weights, and a random
*symmetric* density matrix P.  From these we can:

  * compute the Libxc ingredients (rho, sigma, lapl, tau) as bilinear forms in P,
  * call pylibxc to get the energy density and its derivatives (vrho, ...),
  * assemble Exc = sum_g w_g rho_g zk_g,
  * evaluate our analytic F_uv = dExc/dP_uv (the AD integrand, contracted over g),
  * and independently estimate dExc/dP_uv by central finite differences.

Agreement to ~1e-6 confirms the AD backend computes the exact Fock matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import sympy as sp
from pylibxc import LibXCFunctional

from ..basis import AXES
from ..fock import fock_integrand

# A representative functional for each family, chosen for which variables it
# genuinely consumes.
FAMILY_FUNCTIONAL = {
    "lda": "LDA_X",
    "gga": "GGA_X_PBE",
    "mgga_tau": "MGGA_X_SCAN",   # uses tau, not lapl
    "mgga": "MGGA_X_BR89",       # uses the density Laplacian
}

FAMILY_VARS = {
    "lda": ["rho"],
    "gga": ["rho", "sigma"],
    "mgga_tau": ["rho", "sigma", "tau"],
    "mgga": ["rho", "sigma", "lapl", "tau"],
}


@dataclass
class Grid:
    """Fabricated per-(basis, grid) data. Arrays are shape (nbf, npts) unless
    noted; gradients are (nbf, 3, npts)."""

    w: np.ndarray            # (npts,)
    chi: np.ndarray          # (nbf, npts)
    dchi: np.ndarray         # (nbf, 3, npts)
    lapl_chi: np.ndarray     # (nbf, npts)
    P: np.ndarray            # (nbf, nbf) symmetric


def make_grid(nbf: int = 4, npts: int = 200, seed: int = 1) -> Grid:
    rng = np.random.default_rng(seed)
    w = rng.uniform(0.1, 1.0, npts)
    chi = rng.standard_normal((nbf, npts))
    dchi = rng.standard_normal((nbf, 3, npts))
    lapl_chi = rng.standard_normal((nbf, npts))
    A = rng.standard_normal((nbf, nbf))
    P = A @ A.T                     # symmetric positive-semidefinite
    return Grid(w=w, chi=chi, dchi=dchi, lapl_chi=lapl_chi, P=P)


def ingredients_from_P(g: Grid, P: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute the Libxc input variables at every grid point from P."""
    # rho_g = sum_uv P_uv chi_u chi_v
    rho = np.einsum("uv,ug,vg->g", P, g.chi, g.chi)
    # grad rho_g,i = sum_uv P_uv (dchi_u,i chi_v + chi_u dchi_v,i)
    grad = np.einsum("uv,uig,vg->ig", P, g.dchi, g.chi) \
        + np.einsum("uv,ug,vig->ig", P, g.chi, g.dchi)
    sigma = np.einsum("ig,ig->g", grad, grad)
    # tau_g = 1/2 sum_uv P_uv dchi_u . dchi_v
    tau = 0.5 * np.einsum("uv,uig,vig->g", P, g.dchi, g.dchi)
    # lapl rho = sum_uv P_uv (lapl_u chi_v + 2 dchi_u.dchi_v + chi_u lapl_v)
    lapl = np.einsum("uv,ug,vg->g", P, g.lapl_chi, g.chi) \
        + 2.0 * np.einsum("uv,uig,vig->g", P, g.dchi, g.dchi) \
        + np.einsum("uv,ug,vg->g", P, g.chi, g.lapl_chi)
    return {"rho": rho, "sigma": sigma, "tau": tau, "lapl": lapl, "_grad": grad}


def libxc_eval(name: str, family: str, ing: Dict[str, np.ndarray],
               do_fxc: bool = False):
    func = LibXCFunctional(name, "unpolarized")
    inp = {v: ing[v] for v in FAMILY_VARS[family]}
    return func.compute(inp, do_fxc=do_fxc)


def exc_of_P(name: str, family: str, g: Grid, P: np.ndarray) -> float:
    ing = ingredients_from_P(g, P)
    out = libxc_eval(name, family, ing)
    zk = out["zk"].reshape(-1)
    return float(np.sum(g.w * ing["rho"] * zk))


def analytic_fock(family: str, name: str, g: Grid) -> np.ndarray:
    """Contract our AD integrand over the grid into a full F_uv matrix."""
    fi = fock_integrand(family)
    ing = ingredients_from_P(g, g.P)
    out = libxc_eval(name, family, ing)

    # Per-grid field values shared by every (u, v) pair.
    field = {
        "w": g.w,
        "grad_rho_x": ing["_grad"][0],
        "grad_rho_y": ing["_grad"][1],
        "grad_rho_z": ing["_grad"][2],
        "vrho": out["vrho"].reshape(-1),
    }
    for var, vname in (("sigma", "vsigma"), ("lapl", "vlapl"), ("tau", "vtau")):
        if vname in out and out[vname] is not None:
            field[vname] = out[vname].reshape(-1)

    symbols = sorted(fi.expr.free_symbols, key=lambda s: s.name)
    fn = sp.lambdify(symbols, fi.expr, "numpy")

    nbf, npts = g.chi.shape
    F = np.zeros((nbf, nbf))
    npts_ones = np.ones(npts)
    for u in range(nbf):
        for v in range(nbf):
            env = dict(field)
            env["chi_u"], env["chi_v"] = g.chi[u], g.chi[v]
            env["lapl_chi_u"], env["lapl_chi_v"] = g.lapl_chi[u], g.lapl_chi[v]
            for i, ax in enumerate(AXES):
                env[f"dchi_u_{ax}"] = g.dchi[u, i]
                env[f"dchi_v_{ax}"] = g.dchi[v, i]
            args = [np.broadcast_to(env[s.name], (npts,)) * npts_ones
                    for s in symbols]
            F[u, v] = float(np.sum(fn(*args)))
    return F


def fd_fock(family: str, name: str, g: Grid, h: float = 1e-6) -> np.ndarray:
    """Central-difference dExc/dP_uv, perturbing single entries independently."""
    nbf = g.P.shape[0]
    F = np.zeros((nbf, nbf))
    for u in range(nbf):
        for v in range(nbf):
            Pp = g.P.copy(); Pp[u, v] += h
            Pm = g.P.copy(); Pm[u, v] -= h
            F[u, v] = (exc_of_P(name, family, g, Pp)
                       - exc_of_P(name, family, g, Pm)) / (2 * h)
    return F


def check(family: str) -> Dict[str, float]:
    name = FAMILY_FUNCTIONAL[family]
    g = make_grid()
    Fa = analytic_fock(family, name, g)
    Fd = fd_fock(family, name, g)
    err = float(np.max(np.abs(Fa - Fd)))
    scale = float(np.max(np.abs(Fd))) or 1.0
    return {"family": family, "functional": name,
            "max_abs_err": err, "max_rel_err": err / scale}


# --- second derivative: AO-basis XC kernel g_uv,ts -------------------------

def analytic_kernel(family: str, name: str, g: Grid) -> np.ndarray:
    """Contract the AD kernel integrand into a full g[u,v,t,s] tensor."""
    from ..kernel import xc_kernel
    ki = xc_kernel(family)
    ing = ingredients_from_P(g, g.P)
    out = libxc_eval(name, family, ing, do_fxc=True)

    field = {"w": g.w,
             "grad_rho_x": ing["_grad"][0],
             "grad_rho_y": ing["_grad"][1],
             "grad_rho_z": ing["_grad"][2]}
    # every Libxc first/second derivative array present in the output
    for key, arr in out.items():
        if key in ("zk",) or arr is None:
            continue
        if key.startswith("v"):
            field[key] = arr.reshape(-1)

    symbols = sorted(ki.expr.free_symbols, key=lambda s: s.name)
    fn = sp.lambdify(symbols, ki.expr, "numpy")

    nbf, npts = g.chi.shape
    ones = np.ones(npts)

    def orb_env(label: str, idx: int) -> Dict[str, np.ndarray]:
        e = {f"chi_{label}": g.chi[idx], f"lapl_chi_{label}": g.lapl_chi[idx]}
        for i, ax in enumerate(AXES):
            e[f"dchi_{label}_{ax}"] = g.dchi[idx, i]
        return e

    G = np.zeros((nbf, nbf, nbf, nbf))
    for u in range(nbf):
        for v in range(nbf):
            for t in range(nbf):
                for s in range(nbf):
                    env = dict(field)
                    env.update(orb_env("u", u)); env.update(orb_env("v", v))
                    env.update(orb_env("t", t)); env.update(orb_env("s", s))
                    args = [np.broadcast_to(env[sym.name], (npts,)) * ones
                            for sym in symbols]
                    G[u, v, t, s] = float(np.sum(fn(*args)))
    return G


def fd_kernel(family: str, name: str, g: Grid, h: float = 2e-4) -> np.ndarray:
    """Mixed central 2nd difference d2 Exc / dP_uv dP_ts (independent entries)."""
    nbf = g.P.shape[0]
    G = np.zeros((nbf, nbf, nbf, nbf))

    def E(dp):
        return exc_of_P(name, family, g, g.P + dp)

    for u in range(nbf):
        for v in range(nbf):
            for t in range(nbf):
                for s in range(nbf):
                    dpp = np.zeros_like(g.P); dpp[u, v] += h; dpp[t, s] += h
                    dpm = np.zeros_like(g.P); dpm[u, v] += h; dpm[t, s] -= h
                    dmp = np.zeros_like(g.P); dmp[u, v] -= h; dmp[t, s] += h
                    dmm = np.zeros_like(g.P); dmm[u, v] -= h; dmm[t, s] -= h
                    G[u, v, t, s] = (E(dpp) - E(dpm) - E(dmp) + E(dmm)) / (4 * h * h)
    return G


def check_kernel(family: str) -> Dict[str, float]:
    name = FAMILY_FUNCTIONAL[family]
    g = make_grid(nbf=3, npts=120, seed=2)
    Ga = analytic_kernel(family, name, g)
    Gd = fd_kernel(family, name, g)
    err = float(np.max(np.abs(Ga - Gd)))
    scale = float(np.max(np.abs(Gd))) or 1.0
    return {"family": family, "functional": name,
            "max_abs_err": err, "max_rel_err": err / scale}


if __name__ == "__main__":
    print("Fock  F_uv = dExc/dP_uv")
    for fam in ("lda", "gga", "mgga_tau", "mgga"):
        r = check(fam)
        ok = "OK " if r["max_rel_err"] < 1e-5 else "FAIL"
        print(f"  [{ok}] {fam:9s} {r['functional']:14s} "
              f"max_abs_err={r['max_abs_err']:.3e} "
              f"max_rel_err={r['max_rel_err']:.3e}")

    print("Kernel g_uv,ts = d2Exc/dP_uv dP_ts")
    for fam in ("lda", "gga", "mgga_tau", "mgga"):
        r = check_kernel(fam)
        ok = "OK " if r["max_rel_err"] < 1e-3 else "FAIL"
        print(f"  [{ok}] {fam:9s} {r['functional']:14s} "
              f"max_abs_err={r['max_abs_err']:.3e} "
              f"max_rel_err={r['max_rel_err']:.3e}")
