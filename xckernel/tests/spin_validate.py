"""Validate the spin-polarized Fock and kernel vs pylibxc + finite differences.

Same fabricated-grid strategy as tests/validate.py, but with two spin density
matrices P_a, P_b and Libxc in 'polarized' mode.  We check

    F^a_uv       = dExc/dP^a_uv
    g^{ab}_uv,ts = dExc/dP^a_uv dP^b_ts   (a representative spin-off-diagonal)

against finite differences of the polarized Libxc energy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import sympy as sp
from pylibxc import LibXCFunctional

from ..inputs.basis import AXES
from ..engine.spin_kernel import fock_spin, kernel_spin

FAMILY_FUNCTIONAL = {"lda": "LDA_X", "gga": "GGA_X_PBE",
                     "mgga_tau": "MGGA_X_SCAN", "mgga": "MGGA_X_BR89"}
FAMILY_VARS = {"lda": ["rho"], "gga": ["rho", "sigma"],
               "mgga_tau": ["rho", "sigma", "tau"],
               "mgga": ["rho", "sigma", "lapl", "tau"]}


@dataclass
class Grid:
    w: np.ndarray
    chi: np.ndarray          # (nbf, npts)
    dchi: np.ndarray         # (nbf, 3, npts)
    lapl_chi: np.ndarray     # (nbf, npts)
    Pa: np.ndarray
    Pb: np.ndarray


def make_grid(nbf=3, npts=150, seed=3) -> Grid:
    rng = np.random.default_rng(seed)
    w = rng.uniform(0.1, 1.0, npts)
    chi = rng.standard_normal((nbf, npts))
    dchi = rng.standard_normal((nbf, 3, npts))
    lapl_chi = rng.standard_normal((nbf, npts))
    A = rng.standard_normal((nbf, nbf)); B = rng.standard_normal((nbf, nbf))
    return Grid(w, chi, dchi, lapl_chi, A @ A.T, B @ B.T)


def _channel(g: Grid, P):
    rho = np.einsum("uv,ug,vg->g", P, g.chi, g.chi)
    grad = np.einsum("uv,uig,vg->ig", P, g.dchi, g.chi) \
        + np.einsum("uv,ug,vig->ig", P, g.chi, g.dchi)
    tau = 0.5 * np.einsum("uv,uig,vig->g", P, g.dchi, g.dchi)
    lapl = np.einsum("uv,ug,vg->g", P, g.lapl_chi, g.chi) \
        + 2 * np.einsum("uv,uig,vig->g", P, g.dchi, g.dchi) \
        + np.einsum("uv,ug,vg->g", P, g.chi, g.lapl_chi)
    return rho, grad, tau, lapl


def ingredients(g: Grid, Pa, Pb) -> Dict[str, np.ndarray]:
    ra, ga, ta, la = _channel(g, Pa)
    rb, gb, tb, lb = _channel(g, Pb)
    return {
        "rho": np.column_stack([ra, rb]),
        "sigma": np.column_stack([np.einsum("ig,ig->g", ga, ga),
                                  np.einsum("ig,ig->g", ga, gb),
                                  np.einsum("ig,ig->g", gb, gb)]),
        "lapl": np.column_stack([la, lb]),
        "tau": np.column_stack([ta, tb]),
        "_grad_a": ga, "_grad_b": gb, "_rho_tot": ra + rb,
    }


def libxc_eval(name, family, ing, do_fxc=False):
    f = LibXCFunctional(name, "polarized")
    inp = {v: ing[v] for v in FAMILY_VARS[family]}
    return f.compute(inp, do_fxc=do_fxc)


def exc_of_P(name, family, g, Pa, Pb) -> float:
    ing = ingredients(g, Pa, Pb)
    out = libxc_eval(name, family, ing)
    return float(np.sum(g.w * ing["_rho_tot"] * out["zk"].reshape(-1)))


def _fill_libxc(name: str, arr: np.ndarray) -> np.ndarray:
    return arr


def _field_env(g: Grid, ing, out) -> Dict[str, np.ndarray]:
    env = {"w": g.w}
    for i, ax in enumerate(AXES):
        env[f"grad_rho_a_{ax}"] = ing["_grad_a"][i]
        env[f"grad_rho_b_{ax}"] = ing["_grad_b"][i]
    # Libxc derivative arrays, referenced by symbol name '<array>_<comp>'.
    for key, arr in out.items():
        if key == "zk" or arr is None:
            continue
        A = np.atleast_2d(arr.T).T if arr.ndim == 1 else arr
        for c in range(A.shape[1]):
            env[f"{key}_{c}"] = A[:, c]
    return env


def _orb_env(g: Grid, label: str, idx: int) -> Dict[str, np.ndarray]:
    e = {f"chi_{label}": g.chi[idx], f"lapl_chi_{label}": g.lapl_chi[idx]}
    for i, ax in enumerate(AXES):
        e[f"dchi_{label}_{ax}"] = g.dchi[idx, i]
    return e


def contract(expr: sp.Expr, g: Grid, field: Dict[str, np.ndarray],
             labels: List[str]) -> np.ndarray:
    syms = sorted(expr.free_symbols, key=lambda s: s.name)
    fn = sp.lambdify(syms, expr, "numpy")
    nbf, npts = g.chi.shape
    ones = np.ones(npts)
    out = np.zeros((nbf,) * len(labels))
    import itertools
    for combo in itertools.product(range(nbf), repeat=len(labels)):
        env = dict(field)
        for lbl, idx in zip(labels, combo):
            env.update(_orb_env(g, lbl, idx))
        args = [np.broadcast_to(env[s.name], (npts,)) * ones for s in syms]
        out[combo] = np.sum(fn(*args))
    return out


def check_fock(family: str):
    name = FAMILY_FUNCTIONAL[family]
    g = make_grid()
    ing = ingredients(g, g.Pa, g.Pb)
    out = libxc_eval(name, family, ing)
    field = _field_env(g, ing, out)
    Fa = contract(fock_spin(family, "a").expr, g, field, ["u", "v"])

    nbf = g.Pa.shape[0]; h = 1e-6
    Fd = np.zeros((nbf, nbf))
    for u in range(nbf):
        for v in range(nbf):
            Pp = g.Pa.copy(); Pp[u, v] += h
            Pm = g.Pa.copy(); Pm[u, v] -= h
            Fd[u, v] = (exc_of_P(name, family, g, Pp, g.Pb)
                        - exc_of_P(name, family, g, Pm, g.Pb)) / (2 * h)
    err = np.max(np.abs(Fa - Fd)); scale = np.max(np.abs(Fd)) or 1
    return err, err / scale


# Same-spin coupling is nonzero for exchange; opposite-spin coupling is nonzero
# only for correlation (exchange is spin-separable, d2Ex/drho_a drho_b = 0).
KERNEL_FUNCTIONAL = {
    ("lda", "a", "a"): "LDA_X",   ("lda", "a", "b"): "LDA_C_PW",
    ("gga", "a", "a"): "GGA_X_PBE", ("gga", "a", "b"): "GGA_C_PBE",
    ("mgga_tau", "a", "a"): "MGGA_X_SCAN", ("mgga_tau", "a", "b"): "MGGA_C_SCAN",
    ("mgga", "a", "a"): "MGGA_X_BR89",
}


def check_kernel(family: str, s1: str, s2: str):
    name = KERNEL_FUNCTIONAL[(family, s1, s2)]
    g = make_grid()
    ing = ingredients(g, g.Pa, g.Pb)
    out = libxc_eval(name, family, ing, do_fxc=True)
    field = _field_env(g, ing, out)
    Ga = contract(kernel_spin(family, s1, s2).expr, g, field,
                  ["u", "v", "t", "s"])

    nbf = g.Pa.shape[0]; h = 2e-4
    Gd = np.zeros((nbf,) * 4)

    def E(e1, sg1, e2, sg2):
        # perturb entry e1 of channel s1 by sg1*h and entry e2 of s2 by sg2*h
        Pa = g.Pa.copy(); Pb = g.Pb.copy()
        {"a": Pa, "b": Pb}[s1][e1] += sg1 * h
        {"a": Pa, "b": Pb}[s2][e2] += sg2 * h
        return exc_of_P(name, family, g, Pa, Pb)

    for u in range(nbf):
        for v in range(nbf):
            for t in range(nbf):
                for s in range(nbf):
                    Gd[u, v, t, s] = (E((u, v), +1, (t, s), +1)
                                      - E((u, v), +1, (t, s), -1)
                                      - E((u, v), -1, (t, s), +1)
                                      + E((u, v), -1, (t, s), -1)) / (4 * h * h)
    err = np.max(np.abs(Ga - Gd)); scale = np.max(np.abs(Gd)) or 1
    return err, err / scale


if __name__ == "__main__":
    print("Spin-polarized Fock  F^a_uv = dExc/dP^a_uv")
    for fam in ("lda", "gga", "mgga_tau", "mgga"):
        err, rel = check_fock(fam)
        print(f"  [{'OK ' if rel < 1e-5 else 'FAIL'}] {fam:9s} "
              f"{FAMILY_FUNCTIONAL[fam]:14s} abs={err:.3e} rel={rel:.3e}")
    print("Spin-polarized kernel  g^{s1 s2}_uv,ts = dExc/dP^s1_uv dP^s2_ts")
    for (fam, s1, s2), name in KERNEL_FUNCTIONAL.items():
        err, rel = check_kernel(fam, s1, s2)
        print(f"  [{'OK ' if rel < 1e-3 else 'FAIL'}] g^{{{s1}{s2}}} {fam:9s} "
              f"{name:14s} abs={err:.3e} rel={rel:.3e}")
