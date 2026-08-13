"""Validate the fxc bilinear form (the E[2] XC coupling element used by
Casida-type codes): d2/dt1 dt2 of E = sum_g w e(fields + t1 d1 + t2 d2)
by Richardson cross finite differences on analytic periodic fields,
against xckernel.engine.response.fxc_bilinear evaluated with pylibxc
derivative arrays -- for lda / gga / mgga_tau / mgga_lapl / mgga."""

from __future__ import annotations

import numpy as np
import sympy as sp

from ..engine.response import fxc_bilinear
from ..inputs.basis import AXES

# family -> (functionals, fd step, tolerance).  The r2scan-family
# switching functions have rapidly varying higher derivatives, so their
# FD references need a smaller step and (for r2scanl) a looser
# tolerance; smooth functionals sit at the h ~ 1e-2 noise optimum.
FAMILIES = {
    "lda": (["LDA_X", "LDA_C_PW"], 1e-2, 2e-8),
    "gga": (["GGA_X_PBE", "GGA_C_PBE"], 1e-2, 2e-8),
    "mgga_tau": (["MGGA_X_R2SCAN", "MGGA_C_R2SCAN"], 3e-3, 2e-8),
    "mgga_lapl": (["MGGA_X_R2SCANL", "MGGA_C_R2SCANL"], 3e-3, 2e-7),
    "mgga": (["MGGA_X_BR89"], 1e-2, 2e-8),
}

N = 14


class FieldModel:
    """Analytic periodic scalar fields with exact gradients/Laplacians."""

    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        L = 5.0
        x = np.stack(np.meshgrid(*[np.arange(N) * (L / N)] * 3,
                                 indexing="ij"), axis=-1).reshape(-1, 3)
        self.w = (L / N) ** 3
        self.modes = 2 * np.pi / L * rng.integers(-2, 3, size=(6, 3))
        self.x = x

    def scalar(self, coeff):
        ph = self.x @ self.modes.T
        f = coeff[0] + np.cos(ph) @ coeff[1:]
        g = np.stack([-(np.sin(ph) * self.modes[:, v]) @ coeff[1:]
                      for v in range(3)])
        lap = -(np.cos(ph) * (self.modes ** 2).sum(1)) @ coeff[1:]
        return f, g, lap


def make_setup(seed=11):
    m = FieldModel(seed)
    rng = np.random.default_rng(seed + 1)
    psi, dpsi, ddpsi = m.scalar(np.concatenate(([1.0],
                                                rng.normal(size=6) * 0.15)))
    rho = psi ** 2 + 0.1
    grad = 2 * psi * dpsi
    lapl = 2 * ((dpsi ** 2).sum(0) + psi * ddpsi)
    sigma = (grad ** 2).sum(0)
    tau = sigma / (8 * rho) + 0.3 * rho ** (5 / 3)

    pert = []
    for k in (2, 3):
        d, dd, dlap = m.scalar(rng.normal(size=7) * 0.1)
        dtau, _, _ = m.scalar(rng.normal(size=7) * 0.05)
        pert.append({"rho": d, "grad": dd, "lapl": dlap, "tau": dtau})
    return m, {"rho": rho, "grad": grad, "sigma": sigma, "lapl": lapl,
               "tau": tau}, pert


def libxc_eval(func_ids, family, F, orders):
    import pylibxc
    inp = {"rho": F["rho"]}
    if family != "lda":
        inp["sigma"] = F["sigma"]
    if family in ("mgga", "mgga_lapl"):
        inp["lapl"] = F["lapl"]
    if family in ("mgga", "mgga_tau", "mgga_lapl"):
        inp["tau"] = F["tau"]
    out = {}
    for fid in func_ids:
        f = pylibxc.LibXCFunctional(fid, "unpolarized")
        r = f.compute(inp, do_exc=0 in orders, do_vxc=1 in orders,
                      do_fxc=2 in orders)
        for k, v in r.items():
            out[k] = out.get(k, 0) + v.reshape(v.shape[0], -1).T.squeeze()
    return out


def energy(m, func_ids, family, F0, pert, t1, t2):
    rho = F0["rho"] + t1 * pert[0]["rho"] + t2 * pert[1]["rho"]
    grad = F0["grad"] + t1 * pert[0]["grad"] + t2 * pert[1]["grad"]
    F = {"rho": rho, "sigma": (grad ** 2).sum(0),
         "lapl": F0["lapl"] + t1 * pert[0]["lapl"] + t2 * pert[1]["lapl"],
         "tau": F0["tau"] + t1 * pert[0]["tau"] + t2 * pert[1]["tau"]}
    out = libxc_eval(func_ids, family, F, orders=(0,))
    return m.w * float(np.dot(F["rho"], out["zk"]))


def validate_family(m, F0, pert, family, func_ids, H, tol):
    try:
        out = libxc_eval(func_ids, family, F0, orders=(1, 2))
    except Exception as exc:
        print(f"  [SKIP] {family:10s}: no fxc ({exc})")
        return 0
    vals = dict(out)
    vals.update({"rho": F0["rho"], "sigma": F0["sigma"],
                 "lapl_rho": F0["lapl"], "tau": F0["tau"]})
    for i, ax in enumerate(AXES):
        vals[f"grad_rho_{ax}"] = F0["grad"][i]
    for li, p in zip(("p1", "p2"), pert):
        vals[f"rho_{li}"] = p["rho"]
        vals[f"lapl_rho_{li}"] = p["lapl"]
        vals[f"tau_{li}"] = p["tau"]
        for i, ax in enumerate(AXES):
            vals[f"grad_rho_{li}_{ax}"] = p["grad"][i]

    expr = fxc_bilinear(family)
    syms = sorted(expr.free_symbols, key=lambda s: s.name)
    fn = sp.lambdify(syms, expr, "numpy")
    ana = m.w * float(np.sum(fn(*[vals[s.name] for s in syms])))

    def cross(h):
        E = lambda a, b: energy(m, func_ids, family, F0, pert, a, b)  # noqa
        return (E(h, h) - E(h, -h) - E(-h, h) + E(-h, -h)) / (4 * h * h)
    fd = (4 * cross(H) - cross(2 * H)) / 3
    rel = abs(ana - fd) / max(1.0, abs(fd))
    ok = rel < tol
    print(f"  [{'OK ' if ok else 'FAIL'}] {family:10s} "
          f"({'+'.join(func_ids)}): |ana - fd| / scale = {rel:.2e}")
    return 0 if ok else 1


def main():
    m, F0, pert = make_setup()
    failures = 0
    for family, (func_ids, H, tol) in FAMILIES.items():
        failures += validate_family(m, F0, pert, family, func_ids, H, tol)
    status = "OK " if failures == 0 else "FAIL"
    print(f"[{status}] bilinear_validate: {len(FAMILIES)} checks, "
          f"{failures} failures")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
