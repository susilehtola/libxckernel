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


# The spin setup has a larger energy scale (worse FD truncation) and
# the r2scan switching functions have kinks that h=3e-3 straddles, so
# the spin checks use smaller steps.
SPIN_FAMILIES = {
    "lda": (["LDA_X", "LDA_C_PW"], 3e-3, 2e-8),
    "gga": (["GGA_X_PBE", "GGA_C_PBE"], 1e-3, 2e-8),
    "mgga_tau": (["MGGA_X_R2SCAN", "MGGA_C_R2SCAN"], 1e-3, 2e-7),
}


def make_spin_setup(seed=21):
    m = FieldModel(seed)
    rng = np.random.default_rng(seed + 1)
    chans = []
    for scale in (1.0, 0.7):
        psi, dpsi, _ = m.scalar(np.concatenate(
            ([scale], rng.normal(size=6) * 0.15)))
        rho = psi ** 2 + 0.1 * scale
        grad = 2 * psi * dpsi
        tau = (grad ** 2).sum(0) / (8 * rho) + 0.3 * rho ** (5 / 3)
        chans.append({"rho": rho, "grad": grad, "tau": tau})
    pert = []
    for k in range(2):
        p = {}
        for s in "ab":
            d, dd, _ = m.scalar(rng.normal(size=7) * 0.1)
            dtau, _, _ = m.scalar(rng.normal(size=7) * 0.05)
            p[s] = {"rho": d, "grad": dd, "tau": dtau}
        pert.append(p)
    return m, {"a": chans[0], "b": chans[1]}, pert


def spin_libxc_eval(func_ids, family, F, orders):
    import pylibxc
    npts = len(F["a"]["rho"])
    inp = {"rho": np.stack([F["a"]["rho"], F["b"]["rho"]], axis=-1)}
    if family != "lda":
        inp["sigma"] = np.stack(
            [(F["a"]["grad"] ** 2).sum(0),
             (F["a"]["grad"] * F["b"]["grad"]).sum(0),
             (F["b"]["grad"] ** 2).sum(0)], axis=-1)
    if family == "mgga_tau":
        inp["tau"] = np.stack([F["a"]["tau"], F["b"]["tau"]], axis=-1)
    out = {}
    for fid in func_ids:
        f = pylibxc.LibXCFunctional(fid, "polarized")
        r = f.compute(inp, do_exc=0 in orders, do_vxc=1 in orders,
                      do_fxc=2 in orders)
        for k, v in r.items():
            out[k] = out.get(k, 0) + v.reshape(npts, -1)
    return out


def spin_energy(m, func_ids, family, F0, pert, t1, t2):
    F = {}
    for s in "ab":
        F[s] = {k: F0[s][k] + t1 * pert[0][s][k] + t2 * pert[1][s][k]
                for k in ("rho", "grad", "tau")}
    out = spin_libxc_eval(func_ids, family, F, orders=(0,))
    ntot = F["a"]["rho"] + F["b"]["rho"]
    return m.w * float(np.dot(ntot, out["zk"][:, 0]))


def validate_family_spin(m, F0, pert, family, func_ids, H, tol):
    from ..engine.spin_kernel import fxc_bilinear_spin
    try:
        out = spin_libxc_eval(func_ids, family, F0, orders=(1, 2))
    except Exception as exc:
        print(f"  [SKIP] spin {family:10s}: no fxc ({exc})")
        return 0
    vals = {}
    for name, arr in out.items():
        for k in range(arr.shape[1]):
            vals[f"{name}_{k}"] = arr[:, k]
    for s in "ab":
        vals[f"rho_{s}"] = F0[s]["rho"]
        vals[f"tau_{s}"] = F0[s]["tau"]
        for i, ax in enumerate(AXES):
            vals[f"grad_rho_{s}_{ax}"] = F0[s]["grad"][i]
        for li, p in zip(("p1", "p2"), pert):
            vals[f"rho_{s}_{li}"] = p[s]["rho"]
            vals[f"tau_{s}_{li}"] = p[s]["tau"]
            for i, ax in enumerate(AXES):
                vals[f"grad_rho_{s}_{li}_{ax}"] = p[s]["grad"][i]

    expr = fxc_bilinear_spin(family)
    syms = sorted(expr.free_symbols, key=lambda s_: s_.name)
    fn = sp.lambdify(syms, expr, "numpy")
    ana = m.w * float(np.sum(fn(*[vals[s_.name] for s_ in syms])))

    def cross(h):
        E = lambda a, b: spin_energy(m, func_ids, family,  # noqa
                                     F0, pert, a, b)
        return (E(h, h) - E(h, -h) - E(-h, h) + E(-h, -h)) / (4 * h * h)
    fd = (4 * cross(H) - cross(2 * H)) / 3
    rel = abs(ana - fd) / max(1.0, abs(fd))
    ok = rel < tol
    print(f"  [{'OK ' if ok else 'FAIL'}] spin {family:10s} "
          f"({'+'.join(func_ids)}): |ana - fd| / scale = {rel:.2e}")
    return 0 if ok else 1


def main():
    m, F0, pert = make_setup()
    failures = 0
    for family, (func_ids, H, tol) in FAMILIES.items():
        failures += validate_family(m, F0, pert, family, func_ids, H, tol)
    ms, FS0, spert = make_spin_setup()
    for family, (func_ids, H, tol) in SPIN_FAMILIES.items():
        failures += validate_family_spin(ms, FS0, spert, family,
                                         func_ids, H, tol)
    n = len(FAMILIES) + len(SPIN_FAMILIES)
    status = "OK " if failures == 0 else "FAIL"
    print(f"[{status}] bilinear_validate: {n} checks, "
          f"{failures} failures")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
