"""Validate the fxc bilinear form (the E[2] XC coupling element used by
Casida-type codes): d2/dt1 dt2 of E = sum_g w e(fields + t1 d1 + t2 d2)
by Richardson cross finite differences on analytic periodic fields,
against xckernel.engine.response.fxc_bilinear evaluated with pylibxc
derivative arrays -- for lda / gga / mgga_tau / mgga_lapl / mgga; and the
hmgga bilinears exactly against a toy functional (validate_composite_exact),
and the completeness of the potential and fxc channels of every family."""

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


def validate_st(m, F0, pert):
    """Closed-shell singlet/triplet bilinears at the spin-compensated
    point: the singlet combination must reproduce the unpolarized
    bilinear exactly, and the triplet combination is checked against
    finite differences with antisymmetric channel perturbations."""
    from ..engine.spin_kernel import fxc_bilinear_st

    fails = 0
    FS = {s: {"rho": F0["rho"] / 2, "grad": F0["grad"] / 2,
              "tau": F0["tau"] / 2} for s in "ab"}
    for family, (func_ids, H, tol) in SPIN_FAMILIES.items():
        try:
            out = spin_libxc_eval(func_ids, family, FS, orders=(1, 2))
        except Exception as exc:
            print(f"  [SKIP] st {family:10s}: no fxc ({exc})")
            continue
        vals = {}
        for name, arr in out.items():
            for k in range(arr.shape[1]):
                vals[f"{name}_{k}"] = arr[:, k]
        vals["rho_a"] = FS["a"]["rho"]
        vals["tau_a"] = FS["a"]["tau"]
        for i, ax in enumerate(AXES):
            vals[f"grad_rho_a_{ax}"] = FS["a"]["grad"][i]
        for li, p in zip(("p1", "p2"), pert):
            vals[f"rho_a_{li}"] = p["rho"] / 2
            vals[f"tau_a_{li}"] = p["tau"] / 2
            for i, ax in enumerate(AXES):
                vals[f"grad_rho_a_{li}_{ax}"] = p["grad"][i] / 2

        def evaluate(parities):
            expr = fxc_bilinear_st(family, parities)
            syms = sorted(expr.free_symbols, key=lambda s_: s_.name)
            fn = sp.lambdify(syms, expr, "numpy")
            per_point = np.broadcast_to(
                fn(*[vals[s_.name] for s_ in syms]), F0["rho"].shape)
            return m.w * float(np.sum(per_point))

        # singlet identity vs the unpolarized bilinear
        unpol_ids = func_ids
        uout = libxc_eval(unpol_ids, family, F0, orders=(1, 2))
        uvals = dict(uout)
        uvals.update({"rho": F0["rho"], "sigma": F0["sigma"],
                      "lapl_rho": F0["lapl"], "tau": F0["tau"]})
        for i, ax in enumerate(AXES):
            uvals[f"grad_rho_{ax}"] = F0["grad"][i]
        for li, p in zip(("p1", "p2"), pert):
            uvals[f"rho_{li}"] = p["rho"]
            uvals[f"tau_{li}"] = p["tau"]
            uvals[f"lapl_rho_{li}"] = p["lapl"]
            for i, ax in enumerate(AXES):
                uvals[f"grad_rho_{li}_{ax}"] = p["grad"][i]
        uexpr = fxc_bilinear(family)
        usyms = sorted(uexpr.free_symbols, key=lambda s_: s_.name)
        uref = m.w * float(np.sum(sp.lambdify(usyms, uexpr, "numpy")(
            *[uvals[s_.name] for s_ in usyms])))
        singlet = evaluate((+1, +1))
        rel_s = abs(singlet - uref) / abs(uref)
        ok_s = rel_s < 1e-12

        # triplet vs antisymmetric-channel finite differences
        spert = [{"a": {k: p[k] / 2 for k in ("rho", "grad", "tau")},
                  "b": {k: -p[k] / 2 for k in ("rho", "grad", "tau")}}
                 for p in pert]
        triplet = evaluate((-1, -1))

        def cross(h):
            E = lambda a, b: spin_energy(m, func_ids, family,  # noqa
                                         FS, spert, a, b)
            return (E(h, h) - E(h, -h) - E(-h, h)
                    + E(-h, -h)) / (4 * h * h)
        fd = (4 * cross(H) - cross(2 * H)) / 3
        rel_t = abs(triplet - fd) / max(1.0, abs(fd))
        ok_t = rel_t < tol
        print(f"  [{'OK ' if ok_s and ok_t else 'FAIL'}] st {family:10s}: "
              f"singlet-vs-unpol {rel_s:.2e}, triplet-vs-fd {rel_t:.2e}")
        fails += 0 if (ok_s and ok_t) else 1
    return fails


def _toy_point(spins, seed=3):
    """Rational ground-state and perturbation fields at one point, as
    linear functions of (t1, t2), and their symbol values."""
    import random
    from ..engine.spin import HESS_INDEX
    rnd = random.Random(seed)
    R = lambda: sp.Rational(rnd.randint(-9, 9), 7)  # noqa: E731
    t1, t2 = sp.symbols("t1 t2")
    vals, F = {}, {}

    def line(name, base, ins=""):
        b = (base, R(), R())
        vals[name] = b[0]
        for lab, x in zip(("p1", "p2"), b[1:]):
            stem, comp = (name.rsplit("_", 1) if ins else (name, None))
            vals[f"{stem}_{lab}_{comp}" if ins else f"{name}_{lab}"] = x
        return b[0] + t1 * b[1] + t2 * b[2]

    for s in spins:
        t = f"_{s}" if s else ""
        pos = lambda: sp.Rational(rnd.randint(3, 9), 5)  # noqa: E731
        F[s] = {k: line(f"{k}{t}", pos()) for k in ("rho", "lapl_rho",
                                                     "tau")}
        F[s]["grad"] = [line(f"grad_rho{t}_{ax}", R(), "ax") for ax in AXES]
        H = {}
        for (i, j) in HESS_INDEX:
            if i <= j:
                H[(i, j)] = H[(j, i)] = line(
                    f"hess_rho{t}_{AXES[i]}{AXES[j]}", R(), "ax")
        F[s]["hess"] = H
    return (t1, t2), vals, F


def _toy_scalar(F, group, comp):
    """Libxc scalar from the toy fields (comp: spin tag(s), '' restricted)."""
    if group == "rho":
        return F[comp]["rho"]
    if group == "lapl":
        return F[comp]["lapl_rho"]
    if group == "tau":
        return F[comp]["tau"]
    if group == "sigma":
        s, t = (comp[0], comp[1]) if comp else ("", "")
        return sum(a * b for a, b in zip(F[s]["grad"], F[t]["grad"]))
    g, H = F[comp]["grad"], F[comp]["hess"]
    return sum(g[i] * H[(i, j)] * g[j] for i in range(3) for j in range(3))


def validate_composite_exact():
    """The density-Hessian eta, a composite scalar, enters the bilinear
    through its own curvature, which no Libxc functional reachable from
    pylibxc exercises.  Check the restricted and polarized hmgga
    bilinears exactly against d2/dt1 dt2 of a toy f(scalars) of known
    derivatives, in rational arithmetic."""
    import re
    from ..engine.spin import family_scalars
    from ..engine.spin_kernel import _SYM_SCALARS, fxc_bilinear_spin

    groups = ("rho", "sigma", "lapl", "tau", "eta")
    fails = 0
    for spin in (False, True):
        spins = ("a", "b") if spin else ("",)
        (t1, t2), vals, F = _toy_point(spins)
        if spin:
            keys = [(sc.group, sc.comp) for sc in family_scalars("hmgga")]
        else:
            keys = [(g, "") for g in groups]
        X = {k: sp.Symbol("X_" + "_".join(k)) for k in keys}
        xs = list(X.values())
        rnd = __import__("random").Random(5)
        f = sum(sp.Rational(rnd.randint(-9, 9), 11) * xs[i] * xs[j] * xs[k]
                for i in range(len(xs)) for j in range(i, len(xs))
                for k in range(j, len(xs)))
        on = {X[k]: _toy_scalar(F, *k) for k in keys}
        at0 = {t1: 0, t2: 0}
        ref = sp.diff(f.subs(on, simultaneous=True), t1, t2).subs(at0)
        x0 = {x: v.subs(at0) for x, v in on.items()}

        def deriv(ks):
            d = f
            for k in ks:
                d = sp.diff(d, X[k])
            return d.subs(x0)

        if spin:
            expr = fxc_bilinear_spin("hmgga")
            sub = {a: deriv([(sc.group, sc.comp)
                             for sc in _SYM_SCALARS[a.name]])
                   for a in expr.free_symbols if a.name in _SYM_SCALARS}
        else:
            expr = fxc_bilinear("hmgga")
            sub = {}
            for a in expr.free_symbols:
                m = re.fullmatch(r"v2?((?:%s)+?)2?" % "|".join(groups),
                                 a.name)
                if m is None:
                    continue
                gs = re.findall("|".join(groups), m.group(1))
                if a.name.startswith("v2") and len(gs) == 1:
                    gs = gs * 2
                sub[a] = deriv([(g, "") for g in gs])
        sub.update({a: vals[a.name] for a in expr.free_symbols
                    if a not in sub})
        diff = sp.nsimplify(expr.subs(sub) - ref)
        ok = diff == 0
        fails += 0 if ok else 1
        print(f"  [{'OK ' if ok else 'FAIL'}] exact "
              f"{'spin ' if spin else ''}hmgga (eta curvature): "
              f"ana - exact = {diff}")
    return fails


CHANNEL_FAMILIES = ("lda", "gga", "mgga_tau", "mgga_lapl", "mgga",
                    "cmgga_tau", "hmgga")


def _channel_key(q: sp.Symbol) -> str:
    """Channel key of a second-perturbation operand: grad_rho_a__q_x ->
    grad_a_x, lapl_rho__q -> lapl, hess_rho_a__q_xy -> hess_rho_a_xy."""
    name = q.name.replace("__q", "").replace("grad_rho_", "grad_")
    return "lapl" + name[len("lapl_rho"):] if name.startswith("lapl_rho") \
        else name


def validate_channels_complete():
    """The channel helpers are only a regrouping, so they must lose
    nothing, for every family and spin treatment: the potential channels
    reproduce dExc/dP (check_vxc_channels), and the fxc channels rebuild
    the bilinear exactly, sum_X c_X X^q, one channel per operand of the
    second perturbation."""
    from ..engine.fock import check_vxc_channels
    from ..engine.response import fxc_channels
    from ..engine.spin_kernel import (check_vxc_channels_spin,
                                      fxc_bilinear_spin, fxc_channels_spin)
    fails = 0
    for family in CHANNEL_FAMILIES:
        for spin in (False, True):
            check = check_vxc_channels_spin if spin else check_vxc_channels
            bil = fxc_bilinear_spin if spin else fxc_bilinear
            ch = (fxc_channels_spin if spin else fxc_channels)(family)
            b = bil(family, l1="p1", l2="_q")
            ops = [a for a in b.free_symbols if "__q" in a.name]
            missing = sorted(_channel_key(q) for q in ops
                             if _channel_key(q) not in ch)
            res = sp.expand(b - sum((ch[_channel_key(q)] * q for q in ops
                                     if _channel_key(q) in ch),
                                    sp.Integer(0)))
            ok = check(family) == 0 and res == 0 and not missing
            fails += 0 if ok else 1
            print(f"  [{'OK ' if ok else 'FAIL'}] channels "
                  f"{'spin ' if spin else ''}{family:10s}"
                  + (f" missing {missing}" if missing else ""))
    return fails


def main():
    m, F0, pert = make_setup()
    failures = 0
    for family, (func_ids, H, tol) in FAMILIES.items():
        failures += validate_family(m, F0, pert, family, func_ids, H, tol)
    ms, FS0, spert = make_spin_setup()
    for family, (func_ids, H, tol) in SPIN_FAMILIES.items():
        failures += validate_family_spin(ms, FS0, spert, family,
                                         func_ids, H, tol)
    failures += validate_st(m, F0, pert)
    failures += validate_composite_exact()
    failures += validate_channels_complete()
    n = len(FAMILIES) + 2 * len(SPIN_FAMILIES) + 2 + 2 * len(CHANNEL_FAMILIES)
    status = "OK " if failures == 0 else "FAIL"
    print(f"[{status}] bilinear_validate: {n} checks, "
          f"{failures} failures")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
