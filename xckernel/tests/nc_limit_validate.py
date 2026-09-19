"""The collinear (|m| -> 0) limit of the noncollinear kernel.

nc_fxc_collinear_limit is what a host evaluates below its magnetization
cutoff. Checked against the FULL locally collinear kernel of
engine.noncollinear, on a spin-symmetric model functional, as m -> 0:

* LDA: every trial direction, transverse included -- the LDA kernel
  has a unique limit, so full and limit must agree.
* GGA: charge-only and collinear-spin trials. The GGA map is not twice
  differentiable at m = 0: the transverse response depends on how m
  vanishes (through grad|m|/|m|), and the Scalmani-Frisch gamma map adds
  a 1/|grad rho_s . grad m| singularity at finite |m|. The limit kernel
  is the unique rotation-invariant form that agrees with every collinear
  perturbation; transverse agreement is not expected and not tested.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from ..engine.noncollinear import (F_NABLA, libxc_args, nc_fxc_collinear_limit,
                                   nc_fxc_matrix, nc_map)
from .noncollinear_validate import _deriv_arrays


def run() -> int:
    rng = np.random.default_rng(3)
    failures = checks = 0
    for fam in ("lda", "gga"):
        fields, C = nc_fxc_matrix(fam)
        names = [s.name for s in fields]
        idx = {n: i for i, n in enumerate(names)}
        U = nc_map(fam)
        a2 = libxc_args(fam, 2)
        Cf = sp.lambdify(fields + [sp.Symbol(a) for a in a2] + [F_NABLA], C, "numpy")
        Uf = {a: sp.lambdify(fields + [F_NABLA], e, "numpy") for a, e in U.items()}
        refs, trial, K = nc_fxc_collinear_limit(fam)
        Kf = sp.lambdify(refs + [sp.Symbol(a) for a in a2] + trial, K, "numpy")

        rho, gs = 0.7, rng.normal(size=3)
        e = rng.normal(size=3); e /= np.linalg.norm(e)
        g, h = rng.normal(size=3), rng.normal(size=3)
        eps = 1e-6
        V = {"rho_s": rho}
        for J, a in enumerate("xyz"):
            V[f"rho_{a}"] = eps * e[J]
        if fam == "gga":
            for c, a in enumerate("xyz"):
                V[f"grad_rho_s_{a}"] = gs[c]
            for J, a in enumerate("xyz"):
                for c, b in enumerate("xyz"):
                    V[f"grad_rho_{a}_{b}"] = eps * e[J] * g[c]
        vals = [V[n] for n in names]
        fn = 1.0 if np.dot(gs, g) >= 0 else -1.0
        u = [float(Uf[a](*vals, fn)) for a in U] + [0.0] * (5 - len(U))
        d = _deriv_arrays(a2, [np.array(x) for x in u])
        dv = [float(d[a]) for a in a2]
        Cm = np.array(Cf(*vals, *dv, fn), dtype=float)
        refv = [rho] + (list(gs) if fam == "gga" else [])

        trials = {}
        if fam == "lda":
            trials["general"] = rng.normal(size=len(names))
        else:
            tc = np.zeros(len(names)); tc[idx["rho_s"]] = 0.3
            for c, a in enumerate("xyz"):
                tc[idx[f"grad_rho_s_{a}"]] = rng.normal()
            tl = np.zeros(len(names))
            for J, a in enumerate("xyz"):
                tl[idx[f"rho_{a}"]] = 0.4 * e[J]
                for c, b in enumerate("xyz"):
                    tl[idx[f"grad_rho_{a}_{b}"]] = e[J] * h[c]
            trials["charge"], trials["collinear spin"] = tc, tl
        for lab, t in trials.items():
            full = Cm @ t
            lim = np.array(Kf(*refv, *dv, *t), dtype=float)
            err = np.max(np.abs(full - lim)) / np.max(np.abs(lim))
            ok = err < 1e-5
            checks += 1; failures += 0 if ok else 1
            print(f"  [{'OK' if ok else 'FAIL'}] {fam} full kernel -> limit "
                  f"({lab}, |m| = {eps:.0e}): rel {err:.2e}")
    tag = "OK " if not failures else "FAIL"
    print(f"[{tag}] nc_limit_validate: {checks} checks, {failures} failures")
    return failures


if __name__ == "__main__":
    raise SystemExit(run())
