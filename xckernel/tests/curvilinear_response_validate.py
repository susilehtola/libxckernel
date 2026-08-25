"""Finite-difference validation of the CURVILINEAR potential and kernel.

``validate.py`` and ``response_validate.py`` prove the Cartesian Fock
matrix and fxc contraction against finite differences of the energy.
The curvilinear systems deserve the same standard of proof, and cannot
get it the same way: only one of the four has a host with a
second-order optimizer to compare against, so the spherical, prolate
spheroidal and pure-m kernels would otherwise rest on the metric check
of ``curvilinear_validate.py`` alone.

This script closes that gap without a host.  It fabricates a small
curvilinear "code" -- random basis values and coordinate derivatives at
random grid points, random positive Lame factors, and a random density
matrix -- assembles the ingredients from the GENERATOR'S OWN seeds
(nothing is re-derived here), evaluates Libxc on them, and checks

  * the potential channels of ``fock.vxc_channels`` against finite
    differences of Exc with respect to the density matrix, and
  * the kernel channels of ``response.fxc_channels`` against finite
    differences of that potential,

for all four coordinate systems.  The two reduced systems carry a
BLOCKED density matrix, exactly as their host codes do: the density is a
sum over angular-momentum (or azimuthal) blocks, and block b contributes
its own centrifugal term to the kinetic energy density.  Getting the
Fock matrix of a single block right is precisely what a reduction
demands and what a Cartesian generator cannot express.

Run with: python -m xckernel.tests.curvilinear_response_validate
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import sympy as sp
from pylibxc import LibXCFunctional

from ..engine.fock import vxc_channels
from ..engine.response import fxc_channels
from ..engine.spin_kernel import fxc_channels_spin, vxc_channels_spin
from ..inputs.basis import (Orbital, PROLATE, PROLATE_PUREM, RADIAL,
                            SPHERICAL)
from ..inputs.ingredients import primitives_for

#: well-behaved representatives; SCAN and r2SCAN are avoided on purpose,
#: their numerics being poor enough to swamp a finite-difference check.
FAMILY_FUNCTIONAL = {
    "lda": "LDA_X",
    "gga": "GGA_X_PBE",
    "mgga_tau": "MGGA_X_TPSS",
}

#: coordinate system -> the per-block angular factors it carries.  A
#: system with no reduction has a single block and no angular term.
SYSTEMS = {
    "spherical": (SPHERICAL, [0.0]),
    "prolate": (PROLATE, [0.0]),
    # l(l+1) for l = 0, 1, 2 -- a genuinely blocked radial density
    "radial": (RADIAL, [0.0, 2.0, 6.0]),
    # m^2 for m = 0, 1, 2
    "prolate_purem": (PROLATE_PUREM, [0.0, 1.0, 4.0]),
}


@dataclass
class Grid:
    """Fabricated basis data on a curvilinear grid."""

    w: np.ndarray                       # (npts,)
    chi: np.ndarray                     # (nbf, npts)
    dchi: Dict[str, np.ndarray]         # axis -> (nbf, npts)
    geom: Dict[str, np.ndarray]         # geometry operand -> (npts,)
    P: List[np.ndarray]                 # one (nbf, nbf) block per angular block


def _geometry_values(coords, npts, rng):
    """Numerical values for whatever geometry symbols the seeds contain.

    The scale factors are drawn well away from zero: their vanishing on
    the axis or at the origin is a property of a real grid, not of the
    chain rule, and is the subject of the host-side regularization
    rather than of this check.
    """
    vals = {}
    for name in ("r",):
        vals[name] = rng.uniform(0.5, 2.0, npts)
    vals["sin_theta"] = rng.uniform(0.3, 1.0, npts)
    for name in ("scale_mu", "scale_nu", "scale_phi"):
        vals[name] = rng.uniform(0.5, 2.0, npts)
    return vals


def make_grid(coords, nblocks, nbf=4, npts=40, seed=7) -> Grid:
    rng = np.random.default_rng(seed)
    w = rng.uniform(0.1, 1.0, npts)
    chi = rng.standard_normal((nbf, npts))
    dchi = {ax: rng.standard_normal((nbf, npts)) for ax in coords.axes}
    P = []
    for _ in range(nblocks):
        A = rng.standard_normal((nbf, nbf))
        P.append(A @ A.T / nblocks)     # symmetric positive-semidefinite
    return Grid(w=w, chi=chi, dchi=dchi,
                geom=_geometry_values(coords, npts, rng), P=P)


class Seeds:
    """The generator's seeds, lambdified and evaluated on the grid.

    ``block_factor`` is substituted into the coordinate system's angular
    term, so each block gets the seed its own quantum number implies.
    """

    def __init__(self, coords, g: Grid, block_factor: float):
        prims = primitives_for(coords)
        u, v = Orbital.make("u", coords), Orbital.make("v", coords)
        self.keys, exprs = [], []
        self.keys.append("rho")
        exprs.append(prims["rho"].seed(u, v))
        for p in prims["grad"]:
            self.keys.append("grad_" + p.name[len("grad_rho_"):])
            exprs.append(p.seed(u, v))
        self.keys.append("tau")
        exprs.append(prims["tau"].seed(u, v))

        # the angular term's factor is a property of the block
        sub = {}
        for s in set().union(*[e.free_symbols for e in exprs]):
            if s.name in ("l_factor", "m_factor"):
                sub[s] = block_factor
        exprs = [sp.expand(e.subs(sub)) for e in exprs]

        nbf, npts = g.chi.shape
        # Name-keyed pool: a symbol's assumptions vary between seeds, so
        # matching on the sympy object would register two spellings of
        # the same operand and lambdify would see a duplicate argument.
        pool = {"chi_u": g.chi[:, None, :], "chi_v": g.chi[None, :, :]}
        for ax in coords.axes:
            pool[f"dchi_u_{ax}"] = g.dchi[ax][:, None, :]
            pool[f"dchi_v_{ax}"] = g.dchi[ax][None, :, :]
        for name, arr in g.geom.items():
            pool[name] = arr[None, None, :]

        self.value = {}
        for key, e in zip(self.keys, exprs):
            syms = sorted(e.free_symbols, key=lambda t: t.name)
            missing = [t.name for t in syms if t.name not in pool]
            if missing:
                raise KeyError(f"seed {key} wants {missing}")
            f = sp.lambdify(syms, e, "numpy")
            val = f(*[np.broadcast_to(pool[t.name], (nbf, nbf, npts))
                      for t in syms])
            self.value[key] = np.broadcast_to(
                np.asarray(val, dtype=float), (nbf, nbf, npts)).copy()


def ingredients(coords, g: Grid, seeds: List[Seeds], P: List[np.ndarray]):
    """Libxc input variables, summed over the density-matrix blocks."""
    npts = g.chi.shape[1]
    out = {"rho": np.zeros(npts), "tau": np.zeros(npts)}
    grads = {k: np.zeros(npts) for k in seeds[0].keys if k.startswith("grad")}
    for Pb, sd in zip(P, seeds):
        out["rho"] += np.einsum("uv,uvg->g", Pb, sd.value["rho"])
        out["tau"] += np.einsum("uv,uvg->g", Pb, sd.value["tau"])
        for k in grads:
            grads[k] += np.einsum("uv,uvg->g", Pb, sd.value[k])
    out.update(grads)
    out["sigma"] = sum(grads[k] ** 2 for k in grads)
    return out


def _libxc(family, ing, order):
    """Libxc arrays for the fabricated ingredients."""
    fn = LibXCFunctional(FAMILY_FUNCTIONAL[family], "unpolarized")
    inp = {"rho": ing["rho"]}
    if family != "lda":
        inp["sigma"] = ing["sigma"]
    if family == "mgga_tau":
        inp["tau"] = ing["tau"]
        inp["lapl"] = np.zeros_like(ing["rho"])
    return fn.compute(inp, do_exc=True, do_vxc=order >= 1, do_fxc=order >= 2)


def energy(family, g: Grid, coords, seeds, P):
    ing = ingredients(coords, g, seeds, P)
    res = _libxc(family, ing, 0)
    return float(np.sum(g.w * ing["rho"] * res["zk"].ravel()))


def _channel_values(exprs, ing, res, extra=None):
    """Evaluate generated channel expressions on the fabricated data."""
    pool = {"rho": ing["rho"], "sigma": ing.get("sigma"),
            "tau": ing.get("tau")}
    for k, v in ing.items():
        if k.startswith("grad_"):
            pool["grad_rho_" + k[len("grad_"):]] = v
    for name, arr in res.items():
        pool[name] = np.asarray(arr).ravel()
    if extra:
        pool.update(extra)
    out = {}
    for key, e in exprs.items():
        syms = sorted(e.free_symbols, key=lambda s: s.name)
        missing = [s.name for s in syms if s.name not in pool]
        if missing:
            raise KeyError(f"channel {key} wants {missing}")
        f = sp.lambdify(syms, e, "numpy")
        val = f(*[pool[s.name] for s in syms])
        out[key] = np.broadcast_to(np.asarray(val, dtype=float),
                                   ing["rho"].shape).copy()
    return out


def analytic_fock(family, coords, g: Grid, seeds, P):
    """F_b,uv = sum_g w_g sum_p channel_p(g) seed_p(u,v,g), per block."""
    ing = ingredients(coords, g, seeds, P)
    res = _libxc(family, ing, 1)
    ch = _channel_values(vxc_channels(family, coords), ing, res)
    out = []
    for sd in seeds:
        F = np.zeros(P[0].shape)
        for key, c in ch.items():
            F += np.einsum("g,uvg->uv", g.w * c, sd.value[key])
        out.append(F)
    return out


def check_fock(family, name, coords, factors, tol=2e-6):
    g = make_grid(coords, len(factors))
    seeds = [Seeds(coords, g, f) for f in factors]
    F = analytic_fock(family, coords, g, seeds, g.P)
    h, worst = 1e-6, 0.0
    rng = np.random.default_rng(3)
    nbf = g.chi.shape[0]
    for b in range(len(factors)):
        for _ in range(4):                       # a sample of entries
            i, j = rng.integers(0, nbf, 2)
            Pp = [x.copy() for x in g.P]
            Pm = [x.copy() for x in g.P]
            Pp[b][i, j] += h
            Pm[b][i, j] -= h
            fd = (energy(family, g, coords, seeds, Pp)
                  - energy(family, g, coords, seeds, Pm)) / (2 * h)
            worst = max(worst, abs(F[b][i, j] - fd)
                        / max(1e-8, abs(fd)))
    return (f"{name:14s} {family:9s} vxc vs FD(Exc)", worst < tol, worst)


def check_fxc(family, name, coords, factors, tol=2e-5):
    """fxc channels against a directional derivative of the potential."""
    g = make_grid(coords, len(factors))
    seeds = [Seeds(coords, g, f) for f in factors]
    rng = np.random.default_rng(11)
    nbf = g.chi.shape[0]
    dP = []
    for _ in factors:
        A = rng.standard_normal((nbf, nbf)) * 0.05
        dP.append(A + A.T)

    ing = ingredients(coords, g, seeds, g.P)
    ping = ingredients(coords, g, seeds, dP)          # perturbed fields
    res = _libxc(family, ing, 2)
    extra = {"rho_p1": ping["rho"], "tau_p1": ping["tau"]}
    for k, v in ping.items():
        if k.startswith("grad_"):
            extra["grad_rho_p1_" + k[len("grad_"):]] = v
    ch = _channel_values(fxc_channels(family, coords=coords), ing, res, extra)

    G = []
    for sd in seeds:
        M = np.zeros((nbf, nbf))
        for key, c in ch.items():
            M += np.einsum("g,uvg->uv", g.w * c, sd.value[key])
        G.append(M)

    h = 1e-5
    Pp = [p + h * d for p, d in zip(g.P, dP)]
    Pm = [p - h * d for p, d in zip(g.P, dP)]
    Fp = analytic_fock(family, coords, g, seeds, Pp)
    Fm = analytic_fock(family, coords, g, seeds, Pm)
    worst = 0.0
    for b in range(len(factors)):
        fd = (Fp[b] - Fm[b]) / (2 * h)
        scale = max(1e-8, np.abs(fd).max())
        worst = max(worst, np.abs(G[b] - fd).max() / scale)
    return (f"{name:14s} {family:9s} fxc vs FD(vxc)", worst < tol, worst)


# --- spin-polarized counterpart -------------------------------------------
#
# HelFEM validates the polarized RADIAL path against its own finite
# differences, but radial is the one system whose single axis "r" is a
# single character -- precisely the case in which an axis-naming mistake
# stays invisible. The polarized spherical, prolate and pure-m channels
# have no host at all, so they are checked here the same way.

def spin_ingredients(coords, g: Grid, seeds, Pa, Pb):
    """Polarized Libxc input variables, summed over the blocks."""
    a = ingredients(coords, g, seeds, Pa)
    b = ingredients(coords, g, seeds, Pb)
    gk = [k for k in a if k.startswith("grad_")]
    out = {"rho_a": a["rho"], "rho_b": b["rho"],
           "tau_a": a["tau"], "tau_b": b["tau"]}
    for k in gk:
        ax = k[len("grad_"):]               # grad_r -> grad_a_r
        out[f"grad_a_{ax}"] = a[k]
        out[f"grad_b_{ax}"] = b[k]
    out["_saa"] = sum(a[k] * a[k] for k in gk)
    out["_sab"] = sum(a[k] * b[k] for k in gk)
    out["_sbb"] = sum(b[k] * b[k] for k in gk)
    return out


def _libxc_spin(family, ing, order):
    fn = LibXCFunctional(FAMILY_FUNCTIONAL[family], "polarized")
    npts = ing["rho_a"].size
    inp = {"rho": np.column_stack([ing["rho_a"], ing["rho_b"]])}
    if family != "lda":
        inp["sigma"] = np.column_stack([ing["_saa"], ing["_sab"], ing["_sbb"]])
    if family == "mgga_tau":
        inp["tau"] = np.column_stack([ing["tau_a"], ing["tau_b"]])
        inp["lapl"] = np.zeros((npts, 2))
    res = fn.compute(inp, do_exc=True, do_vxc=order >= 1, do_fxc=order >= 2)
    # flatten (npts, ncomp) arrays into the flat component names the
    # generated kernels take: vsigma_1 is the ab column of vsigma
    flat = {}
    for name, arr in res.items():
        a = np.asarray(arr)
        a = a.reshape(npts, -1) if a.size % npts == 0 else a
        if a.ndim == 2 and a.shape[0] == npts:
            for c in range(a.shape[1]):
                flat[f"{name}_{c}"] = a[:, c].copy()
        flat[name] = a
    return flat


def spin_energy(family, g, coords, seeds, Pa, Pb):
    ing = spin_ingredients(coords, g, seeds, Pa, Pb)
    res = _libxc_spin(family, ing, 0)
    zk = np.asarray(res["zk"]).ravel()
    return float(np.sum(g.w * (ing["rho_a"] + ing["rho_b"]) * zk))


def _spin_pool(ing, res, extra=None):
    pool = dict(res)
    for k, v in ing.items():
        if k.startswith("_"):
            continue
        pool[k] = v
        if k.startswith("grad_"):           # grad_a_r -> grad_rho_a_r
            pool["grad_rho_" + k[len("grad_"):]] = v
    pool["sigma_0"], pool["sigma_1"], pool["sigma_2"] = (
        ing["_saa"], ing["_sab"], ing["_sbb"])
    if extra:
        pool.update(extra)
    return pool


def _eval_spin(exprs, pool, npts):
    out = {}
    for key, e in exprs.items():
        syms = sorted(e.free_symbols, key=lambda t: t.name)
        missing = [t.name for t in syms if t.name not in pool]
        if missing:
            raise KeyError(f"spin channel {key} wants {missing}")
        f = sp.lambdify(syms, e, "numpy")
        val = f(*[pool[t.name] for t in syms])
        out[key] = np.broadcast_to(np.asarray(val, dtype=float),
                                   (npts,)).copy()
    return out


def spin_fock(family, coords, g, seeds, Pa, Pb):
    """(F_a, F_b), each a list over the density-matrix blocks."""
    ing = spin_ingredients(coords, g, seeds, Pa, Pb)
    res = _libxc_spin(family, ing, 1)
    npts = ing["rho_a"].size
    ch = _eval_spin(vxc_channels_spin(family, coords),
                    _spin_pool(ing, res), npts)
    out = {}
    for sp_ in ("a", "b"):
        mats = []
        for sd in seeds:
            F = np.zeros(Pa[0].shape)
            for key, c in ch.items():
                if not key.endswith(f"_{sp_}") and f"_{sp_}_" not in key:
                    continue
                seed_key = ("rho" if key == f"rho_{sp_}" else
                            "tau" if key == f"tau_{sp_}" else
                            "grad_" + key.split(f"_{sp_}_", 1)[1])
                F += np.einsum("g,uvg->uv", g.w * c, sd.value[seed_key])
            mats.append(F)
        out[sp_] = mats
    return out


def check_spin_fock(family, name, coords, factors, tol=2e-6):
    g = make_grid(coords, len(factors), seed=13)
    g2 = make_grid(coords, len(factors), seed=29)
    Pa, Pb = g.P, g2.P
    seeds = [Seeds(coords, g, f) for f in factors]
    F = spin_fock(family, coords, g, seeds, Pa, Pb)
    h, worst = 1e-6, 0.0
    rng = np.random.default_rng(5)
    nbf = g.chi.shape[0]
    for sp_ in ("a", "b"):
        for b in range(len(factors)):
            for _ in range(3):
                i, j = rng.integers(0, nbf, 2)
                Ap = [x.copy() for x in Pa]
                Am = [x.copy() for x in Pa]
                Bp = [x.copy() for x in Pb]
                Bm = [x.copy() for x in Pb]
                if sp_ == "a":
                    Ap[b][i, j] += h
                    Am[b][i, j] -= h
                else:
                    Bp[b][i, j] += h
                    Bm[b][i, j] -= h
                fd = (spin_energy(family, g, coords, seeds, Ap, Bp)
                      - spin_energy(family, g, coords, seeds, Am, Bm)) / (2 * h)
                worst = max(worst, abs(F[sp_][b][i, j] - fd)
                            / max(1e-8, abs(fd)))
    return (f"{name:14s} {family:9s} vxc_spin vs FD(Exc)", worst < tol, worst)


def check_spin_fxc(family, name, coords, factors, tol=5e-5):
    g = make_grid(coords, len(factors), seed=13)
    g2 = make_grid(coords, len(factors), seed=29)
    Pa, Pb = g.P, g2.P
    seeds = [Seeds(coords, g, f) for f in factors]
    rng = np.random.default_rng(17)
    nbf = g.chi.shape[0]
    dA, dB = [], []
    for _ in factors:
        X = rng.standard_normal((nbf, nbf)) * 0.05
        Y = rng.standard_normal((nbf, nbf)) * 0.05
        dA.append(X + X.T)
        dB.append(Y + Y.T)

    ing = spin_ingredients(coords, g, seeds, Pa, Pb)
    ping = spin_ingredients(coords, g, seeds, dA, dB)
    res = _libxc_spin(family, ing, 2)
    extra = {}
    for k, v in ping.items():
        if k.startswith("_"):
            continue
        if k.startswith("grad_"):
            extra["grad_rho_" + k[len("grad_"):].replace("_", "_p1_", 1)] = v
        else:
            extra[f"{k}_p1"] = v
    npts = ing["rho_a"].size
    ch = _eval_spin(fxc_channels_spin(family, coords=coords),
                    _spin_pool(ing, res, extra), npts)

    G = {}
    for sp_ in ("a", "b"):
        mats = []
        for sd in seeds:
            M = np.zeros((nbf, nbf))
            for key, c in ch.items():
                if not key.endswith(f"_{sp_}") and f"_{sp_}_" not in key:
                    continue
                seed_key = ("rho" if key == f"rho_{sp_}" else
                            "tau" if key == f"tau_{sp_}" else
                            "grad_" + key.split(f"_{sp_}_", 1)[1])
                M += np.einsum("g,uvg->uv", g.w * c, sd.value[seed_key])
            mats.append(M)
        G[sp_] = mats

    h = 1e-5
    Fp = spin_fock(family, coords, g, seeds,
                   [x + h * d for x, d in zip(Pa, dA)],
                   [x + h * d for x, d in zip(Pb, dB)])
    Fm = spin_fock(family, coords, g, seeds,
                   [x - h * d for x, d in zip(Pa, dA)],
                   [x - h * d for x, d in zip(Pb, dB)])
    worst = 0.0
    for sp_ in ("a", "b"):
        for b in range(len(factors)):
            fd = (Fp[sp_][b] - Fm[sp_][b]) / (2 * h)
            scale = max(1e-8, np.abs(fd).max())
            worst = max(worst, np.abs(G[sp_][b] - fd).max() / scale)
    return (f"{name:14s} {family:9s} fxc_spin vs FD(vxc)", worst < tol, worst)


def main():
    checks = []
    for name, (coords, factors) in SYSTEMS.items():
        for family in FAMILY_FUNCTIONAL:
            checks.append(check_fock(family, name, coords, factors))
            checks.append(check_fxc(family, name, coords, factors))
            checks.append(check_spin_fock(family, name, coords, factors))
            checks.append(check_spin_fxc(family, name, coords, factors))
    bad = [c for c in checks if not c[1]]
    for label, ok, err in checks:
        print(f"  {'ok  ' if ok else 'FAIL'} {label}  rel={err:.2e}")
    tag = "OK " if not bad else "FAIL"
    print(f"[{tag}] curvilinear_response_validate: {len(checks)} checks, "
          f"{len(bad)} failures")
    raise SystemExit(0 if not bad else 1)


if __name__ == "__main__":
    main()
