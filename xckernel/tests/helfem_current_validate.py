"""Validate HelFEM's current-corrected tau-meta-GGA channels in HelFEM's
convention: complex basis functions, real symmetric density matrices.

The emitted header (``helfemwriter.emit_header``) is compiled behind a
thin extern "C" shim and called through ctypes, so what is checked is
the code HelFEM includes.  The model: a few functions

    chi_k = (x + i s_k y)^|m_k| z^n_k exp(-a_k r^2),   s_k = sign(m_k),

eigenfunctions of L_z with mixed m (including a +1/-1 pair with equal
radial parts), on random points with random positive weights, and a
polynomial functional f(rho, sigma, tau~) whose Libxc arrays come from
SymPy.  Finite differences act on E[P] = sum_g w f directly, so they
share no packing or chain rule with the channels.  Fields:

    rho = sum P_uv Re(chi_u^* chi_v)
    j   = sum P_uv Im(chi_u^* grad chi_v)
    tau = 1/2 sum P_uv Re(grad chi_u^* . grad chi_v)

Checks:
  * Fock: sum_g w [u rho_D + v.grad rho_D + w_tau tau_D + y.j_D] equals
    dE/dt at P + t D for random real symmetric D (restricted, and per
    spin);
  * fxc: the channels of a perturbation D1 (with j_D1 != 0), contracted
    with the fields of D2, equal d/dt of that Fock contraction at P + t D1;
  * single orbital: tau~ = |grad rho|^2 / (8 rho) for psi = sum c chi;
  * j = 0 (equal +1/-1 occupations): the cmgga_tau channels equal the
    mgga_tau ones (u, v, w) and the current channels vanish;
  * pure-m: for an m-blocked density, the _2d kernels on the (s, z)
    gradient and phi current equal the _3d kernels projected on the
    cylindrical frame; and the 3D expressions reduce to the 2D ones
    symbolically.

Run: python -m xckernel.tests.helfem_current_validate
"""

from __future__ import annotations

import ctypes
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import sympy as sp

from ..emitters.helfemwriter import emit_header

BASIS = [(0, 0, 0.8), (0, 1, 1.1), (1, 0, 0.9), (-1, 0, 0.9),
         (1, 1, 0.7), (-2, 0, 1.0)]


class Report:
    def __init__(self):
        self.tested = self.failures = 0

    def check(self, label, got, ref, tol):
        got, ref = np.asarray(got, float), np.asarray(ref, float)
        err = np.max(np.abs(got - ref)) / max(np.max(np.abs(ref)), 1e-300)
        ok = err <= tol
        self.tested += 1
        self.failures += not ok
        print(f"  [{'OK' if ok else 'FAIL'}] {label}: max rel {err:.2e}")


# --------------------------------------------------------------------------
# the emitted header, compiled
# --------------------------------------------------------------------------

class Kernels:
    """Every emitted function, callable as k(name, **operands) -> dict."""

    def __init__(self, td: Path):
        src = emit_header()
        self.sigs = {}
        for m in re.finditer(r"static inline void (\w+)\((.*?)\) \{", src):
            ins, outs = [], []
            for a in m.group(2).split(","):
                a = a.strip()
                (outs if "&" in a else ins).append(a.split()[-1])
            self.sigs[m.group(1)] = (ins, outs)
        shim = ['#include "xck.h"', 'extern "C" {']
        for n, (ins, outs) in self.sigs.items():
            args = [f"i[{k}]" for k in range(len(ins))] + \
                [f"o[{k}]" for k in range(len(outs))]
            shim.append(f"void w_{n}(const double* i, double* o) "
                        f"{{ helfem::xckernel::{n}({', '.join(args)}); }}")
        shim.append("}")
        (td / "xck.h").write_text(src)
        (td / "shim.cpp").write_text("\n".join(shim))
        subprocess.run(["c++", "-std=c++17", "-O1", "-Wall", "-Werror",
                        "-shared", "-fPIC", str(td / "shim.cpp"),
                        "-o", str(td / "shim.so")], check=True, cwd=td)
        self.dll = ctypes.CDLL(str(td / "shim.so"))

    def __call__(self, name, **ops):
        ins, outs = self.sigs[name]
        ng = len(next(iter(ops.values())))
        res = {o: np.empty(ng) for o in outs}
        fn = getattr(self.dll, f"w_{name}")
        P = ctypes.POINTER(ctypes.c_double)
        for g in range(ng):
            i = (ctypes.c_double * len(ins))(*[float(ops[n][g])
                                                for n in ins])
            o = (ctypes.c_double * len(outs))()
            fn(ctypes.cast(i, P), ctypes.cast(o, P))
            for k, n in enumerate(outs):
                res[n][g] = o[k]
        return res


# --------------------------------------------------------------------------
# the model: basis, fields, functional
# --------------------------------------------------------------------------

def basis(pts):
    """chi (K, ng) and grad chi (K, 3, ng), complex."""
    x, y, z = sp.symbols("x y z", real=True)
    chis, grads = [], []
    for m, n, a in BASIS:
        s = 1 if m >= 0 else -1
        e = (x + sp.I * s * y) ** abs(m) * z ** n * sp.exp(-a * (x*x + y*y
                                                              + z*z))
        f = sp.lambdify((x, y, z), e, "numpy")
        df = [sp.lambdify((x, y, z), sp.diff(e, c), "numpy") for c in
              (x, y, z)]
        X, Y, Z = pts
        chis.append(np.broadcast_to(f(X, Y, Z), X.shape).astype(complex))
        grads.append([np.broadcast_to(d(X, Y, Z), X.shape).astype(complex)
                      for d in df])
    return np.array(chis), np.array(grads)


def fields(P, chi, dchi):
    """rho, grad rho (3), tau, j (3) of a real symmetric P."""
    rho = np.einsum("uv,ug,vg->g", P, chi.conj(), chi).real
    grad = 2 * np.einsum("uv,ug,vcg->cg", P, chi.conj(), dchi).real
    tau = 0.5 * np.einsum("uv,ucg,vcg->g", P, dchi.conj(), dchi).real
    j = np.einsum("uv,ug,vcg->cg", P, chi.conj(), dchi).imag
    return rho, grad, tau, j


def functional(polarized, rng):
    """f, its argument symbols, and lambdified Libxc arrays by name."""
    from ..engine.current_channels import _v1, _v2, libxc_derivative_names
    if polarized:
        args = sp.symbols("ra rb saa sab sbb ta tb", real=True)
        kinds = [("rho", 0), ("rho", 1), ("sigma", 0), ("sigma", 1),
                 ("sigma", 2), ("tau", 0), ("tau", 1)]
    else:
        args = sp.symbols("r s t", real=True)
        kinds = [("rho", None), ("sigma", None), ("tau", None)]
    c = rng.uniform(0.2, 1.0, (len(args), len(args)))
    f = sum(c[i, j] * args[i] * args[j] for i in range(len(args))
            for j in range(i, len(args)))
    f += sum(rng.uniform(0.1, 0.5) * a ** 3 for a in args)
    first, second = libxc_derivative_names(polarized)
    arrays = {}
    pos = {k: i for i, k in enumerate(kinds)}
    for k in first:
        arrays[_v1(k[0], k[1], polarized).name] = sp.lambdify(
            args, sp.diff(f, args[pos[k]]), "numpy")
    for k1, k2 in second:
        arrays[_v2(k1[0], k1[1], k2[0], k2[1], polarized).name] = \
            sp.lambdify(args, sp.diff(f, args[pos[k1]], args[pos[k2]]),
                        "numpy")
    return sp.lambdify(args, f, "numpy"), arrays


def _ingr(F, polarized):
    """f's arguments from the fields (per spin when polarized)."""
    if not polarized:
        rho, g, tau, j = F
        return (rho, (g * g).sum(0), tau - (j * j).sum(0) / (2 * rho))
    (ra, ga, ta, ja), (rb, gb, tb, jb) = F
    return (ra, rb, (ga * ga).sum(0), (ga * gb).sum(0), (gb * gb).sum(0),
            ta - (ja * ja).sum(0) / (2 * ra), tb - (jb * jb).sum(0) / (2 * rb))


def _libxc(arrays, F, polarized):
    args = _ingr(F, polarized)
    ng = len(args[0])
    return {n: np.broadcast_to(f(*args), (ng,)).astype(float)
            for n, f in arrays.items()}


AX = ("x", "y", "z")


def _ops(F, F1=None, polarized=False):
    """Operand dict of the 3D kernels."""
    ops = {}
    spins = ("a", "b") if polarized else ("",)
    Fs = F if polarized else (F,)
    F1s = (F1 if polarized else (F1,)) if F1 is not None else None
    for k, s in enumerate(spins):
        t = f"_{s}" if s else ""
        rho, g, tau, j = Fs[k]
        ops[f"rho{t}"] = rho
        for c, ax in enumerate(AX):
            ops[f"grad_rho{t}_{ax}"] = g[c]
            ops[f"jp{t}_{ax}"] = j[c]
        if F1s is not None:
            rho1, g1, tau1, j1 = F1s[k]
            ops[f"rho{t}_p1"] = rho1
            ops[f"tau{t}_p1"] = tau1
            for c, ax in enumerate(AX):
                ops[f"grad_rho{t}_p1_{ax}"] = g1[c]
                ops[f"jp{t}_p1_{ax}"] = j1[c]
    return ops


def _contract(ch, FD, s=""):
    """sum_g [u rho_D + v.grad_D + w tau_D + y.j_D] per point (no weight)."""
    rho, g, tau, j = FD
    t = f"_{s}" if s else ""
    u = ch[f"u{t}"]
    w = ch[f"w{t}"] if f"w{t}" in ch else ch.get("w_tau")
    v = [ch[f"v{t}_{ax}"] for ax in AX]
    y = [ch[f"y{t}_{ax}"] for ax in AX]
    return u * rho + sum(v[c] * g[c] + y[c] * j[c] for c in range(3)) \
        + w * tau


def _vxc_channels(K, F, lx, polarized):
    """Ground-state channels: u, y from the cmgga kernel; v from the shared
    gradient kernel; w = vtau."""
    if not polarized:
        ops = _ops(F)
        ops.update(lx)
        ch = K("xck_helfem_vxc_cmgga_tau_3d", **{n: ops[n] for n in
                                                  K.sigs["xck_helfem_vxc_cmgga_tau_3d"][0]})
        for c, ax in enumerate(AX):
            ch[f"v_{ax}"] = K("xck_helfem_vxc_grad", grad_rho_r=F[1][c],
                              vsigma=lx["vsigma"])["v_r"]
        ch["w_tau"] = lx["vtau"]
        return ch
    ops = _ops(F, polarized=True)
    ops.update(lx)
    nm = "xck_helfem_vxc_cmgga_tau_3d_spin"
    ch = K(nm, **{n: ops[n] for n in K.sigs[nm][0]})
    gname = "xck_helfem_vxc_grad_spin"
    for c, ax in enumerate(AX):
        gops = {"grad_rho_a_r": F[0][1][c], "grad_rho_b_r": F[1][1][c],
                "vsigma_0": lx["vsigma_0"], "vsigma_1": lx["vsigma_1"],
                "vsigma_2": lx["vsigma_2"]}
        r = K(gname, **{n: gops[n] for n in K.sigs[gname][0]})
        ch[f"v_a_{ax}"], ch[f"v_b_{ax}"] = r["v_a_r"], r["v_b_r"]
    ch["w_a"], ch["w_b"] = lx["vtau_0"], lx["vtau_1"]
    return ch


def _fxc_channels(K, F, F1, lx, polarized):
    nm = "xck_helfem_fxc_cmgga_tau_3d" + ("_spin" if polarized else "")
    ops = _ops(F, F1, polarized)
    ops.update(lx)
    return K(nm, **{n: ops[n] for n in K.sigs[nm][0]})


def _richardson(fun, h=1e-3):
    d = lambda hh: (fun(hh) - fun(-hh)) / (2 * hh)
    return (4 * d(h / 2) - d(h)) / 3


def _sym(rng, n, scale=1.0):
    A = rng.standard_normal((n, n)) * scale
    return A + A.T


def run(K, rep, rng):
    pts = rng.uniform(-1.3, 1.3, (3, 160))
    w = rng.uniform(0.5, 1.5, 160)
    chi, dchi = basis(pts)
    nb = len(BASIS)
    C = rng.standard_normal((nb, 3)) * 0.5
    C[0] += 1.5                       # keep rho comfortably positive
    P = C @ C.T
    Pb = P + 0.3 * np.outer(C[:, 1], C[:, 1])

    for polarized in (False, True):
        tag = "spin" if polarized else "restricted"
        f, arrays = functional(polarized, rng)

        def Fof(Pm):
            if polarized:
                return tuple(fields(Q, chi, dchi) for Q in Pm)
            return fields(Pm, chi, dchi)

        def E(Pm):
            return float(np.sum(w * f(*_ingr(Fof(Pm), polarized))))

        P0 = (P, Pb) if polarized else P
        F0 = Fof(P0)
        jn = np.abs(F0[0][3] if polarized else F0[3]).max()
        print(f"{tag}: |j| up to {jn:.2f}")

        def fock(Pm, D):
            F = Fof(Pm)
            ch = _vxc_channels(K, F, _libxc(arrays, F, polarized), polarized)
            if polarized:
                return sum(np.sum(w * _contract(ch, fields(D[k], chi, dchi),
                                                s))
                           for k, s in enumerate("ab"))
            return np.sum(w * _contract(ch, fields(D, chi, dchi)))

        def add(Pm, D, t):
            return tuple(a + t * b for a, b in zip(Pm, D)) if polarized \
                else Pm + t * D

        for k in range(3):
            D = tuple(_sym(rng, nb) for _ in "ab") if polarized \
                else _sym(rng, nb)
            ref = _richardson(lambda t: E(add(P0, D, t)))
            rep.check(f"{tag} Fock vs FD of E, direction {k}",
                      fock(P0, D), ref, 1e-8)

        for k in range(3):
            D1 = tuple(_sym(rng, nb) for _ in "ab") if polarized \
                else _sym(rng, nb)
            D2 = tuple(_sym(rng, nb) for _ in "ab") if polarized \
                else _sym(rng, nb)
            F1 = Fof(D1)
            j1 = np.abs(F1[0][3] if polarized else F1[3]).max()
            ch = _fxc_channels(K, F0, F1, _libxc(arrays, F0, polarized),
                               polarized)
            if polarized:
                got = sum(np.sum(w * _contract(ch, fields(D2[i], chi, dchi),
                                               s))
                          for i, s in enumerate("ab"))
            else:
                got = np.sum(w * _contract(ch, fields(D2, chi, dchi)))
            ref = _richardson(lambda t: fock(add(P0, D1, t), D2))
            rep.check(f"{tag} fxc vs FD of the Fock, pair {k} "
                      f"(|j_p1| up to {j1:.2f})", got, ref, 1e-8)


def single_orbital(rep, rng):
    pts = rng.uniform(-1.3, 1.3, (3, 200))
    chi, dchi = basis(pts)
    for label, c in (("chi(m=+1)", np.eye(len(BASIS))[2]),
                     ("m = 0, +1, -2 mix", np.array([0.9, 0, 0.6, 0, 0,
                                                     0.4]))):
        P = np.outer(c, c)
        rho, g, tau, j = fields(P, chi, dchi)
        tt = tau - (j * j).sum(0) / (2 * rho)
        tw = (g * g).sum(0) / (8 * rho)
        print(f"single orbital {label}: |j| up to {np.abs(j).max():.2f}")
        rep.check(f"single orbital {label}: tau~ = tau_W", tt, tw, 1e-12)


def zero_current(K, rep, rng):
    """Equal +1/-1 occupations (identical radial parts): j = 0, and the
    cmgga_tau kernels reduce to the mgga_tau ones."""
    pts = rng.uniform(-1.3, 1.3, (3, 120))
    chi, dchi = basis(pts)
    P = np.diag([1.0, 0.4, 0.7, 0.7, 0.0, 0.0])
    D = np.diag([0.3, -0.2, 0.5, 0.5, 0.0, 0.0])
    F, F1 = fields(P, chi, dchi), fields(D, chi, dchi)
    print(f"zero current: |j| = {np.abs(F[3]).max():.1e}, "
          f"|j_p1| = {np.abs(F1[3]).max():.1e}")
    f, arrays = functional(False, rng)
    lx = _libxc(arrays, F, False)
    ch = _fxc_channels(K, F, F1, lx, False)
    ops = _ops(F, F1)
    ops.update(lx)
    nm = "xck_helfem_fxc_mgga_tau_3d"
    ref = K(nm, **{n: ops[n] for n in K.sigs[nm][0]})
    for k in ("u", "v_x", "v_y", "v_z", "w_tau"):
        rep.check(f"j = 0: cmgga_tau {k} = mgga_tau {k}", ch[k], ref[k],
                  1e-13)
    _zero(rep, ch)


def _zero(rep, ch):
    m = max(np.abs(ch[f"y_{a}"]).max() for a in AX)
    rep.tested += 1
    ok = m < 1e-12
    rep.failures += not ok
    print(f"  [{'OK' if ok else 'FAIL'}] j = 0: current channels vanish: "
          f"max |y| {m:.1e}")


def purem(K, rep, rng):
    """m-blocked density: the _2d kernels on (s, z) and phi equal the _3d
    ones projected on the cylindrical frame."""
    pts = rng.uniform(-1.3, 1.3, (3, 120))
    chi, dchi = basis(pts)
    X, Y, _ = pts
    s = np.hypot(X, Y)
    es = np.array([X / s, Y / s, 0 * s])
    ez = np.array([0 * s, 0 * s, 1 + 0 * s])
    ephi = np.array([-Y / s, X / s, 0 * s])
    # m-blocks: {0: functions 0, 1}, {+1: 2, 4}, {-1: 3}, {-2: 5}
    blocks = [[0, 1], [2, 4], [3], [5]]

    def blocked(scale):
        Pm = np.zeros((len(BASIS), len(BASIS)))
        for b in blocks:
            A = rng.standard_normal((len(b), len(b))) * scale
            Pm[np.ix_(b, b)] = A @ A.T + (np.eye(len(b)) if scale > 0.5
                                          else 0)
        return Pm
    P, D = blocked(1.0), blocked(0.3)
    for polarized in (False, True):
        tag = "spin" if polarized else "restricted"
        f, arrays = functional(polarized, rng)
        if polarized:
            Pb = blocked(1.0)
            F = (fields(P, chi, dchi), fields(Pb, chi, dchi))
            F1 = (fields(D, chi, dchi), fields(blocked(0.3), chi, dchi))
        else:
            F, F1 = fields(P, chi, dchi), fields(D, chi, dchi)
        Fs = F if polarized else (F,)
        gphi = max(np.abs((ff[1] * ephi).sum(0)).max() for ff in Fs)
        jsz = max(np.abs((ff[3] * es).sum(0)).max() + np.abs(ff[3][2]).max()
                  for ff in Fs)
        print(f"pure-m {tag}: |grad_phi rho| {gphi:.1e}, |j_s|+|j_z| "
              f"{jsz:.1e}")
        lx = _libxc(arrays, F, polarized)
        three = _fxc_channels(K, F, F1, lx, polarized)
        ops = {}
        spins = ("a", "b") if polarized else ("",)
        F1s = F1 if polarized else (F1,)
        for k, sp_ in enumerate(spins):
            t = f"_{sp_}" if sp_ else ""
            for src, lab in ((Fs[k], ""), (F1s[k], "_p1")):
                rho, g, tau, j = src
                if lab:
                    ops[f"rho{t}_p1"], ops[f"tau{t}_p1"] = rho, tau
                else:
                    ops[f"rho{t}"] = rho
                ops[f"grad_rho{t}{lab}_mu"] = (g * es).sum(0)
                ops[f"grad_rho{t}{lab}_nu"] = (g * ez).sum(0)
                ops[f"jp{t}{lab}_phi"] = (j * ephi).sum(0)
        ops.update(lx)
        nm = "xck_helfem_fxc_cmgga_tau_2d" + ("_spin" if polarized else "")
        two = K(nm, **{n: ops[n] for n in K.sigs[nm][0]})
        for sp_ in spins:
            t = f"_{sp_}" if sp_ else ""
            v3 = np.array([three[f"v{t}_{a}"] for a in AX])
            y3 = np.array([three[f"y{t}_{a}"] for a in AX])
            wk = f"w{t}" if polarized else "w_tau"
            got = np.concatenate([two[f"u{t}"], two[f"v{t}_mu"],
                                  two[f"v{t}_nu"], two[wk],
                                  two[f"y{t}_phi"]])
            ref = np.concatenate([three[f"u{t}"], (v3 * es).sum(0),
                                  (v3 * ez).sum(0), three[wk],
                                  (y3 * ephi).sum(0)])
            rep.check(f"pure-m {tag}{t}: _2d = _3d on the cylindrical "
                      "frame", got, ref, 1e-12)


def symbolic(rep):
    """The 3D expressions with grad_phi = 0 and j_s = j_z = 0, written in
    (mu, nu, phi), are the 2D ones; at j = j_p1 = 0 the u, v, w channels
    are the engine's mgga_tau channels and y vanishes."""
    from ..engine.current_channels import fxc_current_channels as fx
    from ..engine.response import fxc_channels
    from ..engine.spin_kernel import fxc_channels_spin
    from ..inputs.basis import PROLATE_PUREM
    X = ("x", "y", "z")
    for pol in (False, True):
        tag = "spin" if pol else "restricted"
        three = fx(X, X, pol)
        two = fx(("mu", "nu"), ("phi",), pol)
        sub = {}
        for e in three.values():
            for s in e.free_symbols:
                n = s.name
                if n.startswith("grad_rho") and n.endswith("_z"):
                    sub[s] = 0
                elif n.startswith("jp") and n[-2:] in ("_x", "_y"):
                    sub[s] = 0
                elif n.startswith("grad_rho"):
                    sub[s] = sp.Symbol(n[:-1] + {"x": "mu", "y": "nu"}[n[-1]],
                                       real=True)
                elif n.startswith("jp"):
                    sub[s] = sp.Symbol(n[:-1] + "phi", real=True)
        bad = 0
        for k, e in two.items():
            k3 = k
            for a, b in (("_mu", "_x"), ("_nu", "_y"), ("_phi", "_z")):
                if k.endswith(a):
                    k3 = k[:-len(a)] + b
            bad += sp.expand(three[k3].subs(sub) - e) != 0
        rep.tested += 1
        rep.failures += bad != 0
        print(f"  [{'OK' if not bad else 'FAIL'}] symbolic {tag}: 3D "
              f"reduces to 2D ({len(two)} channels, {bad} mismatches)")
        zero = lambda d: {k: sp.expand(e.subs({s: 0 for s in e.free_symbols
                                               if s.name.startswith("jp")}))
                          for k, e in d.items()}
        for gax, cax, coords in ((X, X, None),
                                 (("mu", "nu"), ("phi",), PROLATE_PUREM)):
            mine = zero(fx(gax, cax, pol))
            ref = (fxc_channels_spin("mgga_tau", coords=coords) if pol
                   else fxc_channels("mgga_tau", coords=coords))
            bad = sum(sp.expand(mine[k] - ref[k]) != 0 for k in ref)
            bad += sum(mine[k] != 0 for k in mine if k not in ref)
            rep.tested += 1
            rep.failures += bad != 0
            print(f"  [{'OK' if not bad else 'FAIL'}] symbolic {tag} "
                  f"{'3D' if coords is None else 'pure-m'}: at j = 0 the "
                  f"channels are mgga_tau's ({len(ref)} compared, "
                  f"{bad} mismatches)")


def main():
    rng = np.random.default_rng(11)
    rep = Report()
    with tempfile.TemporaryDirectory() as td:
        print("compiling the emitted header ...", flush=True)
        K = Kernels(Path(td))
        print(f"{len(K.sigs)} functions")
        run(K, rep, rng)
        single_orbital(rep, rng)
        zero_current(K, rep, rng)
        purem(K, rep, rng)
    symbolic(rep)
    status = "OK " if rep.failures == 0 else "FAIL"
    print(f"[{status}] helfem_current_validate: {rep.tested} checks, "
          f"{rep.failures} failures")
    return rep.failures


if __name__ == "__main__":
    sys.exit(main())
