"""Validate the one-free-index and pointwise C kernels of the catalog: the
Fock diagonal (xck_*_o1_diag) and the XC nuclear gradient (xck_*_g1,
xck_*_gg), compiled from the emitted C++ exactly as the catalog ships it.

  1. o1_diag == diag(o1) on random operands, every family, r/ua/ub;
  2. basis class: sum over atom A's functions of the g1 rows == FD of Exc
     moving A's basis centers only (per spin channel for ua/ub);
  3. grid class: sum over A's points of gg == FD of Exc moving A's grid
     points only;
  4. weight class: the order-0 kernel with w := dw/dX (toy weights that
     depend on the points and centers, translation invariant);
  5. basis + grid + weight == FD of Exc moving atom A completely, and the
     translational sum rule: summed over atoms, the three classes cancel.

The basis functions are Gaussians with polynomial prefactors (s, p and
d-like) differentiated symbolically, so every collocation derivative is
exact; explicit polynomial functionals stand in for Libxc. FD is
Richardson-extrapolated central differences.
"""

from __future__ import annotations

import ctypes
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import sympy as sp

from ..catalog import FAMILIES, CatalogEntry, _emit_kind, _integrand_for
from ..emitters.cbackend import (_EVALUATOR_HPP, emit_exc_cpp, emit_exc_hpp,
                                 emit_kernel_cpp, emit_kernel_hpp, scal_order)
from ..emitters.codegen import collapse
from ..engine.gradient import GRADIENT_FAMILIES

P = ctypes.POINTER(ctypes.c_double)
_H6 = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
_AX = "xyz"

# --- the compiled kernels ------------------------------------------------------


def build_library(td: Path):
    """Emit and compile every diag/g1/gg kernel plus the o1 and o0 kernels
    they are checked against."""
    inc = td / "include" / "xckernel"
    (inc / "kernels").mkdir(parents=True)
    (inc / "evaluator.hpp").write_text(_EVALUATOR_HPP)
    src = td / "src"
    src.mkdir()
    scal = {}
    for fam in FAMILIES:
        spins = ("r", "ua", "ub")
        kinds = [(s, "diag") for s in spins]
        if fam in GRADIENT_FAMILIES:
            kinds += [(s, "g1") for s in spins] + [("r", "gg"), ("u", "gg")]
        for spin, kind in kinds:
            e = CatalogEntry(fam, spin, 1, kind=kind)
            hpp, cpp, m = _emit_kind(e)
            (inc / "kernels" / f"{e.name}.hpp").write_text(hpp)
            (src / f"{e.name}.cpp").write_text(cpp)
            scal[e.name] = m["scal_names"]
        for spin in spins:
            e = CatalogEntry(fam, spin, 1)
            ck = collapse(_integrand_for(e))
            (inc / "kernels" / f"{e.name}.hpp").write_text(
                emit_kernel_hpp(ck, e.name))
            (src / f"{e.name}.cpp").write_text(emit_kernel_cpp(ck, e.name))
            scal[e.name] = scal_order(ck)
        name0 = CatalogEntry(fam, "r", 0).name
        (inc / "kernels" / f"{name0}.hpp").write_text(emit_exc_hpp(name0))
        (src / f"{name0}.cpp").write_text(emit_exc_cpp(name0))
    lib = td / "libxckt.so"
    # a small grid block: several blocks and a remainder at the test sizes
    subprocess.run(["c++", "-std=c++17", "-O1", "-shared", "-fPIC",
                    "-DXCKERNEL_GRID_BLOCK=16", "-I", str(td / "include"),
                    *sorted(str(p) for p in src.glob("*.cpp")),
                    "-o", str(lib)], check=True)
    return ctypes.CDLL(str(lib)), scal


def _ptr(a):
    return None if a is None else np.ascontiguousarray(a).ctypes.data_as(P)


def _scal(names, ops):
    arrs = [np.ascontiguousarray(ops[n], dtype=float) for n in names]
    return (P * len(arrs))(*[a.ctypes.data_as(P) for a in arrs]), arrs


def call_rows(lib, scal, name, ng, nbf, nrow, arrays, ops):
    """diag (nrow=1, 4 arrays) or g1 (nrow=3, 8 arrays) kernel."""
    f = getattr(lib, name)
    f.restype = ctypes.c_int
    sp_, keep = _scal(scal[name], ops)
    out = np.zeros((nrow, nbf))
    arr = [np.ascontiguousarray(a) if a is not None else None for a in arrays]
    rc = f(ctypes.c_int64(ng), ctypes.c_int64(nbf), *[_ptr(a) for a in arr],
           sp_, out.ctypes.data_as(P))
    assert rc == 0
    return out


def call_matrix(lib, scal, name, ng, nbf, arrays, ops):
    f = getattr(lib, name)
    f.restype = ctypes.c_int
    sp_, keep = _scal(scal[name], ops)
    out = np.zeros((nbf, nbf))
    arr = [np.ascontiguousarray(a) for a in arrays]
    rc = f(ctypes.c_int64(ng), ctypes.c_int64(nbf), *[_ptr(a) for a in arr],
           sp_, out.ctypes.data_as(P))
    assert rc == 0
    return out


def call_points(lib, scal, name, ng, ops):
    f = getattr(lib, name)
    f.restype = ctypes.c_int
    sp_, keep = _scal(scal[name], ops)
    out = np.zeros((3, ng))
    rc = f(ctypes.c_int64(ng), sp_, out.ctypes.data_as(P))
    assert rc == 0
    return out


def call_exc(lib, name, w, rho, zk):
    f = getattr(lib, name)
    f.restype = ctypes.c_double
    w, rho, zk = (np.ascontiguousarray(a) for a in (w, rho, zk))
    return f(ctypes.c_int64(len(w)), w.ctypes.data_as(P),
             rho.ctypes.data_as(P), zk.ctypes.data_as(P))


# --- exact collocation ---------------------------------------------------------

_R = sp.symbols("x y z", real=True)
_C = sp.symbols("cx cy cz", real=True)
#: (atom, exponent, prefactor monomial exponents)
BASIS = [(0, 0.9, (0, 0, 0)), (0, 1.3, (1, 0, 0)), (1, 0.7, (0, 1, 0)),
         (1, 1.1, (0, 0, 1)), (2, 0.8, (1, 1, 0)), (2, 1.2, (0, 0, 0)),
         (2, 1.0, (0, 1, 1))]
BF_ATOM = np.array([b[0] for b in BASIS])


def _colloc_funcs():
    d = [r - c for r, c in zip(_R, _C)]
    out = []
    for _, a, (px, py, pz) in BASIS:
        chi = d[0]**px * d[1]**py * d[2]**pz \
            * sp.exp(-a * (d[0]**2 + d[1]**2 + d[2]**2))
        g = [sp.diff(chi, r) for r in _R]
        h = [[sp.diff(g[i], _R[j]) for j in range(3)] for i in range(3)]
        lap = h[0][0] + h[1][1] + h[2][2]
        dl = [sp.diff(lap, r) for r in _R]
        exprs = [chi] + g + [h[i][j] for i in range(3) for j in range(3)] \
            + [lap] + dl
        out.append(sp.lambdify(_R + _C, exprs, "numpy"))
    return out


_FUNCS = _colloc_funcs()


def colloc(centers, pts):
    """chi (nbf,ng), dchi (3,nbf,ng), hess (3,3,nbf,ng), lapl (nbf,ng),
    dlapl (3,nbf,ng)."""
    ng = len(pts)
    vals = np.empty((len(BASIS), 17, ng))
    for u, (f, (atom, _, _)) in enumerate(zip(_FUNCS, BASIS)):
        res = f(pts[:, 0], pts[:, 1], pts[:, 2], *centers[atom])
        vals[u] = [np.broadcast_to(np.asarray(r, float), (ng,)) for r in res]
    chi = vals[:, 0]
    dchi = np.transpose(vals[:, 1:4], (1, 0, 2))
    hess = np.transpose(vals[:, 4:13].reshape(-1, 3, 3, ng), (1, 2, 0, 3))
    return chi, dchi, hess, vals[:, 13], np.transpose(vals[:, 14:17],
                                                      (1, 0, 2))


def pack6(h):
    return np.stack([h[i, j] for i, j in _H6])


def fields(D, col):
    """Every field of one density matrix (D symmetric)."""
    chi, dchi, hess, lap, dlap = col
    rho = np.einsum("uv,ug,vg->g", D, chi, chi)
    grad = 2 * np.einsum("uv,iug,vg->ig", D, dchi, chi)
    tau = 0.5 * np.einsum("uv,iug,ivg->g", D, dchi, dchi)
    lapl = 2 * (np.einsum("uv,ug,vg->g", D, lap, chi)
                + np.einsum("uv,iug,ivg->g", D, dchi, dchi))
    hrho = 2 * (np.einsum("uv,ijug,vg->ijg", D, hess, chi)
                + np.einsum("uv,iug,jvg->ijg", D, dchi, dchi))
    gtau = np.einsum("uv,kiug,ivg->kg", D, hess, dchi)
    glapl = 2 * (np.einsum("uv,kug,vg->kg", D, dchi, lap)
                 + np.einsum("uv,kug,vg->kg", D, dlap, chi)
                 + 2 * np.einsum("uv,kiug,ivg->kg", D, hess, dchi))
    return {"rho": rho, "grad": grad, "tau": tau, "lapl": lapl,
            "hess": hrho, "gtau": gtau, "glapl": glapl}


# --- explicit test functionals ---------------------------------------------------

_FAM_VARS = {"lda": ("rho",), "gga": ("rho", "sigma"),
             "mgga_tau": ("rho", "sigma", "tau"),
             "mgga_lapl": ("rho", "sigma", "lapl"),
             "mgga": ("rho", "sigma", "lapl", "tau")}

_rho, _sig, _tau, _lap = sp.symbols("rho sigma tau lapl", real=True)
_FR = (_rho**2 + sp.Rational(3, 10) * _sig * _rho + sp.Rational(1, 5) * _tau**2
       + sp.Rational(1, 10) * _rho * _tau + sp.Rational(1, 20) * _sig * _tau
       + sp.Rational(3, 20) * _lap * _rho + sp.Rational(1, 15) * _lap**2
       + sp.Rational(1, 25) * _lap * _sig)
_RVARS = {"rho": _rho, "sigma": _sig, "tau": _tau, "lapl": _lap}

_ra, _rb, _saa, _sab, _sbb, _la, _lb, _ta, _tb = sp.symbols(
    "ra rb saa sab sbb la lb ta tb", real=True)
_FU = (_ra**2 + sp.Rational(4, 5) * _rb**2 + sp.Rational(3, 10) * _ra * _rb
       + sp.Rational(1, 5) * _saa * _rb + sp.Rational(1, 4) * _sab * (_ra + _rb)
       + sp.Rational(1, 10) * _sbb * _ra + sp.Rational(1, 5) * _ta * _rb
       + sp.Rational(3, 20) * _tb**2 + sp.Rational(1, 8) * _ta * _tb
       + sp.Rational(3, 25) * _la * _rb + sp.Rational(1, 11) * _lb * _ra
       + sp.Rational(1, 20) * _la * _lb + sp.Rational(1, 30) * _saa * _tb)
#: polarized variable -> (Libxc array, component)
_UVARS = {"rho": [(_ra, "vrho_0"), (_rb, "vrho_1")],
          "sigma": [(_saa, "vsigma_0"), (_sab, "vsigma_1"),
                    (_sbb, "vsigma_2")],
          "lapl": [(_la, "vlapl_0"), (_lb, "vlapl_1")],
          "tau": [(_ta, "vtau_0"), (_tb, "vtau_1")]}


def functional_r(fam, f):
    """(e, {v-array name: values}) restricted."""
    zero = {s: 0 for v, s in _RVARS.items() if v not in _FAM_VARS[fam]}
    F = _FR.subs(zero)
    syms = list(_RVARS.values())
    vals = [f["rho"], np.einsum("ig,ig->g", f["grad"], f["grad"]), f["tau"],
            f["lapl"]]
    e = sp.lambdify(syms, F, "numpy")(*vals) * np.ones_like(f["rho"])
    vs = {f"v{v}": sp.lambdify(syms, sp.diff(F, _RVARS[v]), "numpy")(*vals)
          * np.ones_like(f["rho"]) for v in _FAM_VARS[fam]}
    return e, vs


def functional_u(fam, fa, fb):
    zero = {s: 0 for v, lst in _UVARS.items() if v not in _FAM_VARS[fam]
            for s, _ in lst}
    F = _FU.subs(zero)
    syms = [_ra, _rb, _saa, _sab, _sbb, _la, _lb, _ta, _tb]
    dot = lambda a, b: np.einsum("ig,ig->g", a, b)
    vals = [fa["rho"], fb["rho"], dot(fa["grad"], fa["grad"]),
            dot(fa["grad"], fb["grad"]), dot(fb["grad"], fb["grad"]),
            fa["lapl"], fb["lapl"], fa["tau"], fb["tau"]]
    one = np.ones_like(fa["rho"])
    e = sp.lambdify(syms, F, "numpy")(*vals) * one
    vs = {name: sp.lambdify(syms, sp.diff(F, s), "numpy")(*vals) * one
          for v in _FAM_VARS[fam] for s, name in _UVARS[v]}
    return e, vs


def field_ops(f, sfx=""):
    """Per-point operands of one channel's fields, by scal name."""
    ops = {}
    for i, a in enumerate(_AX):
        ops[f"grad_rho{sfx}_{a}"] = f["grad"][i]
        ops[f"grad_tau{sfx}_{a}"] = f["gtau"][i]
        ops[f"grad_lapl_rho{sfx}_{a}"] = f["glapl"][i]
    for (i, j) in _H6:
        ops[f"hess_rho{sfx}_{_AX[i]}{_AX[j]}"] = f["hess"][i, j]
    return ops


# --- the synthetic system ----------------------------------------------------------

class System:
    natom, ng, beta = 3, 60, 0.15

    def __init__(self, seed=5):
        rng = np.random.default_rng(seed)
        self.centers = rng.uniform(-0.7, 0.7, (self.natom, 3))
        self.pts = rng.uniform(-1.4, 1.4, (self.ng, 3))
        self.parent = np.arange(self.ng) % self.natom
        self.w0 = rng.uniform(0.2, 1.0, self.ng)
        nbf = len(BASIS)

        def dm():
            M = rng.standard_normal((nbf, nbf))
            return 0.2 * M @ M.T / nbf + 0.6 * np.eye(nbf)
        self.Da, self.Db = dm(), dm()

    def weights(self, centers, pts):
        """Toy partition weights: depend on every center and the point,
        invariant under a rigid translation."""
        d2 = ((pts[:, None, :] - centers[None, :, :]) ** 2).sum(-1).sum(-1)
        return self.w0 * np.exp(-self.beta * d2)

    def dweights(self, A, d):
        """dw/dX_{A,d}: the center moves, and so do A's points."""
        w = self.weights(self.centers, self.pts)
        dwdC = 2 * self.beta * (self.pts[:, d] - self.centers[A, d]) * w
        dwdr = -2 * self.beta * (self.pts[:, d][:, None]
                                 - self.centers[None, :, d]).sum(1) * w
        return dwdC + (self.parent == A) * dwdr

    def energy(self, fam, spin, centers_a, centers_b, pts, w):
        """Exc with the alpha and beta basis functions at independent
        centers (so each channel's basis class can be isolated)."""
        ca, cb = colloc(centers_a, pts), colloc(centers_b, pts)
        if spin == "r":
            # restricted: one density matrix D = Da + Db on one basis
            e, _ = functional_r(fam, fields(self.Da + self.Db, ca))
        else:
            e, _ = functional_u(fam, fields(self.Da, ca), fields(self.Db, cb))
        return float(np.dot(w, e))


def richardson(f, h=1e-3):
    return (8 * (f(h) - f(-h)) - (f(2 * h) - f(-2 * h))) / (12 * h)


# --- the checks ------------------------------------------------------------------

class Report:
    def __init__(self):
        self.tested = self.failures = 0

    def check(self, label, got, ref, tol, scale=None):
        got, ref = np.asarray(got), np.asarray(ref)
        self.tested += 1
        if scale is None:
            scale = max(np.abs(ref).max(), np.abs(got).max(), 1e-14)
        rel = np.abs(got - ref).max() / scale
        ok = rel < tol
        if not ok:
            self.failures += 1
        print(f"  [{'OK' if ok else 'FAIL'}] {label}: max rel {rel:.2e}")


def check_diag(lib, scal, rep, nbf=5, ng=37, seed=2):
    rng = np.random.default_rng(seed)
    chi = rng.standard_normal((nbf, ng))
    dchi = rng.standard_normal((3, nbf, ng))
    lap = rng.standard_normal((nbf, ng))
    hess = rng.standard_normal((6, nbf, ng))
    for fam in FAMILIES:
        for spin in ("r", "ua", "ub"):
            full = CatalogEntry(fam, spin, 1).name
            diag = CatalogEntry(fam, spin, 1, kind="diag").name
            if scal[diag] != scal[full]:
                rep.check(f"{diag} operand order", 1.0, 0.0, 0.5)
                continue
            ops = {n: rng.standard_normal(ng) for n in scal[full]}
            arrays = (chi, dchi, lap, hess)
            F = call_matrix(lib, scal, full, ng, nbf, arrays, ops)
            d = call_rows(lib, scal, diag, ng, nbf, 1, arrays, ops)[0]
            rep.check(f"{diag} == diag({full})", d, np.diag(F), 1e-13)


def check_gradient(lib, scal, rep, fam, spin, sysm):
    """spin 'r' or 'u'."""
    c0, p0 = sysm.centers, sysm.pts
    w = sysm.weights(c0, p0)
    col = colloc(c0, p0)
    chi, dchi, hess, lap, dlap = col
    nbf, ng = chi.shape
    Dr = sysm.Da + sysm.Db
    if spin == "r":
        f = fields(Dr, col)
        e, vs = functional_r(fam, f)
        ops = {"w": w, **field_ops(f), **vs}
        chans = [("r", Dr)]
        rho_tot = f["rho"]
    else:
        fa, fb = fields(sysm.Da, col), fields(sysm.Db, col)
        e, vs = functional_u(fam, fa, fb)
        ops = {"w": w, **field_ops(fa, "_a"), **field_ops(fb, "_b"), **vs}
        chans = [("ua", sysm.Da), ("ub", sysm.Db)]
        rho_tot = fa["rho"] + fb["rho"]

    # ----- basis class: per channel -----
    rows = {}
    for ch, D in chans:
        arrays = (chi, dchi, lap, pack6(hess), dlap,
                  D @ chi, np.einsum("uv,ivg->iug", D, dchi), D @ lap)
        rows[ch] = call_rows(lib, scal, f"xck_{fam}_{ch}_g1", ng, nbf, 3,
                             arrays, ops)
    basis = np.zeros((sysm.natom, 3))
    for ch, _ in chans:
        for A in range(sysm.natom):
            basis[A] += rows[ch][:, BF_ATOM == A].sum(1)

    for ch, _ in chans:
        got, fd = [], []
        for A in range(sysm.natom):
            for d in range(3):
                def E(h, A=A, d=d, ch=ch):
                    c = c0.copy()
                    c[A, d] += h
                    ca = c if ch in ("r", "ua") else c0
                    cb = c if ch == "ub" else c0
                    return sysm.energy(fam, spin, ca, cb, p0, w)
                fd.append(richardson(E))
                got.append(rows[ch][d, BF_ATOM == A].sum())
        rep.check(f"{fam:9s} {ch:2s} g1 basis class vs FD", got, fd, 1e-9)

    # ----- grid class -----
    gg = call_points(lib, scal, f"xck_{fam}_{spin}_gg", ng, ops)
    grid = np.array([[gg[d, sysm.parent == A].sum() for d in range(3)]
                     for A in range(sysm.natom)])
    fd = np.zeros_like(grid)
    for A in range(sysm.natom):
        for d in range(3):
            def E(h, A=A, d=d):
                p = p0.copy()
                p[sysm.parent == A, d] += h
                return sysm.energy(fam, spin, c0, c0, p, w)
            fd[A, d] = richardson(E)
    rep.check(f"{fam:9s} {spin:2s} gg grid class vs FD", grid, fd, 1e-9)

    # ----- weight class: the order-0 kernel with w := dw/dX -----
    weight = np.zeros_like(grid)
    zk = e / rho_tot
    for A in range(sysm.natom):
        for d in range(3):
            weight[A, d] = call_exc(lib, f"xck_{fam}_r_o0",
                                    sysm.dweights(A, d), rho_tot, zk)

    # ----- all three classes vs moving atom A completely -----
    total = basis + grid + weight
    fd = np.zeros_like(total)
    for A in range(sysm.natom):
        for d in range(3):
            def E(h, A=A, d=d):
                c, p = c0.copy(), p0.copy()
                c[A, d] += h
                p[sysm.parent == A, d] += h
                return sysm.energy(fam, spin, c, c, p, sysm.weights(c, p))
            fd[A, d] = richardson(E)
    rep.check(f"{fam:9s} {spin:2s} basis+grid+weight vs FD", total, fd, 1e-9)

    # ----- translational sum rule -----
    scale = max(np.abs(basis).max(), np.abs(grid).max(), np.abs(weight).max())
    rep.check(f"{fam:9s} {spin:2s} translational sum rule",
              total.sum(0), np.zeros(3), 1e-12, scale=scale)


def main():
    rep = Report()
    with tempfile.TemporaryDirectory() as td:
        print("building the diag/g1/gg kernels ...", flush=True)
        lib, scal = build_library(Path(td))
        print("Fock diagonal")
        check_diag(lib, scal, rep)
        sysm = System()
        for fam in GRADIENT_FAMILIES:
            print(f"nuclear gradient: {fam}")
            for spin in ("r", "u"):
                check_gradient(lib, scal, rep, fam, spin, sysm)
    status = "OK " if rep.failures == 0 else "FAIL"
    print(f"[{status}] gradient_validate: {rep.tested} checks, "
          f"{rep.failures} failures")
    return rep.failures


if __name__ == "__main__":
    raise SystemExit(main())
