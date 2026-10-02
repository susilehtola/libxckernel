"""Validate the compiled C catalog on its derivative-tower interface
against physics: the kernels are emitted as the catalog ships them,
compiled, and called through runtime.Library with collocation towers and
tower-named fields computed independently here.

  1. Fock diagonal: o1_diag == diag(o1), every family, r/ua/ub (random
     towers);
  2. Fock matrix: o1 == dExc/dD by FD, r and ua;
  3. linear response: o2 == FD of the o1 kernel along a perturbation D1
     (perturbed-field towers, including the Laplacian the kernel forms);
  4. London orbitals: the GIAO field-derivative kernels (chi tower and
     basis-function centers) == the NumPy GIAO kernels fed the explicit
     center-scaled operands (validated against FD in london_validate);
  5. nuclear gradient: basis class (g1, per spin channel), grid class (gg)
     and weight class (o0 with w := dw/dX) against FD of Exc with each
     R-dependence isolated, their sum against moving an atom completely,
     and the translational sum rule.

The basis functions are Gaussians with polynomial prefactors (s, p and
d-like) differentiated symbolically to third order; explicit polynomial
functionals stand in for Libxc, with derivative arrays of any order from
the engine's own name registry. FD is Richardson-extrapolated.
"""

from __future__ import annotations

import ctypes
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import sympy as sp

from ..catalog import (FAMILIES, CatalogEntry, _emit_kind, _integrand_for,
                       _one_index_entries, gradient_families)
from ..emitters.cbackend import (_EVALUATOR_HPP, emit_exc_cpp, emit_exc_hpp,
                                 emit_kernel_cpp, emit_kernel_hpp)
from ..emitters.codegen import collapse
from ..emitters.tower import comp_index, components
from ..engine.deriv import LIBXC_MULTISET
from ..engine.gradient import GRADIENT_FAMILIES
from ..runtime import Library

P = ctypes.POINTER(ctypes.c_double)
_AX = "xyz"

# --- the compiled kernels ------------------------------------------------------


def build_library(td: Path) -> Library:
    """Emit and compile every diag/g1/gg kernel, the o1 kernels of every
    family, the o2 kernels of the gradient families and the o0 kernels."""
    inc = td / "include" / "xckernel"
    (inc / "kernels").mkdir(parents=True)
    (inc / "evaluator.hpp").write_text(_EVALUATOR_HPP)
    src = td / "src"
    src.mkdir()

    def write(name, hpp, cpp):
        (inc / "kernels" / f"{name}.hpp").write_text(hpp)
        (src / f"{name}.cpp").write_text(cpp)

    for fam in FAMILIES:
        matrix = [(s, 1, ()) for s in ("r", "ua", "ub")]
        # every o2 (each has a batched twin in _one_index_entries)
        matrix += [(s, 2, ()) for s in ("r", "ua", "ub")]
        matrix += [("st", 2, (+1,)), ("st", 2, (-1,))]
        for e in _one_index_entries(fam):
            hpp, cpp, _ = _emit_kind(e)
            write(e.name, hpp, cpp)
        for spin, order, par in matrix:
            e = CatalogEntry(fam, spin, order, par)
            ck = collapse(_integrand_for(e))
            write(e.name, emit_kernel_hpp(ck, e.name),
                  emit_kernel_cpp(ck, e.name))
        from ..catalog import GIAO_FAMILIES
        if fam in GIAO_FAMILIES:
            for spin in ("r", "ua", "ub"):
                e = CatalogEntry(fam, spin, 1, giao=True)
                hpp, cpp, _ = _emit_kind(e)
                write(e.name, hpp, cpp)
        n0 = CatalogEntry(fam, "r", 0).name
        write(n0, emit_exc_hpp(n0), emit_exc_cpp(n0))
    lib = td / "libxckt.so"
    # a small grid block: several blocks and a remainder at the test sizes
    subprocess.run(["c++", "-std=c++17", "-O1", "-shared", "-fPIC",
                    "-DXCKERNEL_GRID_BLOCK=16", "-I", str(td / "include"),
                    *sorted(str(p) for p in src.glob("*.cpp")),
                    "-o", str(lib)], check=True)
    return Library(str(lib))


def call_exc(lib: Library, name, w, rho, zk):
    f = getattr(lib._dll, name)
    f.restype = ctypes.c_double
    w, rho, zk = (np.ascontiguousarray(a) for a in (w, rho, zk))
    return f(ctypes.c_int64(len(w)), w.ctypes.data_as(P),
             rho.ctypes.data_as(P), zk.ctypes.data_as(P))


# --- exact collocation towers --------------------------------------------------

_R = sp.symbols("x y z", real=True)
_C = sp.symbols("cx cy cz", real=True)
#: families whose density matrix is general (complex orbitals: the
#: antisymmetric part is Im P and carries the paramagnetic current)
GENERAL_DM = ("cmgga_tau",)

#: (atom, exponent, prefactor monomial exponents)
BASIS = [(0, 0.9, (0, 0, 0)), (0, 1.3, (1, 0, 0)), (1, 0.7, (0, 1, 0)),
         (1, 1.1, (0, 0, 1)), (2, 0.8, (1, 1, 0)), (2, 1.2, (0, 0, 0)),
         (2, 1.0, (0, 1, 1))]
BF_ATOM = np.array([b[0] for b in BASIS])
COMPS = components(3)


def _colloc_funcs():
    d = [r - c for r, c in zip(_R, _C)]
    out = []
    for _, a, (px, py, pz) in BASIS:
        chi = d[0]**px * d[1]**py * d[2]**pz \
            * sp.exp(-a * (d[0]**2 + d[1]**2 + d[2]**2))
        exprs = []
        for ax in COMPS:
            e = chi
            for c in ax:
                e = sp.diff(e, _R[_AX.index(c)])
            exprs.append(e)
        out.append(sp.lambdify(_R + _C, exprs, "numpy"))
    return out


_FUNCS = _colloc_funcs()


def colloc(centers, pts):
    """The collocation tower (20, nbf, ng) to third order."""
    ng = len(pts)
    T = np.empty((len(COMPS), len(BASIS), ng))
    for u, (f, (atom, _, _)) in enumerate(zip(_FUNCS, BASIS)):
        res = f(pts[:, 0], pts[:, 1], pts[:, 2], *centers[atom])
        for k, r in enumerate(res):
            T[k, u] = np.broadcast_to(np.asarray(r, float), (ng,))
    return T


def _subsets(axes):
    n = len(axes)
    for mask in range(1 << n):
        a = "".join(axes[i] for i in range(n) if mask >> i & 1)
        b = "".join(axes[i] for i in range(n) if not mask >> i & 1)
        yield a, b


def fields(D, T, order=3):
    """The density tower (Leibniz rule), the tau tower to first order and
    the paramagnetic-current towers jp[i] to first order. D need not be
    symmetric: its antisymmetric part (Im P) carries the current, with the
    engine's convention jp_i = sum_uv D_uv (chi_u d_i chi_v - d_i chi_u
    chi_v) / 2."""
    t = lambda ax: T[comp_index(ax)]
    rho = np.stack([sum(np.einsum("uv,ug,vg->g", D, t(a), t(b))
                        for a, b in _subsets(ax))
                    for ax in components(order)])
    tau = [0.5 * sum(np.einsum("uv,ug,vg->g", D, t(i), t(i)) for i in _AX)]
    for k in _AX:
        tau.append(0.5 * sum(np.einsum("uv,ug,vg->g", D, t(k + i), t(i))
                             + np.einsum("uv,ug,vg->g", D, t(i), t(k + i))
                             for i in _AX))
    jp = []
    for i in _AX:
        comp = [0.5 * (np.einsum("uv,ug,vg->g", D, t(""), t(i))
                       - np.einsum("uv,ug,vg->g", D, t(i), t("")))]
        for d in _AX:
            comp.append(0.5 * (np.einsum("uv,ug,vg->g", D, t(d), t(i))
                               + np.einsum("uv,ug,vg->g", D, t(""), t(d + i))
                               - np.einsum("uv,ug,vg->g", D, t(d + i), t(""))
                               - np.einsum("uv,ug,vg->g", D, t(i), t(d))))
        jp.append(np.stack(comp))
    return {"rho": rho, "tau": np.stack(tau), "jp": np.stack(jp)}


def _tau_tilde(f):
    """The gauge-corrected tau~ = tau - jp^2 / (2 rho) (tau without
    current)."""
    jp2 = (f["jp"][:, 0] ** 2).sum(0)
    return f["tau"][0] - jp2 / (2 * f["rho"][0])


# --- explicit test functionals ---------------------------------------------------

_FAM_VARS = {"lda": ("rho",), "gga": ("rho", "sigma"),
             "mgga_tau": ("rho", "sigma", "tau"),
             "mgga_lapl": ("rho", "sigma", "lapl"),
             "mgga": ("rho", "sigma", "lapl", "tau"),
             "cmgga_tau": ("rho", "sigma", "tau"),
             "hmgga": ("rho", "sigma", "lapl", "tau", "eta")}

_RV = {v: sp.Symbol(v, real=True)
       for v in ("rho", "sigma", "tau", "lapl", "eta")}
_FR = (_RV["rho"]**3 / 3 + sp.Rational(3, 10) * _RV["sigma"] * _RV["rho"]
       + sp.Rational(1, 5) * _RV["tau"]**2
       + sp.Rational(1, 10) * _RV["rho"]**2 * _RV["tau"]
       + sp.Rational(1, 20) * _RV["sigma"] * _RV["tau"]
       + sp.Rational(3, 20) * _RV["lapl"] * _RV["rho"]**2
       + sp.Rational(1, 15) * _RV["lapl"]**2
       + sp.Rational(1, 25) * _RV["lapl"] * _RV["sigma"]
       + sp.Rational(1, 30) * _RV["sigma"]**2
       + sp.Rational(1, 12) * _RV["eta"] * _RV["rho"]
       + sp.Rational(1, 40) * _RV["eta"]**2
       + sp.Rational(1, 35) * _RV["eta"] * _RV["tau"])

_UV = {(g, c): sp.Symbol(f"{g}_{c}", real=True)
       for g, cs in (("rho", "ab"), ("sigma", ("aa", "ab", "bb")),
                     ("lapl", "ab"), ("tau", "ab"), ("eta", "ab"))
       for c in cs}
_U = lambda g, c: _UV[(g, c)]
_FU = (_U("rho", "a")**3 / 3 + sp.Rational(2, 5) * _U("rho", "b")**3
       + sp.Rational(3, 10) * _U("rho", "a") * _U("rho", "b")**2
       + sp.Rational(1, 5) * _U("sigma", "aa") * _U("rho", "b")
       + sp.Rational(1, 4) * _U("sigma", "ab") * (_U("rho", "a") + _U("rho", "b"))
       + sp.Rational(1, 10) * _U("sigma", "bb") * _U("rho", "a")**2
       + sp.Rational(1, 30) * _U("sigma", "aa") * _U("sigma", "ab")
       + sp.Rational(1, 5) * _U("tau", "a") * _U("rho", "b")**2
       + sp.Rational(3, 20) * _U("tau", "b")**2
       + sp.Rational(1, 8) * _U("tau", "a") * _U("tau", "b")
       + sp.Rational(3, 25) * _U("lapl", "a") * _U("rho", "b")
       + sp.Rational(1, 11) * _U("lapl", "b") * _U("rho", "a")**2
       + sp.Rational(1, 20) * _U("lapl", "a") * _U("lapl", "b")
       + sp.Rational(1, 30) * _U("sigma", "aa") * _U("tau", "b")
       + sp.Rational(1, 12) * _U("eta", "a") * _U("rho", "b")
       + sp.Rational(1, 16) * _U("eta", "b") * _U("rho", "a")
       + sp.Rational(1, 40) * _U("eta", "a") * _U("eta", "b"))


def _eta(rho):
    """grad rho . (grad grad rho) . grad rho from a density tower."""
    g = rho[1:4]
    H = np.array([[rho[comp_index(a + b)] for b in _AX] for a in _AX])
    return np.einsum("ig,ijg,jg->g", g, H, g)


def _variables_r(f):
    rho = f["rho"]
    grad = rho[1:4]
    return {_RV["rho"]: rho[0], _RV["sigma"]: (grad * grad).sum(0),
            _RV["tau"]: _tau_tilde(f),
            _RV["lapl"]: sum(rho[comp_index(a + a)] for a in _AX),
            _RV["eta"]: _eta(rho)}


def _variables_u(fa, fb):
    out = {}
    for s, f in (("a", fa), ("b", fb)):
        out[_U("rho", s)] = f["rho"][0]
        out[_U("tau", s)] = _tau_tilde(f)
        out[_U("lapl", s)] = sum(f["rho"][comp_index(a + a)] for a in _AX)
        out[_U("eta", s)] = _eta(f["rho"])
    for c, (f1, f2) in (("aa", (fa, fa)), ("ab", (fa, fb)), ("bb", (fb, fb))):
        out[_U("sigma", c)] = (f1["rho"][1:4] * f2["rho"][1:4]).sum(0)
    return out


def _functional(fam, spin):
    if spin == "r":
        return _FR.subs({_RV[v]: 0 for v in _RV if v not in _FAM_VARS[fam]})
    return _FU.subs({s: 0 for (g, _), s in _UV.items()
                     if g not in _FAM_VARS[fam]})


def energy_density(fam, spin, fa, fb=None):
    F = _functional(fam, spin)
    var = _variables_r(fa) if spin == "r" else _variables_u(fa, fb)
    syms = list(var)
    return sp.lambdify(syms, F, "numpy")(*var.values()) * np.ones(
        fa["rho"].shape[1])


def libxc_arrays(names, fam, spin, fa, fb=None):
    """Derivative arrays by Libxc name (any order), from the registry the
    kernels are generated with."""
    from ..engine.spin_kernel import _SYM_SCALARS
    F = _functional(fam, spin)
    var = _variables_r(fa) if spin == "r" else _variables_u(fa, fb)
    syms = list(var)
    out = {}
    for n in names:
        e = F
        if spin == "r":
            for v, cnt in LIBXC_MULTISET[n].items():
                e = sp.diff(e, _RV[v], cnt)
        else:
            for K in _SYM_SCALARS[n]:
                e = sp.diff(e, _U(K.group, K.comp))
        out[n] = sp.lambdify(syms, e, "numpy")(*var.values()) * np.ones(
            fa["rho"].shape[1])
    return out


def tower_ops(fa, fb=None, sfx=""):
    """Tower-named per-point operands of the ground (or perturbed, sfx
    '_p1') fields: rho, tau, the current jpx/jpy/jpz and, for the ground
    state, inv_rho."""
    out = {}
    chans = [("", fa)] if fb is None else [("_a", fa), ("_b", fb)]
    for c, f in chans:
        out[f"rho{c}{sfx}"] = f["rho"]
        out[f"tau{c}{sfx}"] = f["tau"]
        for i, ax in enumerate(_AX):
            out[f"jp{ax}{c}{sfx}"] = f["jp"][i]
        if not sfx:
            out[f"inv_rho{c}"] = 1.0 / f["rho"][0]
    return out


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

        def dm(scale=0.2, shift=0.6):
            M = rng.standard_normal((nbf, nbf))
            return scale * M @ M.T / nbf + shift * np.eye(nbf)
        self.Da, self.Db = dm(), dm()
        self.D1a = 0.3 * (lambda M: M + M.T)(rng.standard_normal((nbf, nbf)))
        self.D1b = 0.3 * (lambda M: M + M.T)(rng.standard_normal((nbf, nbf)))
        anti = lambda: 0.2 * (lambda M: M - M.T)(rng.standard_normal((nbf, nbf)))
        self.Aa, self.Ab, self.A1a, self.A1b = anti(), anti(), anti(), anti()

    def dms(self, fam):
        """(Ma, Mb, X1a, X1b): ground and perturbation density matrices,
        general (with an antisymmetric, current-carrying part) for the
        current-density family."""
        if fam in GENERAL_DM:
            return (self.Da + self.Aa, self.Db + self.Ab,
                    self.D1a + self.A1a, self.D1b + self.A1b)
        return self.Da, self.Db, self.D1a, self.D1b

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

    def energy(self, fam, spin, centers_a, centers_b, pts, w, Da=None,
               Db=None):
        """Exc with the alpha and beta basis functions at independent
        centers (so each channel's basis class can be isolated)."""
        Da = self.Da if Da is None else Da
        Db = self.Db if Db is None else Db
        Ta = colloc(centers_a, pts)
        if spin == "r":
            e = energy_density(fam, "r", fields(Da + Db, Ta, 2))
        else:
            Tb = Ta if centers_b is centers_a else colloc(centers_b, pts)
            e = energy_density(fam, "u", fields(Da, Ta, 2),
                               fields(Db, Tb, 2))
        return float(np.dot(w, e))


def richardson(f, h=5e-4):
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


def check_diag(lib, rep, nbf=5, ng=37, seed=2):
    rng = np.random.default_rng(seed)
    T = rng.standard_normal((len(components(2)), nbf, ng))
    for fam in FAMILIES:
        for spin in ("r", "ua", "ub"):
            full = CatalogEntry(fam, spin, 1).name
            diag = CatalogEntry(fam, spin, 1, kind="diag").name
            if lib.scal_names(diag) != lib.scal_names(full):
                rep.check(f"{diag} operand order", 1.0, 0.0, 0.5)
                continue
            ops = {n: rng.standard_normal(ng) for n in lib.scal_names(full)}
            F = lib(full, chi=T, **ops)
            d = lib(diag, chi=T, **ops)
            rep.check(f"{diag} == diag({full})", d, np.diag(F), 1e-13)


def _matrix_ops(lib, name, fam, spin, sysm, w, T, Da, Db, D1a=None, D1b=None):
    """Every tower operand of a matrix kernel at (Da, Db) [and D1]."""
    if spin == "r":
        f = fields(Da + Db, T)
        ops = tower_ops(f)
        if D1a is not None:
            ops.update(tower_ops(fields(D1a + D1b, T), sfx="_p1"))
        ops.update(libxc_arrays([n for n in lib.scal_names(name)
                                 if n.startswith("v")], fam, "r", f))
    else:
        fa, fb = fields(Da, T), fields(Db, T)
        ops = tower_ops(fa, fb)
        if D1a is not None:
            ops.update(tower_ops(fields(D1a, T), fields(D1b, T), sfx="_p1"))
        ops.update(libxc_arrays([n for n in lib.scal_names(name)
                                 if n.startswith("v")], fam, "u", fa, fb))
    return ops


def check_matrix(lib, rep, fam, sysm):
    c0, p0 = sysm.centers, sysm.pts
    w = sysm.weights(c0, p0)
    T = colloc(c0, p0)
    nbf = T.shape[1]
    Ma, Mb, X1a, X1b = sysm.dms(fam)
    general = fam in GENERAL_DM
    for spin in ("r", "ua"):
        o1 = CatalogEntry(fam, spin, 1).name
        o2 = CatalogEntry(fam, spin, 2).name
        espin = "r" if spin == "r" else "u"

        def F_at(Da, Db):
            return lib(o1, w=w, chi=T,
                       **_matrix_ops(lib, o1, fam, spin, sysm, w, T, Da, Db))

        # o1 == dExc/dD (alpha channel for ua): every entry independently
        # for a general D, symmetric pairs otherwise
        F = F_at(Ma, Mb)
        fd = np.zeros((nbf, nbf))
        for u in range(nbf):
            for v in range(nbf if general else u + 1):
                dD = np.zeros((nbf, nbf))
                if general:
                    dD[u, v] = 1.0
                else:
                    dD[u, v] += 0.5
                    dD[v, u] += 0.5
                fd[u, v] = richardson(
                    lambda h: sysm.energy(fam, espin, c0, c0, p0, w,
                                          Ma + h * dD, Mb))
                if not general:
                    fd[v, u] = fd[u, v]
        rep.check(f"{fam:9s} {spin:2s} o1 == dExc/dD", F, fd, 1e-9)

        # o2 == directional derivative of o1 along D1 (both channels)
        F2 = lib(o2, w=w, chi=T,
                 **_matrix_ops(lib, o2, fam, spin, sysm, w, T, Ma, Mb,
                               X1a, X1b))
        fd2 = richardson(lambda h: F_at(Ma + h * X1a, Mb + h * X1b))
        rep.check(f"{fam:9s} {spin:2s} o2 == d(o1)/dD . D1", F2, fd2, 1e-9)


def check_batch(lib, rep, nbf=5, ng=37, seed=4):
    """o2_batch == nx single o2 calls, every family and spin case."""
    rng = np.random.default_rng(seed)
    for fam in FAMILIES:
        cases = [CatalogEntry(fam, s, 2) for s in ("r", "ua", "ub")] + \
            [CatalogEntry(fam, "st", 2, (p,)) for p in (+1, -1)]
        for single in cases:
            batch = CatalogEntry(single.family, single.spin, 2,
                                 single.parities, kind="o2b").name
            names = lib.scal_names(single.name)
            if lib.scal_names(batch) != names:
                rep.check(f"{batch} operand order", 1.0, 0.0, 0.5)
                continue
            T = rng.standard_normal((len(components(lib.order(single.name))),
                                     nbf, ng))
            for nx in (1, 3, 11):
                ground = {n: rng.standard_normal(ng) for n in names
                          if "_p1" not in n}
                ground["w"] = np.abs(ground["w"]) + 0.1
                pert = {n: rng.standard_normal((nx, ng)) for n in names
                        if "_p1" in n}
                got = lib(batch, chi=T, **ground, **pert)
                ref = np.stack([lib(single.name, chi=T, **ground,
                                    **{n: v[x] for n, v in pert.items()})
                                for x in range(nx)])
                rep.check(f"{batch} nx={nx} == {nx} x {single.name}", got,
                          ref, 1e-13)


def check_fock_derivative(lib, rep, fam, spin, sysm):
    """dF/dX: basis class (f1, atom mask), grid class (fg, w := w M^A)
    and weight class (o1, w := dw/dX), each against FD of the o1 kernel
    with that R-dependence isolated; their sum against moving the atom
    completely; the translational sum rule."""
    c0, p0 = sysm.centers, sysm.pts
    w0 = sysm.weights(c0, p0)
    Ma, Mb, _, _ = sysm.dms(fam)
    o1 = CatalogEntry(fam, spin, 1).name
    f1, fg = f"xck_{fam}_{spin}_f1", f"xck_{fam}_{spin}_fg"
    T = colloc(c0, p0)

    def F_at(centers, pts, w):
        Tc = colloc(centers, pts)
        return lib(o1, w=w, chi=Tc, **_matrix_ops(lib, o1, fam, spin, sysm,
                                                  w, Tc, Ma, Mb))

    towers = ({"Dchi": np.einsum("uv,kvg->kug", Ma + Mb, T)} if spin == "r"
              else {"Dchi_a": np.einsum("uv,kvg->kug", Ma, T),
                    "Dchi_b": np.einsum("uv,kvg->kug", Mb, T)})
    if fam in GENERAL_DM:
        # the M^T-contracted towers of a general density matrix
        towers.update({"DTchi": np.einsum("vu,kvg->kug", Ma + Mb, T)}
                      if spin == "r" else
                      {"DTchi_a": np.einsum("vu,kvg->kug", Ma, T),
                       "DTchi_b": np.einsum("vu,kvg->kug", Mb, T)})
    ops_b = _matrix_ops(lib, f1, fam, spin, sysm, w0, T, Ma, Mb)
    ops_g = _matrix_ops(lib, fg, fam, spin, sysm, w0, T, Ma, Mb)
    ops_w = _matrix_ops(lib, o1, fam, spin, sysm, w0, T, Ma, Mb)
    natom = sysm.natom
    total = []
    for A in range(natom):
        mask = (BF_ATOM == A).astype(np.int8)
        basis = lib(f1, w=w0, chi=T, atom_mask=mask, **towers, **ops_b)
        grid = lib(fg, w=w0 * (sysm.parent == A), chi=T, **ops_g)
        weight = np.stack([lib(o1, w=sysm.dweights(A, d), chi=T, **ops_w)
                           for d in range(3)])
        fdb, fdg, fdt = [], [], []
        for d in range(3):
            def Eb(h):
                c = c0.copy()
                c[A, d] += h
                return F_at(c, p0, w0)

            def Eg(h):
                p = p0.copy()
                p[sysm.parent == A, d] += h
                return F_at(c0, p, w0)

            def Et(h):
                c, p = c0.copy(), p0.copy()
                c[A, d] += h
                p[sysm.parent == A, d] += h
                return F_at(c, p, sysm.weights(c, p))
            fdb.append(richardson(Eb))
            fdg.append(richardson(Eg))
            fdt.append(richardson(Et))
        rep.check(f"{fam:9s} {spin:2s} A={A} f1 basis class vs FD", basis,
                  np.stack(fdb), 1e-9)
        rep.check(f"{fam:9s} {spin:2s} A={A} fg grid class vs FD", grid,
                  np.stack(fdg), 1e-9)
        total.append(basis + grid + weight)
        rep.check(f"{fam:9s} {spin:2s} A={A} f1+fg+weight vs FD", total[-1],
                  np.stack(fdt), 1e-9)
    total = np.stack(total)
    rep.check(f"{fam:9s} {spin:2s} dF/dX translational sum rule",
              total.sum(0), np.zeros_like(total[0]), 1e-12,
              scale=np.abs(total).max())


def check_giao(lib, rep, fam, spin, sysm, seed=7):
    """The compiled GIAO kernel against the NumPy one on the same data."""
    from ..emitters.codegen import compile_function, generate_collapsed
    from ..engine.london import london_fock, london_fock_spin
    rng = np.random.default_rng(seed)
    T = colloc(sysm.centers, sysm.pts)
    nbf, ng = T.shape[1:]
    R = sysm.centers[BF_ATOM]                         # (nbf, 3)
    name = CatalogEntry(fam, spin, 1, giao=True).name
    ops = {n: rng.standard_normal(ng) for n in lib.scal_names(name)}
    ops["w"] = np.abs(ops["w"]) + 0.1
    K = lib(name, chi=T, bf_centers=np.ascontiguousarray(R.T), **ops)

    # the NumPy reference, with its own explicit operands
    chi, dchi = T[0], T[1:4]
    lapl = sum(T[comp_index(a + a)] for a in _AX)
    args = {"w": ops["w"], "chi": chi, "dchi": dchi, "lapl_chi": lapl,
            "Rchi": np.einsum("ua,ug->aug", R, chi),
            "Rdchi": np.einsum("ua,cug->acug", R, dchi),
            "Rlapl_chi": np.einsum("ua,ug->aug", R, lapl),
            "rg": np.stack([ops[f"rg{a}"] for a in _AX]),
            **{n: v for n, v in ops.items() if n.startswith("v")}}
    if spin == "r":
        if "rho_x" in ops:
            args["grad_rho"] = np.stack([ops[f"rho_{a}"] for a in _AX])
    else:
        for c in "ab":
            if f"rho_{c}_x" in ops:
                args[f"grad_rho_{c}"] = np.stack([ops[f"rho_{c}_{a}"]
                                                  for a in _AX])
    ref = []
    for s in range(3):
        ki = (london_fock(fam, s) if spin == "r"
              else london_fock_spin(fam, spin[1], s))
        gen = generate_collapsed(ki, "kref", batch=False)
        sig = gen.source.split("(", 1)[1].split(")", 1)[0]
        ref.append(compile_function(gen)(*[args[p.strip()]
                                           for p in sig.split(",")]))
    rep.check(f"{fam:9s} {spin:2s} giao == NumPy GIAO kernel", K,
              np.stack(ref), 1e-12)
    rep.check(f"{fam:9s} {spin:2s} giao K^s antisymmetric", K,
              -np.transpose(K, (0, 2, 1)), 1e-12,
              scale=np.abs(K).max())


def check_gradient(lib, rep, fam, spin, sysm):
    """spin 'r' or 'u'."""
    c0, p0 = sysm.centers, sysm.pts
    w = sysm.weights(c0, p0)
    T = colloc(c0, p0)
    nbf, ng = T.shape[1:]
    Ma, Mb, _, _ = sysm.dms(fam)
    general = fam in GENERAL_DM
    dm = {"Da": Ma, "Db": Mb}
    if spin == "r":
        f = fields(Ma + Mb, T)
        e = energy_density(fam, "r", f)
        ops = tower_ops(f)
        vnames = {n for k in ("xck_%s_r_g1", "xck_%s_r_gg")
                  for n in lib.scal_names(k % fam) if n.startswith("v")}
        ops.update(libxc_arrays(vnames, fam, "r", f))
        chans = [("r", Ma + Mb)]
        rho_tot = f["rho"][0]
    else:
        fa, fb = fields(Ma, T), fields(Mb, T)
        e = energy_density(fam, "u", fa, fb)
        ops = tower_ops(fa, fb)
        vnames = {n for k in ("xck_%s_ua_g1", "xck_%s_ub_g1", "xck_%s_u_gg")
                  for n in lib.scal_names(k % fam) if n.startswith("v")}
        ops.update(libxc_arrays(vnames, fam, "u", fa, fb))
        chans = [("ua", Ma), ("ub", Mb)]
        rho_tot = fa["rho"][0] + fb["rho"][0]

    # ----- basis class: per channel -----
    rows = {}
    for ch, D in chans:
        extra = ({"DTchi": np.einsum("vu,kvg->kug", D, T)} if general
                 else {})
        rows[ch] = lib(f"xck_{fam}_{ch}_g1", w=w, chi=T,
                       Dchi=np.einsum("uv,kvg->kug", D, T), **extra, **ops)
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
                    return sysm.energy(fam, spin, ca, cb, p0, w, **dm)
                fd.append(richardson(E))
                got.append(rows[ch][d, BF_ATOM == A].sum())
        rep.check(f"{fam:9s} {ch:2s} g1 basis class vs FD", got, fd, 1e-9)

    # ----- grid class -----
    gg = lib(f"xck_{fam}_{spin}_gg", w=w, **ops)
    grid = np.array([[gg[d, sysm.parent == A].sum() for d in range(3)]
                     for A in range(sysm.natom)])
    fd = np.zeros_like(grid)
    for A in range(sysm.natom):
        for d in range(3):
            def E(h, A=A, d=d):
                p = p0.copy()
                p[sysm.parent == A, d] += h
                return sysm.energy(fam, spin, c0, c0, p, w, **dm)
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
                return sysm.energy(fam, spin, c, c, p, sysm.weights(c, p),
                                   **dm)
            fd[A, d] = richardson(E)
    rep.check(f"{fam:9s} {spin:2s} basis+grid+weight vs FD", total, fd, 1e-9)

    # ----- translational sum rule -----
    scale = max(np.abs(basis).max(), np.abs(grid).max(), np.abs(weight).max())
    rep.check(f"{fam:9s} {spin:2s} translational sum rule",
              total.sum(0), np.zeros(3), 1e-12, scale=scale)


def main():
    rep = Report()
    with tempfile.TemporaryDirectory() as td:
        print("building the kernels ...", flush=True)
        lib = build_library(Path(td))
        print("Fock diagonal")
        check_diag(lib, rep)
        print("batched linear response")
        check_batch(lib, rep)
        sysm = System()
        for fam in GRADIENT_FAMILIES + GENERAL_DM:
            print(f"Fock matrix and linear response: {fam}")
            check_matrix(lib, rep, fam, sysm)
        from ..catalog import GIAO_FAMILIES
        for fam in GIAO_FAMILIES:
            print(f"London orbitals: {fam}")
            for spin in ("r", "ua", "ub"):
                check_giao(lib, rep, fam, spin, sysm)
        from ..engine.geofock import FOCK_DERIV_FAMILIES
        for fam in FOCK_DERIV_FAMILIES:
            print(f"nuclear derivative of the Fock matrix: {fam}")
            for spin in ("r", "ua"):
                check_fock_derivative(lib, rep, fam, spin, sysm)
        for fam in gradient_families():
            print(f"nuclear gradient: {fam}")
            for spin in ("r", "u"):
                check_gradient(lib, rep, fam, spin, sysm)
    status = "OK " if rep.failures == 0 else "FAIL"
    print(f"[{status}] catalog_c_validate: {rep.tested} checks, "
          f"{rep.failures} failures")
    return rep.failures


if __name__ == "__main__":
    raise SystemExit(main())
