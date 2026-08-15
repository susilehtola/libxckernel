"""Validate the machine-generated noncollinear (relativistic) kernels.

The locally collinear map U(V) of :mod:`..engine.noncollinear` composes
collinear functionals with noncollinear fields V = (rho_s, m, gradients);
the emitted nc_vxc/nc_fxc functions are its first and second mechanical
derivatives with the Libxc arrays opaque.  An explicit analytic polarized
functional stands in for Libxc, so all derivative arrays are exact.

Checks:

  1. collinear reduction: for m || z the energy, the potential rows, and
     the transverse rows reduce exactly to the polarized collinear theory
  2. the LDA transverse (spin-flip) kernel identity
     C_xx = (v+ - v-)/(2 m_z) at a collinear point
  3. global spin-rotation invariance of the energy and the fxc bilinear
  4. nc_vxc against finite differences of the energy in the fields, and
     nc_fxc against finite differences of nc_vxc (LDA and GGA)
  5. four-component end to end: fields assembled from FIRST PRINCIPLES
     over an explicit restricted-kinetic-balance spinor basis (Pauli
     matrices and sigma.p small components contracted by einsum -- an
     independent numerical derivation of the density expressions of
     Bersson et al.) with a general Hermitian block density matrix;
     dE/dlambda and d2E/dlambda2 along a random Hermitian perturbation
     of the large- and small-component blocks against the generated
     potential contraction and fxc bilinear.  The speed of light is
     reduced to make the small-component contributions substantial.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from ..emitters.ncwriter import emit
from ..engine.noncollinear import (AXES, F_NABLA, LIBXC_FIRST, LIBXC_SECOND,
                                   libxc_args, nc_fields, nc_map)

# --- the explicit polarized test functional over U ---------------------------

_np_, _nm_, _gpp, _gpm, _gmm = sp.symbols("n_p n_m g_pp g_pm g_mm",
                                          positive=True)
_USYM = {"n_p": _np_, "n_m": _nm_, "g_pp": _gpp, "g_pm": _gpm, "g_mm": _gmm}
_F = (_np_**2 + _nm_**2 + sp.Rational(1, 2) * _np_ * _nm_
      + sp.Rational(3, 10) * (_gpp * _np_ + _gmm * _nm_)
      + sp.Rational(1, 10) * _gpm * (_np_ + _nm_)
      + sp.Rational(1, 20) * _gpp * _gmm)
_ARGS = (_np_, _nm_, _gpp, _gpm, _gmm)
_FNUM = sp.lambdify(_ARGS, _F, "numpy")

#: derivative array name -> differentiation multiset over U
_DMAP = {name: (a,) for a, name in LIBXC_FIRST.items()}
_DMAP.update({name: pair for pair, name in LIBXC_SECOND.items()})


def _deriv_arrays(names, uvals):
    out = {}
    for name in names:
        e = _F
        for a in _DMAP[name]:
            e = sp.diff(e, _USYM[a])
        out[name] = sp.lambdify(_ARGS, e, "numpy")(*uvals) \
            * np.ones_like(uvals[0])
    return out


def _call(fn, params, args):
    return fn(*[args[p] for p in params])


def main():
    rng = np.random.default_rng(41)
    ng = 20

    # emitted module, regenerated in-process
    ns = {}
    exec(compile(emit(("lda", "gga")), "<nc_kernels>", "exec"), ns)

    U_gga = {a: sp.lambdify([sp.Symbol(n) for n in _field_names("gga")]
                            + [F_NABLA], e, "numpy")
             for a, e in nc_map("gga").items()}
    U_lda = {a: sp.lambdify([sp.Symbol(n) for n in _field_names("lda")]
                            + [F_NABLA], e, "numpy")
             for a, e in nc_map("lda").items()}

    tested = failures = 0

    def check(label, err, tol):
        nonlocal tested, failures
        tested += 1
        ok = err < tol
        if not ok:
            failures += 1
        print(f"  [{'OK' if ok else 'FAIL'}] {label}: err {err:.2e}")

    def uvals(family, V, fnab):
        Umap = U_gga if family == "gga" else U_lda
        vals = [Umap[a](*V, fnab) * np.ones(np.shape(V[0]))
                for a in ("n_p", "n_m")]
        if family == "gga":
            vals += [Umap[a](*V, fnab) for a in ("g_pp", "g_pm", "g_mm")]
        else:
            vals += [np.zeros_like(vals[0])] * 3
        return vals

    def energy(family, V, fnab, w):
        u = uvals(family, V, fnab)
        return float(np.sum(w * _FNUM(*u)))

    def fnabla_of(V):
        # sgn(grad rho_s . sum_J rho_J grad rho_J), from the gga field order
        gs = np.array(V[4:7])
        tot = np.zeros_like(V[0])
        for J in range(3):
            gJ = np.array(V[7 + 3 * J: 10 + 3 * J])
            tot += V[1 + J] * np.einsum("cg,cg->g", gs, gJ)
        s = np.sign(tot)
        s[s == 0] = 1.0
        return s

    def vxc(family, V, fnab):
        u = uvals(family, V, fnab)
        lx = _deriv_arrays(libxc_args(family, 1), u)
        args = {n: a for n, a in zip(_field_names(family), V)}
        args.update(lx)
        args["f_nabla"] = fnab
        fn = ns[f"nc_vxc_{family}"]
        return _call(fn, _params(family, 1), args)

    def fxc(family, V, fnab):
        u = uvals(family, V, fnab)
        lx = _deriv_arrays(libxc_args(family, 2), u)
        args = {n: a for n, a in zip(_field_names(family), V)}
        args.update(lx)
        args["f_nabla"] = fnab
        fn = ns[f"nc_fxc_{family}"]
        return _call(fn, _params(family, 2), args)

    w = rng.uniform(0.1, 1.0, ng)

    # generic noncollinear gga fields: |m| in [0.3, 0.4], direction varying
    rho_s = rng.uniform(1.5, 2.5, ng)
    nhat = rng.standard_normal((3, ng))
    nhat /= np.linalg.norm(nhat, axis=0)
    mmag = rng.uniform(0.3, 0.4, ng)
    V = [rho_s] + [nhat[J] * mmag for J in range(3)] \
        + [0.5 * rng.standard_normal(ng) for _ in range(12)]
    fnab = fnabla_of(V)
    assert uvals("gga", V, fnab)[1].min() > 0.02

    # 1. collinear reduction ---------------------------------------------------
    Vc = list(V)
    Vc[1] = np.zeros(ng)
    Vc[2] = np.zeros(ng)
    Vc[3] = rng.uniform(0.25, 0.4, ng)               # m_z > 0
    for k in range(7, 13):                           # grad m_x, grad m_y = 0
        Vc[k] = np.zeros(ng)
    fnc = fnabla_of(Vc)
    n_p = 0.5 * (Vc[0] + Vc[3])
    n_m = 0.5 * (Vc[0] - Vc[3])
    gp = 0.5 * (np.array(Vc[4:7]) + np.array(Vc[13:16]))
    gm = 0.5 * (np.array(Vc[4:7]) - np.array(Vc[13:16]))
    upol = [n_p, n_m, np.einsum("cg,cg->g", gp, gp),
            np.einsum("cg,cg->g", gp, gm), np.einsum("cg,cg->g", gm, gm)]
    Epol = float(np.sum(w * _FNUM(*upol)))
    Enc = energy("gga", Vc, fnc, w)
    check("collinear energy reduction", abs(Enc - Epol) / abs(Epol), 1e-14)

    pot = vxc("gga", Vc, fnc)
    lxc = _deriv_arrays(["vrho_0", "vrho_1"], upol)
    err = max(
        float(np.abs(pot[0] - 0.5 * (lxc["vrho_0"] + lxc["vrho_1"])).max()),
        float(np.abs(pot[3] - 0.5 * (lxc["vrho_0"] - lxc["vrho_1"])).max()))
    check("collinear potential rows (rho_s, rho_z)", err, 1e-13)
    err = max(float(np.abs(pot[i]).max()) for i in (1, 2, 7, 8, 9, 10, 11, 12))
    check("transverse potential rows vanish", err, 1e-13)

    # 2. LDA spin-flip kernel identity ----------------------------------------
    Vl = [Vc[0], Vc[1], Vc[2], Vc[3]]
    ul = uvals("lda", Vl, fnc)
    Cl = fxc("lda", Vl, fnc)
    lx1 = _deriv_arrays(["vrho_0", "vrho_1"], ul)
    ref = (lx1["vrho_0"] - lx1["vrho_1"]) / (2 * Vl[3])
    err = max(float(np.abs(Cl[1, 1] - ref).max()),
              float(np.abs(Cl[2, 2] - ref).max()),
              float(np.abs(Cl[1, 2]).max()))
    check("LDA spin-flip kernel (v+ - v-)/(2 m_z)", err / abs(ref).max(),
          1e-13)

    # 3. spin-rotation invariance ---------------------------------------------
    A = rng.standard_normal((3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    VR = list(V)
    m = np.array(V[1:4])
    gmJ = np.array(V[7:16]).reshape(3, 3, ng)        # (J, c, g)
    mR = np.einsum("JK,Kg->Jg", Q, m)
    gR = np.einsum("JK,Kcg->Jcg", Q, gmJ)
    for J in range(3):
        VR[1 + J] = mR[J]
        for c in range(3):
            VR[7 + 3 * J + c] = gR[J, c]
    fnR = fnabla_of(VR)
    check("rotation invariance of f_nabla",
          float(np.abs(fnR - fnab).max()), 1e-13)
    ER = energy("gga", VR, fnR, w)
    E0 = energy("gga", V, fnab, w)
    check("rotation invariance of the energy", abs(ER - E0) / abs(E0), 1e-13)
    d = [rng.standard_normal(ng) for _ in range(16)]
    dR = list(d)
    dm = np.einsum("JK,Kg->Jg", Q, np.array(d[1:4]))
    dg = np.einsum("JK,Kcg->Jcg", Q, np.array(d[7:16]).reshape(3, 3, ng))
    for J in range(3):
        dR[1 + J] = dm[J]
        for c in range(3):
            dR[7 + 3 * J + c] = dg[J, c]
    C0 = fxc("gga", V, fnab)
    CR = fxc("gga", VR, fnR)
    b0 = float(np.sum(w * np.einsum("ig,ijg,jg->g", np.array(d), C0,
                                    np.array(d))))
    bR = float(np.sum(w * np.einsum("ig,ijg,jg->g", np.array(dR), CR,
                                    np.array(dR))))
    check("rotation invariance of the fxc bilinear", abs(bR - b0) / abs(b0),
          1e-12)

    # 4. FD validation of the generated derivatives ---------------------------
    for family, Vf, nV in (("lda", V[:4], 4), ("gga", V, 16)):
        fn0 = fnab
        eps = [rng.standard_normal(ng) for _ in range(nV)]

        def E_at(lam):
            return energy(family, [v + lam * e for v, e in zip(Vf, eps)],
                          fn0, w)

        h = 1e-5
        fd1 = (8 * (E_at(h) - E_at(-h))
               - (E_at(2 * h) - E_at(-2 * h))) / (12 * h)
        pot0 = vxc(family, Vf, fn0)
        an1 = float(np.sum(w * sum(pot0[i] * eps[i] for i in range(nV))))
        check(f"{family} nc_vxc vs FD of the energy",
              abs(fd1 - an1) / abs(an1), 1e-9)

        def g_at(lam):
            p = vxc(family, [v + lam * e for v, e in zip(Vf, eps)], fn0)
            return float(np.sum(w * sum(p[i] * eps[i] for i in range(nV))))

        fd2 = (8 * (g_at(h) - g_at(-h))
               - (g_at(2 * h) - g_at(-2 * h))) / (12 * h)
        C = fxc(family, Vf, fn0)
        an2 = float(np.sum(w * np.einsum("ig,ijg,jg->g", np.array(eps), C,
                                         np.array(eps))))
        check(f"{family} nc_fxc vs FD of nc_vxc", abs(fd2 - an2) / abs(an2),
              1e-8)

    # 5. four-component end to end --------------------------------------------
    cvel = 5.0                     # reduced speed of light: substantial SS
    nbf = 4
    chi = rng.uniform(0.5, 1.5, (nbf, ng))
    dchi = rng.standard_normal((3, nbf, ng))
    dd = rng.standard_normal((3, 3, nbf, ng))
    ddchi = 0.5 * (dd + dd.transpose(1, 0, 2, 3))

    sig = {"s": np.eye(2, dtype=complex),
           "x": np.array([[0, 1], [1, 0]], dtype=complex),
           "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
           "z": np.array([[1, 0], [0, -1]], dtype=complex)}
    #: spin weights: LL block (sigma_J)_{ab}; SS block via sigma.p on both
    #: sides, (sigma_k sigma_J sigma_l)_{ab} with the 1/4c^2 prefactor
    sws = {J: np.einsum("kab,bc,lcd->klad", np.array([sig[c] for c in AXES]),
                        sig[J], np.array([sig[c] for c in AXES]))
           for J in ("s",) + AXES}

    def herm(scale):
        M = (rng.standard_normal((2, nbf, 2, nbf))
             + 1j * rng.standard_normal((2, nbf, 2, nbf)))
        M = M.reshape(2 * nbf, 2 * nbf)
        return (scale * 0.5 * (M + M.conj().T)).reshape(2, nbf, 2, nbf)

    def block(comps):
        """A Hermitian 2-spinor block from magnitude-controlled Pauli
        components (still a completely general Hermitian block)."""
        out = np.zeros((2, nbf, 2, nbf), dtype=complex)
        for J, M in comps.items():
            out += np.einsum("ab,uv->aubv", sig[J], M)
        return out

    def hcomp(scale):
        M = (rng.standard_normal((nbf, nbf))
             + 1j * rng.standard_normal((nbf, nbf)))
        return scale * 0.5 * (M + M.conj().T)

    PLL = block({"s": np.eye(nbf) + hcomp(0.15), "x": hcomp(0.1),
                 "y": hcomp(0.1), "z": 0.25 * np.eye(nbf) + hcomp(0.1)})
    PSS = block({J: hcomp(0.4) for J in ("s", "x", "y", "z")})

    pref = 1.0 / (4 * cvel**2)

    def fields4(PL, PS):
        V4 = []
        for J in ["s"] + list(AXES):
            e = np.einsum("aubv,ab,ug,vg->g", PL, sig[J], chi, chi)
            e = e + pref * np.einsum("aubv,klab,kug,lvg->g", PS, sws[J],
                                     dchi, dchi)
            assert np.abs(e.imag).max() < 1e-12
            V4.append(e.real)
        for J in ["s"] + list(AXES):
            for c in range(3):
                e = np.einsum("aubv,ab,ug,vg->g", PL, sig[J], dchi[c], chi) \
                    + np.einsum("aubv,ab,ug,vg->g", PL, sig[J], chi, dchi[c])
                e = e + pref * (
                    np.einsum("aubv,klab,kug,lvg->g", PS, sws[J],
                              ddchi[:, c], dchi)
                    + np.einsum("aubv,klab,kug,lvg->g", PS, sws[J],
                                dchi, ddchi[:, c]))
                assert np.abs(e.imag).max() < 1e-12
                V4.append(e.real)
        return V4

    V4 = fields4(PLL, PSS)
    V4noSS = fields4(PLL, 0 * PSS)
    sscontrib = max(np.abs(a - b).max() for a, b in zip(V4, V4noSS))
    assert sscontrib > 1e-3, "small-component contribution too small to test"
    fn4 = fnabla_of(V4)
    assert uvals("gga", V4, fn4)[1].min() > 0.02, \
        "need n- > 0 in the 4C test fields"

    dPL = block({J: hcomp(0.2) for J in ("s", "x", "y", "z")})
    dPS = block({J: hcomp(0.3) for J in ("s", "x", "y", "z")})
    dV4 = fields4(dPL, dPS)          # fields are linear in the blocks

    def E4(lam):
        return energy("gga", [v + lam * d_ for v, d_ in zip(V4, dV4)],
                      fn4, w)

    h = 1e-4
    fd1 = (8 * (E4(h) - E4(-h)) - (E4(2 * h) - E4(-2 * h))) / (12 * h)
    pot4 = vxc("gga", V4, fn4)
    an1 = float(np.sum(w * sum(pot4[i] * dV4[i] for i in range(16))))
    check("4C dE/dlambda vs generated potential", abs(fd1 - an1) / abs(an1),
          1e-9)

    d2 = (E4(h) - 2 * E4(0.0) + E4(-h)) / h**2
    d2b = (E4(2 * h) - 2 * E4(0.0) + E4(-2 * h)) / (4 * h**2)
    fd2 = (4 * d2 - d2b) / 3
    C4 = fxc("gga", V4, fn4)
    an2 = float(np.sum(w * np.einsum("ig,ijg,jg->g", np.array(dV4), C4,
                                     np.array(dV4))))
    check("4C d2E/dlambda2 vs generated fxc kernel",
          abs(fd2 - an2) / abs(an2), 1e-7)

    # 6. the emitted C++ against the NumPy reference ---------------------------
    cxx_err = _check_cxx(V4, fn4, uvals, ns)
    if cxx_err is None:
        print("  [SKIP] emitted C++ vs NumPy (no C++ compiler)")
    else:
        check("emitted C++ vs NumPy (vxc, fxc contraction)", cxx_err, 1e-12)

    status = "OK " if failures == 0 else "FAIL"
    print(f"[{status}] noncollinear_validate: {tested} checks, "
          f"{failures} failures")
    return failures


def _check_cxx(V, fnab, uvals, ns):
    """Compile the emitted C++ header and compare its GGA potential and
    kernel contraction against the NumPy path at one grid point.  Returns
    the max relative deviation, or None when no compiler is available."""
    import shutil
    import subprocess
    import tempfile

    cxx = shutil.which("g++") or shutil.which("clang++")
    if cxx is None:
        return None

    from ..emitters.ncwriter import emit_cxx
    fnames = _field_names("gga")
    args1 = _params("gga", 1)
    args2 = _params("gga", 2)

    rng = np.random.default_rng(5)
    pt = 0                                   # compare at one grid point
    u = uvals("gga", V, fnab)
    lx1 = _deriv_arrays(libxc_args("gga", 1), u)
    lx2 = _deriv_arrays(libxc_args("gga", 2), u)
    trial = [rng.standard_normal(len(V[0])) for _ in fnames]

    vals = {n: float(a[pt]) for n, a in zip(fnames, V)}
    vals.update({k: float(v[pt]) for k, v in lx1.items()})
    vals.update({k: float(v[pt]) for k, v in lx2.items()})
    vals["f_nabla"] = float(fnab[pt])
    tvals = {f"t{n}": float(t[pt]) for n, t in zip(fnames, trial)}

    n = len(fnames)
    with tempfile.TemporaryDirectory() as d:
        with open(f"{d}/nc.hpp", "w") as f:
            f.write(emit_cxx(("gga",)))
        argv = ", ".join(repr(vals[a]) for a in args1)
        argw = ", ".join(repr(vals[a]) for a in args2)
        argt = ", ".join(repr(tvals[f"t{q}"]) for q in fnames)
        src = f'''#include "nc.hpp"
#include <cstdio>
int main() {{
  double v[{n}], k[{n}];
  xckernel::nc_vxc_gga({argv}, {", ".join(f"v[{i}]" for i in range(n))});
  xckernel::nc_fxc_contract_gga({argw}, {argt},
    {", ".join(f"k[{i}]" for i in range(n))});
  for (int i = 0; i < {n}; ++i) printf("%.17e %.17e\\n", v[i], k[i]);
  return 0;
}}
'''
        with open(f"{d}/main.cpp", "w") as f:
            f.write(src)
        r = subprocess.run([cxx, "-std=c++17", "-O2", "-I", d,
                            f"{d}/main.cpp", "-o", f"{d}/a.out"],
                           capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError("generated C++ failed to compile:\n"
                               + r.stderr[:2000])
        out = subprocess.run([f"{d}/a.out"], capture_output=True, text=True,
                             check=True).stdout.split()

    got_v = np.array([float(x) for x in out[0::2]])
    got_k = np.array([float(x) for x in out[1::2]])

    fn_v = ns["nc_vxc_gga"]
    ref_v = np.array([r[pt] for r in _call(fn_v, args1, {
        **{n_: a for n_, a in zip(fnames, V)}, **lx1, "f_nabla": fnab})])
    fn_c = ns["nc_fxc_gga"]
    C = _call(fn_c, args2, {**{n_: a for n_, a in zip(fnames, V)},
                            **lx2, "f_nabla": fnab})
    ref_k = np.einsum("ijg,jg->ig", C, np.array(trial))[:, pt]

    scale = max(np.abs(ref_v).max(), np.abs(ref_k).max(), 1e-30)
    return float(max(np.abs(got_v - ref_v).max(),
                     np.abs(got_k - ref_k).max()) / scale)


def _field_names(family):
    return [s.name for s in nc_fields(family)]


def _params(family, order):
    return _field_names(family) + libxc_args(family, order) + ["f_nabla"]


if __name__ == "__main__":
    raise SystemExit(main())
