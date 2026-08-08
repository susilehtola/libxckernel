"""Validate the complex-orbital-coefficient runtime path over a REAL basis.

Complex MO coefficients need no new generated code: every grid field is a
linear functional of the density matrix, the symmetric bilinear kernels
annihilate the imaginary (antisymmetric) part of a Hermitian P, and the
antisymmetric jp kernel carries it.  This script pins the conventions of
:mod:`..fields` end to end:

  1. field collocation from a complex Hermitian P vs direct MO evaluation
     (rho = sum |psi|^2, jp = Im sum psi* grad psi, ...)
  2. current-free family (mgga_tau): the real Fock matrix from Re P is the
     full complex Fock -- FD in Re and Im displacements of P
  3. current family (cmgga_tau): F = sym(G) + i asym(G) from the real
     general kernel output G -- FD in Re and Im displacements of P
  4. o2 response with a complex Hermitian perturbation: pert fields mapped
     channel-wise (rho-type from dS, jp from dA), result reassembled the
     same way, vs FD of the complex Fock matrix
  5. complex BASIS functions: the sesquilinear emission
     (generate_collapsed(..., sesquilinear=True)) against FD in the real
     and imaginary parts of a Hermitian P, Fock and o2 response

The explicit analytic functional of cdft_validate stands in for Libxc, so
all derivative arrays are exact.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from ..codegen import compile_function, generate_collapsed
from ..deriv import LIBXC_MULTISET
from ..fields import collocate, hermitian_fields, hermitian_fock

# --- the explicit test functional (same as cdft_validate) -------------------

_r, _s, _t = sp.symbols("rho sigma tau", positive=True)
_F = (_r**2 + sp.Rational(3, 10) * _s * _r + sp.Rational(1, 5) * _t**2
      + sp.Rational(1, 10) * _r * _t + sp.Rational(1, 20) * _s * _t)
_VAR = {"rho": _r, "sigma": _s, "tau": _t}
_FNUM = sp.lambdify((_r, _s, _t), _F, "numpy")


def _deriv_arrays(names, rho, sigma, tau):
    out = {}
    for name in names:
        e = _F
        for var, cnt in LIBXC_MULTISET[name].items():
            e = sp.diff(e, _VAR[var], cnt)
        out[name] = sp.lambdify((_r, _s, _t), e, "numpy")(
            rho, sigma, tau) * np.ones_like(rho)
    return out


def _call(fn, gen, args):
    sig = gen.source.split("(", 1)[1].split(")", 1)[0]
    return fn(*[args[p.strip()] for p in sig.split(",")])


# --- energies from a complex Hermitian P -------------------------------------

def _exc_plain(P, chi, dchi, w):
    """Current-free family: E depends on Re P only."""
    f = hermitian_fields(P, chi, dchi)
    return float(np.sum(w * _FNUM(f["rho"], f["sigma"], f["tau"])))


def _exc_current(P, chi, dchi, w):
    """Current family: tau~ = tau - jp^2/(2 rho), jp from -Im P."""
    f = hermitian_fields(P, chi, dchi)
    taut = f["tau"] - 0.5 * np.einsum("cg,cg->g", f["jp"], f["jp"]) / f["rho"]
    return float(np.sum(w * _FNUM(f["rho"], f["sigma"], taut)))


def main():
    rng = np.random.default_rng(11)
    nbf, ng, nocc = 4, 24, 2
    chi = rng.uniform(0.5, 1.5, (nbf, ng))
    dchi = rng.standard_normal((3, nbf, ng))
    w = rng.uniform(0.1, 1.0, ng)

    # complex MO coefficients, Hermitian P = C C^dagger with a dominant
    # real diagonal so the density stays positive
    C = (np.eye(nbf)[:, :nocc] + 0.25 * rng.standard_normal((nbf, nocc))
         + 0.25j * rng.standard_normal((nbf, nocc)))
    P = C.conj() @ C.T
    assert np.abs(P.imag).max() > 1e-3, "test P must be genuinely complex"

    tested = failures = 0

    def check(label, err, tol):
        nonlocal tested, failures
        tested += 1
        ok = err < tol
        if not ok:
            failures += 1
        print(f"  [{'OK' if ok else 'FAIL'}] {label}: err {err:.2e}")

    # 1. fields vs direct MO evaluation ---------------------------------------
    psi = np.einsum("ui,ug->ig", C.conj(), chi)          # psi_i^*(r_g)
    dpsi = np.einsum("ui,cug->cig", C, dchi)             # grad psi_i(r_g)
    psiv = np.einsum("ui,ug->ig", C, chi)                # psi_i(r_g)
    f = hermitian_fields(P, chi, dchi)
    rho_ref = np.einsum("ig,ig->g", psi, psiv).real
    jp_ref = np.einsum("ig,cig->cg", psi, dpsi).imag
    tau_ref = 0.5 * np.einsum("cig,cig->g",
                              dpsi.conj(), dpsi).real
    grad_ref = 2.0 * np.einsum("ig,cig->cg", psi, dpsi).real
    check("rho vs MO", float(np.abs(f["rho"] - rho_ref).max()), 1e-13)
    check("grad rho vs MO", float(np.abs(f["grad_rho"] - grad_ref).max()),
          1e-13)
    check("tau vs MO", float(np.abs(f["tau"] - tau_ref).max()), 1e-13)
    check("jp vs MO", float(np.abs(f["jp"] - jp_ref).max()), 1e-13)
    for key in ("rho", "grad_rho", "tau", "jp"):
        assert not np.iscomplexobj(f[key])

    # Hermitian displacement helpers ------------------------------------------
    def sym_dir(u, v):
        E = np.zeros((nbf, nbf)); E[u, v] += 0.5; E[v, u] += 0.5
        return E.astype(complex)

    def asym_dir(u, v):
        E = np.zeros((nbf, nbf)); E[u, v] += 0.5; E[v, u] -= 0.5
        return 1j * E

    def fd(exc, D, h):
        def d(hh):
            return (exc(P + hh * D, chi, dchi, w)
                    - exc(P - hh * D, chi, dchi, w)) / (2 * hh)
        return (4 * d(h) - d(2 * h)) / 3

    # 2. current-free family: real Fock from Re P is the full answer ----------
    from ..kernel import fock
    ki = fock("mgga_tau")
    gen = generate_collapsed(ki, "mfock")
    fn = compile_function(gen)
    lx = _deriv_arrays(gen.libxc_args, f["rho"], f["sigma"], f["tau"])
    F = _call(fn, gen, {"w": w, "chi": chi, "dchi": dchi,
                        "grad_rho": f["grad_rho"], **lx})
    err = 0.0
    for (u, v) in [(0, 0), (0, 2), (1, 3)]:
        ref = fd(_exc_plain, sym_dir(u, v), 5e-4)
        err = max(err, abs(F[u, v] - ref) / max(abs(ref), 1e-14))
    check("mgga_tau Fock vs FD (sym)", err, 1e-7)
    err = max(abs(fd(_exc_plain, asym_dir(u, v), 5e-4))
              for (u, v) in [(0, 2), (1, 3)])
    check("mgga_tau imag channel inert", err, 1e-9)

    # 3. current family: F = sym(G) + i asym(G) --------------------------------
    ki_c = fock("cmgga_tau")
    gen_c = generate_collapsed(ki_c, "cfock")
    fn_c = compile_function(gen_c)
    taut = f["tau"] - 0.5 * np.einsum("cg,cg->g", f["jp"], f["jp"]) / f["rho"]
    lx_c = _deriv_arrays(gen_c.libxc_args, f["rho"], f["sigma"], taut)
    base_c = {"w": w, "chi": chi, "dchi": dchi, "grad_rho": f["grad_rho"],
              "jp": f["jp"], "inv_rho": 1.0 / f["rho"], **lx_c}
    G = _call(fn_c, gen_c, base_c)
    Fc = hermitian_fock(G)
    assert np.abs(Fc - Fc.conj().T).max() < 1e-13, "Fock must be Hermitian"
    err = 0.0
    for (u, v) in [(0, 0), (0, 2), (1, 3)]:
        # dE = Re tr(Fc dP) for Hermitian dP: sym probes Re Fc, asym Im Fc
        ref_s = fd(_exc_current, sym_dir(u, v), 5e-4)
        got_s = float(Fc[u, v].real)
        ref_a = fd(_exc_current, asym_dir(u, v), 5e-4)
        got_a = -float(Fc[u, v].imag)
        err = max(err, abs(got_s - ref_s) / max(abs(ref_s), 1e-14))
        if (u, v) != (0, 0):
            err = max(err, abs(got_a - ref_a) / max(abs(ref_a), 1e-14))
    check("cmgga_tau complex Fock vs FD", err, 1e-6)

    # 4. o2 response with a complex Hermitian perturbation ---------------------
    from ..response import response_fock
    ki2 = response_fock("cmgga_tau", 2)
    gen2 = generate_collapsed(ki2, "ck2")
    fn2 = compile_function(gen2)
    dP = (rng.standard_normal((nbf, nbf))
          + 1j * rng.standard_normal((nbf, nbf)))
    dP = 0.5 * (dP + dP.conj().T)                        # Hermitian direction
    pf = collocate(dP.real, chi, dchi)                   # rho-type channels
    pj = collocate(dP.imag, chi, dchi)["jp"]             # current channel
    lx2 = _deriv_arrays(gen2.libxc_args, f["rho"], f["sigma"], taut)
    base_c = {**base_c, **lx2}
    dG = _call(fn2, gen2, {**base_c, "rho_p1": pf["rho"],
                           "grad_rho_p1": pf["grad_rho"],
                           "tau_p1": pf["tau"], "jp_p1": pj})
    dFc = hermitian_fock(dG)

    def cfock_at(Ph):
        fh = hermitian_fields(Ph, chi, dchi)
        tt = fh["tau"] - 0.5 * np.einsum("cg,cg->g", fh["jp"],
                                         fh["jp"]) / fh["rho"]
        lxh = _deriv_arrays(gen_c.libxc_args, fh["rho"], fh["sigma"], tt)
        Gh = _call(fn_c, gen_c, {"w": w, "chi": chi, "dchi": dchi,
                                 "grad_rho": fh["grad_rho"], "jp": fh["jp"],
                                 "inv_rho": 1.0 / fh["rho"], **lxh})
        return hermitian_fock(Gh)

    def dfd(h):
        return (cfock_at(P + h * dP) - cfock_at(P - h * dP)) / (2 * h)
    ref = (4 * dfd(5e-4) - dfd(1e-3)) / 3
    err = float(np.abs(dFc - ref).max() / np.abs(ref).max())
    check("o2 response, complex Hermitian pert vs FD", err, 1e-7)

    # 5. complex basis functions: sesquilinear emission ------------------------
    chix = chi + 0.3j * rng.standard_normal((nbf, ng))
    dchix = dchi + 0.3j * rng.standard_normal((3, nbf, ng))
    chix_c, dchix_c = np.conj(chix), np.conj(dchix)
    f5 = collocate(P, chix, dchix)
    assert max(np.abs(f5[k].imag).max()
               for k in ("rho", "grad_rho", "tau")) < 1e-13, \
        "Hermitian P must give real fields over a complex basis"
    assert f5["rho"].real.min() > 0.02, "test density not positive"
    rho5, grad5 = f5["rho"].real, f5["grad_rho"].real
    sig5, tau5 = f5["sigma"].real, f5["tau"].real

    def exc5(Pm, *_):
        fm = collocate(Pm, chix, dchix)
        return float(np.sum(w * _FNUM(fm["rho"].real, fm["sigma"].real,
                                      fm["tau"].real)))

    gen5 = generate_collapsed(ki, "sfock", sesquilinear=True)
    fn5 = compile_function(gen5)
    lx5 = _deriv_arrays(gen5.libxc_args, rho5, sig5, tau5)
    F5 = _call(fn5, gen5, {"w": w, "chi": chix, "chi_c": chix_c,
                           "dchi": dchix, "dchi_c": dchix_c,
                           "grad_rho": grad5, **lx5})
    assert np.abs(F5 - F5.conj().T).max() < 1e-12, "Fock must be Hermitian"
    err = 0.0
    for (u, v) in [(0, 0), (0, 2), (1, 3)]:
        ref_s = fd(exc5, sym_dir(u, v), 5e-4)
        err = max(err, abs(float(F5[u, v].real) - ref_s)
                  / max(abs(ref_s), 1e-14))
        if (u, v) != (0, 0):
            ref_a = fd(exc5, asym_dir(u, v), 5e-4)
            err = max(err, abs(-float(F5[u, v].imag) - ref_a)
                      / max(abs(ref_a), 1e-14))
    check("sesquilinear Fock (complex basis) vs FD", err, 1e-6)

    ki25 = response_fock("mgga_tau", 2)
    gen25 = generate_collapsed(ki25, "sk2", sesquilinear=True)
    fn25 = compile_function(gen25)
    dP5 = rng.standard_normal((nbf, nbf)) \
        + 1j * rng.standard_normal((nbf, nbf))
    dP5 = 0.5 * (dP5 + dP5.conj().T)
    pf5 = collocate(dP5, chix, dchix)
    lx25 = _deriv_arrays(gen25.libxc_args, rho5, sig5, tau5)
    dF5 = _call(fn25, gen25, {"w": w, "chi": chix, "chi_c": chix_c,
                              "dchi": dchix, "dchi_c": dchix_c,
                              "grad_rho": grad5,
                              "rho_p1": pf5["rho"].real,
                              "grad_rho_p1": pf5["grad_rho"].real,
                              "tau_p1": pf5["tau"].real, **lx25})

    def sfock_at(Pm):
        fm = collocate(Pm, chix, dchix)
        lxm = _deriv_arrays(gen5.libxc_args, fm["rho"].real,
                            fm["sigma"].real, fm["tau"].real)
        return _call(fn5, gen5, {"w": w, "chi": chix, "chi_c": chix_c,
                                 "dchi": dchix, "dchi_c": dchix_c,
                                 "grad_rho": fm["grad_rho"].real, **lxm})

    def dfd5(h):
        return (sfock_at(P + h * dP5) - sfock_at(P - h * dP5)) / (2 * h)
    ref = (4 * dfd5(5e-4) - dfd5(1e-3)) / 3
    err = float(np.abs(dF5 - ref).max() / np.abs(ref).max())
    check("sesquilinear o2 response vs FD", err, 1e-7)

    status = "OK " if failures == 0 else "FAIL"
    print(f"[{status}] complex_validate: {tested} checks, {failures} failures")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
