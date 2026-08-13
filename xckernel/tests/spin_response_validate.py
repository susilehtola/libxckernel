"""Validate the spin-polarized response contraction engine.

1. Order 2 (fxc contraction) vs PySCF numint.nr_uks_fxc on the OH radical --
   expect machine precision, both spin channels.
2. Order 3 (kxc contraction) vs mixed finite differences of the spin Fock
   matrix on a fabricated grid, with a correlation functional so all
   spin couplings are exercised.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from ..emitters.codegen import compile_function, generate
from ..engine.spin_kernel import fock_spin, response_fock_spin

SPINS = ("a", "b")


def _pert_fields(D, chi, dchi):
    rho = np.einsum("uv,ug,vg->g", D, chi, chi)
    grad = np.einsum("uv,iug,vg->ig", D, dchi, chi) \
        + np.einsum("uv,ug,ivg->ig", D, chi, dchi)
    return rho, grad


def _call(gen, fn, base, libxc, pert):
    """Assemble args for a generated spin function.

    base: w/chi/dchi plus ground-state grad_rho_a/b.
    libxc: '<array>_<comp>' -> (ng,).
    pert: label 'a_p1' style -> {'grad': (3,ng)}; scalars keyed by full name.
    """
    args = [base["w"], base["chi"], base["dchi"]]
    if gen.uses_lapl_chi:
        args.append(base["lapl_chi"])
    if gen.uses_grad_rho_a:
        args.append(base["grad_rho_a"])
    if gen.uses_grad_rho_b:
        args.append(base["grad_rho_b"])
    args += [pert[f"grad_{lbl}"] for lbl in (gen.pert_grads or [])]
    args += [pert[name] for name in (gen.pert_scalars or [])]
    args += [libxc[name] for name in gen.libxc_args]
    return fn(*args)


def _libxc_env(out: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    env = {}
    for key, arr in out.items():
        if key == "zk" or arr is None:
            continue
        A = np.atleast_2d(arr.T).T if arr.ndim == 1 else arr
        for c in range(A.shape[1]):
            env[f"{key}_{c}"] = A[:, c]
    return env


# --- 1. order 2 vs PySCF UKS -------------------------------------------------

def check_pyscf(family: str):
    from pyscf import gto, dft
    from pyscf.dft import numint

    xc = {"lda": "LDA_X,", "gga": "PBE"}[family]
    xctype = {"lda": "LDA", "gga": "GGA"}[family]

    mol = gto.M(atom="O 0 0 0; H 0 0 0.97", basis="sto-3g", spin=1, verbose=0)
    grids = dft.gen_grid.Grids(mol); grids.level = 3; grids.build()
    mf = dft.UKS(mol); mf.xc = "LDA,VWN"; mf.verbose = 0; mf.kernel()
    dm = mf.make_rdm1()
    rng = np.random.default_rng(0)
    dm1 = np.array([(lambda m: m + m.T)(rng.standard_normal(dm[0].shape))
                    for _ in SPINS])

    deriv = 0 if xctype == "LDA" else 1
    ao = numint.eval_ao(mol, grids.coords, deriv=deriv)
    rho_a = numint.eval_rho(mol, ao, dm[0], xctype=xctype)
    rho_b = numint.eval_rho(mol, ao, dm[1], xctype=xctype)
    exc, vxc, fxc = dft.libxc.eval_xc(xc, (rho_a, rho_b), spin=1, deriv=2)[:3]
    ng = len(exc)

    def col(arr):
        arr = np.asarray(arr)
        return arr if arr.shape[0] == ng else arr.T

    libxc = {}
    names_v = ["vrho"] if xctype == "LDA" else ["vrho", "vsigma"]
    names_f = (["v2rho2"] if xctype == "LDA"
               else ["v2rho2", "v2rhosigma", "v2sigma2"])
    for name, arr in zip(names_v, vxc):
        A = col(arr)
        for c in range(A.shape[1]):
            libxc[f"{name}_{c}"] = A[:, c]
    for name, arr in zip(names_f, fxc):
        A = col(arr)
        for c in range(A.shape[1]):
            libxc[f"{name}_{c}"] = A[:, c]

    if xctype == "LDA":
        chi = ao.T; dchi = np.zeros((3,) + chi.shape)
        base = {"w": grids.weights, "chi": chi, "dchi": dchi}
    else:
        chi = ao[0].T; dchi = np.transpose(ao[1:4], (0, 2, 1))
        base = {"w": grids.weights, "chi": chi, "dchi": dchi,
                "grad_rho_a": rho_a[1:4], "grad_rho_b": rho_b[1:4]}

    pert = {}
    for s, D in zip(SPINS, dm1):
        r, g = _pert_fields(D, chi, dchi)
        pert[f"rho_{s}_p1"] = r
        pert[f"grad_{s}_p1"] = g

    ni = numint.NumInt()
    Rref = ni.nr_uks_fxc(mol, grids, xc, dm, dm1, hermi=0)

    errs = []
    for i, s in enumerate(SPINS):
        gen = generate(response_fock_spin(family, s, 2), f"r2_{s}")
        R = _call(gen, compile_function(gen), base, libxc, pert)
        errs.append(np.max(np.abs(R - Rref[i])))
    err = max(errs); scale = np.max(np.abs(Rref)) or 1
    return err, err / scale


# --- 2. order 3 vs FD --------------------------------------------------------

def check_third_order(family: str):
    from pylibxc import LibXCFunctional
    name = {"lda": "LDA_C_PW", "gga": "GGA_C_PBE"}[family]
    vars_ = {"lda": ["rho"], "gga": ["rho", "sigma"]}[family]

    rng = np.random.default_rng(11)
    nbf, npts = 3, 150
    w = rng.uniform(0.1, 1.0, npts)
    chi = rng.standard_normal((nbf, npts))
    dchi = rng.standard_normal((3, nbf, npts))

    def sym(m): return m + m.T
    P0 = {s: (lambda A: A @ A.T)(rng.standard_normal((nbf, nbf))) for s in SPINS}
    D1 = {s: sym(rng.standard_normal((nbf, nbf))) for s in SPINS}
    D2 = {s: sym(rng.standard_normal((nbf, nbf))) for s in SPINS}

    func = LibXCFunctional(name, "polarized")

    def fields(P):
        rho = np.einsum("uv,ug,vg->g", P, chi, chi)
        grad = np.einsum("uv,iug,vg->ig", P, dchi, chi) \
            + np.einsum("uv,ug,ivg->ig", P, chi, dchi)
        return rho, grad

    def libxc_at(P, do_kxc=False):
        fa = fields(P["a"]); fb = fields(P["b"])
        inp = {"rho": np.column_stack([fa[0], fb[0]])}
        if "sigma" in vars_:
            inp["sigma"] = np.column_stack([
                np.einsum("ig,ig->g", fa[1], fa[1]),
                np.einsum("ig,ig->g", fa[1], fb[1]),
                np.einsum("ig,ig->g", fb[1], fb[1])])
        out = func.compute(inp, do_fxc=True, do_kxc=do_kxc)
        return _libxc_env(out), fa, fb

    libxc0, fa0, fb0 = libxc_at(P0, do_kxc=True)
    base = {"w": w, "chi": chi, "dchi": dchi,
            "grad_rho_a": fa0[1], "grad_rho_b": fb0[1]}
    pert = {}
    for k, D in (("p1", D1), ("p2", D2)):
        for s in SPINS:
            r, g = _pert_fields(D[s], chi, dchi)
            pert[f"rho_{s}_{k}"] = r
            pert[f"grad_{s}_{k}"] = g

    genF = {s: generate(fock_spin(family, s), f"f_{s}") for s in SPINS}
    fF = {s: compile_function(genF[s]) for s in SPINS}

    def fock_at(P, s):
        lib, fa, fb = libxc_at(P)
        b = {"w": w, "chi": chi, "dchi": dchi,
             "grad_rho_a": fa[1], "grad_rho_b": fb[1]}
        return _call(genF[s], fF[s], b, lib, {})

    h = 2e-4
    errs = []
    for s in SPINS:
        gen3 = generate(response_fock_spin(family, s, 3), f"r3_{s}")
        K = _call(gen3, compile_function(gen3), base, libxc0, pert)

        def P_at(x1, x2):
            return {t: P0[t] + x1*D1[t] + x2*D2[t] for t in SPINS}

        Kfd = (fock_at(P_at(+h, +h), s) - fock_at(P_at(+h, -h), s)
               - fock_at(P_at(-h, +h), s) + fock_at(P_at(-h, -h), s)) / (4*h*h)
        errs.append((np.max(np.abs(K - Kfd)), np.max(np.abs(Kfd)) or 1))
    err = max(e for e, _ in errs); scale = max(sc for _, sc in errs)
    return err, err / scale


# --- 3. singlet/triplet adaptation vs PySCF nr_rks_fxc_st -------------------

def check_st(family: str, singlet: bool):
    from pyscf import gto, dft
    from pyscf.dft import numint
    from ..engine.spin_kernel import response_fock_st

    xc = {"lda": "LDA_X,", "gga": "PBE"}[family]
    xctype = {"lda": "LDA", "gga": "GGA"}[family]

    mol = gto.M(atom="O 0 0 0; H 0 0 0.96; H 0 0.93 -0.24", basis="sto-3g",
                verbose=0)
    grids = dft.gen_grid.Grids(mol); grids.level = 3; grids.build()
    mf = dft.RKS(mol); mf.xc = "LDA,VWN"; mf.verbose = 0; mf.kernel()
    dm = mf.make_rdm1()
    rng = np.random.default_rng(1)
    a = rng.standard_normal(dm.shape)
    dm1 = a + a.T                      # alpha-channel response DM

    deriv = 0 if xctype == "LDA" else 1
    ao = numint.eval_ao(mol, grids.coords, deriv=deriv)
    rho = numint.eval_rho(mol, ao, dm, xctype=xctype)
    # polarized kernel at the closed-shell point (rho/2, rho/2)
    rho_s = rho * .5
    exc, vxc, fxc = dft.libxc.eval_xc(xc, (rho_s, rho_s), spin=1, deriv=2)[:3]
    ng = len(exc)

    def col(arr):
        arr = np.asarray(arr)
        return arr if arr.shape[0] == ng else arr.T

    libxc = {}
    names_v = ["vrho"] if xctype == "LDA" else ["vrho", "vsigma"]
    names_f = (["v2rho2"] if xctype == "LDA"
               else ["v2rho2", "v2rhosigma", "v2sigma2"])
    for name, arr in zip(names_v + names_f, list(vxc[:len(names_v)])
                         + list(fxc[:len(names_f)])):
        A = col(arr)
        for c in range(A.shape[1]):
            libxc[f"{name}_{c}"] = A[:, c]

    if xctype == "LDA":
        chi = ao.T; dchi = np.zeros((3,) + chi.shape)
        base = {"w": grids.weights, "chi": chi, "dchi": dchi}
    else:
        chi = ao[0].T; dchi = np.transpose(ao[1:4], (0, 2, 1))
        base = {"w": grids.weights, "chi": chi, "dchi": dchi,
                "grad_rho_a": rho[1:4] * .5}   # alpha gradient = half of total

    r1, g1 = _pert_fields(dm1, chi, dchi)
    pert = {"rho_a_p1": r1, "grad_a_p1": g1}

    gen = generate(response_fock_st(family, 2, (+1 if singlet else -1,)), "st")
    R = _call(gen, compile_function(gen), base, libxc, pert)

    ni = numint.NumInt()
    Rref = ni.nr_rks_fxc_st(mol, grids, xc, dm, dm1, singlet=singlet)
    Rref = np.asarray(Rref).reshape(R.shape)
    err = np.max(np.abs(R - Rref)); scale = np.max(np.abs(Rref)) or 1
    return err, err / scale


if __name__ == "__main__":
    print("Spin order 2 (fxc) vs PySCF nr_uks_fxc, OH/sto-3g doublet")
    for fam in ("lda", "gga"):
        err, rel = check_pyscf(fam)
        print(f"  [{'OK ' if rel < 1e-12 else 'FAIL'}] {fam:4s} "
              f"abs={err:.3e} rel={rel:.3e}")
    print("Spin order 3 (kxc) vs mixed FD of F^s, fabricated grid")
    for fam in ("lda", "gga"):
        err, rel = check_third_order(fam)
        print(f"  [{'OK ' if rel < 1e-4 else 'FAIL'}] {fam:4s} "
              f"abs={err:.3e} rel={rel:.3e}")
    print("Singlet/triplet adaptation vs PySCF nr_rks_fxc_st, H2O/sto-3g")
    for fam in ("lda", "gga"):
        for singlet in (True, False):
            tag = "singlet" if singlet else "triplet"
            err, rel = check_st(fam, singlet)
            print(f"  [{'OK ' if rel < 1e-12 else 'FAIL'}] {fam:4s} {tag:7s} "
                  f"abs={err:.3e} rel={rel:.3e}")
