"""End-to-end spin-polarized demo: generated code vs PySCF UKS.

Open-shell OH radical.  We build the alpha/beta XC Fock matrices and the
spin-resolved fxc response with xckernel's generated einsum code, and compare to
PySCF's own numint.nr_uks (XC Fock) and numint.nr_uks_fxc (fxc response).
"""

from __future__ import annotations

import numpy as np
from pyscf import gto, dft
from pyscf.dft import numint

from ..engine.spin_kernel import fock_spin, kernel_spin
from ..emitters.codegen import generate, compile_function

XCTYPE = {"lda": "LDA", "gga": "GGA"}
XC = {"lda": "LDA_X,", "gga": "PBE"}
SPINS = ("a", "b")
SIDX = {"a": 0, "b": 1}


def _grid_data(mol, grids, dm, xctype):
    deriv = 0 if xctype == "LDA" else 1
    ao = numint.eval_ao(mol, grids.coords, deriv=deriv)
    rho_a = numint.eval_rho(mol, ao, dm[0], xctype=xctype)
    rho_b = numint.eval_rho(mol, ao, dm[1], xctype=xctype)
    if xctype == "LDA":
        chi = ao.T
        dchi = np.zeros((3,) + chi.shape)
        grad = {"a": None, "b": None}
        rho_in = (rho_a, rho_b)
    else:
        chi = ao[0].T
        dchi = np.transpose(ao[1:4], (0, 2, 1))
        grad = {"a": rho_a[1:4], "b": rho_b[1:4]}
        rho_in = (rho_a, rho_b)
    return ao, chi, dchi, grad, rho_in


def _libxc_dict(xc, rho_in, xctype, deriv):
    exc, vxc, fxc = dft.libxc.eval_xc(xc, rho_in, spin=1, deriv=deriv)[:3]
    ng = len(exc)

    def col(arr):
        arr = np.asarray(arr)
        return arr if arr.shape[0] == ng else arr.T   # -> (ng, ncomp)

    d = {"vrho": col(vxc[0])}
    if xctype != "LDA":
        d["vsigma"] = col(vxc[1])
    if fxc is not None:
        d["v2rho2"] = col(fxc[0])
        if xctype != "LDA":
            d["v2rhosigma"] = col(fxc[1])
            d["v2sigma2"] = col(fxc[2])
    return d


def _value(name, libxc):
    array, comp = name.rsplit("_", 1)
    return libxc[array][:, int(comp)]


def _call(gen, fn, w, chi, dchi, grad, libxc):
    args = [w, chi, dchi]
    if gen.uses_lapl_chi:
        raise NotImplementedError("meta-GGA not in this demo")
    if gen.uses_grad_rho_a:
        args.append(grad["a"])
    if gen.uses_grad_rho_b:
        args.append(grad["b"])
    args += [_value(n, libxc) for n in gen.libxc_args]
    return fn(*args)


def demo_fock(family, mol, grids, dm):
    xc, xctype = XC[family], XCTYPE[family]
    ni = numint.NumInt()
    ao, chi, dchi, grad, rho_in = _grid_data(mol, grids, dm, xctype)
    libxc = _libxc_dict(xc, rho_in, xctype, deriv=1)
    w = grids.weights

    F = {}
    for s in SPINS:
        gen = generate(fock_spin(family, s), f"fock_{s}")
        F[s] = _call(gen, compile_function(gen), w, chi, dchi, grad, libxc)

    _, _, vref = ni.nr_uks(mol, grids, xc, dm)
    errs = [np.max(np.abs(F[s] - vref[SIDX[s]])) for s in SPINS]
    scale = np.max(np.abs(vref)) or 1
    return max(errs), max(errs) / scale


def demo_fxc(family, mol, grids, dm, dm1):
    xc, xctype = XC[family], XCTYPE[family]
    ni = numint.NumInt()
    ao, chi, dchi, grad, rho_in = _grid_data(mol, grids, dm, xctype)
    libxc = _libxc_dict(xc, rho_in, xctype, deriv=2)
    w = grids.weights

    # R^s = sum_t  g^{st} : dm1^t
    R = {}
    for s in SPINS:
        acc = 0.0
        for t in SPINS:
            gen = generate(kernel_spin(family, s, t), f"k_{s}{t}")
            G = _call(gen, compile_function(gen), w, chi, dchi, grad, libxc)
            acc = acc + np.einsum("uvts,ts->uv", G, dm1[SIDX[t]])
        R[s] = acc

    Rref = ni.nr_uks_fxc(mol, grids, xc, dm, dm1, hermi=0)
    errs = [np.max(np.abs(R[s] - Rref[SIDX[s]])) for s in SPINS]
    scale = np.max(np.abs(Rref)) or 1
    return max(errs), max(errs) / scale


def main():
    mol = gto.M(atom="O 0 0 0; H 0 0 0.97", basis="sto-3g", spin=1, verbose=0)
    grids = dft.gen_grid.Grids(mol); grids.level = 3; grids.build()

    mf = dft.UKS(mol); mf.xc = "LDA,VWN"; mf.verbose = 0; mf.kernel()
    dm = mf.make_rdm1()
    rng = np.random.default_rng(0)
    dm1 = np.array([(lambda a: a + a.T)(rng.standard_normal(dm[0].shape))
                    for _ in SPINS])

    print(f"OH/sto-3g (doublet), nao={mol.nao}, ngrid={len(grids.weights)}\n")
    print("UKS XC Fock  F^s vs PySCF numint.nr_uks")
    for fam in ("lda", "gga"):
        err, rel = demo_fock(fam, mol, grids, dm)
        print(f"  [{'OK ' if rel < 1e-9 else 'FAIL'}] {fam:4s} "
              f"abs={err:.3e} rel={rel:.3e}")

    print("\nUKS fxc response  R^s = sum_t g^{st}:dm1^t vs PySCF nr_uks_fxc")
    for fam in ("lda", "gga"):
        err, rel = demo_fxc(fam, mol, grids, dm, dm1)
        print(f"  [{'OK ' if rel < 1e-9 else 'FAIL'}] {fam:4s} "
              f"abs={err:.3e} rel={rel:.3e}")


if __name__ == "__main__":
    main()
