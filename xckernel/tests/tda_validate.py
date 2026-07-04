"""End-to-end layer-2 validation: reproduce PySCF's TDA sigma vector.

(A z)_ia = (e_a - e_i) z_ia + [2(ia|jb) + (ia|fxc_singlet|jb)] z_jb

built from xckernel layers: transition_dm / project_ov (layer 2, algebra.py)
around the batched singlet fxc contraction (layer 1, response_fock_st), with
PySCF supplying only the Coulomb matrix.  Reference: the vind closure from
PySCF's own TDA.gen_vind.  Pure exchange functionals (no HF exchange), LDA and
GGA; expect machine precision.
"""

from __future__ import annotations

import numpy as np

from ..algebra import tda_sigma
from ..codegen import compile_function, generate
from ..spin_kernel import response_fock_st


def check_tda(family: str, nz: int = 3):
    from pyscf import gto, dft
    from pyscf.dft import numint

    xc = {"lda": "LDA_X,", "gga": "GGA_X_PBE,"}[family]
    xctype = {"lda": "LDA", "gga": "GGA"}[family]

    mol = gto.M(atom="O 0 0 0; H 0 0 0.96; H 0 0.93 -0.24", basis="sto-3g",
                verbose=0)
    mf = dft.RKS(mol); mf.xc = xc; mf.verbose = 0; mf.kernel()

    mo_occ = mf.mo_occ
    Co = mf.mo_coeff[:, mo_occ > 0]
    Cv = mf.mo_coeff[:, mo_occ == 0]
    nocc, nvir = Co.shape[1], Cv.shape[1]
    e_ia = mf.mo_energy[mo_occ == 0] - mf.mo_energy[mo_occ > 0, None]

    # ---- grid data + polarized Libxc arrays at the closed-shell point ----
    grids = mf.grids
    deriv = 0 if xctype == "LDA" else 1
    ao = numint.eval_ao(mol, grids.coords, deriv=deriv)
    dm0 = mf.make_rdm1()
    rho = numint.eval_rho(mol, ao, dm0, xctype=xctype)
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
    for name, arr in zip(names_v + names_f,
                         list(vxc[:len(names_v)]) + list(fxc[:len(names_f)])):
        A = col(arr)
        for c in range(A.shape[1]):
            libxc[f"{name}_{c}"] = A[:, c]

    if xctype == "LDA":
        chi = ao.T; dchi = np.zeros((3,) + chi.shape)
    else:
        chi = ao[0].T; dchi = np.transpose(ao[1:4], (0, 2, 1))

    # ---- batched singlet fxc contraction (layer 1) ----
    gen = generate(response_fock_st(family, 2, (+1,)), "st", batch=True)
    fn = compile_function(gen)

    def vxc_st(dms):
        rho1 = np.einsum("xuv,ug,vg->xg", dms, chi, chi)
        args = [grids.weights, chi, dchi]
        if gen.uses_grad_rho_a:
            args.append(rho[1:4] * .5)
        if gen.pert_grads:
            grad1 = np.einsum("xuv,iug,vg->xig", dms, dchi, chi) \
                + np.einsum("xuv,ug,ivg->xig", dms, chi, dchi)
            args.append(grad1)
        args.append(rho1)
        args += [libxc[name] for name in gen.libxc_args]
        return fn(*args)

    def vresp(dms):
        vj = mf.get_j(mol, dms)
        # PySCF convention: the singlet kernel is halved (alpha+beta folding)
        return vj + 0.5 * vxc_st(dms)

    # ---- compare against PySCF's own TDA sigma ----
    td = mf.TDA(); td.singlet = True
    vind, hdiag = td.gen_vind(mf)

    rng = np.random.default_rng(4)
    zs = rng.standard_normal((nz, nocc, nvir))
    ref = vind(zs.reshape(nz, -1)).reshape(nz, nocc, nvir)
    ours = tda_sigma(zs, e_ia, Co, Cv, vresp)

    err = np.max(np.abs(ours - ref)); scale = np.max(np.abs(ref)) or 1
    return err, err / scale


if __name__ == "__main__":
    print("TDA sigma vector (A z) vs PySCF TDA.gen_vind, H2O/sto-3g")
    for fam in ("lda", "gga"):
        err, rel = check_tda(fam)
        print(f"  [{'OK ' if rel < 1e-12 else 'FAIL'}] {fam:4s} "
              f"abs={err:.3e} rel={rel:.3e}")
