"""Validate the response contraction engine.

1. Second order (fxc contraction, linear response): generated O(N^2 ng) code
   vs PySCF's numint.nr_rks_fxc on a real molecule -- expect machine precision.
2. Third order (kxc contraction, quadratic response): generated code vs mixed
   finite differences of the Fock matrix F_uv(P + a D1 + b D2) on a fabricated
   grid -- the first quantity beyond what most surveyed codes implement.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..codegen import compile_function, generate
from ..kernel import fock
from ..response import response_fock


def _pert_fields(D, chi, dchi):
    """Perturbed fields rho^X, grad rho^X, tau^X on the grid from a perturbed
    DM.  chi: (nao, ng); dchi: (3, nao, ng)."""
    rho = np.einsum("uv,ug,vg->g", D, chi, chi)
    grad = np.einsum("uv,iug,vg->ig", D, dchi, chi) \
        + np.einsum("uv,ug,ivg->ig", D, chi, dchi)
    tau = 0.5 * np.einsum("uv,iug,ivg->g", D, dchi, dchi)
    return rho, grad, tau


def _call(gen, fn, base_args: Dict[str, np.ndarray], libxc: Dict[str, np.ndarray],
          pert: Dict[str, Dict[str, np.ndarray]]):
    args = [base_args["w"], base_args["chi"], base_args["dchi"]]
    if gen.uses_lapl_chi:
        args.append(base_args["lapl_chi"])
    if gen.uses_grad_rho:
        args.append(base_args["grad_rho"])
    args += [pert[lbl]["grad"] for lbl in (gen.pert_grads or [])]
    for name in gen.pert_scalars or []:
        field, lbl = name.rsplit("_", 1)
        args.append(pert[lbl][field])
    args += [libxc[name] for name in gen.libxc_args]
    return fn(*args)


# --- 1. second order vs PySCF ----------------------------------------------

def check_pyscf(family: str):
    from pylibxc import LibXCFunctional
    from pyscf import gto, dft
    from pyscf.dft import numint

    libxc_name = {"lda": "LDA_X", "gga": "GGA_X_PBE",
                  "mgga_tau": "MGGA_X_SCAN"}[family]
    xc = libxc_name + ","            # exchange-only, matching pylibxc arrays
    xctype = {"lda": "LDA", "gga": "GGA", "mgga_tau": "MGGA"}[family]

    mol = gto.M(atom="O 0 0 0; H 0 0 0.96; H 0 0.93 -0.24", basis="sto-3g",
                verbose=0)
    grids = dft.gen_grid.Grids(mol); grids.level = 3; grids.build()
    mf = dft.RKS(mol); mf.xc = "LDA,VWN"; mf.verbose = 0; mf.kernel()
    dm = mf.make_rdm1()
    rng = np.random.default_rng(0)
    a = rng.standard_normal(dm.shape)
    dm1 = a + a.T

    deriv = 0 if xctype == "LDA" else 1
    ao = numint.eval_ao(mol, grids.coords, deriv=deriv)
    rho0 = numint.eval_rho(mol, ao, dm, xctype=xctype)

    # Libxc derivative arrays by name, straight from pylibxc (no tuple-order
    # guessing); inputs assembled from the PySCF grid densities.
    if xctype == "LDA":
        chi = ao.T; dchi = np.zeros((3,) + chi.shape)
        grad_rho = None
        inp = {"rho": rho0}
    else:
        chi = ao[0].T; dchi = np.transpose(ao[1:4], (0, 2, 1))
        grad_rho = rho0[1:4]
        inp = {"rho": rho0[0],
               "sigma": np.einsum("ig,ig->g", grad_rho, grad_rho)}
        if xctype == "MGGA":
            inp["tau"] = rho0[-1]     # PySCF MGGA layout: tau is the last row
    fn_xc = LibXCFunctional(libxc_name, "unpolarized")
    out = fn_xc.compute(inp, do_fxc=True)
    libxc = {k: v.reshape(-1) for k, v in out.items()
             if k != "zk" and v is not None}

    rho1, grad1, tau1 = _pert_fields(dm1, chi, dchi)
    base = {"w": grids.weights, "chi": chi, "dchi": dchi, "grad_rho": grad_rho}
    pert = {"p1": {"rho": rho1, "grad": grad1, "tau": tau1}}

    gen = generate(response_fock(family, 2), "resp2")
    R = _call(gen, compile_function(gen), base, libxc, pert)

    ni = numint.NumInt()
    Rref = ni.nr_rks_fxc(mol, grids, xc, dm, dm1, hermi=0)
    err = np.max(np.abs(R - Rref)); scale = np.max(np.abs(Rref)) or 1
    return err, err / scale


# --- 2. third order vs finite differences ----------------------------------

def check_third_order(family: str):
    from pylibxc import LibXCFunctional
    name = {"lda": "LDA_X", "gga": "GGA_X_PBE",
            "mgga_tau": "MGGA_X_SCAN"}[family]
    vars_ = {"lda": ["rho"], "gga": ["rho", "sigma"],
             "mgga_tau": ["rho", "sigma", "tau"]}[family]

    rng = np.random.default_rng(7)
    nbf, npts = 4, 200
    w = rng.uniform(0.1, 1.0, npts)
    chi = rng.standard_normal((nbf, npts))
    dchi = rng.standard_normal((3, nbf, npts))
    A = rng.standard_normal((nbf, nbf)); P0 = A @ A.T
    B1 = rng.standard_normal((nbf, nbf)); D1 = B1 + B1.T
    B2 = rng.standard_normal((nbf, nbf)); D2 = B2 + B2.T

    func = LibXCFunctional(name, "unpolarized")

    def fields(P):
        rho = np.einsum("uv,ug,vg->g", P, chi, chi)
        grad = np.einsum("uv,iug,vg->ig", P, dchi, chi) \
            + np.einsum("uv,ug,ivg->ig", P, chi, dchi)
        tau = 0.5 * np.einsum("uv,iug,ivg->g", P, dchi, dchi)
        return rho, grad, tau

    def libxc_at(P, do_kxc=False):
        rho, grad, tau = fields(P)
        inp = {"rho": rho}
        if "sigma" in vars_:
            inp["sigma"] = np.einsum("ig,ig->g", grad, grad)
        if "tau" in vars_:
            inp["tau"] = tau
        out = func.compute(inp, do_fxc=True, do_kxc=do_kxc)
        d = {k: v.reshape(-1) for k, v in out.items() if v is not None}
        return d, rho, grad

    # analytic third-order response Fock via generated code
    libxc0, rho0, grad0 = libxc_at(P0, do_kxc=True)
    rho1, grad1, tau1 = fields(D1)
    rho2, grad2, tau2 = fields(D2)
    base = {"w": w, "chi": chi, "dchi": dchi, "grad_rho": grad0}
    pert = {"p1": {"rho": rho1, "grad": grad1, "tau": tau1},
            "p2": {"rho": rho2, "grad": grad2, "tau": tau2}}
    gen3 = generate(response_fock(family, 3), "resp3")
    K = _call(gen3, compile_function(gen3), base, libxc0, pert)

    # reference: mixed FD of the (validated) Fock matrix
    genF = generate(fock(family), "fockfn")
    fF = compile_function(genF)

    def fock_at(P):
        lib, rho, grad = libxc_at(P)
        b = {"w": w, "chi": chi, "dchi": dchi, "grad_rho": grad}
        return _call(genF, fF, b, lib, {})

    h = 2e-4
    Kfd = (fock_at(P0 + h*D1 + h*D2) - fock_at(P0 + h*D1 - h*D2)
           - fock_at(P0 - h*D1 + h*D2) + fock_at(P0 - h*D1 - h*D2)) / (4*h*h)

    err = np.max(np.abs(K - Kfd)); scale = np.max(np.abs(Kfd)) or 1
    return err, err / scale


if __name__ == "__main__":
    print("Order 2 (fxc contraction) vs PySCF nr_rks_fxc, H2O/sto-3g")
    for fam in ("lda", "gga", "mgga_tau"):
        err, rel = check_pyscf(fam)
        print(f"  [{'OK ' if rel < 1e-12 else 'FAIL'}] {fam:9s} "
              f"abs={err:.3e} rel={rel:.3e}")
    print("Order 3 (kxc contraction) vs mixed FD of Fock, fabricated grid")
    for fam in ("lda", "gga", "mgga_tau"):
        err, rel = check_third_order(fam)
        print(f"  [{'OK ' if rel < 1e-4 else 'FAIL'}] {fam:9s} "
              f"abs={err:.3e} rel={rel:.3e}")
