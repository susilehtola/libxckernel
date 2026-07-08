"""Validate the C backend against the NumPy backend on identical inputs.

Emits table-driven C kernels, compiles them with the system C compiler
(-O2 -shared), loads via ctypes, and compares against the pattern-collapsed
NumPy kernels on random operands.  Covers all families, response orders up to
4, spin and spin-adapted cases -- including the heavyweight entries whose
coefficient tables hold tens of thousands of monomials.
"""

from __future__ import annotations

import ctypes
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ..cbackend import emit_c, scal_order
from ..codegen import collapse, compile_function, generate_collapsed
from ..kernel import fock
from ..response import response_fock
from ..spin_kernel import fock_spin, response_fock_spin, response_fock_st


_H6_COMPS = ("xx", "xy", "xz", "yy", "yz", "zz")


def _operands(ck, nbf, ng, seed):
    rng = np.random.default_rng(seed)
    chi = np.ascontiguousarray(rng.standard_normal((nbf, ng)))
    dchi = np.ascontiguousarray(rng.standard_normal((3, nbf, ng)))
    lapl_chi = np.ascontiguousarray(rng.standard_normal((nbf, ng)))
    hess_chi = np.ascontiguousarray(rng.standard_normal((6, nbf, ng)))
    scal = {}
    for name in scal_order(ck):
        scal[name] = np.ascontiguousarray(rng.standard_normal(ng))
    scal["w"] = np.abs(scal["w"]) + 0.1
    return chi, dchi, lapl_chi, hess_chi, scal


def _numpy_args(gen, ck, chi, dchi, lapl_chi, hess_chi, scal):
    args = [scal["w"], chi, dchi]
    if gen.uses_lapl_chi:
        args.append(lapl_chi)
    if "hess_chi" in ck.params:
        args.append(hess_chi)
    # vector params, in the same order the generator emits them
    for p in ck.params:
        if p in ("w", "chi", "dchi", "lapl_chi", "hess_chi"):
            continue
        if p.startswith("hess_rho"):
            args.append(np.stack([scal[f"{p}_{c}"] for c in _H6_COMPS]))
        elif p.startswith(("grad_rho", "jp")):
            args.append(np.stack([scal[f"{p}_{ax}"] for ax in "xyz"]))
        else:
            args.append(scal[p])
    return args


def check(name, ki, nbf=4, ng=60, seed=11):
    ck = collapse(ki)
    gen = generate_collapsed(ki, "npk", batch=False)
    fn_np = compile_function(gen)

    chi, dchi, lapl_chi, hess_chi, scal = _operands(ck, nbf, ng, seed)
    ref = fn_np(*_numpy_args(gen, ck, chi, dchi, lapl_chi, hess_chi, scal))

    csrc = emit_c(ck, name)
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / f"{name}.c"
        lib = Path(td) / f"{name}.so"
        src.write_text(csrc)
        subprocess.run(["cc", "-O2", "-shared", "-fPIC", "-o", str(lib),
                        str(src)], check=True)
        dll = ctypes.CDLL(str(lib))
        f = getattr(dll, name)
        f.restype = ctypes.c_int
        P = ctypes.POINTER(ctypes.c_double)
        names = scal_order(ck)
        scal_ptrs = (P * len(names))(*[
            scal[n].ctypes.data_as(P) for n in names])
        out = np.zeros((nbf, nbf))
        rc = f(ctypes.c_int64(ng), ctypes.c_int64(nbf),
               chi.ctypes.data_as(P), dchi.ctypes.data_as(P),
               lapl_chi.ctypes.data_as(P), hess_chi.ctypes.data_as(P),
               scal_ptrs, out.ctypes.data_as(P))
        assert rc == 0

    err = np.max(np.abs(out - ref))
    scale = np.max(np.abs(ref)) or 1.0
    return err, err / scale, len(csrc.splitlines())


if __name__ == "__main__":
    cases = [
        ("xck_lda_r_o1", fock("lda")),
        ("xck_gga_r_o1", fock("gga")),
        ("xck_mgga_r_o1", fock("mgga")),
        ("xck_gga_r_o2", response_fock("gga", 2)),
        ("xck_mgga_tau_r_o3", response_fock("mgga_tau", 3)),
        ("xck_gga_r_o4", response_fock("gga", 4)),
        ("xck_mgga_r_o4", response_fock("mgga", 4)),
        ("xck_gga_ua_o2", response_fock_spin("gga", "a", 2)),
        ("xck_gga_ua_o3", response_fock_spin("gga", "a", 3)),
        ("xck_gga_st_o2_m", response_fock_st("gga", 2, (-1,))),
        ("xck_gga_st_o3_pm", response_fock_st("gga", 3, (+1, -1))),
        ("xck_cmgga_tau_r_o2", response_fock("cmgga_tau", 2)),
        ("xck_hmgga_r_o1", fock("hmgga")),
        ("xck_hmgga_r_o2", response_fock("hmgga", 2)),
        ("xck_hmgga_r_o3", response_fock("hmgga", 3)),
    ]
    print("C backend (table-driven, cc -O2) vs NumPy backend")
    for name, ki in cases:
        err, rel, nloc = check(name, ki)
        print(f"  [{'OK ' if rel < 1e-12 else 'FAIL'}] {name:22s} "
              f"rel={rel:.3e}  ({nloc} LOC)")
