"""End-to-end validation of the libxckernel package: emit the C source
package for a family subset, build it with CMake, load the shared library,
and validate every emitted kernel against the NumPy backend on identical
operands.  Requires cmake + a C compiler (and gfortran for the Fortran
module, built as part of the same package).
"""

from __future__ import annotations

import ctypes
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ..catalog import CatalogEntry, _integrand_for, build_catalog
from ..emitters.cbackend import scal_order
from ..emitters.codegen import collapse, compile_function, generate_collapsed


_H6_COMPS = ("xx", "xy", "xz", "yy", "yz", "zz")


def build_and_validate(families=("lda", "gga", "hmgga"), max_order=3,
                       nbf=4, ng=50, seed=3, verbose=False):
    with tempfile.TemporaryDirectory() as td:
        pkg = Path(td) / "libxck"
        build_catalog(str(pkg), families, max_order, verbose=verbose,
                      backend="c")
        bld = pkg / "build"
        bld.mkdir()
        subprocess.run(["cmake", "..", "-DBUILD_SHARED_LIBS=ON",
                        "-DCMAKE_BUILD_TYPE=Release"],
                       cwd=bld, check=True, capture_output=True)
        subprocess.run(["make", "-j8"], cwd=bld, check=True,
                       capture_output=True)
        lib = ctypes.CDLL(str(bld / "libxckernel.so"))
        man = json.loads((pkg / "manifest.json").read_text())

        P = ctypes.POINTER(ctypes.c_double)
        rng = np.random.default_rng(seed)
        tested = failures = 0
        for k in man["kernels"]:
            # skip cross-backend pointer entries (e.g. the GIAO notes,
            # which carry no "abi"/"order") and the order-0 energy kernels
            if "abi" not in k or k.get("order", 0) == 0:
                continue
            e = CatalogEntry(k["family"], k["spin"], k["order"],
                             tuple(k.get("parities", ())))
            ki = _integrand_for(e)
            ck = collapse(ki)
            gen = generate_collapsed(ki, "npk", batch=False)

            chi = np.ascontiguousarray(rng.standard_normal((nbf, ng)))
            dchi = np.ascontiguousarray(rng.standard_normal((3, nbf, ng)))
            lapl = np.ascontiguousarray(rng.standard_normal((nbf, ng)))
            hess = np.ascontiguousarray(rng.standard_normal((6, nbf, ng)))
            scal = {n: np.ascontiguousarray(rng.standard_normal(ng))
                    for n in scal_order(ck)}

            args = [scal["w"], chi, dchi] \
                + ([lapl] if gen.uses_lapl_chi else []) \
                + ([hess] if "hess_chi" in ck.params else [])
            for p in ck.params:
                if p in ("w", "chi", "dchi", "lapl_chi", "hess_chi"):
                    continue
                if p.startswith("hess_rho"):
                    args.append(np.stack([scal[f"{p}_{c}"]
                                          for c in _H6_COMPS]))
                elif p.startswith(("grad_rho", "jp")):
                    args.append(np.stack([scal[f"{p}_{ax}"]
                                          for ax in "xyz"]))
                else:
                    args.append(scal[p])
            ref = compile_function(gen)(*args)

            f = getattr(lib, k["name"])
            f.restype = ctypes.c_int
            names = scal_order(ck)
            sp = (P * len(names))(*[scal[n].ctypes.data_as(P)
                                    for n in names])
            out = np.zeros((nbf, nbf))
            rc = f(ctypes.c_int64(ng), ctypes.c_int64(nbf),
                   chi.ctypes.data_as(P), dchi.ctypes.data_as(P),
                   lapl.ctypes.data_as(P), hess.ctypes.data_as(P),
                   sp, out.ctypes.data_as(P))
            ok = rc == 0 and np.allclose(out, ref, atol=1e-12, rtol=1e-12)
            tested += 1
            if not ok:
                failures += 1
                print(f"  [FAIL] {k['name']}")

        # the runtime layer: self-describing dispatch through the same .so
        from ..runtime import Library
        rt = Library(str(bld / "libxckernel.so"))
        name = "xck_gga_r_o2"
        rng2 = np.random.default_rng(1)
        chi = rng2.standard_normal((nbf, ng))
        dchi = rng2.standard_normal((3, nbf, ng))
        ops = {}
        for n in rt.scal_names(name):
            if n == "w":
                continue
            base = n[:-2] if n.endswith(("_x", "_y", "_z")) else n
            if base != n:
                ops.setdefault(base, rng2.standard_normal((3, ng)))
            else:
                ops[n] = rng2.standard_normal(ng)
        w2 = rng2.uniform(0.1, 1.0, ng)
        from ..runtime import _NumpyKernel
        F1 = rt(name, chi=chi, dchi=dchi, w=w2, **ops)
        F2 = _NumpyKernel(name)(chi=chi, dchi=dchi, w=w2, **ops)
        tested += 1
        if not np.allclose(F1, F2, atol=1e-12):
            failures += 1
            print("  [FAIL] runtime.Library dispatch")

        # datatype templating: instantiate a kernel at long double through
        # the header-only path and compare against the double ABI result
        prog = pkg / "ld_test.cpp"
        prog.write_text(r'''
#include "xckernel/kernels/xck_gga_r_o2.hpp"
#include <cstdio>
#include <vector>
extern "C" int xck_gga_r_o2(int64_t, int64_t, const double*, const double*,
                            const double*, const double*,
                            const double* const*, double*);
extern "C" const int xck_gga_r_o2_n_scal;
int main() {
    const int64_t nbf = 3, ng = 20;
    const int ns = xck_gga_r_o2_n_scal;
    std::vector<double> chi(nbf*ng), dchi(3*nbf*ng);
    std::vector<std::vector<double>> scal(ns, std::vector<double>(ng));
    unsigned s = 12345;
    auto rnd = [&]() { s = 1664525u*s + 1013904223u;
                       return (double)(s % 1000) / 500.0 - 1.0; };
    for (auto& x : chi) x = rnd();
    for (auto& x : dchi) x = rnd();
    for (auto& v : scal) for (auto& x : v) x = rnd();
    std::vector<const double*> sp(ns);
    for (int i = 0; i < ns; i++) sp[i] = scal[i].data();
    std::vector<double> outd(nbf*nbf, 0.0);
    xck_gga_r_o2(ng, nbf, chi.data(), dchi.data(), nullptr, nullptr,
                 sp.data(), outd.data());
    // long double instantiation
    std::vector<long double> chiL(chi.begin(), chi.end()),
        dchiL(dchi.begin(), dchi.end()), outL(nbf*nbf, 0.0L);
    std::vector<std::vector<long double>> scalL(ns);
    std::vector<const long double*> spL(ns);
    for (int i = 0; i < ns; i++) {
        scalL[i].assign(scal[i].begin(), scal[i].end());
        spL[i] = scalL[i].data();
    }
    // T = long double for basis/fields, Txc = double for Libxc arrays
    const int nfld = ns - 4;   // gga o2: 4 libxc arrays, fields first
    std::vector<const double*> xcp(sp.begin() + nfld, sp.end());
    std::vector<const long double*> fldL(spL.begin(), spL.begin() + nfld);
    xckernel::xck_gga_r_o2_t<long double, double>(
        ng, nbf, chiL.data(), dchiL.data(), nullptr, nullptr,
        fldL.data(), xcp.data(), outL.data());
    long double maxerr = 0.0L;
    for (int i = 0; i < nbf*nbf; i++) {
        long double d = outL[i] - (long double)outd[i];
        if (d < 0) d = -d;
        if (d > maxerr) maxerr = d;
    }
    std::printf("%Lg\n", maxerr);
    return maxerr < 1e-12L ? 0 : 1;
}
''')
        r = subprocess.run(
            ["c++", "-std=c++17", "-O2", "-I", str(pkg / "include"),
             str(prog), str(bld / "libxckernel.so"), "-o",
             str(pkg / "ld_test")], capture_output=True, text=True)
        tested += 1
        if r.returncode != 0:
            failures += 1
            print("  [FAIL] long-double compile:", r.stderr[-300:])
        else:
            r2 = subprocess.run([str(pkg / "ld_test")],
                                capture_output=True, text=True,
                                env={"LD_LIBRARY_PATH": str(bld)})
            if r2.returncode != 0:
                failures += 1
                print("  [FAIL] long-double mismatch:", r2.stdout)
        return tested, failures


if __name__ == "__main__":
    tested, failures = build_and_validate()
    status = "OK " if failures == 0 else "FAIL"
    print(f"[{status}] libxckernel package: {tested} kernels built via "
          f"CMake and validated vs NumPy, {failures} failures")
