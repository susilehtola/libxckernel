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
from ..cbackend import scal_order
from ..codegen import collapse, compile_function, generate_collapsed


def build_and_validate(families=("lda", "gga"), max_order=3,
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
            if not k.get("emitted", True) or k["order"] == 0:
                continue
            e = CatalogEntry(k["family"], k["spin"], k["order"],
                             tuple(k.get("parities", ())))
            ki = _integrand_for(e)
            ck = collapse(ki)
            gen = generate_collapsed(ki, "npk", batch=False)

            chi = np.ascontiguousarray(rng.standard_normal((nbf, ng)))
            dchi = np.ascontiguousarray(rng.standard_normal((3, nbf, ng)))
            lapl = np.ascontiguousarray(rng.standard_normal((nbf, ng)))
            scal = {n: np.ascontiguousarray(rng.standard_normal(ng))
                    for n in scal_order(ck)}

            args = [scal["w"], chi, dchi] \
                + ([lapl] if gen.uses_lapl_chi else [])
            for p in ck.params:
                if p in ("w", "chi", "dchi", "lapl_chi"):
                    continue
                if p.startswith("grad_rho"):
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
                   lapl.ctypes.data_as(P), sp, out.ctypes.data_as(P))
            ok = rc == 0 and np.allclose(out, ref, atol=1e-12, rtol=1e-12)
            tested += 1
            if not ok:
                failures += 1
                print(f"  [FAIL] {k['name']}")
        return tested, failures


if __name__ == "__main__":
    tested, failures = build_and_validate()
    status = "OK " if failures == 0 else "FAIL"
    print(f"[{status}] libxckernel package: {tested} kernels built via "
          f"CMake and validated vs NumPy, {failures} failures")
