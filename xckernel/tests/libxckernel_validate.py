"""End-to-end validation of the libxckernel package: emit the C source
package for a family subset, build it with CMake, load the shared library,
and validate every emitted kernel against the NumPy backend on identical
operands.  Requires cmake + a C compiler (and gfortran for the Fortran
module, built as part of the same package).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from ..catalog import build_catalog


def _build(pkg: Path) -> Path:
    """Configure and build a generated package; the build directory."""
    bld = pkg / "build"
    bld.mkdir()
    # a small grid block exercises the blocked GEMM path (several
    # blocks and a remainder) at the test's grid size; the prefix
    # lets FindBLAS see an environment's BLAS (conda, venv)
    subprocess.run(["cmake", "..", "-DBUILD_SHARED_LIBS=ON",
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DXCKERNEL_GRID_BLOCK=16",
                    f"-DCMAKE_PREFIX_PATH={sys.prefix}"],
                   cwd=bld, check=True, capture_output=True)
    subprocess.run(["make", "-j3"], cwd=bld, check=True,
                   capture_output=True)
    return bld


def check_index(dll, man, verbose=False):
    """The kernel index against the manifest and the per-kernel exports:
    it lists exactly the manifest's kernels; each entry's kind, rank and
    shape, entry point, operand names and tower orders match. Returns
    (tested, failures)."""
    import ctypes

    from ..emitters.cbackend import ABI_KINDS, out_rank

    class _Info(ctypes.Structure):
        _fields_ = [("name", ctypes.c_char_p), ("kind", ctypes.c_char_p),
                    ("order", ctypes.c_int), ("out_rank", ctypes.c_int),
                    ("out_shape", ctypes.c_char_p),
                    ("fn", ctypes.c_void_p),
                    ("scal_names", ctypes.POINTER(ctypes.c_char_p)),
                    ("n_scal", ctypes.c_int), ("n_fields", ctypes.c_int),
                    ("chi_order", ctypes.c_int), ("dchi_order", ctypes.c_int),
                    ("dtchi_order", ctypes.c_int)]
    tested = failures = 0
    n = ctypes.c_int.in_dll(dll, "xckernel_n_kernels").value
    table = (_Info * n).in_dll(dll, "xckernel_kernels")
    index = {t.name.decode(): t for t in table}
    built = {k["name"]: k for k in man["kernels"] if "abi" in k}
    tested += 1
    if set(index) != set(built):
        failures += 1
        print("  [FAIL] kernel index:", sorted(set(index) ^ set(built)))
    for name, k in built.items():
        t = index.get(name)
        got = (ctypes.c_char_p.in_dll(dll, f"{name}_kind").value.decode(),
               ctypes.c_int.in_dll(dll, f"{name}_out_rank").value,
               ctypes.c_char_p.in_dll(dll, f"{name}_out_shape").value
               .decode())
        want = (k["abi_kind"], out_rank(k["output_shape"]),
                k["output_shape"])
        tested += 1
        if t is None or got != want or \
                (t.kind.decode(), t.out_rank, t.out_shape.decode()) != got:
            failures += 1
            print(f"  [FAIL] kind export {name}: {got} vs {want}")
            continue
        # dispatch metadata: the entry point is the symbol; the operand
        # names, field count and tower orders are the kernel's exports
        bad = []
        if t.fn != ctypes.cast(getattr(dll, name), ctypes.c_void_p).value:
            bad.append("fn")
        kind = got[0]
        if kind == "exc":
            if t.scal_names or t.n_scal or \
                    (t.chi_order, t.dchi_order, t.dtchi_order) != (-1,) * 3:
                bad.append("exc metadata")
        else:
            ns = ctypes.c_int.in_dll(dll, f"{name}_n_scal").value
            nf = ctypes.c_int.in_dll(dll, f"{name}_n_fields").value
            names = [t.scal_names[i].decode() for i in range(t.n_scal)]
            if (t.n_scal, t.n_fields) != (ns, nf) or \
                    names != k["scal_names"]:
                bad.append("operands")
            towers = ABI_KINDS[kind][1]

            def order(prefix):
                for a in towers:
                    if a.split("_")[0] in prefix:
                        return ctypes.c_int.in_dll(
                            dll, f"{name}_{a}_order").value
                return -1
            if (t.chi_order, t.dchi_order, t.dtchi_order) != (
                    order(("chi", "phi")), order(("Dchi",)),
                    order(("DTchi",))):
                bad.append("tower orders")
        tested += 1
        if bad:
            failures += 1
            print(f"  [FAIL] index entry {name}: {', '.join(bad)}")
    return tested, failures


def validate_selection(families=("gga", "mgga_tau"), max_order=2,
                       kinds=("exc", "matrix", "diag", "o2b", "mo2"),
                       verbose=False):
    """A kind-selective package: the manifest records the selection and
    lists exactly the selected entries, the index matches it, the header
    declares only them, and the evaluator carries only the helpers they
    call. Returns (tested, failures)."""
    import ctypes
    import re

    from ..catalog import entries, entry_kind, parse_kinds
    from ..emitters.cbackend import OPTIONAL_HELPERS
    tested = failures = 0
    sel = parse_kinds(",".join(kinds))
    with tempfile.TemporaryDirectory() as td:
        pkg = Path(td) / "libxck"
        build_catalog(str(pkg), families, max_order, verbose=verbose,
                      backend="c", kinds=sel)
        man = json.loads((pkg / "manifest.json").read_text())
        want = {e.name for e in entries(families, max_order, kinds=sel)}
        got = {k["name"] for k in man["kernels"]}
        tested += 1
        if man["kinds"] != list(sel) or got != want or \
                {entry_kind(e) for e in entries(families, max_order)
                 if e.name in got} - set(sel):
            failures += 1
            print(f"  [FAIL] selection manifest: kinds {man['kinds']}, "
                  f"{sorted(got ^ want)}")
        header = (pkg / "include" / "xckernel.h").read_text()
        declared = set(re.findall(r"\b(xck_\w+)\(", header))
        tested += 1
        if declared != want or \
                f'#define XCKERNEL_KINDS "{",".join(sel)}"' not in header:
            failures += 1
            print("  [FAIL] selection header:", sorted(declared ^ want))
        ev = (pkg / "include" / "xckernel" / "evaluator.hpp").read_text()
        srcs = "\n".join(p.read_text() for p in
                         (pkg / "include" / "xckernel" / "kernels")
                         .glob("*.hpp"))
        for h in OPTIONAL_HELPERS:
            used = re.search(rf"\b{h}\s*[<(]", srcs) is not None
            present = re.search(rf"inline void {h}\(", ev) is not None
            tested += 1
            if used != present:
                failures += 1
                print(f"  [FAIL] evaluator helper {h}: used {used}, "
                      f"present {present}")
        bld = _build(pkg)
        dll = ctypes.CDLL(str(bld / "libxckernel.so"))
        t, f = check_index(dll, man)
        tested, failures = tested + t, failures + f
    return tested, failures


def build_and_validate(families=("lda", "gga", "hmgga"), max_order=3,
                       nbf=4, ng=50, seed=3, verbose=False):
    with tempfile.TemporaryDirectory() as td:
        pkg = Path(td) / "libxck"
        build_catalog(str(pkg), families, max_order, verbose=verbose,
                      backend="c")
        bld = _build(pkg)
        man = json.loads((pkg / "manifest.json").read_text())

        # every response kernel through the self-describing runtime layer
        # (collocation and field towers, operands by name), against the
        # NumPy backend fed the same towers
        from ..emitters.tower import ncomp
        from ..runtime import Library, _NumpyKernel
        rt = Library(str(bld / "libxckernel.so"))
        rng = np.random.default_rng(seed)
        tested = failures = 0
        for k in man["kernels"]:
            # skip cross-backend pointer entries (e.g. the GIAO notes,
            # which carry no "abi"/"order") and the order-0 energy kernels
            if "abi" not in k or k.get("order", 0) == 0:
                continue
            # the Fock-diagonal and nuclear-gradient ABIs are checked
            # against physics by catalog_c_validate
            if k.get("kind", "matrix") != "matrix":
                continue
            name = k["name"]
            chi = rng.standard_normal((ncomp(rt.order(name)), nbf, ng))
            ops = {n: rng.standard_normal(ng) for n in rt.scal_names(name)}
            ops["w"] = np.abs(ops["w"]) + 0.1
            out = rt(name, chi=chi, **ops)
            ref = _NumpyKernel(name)(chi=chi, **ops)
            ok = np.allclose(out, ref, atol=1e-12, rtol=1e-12)
            tested += 1
            if not ok:
                failures += 1
                print(f"  [FAIL] {name}")

        # machine-readable kinds and the dispatch index
        t, f = check_index(rt._dll, man)
        tested, failures = tested + t, failures + f

        # datatype templating: instantiate a kernel at long double through
        # the header-only path and compare against the double ABI result
        prog = pkg / "ld_test.cpp"
        prog.write_text(r'''
#include "xckernel/kernels/xck_gga_r_o2.hpp"
#include <cstdio>
#include <vector>
extern "C" int xck_gga_r_o2(int64_t, int64_t, const double*,
                            const double* const*, double*);
extern "C" const int xck_gga_r_o2_n_scal;
extern "C" const int xck_gga_r_o2_n_fields;
extern "C" const int xck_gga_r_o2_chi_order;
int main() {
    const int64_t nbf = 3, ng = 20;
    const int ns = xck_gga_r_o2_n_scal, nfld = xck_gga_r_o2_n_fields;
    const int o = xck_gga_r_o2_chi_order;
    const int64_t nc = (o + 1) * (o + 2) * (o + 3) / 6;
    std::vector<double> chi(nc*nbf*ng);
    std::vector<std::vector<double>> scal(ns, std::vector<double>(ng));
    unsigned s = 12345;
    auto rnd = [&]() { s = 1664525u*s + 1013904223u;
                       return (double)(s % 1000) / 500.0 - 1.0; };
    for (auto& x : chi) x = rnd();
    for (auto& v : scal) for (auto& x : v) x = rnd();
    std::vector<const double*> sp(ns);
    for (int i = 0; i < ns; i++) sp[i] = scal[i].data();
    std::vector<double> outd(nbf*nbf, 0.0);
    xck_gga_r_o2(ng, nbf, chi.data(), sp.data(), outd.data());
    // T = long double for collocation and fields, Txc = double for Libxc
    std::vector<long double> chiL(chi.begin(), chi.end()), outL(nbf*nbf, 0.0L);
    std::vector<std::vector<long double>> scalL(nfld);
    std::vector<const long double*> fldL(nfld);
    for (int i = 0; i < nfld; i++) {
        scalL[i].assign(scal[i].begin(), scal[i].end());
        fldL[i] = scalL[i].data();
    }
    std::vector<const double*> xcp(sp.begin() + nfld, sp.end());
    xckernel::xck_gga_r_o2_t<long double, double>(
        ng, nbf, chiL.data(), fldL.data(), xcp.data(), outL.data());
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
    t, f = validate_selection()
    tested, failures = tested + t, failures + f
    status = "OK " if failures == 0 else "FAIL"
    print(f"[{status}] libxckernel package: {tested} checks (kernels built "
          f"via CMake vs NumPy, kind index, kind selection), "
          f"{failures} failures")
