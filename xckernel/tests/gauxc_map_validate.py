"""Cross-validate the noncollinear map against GauXC's production code.

GauXC implements the locally collinear (generalized Kohn-Sham) map by
hand, in ``ReferenceLocalHostWorkDriver::eval_uvvar_gga_gks``, and uses
it for relativistic (X2C) DFT on GPUs.  :mod:`..engine.noncollinear`
derives the same map symbolically and differentiates it to obtain the
potential and -- new -- the noncollinear fxc kernel.

This script compiles a small harness against a built GauXC, feeds its
hand-written kernel randomized noncollinear densities, and compares the
collinear variables it produces (n+-, gamma^{++,+-,--}) against the
generated map evaluated on the SAME fields.  Agreement to roundoff is
the correctness gate for everything derived from the map.

Requires a GauXC build tree; point ``GAUXC_BUILD`` at it (and optionally
``GAUXC_SRC`` at the sources).  Skips cleanly when absent, in the manner
of gpaw_stress_validate.

    GAUXC_BUILD=/path/to/build python -m xckernel.tests.gauxc_map_validate
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import numpy as np
import sympy as sp

from ..engine.noncollinear import F_NABLA, nc_fields, nc_map

_HARNESS = r'''
// Feed GauXC's hand-written noncollinear map randomized densities and
// print both its inputs (the noncollinear fields it derived) and its
// outputs (n+-, gamma), for comparison against the generated map.
#include <gauxc/xc_integrator/local_work_driver.hpp>
#include "xc_integrator/local_work_driver/host/local_host_work_driver.hpp"
#include <cstdio>
#include <vector>
#include <random>

int main() {
  using namespace GauXC;
  auto base = LocalWorkDriverFactory::make_local_work_driver(
      ExecutionSpace::Host, "REFERENCE");
  auto* lwd = dynamic_cast<LocalHostWorkDriver*>(base.get());
  if (!lwd) { printf("cast failed\n"); return 1; }

  const size_t npts = 16, nbe = 4;
  std::mt19937 gen(20260815);
  std::uniform_real_distribution<double> U(0.3, 1.0);

  std::vector<double> B(nbe*npts), dBx(nbe*npts), dBy(nbe*npts),
                      dBz(nbe*npts);
  for (auto& v : B)   v = U(gen);
  for (auto& v : dBx) v = U(gen) - 0.6;
  for (auto& v : dBy) v = U(gen) - 0.6;
  for (auto& v : dBz) v = U(gen) - 0.6;

  std::vector<double> Xs(nbe*npts), Xz(nbe*npts), Xx(nbe*npts), Xy(nbe*npts);
  for (auto& v : Xs) v = U(gen) + 0.8;          // dominant scalar channel
  for (auto& v : Xz) v = 0.12*(U(gen) - 0.6);
  for (auto& v : Xx) v = 0.12*(U(gen) - 0.6);
  for (auto& v : Xy) v = 0.12*(U(gen) - 0.6);

  std::vector<double> den(2*npts), ddx(4*npts), ddy(4*npts), ddz(4*npts),
                      gamma(3*npts), K(3*npts), H(3*npts);
  lwd->eval_uvvar_gga_gks(npts, nbe, B.data(), dBx.data(), dBy.data(),
      dBz.data(), Xs.data(), nbe, Xz.data(), nbe, Xx.data(), nbe,
      Xy.data(), nbe, den.data(), ddx.data(), ddy.data(), ddz.data(),
      gamma.data(), K.data(), H.data(), 1e-12);

  // channel order of the gradient arrays is (s, z, y, x)
  for (size_t i = 0; i < npts; ++i) {
    double rs = 0., rz = 0., ry = 0., rx = 0.;
    for (size_t j = 0; j < nbe; ++j) {
      rs += B[i*nbe+j]*Xs[i*nbe+j];  rz += B[i*nbe+j]*Xz[i*nbe+j];
      ry += B[i*nbe+j]*Xy[i*nbe+j];  rx += B[i*nbe+j]*Xx[i*nbe+j];
    }
    printf("%.17e %.17e %.17e %.17e", rs, rx, ry, rz);
    for (int c = 0; c < 4; ++c) printf(" %.17e", ddx[4*i+c]);
    for (int c = 0; c < 4; ++c) printf(" %.17e", ddy[4*i+c]);
    for (int c = 0; c < 4; ++c) printf(" %.17e", ddz[4*i+c]);
    printf(" %.17e %.17e %.17e %.17e %.17e\n", den[2*i], den[2*i+1],
           gamma[3*i], gamma[3*i+1], gamma[3*i+2]);
  }
  return 0;
}
'''

#: gau2grid entry points the harness never calls; stubbed so the static
#: library links without an external gau2grid.
_STUBS = r'''
#include <stdlib.h>
void gg_collocation(void){ abort(); }
void gg_collocation_deriv1(void){ abort(); }
void gg_collocation_deriv2(void){ abort(); }
void gg_collocation_deriv3(void){ abort(); }
void gg_fast_transpose(void){ abort(); }
'''


def _build_and_run(build, src, d):
    cxx = shutil.which("g++") or shutil.which("clang++")
    if cxx is None:
        return None
    inc = [f"{src}/include", f"{src}/src", f"{build}/include", f"{build}/src",
           f"{build}/_deps/exchcxx-src/include",
           f"{build}/_deps/exchcxx-build/include",
           f"{build}/_deps/integratorxx-src/include"]
    libs = [f"{build}/src/libgauxc.a",
            f"{build}/_deps/exchcxx-build/src/libexchcxx.a"]
    for lib in libs:
        if not os.path.exists(lib):
            return None
    open(f"{d}/h.cpp", "w").write(_HARNESS)
    open(f"{d}/s.c", "w").write(_STUBS)
    subprocess.run(["cc", "-c", f"{d}/s.c", "-o", f"{d}/s.o"], check=True)
    cmd = ([cxx, "-std=c++17", "-O2"] + [f"-I{i}" for i in inc]
           + [f"{d}/h.cpp", "-o", f"{d}/h"] + libs + [f"{d}/s.o"]
           + ["-lxc", "-lopenblas", "-lgomp"])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError("harness failed to build:\n" + r.stderr[-2000:])
    out = subprocess.run([f"{d}/h"], capture_output=True, text=True,
                         check=True).stdout
    return np.array([[float(x) for x in line.split()]
                     for line in out.strip().splitlines()])


def main():
    build = os.environ.get("GAUXC_BUILD")
    src = os.environ.get("GAUXC_SRC", "/home/work/GauXC")
    if not build or not os.path.isdir(build):
        print("[SKIP] gauxc_map_validate: set GAUXC_BUILD to a GauXC build")
        return 0

    with tempfile.TemporaryDirectory() as d:
        D = _build_and_run(build, src, d)
    if D is None:
        print("[SKIP] gauxc_map_validate: no compiler or incomplete build")
        return 0

    rho_s, rho_x, rho_y, rho_z = D[:, 0], D[:, 1], D[:, 2], D[:, 3]
    ddx, ddy, ddz = D[:, 4:8], D[:, 8:12], D[:, 12:16]
    gx = {"n_p": D[:, 16], "n_m": D[:, 17],
          "g_pp": D[:, 18], "g_pm": D[:, 19], "g_mm": D[:, 20]}

    # GauXC packs the gradient channels as (s, z, y, x)
    g = {c: np.stack([ddx[:, k], ddy[:, k], ddz[:, k]])
         for k, c in enumerate(("s", "z", "y", "x"))}
    V = ([rho_s, rho_x, rho_y, rho_z]
         + list(g["s"]) + list(g["x"]) + list(g["y"]) + list(g["z"]))

    tot = sum(m * np.einsum("cg,cg->g", g["s"], g[c])
              for m, c in ((rho_x, "x"), (rho_y, "y"), (rho_z, "z")))
    fnab = np.where(np.sign(tot) == 0, 1.0, np.sign(tot))

    syms = [sp.Symbol(s.name) for s in nc_fields("gga")] + [F_NABLA]
    ours = {k: sp.lambdify(syms, e, "numpy")(*V, fnab)
            for k, e in nc_map("gga").items()}

    failures = 0
    worst = 0.0
    for key, label in (("n_p", "n+"), ("n_m", "n-"), ("g_pp", "gamma++"),
                       ("g_pm", "gamma+-"), ("g_mm", "gamma--")):
        err = float(np.abs(ours[key] - gx[key]).max()
                    / max(float(np.abs(gx[key]).max()), 1e-30))
        worst = max(worst, err)
        ok = err < 1e-13
        failures += not ok
        print(f"  [{'OK' if ok else 'FAIL'}] {label:8s} vs GauXC: {err:.2e}")

    status = "OK " if failures == 0 else "FAIL"
    print(f"[{status}] gauxc_map_validate: 5 checks, {failures} failures "
          f"(worst {worst:.2e})")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
