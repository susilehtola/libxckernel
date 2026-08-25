"""Build a libxckernel release artifact: the complete generated C source
package, distributable and buildable with only CMake + a C compiler
(no Python/SymPy/NumPy on the consumer side -- the Libxc distribution model:
generators are developer tools, releases ship generated source).

    python -m xckernel.emitters.release [outdir] [version]

produces outdir/libxckernel-<version>/ and libxckernel-<version>.tar.gz.
"""

from __future__ import annotations

import tarfile
from pathlib import Path

from .. import catalog

README = """\
libxckernel {version}
====================

Generated exchange-correlation kernel contractions in LCAO basis sets --
an automatic-differentiation backend for Libxc. This package contains
GENERATED C++17 SOURCE: kernels are header-only templates over the
floating-point type (double, long double, __float128, ...), with
extern "C" double instantiations providing the stable C ABI. Do not
edit; regenerate with the xckernel Python package.
Building requires only CMake >= 3.16 and a C++17 compiler
(plus a Fortran compiler for the optional xckernel_f03 module):

    cmake -B build -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build build
    cmake --install build

Interface: include/xckernel.h (C; ABI, operand ordering, and the
linear-mixing contract for user-mixed functional-derivative arrays are
documented there), fortran/xckernel_f03.f90 (ISO_C_BINDING module), and
manifest.json (machine-readable kernel descriptions). Kernels contain
exchange-correlation terms only; Coulomb and exact exchange are the
host's. License: BSD-3-Clause.
"""


def build_release(outdir: str = "dist", version: str = catalog.VERSION,
                  families=catalog.FAMILIES, max_order: int = 4,
                  verbose: bool = True) -> Path:
    out = Path(outdir)
    pkg = out / f"libxckernel-{version}"
    catalog.build_catalog(str(pkg), families, max_order, verbose=verbose,
                          backend="c")
    (pkg / "README").write_text(README.format(version=version))
    # ship the license alongside the generated code
    lic = Path(__file__).resolve().parent.parent / "LICENSE"
    if lic.exists():
        (pkg / "LICENSE").write_text(lic.read_text())
    tarball = out / f"libxckernel-{version}.tar.gz"
    with tarfile.open(tarball, "w:gz") as tf:
        tf.add(pkg, arcname=pkg.name)
    return tarball


def main(argv=None):
    """Assemble the release tarball.

    Positional and optional as before, but parsed rather than indexed
    off sys.argv: read by position, ``--help`` was taken as the output
    directory and a distribution was written into a folder of that name.
    """
    import argparse
    p = argparse.ArgumentParser(
        prog="python -m xckernel.emitters.release",
        description="Assemble the self-contained C source package.")
    p.add_argument("outdir", nargs="?", default="dist",
                   help="output directory (default: dist)")
    p.add_argument("version", nargs="?", default=catalog.VERSION,
                   help=f"release version (default: {catalog.VERSION})")
    a = p.parse_args(argv)
    tb = build_release(a.outdir, a.version)
    print(f"release: {tb} ({tb.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
