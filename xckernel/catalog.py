"""The kernel catalog: enumerate, generate, and manifest every XC matrix
element up to a given derivative order (interfacing-plan phase 2).

The catalog spans the finite product space

    quantity   exc (order 0), Fock (order 1), response contractions (2..N)
    family     lda, gga, mgga_tau (no Laplacian), mgga_lapl (no tau), mgga (full)
    spin case  'r'  unpolarized (restricted)
               'ua'/'ub'  unrestricted, alpha/beta output channel
               'st' closed-shell spin-adapted, one parity (+1 singlet /
                    -1 triplet) per perturbation (multisets: perturbation
                    slots are relabelable)
    batch      response kernels carry a leading batch axis over perturbations

Every entry yields (a) generated source (NumPy backend today; compiled
backends per the interfacing plan) in the pattern-collapsed form, and (b) a
machine-readable manifest: parameters in call order with shapes and kinds,
the Libxc derivative arrays needed (by Libxc name; spin components flattened
as '<array>_<comp>' in Libxc's packing), the Libxc evaluation requirements,
and the term-ownership declaration (XC only -- Coulomb/HF/RSH exchange is
host-owned).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

FAMILIES = ("lda", "gga", "mgga_tau", "mgga_lapl", "mgga", "cmgga_tau", "hmgga")

#: families restricted to the unpolarized case (none since the spin
#: resolution of the current-density and density-Hessian families; the spin
#: components pack in analogy to the Libxc variables: jp and the spin-pure
#: eta are density-like, with a and b channels).
UNPOLARIZED_ONLY: set = set()

#: Libxc input variables per family.
FAMILY_VARS = {
    "lda": ["rho"],
    "gga": ["rho", "sigma"],
    "mgga_tau": ["rho", "sigma", "tau"],
    "mgga_lapl": ["rho", "sigma", "lapl"],
    "mgga": ["rho", "sigma", "lapl", "tau"],
    # current-density DFT: a tau-meta-GGA evaluated at the gauge-corrected
    # tau~ = tau - j_p^2/(2 rho); host supplies jp (3,ng) and inv_rho (ng,)
    "cmgga_tau": ["rho", "sigma", "tau"],
    # local-hybrid calibration-function set (CF concept: Arbuznikov & Kaupp
    # 2014; the density-Hessian variable: Maier et al. 2016, Eqs. 22-23;
    # eta notation: Schattenberg & Kaupp 2021): the
    # meta-GGA variables plus eta = grad rho . (grad grad rho) . grad rho.
    # eta is beyond Libxc; its derivative arrays (veta, v2rhoeta, ...) follow
    # the same naming scheme and are supplied by the host's functional
    # implementation.  Host supplies hess_rho (6,ng) and hess_chi (6,nbf,ng).
    "hmgga": ["rho", "sigma", "lapl", "tau", "eta"],
}

OWNERSHIP = ("xc-only: Coulomb, HF and range-separated exchange are "
             "host-owned; kernels contain exclusively density-functional "
             "exchange-correlation terms")

#: per-family cap on the shipped derivative order.  None at present: every
#: family is generated through ``max_order``.  The largest entry is the
#: spin-resolved order-4 hmgga contraction (6.6e6 monomials, ~0.5 GB of
#: expression source), which the table-driven C backend emits as data.
FAMILY_MAX_ORDER: Dict[str, int] = {}

#: families with GIAO magnetic-field derivative kernels (London orbitals).
GIAO_FAMILIES = ("lda", "gga", "mgga_tau", "mgga_lapl", "mgga")

#: additional cap for the SPIN-RESOLVED cases.  None at present.
FAMILY_SPIN_MAX_ORDER: Dict[str, int] = {}


@dataclass
class CatalogEntry:
    family: str
    spin: str                      # 'r' | 'ua' | 'ub' | 'st'
    order: int                     # 0=exc, 1=Fock, >=2 response contraction
    parities: Tuple[int, ...] = ()   # 'st' only, one per perturbation
    batch: bool = False
    #: explicit GIAO magnetic-field derivative of the Fock matrix
    #: (London orbitals; F^{B_s} = (i/2c) K_s at a real reference)
    giao: bool = False

    @property
    def name(self) -> str:
        if self.giao:
            return f"xck_{self.family}_{self.spin}_giao"
        parts = ["xck", self.family, self.spin, f"o{self.order}"]
        if self.parities:
            parts.append("".join("p" if p > 0 else "m" for p in self.parities))
        return "_".join(parts)

    @property
    def description(self) -> str:
        if self.giao:
            sd = {"r": "unpolarized",
                  "ua": "unrestricted (alpha channel)",
                  "ub": "unrestricted (beta channel)"}[self.spin]
            return (f"explicit GIAO magnetic-field derivative of the XC "
                    f"Fock matrix, {self.family}, {sd}; "
                    f"F^(B_s) = (i/2c) K_s at a real reference")
        q = {0: "XC energy", 1: "XC Fock matrix"}.get(
            self.order, f"order-{self.order} XC response contraction")
        s = {"r": "unpolarized", "ua": "unrestricted (alpha channel)",
             "ub": "unrestricted (beta channel)",
             "st": "closed-shell spin-adapted"}[self.spin]
        p = ""
        if self.parities:
            p = " with perturbation parities (" + ", ".join(
                "singlet" if x > 0 else "triplet" for x in self.parities) + ")"
        return f"{q}, {self.family}, {s}{p}"


def entries(families=FAMILIES, max_order: int = 4) -> Iterator[CatalogEntry]:
    """Enumerate the catalog."""
    for fam in families:
        fmax = min(max_order, FAMILY_MAX_ORDER.get(fam, max_order))
        yield CatalogEntry(fam, "r", 0)                      # exc
        for o in range(1, fmax + 1):                         # unpolarized
            yield CatalogEntry(fam, "r", o, batch=(o >= 2))
        if fam in UNPOLARIZED_ONLY:
            continue
        smax = min(fmax, FAMILY_SPIN_MAX_ORDER.get(fam, fmax))
        for spin in ("ua", "ub"):                            # unrestricted
            for o in range(1, smax + 1):
                yield CatalogEntry(fam, spin, o, batch=(o >= 2))
        for o in range(2, smax + 1):                         # spin-adapted
            for pars in combinations_with_replacement((+1, -1), o - 1):
                yield CatalogEntry(fam, "st", o, parities=pars, batch=True)
        if fam in GIAO_FAMILIES:                             # GIAO B-derivative
            yield CatalogEntry(fam, "r", 1, giao=True)
            for spin in ("ua", "ub"):
                yield CatalogEntry(fam, spin, 1, giao=True)


# --- source generation -------------------------------------------------------

_EXC_SOURCE = """\
def {name}(w, rho, zk):
    # XC energy: Exc = sum_g w_g rho_g zk_g   (zk = Libxc energy per particle)
    import numpy as np
    return float(np.sum(w * rho * zk))
"""


def build_entry(e: CatalogEntry):
    """Generate the entry's source. Returns (source, GeneratedFunction|None)."""
    from .emitters.codegen import generate_collapsed
    if e.order == 0 and not e.giao:
        return _EXC_SOURCE.format(name=e.name), None

    if e.giao:
        from .engine.london import london_fock, london_fock_spin
        parts, gen0 = [], None
        for si, ax in enumerate("xyz"):
            if e.spin == "r":
                ki = london_fock(e.family, si)
            else:
                ki = london_fock_spin(e.family, e.spin[1], si)
            gen = generate_collapsed(ki, f"{e.name}_{ax}")
            parts.append(gen.source)
            gen0 = gen0 or gen
        sig = gen0.source.split("(", 1)[1].split(")", 1)[0]
        parts.append(
            f"def {e.name}({sig}):\n"
            f"    # stacked (3, nao, nao); F^(B_s) = (i/2c) * result[s]\n"
            f"    import numpy as np\n"
            f"    return np.stack([{e.name}_x({sig}), {e.name}_y({sig}), "
            f"{e.name}_z({sig})])\n")
        return "\n\n".join(parts), gen0

    if e.spin == "r":
        if e.order == 1:
            from .engine.kernel import fock
            ki = fock(e.family)
        else:
            from .engine.response import response_fock
            ki = response_fock(e.family, e.order)
    elif e.spin in ("ua", "ub"):
        s = e.spin[1]
        if e.order == 1:
            from .engine.spin_kernel import fock_spin
            ki = fock_spin(e.family, s)
        else:
            from .engine.spin_kernel import response_fock_spin
            ki = response_fock_spin(e.family, s, e.order)
    else:  # 'st'
        from .engine.spin_kernel import response_fock_st
        ki = response_fock_st(e.family, e.order, e.parities)

    gen = generate_collapsed(ki, e.name, batch=e.batch)
    return gen.source, gen


# --- manifests ---------------------------------------------------------------

def _param_meta(name: str, batch: bool) -> Dict[str, str]:
    """Shape and kind for a generated-function parameter."""
    import re
    if name == "w":
        return {"shape": "(ng,)", "kind": "grid_weights"}
    if name == "chi":
        return {"shape": "(nbf, ng)", "kind": "collocation"}
    if name == "dchi":
        return {"shape": "(3, nbf, ng)", "kind": "collocation_gradient"}
    if name == "lapl_chi":
        return {"shape": "(nbf, ng)", "kind": "collocation_laplacian"}
    if name == "hess_chi":
        return {"shape": "(6, nbf, ng)", "kind": "collocation_hessian"}
    if re.match(r"^(grad_rho|jp)(_[ab])?$", name):
        return {"shape": "(3, ng)", "kind": "gs_field"}
    if name == "rg":
        return {"shape": "(3, ng)", "kind": "grid_coordinates"}
    if name == "Rchi":
        return {"shape": "(3, nbf, ng)", "kind": "center_scaled_collocation"}
    if name == "Rdchi":
        return {"shape": "(3, 3, nbf, ng)",
                "kind": "center_scaled_collocation_gradient"}
    if name == "Rlapl_chi":
        return {"shape": "(3, nbf, ng)",
                "kind": "center_scaled_collocation_laplacian"}
    if re.match(r"^hess_rho(_[ab])?$", name):
        return {"shape": "(6, ng)", "kind": "gs_field"}
    if re.match(r"^inv_rho(_[ab])?$", name):
        return {"shape": "(ng,)", "kind": "gs_field"}
    if re.match(r"^(grad_rho|jp)(_[ab])?_p\d+$", name):
        return {"shape": "(nx, 3, ng)" if batch else "(3, ng)",
                "kind": "pert_field"}
    if re.match(r"^hess_rho(_[ab])?_p\d+$", name):
        return {"shape": "(nx, 6, ng)" if batch else "(6, ng)",
                "kind": "pert_field"}
    if re.match(r"^(rho|lapl_rho|tau)(_[ab])?_p\d+$", name):
        return {"shape": "(nx, ng)" if batch else "(ng,)",
                "kind": "pert_field"}
    # Libxc derivative array (possibly '<array>_<comp>' spin component)
    return {"shape": "(ng,)", "kind": "libxc_deriv"}


def manifest_for(e: CatalogEntry, gen) -> Dict:
    m: Dict = {
        "name": e.name,
        "description": e.description,
        "family": e.family,
        "spin": e.spin,
        "order": e.order,
        "batch": e.batch,
        "generator": "machine-generated by xckernel; do not edit",
        "copyright": "Copyright (c) 2026 Susi Lehtola",
        "ownership": OWNERSHIP,
        "libxc": {
            "input_variables": FAMILY_VARS[e.family],
            "spin_mode": ("unpolarized" if e.spin == "r"
                          else "polarized"),
            "max_derivative_order": max(e.order, 1),
            "component_packing": (
                "unpolarized arrays" if e.spin == "r" else
                "spin components flattened as '<array>_<index>' in Libxc's "
                "canonical packing (e.g. v2rho2_0/1/2 = aa/ab/bb)"),
        },
    }
    if e.parities:
        m["parities"] = list(e.parities)
        m["libxc"]["evaluation_point"] = \
            "polarized kernel at the closed-shell density (rho/2, rho/2)"
    if e.spin == "st":
        m["pert_dm_convention"] = \
            "alpha-channel perturbed DM; D^{X,b} = parity * D^{X,a}"

    if e.order == 0:
        m["params"] = [
            {"name": "w", "shape": "(ng,)", "kind": "grid_weights"},
            {"name": "rho", "shape": "(ng,)", "kind": "gs_field"},
            {"name": "zk", "shape": "(ng,)", "kind": "libxc_energy_density"},
        ]
        return m

    import re
    # GIAO entries generate one function per Cartesian axis plus a stacking
    # wrapper; gen is the x-axis generator, whose signature the wrapper shares.
    m_sig = re.search(rf"def {re.escape(gen.name)}\(([^)]*)\)", gen.source)
    if m_sig is None:
        raise RuntimeError(f"no signature for {gen.name} in generated source")
    sig = m_sig.group(1)
    params = [p.strip() for p in sig.split(",")]
    m["params"] = [{"name": p, **_param_meta(p, e.batch)} for p in params]
    m["libxc"]["derivative_arrays"] = gen.libxc_args
    m["n_patterns"] = gen.n_patterns
    m["n_products"] = gen.n_products
    return m


# --- builder ------------------------------------------------------------------

def _integrand_for(e: CatalogEntry):
    """The symbolic integrand behind a (non-energy) catalog entry."""
    if e.giao:
        raise ValueError("GIAO entries carry three components; "
                         "dispatch them via build_entry")
    if e.spin == "r":
        if e.order == 1:
            from .engine.kernel import fock
            return fock(e.family)
        from .engine.response import response_fock
        return response_fock(e.family, e.order)
    if e.spin in ("ua", "ub"):
        s = e.spin[1]
        if e.order == 1:
            from .engine.spin_kernel import fock_spin
            return fock_spin(e.family, s)
        from .engine.spin_kernel import response_fock_spin
        return response_fock_spin(e.family, s, e.order)
    from .engine.spin_kernel import response_fock_st
    return response_fock_st(e.family, e.order, e.parities)


VERSION = "0.1.0"


def build_catalog(outdir: str, families=FAMILIES, max_order: int = 4,
                  verbose: bool = True, backend: str = "numpy") -> Dict:
    """Generate the full catalog.

    backend='numpy': outdir/kernels/*.py + manifest.json (batched kernels).
    backend='c':     the complete libxckernel source package -- outdir/src/*.c,
                     include/xckernel.h, fortran/xckernel_f03.f90,
                     CMakeLists.txt, manifest.json. Energy (order-0) entries
                     are manifest-only in the C package (the contraction
                     sum(w*rho*zk) is left to the host); response kernels
                     take one perturbation-batch entry per call.
    """
    out = Path(outdir)
    manifest: Dict = {"generator": "xckernel", "backend": backend,
                      "version": VERSION, "max_order": max_order,
                      "kernels": []}

    if backend == "numpy":
        (out / "kernels").mkdir(parents=True, exist_ok=True)
        for e in entries(families, max_order):
            t0 = time.time()
            source, gen = build_entry(e)
            (out / "kernels" / f"{e.name}.py").write_text(
                "import numpy as np\n\n" + source)
            manifest["kernels"].append(manifest_for(e, gen))
            if verbose:
                npat = gen.n_patterns if gen else 0
                nprod = gen.n_products if gen else 0
                print(f"  {e.name:28s} {time.time()-t0:7.1f}s  "
                      f"{npat:3d} patterns {nprod:3d} products",
                      flush=True)
    elif backend == "c":
        from .emitters.cbackend import (_EVALUATOR_HPP, emit_cmake, emit_exc_cpp,
                               emit_exc_hpp, emit_f03, emit_header,
                               emit_kernel_cpp, emit_kernel_hpp)
        from .emitters.codegen import collapse, generate_collapsed
        (out / "src").mkdir(parents=True, exist_ok=True)
        (out / "include" / "xckernel" / "kernels").mkdir(parents=True,
                                                         exist_ok=True)
        (out / "fortran").mkdir(exist_ok=True)
        (out / "include" / "xckernel" / "evaluator.hpp").write_text(
            _EVALUATOR_HPP)
        names: List = []
        for e in entries(families, max_order):
            t0 = time.time()
            if e.giao:
                # the C ABI does not yet carry the center-scaled
                # collocation operands; GIAO kernels ship NumPy-only
                m = manifest_for(e, None) if e.order == 0 else None
                m = {"name": e.name, "description": e.description,
                     "generator": "machine-generated by xckernel; do not edit",
                     "copyright": "Copyright (c) 2026 Susi Lehtola",
                     "backends": ["numpy"],
                     "note": ("not included in the C package; requires the "
                              "center-scaled collocation operands Rchi, "
                              "Rdchi, Rlapl_chi and grid coordinates rg")}
                manifest["kernels"].append(m)
                continue
            if e.order == 0:
                (out / "include" / "xckernel" / "kernels"
                 / f"{e.name}.hpp").write_text(emit_exc_hpp(e.name))
                (out / "src" / f"{e.name}.cpp").write_text(
                    emit_exc_cpp(e.name))
                m = manifest_for(e, None)
                m["abi"] = "xckernel.h"
                manifest["kernels"].append(m)
                names.append((e.name, 0))
                continue
            ki = _integrand_for(e)
            ck = collapse(ki)
            (out / "include" / "xckernel" / "kernels"
             / f"{e.name}.hpp").write_text(emit_kernel_hpp(ck, e.name))
            (out / "src" / f"{e.name}.cpp").write_text(
                emit_kernel_cpp(ck, e.name))
            # manifest from the (unbatched-ABI) generated form
            gen = generate_collapsed(ki, e.name, batch=False)
            m = manifest_for(e, gen)
            m["batch"] = False
            m["abi"] = "xckernel.h"
            manifest["kernels"].append(m)
            names.append((e.name, e.order))
            if verbose:
                print(f"  {e.name:28s} {time.time()-t0:7.1f}s  "
                      f"{len(ck.patterns):3d} patterns", flush=True)
        (out / "include" / "xckernel.h").write_text(
            emit_header(names, VERSION))
        (out / "fortran" / "xckernel_f03.f90").write_text(
            emit_f03(names, VERSION))
        (out / "CMakeLists.txt").write_text(emit_cmake(names, VERSION))
    else:
        raise ValueError(f"unknown backend {backend!r}")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main(argv=None):
    """Build the catalog into a directory.

    The arguments are positional for backwards compatibility, but they
    go through argparse so that ``--help`` prints usage: read straight
    off sys.argv, it was taken as the output directory, and the catalog
    was cheerfully generated into a folder named ``--help``.
    """
    import argparse
    p = argparse.ArgumentParser(
        prog="python -m xckernel.catalog",
        description="Generate the kernel catalog and its manifests.")
    p.add_argument("outdir", nargs="?", default="catalog",
                   help="output directory (default: catalog)")
    p.add_argument("families", nargs="?", default=None,
                   help="comma-separated family names "
                        f"(default: all of {','.join(FAMILIES)})")
    p.add_argument("max_order", nargs="?", type=int, default=4,
                   help="highest derivative order to generate (default: 4)")
    p.add_argument("backend", nargs="?", default="numpy",
                   help="emission backend (default: numpy)")
    a = p.parse_args(argv)
    families = a.families.split(",") if a.families else FAMILIES
    unknown = [f for f in families if f not in FAMILIES]
    if unknown:
        p.error(f"unknown families {unknown}; known: {', '.join(FAMILIES)}")
    m = build_catalog(a.outdir, families, a.max_order, backend=a.backend)
    print(f"{len(m['kernels'])} kernels -> {a.outdir}/ [{a.backend}]")


if __name__ == "__main__":
    main()
