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

#: entries left out of a default build, generated only on request
#: (include_heavy=True, --include-heavy, -DXCKERNEL_INCLUDE_HEAVY=ON).
#: The order-4 open-shell hmgga contractions dwarf everything else:
#: xck_hmgga_ua_o4 takes ~45 min and ~16 GB to generate -- more memory
#: than a standard CI runner has -- and the spin-adapted st_o4 entries
#: rebuild that same polynomial before substituting the parities (~10 min
#: each before collapse, 3.5e6 monomials). None has a known consumer.
HEAVY = frozenset({"xck_hmgga_ua_o4", "xck_hmgga_ub_o4",
                   "xck_hmgga_st_o4_ppp", "xck_hmgga_st_o4_ppm",
                   "xck_hmgga_st_o4_pmm", "xck_hmgga_st_o4_mmm"})


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
    #: one-free-index and pointwise kernels (C backend): 'diag' (the
    #: order-1 diagonal F_uu), 'g1' (nuclear-gradient basis-class rows),
    #: 'gg' (nuclear-gradient grid class, per point); '' for the rest
    kind: str = ""

    @property
    def name(self) -> str:
        if self.giao:
            return f"xck_{self.family}_{self.spin}_giao"
        if self.kind == "diag":
            return f"xck_{self.family}_{self.spin}_o1_diag"
        if self.kind in ("g1", "gg", "f1", "fg"):
            return f"xck_{self.family}_{self.spin}_{self.kind}"
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
        if self.kind:
            sd = {"r": "unpolarized", "ua": "unrestricted (alpha channel)",
                  "ub": "unrestricted (beta channel)",
                  "u": "unrestricted (both channels)"}[self.spin]
            what = {
                "diag": "diagonal F_uu of the XC Fock matrix",
                "g1": ("XC nuclear gradient, basis class: per-function rows "
                       "g_{d,u}, summed over the functions on atom A to "
                       "+dE/dX_{A,d}"),
                "gg": ("XC nuclear gradient, grid class: per-point "
                       "w * d_d e(r_g), summed over the points of atom A"),
                "f1": ("nuclear derivative of the XC Fock matrix "
                       "dF/dX_{A,d}, basis class, one atom per call"),
                "fg": ("nuclear derivative of the XC Fock matrix "
                       "dF/dX_{A,d}, grid class, one atom per call"),
            }[self.kind]
            return f"{what}, {self.family}, {sd}"
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


def gradient_families() -> Tuple[str, ...]:
    from .engine.gradient import GRADIENT_FAMILIES
    return tuple(f for f in FAMILIES if f in GRADIENT_FAMILIES)


def _one_index_entries(fam: str, max_order: int = 2) -> Iterator[CatalogEntry]:
    """The C-backend Fock-diagonal and nuclear-derivative entries (dF/dX
    contains the linear-response contraction: max_order >= 2)."""
    from .engine.geofock import FOCK_DERIV_FAMILIES
    polarized = fam not in UNPOLARIZED_ONLY
    spins = ("r", "ua", "ub") if polarized else ("r",)
    for spin in spins:
        yield CatalogEntry(fam, spin, 1, kind="diag")
    if fam in gradient_families():
        for spin in spins:
            yield CatalogEntry(fam, spin, 1, kind="g1")
        for spin in (("r", "u") if polarized else ("r",)):
            yield CatalogEntry(fam, spin, 1, kind="gg")
    if fam in FOCK_DERIV_FAMILIES and max_order >= 2:
        for kind in ("f1", "fg"):
            for spin in spins:
                yield CatalogEntry(fam, spin, 2, kind=kind)


def entries(families=FAMILIES, max_order: int = 4,
            include_heavy: bool = False) -> Iterator[CatalogEntry]:
    """Enumerate the catalog; HEAVY entries only with include_heavy."""
    for e in _all_entries(families, max_order):
        if include_heavy or e.name not in HEAVY:
            yield e


def _all_entries(families, max_order: int) -> Iterator[CatalogEntry]:
    for fam in families:
        fmax = min(max_order, FAMILY_MAX_ORDER.get(fam, max_order))
        yield CatalogEntry(fam, "r", 0)                      # exc
        for o in range(1, fmax + 1):                         # unpolarized
            yield CatalogEntry(fam, "r", o, batch=(o >= 2))
        if fmax >= 1:                       # Fock diagonal, gradient, dF/dX
            yield from _one_index_entries(fam, fmax)
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
    if re.match(r"^(grad_tau|grad_lapl_rho)(_[ab])?$", name):
        return {"shape": "(3, ng)", "kind": "gs_field_gradient"}
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


def _kind_kernels(e: CatalogEntry):
    """(row blocks, collapsed form carrying the operand order, computed
    per-point fields or None) of a diag/g1/gg/f1/fg entry."""
    import sympy as sp

    from .emitters.cbackend import collapse_pointwise
    from .emitters.codegen import collapse
    from .engine.kernel import KernelIntegrand
    if e.giao:
        from .engine.london import london_fock, london_fock_spin
        from .inputs.functional import Functional
        kis = [london_fock(e.family, s) if e.spin == "r"
               else london_fock_spin(e.family, e.spin[1], s)
               for s in range(3)]
        whole = KernelIntegrand(functional=Functional.of_family(e.family),
                                index_pairs=[("u", "v")],
                                expr=sp.Add(*[k.expr for k in kis]))
        return [collapse(k) for k in kis], collapse(whole), None
    if e.kind == "diag":
        ck = collapse(_integrand_for(CatalogEntry(e.family, e.spin, 1)))
        return [ck], ck, None
    if e.kind in ("f1", "fg"):
        from .engine import geofock
        from .emitters.cbackend import scal_order
        make = (geofock.fock_basis_class if e.kind == "f1"
                else geofock.fock_grid_class)
        kis = [make(e.family, e.spin, d) for d in range(3)]
        whole = collapse(KernelIntegrand(
            functional=kis[0].functional, index_pairs=[("u", "v")],
            expr=sp.Add(*[k.expr for k in kis])))
        computed = None
        if e.kind == "f1":
            from .engine.gradient import GENERAL_DM_FAMILIES
            general = e.family in GENERAL_DM_FAMILIES
            computed = {n: [geofock.perturbed_field_terms(n, d, general)
                            for d in range(3)]
                        for n in scal_order(whole) if "_p1" in n}
        return [collapse(k) for k in kis], whole, computed
    if e.kind == "g1":
        from .engine.gradient import energy_gradient_rows
        kis = [energy_gradient_rows(e.family, e.spin, d) for d in range(3)]
        whole = KernelIntegrand(functional=kis[0].functional,
                                index_pairs=[("u", "v")],
                                expr=sp.Add(*[k.expr for k in kis]))
        return [collapse(k) for k in kis], collapse(whole), None
    from .engine.gradient import energy_grid_gradient
    from .inputs.functional import Functional
    func = Functional.of_family(e.family)
    exprs = [energy_grid_gradient(e.family, e.spin, d) for d in range(3)]
    return ([collapse_pointwise(x, func) for x in exprs],
            collapse_pointwise(sp.Add(*exprs), func), None)


def _tower_params(layout, kind: str) -> List[Dict]:
    """Manifest operands of a kernel on the derivative-tower interface."""
    import re

    from .emitters.cbackend import ABI_KINDS
    from .emitters.tower import ncomp
    params: List[Dict] = []
    for a in ABI_KINDS[kind][1]:
        n = layout.orders.get(a, 0)
        meta = {"name": a, "shape": "(ncomp, nbf, ng)", "order": n,
                "ncomp": ncomp(n), "kind": "collocation_tower"}
        if a.startswith(("Dchi", "DTchi")):
            meta["kind"] = "dm_contracted_collocation_tower"
            mat = {"Dchi": "D", "DTchi": "D^T", "Dchi_a": "D^a",
                   "Dchi_b": "D^b", "DTchi_a": "(D^a)^T",
                   "DTchi_b": "(D^b)^T"}[a]
            meta["definition"] = f"{a}[k,u,g] = sum_v ({mat})[u,v] chi[k,v,g]"
        params.append(meta)
    for f in layout.fields:
        kind_f = ("grid_weights" if f == "w" else
                  "pert_field" if re.search(r"_p\d+", f) else "gs_field")
        params.append({"name": f, "shape": "(ng,)", "kind": kind_f})
    for x in layout.libxc:
        params.append({"name": x, "shape": "(ng,)", "kind": "libxc_deriv"})
    return params


def _tower_manifest(m: Dict, blocks, scal_ck, kind: str,
                    computed=None) -> Dict:
    """Overwrite a manifest entry's operands with the C ABI's."""
    from .emitters.cbackend import MASKED_KINDS, kernel_layout
    layout = kernel_layout(blocks, scal_ck, kind, computed)
    m["params"] = _tower_params(layout, kind)
    if kind in MASKED_KINDS:
        m["params"].insert(len([p for p in m["params"]
                                if p["kind"].endswith("tower")]),
                           {"name": "atom_mask", "shape": "(nbf,)",
                            "kind": "atom_mask", "dtype": "int8",
                            "definition": "1 for the functions on the "
                                          "displaced atom, 0 elsewhere"})
    m["scal_names"] = layout.scal_names
    m["tower_orders"] = dict(layout.orders)
    m["formed_in_kernel"] = layout.definitions()
    m["libxc"]["derivative_arrays"] = list(scal_ck.libxc_args)
    return m


def abi_kind(e: CatalogEntry) -> str:
    """The C ABI of an entry: its kind, except that the gradient rows of a
    general-density-matrix family also take DTchi ('g1c')."""
    from .engine.gradient import GENERAL_DM_FAMILIES
    if e.giao:
        return "giao"
    if e.kind == "g1" and e.family in GENERAL_DM_FAMILIES:
        return "g1c"
    if e.kind == "f1":
        c = "c" if e.family in GENERAL_DM_FAMILIES else ""
        return f"f1{c}" if e.spin == "r" else f"f1{c}u"
    return e.kind or "matrix"


def _emit_kind(e: CatalogEntry):
    """(hpp, cpp, manifest entry) of a diag/g1/gg entry."""
    from .emitters.cbackend import ABI_KINDS, emit_tower_cpp, emit_tower_hpp
    blocks, scal_ck, computed = _kind_kernels(e)
    abi = abi_kind(e)
    hpp = emit_tower_hpp(blocks, scal_ck, e.name, abi, computed)
    cpp = emit_tower_cpp(blocks, scal_ck, e.name, abi, computed)
    m: Dict = {
        "name": e.name, "description": e.description,
        "family": e.family, "spin": e.spin, "order": e.order, "batch": False,
        "kind": e.kind or abi, "abi_kind": abi,
        "output_shape": ABI_KINDS[abi][2],
        "abi": "xckernel.h",
        "generator": "machine-generated by xckernel; do not edit",
        "copyright": "Copyright (c) 2026 Susi Lehtola",
        "ownership": OWNERSHIP,
        "libxc": {
            "input_variables": FAMILY_VARS[e.family],
            "spin_mode": "unpolarized" if e.spin == "r" else "polarized",
            "max_derivative_order": e.order,
        },
    }
    _tower_manifest(m, blocks, scal_ck, abi, computed)
    if e.kind == "diag":
        m["convention"] = "out[u] += F_uu, the diagonal of " + \
            CatalogEntry(e.family, e.spin, 1).name
    elif e.kind == "g1":
        m["convention"] = (
            "out[d*nbf + u] += g_{d,u}; dE/dX_{A,d} (basis class) = sum over "
            "u on atom A of g_{d,u}. SIGN: the output is +dE/dX -- the "
            "-d/dr of d chi/dX is folded into the kernel, so chi is the "
            "plain collocation tower. Dchi = D chi with D symmetric.")
        if abi == "g1c":
            m["convention"] = m["convention"].replace(
                "Dchi = D chi with D symmetric.",
                "The density matrix M is general (complex orbitals in a real "
                "basis: symmetric part Re P, antisymmetric part Im P, which "
                "carries the paramagnetic current): Dchi = M chi and "
                "DTchi = M^T chi.")
        if e.spin != "r":
            m["convention"] += (
                f" Unrestricted: Dchi is formed with the {e.spin[1]}-channel "
                "spin density matrix and the output is that channel's "
                "contribution; the gradient is the sum of the ua and ub calls.")
    elif e.giao:
        m["convention"] = (
            "out[(s*nbf + u)*nbf + v] += K^s_uv, s = x, y, z: the explicit "
            "field derivative of the XC Fock matrix with London orbitals "
            "chi^B = exp[-(i/2c) B.(R_u x r)] chi, dF/dB_s = (i/2c) K^s at a "
            "real reference (K^s antisymmetric). bf_centers[a*nbf + u] is "
            "the a coordinate of function u's center R_u; rgx/rgy/rgz are "
            "the grid coordinates, in the same origin.")
    elif e.kind == "f1":
        m["convention"] = (
            "out[(d*nbf + u)*nbf + v] += dF_uv/dX_{A,d}, basis class (fixed "
            "grid), for the atom whose functions atom_mask flags; one call "
            "per atom. chi is the plain collocation tower (the -d/dr of "
            "d chi/dX is folded in); the kernel forms the perturbed fields "
            "of the displacement from chi, the D chi towers and the mask. "
            + ("D symmetric." if abi in ("f1", "f1u") else
               "The density matrix M is general (complex orbitals in a real "
               "basis): Dchi = M chi and DTchi = M^T chi.")
            + ("" if e.spin == "r" else
                              " Unrestricted: dF^s/dX responds to both "
                              "channels, so both Dchi_a and Dchi_b are "
                              "passed."))
    elif e.kind == "fg":
        m["convention"] = (
            "out[(d*nbf + u)*nbf + v] += dF_uv/dX_{A,d}, grid class: call "
            "with w := w M^A (the weights of atom A's points, zero "
            "elsewhere); one call per atom. The weight class is the o1 "
            "kernel with w := dw/dX.")
    else:
        m["convention"] = (
            "out[d*ng + g] += w_g d_d e(r_g), e the XC energy density per "
            "volume; the grid class of dE/dX_{A,d} is the sum over the points "
            "whose parent atom is A (call with the plain weights; no atom "
            "masking in the kernel).")
    m["gradient_classes"] = _GRADIENT_CLASSES
    return hpp, cpp, m


#: the dispatch table of the XC nuclear gradient
_GRADIENT_CLASSES = {
    "basis": "xck_<family>_<r|ua|ub>_g1 (per-function rows, summed per atom)",
    "grid": "xck_<family>_<r|u>_gg (per-point, summed over each atom's points)",
    "weight": ("no new kernel: sum_g (dw_g/dX_{A,d}) e(r_g), e.g. "
               "xck_<family>_r_o0 called with w := dw/dX (rho the total "
               "density, zk the Libxc energy per particle); the host owns "
               "the partition-weight derivative"),
    "invariance": ("sum over atoms of basis + grid + weight classes is zero "
                   "for each direction"),
    "fock_derivative": ("dF/dX_{A,d} = xck_<family>_<spin>_f1 (atom mask) + "
                        "xck_<family>_<spin>_fg (w := w M^A) + "
                        "xck_<family>_<spin>_o1 (w := dw/dX); summed over "
                        "atoms the three are zero"),
}


VERSION = "0.4.0"


def build_catalog(outdir: str, families=FAMILIES, max_order: int = 4,
                  verbose: bool = True, backend: str = "numpy",
                  include_heavy: bool = False) -> Dict:
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
                      "gradient_classes": _GRADIENT_CLASSES,
                      "kernels": []}

    if backend == "numpy":
        (out / "kernels").mkdir(parents=True, exist_ok=True)
        for e in entries(families, max_order, include_heavy):
            t0 = time.time()
            if e.kind:
                manifest["kernels"].append({
                    "name": e.name, "description": e.description,
                    "backends": ["c"],
                    "note": "C backend only (one-free-index/pointwise ABI)"})
                continue
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
        from .emitters.cbackend import (_CONFIG_H_IN, _EVALUATOR_HPP,
                               emit_cmake, emit_exc_cpp,
                               emit_exc_hpp, emit_f03, emit_header,
                               emit_kernel_cpp, emit_kernel_hpp)
        from .emitters.codegen import collapse, generate_collapsed
        (out / "src").mkdir(parents=True, exist_ok=True)
        (out / "include" / "xckernel" / "kernels").mkdir(parents=True,
                                                         exist_ok=True)
        (out / "fortran").mkdir(exist_ok=True)
        (out / "include" / "xckernel" / "evaluator.hpp").write_text(
            _EVALUATOR_HPP)
        (out / "include" / "xckernel" / "config.h.in").write_text(
            _CONFIG_H_IN)
        names: List = []
        for e in entries(families, max_order, include_heavy):
            t0 = time.time()
            if e.kind or e.giao:
                hpp, cpp, m = _emit_kind(e)
                (out / "include" / "xckernel" / "kernels"
                 / f"{e.name}.hpp").write_text(hpp)
                (out / "src" / f"{e.name}.cpp").write_text(cpp)
                manifest["kernels"].append(m)
                names.append((e.name, e.order, abi_kind(e)))
                if verbose:
                    print(f"  {e.name:28s} {time.time()-t0:7.1f}s  "
                          f"{e.kind}", flush=True)
                continue
            if e.order == 0:
                (out / "include" / "xckernel" / "kernels"
                 / f"{e.name}.hpp").write_text(emit_exc_hpp(e.name))
                (out / "src" / f"{e.name}.cpp").write_text(
                    emit_exc_cpp(e.name))
                m = manifest_for(e, None)
                m["abi"] = "xckernel.h"
                m["abi_kind"] = "exc"
                m["output_shape"] = "()"
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
            m["kind"] = "matrix"
            m["abi_kind"] = "matrix"
            m["output_shape"] = "(nbf, nbf)"
            _tower_manifest(m, [ck], ck, "matrix")
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
        from .emitters.cbackend import emit_index_cpp
        (out / "src" / "xckernel_index.cpp").write_text(emit_index_cpp(names))
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
    p.add_argument("--include-heavy", action="store_true",
                   help="also generate the HEAVY entries "
                        f"({', '.join(sorted(HEAVY))})")
    a = p.parse_args(argv)
    families = a.families.split(",") if a.families else FAMILIES
    unknown = [f for f in families if f not in FAMILIES]
    if unknown:
        p.error(f"unknown families {unknown}; known: {', '.join(FAMILIES)}")
    m = build_catalog(a.outdir, families, a.max_order, backend=a.backend,
                      include_heavy=a.include_heavy)
    print(f"{len(m['kernels'])} kernels -> {a.outdir}/ [{a.backend}]")


if __name__ == "__main__":
    main()
