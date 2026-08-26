"""VeloxChem writer plugin: regenerate DensityGridQuad mode branches.

VeloxChem organizes quadratic (and cubic) response as a generic
derivative-array contraction (XCIntegratorFor*.cpp) fed by mode-specific
pointwise products of perturbed densities (DensityGridQuad.cpp,
DensityGridCubic.cpp).  The product code is fully determined by a small
pairing table: which pairs of first-order densities multiply into which
output slot, with what prefactor, and whether the densities are complex.
The hand-written files expand this table times the product template times
the real/imaginary decomposition, which is what makes them tens of
thousands of lines long.

This module emits those branches from the pairing table.  The product
template is fixed by VeloxChem's generic contraction stage and is stated
ONCE, as :data:`PRODUCT_ROWS`; ``emit_branch`` walks it rather than
restating it in control flow.  Each row is tied to the functional
ingredient it serves, so which rows apply to a family follows from that
family's ingredient set -- a tau family is data, not new branching.

Complex products expand through VeloxChem's prod2_r/prod2_i helpers,
whose algebra is owned by :func:`..emitters.codegen.part_product`; only
the helper NAMES are VeloxChem's, and they live in one table here.
The pairing tables below follow the density layout documented in each
python response driver's get_densities().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

AXES = ("x", "y", "z")


@dataclass(frozen=True)
class PairSpec:
    """One output slot of a quadratic-response product branch.

    out:    output slot index (per repetition block)
    a, b:   first-order density labels (indices into the block's densities)
    factor: overall prefactor in front of the symmetrized product
    """
    out: int
    a: str
    b: str
    factor: float = 1.0


@dataclass(frozen=True)
class ModeSpec:
    """Pairing table of a quadratic-response mode.

    densities: complex first-order densities per block, in storage order;
               each occupies two real slots (Re, Im) of the density grid.
    outputs:   complex output slots per block; each occupies two real gam
               slots (Re, Im).
    pairs:     the products.
    """
    name: str
    densities: Tuple[str, ...]
    n_out: int
    pairs: Tuple[PairSpec, ...]


#: Density layouts follow get_densities() in the python response drivers.
MODES = {
    # quadraticresponsedriver.get_densities: Db, Dc; one output Fbc
    "QRF": ModeSpec(name="QRF", densities=("B", "C"), n_out=1,
                    pairs=(PairSpec(0, "B", "C"),)),
}


def _gam_decl(n_out: int, rows, ind: str) -> List[str]:
    """Declare the gam output slots each product row writes into.

    The accessor name is ``gam`` suffixed with the slot's uppercased
    Cartesian indices, which is exactly the row's axis structure."""
    lines = []
    for slot in range(n_out):
        pos = f"{2 * n_out} * j + {2 * slot}" if n_out > 1 else "2 * j"
        pos_i = (f"{2 * n_out} * j + {2 * slot + 1}" if n_out > 1
                 else "2 * j + 1")
        for row in rows:
            for axes in _slot_axes(row):
                sfx = "".join(axes)
                acc = "gam" + "".join(a.upper() for a in axes)
                name = f"gam{slot}" + (f"_{sfx}" if sfx else "")
                lines.append(f"{ind}auto {name}_r = {acc}({pos});")
                lines.append(f"{ind}auto {name}_i = {acc}({pos_i});")
    return lines


#: VeloxChem's pointwise helpers, one row per part: (canonical part name
#: as used by codegen.part_product, VeloxChem's tag for it, helper name).
#: The decomposition itself -- prod2_r(ar,ai,br,bi) = ar*br - ai*bi and
#: prod2_i = ar*bi + ai*br -- is owned by part_product; this table only
#: names the helpers that realize it.
_HELPERS = (("re", "r", "prod2_r"), ("im", "i", "prod2_i"))
_VLX_PART = {tag: part for part, tag, _h in _HELPERS}
_PROD_HELPER = {part: h for part, _tag, h in _HELPERS}
#: VeloxChem's part tags, in emission order
PARTS = tuple(tag for _p, tag, _h in _HELPERS)


@dataclass(frozen=True)
class ProductRow:
    """One row of the generic contraction stage's product template.

    ingredient: the functional ingredient this row serves; the row applies
                to a family exactly when the family has that ingredient
    naxes:      Cartesian indices carried by the output slot (0, 1 or 2)
    factor:     prefactor in front of the symmetrized product
    left/right: which field of each density enters, per side; the axes of
                the slot are handed to the sides in order
    """

    ingredient: str
    naxes: int
    factor: float
    left: str
    right: str


#: THE product template of VeloxChem's generic contraction stage:
#:     gam    +=     rho_A rho_B            (both orderings)
#:     gamK   += 2 * dK rho_A rho_B         (both orderings, K in xyz)
#:     gamKL  +=     dK rho_A dL rho_B      (both orderings)
#: stated here only.
PRODUCT_ROWS = (
    ProductRow("rho",   0, 1.0, "rho",  "rho"),
    ProductRow("sigma", 1, 2.0, "grad", "rho"),
    ProductRow("sigma", 2, 1.0, "grad", "grad"),
)


def rows_for(family: str):
    """The product rows a family needs, from its ingredient set."""
    from ..inputs.functional import Functional
    have = {i.name for i in Functional.of_family(family).ingredients}
    rows = tuple(r for r in PRODUCT_ROWS if r.ingredient in have)
    missing = have - {r.ingredient for r in PRODUCT_ROWS}
    if missing:
        raise NotImplementedError(
            f"family {family!r} needs ingredient(s) {sorted(missing)}, for "
            "which no VeloxChem product rows are declared; add them to "
            "PRODUCT_ROWS together with the matching gam accessors")
    return rows


def _slot_axes(row: ProductRow):
    """The Cartesian index tuples a row's output slot ranges over."""
    if row.naxes == 0:
        return ((),)
    if row.naxes == 1:
        return tuple((a,) for a in AXES)
    return tuple((a1, a2) for a1 in AXES for a2 in AXES)


def _field(kind: str, label: str, axes) -> str:
    """The split-storage field stem for one side of a product."""
    if kind == "rho":
        return f"rho{label}"
    return f"{kind}{label}_{axes[0]}"


def prod_expansion(part: str) -> str:
    """The expanded expression a helper must implement, from the shared
    decomposition table -- pinned against the emitted helper calls by the
    validation suite."""
    from .codegen import part_product
    rows = part_product(_VLX_PART[part], conjugate_left=False)
    return " ".join(f"{'+' if s > 0 else '-'} a_{lp[:1]}*b_{rp[:1]}"
                    for s, lp, rp in rows).lstrip("+ ")


def _prod(part: str, fa: str, fb: str) -> str:
    """The ``part`` of the plain complex product fa*fb of two split-storage
    fields, emitted as the VeloxChem helper call for that decomposition."""
    helper = _PROD_HELPER[_VLX_PART[part]]
    return (f"{helper}({fa}_r[i],{fa}_i[i],{fb}_r[i],{fb}_i[i])")


def _fmt_factor(x: float) -> str:
    return f"{x:.1f} * " if x != 1.0 else ""


def _accum(target: str, factor: float, part: str,
           fa1: str, fb1: str, fa2: str, fb2: str, ind: str) -> List[str]:
    """target += factor * (prod(a1,b1) + prod(a2,b2)); both orderings."""
    f = _fmt_factor(factor)
    return [f"{ind}{target}[i] += {f}{_prod(part, fa1, fb1)}",
            f"{ind}            + {f}{_prod(part, fa2, fb2)};"]


def emit_branch(mode: str, family: str, indent: int = 8) -> str:
    """The full quadMode branch body for DensityProdFor{LDA,GGA}."""
    spec = MODES[mode.upper()]
    rows = rows_for(family)
    ind0 = " " * indent
    ind1 = ind0 + " " * 4
    ind2 = ind1 + " " * 4

    #: which per-density fields the rows reference at all
    kinds = {k for row in rows for k in (row.left, row.right)}

    nden = 2 * len(spec.densities)   # real slots per block
    lines = [f"{ind0}// generated by xckernel vlxwriter: mode {spec.name}, "
             f"family {family.upper()}",
             f"{ind0}for (int j = 0; j < numdens / {2 * spec.n_out}; j++)",
             ind0 + "{"]
    for k, lbl in enumerate(spec.densities):
        for part, off in zip(PARTS, (0, 1)):
            lines.append(f"{ind1}auto rho{lbl}_{part} = "
                         f"rwDensityGrid.alphaDensity"
                         f"({nden} * j + {2 * k + off});")
            if "grad" in kinds:
                for ax in AXES:
                    lines.append(f"{ind1}auto grad{lbl}_{ax}_{part} = "
                                 f"rwDensityGrid.alphaDensityGradient"
                                 f"{ax.upper()}({nden} * j + {2 * k + off});")
        lines.append("")
    lines += _gam_decl(spec.n_out, rows, ind1)
    lines.append("")
    lines.append(f"{ind1}for (int i = 0; i < npoints; i++)")
    lines.append(ind1 + "{")
    for p in spec.pairs:
        A, B, s_ = p.a, p.b, p.out
        for row in rows:
            for axes in _slot_axes(row):
                sfx = "".join(axes)
                target = f"gam{s_}" + (f"_{sfx}" if sfx else "")
                # the slot's axes are handed to the sides in order
                la = axes[:1] if row.left == "grad" else ()
                ra = axes[-1:] if row.right == "grad" else ()
                for part in PARTS:
                    lines += _accum(
                        f"{target}_{part}", row.factor * p.factor, part,
                        _field(row.left, A, la), _field(row.right, B, ra),
                        _field(row.left, B, la), _field(row.right, A, ra),
                        ind2)
    lines.append(ind1 + "}")
    lines.append(ind0 + "}")
    return "\n".join(lines)


def main(argv=None) -> None:
    import argparse
    p = argparse.ArgumentParser(
        prog="python -m xckernel.emitters.vlxwriter",
        description="Emit a VeloxChem quadMode branch body.")
    p.add_argument("mode", nargs="?", default="QRF",
                   choices=sorted(MODES), type=str.upper,
                   help="response mode (default: QRF)")
    p.add_argument("family", nargs="?", default="gga",
                   help="functional family (default: gga)")
    p.add_argument("--emit-dir", metavar="DIR",
                   help="write the generated regions as .inc files here "
                        "instead of printing a branch body")
    a = p.parse_args(argv)
    if a.emit_dir:
        for fname in write_include_files(a.emit_dir):
            print(fname)
        return
    print(emit_branch(a.mode, a.family))



# --- open-shell response contractions -------------------------------------
#
# VeloxChem has no open-shell kxc or lxc on any rung, and no open-shell
# fxc for meta-GGAs: XCIntegrator throws "Not implemented for open-shell".
# The missing piece is not the driver scaffolding, which their existing
# open-shell fxc already demonstrates, but the spin-resolved chain rule --
# 130,566 terms for the meta-GGA fourth-order case alone, which is why it
# was never hand-derived. Here it is generated instead.

#: Libxc flat component names per derivative array, in libxc's own
#: packing order, spelled the way VeloxChem spells them.
SPIN_COMPONENTS = {
    "vrho": ("a", "b"),
    "v2rho2": ("aa", "ab", "bb"),
    "v3rho3": ("aaa", "aab", "abb", "bbb"),
    "v4rho4": ("aaaa", "aaab", "aabb", "abbb", "bbbb"),
}


def _vlx_deriv_name(sym: str) -> str:
    """xckernel's flat index -> VeloxChem's spelled component."""
    base, _, idx = sym.rpartition("_")
    comps = SPIN_COMPONENTS.get(base)
    if comps is None or not idx.isdigit():
        return sym
    return f"{base}_{comps[int(idx)]}"


def _density_symbol(spins, labels):
    """The density-grid component a monomial's perturbed densities name.

    The perturbed densities are complex, and their products are already
    formed by the density-product layer -- VeloxChem's gam arrays, built
    with prod2_r/prod2_i.  The contraction therefore consumes gam
    components rather than re-deriving the products, which keeps the
    complex expansion in the one place that already owns it.

    The spin pair is kept ORDERED: gam_ab is rhoB_a * rhoC_b and gam_ba
    is rhoB_b * rhoC_a, as separate components.  Folding them into a
    single symmetrized "gam_ab" with a multiplicity of two would be
    valid only when the two perturbations carry the same density, and it
    disagrees with a convention that also doubles the diagonal.  Kept
    ordered, every coefficient maps one-to-one with no factor at all,
    and the ambiguity cannot arise.
    """
    ordered = [s for _, s in sorted(zip(labels, spins))]
    return "gam_" + "".join(ordered)


def openshell_contraction(family: str, order: int, indent: int = 24):
    """Loop-body lines assigning G_a_val / G_b_val for one order.

    Returns VeloxChem-idiom source: the same shape as the hand-written
    closed-shell bodies, but with the spin sums carried explicitly
    instead of folded into binomial combinations that only hold when
    rho_a == rho_b.
    """
    import sympy as sp
    from ..engine.spin_kernel import response_fock_spin

    ind = " " * indent
    out = []
    for spin in ("a", "b"):
        ri = response_fock_spin(family, spin, order)
        cu = sp.Symbol("chi_u", real=True)
        cv = sp.Symbol("chi_v", real=True)
        w = sp.Symbol("w", real=True, positive=True)
        expr = sp.expand(ri.expr / (cu * cv * w))

        groups = {}
        for mono in sp.Add.make_args(expr):
            deriv, spins, labels, coeff = None, [], [], sp.Integer(1)
            for fac in mono.as_ordered_factors():
                base = fac.as_base_exp()[0]
                name = getattr(base, "name", None)
                if name is None:
                    coeff *= fac
                    continue
                exp = int(fac.as_base_exp()[1])
                if name.startswith("rho_"):
                    _, s, lab = name.split("_")
                    spins += [s] * exp
                    labels += [lab] * exp
                elif name.startswith(("v2", "v3", "v4", "vrho")):
                    deriv = name
                else:
                    coeff *= fac
            key = (_vlx_deriv_name(deriv), _density_symbol(spins, labels))
            groups[key] = groups.get(key, sp.Integer(0)) + coeff

        terms = []
        for (d, rho), c in sorted(groups.items()):
            c = sp.nsimplify(c)
            pre = "" if c == 1 else f"{float(c)} * "
            terms.append(f"{pre}{d} * {rho}")
        body = "\n".join(f"{ind}    {'  ' if i else ''}{'+ ' if i else ''}{t}"
                         for i, t in enumerate(terms))
        out.append(f"{ind}G_{spin}_val[nu_offset + g] += weights[g] *\n"
                   f"{body}\n{ind}    ;")
    return "\n".join(out)


# --- include-file emission -------------------------------------------------
#
# The generated regions go into standalone .inc files that the VeloxChem
# sources #include, rather than being spliced between markers. Splicing
# makes regeneration a merge problem; an include makes it a file
# overwrite. This mirrors the psi4backend --emit-dir convention, and it
# also matches VeloxChem's own practice of committing generated sources
# (src/onee_ints carries 15 of them) -- with the improvement that these
# name a public generator they can re-run.

INCLUDE_FILE_NOTICE = (
    "// This file is machine-generated by xckernel in its entirety;\n"
    "// do not edit.  Reproduce with:\n"
    "//     python -m xckernel.emitters.vlxwriter --emit-dir <this directory>\n"
    "// Copyright (c) 2026 Susi Lehtola.\n")

#: Open-shell response contractions VeloxChem lacks entirely.  The
#: closed-shell counterparts exist and are trusted; these are the spin
#: sums they fold away, restored.
OPENSHELL_REGIONS = (
    ("lda", 3, "vlx_openshell_lda_kxc"),
    ("lda", 4, "vlx_openshell_lda_lxc"),
)


def emit_openshell_regions() -> "dict[str, str]":
    """Every open-shell contraction region, keyed by include-file stem."""
    out = {}
    # The meta-GGA fxc is the one region reachable from Python today:
    # the unrestricted linear-response solvers call integrate_fxc_fock,
    # which currently throws for a meta-GGA.
    out["vlx_openshell_mgga_fxc"] = (
        "// Open-shell tau-meta-GGA fxc contraction (order 2).\n"
        "// Included inside the (nu, g) loop of the generated driver,\n"
        "// which declares the operands this uses.  Assigns the value\n"
        "// channel, the FUSED gradient channel (the three Cartesian\n"
        "// components pre-contracted, as VeloxChem already does), and\n"
        "// the tau channel per component.\n"
        "// Tau-only, matching VeloxChem's own convention: their\n"
        "// closed-shell meta-GGA fxc allocates the Laplacian arrays but\n"
        "// has every Laplacian contribution commented out.\n"
        + openshell_fxc_mgga(indent=24))
    out["vlx_openshell_mgga_fxc_driver"] = (
        "// Open-shell tau-meta-GGA fxc driver.  Include at namespace\n"
        "// scope inside XCIntegratorForMGGA.cpp (namespace xcintmgga).\n"
        + emit_mgga_fxc_driver())
    out["vlx_openshell_mgga_fxc_decl"] = (
        "// Declaration; include inside namespace xcintmgga in\n"
        "// XCIntegratorForMGGA.hpp.\n"
        + DRIVER_MGGA_FXC.split("{{")[0].replace("\nauto\n", "auto\n", 1)
                         .rstrip().rstrip("\n") + ";\n")
    out["vlx_openshell_mgga_fxc_dispatch"] = (
        "// Replaces the \"Only implemented for open-shell LDA/GGA\" branch\n"
        "// of XCIntegrator::integrateFxcFock.\n"
        "else if (xcfuntype == xcfun::mgga)\n"
        "{\n"
        "    xcintmgga::integrateFxcFockForMetaGgaOpenShell(\n"
        "        aoFockPointers, molecule, basis, rwDensityPointers,"
        " gsDensityPointers, molecularGrid, _screeningThresholdForGTOValues,"
        " fvxc);\n"
        "}\n")
    for family, order, stem in OPENSHELL_REGIONS:
        nm = {2: "fxc", 3: "kxc", 4: "lxc"}[order]
        head = (f"// Open-shell {family.upper()} {nm} contraction "
                f"(order {order}).\n"
                f"// Included inside the (nu, g) loop; assigns G_a_val and\n"
                f"// G_b_val.  Operands: weights, the perturbed densities\n"
                f"// rhoB_a/rhoB_b/... per spin, and the Libxc arrays with\n"
                f"// their components spelled out.  Setting rho_a = rho_b\n"
                f"// reproduces VeloxChem's hand-folded closed-shell\n"
                f"// coefficients exactly (see xckernel.tests."
                f"vlx_openshell_validate).\n")
        out[stem] = head + openshell_contraction(family, order, indent=16)
    return out


def write_include_files(directory: str) -> "list[str]":
    """Write each region to <directory>/<stem>.inc; return the file names."""
    import os
    os.makedirs(directory, exist_ok=True)
    written = []
    for stem, text in sorted(emit_openshell_regions().items()):
        fname = f"{stem}.inc"
        with open(os.path.join(directory, fname), "w") as f:
            f.write(INCLUDE_FILE_NOTICE)
            f.write(text)
            f.write("\n")
        written.append(fname)
    return written


# --- open-shell meta-GGA fxc ----------------------------------------------
#
# This is the one missing piece that is reachable from Python today: the
# unrestricted linear-response solvers exist (lreigensolverunrest,
# lrsolverunrest, tdaeigensolverunrest, cppsolverunrest) and reach
# integrate_fxc_fock, which throws "Only implemented for open-shell
# LDA/GGA" for a meta-GGA. An unrestricted TD-DFT run with a
# meta-GGA therefore fails outright.
#
# VeloxChem's meta-GGA XC is tau-only IN PRACTICE: their closed-shell
# fxc allocates the Laplacian arrays and asks Libxc for them, but every
# Laplacian contribution is commented out. The generated open-shell
# counterpart matches that convention rather than Libxc's full mGGA, so
# the spin-compensated limit is comparable against their own routine.

#: xckernel operand -> the name our generated driver declares for it.
#: Ground-state gradients, the perturbed density/gradient/tau, and the
#: quadrature weight. Libxc derivative components are emitted as flat
#: indexed reads instead, which cannot be misspelled.
VLX_OPERANDS = {
    "w": "w",
    "rho_a_p1": "rwa", "rho_b_p1": "rwb",
    "tau_a_p1": "tauwa", "tau_b_p1": "tauwb",
}
for _s in ("a", "b"):
    for _ax in ("x", "y", "z"):
        VLX_OPERANDS[f"grad_rho_{_s}_{_ax}"] = f"grad{_s}_{_ax}"
        VLX_OPERANDS[f"grad_rho_{_s}_p1_{_ax}"] = f"rw{_s}_{_ax}"

#: Libxc polarized component spellings, in Libxc's packing order.
VLX_COMPONENTS = {
    "vrho": ("a", "b"),
    "vsigma": ("aa", "ab", "bb"),
    "vtau": ("a", "b"),
    "v2rho2": ("aa", "ab", "bb"),
    "v2rhosigma": ("a_aa", "a_ab", "a_bb", "b_aa", "b_ab", "b_bb"),
    "v2rhotau": ("aa", "ab", "ba", "bb"),
    "v2sigma2": ("aa_aa", "aa_ab", "aa_bb", "ab_ab", "ab_bb", "bb_bb"),
    "v2sigmatau": ("aa_a", "aa_b", "ab_a", "ab_b", "bb_a", "bb_b"),
    "v2tau2": ("aa", "ab", "bb"),
}


def _vlx_operand(name: str) -> str:
    """Render one xckernel operand in the generated driver's vocabulary."""
    if name in VLX_OPERANDS:
        return VLX_OPERANDS[name]
    base, _, idx = name.rpartition("_")
    comps = VLX_COMPONENTS.get(base)
    if comps is not None and idx.isdigit():
        return f"{base}_{comps[int(idx)]}"
    return name


def _render(expr) -> str:
    """C++ expression for one channel, in the driver's operand names.

    Integer powers are written out as products: sympy's default ccode
    emits pow(x, 2), and this sits in the innermost loop over grid
    points, where a libm call per term would be a real cost. VeloxChem's
    own hand-written kernels write the products out for the same reason.
    """
    import sympy as sp
    from .fieldkernel import CxxPrinter
    sub = {t: sp.Symbol(_vlx_operand(t.name), real=True)
           for t in expr.free_symbols}
    return CxxPrinter().doprint(sp.expand(expr.subs(sub)))


def openshell_fxc_mgga(indent: int = 24) -> str:
    """Loop body for the open-shell tau-meta-GGA fxc.

    Assigns the value channel (G), the FUSED gradient channel (G_gga,
    the three Cartesian components pre-contracted against the basis
    gradients as VeloxChem already does for its closed-shell routines),
    and the tau channel per component (G_gga_x/y/z).
    """
    from ..engine.spin_kernel import fxc_channels_spin

    ch = fxc_channels_spin("mgga_tau")
    ind = " " * indent
    out = []
    for s in ("a", "b"):
        out.append(f"{ind}G_{s}_val[nu_offset + g] = w * ("
                   f"{_render(ch[f'rho_{s}'])}) * chi_val[nu_offset + g];")
        fused = " + ".join(
            f"({_render(ch[f'grad_{s}_{ax}'])}) * chi_{ax}_val[nu_offset + g]"
            for ax in ("x", "y", "z"))
        out.append(f"{ind}G_gga_{s}_val[nu_offset + g] = w * ({fused});")
        tau = _render(ch[f"tau_{s}"])
        for ax in ("x", "y", "z"):
            out.append(f"{ind}G_gga_{s}_{ax}_val[nu_offset + g] = "
                       f"w * ({tau}) * chi_{ax}_val[nu_offset + g];")
    return "\n".join(out)


DRIVER_MGGA_FXC = '''\
auto
integrateFxcFockForMetaGgaOpenShell(const std::vector<double*>&       aoFockPointers,
                                    const CMolecule&                  molecule,
                                    const CMolecularBasis&            basis,
                                    const std::vector<const double*>& rwDensityPointers,
                                    const std::vector<const double*>& gsDensityPointers,
                                    const CMolecularGrid&             molecularGrid,
                                    const double                      screeningThresholdForGTOValues,
                                    const CXCFunctional&              xcFunctional) -> void
{{
    CMultiTimer timer;

    timer.start("Total timing");

    auto nthreads = omp_get_max_threads();

    std::vector<CMultiTimer> omptimers(nthreads);

    const auto gto_blocks = gtofunc::make_gto_blocks(basis, molecule);

    const auto naos = gtofunc::getNumberOfAtomicOrbitals(gto_blocks);

    auto xcoords = molecularGrid.getCoordinatesX();
    auto ycoords = molecularGrid.getCoordinatesY();
    auto zcoords = molecularGrid.getCoordinatesZ();

    auto weights = molecularGrid.getWeights();

    auto counts = molecularGrid.getGridPointCounts();

    auto displacements = molecularGrid.getGridPointDisplacements();

    const auto n_boxes = counts.size();

    const auto n_gto_blocks = gto_blocks.size();

    // two spin channels per perturbed density
    const auto n_rw_densities = rwDensityPointers.size() / 2;

    auto ptr_gto_blocks = gto_blocks.data();

    auto ptr_xcFunctional = &xcFunctional;

#pragma omp parallel shared(displacements, xcoords, ycoords, zcoords, \\
                            ptr_gto_blocks, gsDensityPointers, ptr_xcFunctional, \\
                            n_boxes, n_gto_blocks, n_rw_densities, naos, \\
                            aoFockPointers, rwDensityPointers)
    {{

#pragma omp single nowait
    {{

    for (size_t box_id = 0; box_id < n_boxes; box_id++)
    {{

    #pragma omp task firstprivate(box_id)
    {{
        auto thread_id = omp_get_thread_num();

        auto npoints = counts.data()[box_id];

        auto gridblockpos = displacements.data()[box_id];

        auto boxdim = prescr::getGridBoxDimension(gridblockpos, npoints, xcoords, ycoords, zcoords);

        omptimers[thread_id].start("GTO pre-screening");

        std::vector<std::vector<int>> cgto_mask_blocks, pre_ao_inds_blocks;

        std::vector<int> aoinds;

        cgto_mask_blocks.reserve(n_gto_blocks);
        pre_ao_inds_blocks.reserve(n_gto_blocks);
        aoinds.reserve(naos);

        for (size_t i = 0; i < n_gto_blocks; i++)
        {{
            // 1st order GTO derivative for a meta-GGA
            auto [cgto_mask, pre_ao_inds] = prescr::preScreenGtoBlock(ptr_gto_blocks[i], 1, screeningThresholdForGTOValues, boxdim);

            cgto_mask_blocks.push_back(cgto_mask);
            pre_ao_inds_blocks.push_back(pre_ao_inds);

            for (const auto nu : pre_ao_inds)
            {{
                aoinds.push_back(nu);
            }}
        }}

        const auto aocount = static_cast<int>(aoinds.size());

        omptimers[thread_id].stop("GTO pre-screening");

        if (aocount > 0)
        {{
            omptimers[thread_id].start("Density matrix slicing");

            auto sub_dens_mat_a = dftsubmat::getSubDensityMatrix(gsDensityPointers[0], aoinds, naos);
            auto sub_dens_mat_b = dftsubmat::getSubDensityMatrix(gsDensityPointers[1], aoinds, naos);

            std::vector<CDenseMatrix> rw_sub_dens_mat_vec_a(n_rw_densities);
            std::vector<CDenseMatrix> rw_sub_dens_mat_vec_b(n_rw_densities);

            for (int idensity = 0; idensity < static_cast<int>(n_rw_densities); idensity++)
            {{
                rw_sub_dens_mat_vec_a[idensity] = dftsubmat::getSubDensityMatrix(rwDensityPointers[idensity * 2 + 0], aoinds, naos);
                rw_sub_dens_mat_vec_b[idensity] = dftsubmat::getSubDensityMatrix(rwDensityPointers[idensity * 2 + 1], aoinds, naos);
            }}

            omptimers[thread_id].stop("Density matrix slicing");

            omptimers[thread_id].start("gtoeval");

            CDenseMatrix mat_chi(aocount, npoints);
            CDenseMatrix mat_chi_x(aocount, npoints);
            CDenseMatrix mat_chi_y(aocount, npoints);
            CDenseMatrix mat_chi_z(aocount, npoints);

            const auto grid_x_ptr = xcoords + gridblockpos;
            const auto grid_y_ptr = ycoords + gridblockpos;
            const auto grid_z_ptr = zcoords + gridblockpos;

            std::vector<double> grid_x(grid_x_ptr, grid_x_ptr + npoints);
            std::vector<double> grid_y(grid_y_ptr, grid_y_ptr + npoints);
            std::vector<double> grid_z(grid_z_ptr, grid_z_ptr + npoints);

            for (int i_block = 0, idx = 0; i_block < static_cast<int>(n_gto_blocks); i_block++)
            {{
                const auto& gto_block = ptr_gto_blocks[i_block];

                const auto& cgto_mask = cgto_mask_blocks[i_block];

                const auto& pre_ao_inds = pre_ao_inds_blocks[i_block];

                auto cmat = gtoval::get_gto_values_for_gga(gto_block, grid_x, grid_y, grid_z, cgto_mask);

                if (cmat.is_empty()) continue;

                auto submat_0_ptr = cmat.sub_matrix({{0, 0}});
                auto submat_x_ptr = cmat.sub_matrix({{1, 0}});
                auto submat_y_ptr = cmat.sub_matrix({{1, 1}});
                auto submat_z_ptr = cmat.sub_matrix({{1, 2}});

                auto submat_0_data = submat_0_ptr->data();
                auto submat_x_data = submat_x_ptr->data();
                auto submat_y_data = submat_y_ptr->data();
                auto submat_z_data = submat_z_ptr->data();

                for (int nu = 0; nu < static_cast<int>(pre_ao_inds.size()); nu++, idx++)
                {{
                    std::memcpy(mat_chi.row(idx), submat_0_data + nu * npoints, npoints * sizeof(double));
                    std::memcpy(mat_chi_x.row(idx), submat_x_data + nu * npoints, npoints * sizeof(double));
                    std::memcpy(mat_chi_y.row(idx), submat_y_data + nu * npoints, npoints * sizeof(double));
                    std::memcpy(mat_chi_z.row(idx), submat_z_data + nu * npoints, npoints * sizeof(double));
                }}
            }}

            omptimers[thread_id].stop("gtoeval");

            omptimers[thread_id].start("Generate density grid");

            auto local_xcfunc = CXCFunctional(*ptr_xcFunctional);

            auto       mggafunc = local_xcfunc.getFunctionalPointerToMetaGgaComponent();
            const auto dim      = &(mggafunc->dim);

            std::vector<double> local_weights_data(weights + gridblockpos, weights + gridblockpos + npoints);

            std::vector<double> rho_data(dim->rho * npoints);
            std::vector<double> rhograd_data(dim->rho * 3 * npoints);
            std::vector<double> sigma_data(dim->sigma * npoints);
            std::vector<double> lapl_data(dim->lapl * npoints);
            std::vector<double> tau_data(dim->tau * npoints);

            std::vector<double> rhow_data(dim->rho * npoints);
            std::vector<double> rhowgrad_data(dim->rho * 3 * npoints);
            std::vector<double> laplw_data(dim->lapl * npoints);
            std::vector<double> tauw_data(dim->tau * npoints);

            std::vector<double> vrho_data(dim->vrho * npoints);
            std::vector<double> vsigma_data(dim->vsigma * npoints);
            std::vector<double> vlapl_data(dim->vlapl * npoints);
            std::vector<double> vtau_data(dim->vtau * npoints);

            std::vector<double> v2rho2_data(dim->v2rho2 * npoints);
            std::vector<double> v2rhosigma_data(dim->v2rhosigma * npoints);
            std::vector<double> v2rholapl_data(dim->v2rholapl * npoints);
            std::vector<double> v2rhotau_data(dim->v2rhotau * npoints);
            std::vector<double> v2sigma2_data(dim->v2sigma2 * npoints);
            std::vector<double> v2sigmalapl_data(dim->v2sigmalapl * npoints);
            std::vector<double> v2sigmatau_data(dim->v2sigmatau * npoints);
            std::vector<double> v2lapl2_data(dim->v2lapl2 * npoints);
            std::vector<double> v2lapltau_data(dim->v2lapltau * npoints);
            std::vector<double> v2tau2_data(dim->v2tau2 * npoints);

            auto local_weights = local_weights_data.data();

            auto rho     = rho_data.data();
            auto rhograd = rhograd_data.data();
            auto sigma   = sigma_data.data();
            auto lapl    = lapl_data.data();
            auto tau     = tau_data.data();

            auto rhow     = rhow_data.data();
            auto rhowgrad = rhowgrad_data.data();
            auto laplw    = laplw_data.data();
            auto tauw     = tauw_data.data();

            auto vrho   = vrho_data.data();
            auto vsigma = vsigma_data.data();
            auto vlapl  = vlapl_data.data();
            auto vtau   = vtau_data.data();

            auto v2rho2      = v2rho2_data.data();
            auto v2rhosigma  = v2rhosigma_data.data();
            auto v2rholapl   = v2rholapl_data.data();
            auto v2rhotau    = v2rhotau_data.data();
            auto v2sigma2    = v2sigma2_data.data();
            auto v2sigmalapl = v2sigmalapl_data.data();
            auto v2sigmatau  = v2sigmatau_data.data();
            auto v2lapl2     = v2lapl2_data.data();
            auto v2lapltau   = v2lapltau_data.data();
            auto v2tau2      = v2tau2_data.data();

            sdengridgen::serialGenerateDensityForMGGA(
                rho, rhograd, sigma, lapl, tau, mat_chi, mat_chi_x, mat_chi_y, mat_chi_z, sub_dens_mat_a, sub_dens_mat_b);

            omptimers[thread_id].stop("Generate density grid");

            omptimers[thread_id].start("XC functional eval.");

            local_xcfunc.compute_vxc_for_mgga(npoints, rho, sigma, lapl, tau, vrho, vsigma, vlapl, vtau);

            local_xcfunc.compute_fxc_for_mgga(npoints, rho, sigma, lapl, tau, v2rho2, v2rhosigma, v2rholapl, v2rhotau,
                                              v2sigma2, v2sigmalapl, v2sigmatau, v2lapl2, v2lapltau, v2tau2);

            omptimers[thread_id].stop("XC functional eval.");

            for (int idensity = 0; idensity < static_cast<int>(n_rw_densities); idensity++)
            {{
                omptimers[thread_id].start("Generate density grid");

                sdengridgen::serialGenerateDensityForMGGA(rhow, rhowgrad, nullptr, laplw, tauw, mat_chi, mat_chi_x, mat_chi_y, mat_chi_z,
                                                          rw_sub_dens_mat_vec_a[idensity], rw_sub_dens_mat_vec_b[idensity]);

                omptimers[thread_id].stop("Generate density grid");

                omptimers[thread_id].start("Fxc matrix G");

                CDenseMatrix mat_G_a(aocount, npoints);
                CDenseMatrix mat_G_b(aocount, npoints);
                CDenseMatrix mat_G_a_gga(aocount, npoints);
                CDenseMatrix mat_G_b_gga(aocount, npoints);
                CDenseMatrix mat_G_a_gga_x(aocount, npoints);
                CDenseMatrix mat_G_a_gga_y(aocount, npoints);
                CDenseMatrix mat_G_a_gga_z(aocount, npoints);
                CDenseMatrix mat_G_b_gga_x(aocount, npoints);
                CDenseMatrix mat_G_b_gga_y(aocount, npoints);
                CDenseMatrix mat_G_b_gga_z(aocount, npoints);

                auto G_a_val = mat_G_a.values();
                auto G_b_val = mat_G_b.values();
                auto G_a_gga_val = mat_G_a_gga.values();
                auto G_b_gga_val = mat_G_b_gga.values();
                auto G_a_gga_x_val = mat_G_a_gga_x.values();
                auto G_a_gga_y_val = mat_G_a_gga_y.values();
                auto G_a_gga_z_val = mat_G_a_gga_z.values();
                auto G_b_gga_x_val = mat_G_b_gga_x.values();
                auto G_b_gga_y_val = mat_G_b_gga_y.values();
                auto G_b_gga_z_val = mat_G_b_gga_z.values();

                auto chi_val   = mat_chi.values();
                auto chi_x_val = mat_chi_x.values();
                auto chi_y_val = mat_chi_y.values();
                auto chi_z_val = mat_chi_z.values();

                for (int nu = 0; nu < aocount; nu++)
                {{
                    auto nu_offset = nu * npoints;

                    #pragma omp simd
                    for (int g = 0; g < npoints; g++)
                    {{
                        double w = local_weights[g];

                        // ground-state gradient
                        double grada_x = rhograd[6 * g + 0];
                        double grada_y = rhograd[6 * g + 1];
                        double grada_z = rhograd[6 * g + 2];
                        double gradb_x = rhograd[6 * g + 3];
                        double gradb_y = rhograd[6 * g + 4];
                        double gradb_z = rhograd[6 * g + 5];

                        // perturbed density, gradient and kinetic energy density
                        double rwa = rhow[2 * g + 0];
                        double rwb = rhow[2 * g + 1];

                        double rwa_x = rhowgrad[6 * g + 0];
                        double rwa_y = rhowgrad[6 * g + 1];
                        double rwa_z = rhowgrad[6 * g + 2];
                        double rwb_x = rhowgrad[6 * g + 3];
                        double rwb_y = rhowgrad[6 * g + 4];
                        double rwb_z = rhowgrad[6 * g + 5];

                        double tauwa = tauw[2 * g + 0];
                        double tauwb = tauw[2 * g + 1];

{decls}

{body}
                    }}
                }}

                omptimers[thread_id].stop("Fxc matrix G");

                omptimers[thread_id].start("Fxc matmul and symm.");

                // One matrix product for the value and gradient channels
                // together, as the meta-GGA Vxc of this file already does:
                // both contract against mat_chi, so they fuse into a single
                // operand before the product.
                auto partial_mat_Fxc_a = sdenblas::serialMultABt(mat_chi, sdenblas::serialAddAB(mat_G_a, mat_G_a_gga, 2.0));
                auto partial_mat_Fxc_b = sdenblas::serialMultABt(mat_chi, sdenblas::serialAddAB(mat_G_b, mat_G_b_gga, 2.0));

                // tau contribution
                auto partial_mat_Fxc_a_x = sdenblas::serialMultABt(mat_chi_x, mat_G_a_gga_x);
                auto partial_mat_Fxc_a_y = sdenblas::serialMultABt(mat_chi_y, mat_G_a_gga_y);
                auto partial_mat_Fxc_a_z = sdenblas::serialMultABt(mat_chi_z, mat_G_a_gga_z);
                auto partial_mat_Fxc_b_x = sdenblas::serialMultABt(mat_chi_x, mat_G_b_gga_x);
                auto partial_mat_Fxc_b_y = sdenblas::serialMultABt(mat_chi_y, mat_G_b_gga_y);
                auto partial_mat_Fxc_b_z = sdenblas::serialMultABt(mat_chi_z, mat_G_b_gga_z);

                sdenblas::serialInPlaceAddAB(partial_mat_Fxc_a, partial_mat_Fxc_a_x, 0.5);
                sdenblas::serialInPlaceAddAB(partial_mat_Fxc_a, partial_mat_Fxc_a_y, 0.5);
                sdenblas::serialInPlaceAddAB(partial_mat_Fxc_a, partial_mat_Fxc_a_z, 0.5);
                sdenblas::serialInPlaceAddAB(partial_mat_Fxc_b, partial_mat_Fxc_b_x, 0.5);
                sdenblas::serialInPlaceAddAB(partial_mat_Fxc_b, partial_mat_Fxc_b_y, 0.5);
                sdenblas::serialInPlaceAddAB(partial_mat_Fxc_b, partial_mat_Fxc_b_z, 0.5);

                partial_mat_Fxc_a.symmetrizeAndScale(0.5);
                partial_mat_Fxc_b.symmetrizeAndScale(0.5);

                omptimers[thread_id].stop("Fxc matmul and symm.");

                omptimers[thread_id].start("Fxc dist.");

                #pragma omp critical
                {{
                    dftsubmat::distributeSubMatrixToFock(aoFockPointers, idensity * 2 + 0, partial_mat_Fxc_a, aoinds, naos);
                    dftsubmat::distributeSubMatrixToFock(aoFockPointers, idensity * 2 + 1, partial_mat_Fxc_b, aoinds, naos);
                }}

                omptimers[thread_id].stop("Fxc dist.");
            }}
        }}
    }}
    }}
    }}
    }}

    timer.stop("Total timing");
}}
'''


def _libxc_decls(exprs, indent: int = 24) -> str:
    """Declare every Libxc component the channels use, from flat indices.

    Read by flat index rather than by the host's spelling: the index is
    Libxc's own packing and cannot be misspelled, whereas VeloxChem
    names the sigma components a/c/b where Libxc packs them aa/ab/bb.
    """
    import sympy as sp
    ind = " " * indent
    wanted = set()
    for e in exprs.values():
        for t in e.free_symbols:
            base, _, idx = t.name.rpartition("_")
            if base in VLX_COMPONENTS and idx.isdigit():
                wanted.add((base, int(idx)))
    lines = []
    for base, idx in sorted(wanted):
        nm = f"{base}_{VLX_COMPONENTS[base][idx]}"
        lines.append(f"{ind}double {nm} = {base}[dim->{base} * g + {idx}];")
    return "\n".join(lines)


def emit_mgga_fxc_driver() -> str:
    """The complete open-shell meta-GGA fxc driver, ready to #include."""
    from ..engine.spin_kernel import fxc_channels_spin
    ch = fxc_channels_spin("mgga_tau")
    return DRIVER_MGGA_FXC.format(decls=_libxc_decls(ch),
                                  body=openshell_fxc_mgga(indent=24))

if __name__ == "__main__":
    main()
