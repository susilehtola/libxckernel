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


def main() -> None:
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "QRF"
    family = sys.argv[2] if len(sys.argv) > 2 else "gga"
    print(emit_branch(mode, family))


if __name__ == "__main__":
    main()
