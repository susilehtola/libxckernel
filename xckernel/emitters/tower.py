"""The derivative-tower interface of the compiled kernels.

Every spatially varying operand a host passes is a component of a
Cartesian derivative tower, named by its sorted axis string:

    <base>[_<spin>][_<pert>][_<axes>]

base   chi (basis functions), Dchi (density-contracted basis rows,
       Dchi[k,u,g] = sum_v D[u,v] chi[k,v,g]), rho, tau, or a vector
       component such as jpx (the x component of the paramagnetic current;
       a vector's component is part of its base, so that a trailing axis
       string always means a derivative);
spin   a | b (polarized channels);
pert   p1, p2, ... (perturbed fields of the response kernels);
axes   the derivative, axes sorted: '' (the value), x, xy, xyz, xxy, ...

rho_a_p1_xy is d^2/dxdy of the alpha perturbed density of perturbation 1.
Collocation is one array per tower, chi[k, u, g], with the components k in
the canonical order of ``components`` (that of PySCF's eval_ao(deriv=n)):
1, x, y, z, xx, xy, xz, yy, yz, zz, xxx, xxy, ... .

Combinations the engine works with -- the Laplacians lapl chi and
lapl rho, the Laplacian gradient d_d lapl chi -- are not operands: the
kernel forms them from the tower (basis-level ones once per grid block,
per-point ones once per call), so the contraction runs the same number of
GEMMs as with a host-supplied Laplacian.

This module is the single translation table between the engine's operand
symbols and the tower.
"""

from __future__ import annotations

import re
from itertools import combinations_with_replacement
from typing import Dict, List, Tuple

AXES = "xyz"
#: packed symmetric-tensor component order of the engine (hess_chi[k])
H6 = ("xx", "xy", "xz", "yy", "yz", "zz")

Combo = List[Tuple[str, int]]          # [(component axes, weight)]


def components(order: int) -> List[str]:
    """Axis strings of every derivative up to ``order``, canonical order."""
    out = []
    for k in range(order + 1):
        out += ["".join(c) for c in combinations_with_replacement(AXES, k)]
    return out


def ncomp(order: int) -> int:
    return (order + 1) * (order + 2) * (order + 3) // 6


def comp_index(axes: str) -> int:
    """Position of a derivative in the canonical tower order."""
    axes = "".join(sorted(axes))
    return components(len(axes)).index(axes)


def _lapl(prefix: str = "") -> Combo:
    return [("".join(sorted(prefix + a + a)), 1) for a in AXES]


# --- basis operands -------------------------------------------------------------

def basis_operand(code: str) -> Tuple[str, Combo]:
    """(tower array, combination of components) of an engine basis code."""
    m = re.fullmatch(r"(dchi|hess_chi|dlapl_chi)\[(\d)\]", code)
    if code == "chi":
        return "chi", [("", 1)]
    if m and m.group(1) == "dchi":
        return "chi", [(AXES[int(m.group(2))], 1)]
    if m and m.group(1) == "hess_chi":
        return "chi", [(H6[int(m.group(2))], 1)]
    if m and m.group(1) == "dlapl_chi":
        return "chi", _lapl(AXES[int(m.group(2))])
    if code == "lapl_chi":
        return "chi", _lapl()
    m = re.fullmatch(r"tchi\[([xyz]{3})\]", code)
    if m:
        return "chi", [(m.group(1), 1)]
    m = re.fullmatch(r"Uh\[([xyz]{2})\]", code)
    if m:
        return "Dchi", [(m.group(1), 1)]
    if code == "UT0":
        return "DTchi", [("", 1)]
    m = re.fullmatch(r"UT([123])", code)
    if m:
        return "DTchi", [(AXES[int(m.group(1)) - 1], 1)]
    if code == "UTL":
        return "DTchi", _lapl()
    m = re.fullmatch(r"UTh\[([xyz]{2})\]", code)
    if m:
        return "DTchi", [(m.group(1), 1)]
    if code == "U0":
        return "Dchi", [("", 1)]
    m = re.fullmatch(r"U([123])", code)
    if m:
        return "Dchi", [(AXES[int(m.group(1)) - 1], 1)]
    if code == "UL":
        return "Dchi", _lapl()
    raise ValueError(f"basis operand {code!r} has no tower form")


def is_component(combo: Combo) -> bool:
    return len(combo) == 1 and combo[0][1] == 1


# --- per-point operands -----------------------------------------------------------

def _name(base: str, spin, pert, axes: str) -> str:
    parts = [base] + [p for p in (spin, pert) if p] + ([axes] if axes else [])
    return "_".join(parts)


_SP = r"(?:_([ab]))?"
_PT = r"(?:_(p\d+))?"


def scalar_operand(name: str) -> List[Tuple[str, int]]:
    """Tower names and weights of an engine per-point operand (Libxc
    derivative arrays excluded): a single entry with weight 1 for a plain
    rename, several for a combination the kernel forms."""
    if name in ("w",) or re.fullmatch(r"inv_rho" + _SP, name):
        return [(name, 1)]
    m = re.fullmatch(r"grad_rho" + _SP + _PT + r"_([xyz])", name)
    if m:
        return [(_name("rho", m.group(1), m.group(2), m.group(3)), 1)]
    m = re.fullmatch(r"hess_rho" + _SP + _PT + r"_(xx|xy|xz|yy|yz|zz)", name)
    if m:
        return [(_name("rho", m.group(1), m.group(2), m.group(3)), 1)]
    m = re.fullmatch(r"(rho|tau)" + _SP + r"_(p\d+)", name)
    if m:
        return [(_name(m.group(1), m.group(2), m.group(3), ""), 1)]
    m = re.fullmatch(r"lapl_rho" + _SP + r"_(p\d+)", name)
    if m:
        return [(_name("rho", m.group(1), m.group(2), ax), wt)
                for ax, wt in _lapl()]
    m = re.fullmatch(r"d3rho" + _SP + r"_([xyz]{3})", name)
    if m:
        return [(_name("rho", m.group(1), None, m.group(2)), 1)]
    m = re.fullmatch(r"grad_tau" + _SP + r"_([xyz])", name)
    if m:
        return [(_name("tau", m.group(1), None, m.group(2)), 1)]
    m = re.fullmatch(r"grad_lapl_rho" + _SP + r"_([xyz])", name)
    if m:
        return [(_name("rho", m.group(1), None, ax), wt)
                for ax, wt in _lapl(m.group(2))]
    m = re.fullmatch(r"jpgrad" + _SP + r"_([xyz])_([xyz])", name)
    if m:
        return [(_name(f"jp{m.group(2)}", m.group(1), None, m.group(3)), 1)]
    m = re.fullmatch(r"jp" + _SP + _PT + r"_([xyz])", name)
    if m:
        return [(_name(f"jp{m.group(3)}", m.group(1), m.group(2), ""), 1)]
    raise ValueError(f"per-point operand {name!r} has no tower form")


class Layout:
    """The tower interface of one kernel: which arrays at which derivative
    order, the ordered per-point operand names, and the combinations the
    kernel forms.

    internal_fields: the engine's per-point field operands, in table order
    (fields first, as in scal_order); libxc: the derivative arrays."""

    def __init__(self, internal_fields: List[str], libxc: List[str],
                 basis_codes):
        self.internal_fields = list(internal_fields)
        self.libxc = list(libxc)
        self.field_map: Dict[str, List[Tuple[int, int]]] = {}
        abi: List[str] = []
        for n in self.internal_fields:
            entry = []
            for comp, wt in scalar_operand(n):
                if comp not in abi:
                    abi.append(comp)
                entry.append((abi.index(comp), wt))
            self.field_map[n] = entry
        self.fields = abi
        #: engine basis code -> (array, combo)
        self.basis = {c: basis_operand(c) for c in sorted(basis_codes)}
        self.orders: Dict[str, int] = {}
        for arr, combo in self.basis.values():
            k = max(len(ax) for ax, _ in combo)
            self.orders[arr] = max(self.orders.get(arr, 0), k)

    @property
    def scal_names(self) -> List[str]:
        return self.fields + self.libxc

    def derived_fields(self) -> List[str]:
        return [n for n in self.internal_fields
                if not (len(self.field_map[n]) == 1
                        and self.field_map[n][0][1] == 1)]

    def derived_basis(self) -> List[str]:
        return [c for c, (_, combo) in self.basis.items()
                if not is_component(combo)]

    def definitions(self) -> Dict[str, str]:
        """Human-readable definitions of the combinations formed inside."""
        out = {}
        for n in self.derived_fields():
            out[n] = " + ".join(
                (f"{wt}*" if wt != 1 else "") + self.fields[i]
                for i, wt in self.field_map[n])
        for c in self.derived_basis():
            arr, combo = self.basis[c]
            out[c] = " + ".join((f"{wt}*" if wt != 1 else "")
                                + f"{arr}_{ax}" for ax, wt in combo)
        return out


def translate_operands(layout: Layout, tower_ops: Dict) -> Dict:
    """Engine per-point operands from tower-named ones (for reference
    evaluation through the NumPy backend)."""
    out = {}
    for n in layout.internal_fields:
        out[n] = sum(wt * tower_ops[layout.fields[i]]
                     for i, wt in layout.field_map[n])
    return out


def internal_collocation(chi):
    """The engine's collocation arrays (chi, dchi, lapl_chi, hess_chi)
    from a tower chi[k, u, g] of order >= 2 (hess/lapl None below)."""
    n = chi.shape[0]
    c = lambda ax: chi[comp_index(ax)]
    import numpy as np
    out = {"chi": chi[0]}
    if n >= 4:
        out["dchi"] = np.stack([c(a) for a in AXES])
    if n >= 10:
        out["hess_chi"] = np.stack([c(h) for h in H6])
        out["lapl_chi"] = c("xx") + c("yy") + c("zz")
    return out
