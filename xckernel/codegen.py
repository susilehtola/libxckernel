"""Generate einsum-based Python from a symbolic kernel integrand.

A kernel integrand is a sum of monomials over a fixed vocabulary of symbols:

* basis factors tagged by a free index label (u, v, t, s, ...):
  ``chi_u``, ``dchi_u_x``, ``lapl_chi_u`` -- arrays indexed (label, grid);
* per-grid fields ``grad_rho_{x,y,z}`` and the grid weight ``w`` -- arrays (grid,);
* Libxc derivative outputs ``vrho``, ``vsigma``, ``v2rho2``, ... -- arrays (grid,).

Each monomial therefore contracts to one ``np.einsum`` over the grid index with
one output index per free label.  We emit a standalone Python function that sums
these einsums.  Operands:

    w         (ng,)
    chi       (nao, ng)
    dchi      (3, nao, ng)
    lapl_chi  (nao, ng)          [only if used]
    grad_rho  (3, ng)            [only if used]
    <vname>   (ng,)              one parameter per Libxc derivative used

The generated code is exactly the "AD backend for Libxc" contraction a host code
(e.g. PySCF) would run on the grid.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import sympy as sp

from .basis import AXES
from .deriv import LIBXC_MULTISET
from .kernel import KernelIntegrand

_AX = {ax: i for i, ax in enumerate(AXES)}
_CHI = re.compile(r"^chi_(\w+)$")
_DCHI = re.compile(r"^dchi_(\w+)_([xyz])$")
_LAPL = re.compile(r"^lapl_chi_(\w+)$")
_GRAD = re.compile(r"^grad_rho_([xyz])$")


@dataclass
class Operand:
    """One factor of a monomial: how to write it in an einsum call."""

    code: str        # python expression for the array, e.g. "chi" or "dchi[0]"
    subscript: str   # einsum subscript, e.g. "ug" or "g"


def _classify(name: str) -> Tuple[Operand, str]:
    """Map a symbol name to (Operand, kind). kind in {basis, grad, weight, libxc}."""
    m = _CHI.match(name)
    if m:
        return Operand("chi", f"{m.group(1)}g"), "basis"
    m = _DCHI.match(name)
    if m:
        return Operand(f"dchi[{_AX[m.group(2)]}]", f"{m.group(1)}g"), "basis"
    m = _LAPL.match(name)
    if m:
        return Operand("lapl_chi", f"{m.group(1)}g"), "basis"
    m = _GRAD.match(name)
    if m:
        return Operand(f"grad_rho[{_AX[m.group(1)]}]", "g"), "grad"
    if name == "w":
        return Operand("w", "g"), "weight"
    if name in LIBXC_MULTISET:
        return Operand(name, "g"), "libxc"
    raise ValueError(f"unrecognised symbol in integrand: {name}")


@dataclass
class GeneratedFunction:
    name: str
    source: str
    out_indices: str            # e.g. "uv" or "uvts"
    libxc_args: List[str]       # Libxc derivative arrays required, in order
    uses_lapl_chi: bool
    uses_grad_rho: bool


def _term_einsum(term: sp.Expr, out_indices: str) -> Tuple[str, float,
                                                           List[str], bool, bool]:
    """Build the einsum ('subs', [operand codes]) for one monomial term."""
    coeff, rest = term.as_coeff_Mul()
    powers = rest.as_powers_dict() if rest != 1 else {}

    subs: List[str] = []
    codes: List[str] = []
    libxc: List[str] = []
    uses_lapl = uses_grad = False

    for base, exp in powers.items():
        e = int(exp)
        op, kind = _classify(base.name)
        if kind == "libxc":
            libxc.append(base.name)
        elif kind == "grad":
            uses_grad = True
        elif kind == "basis" and op.code == "lapl_chi":
            uses_lapl = True
        for _ in range(e):
            subs.append(op.subscript)
            codes.append(op.code)

    einsum_str = ",".join(subs) + "->" + out_indices
    return einsum_str, float(coeff), codes, libxc, (uses_lapl, uses_grad)


def generate(ki: KernelIntegrand, func_name: str = "kernel") -> GeneratedFunction:
    out_indices = "".join(lbl for pair in ki.index_pairs for lbl in pair)
    nidx = len(out_indices)

    terms = sp.Add.make_args(sp.expand(ki.expr))
    lines: List[str] = []
    libxc_used: Dict[str, None] = {}
    uses_lapl = uses_grad = False

    for term in terms:
        subs, coeff, codes, libxc, (ul, ug) = _term_einsum(term, out_indices)
        for name in libxc:
            libxc_used.setdefault(name, None)
        uses_lapl = uses_lapl or ul
        uses_grad = uses_grad or ug
        # Reorder operand codes to match the subscript order already built by
        # _term_einsum (they are parallel lists, so just zip).
        operands = ", ".join(codes)
        c = "" if coeff == 1.0 else f"{coeff!r} * "
        lines.append(f"    out += {c}np.einsum('{subs}', {operands})")

    # Libxc args in canonical (order, name) sorting for a stable signature.
    libxc_args = sorted(libxc_used, key=lambda n: (len(LIBXC_MULTISET[n]), n))

    params = ["w", "chi", "dchi"]
    if uses_lapl:
        params.append("lapl_chi")
    if uses_grad:
        params.append("grad_rho")
    params += libxc_args

    idx_letters = ", ".join(f"n{c}" for c in out_indices)
    shape = ", ".join("nao" for _ in out_indices)
    header = [
        f"def {func_name}({', '.join(params)}):",
        f"    # F/kernel element with free indices ({', '.join(out_indices)})",
        f"    nao = chi.shape[0]",
        f"    out = np.zeros(({shape}))",
    ]
    source = "\n".join(header + lines + ["    return out", ""])

    return GeneratedFunction(
        name=func_name, source=source, out_indices=out_indices,
        libxc_args=libxc_args, uses_lapl_chi=uses_lapl, uses_grad_rho=uses_grad)


def compile_function(gen: GeneratedFunction):
    """exec the generated source and return the live callable."""
    import numpy as np
    ns: Dict[str, object] = {"np": np}
    exec(gen.source, ns)
    return ns[gen.name]
