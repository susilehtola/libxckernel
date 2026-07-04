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
_GRAD_SPIN = re.compile(r"^grad_rho_([ab])_([xyz])$")
_LIBXC_SPIN = re.compile(r"^v\w+_\d+$")
# perturbed fields of the response contraction engine (labels p1, p2, ...)
_PERT_SCALAR = re.compile(r"^(?:rho|lapl_rho|tau)_(p\d+)$")
_PERT_GRAD = re.compile(r"^grad_rho_(p\d+)_([xyz])$")
# spin-resolved perturbed fields (rho_a_p1, grad_rho_a_p1_x, ...)
_PERT_SCALAR_SPIN = re.compile(r"^(?:rho|lapl_rho|tau)_([ab])_(p\d+)$")
_PERT_GRAD_SPIN = re.compile(r"^grad_rho_([ab])_(p\d+)_([xyz])$")


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
    m = _GRAD_SPIN.match(name)
    if m:
        return Operand(f"grad_rho_{m.group(1)}[{_AX[m.group(2)]}]", "g"), \
            f"grad_{m.group(1)}"
    m = _PERT_GRAD.match(name)
    if m:
        return Operand(f"grad_rho_{m.group(1)}[{_AX[m.group(2)]}]", "g"), \
            f"pgrad:{m.group(1)}"
    m = _PERT_GRAD_SPIN.match(name)
    if m:
        lbl = f"{m.group(1)}_{m.group(2)}"
        return Operand(f"grad_rho_{lbl}[{_AX[m.group(3)]}]", "g"), f"pgrad:{lbl}"
    m = _PERT_SCALAR.match(name) or _PERT_SCALAR_SPIN.match(name)
    if m:
        # each perturbed scalar field is its own (ng,) parameter
        return Operand(name, "g"), f"pscalar:{name}"
    if name == "w":
        return Operand("w", "g"), "weight"
    if name in LIBXC_MULTISET or _LIBXC_SPIN.match(name):
        # each Libxc derivative component is passed as its own (ng,) parameter.
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
    uses_grad_rho_a: bool = False
    uses_grad_rho_b: bool = False
    pert_grads: List[str] = None      # perturbation labels needing grad_rho_pN
    pert_scalars: List[str] = None    # perturbed scalar field parameter names


def _term_einsum(term: sp.Expr, out_indices: str) -> Tuple[str, float,
                                                           List[str], bool, bool]:
    """Build the einsum ('subs', [operand codes]) for one monomial term."""
    coeff, rest = term.as_coeff_Mul()
    powers = rest.as_powers_dict() if rest != 1 else {}

    subs: List[str] = []
    codes: List[str] = []
    scalar: List[str] = []       # Libxc derivative components: own (ng,) params
    uses: set = set()

    for base, exp in powers.items():
        e = int(exp)
        op, kind = _classify(base.name)
        if kind == "libxc":
            scalar.append(base.name)
        elif kind == "grad":
            uses.add("grad")
        elif kind in ("grad_a", "grad_b"):
            uses.add(kind)
        elif kind.startswith(("pgrad:", "pscalar:")):
            uses.add(kind)
        elif kind == "basis" and op.code == "lapl_chi":
            uses.add("lapl")
        for _ in range(e):
            subs.append(op.subscript)
            codes.append(op.code)

    einsum_str = ",".join(subs) + "->" + out_indices
    return einsum_str, float(coeff), codes, scalar, uses


def _libxc_sort_key(n: str):
    if n in LIBXC_MULTISET:
        return (len(LIBXC_MULTISET[n]), n)
    # spin component name '<array>_<comp>': order by derivative order then name
    return (int(n[1]) if n[1].isdigit() else 1, n)


def generate(ki: KernelIntegrand, func_name: str = "kernel") -> GeneratedFunction:
    out_indices = "".join(lbl for pair in ki.index_pairs for lbl in pair)

    terms = sp.Add.make_args(sp.expand(ki.expr))
    lines: List[str] = []
    libxc_used: Dict[str, None] = {}
    uses: set = set()

    for term in terms:
        subs, coeff, codes, libxc, term_uses = _term_einsum(term, out_indices)
        for name in libxc:
            libxc_used.setdefault(name, None)
        uses |= term_uses
        operands = ", ".join(codes)
        c = "" if coeff == 1.0 else f"{coeff!r} * "
        lines.append(f"    out += {c}np.einsum('{subs}', {operands})")

    libxc_args = sorted(libxc_used, key=_libxc_sort_key)

    params = ["w", "chi", "dchi"]
    if "lapl" in uses:
        params.append("lapl_chi")
    if "grad" in uses:
        params.append("grad_rho")
    if "grad_a" in uses:
        params.append("grad_rho_a")
    if "grad_b" in uses:
        params.append("grad_rho_b")
    # perturbed fields, grouped per label in sorted order: grad first (3,ng),
    # then the scalar fields (ng,) sorted by name
    pert_grads = sorted(u.split(":", 1)[1] for u in uses if u.startswith("pgrad:"))
    pert_scalars = sorted(u.split(":", 1)[1] for u in uses if u.startswith("pscalar:"))
    params += [f"grad_rho_{lbl}" for lbl in pert_grads]
    params += pert_scalars
    params += libxc_args

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
        libxc_args=libxc_args, uses_lapl_chi=("lapl" in uses),
        uses_grad_rho=("grad" in uses),
        uses_grad_rho_a=("grad_a" in uses), uses_grad_rho_b=("grad_b" in uses),
        pert_grads=pert_grads, pert_scalars=pert_scalars)


def compile_function(gen: GeneratedFunction):
    """exec the generated source and return the live callable."""
    import numpy as np
    ns: Dict[str, object] = {"np": np}
    exec(gen.source, ns)
    return ns[gen.name]
