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
# operand *code* for a perturbed gradient component, e.g. grad_rho_p1[0]
_PERT_GRAD_CODE = re.compile(r"^grad_rho_(?:[ab]_)?p\d+\[\d\]$")


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
    batch: bool = False               # perturbed operands carry a batch axis


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


def generate(ki: KernelIntegrand, func_name: str = "kernel",
             batch: bool = False) -> GeneratedFunction:
    """Generate einsum source for an integrand.

    With ``batch=True`` every perturbed-field operand carries a leading batch
    axis x (rho_pN: (nx,ng), grad_rho_pN: (nx,3,ng)) shared across all
    perturbation slots -- the i-th entries pair up, and the output gains a
    leading batch index (nx, nao, nao). This amortizes basis/ground-state work
    over many simultaneous perturbations (Dalton's NOSIM batching).
    """
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
        if batch:
            # add the batch axis to perturbed operands and the output
            in_subs, out_sub = subs.split("->")
            parts = in_subs.split(",")
            new_parts = []
            any_pert = False
            for code, sub in zip(codes, parts):
                if _PERT_SCALAR.match(code) or _PERT_SCALAR_SPIN.match(code) \
                        or _PERT_GRAD_CODE.match(code):
                    new_parts.append("x" + sub)
                    any_pert = True
                else:
                    new_parts.append(sub)
            if any_pert:
                subs = ",".join(new_parts) + "->x" + out_sub
            # grad operand indexing: grad_rho_p1[0] -> grad_rho_p1[:, 0]
            codes = [re.sub(r"^(grad_rho_(?:[ab]_)?p\d+)\[(\d)\]$",
                            r"\1[:, \2]", c) for c in codes]
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
    ]
    if batch:
        if not (pert_scalars or pert_grads):
            raise ValueError("batch=True requires perturbed-field operands")
        nx_src = pert_scalars[0] if pert_scalars \
            else f"grad_rho_{pert_grads[0]}"
        header += [f"    nx = {nx_src}.shape[0]",
                   f"    out = np.zeros((nx, {shape}))"]
    else:
        header += [f"    out = np.zeros(({shape}))"]
    source = "\n".join(header + lines + ["    return out", ""])

    return GeneratedFunction(
        name=func_name, source=source, out_indices=out_indices,
        libxc_args=libxc_args, uses_lapl_chi=("lapl" in uses),
        uses_grad_rho=("grad" in uses),
        uses_grad_rho_a=("grad_a" in uses), uses_grad_rho_b=("grad_b" in uses),
        pert_grads=pert_grads, pert_scalars=pert_scalars, batch=batch)


def generate_collapsed(ki: KernelIntegrand, func_name: str = "kernel",
                       batch: bool = False) -> GeneratedFunction:
    """Pattern-collapsed emission (the production lowering).

    Every monomial of a one-free-pair integrand factorizes as
    (basis factor at u) x (basis factor at v) x (per-point scalar), and only a
    handful of basis-pair patterns exist at ANY derivative order (chi*chi,
    chi*dchi_c, dchi_c*dchi_c', chi*lapl_chi, ...).  Grouping by pattern and
    factoring out the scalar yields the three-stage form every production code
    hand-writes: (A) pointwise coefficient vectors, (B) one GEMM-shaped
    distribute per pattern.  The order-4 GGA kernel collapses from 862 einsum
    terms to 16 patterns.

    Drop-in replacement for generate(): same parameters, same output,
    different (faster) body.  Only single-free-pair integrands (Fock/response
    kernels) are supported; multi-pair kernels keep the per-term path.
    """
    if len(ki.index_pairs) != 1:
        raise ValueError("pattern collapse requires exactly one free pair")
    (u_lbl, v_lbl), = ki.index_pairs

    terms = sp.Add.make_args(sp.expand(ki.expr))
    patterns: Dict[Tuple[str, str], sp.Expr] = {}
    libxc_used: Dict[str, None] = {}
    uses: set = set()

    for term in terms:
        coeff, rest = term.as_coeff_Mul()
        powers = rest.as_powers_dict() if rest != 1 else {}
        ufac = vfac = None
        scalar = sp.Integer(1) * coeff
        for base, exp in powers.items():
            e = int(exp)
            op, kind = _classify(base.name)
            if kind == "basis":
                lbl = op.subscript[:-1]      # 'u' or 'v'
                if e != 1:
                    raise ValueError(f"unexpected basis power {base}**{e}")
                if lbl == u_lbl:
                    ufac = op.code
                elif lbl == v_lbl:
                    vfac = op.code
                else:
                    raise ValueError(f"unknown basis label in {base}")
                if op.code == "lapl_chi":
                    uses.add("lapl")
            else:
                if kind == "libxc":
                    libxc_used.setdefault(base.name, None)
                elif kind == "grad":
                    uses.add("grad")
                elif kind in ("grad_a", "grad_b"):
                    uses.add(kind)
                elif kind.startswith(("pgrad:", "pscalar:")):
                    uses.add(kind)
                scalar *= base ** e
        if ufac is None or vfac is None:
            raise ValueError(f"term without both basis factors: {term}")
        key = (ufac, vfac)
        patterns[key] = patterns.get(key, sp.Integer(0)) + scalar

    libxc_args = sorted(libxc_used, key=_libxc_sort_key)
    pert_grads = sorted(u.split(":", 1)[1] for u in uses
                        if u.startswith("pgrad:"))
    pert_scalars = sorted(u.split(":", 1)[1] for u in uses
                          if u.startswith("pscalar:"))

    params = ["w", "chi", "dchi"]
    if "lapl" in uses:
        params.append("lapl_chi")
    if "grad" in uses:
        params.append("grad_rho")
    if "grad_a" in uses:
        params.append("grad_rho_a")
    if "grad_b" in uses:
        params.append("grad_rho_b")
    params += [f"grad_rho_{lbl}" for lbl in pert_grads]
    params += pert_scalars
    params += libxc_args

    # map scalar symbols to python expressions for coefficient printing
    def scalar_code(name: str) -> str:
        op, kind = _classify(name)
        code = op.code
        if batch and kind.startswith("pgrad:"):
            code = re.sub(r"\[(\d)\]$", r"[:, \1]", code)
        return code

    lines: List[str] = []
    for k, ((ufac, vfac), cexpr) in enumerate(sorted(patterns.items())):
        sub = {s: sp.Symbol(scalar_code(s.name))
               for s in cexpr.free_symbols}
        code = str(sp.expand(cexpr.subs(sub, simultaneous=True)))
        lines.append(f"    c = {code}")
        if batch:
            lines.append(f"    out += np.einsum('ug,xg,vg->xuv', {ufac}, "
                         f"c, {vfac}, optimize=True)")
        else:
            lines.append(f"    out += ({ufac} * c) @ {vfac}.T")

    header = [
        f"def {func_name}({', '.join(params)}):",
        f"    # pattern-collapsed: {len(patterns)} patterns "
        f"from {len(terms)} terms",
        f"    nao = chi.shape[0]",
    ]
    if batch:
        if not (pert_scalars or pert_grads):
            raise ValueError("batch=True requires perturbed-field operands")
        nx_src = pert_scalars[0] if pert_scalars \
            else f"grad_rho_{pert_grads[0]}"
        header += [f"    nx = {nx_src}.shape[0]",
                   f"    out = np.zeros((nx, nao, nao))"]
    else:
        header += ["    out = np.zeros((nao, nao))"]
    source = "\n".join(header + lines + ["    return out", ""])

    return GeneratedFunction(
        name=func_name, source=source, out_indices=u_lbl + v_lbl,
        libxc_args=libxc_args, uses_lapl_chi=("lapl" in uses),
        uses_grad_rho=("grad" in uses),
        uses_grad_rho_a=("grad_a" in uses), uses_grad_rho_b=("grad_b" in uses),
        pert_grads=pert_grads, pert_scalars=pert_scalars, batch=batch)


def compile_function(gen: GeneratedFunction):
    """exec the generated source and return the live callable."""
    import numpy as np
    ns: Dict[str, object] = {"np": np}
    exec(gen.source, ns)
    return ns[gen.name]
