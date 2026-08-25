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

from ..inputs.basis import (AXES, PROLATE, PROLATE_PUREM, RADIAL,
                            SPHERICAL)
from ..engine.deriv import LIBXC_MULTISET
from ..engine.kernel import KernelIntegrand

_AX = {ax: i for i, ax in enumerate(AXES)}

#: Symbols a curvilinear coordinate system introduces into the integrand:
#: the scale factors themselves and the angular-momentum factor of an
#: l-blocked density matrix. They are per-point host data, not fields
#: derived from the density.
_GEOMETRY_OPERANDS = {"r", "sin_theta", "l_factor", "m_factor",
                      "scale_mu", "scale_nu", "scale_phi"}

_AX_ANY = dict(_AX)
for _c in (SPHERICAL, PROLATE, RADIAL, PROLATE_PUREM):
    for _i, _a in enumerate(_c.axes):
        _AX_ANY.setdefault(_a, _i)

#: alternation of every known axis name, longest first so that
#: 'theta' is preferred over any prefix of it
_AXRE = "|".join(sorted(_AX_ANY, key=len, reverse=True))

#: Axis index for every coordinate system the generator knows. Names are
#: unique across systems, so one flat table suffices and no context has to
#: be threaded into the classifier.

#: packed symmetric-tensor component -> index (xx,xy,xz,yy,yz,zz)
_H6 = {"xx": 0, "xy": 1, "xz": 2, "yy": 3, "yz": 4, "zz": 5}
_CHI = re.compile(r"^chi_(\w+)$")
#: Basis-function derivative component. The axis suffix is whatever the
#: coordinate system calls it -- x/y/z in Cartesian, r/theta/phi in
#: spherical -- so the pattern lists the KNOWN axis names rather than any
#: token, which would otherwise swallow operands such as ``dchi_g_v``
#: whose trailing label is a perturbation index, not an axis.
_DCHI = re.compile(rf"^dchi_(\w+?)_({_AXRE})$")
_LAPL = re.compile(r"^lapl_chi_(\w+)$")
_HESS_CHI = re.compile(r"^hess_chi_(\w+?)_(xx|xy|xz|yy|yz|zz)$")
_GRAD = re.compile(rf"^grad_rho_({_AXRE})$")
_GRAD_SPIN = re.compile(rf"^grad_rho_([ab])_({_AXRE})$")
_LIBXC_SPIN = re.compile(r"^v\w+_\d+$")
# perturbed fields of the response contraction engine (labels p1, p2, ...)
_PERT_SCALAR = re.compile(r"^(?:rho|lapl_rho|tau)_(p\d+)$")
_PERT_GRAD = re.compile(rf"^grad_rho_(p\d+)_({_AXRE})$")
# spin-resolved perturbed fields (rho_a_p1, grad_rho_a_p1_x, ...)
_PERT_SCALAR_SPIN = re.compile(r"^(?:rho|lapl_rho|tau)_([ab])_(p\d+)$")
_PERT_GRAD_SPIN = re.compile(r"^grad_rho_([ab])_(p\d+)_([xyz])$")
# current-density ingredients: jp vector (gs + perturbed) and the inv_rho field
_JP = re.compile(r"^jp_([xyz])$")
_JP_SPIN = re.compile(r"^jp_([ab])_([xyz])$")
_PERT_JP = re.compile(r"^jp_(p\d+)_([xyz])$")
_PERT_JP_SPIN = re.compile(r"^jp_([ab])_(p\d+)_([xyz])$")
# density-Hessian ingredients: the packed 6-component tensor (gs + perturbed)
_HRHO = re.compile(r"^hess_rho_(xx|xy|xz|yy|yz|zz)$")
_HRHO_SPIN = re.compile(r"^hess_rho_([ab])_(xx|xy|xz|yy|yz|zz)$")
_PERT_HRHO = re.compile(r"^hess_rho_(p\d+)_(xx|xy|xz|yy|yz|zz)$")
_PERT_HRHO_SPIN = re.compile(r"^hess_rho_([ab])_(p\d+)_(xx|xy|xz|yy|yz|zz)$")
_GS_SCALAR = re.compile(
    r"^(inv_rho(_[ab])?|drho_g|dtau_g|drho_[ab]_g|dtau_[ab]_g)$")
# London/GIAO operands: center-scaled collocations and grid coordinates
_RCHI = re.compile(r"^Rchi_(\w+)_([xyz])$")
_RDCHI = re.compile(r"^Rdchi_(\w+)_([xyz])_([xyz])$")
_RLAPL = re.compile(r"^Rlapl_chi_(\w+)_([xyz])$")
_RG = re.compile(r"^rg_([xyz])$")
# geometric operands: spatial-gradient basis factors (dchi_g, ddchi_g), their
# atom-masked fixed-grid analogues (dchi_gA, ddchi_gA), and the direction-
# resolved density-Hessian row dgrad_rho_g
_DCHI_G = re.compile(r"^dchi_(g|gA|gB)_(\w+)$")
_DDCHI_G = re.compile(r"^ddchi_(g|gA|gB)_(\w+)_([xyz])$")
# second-displacement masked collocations (both nuclear derivatives on the
# same function) and density-contracted collocation rows
_D2CHI_G2 = re.compile(r"^d2chi_g2_(\w+)$")
_D3CHI_G2 = re.compile(r"^d3chi_g2_(\w+)_([xyz])$")
_UROW = re.compile(r"^U(0|[123])_(\w+)$")
# the local density-matrix pair factor (two free labels)
_DPAIR = re.compile(r"^D_(\w+)_(\w+)$")
_DGRAD_G = re.compile(r"^dgrad_rho_g_([xyz])$")
_DGRAD_G_SPIN = re.compile(r"^dgrad_rho_([ab])_g_([xyz])$")
# operand *code* for a perturbed vector/tensor component, e.g. grad_rho_p1[0]
_PERT_GRAD_CODE = re.compile(
    r"^(?:grad_rho|jp|hess_rho)_(?:[ab]_)?p\d+\[\d\]$")


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
        return Operand(f"dchi[{_AX_ANY[m.group(2)]}]",
                       f"{m.group(1)}g"), "basis"
    m = _LAPL.match(name)
    if m:
        return Operand("lapl_chi", f"{m.group(1)}g"), "basis"
    m = _HESS_CHI.match(name)
    if m:
        return Operand(f"hess_chi[{_H6[m.group(2)]}]",
                       f"{m.group(1)}g"), "basis"
    m = _RCHI.match(name)
    if m:
        return Operand(f"Rchi[{_AX_ANY[m.group(2)]}]", f"{m.group(1)}g"), "basis"
    m = _RDCHI.match(name)
    if m:
        return Operand(f"Rdchi[{_AX_ANY[m.group(2)]}][{_AX_ANY[m.group(3)]}]",
                       f"{m.group(1)}g"), "basis"
    m = _RLAPL.match(name)
    if m:
        return Operand(f"Rlapl_chi[{_AX_ANY[m.group(2)]}]",
                       f"{m.group(1)}g"), "basis"
    m = _RG.match(name)
    if m:
        return Operand(f"rg[{_AX_ANY[m.group(1)]}]", "g"), "gscalar:rg"
    m = _DDCHI_G.match(name)
    if m:
        return Operand(f"ddchi_{m.group(1)}[{_AX_ANY[m.group(3)]}]",
                       f"{m.group(2)}g"), "basis"
    m = _DCHI_G.match(name)
    if m:
        return Operand(f"dchi_{m.group(1)}", f"{m.group(2)}g"), "basis"
    m = _D2CHI_G2.match(name)
    if m:
        return Operand("d2chi_g2", f"{m.group(1)}g"), "basis"
    m = _D3CHI_G2.match(name)
    if m:
        return Operand(f"d3chi_g2[{_AX_ANY[m.group(2)]}]", f"{m.group(1)}g"), "basis"
    m = _UROW.match(name)
    if m:
        return Operand(f"U{m.group(1)}", f"{m.group(2)}g"), "basis"
    m = _DPAIR.match(name)
    if m:
        return Operand("Dloc", f"{m.group(1)}{m.group(2)}"), "dpair"
    m = _DGRAD_G.match(name)
    if m:
        return Operand(f"dgrad_rho_g[{_AX_ANY[m.group(1)]}]", "g"), "dgrad_g"
    m = _DGRAD_G_SPIN.match(name)
    if m:
        return Operand(f"dgrad_rho_{m.group(1)}_g[{_AX_ANY[m.group(2)]}]", "g"), \
            f"dgrad_g_{m.group(1)}"
    m = None if (_GRAD_SPIN.match(name) or _PERT_GRAD.match(name)) \
        else _GRAD.match(name)
    if m:
        return Operand(f"grad_rho[{_AX_ANY[m.group(1)]}]", "g"), "grad"
    m = _GRAD_SPIN.match(name)
    if m:
        return Operand(f"grad_rho_{m.group(1)}[{_AX_ANY[m.group(2)]}]", "g"), \
            f"grad_{m.group(1)}"
    m = _PERT_GRAD.match(name)
    if m:
        return Operand(f"grad_rho_{m.group(1)}[{_AX_ANY[m.group(2)]}]", "g"), \
            f"pgrad:{m.group(1)}"
    m = _PERT_GRAD_SPIN.match(name)
    if m:
        lbl = f"{m.group(1)}_{m.group(2)}"
        return Operand(f"grad_rho_{lbl}[{_AX_ANY[m.group(3)]}]", "g"), f"pgrad:{lbl}"
    m = _JP.match(name)
    if m:
        return Operand(f"jp[{_AX_ANY[m.group(1)]}]", "g"), "jp"
    m = _JP_SPIN.match(name)
    if m:
        return Operand(f"jp_{m.group(1)}[{_AX_ANY[m.group(2)]}]", "g"), \
            f"jp_{m.group(1)}"
    m = _PERT_JP.match(name)
    if m:
        return Operand(f"jp_{m.group(1)}[{_AX_ANY[m.group(2)]}]", "g"), \
            f"pjp:{m.group(1)}"
    m = _PERT_JP_SPIN.match(name)
    if m:
        lbl = f"{m.group(1)}_{m.group(2)}"
        return Operand(f"jp_{lbl}[{_AX_ANY[m.group(3)]}]", "g"), f"pjp:{lbl}"
    m = _HRHO.match(name)
    if m:
        return Operand(f"hess_rho[{_H6[m.group(1)]}]", "g"), "hrho"
    m = _HRHO_SPIN.match(name)
    if m:
        return Operand(f"hess_rho_{m.group(1)}[{_H6[m.group(2)]}]", "g"), \
            f"hrho_{m.group(1)}"
    m = _PERT_HRHO.match(name)
    if m:
        return Operand(f"hess_rho_{m.group(1)}[{_H6[m.group(2)]}]", "g"), \
            f"phrho:{m.group(1)}"
    m = _PERT_HRHO_SPIN.match(name)
    if m:
        lbl = f"{m.group(1)}_{m.group(2)}"
        return Operand(f"hess_rho_{lbl}[{_H6[m.group(3)]}]", "g"), \
            f"phrho:{lbl}"
    m = _GS_SCALAR.match(name)
    if m:
        # ground-state scalar field passed as its own (ng,) parameter
        return Operand(name, "g"), f"gscalar:{name}"
    m = _PERT_SCALAR.match(name) or _PERT_SCALAR_SPIN.match(name)
    if m:
        # each perturbed scalar field is its own (ng,) parameter
        return Operand(name, "g"), f"pscalar:{name}"
    if name == "w":
        return Operand("w", "g"), "weight"
    if name in _GEOMETRY_OPERANDS:
        # Coordinate-system data the host supplies per grid point: the
        # Lame factors and, for a density matrix blocked by angular
        # momentum, the l(l+1) of the block being contracted.
        return Operand(name, "g"), f"geometry:{name}"
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
    part: str = None                  # split-storage output part, if any
    n_patterns: int = 0               # collapsed basis-pair patterns
    n_products: int = 0               # emitted matrix products after fusion


def _iter_terms(ki: KernelIntegrand):
    """Yield (coeff, [(symbol, int_exp), ...]) monomials from an integrand,
    straight from its Poly dict when present (avoiding expression
    materialization), else from the expanded expression."""
    poly = getattr(ki, "poly", None)
    if poly is not None:
        for key, coeff in poly.items():
            yield coeff, list(key)
        return
    for term in sp.Add.make_args(sp.expand(ki.expr)):
        coeff, rest = term.as_coeff_Mul()
        powers = rest.as_powers_dict() if rest != 1 else {}
        yield coeff, [(s, int(e)) for s, e in powers.items()]


def _term_einsum(coeff, powers, out_indices: str) -> Tuple[str, float,
                                                           List[str], bool, bool]:
    """Build the einsum ('subs', [operand codes]) for one monomial term."""
    subs: List[str] = []
    codes: List[str] = []
    scalar: List[str] = []       # Libxc derivative components: own (ng,) params
    uses: set = set()

    for base, e in powers:
        op, kind = _classify(base.name)
        if kind == "libxc":
            scalar.append(base.name)
        elif kind == "grad":
            uses.add("grad")
        elif kind in ("grad_a", "grad_b", "jp", "jp_a", "jp_b",
                      "hrho", "hrho_a", "hrho_b", "dgrad_g"):
            uses.add(kind)
        elif kind.startswith(("pgrad:", "pscalar:", "pjp:", "phrho:",
                              "gscalar:")):
            uses.add(kind)
        elif kind == "basis" and op.code == "lapl_chi":
            uses.add("lapl")
        elif kind == "basis" and op.code.startswith("hess_chi"):
            uses.add("hess")
        elif kind == "basis" and op.code.startswith(
                ("Rchi", "Rdchi", "Rlapl_chi")):
            uses.add(op.code.split("[", 1)[0])
        elif kind == "basis" and op.code.startswith(("dchi_g", "ddchi_g")):
            uses.add(op.code.split("[", 1)[0])
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

    lines: List[str] = []
    libxc_used: Dict[str, None] = {}
    uses: set = set()

    for tcoeff, tpowers in _iter_terms(ki):
        subs, coeff, codes, libxc, term_uses = _term_einsum(
            tcoeff, tpowers, out_indices)
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
            codes = [re.sub(
                r"^((?:grad_rho|jp|hess_rho)_(?:[ab]_)?p\d+)\[(\d)\]$",
                r"\1[:, \2]", c) for c in codes]
        operands = ", ".join(codes)
        c = "" if coeff == 1.0 else f"{coeff!r} * "
        lines.append(f"    out += {c}np.einsum('{subs}', {operands})")

    libxc_args = sorted(libxc_used, key=_libxc_sort_key)

    params = ["w", "chi", "dchi"]
    if "lapl" in uses:
        params.append("lapl_chi")
    if "hess" in uses:
        params.append("hess_chi")
    for b in ("Rchi", "Rdchi", "Rlapl_chi"):
        if b in uses:
            params.append(b)
    for b in ("dchi_g", "ddchi_g", "dchi_gA", "ddchi_gA"):
        if b in uses:
            params.append(b)
    if "grad" in uses:
        params.append("grad_rho")
    if "grad_a" in uses:
        params.append("grad_rho_a")
    if "grad_b" in uses:
        params.append("grad_rho_b")
    # perturbed fields, grouped per label in sorted order: grad first (3,ng),
    # then the scalar fields (ng,) sorted by name
    if "jp" in uses:
        params.append("jp")
    for sk in ("jp_a", "jp_b"):
        if sk in uses:
            params.append(sk)
    if "hrho" in uses:
        params.append("hess_rho")
    for sk in ("hrho_a", "hrho_b"):
        if sk in uses:
            params.append(f"hess_rho_{sk[-1]}")
    if "dgrad_g" in uses:
        params.append("dgrad_rho_g")
    params += sorted(u.split(":", 1)[1] for u in uses
                     if u.startswith("gscalar:"))
    pert_grads = sorted(u.split(":", 1)[1] for u in uses if u.startswith("pgrad:"))
    pert_scalars = sorted(u.split(":", 1)[1] for u in uses if u.startswith("pscalar:"))
    pert_jps = sorted(u.split(":", 1)[1] for u in uses if u.startswith("pjp:"))
    pert_hrhos = sorted(u.split(":", 1)[1] for u in uses if u.startswith("phrho:"))
    params += [f"grad_rho_{lbl}" for lbl in pert_grads]
    params += [f"jp_{lbl}" for lbl in pert_jps]
    params += [f"hess_rho_{lbl}" for lbl in pert_hrhos]
    params += pert_scalars
    # coordinate-system data (scale factors, the l(l+1) of the block)
    params += sorted(u.split(":", 1)[1] for u in uses
                     if u.startswith("geometry:"))
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


@dataclass
class CollapsedKernel:
    """Structured pattern-collapsed form, consumed by all emitters.

    patterns: sorted list of (ufac_code, vfac_code, monomials) with
    monomials = list of (float_coeff, ((scalar_name, exp), ...)).
    """
    u_lbl: str
    v_lbl: str
    patterns: List[Tuple[str, str, List[Tuple[float, Tuple]]]]
    params: List[str]
    libxc_args: List[str]
    pert_grads: List[str]
    pert_scalars: List[str]
    uses: set
    n_terms: int


def collapse(ki: KernelIntegrand) -> CollapsedKernel:
    """Group integrand monomials by basis-pair pattern, factoring out the
    per-point scalar coefficient (the production three-stage lowering)."""
    if len(ki.index_pairs) != 1:
        raise ValueError("pattern collapse requires exactly one free pair")
    (u_lbl, v_lbl), = ki.index_pairs

    # patterns: (ufac, vfac) -> {scalar-monomial: coefficient}
    patterns: Dict[Tuple[str, str], Dict] = {}
    libxc_used: Dict[str, None] = {}
    uses: set = set()
    n_terms = 0

    for coeff, powers in _iter_terms(ki):
        n_terms += 1
        ufac = vfac = None
        smono: List[Tuple[sp.Symbol, int]] = []
        for base, e in powers:
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
                elif op.code.startswith("hess_chi"):
                    uses.add("hess")
                elif op.code.startswith(("Rchi", "Rdchi", "Rlapl_chi")):
                    uses.add(op.code.split("[", 1)[0])
                elif op.code.startswith(("dchi_g", "ddchi_g")):
                    uses.add(op.code.split("[", 1)[0])
            else:
                if kind == "libxc":
                    libxc_used.setdefault(base.name, None)
                elif kind == "grad":
                    uses.add("grad")
                elif kind in ("grad_a", "grad_b", "jp", "jp_a", "jp_b",
                      "hrho", "hrho_a", "hrho_b", "dgrad_g"):
                    uses.add(kind)
                elif kind.startswith(("pgrad:", "pscalar:", "pjp:", "phrho:",
                                      "gscalar:", "geometry:")):
                    uses.add(kind)
                smono.append((base, e))
        if ufac is None or vfac is None:
            raise ValueError(f"term without both basis factors: {powers}")
        # The mixing contract: every monomial must carry EXACTLY ONE
        # functional-derivative factor (to first power), so kernels are
        # jointly linear in the derivative arrays and hosts may pass
        # coefficient-mixed (superfunctional) arrays. Provable from the
        # tower structure; enforced here so extensions cannot break it.
        n_deriv = sum(e for s, e in smono
                      if _classify(s.name)[1] == "libxc")
        if n_deriv != 1:
            raise ValueError(
                f"monomial with {n_deriv} functional-derivative factors "
                f"violates the linear-mixing contract: {powers}")
        key = (ufac, vfac)
        skey = tuple(sorted(smono, key=lambda p: p[0].name))
        pat = patterns.setdefault(key, {})
        pat[skey] = pat.get(skey, sp.Integer(0)) + coeff

    libxc_args = sorted(libxc_used, key=_libxc_sort_key)
    pert_grads = sorted(u.split(":", 1)[1] for u in uses
                        if u.startswith("pgrad:"))
    pert_scalars = sorted(u.split(":", 1)[1] for u in uses
                          if u.startswith("pscalar:"))
    pert_jps = sorted(u.split(":", 1)[1] for u in uses
                      if u.startswith("pjp:"))
    pert_hrhos = sorted(u.split(":", 1)[1] for u in uses
                        if u.startswith("phrho:"))

    params = ["w", "chi", "dchi"]
    if "lapl" in uses:
        params.append("lapl_chi")
    if "hess" in uses:
        params.append("hess_chi")
    for b in ("Rchi", "Rdchi", "Rlapl_chi"):
        if b in uses:
            params.append(b)
    for b in ("dchi_g", "ddchi_g", "dchi_gA", "ddchi_gA"):
        if b in uses:
            params.append(b)
    if "grad" in uses:
        params.append("grad_rho")
    if "grad_a" in uses:
        params.append("grad_rho_a")
    if "grad_b" in uses:
        params.append("grad_rho_b")
    if "jp" in uses:
        params.append("jp")
    for sk in ("jp_a", "jp_b"):
        if sk in uses:
            params.append(sk)
    if "hrho" in uses:
        params.append("hess_rho")
    for sk in ("hrho_a", "hrho_b"):
        if sk in uses:
            params.append(f"hess_rho_{sk[-1]}")
    if "dgrad_g" in uses:
        params.append("dgrad_rho_g")
    params += sorted(u.split(":", 1)[1] for u in uses
                     if u.startswith("gscalar:"))
    params += [f"grad_rho_{lbl}" for lbl in pert_grads]
    params += [f"jp_{lbl}" for lbl in pert_jps]
    params += [f"hess_rho_{lbl}" for lbl in pert_hrhos]
    params += pert_scalars
    # coordinate-system data: Lame factors, and the l(l+1) of the block
    # being contracted when the density matrix is blocked by angular
    # momentum. Empty for Cartesian, so the signature is unchanged there.
    params += sorted(u.split(":", 1)[1] for u in uses
                     if u.startswith("geometry:"))
    params += libxc_args

    plist = []
    for (ufac, vfac), cpoly in sorted(patterns.items()):
        monos = [(float(coeff), tuple((s.name, e) for s, e in skey))
                 for skey, coeff in cpoly.items()]
        plist.append((ufac, vfac, monos))

    return CollapsedKernel(u_lbl=u_lbl, v_lbl=v_lbl, patterns=plist,
                           params=params, libxc_args=libxc_args,
                           pert_grads=pert_grads, pert_scalars=pert_scalars,
                           uses=uses, n_terms=n_terms)


#: basis arrays that gain a conjugated companion in sesquilinear emission.
_SESQUI_BASES = ("chi", "dchi", "lapl_chi", "hess_chi")

#: Re/Im decomposition of a complex product a*b (plain) or conj(a)*b
#: (conjugate_left): rows (sign, part of a, part of b) such that
#: part(product) = sum sign * a_part * b_part.  The single source of truth
#: for every split-storage lowering: the part-extracted matrix emission
#: below and the pointwise complex products of the VeloxChem writer.
_PART_PRODUCT = {
    ("re", False): ((1, "re", "re"), (-1, "im", "im")),
    ("im", False): ((1, "re", "im"), (1, "im", "re")),
    ("re", True): ((1, "re", "re"), (1, "im", "im")),
    ("im", True): ((1, "re", "im"), (-1, "im", "re")),
}


def part_product(part: str, conjugate_left: bool):
    """Decomposition rows (sign, left part, right part) of the real or
    imaginary part of a complex product; see :data:`_PART_PRODUCT`."""
    return _PART_PRODUCT[(part, bool(conjugate_left))]


def _side_factor(fac: str, suffix: str, why: str) -> str:
    """A basis factor code with a side suffix, e.g. chi -> chi_l."""
    base = fac.split("[", 1)[0]
    if base not in _SESQUI_BASES:
        raise ValueError(
            f"{why} emission does not support basis factor {fac!r} "
            "(geometric-derivative kernels are bilinear-only for now)")
    return fac.replace(base, base + suffix, 1)


def generate_collapsed(ki: KernelIntegrand, func_name: str = "kernel",
                       batch: bool = False,
                       sesquilinear: bool = False,
                       two_sided: bool = False,
                       part: str = None,
                       input_parts: Dict[str, str] = None) -> GeneratedFunction:
    """Pattern-collapsed NumPy emission (the production lowering).

    Every monomial of a one-free-pair integrand factorizes as
    (basis factor at u) x (basis factor at v) x (per-point scalar), and only a
    handful of basis-pair patterns exist at ANY derivative order.  See
    collapse() for the structured intermediate shared with other backends.

    With ``sesquilinear=True`` (complex basis functions), the free pair
    contracts the CONJUGATED basis values on the u side, returning
    ``F_uv = dE/dP_uv`` for the convention ``rho = sum_uv P_uv chi_u^* chi_v``.
    The per-point scalar coefficients are untouched: all ingredient fields of
    a Hermitian density matrix remain real for a complex basis as well.
    The RECOMMENDED storage convention for complex quantities is separate
    real and imaginary arrays (``part`` below); the interleaved-complex
    lowering (``part=None``), which takes companion arrays ``chi_c``
    (= conj(chi)), ``dchi_c``, ... after each plain basis array and
    accumulates in complex arithmetic, is kept for hosts with native
    complex storage.

    With ``part`` in ``{'re', 'im', 'both'}`` (requires ``sesquilinear``),
    the emission switches to SPLIT STORAGE: every basis array is passed as
    separate real and imaginary parts (``chi_re``, ``chi_im``, ...), the
    requested part of the output matrix is accumulated in real arithmetic,
    and each basis-pair pattern lowers to two real matrix products instead
    of one complex one (four real products' work) -- 50% fewer
    multiplications when a single part is requested, identical work but
    real-arithmetic-only when both are.  ``input_parts`` maps a basis array
    base (e.g. ``'chi'``) to ``'re'`` or ``'im'`` when that operand is known
    purely real or imaginary: its zero part is eliminated at generation
    time, only the surviving array is taken as a parameter, and pattern
    terms touching the zero part are dropped (a real basis reduces the Re
    output to single real products and empties the Im output).
    ``part='both'`` returns the tuple ``(out_re, out_im)``.

    With ``two_sided=True``, the u and v sides take INDEPENDENT collocation
    arrays (``chi_l``/``chi_r``, ``dchi_l``/``dchi_r``, ...), and the output
    is (nl, nr) with nl and nr free.  Seeding the sides with occupied and
    virtual MOLECULAR-orbital values on the grid emits the sigma-vector
    contraction sigma_ia directly, with no atomic-orbital matrix ever
    materialized -- the matrix-free mode of plane-wave and Davidson-solver
    hosts.  The caller supplies conjugated left-side values (bra side) when
    the orbitals are complex.
    """
    if sesquilinear and two_sided:
        raise ValueError("sesquilinear and two_sided are mutually exclusive; "
                         "two_sided callers pass conjugated left arrays")
    if part is not None:
        if part not in ("re", "im", "both"):
            raise ValueError(f"unknown part {part!r}")
        if not sesquilinear:
            raise ValueError("part extraction applies to the sesquilinear "
                             "(complex-basis) emission; pass sesquilinear="
                             "True")
    elif input_parts:
        raise ValueError("input_parts requires part extraction")
    input_parts = dict(input_parts or {})
    for base, msk in input_parts.items():
        if base not in _SESQUI_BASES or msk not in ("re", "im"):
            raise ValueError(f"bad input_parts entry {base!r}: {msk!r}")

    def _mask(fac: str) -> Tuple[str, ...]:
        m = input_parts.get(fac.split("[", 1)[0])
        return ("re", "im") if m is None else (m,)

    ck = collapse(ki)
    u_lbl, v_lbl = ck.u_lbl, ck.v_lbl
    libxc_args = ck.libxc_args
    pert_grads, pert_scalars = ck.pert_grads, ck.pert_scalars
    uses, params, n_terms = ck.uses, ck.params, ck.n_terms

    if sesquilinear and part is None:
        for ufac, _vfac, _monos in ck.patterns:
            _side_factor(ufac, "_c", "sesquilinear")   # reject early
        params = list(params)
        for base in reversed(_SESQUI_BASES):
            if base in params:
                params.insert(params.index(base) + 1, base + "_c")
    if part is not None:
        for ufac, vfac, _monos in ck.patterns:
            _side_factor(ufac, "_re", "split")         # reject early
            _side_factor(vfac, "_re", "split")
        params = list(params)
        for base in reversed(_SESQUI_BASES):
            if base in params:
                i = params.index(base)
                params[i:i + 1] = [base + "_" + p for p in _mask(base)]
    if two_sided:
        for ufac, vfac, _monos in ck.patterns:
            _side_factor(ufac, "_l", "two-sided")      # reject early
            _side_factor(vfac, "_r", "two-sided")
        # replace each basis array by its _l/_r pair, keeping the order
        params = []
        for p in ck.params:
            if p in _SESQUI_BASES:
                params += [p + "_l", p + "_r"]
            else:
                params.append(p)

    # map scalar symbol names to python expressions for coefficient printing
    def scalar_code(name: str) -> str:
        op, kind = _classify(name)
        code = op.code
        if batch and kind.startswith(("pgrad:", "pjp:", "phrho:")):
            code = re.sub(r"\[(\d)\]$", r"[:, \1]", code)
        return code

    def mono_code(coeff, factors_) -> str:
        """One scalar monomial as python source, e.g. '2.0*w*vsigma*rho_p1'."""
        factors = [repr(coeff)]
        for name, e in factors_:
            c = scalar_code(name)
            if e == 1:
                factors.append(c)
            elif e > 1:
                factors.append(f"{c}**{e}")
            else:
                # negative powers arise from curvilinear scale factors,
                # e.g. the 1/r^2 of an l-blocked centrifugal term
                factors.append(f"{c}**({e})")
        return "*".join(factors)

    # Transpose-partner deduplication: patterns (u, v) and (v, u) whose
    # coefficient polynomials are identical (or exactly negated, as in the
    # antisymmetric current channel) are evaluated with a single matrix
    # multiplication and a (subtracted) transposed accumulation. In the
    # sesquilinear mode the partner accumulates through the conjugate
    # transpose instead, which is exact because the per-point coefficients
    # are real (all ingredients of a Hermitian density matrix are real
    # also for a complex basis). The two-sided mode emits every pattern
    # separately, as its bra and ket collocations are different arrays.
    def _mono_dict(monos):
        return {fac: coeff for coeff, fac in monos}

    plan: List[tuple] = []          # (pattern index, +1 / -1 / None)
    if two_sided:
        plan = [(k, None) for k in range(len(ck.patterns))]
    else:
        index = {(u, v): k for k, (u, v, _) in enumerate(ck.patterns)}
        done = set()
        for k, (u, v, m) in enumerate(ck.patterns):
            if k in done:
                continue
            done.add(k)
            j = index.get((v, u))
            sign = None
            if u != v and j is not None and j not in done:
                mk = _mono_dict(m)
                mj = _mono_dict(ck.patterns[j][2])
                if mj == mk:
                    sign = 1
                elif mj == {fac: -c for fac, c in mk.items()}:
                    sign = -1
                if sign is not None:
                    done.add(j)
            plan.append((k, sign))

    # Cartesian fusion: patterns that share their u-side factor and whose
    # v-side factors are components of one collocation array contract
    # against the SAME left operand, so the Cartesian sum can be carried
    # out on the grid before the matrix product,
    #
    #   sum_c (u * c_c) @ dchi[c].T  =  u @ (sum_c c_c * dchi[c]).T,
    #
    # trading n_c matrix products for n_c elementwise scalings of an
    # (nbf, ngrid) array.  A diagonal pattern (u == v) joins the group with
    # half its coefficient, since its product is (anti)symmetric and the
    # transposed accumulation restores the other half.  This is the fusion
    # that hand-written GGA Fock builds perform by folding the potential
    # into a single modified collocation array.  Restricted to the plain
    # single-array lowering: the two-sided mode has different bra and ket
    # arrays, and split storage fuses per part elsewhere.  Batched kernels
    # fuse inside an explicit loop over the perturbation index, so that the
    # fused array stays (nbf, ngrid) rather than growing an nbatch axis;
    # the coefficient assembly, which carries the monomials, stays
    # vectorized over the batch.
    _COMPONENT = re.compile(r"^(.*)\[(\d)\]$")

    def _fuse(plan_in):
        """Group plan entries into ``(entries, sign)`` fusion bundles.

        Entries are ``(pattern index, halve)``.  Patterns that share their
        u-side factor contract against the same left operand, so their sum
        can be formed on the grid before the matrix product.  Only patterns
        accumulated the same way may share a bundle: a transpose-paired
        group closes as ``t +- t^T``, whereas an unpaired group closes as
        ``t`` alone.  A diagonal pattern (u == v) is symmetric and has no
        distinct partner; it joins a transpose-paired group of its own
        u-side factor with half its coefficient, the transposed
        accumulation supplying the other half, and otherwise joins the
        unpaired group at full strength.
        """
        buckets: Dict[tuple, List[tuple]] = {}
        order: List[tuple] = []

        def _put(key, entry):
            if key not in buckets:
                buckets[key] = []
                order.append(key)
            buckets[key].append(entry)

        pending = []
        for k, sign in plan_in:
            ufac, vfac, _ = ck.patterns[k]
            if sign is not None and not two_sided:
                _put((ufac, sign), (k, False))
            else:
                pending.append((k, ufac, vfac))
        for k, ufac, vfac in pending:
            if not two_sided and ufac == vfac and (ufac, 1) in buckets:
                _put((ufac, 1), (k, True))       # half now, half by t^T
            else:
                _put((ufac, None), (k, False))
        return [(buckets[key], key[1]) for key in order]

    # Emit big sums as chunked += accumulation: a single expression over
    # tens of thousands of monomials builds an AST deep enough to blow
    # CPython's fixed compiler recursion cap (<= 3.13).
    _CHUNK = 64

    #: output array name per requested part
    outs = [(part, "out")] if part in ("re", "im") else \
        [("re", "out_re"), ("im", "out_im")]

    def _gemm(ucode, vcode):
        if batch:
            return (f"np.einsum('ug,xg,vg->xuv', {ucode}, "
                    f"c, {vcode}, optimize=True)")
        return f"({ucode} * c) @ {vcode}.T"

    def _chunked(name, monos):
        """Emit ``name`` as a chunked accumulation of the monomial sum."""
        ch = [" + ".join(mono_code(coeff, fac) for coeff, fac in
                         monos[i:i + _CHUNK])
              for i in range(0, len(monos), _CHUNK)]
        return [f"    {name} = {ch[0]}"] + [f"    {name} += {p}" for p in ch[1:]]

    lines: List[str] = []
    for bundle, bsign in _fuse(plan):
        if len(bundle) > 1 and part is not None:
            # --- fusion in split storage: the part decomposition turns each
            # pattern into real products (sign, a, b); those sharing a left
            # operand a fuse into one product against the summed right one
            for i, (k, _h) in enumerate(bundle):
                lines += _chunked(f"c{i}", ck.patterns[k][2])
            tT = "np.transpose(t, (0, 2, 1))" if batch else "t.T"
            bx = "[:, None, :]" if batch else ""
            for p, out_name in outs:
                groups: Dict[str, List[tuple]] = {}
                gorder: List[str] = []
                for i, (k, halve) in enumerate(bundle):
                    kufac, kvfac, _ = ck.patterns[k]
                    umask, vmask = _mask(kufac), _mask(kvfac)
                    half = halve
                    for sg, lp, rp in part_product(p, True):
                        if lp not in umask or rp not in vmask:
                            continue
                        a = _side_factor(kufac, "_" + lp, "split")
                        b = _side_factor(kvfac, "_" + rp, "split")
                        groups.setdefault(a, []) or gorder.append(a) \
                            if a not in gorder else None
                        groups.setdefault(a, []).append((sg, b, i, half))
                if not groups:
                    continue
                first = True
                for a in gorder:
                    for j, (sg, b, i, half) in enumerate(groups[a]):
                        sc = ("-" if sg < 0 else "") + \
                             ("0.5 * " if half else "")
                        op = "d = " if j == 0 else "d += "
                        lines.append(f"    {op}{sc}c{i}{bx} * {b}")
                    prod = (f"np.einsum('ug,xvg->xuv', {a}, d, optimize=True)"
                            if batch else f"{a} @ d.T")
                    lines.append(f"    t {'=' if first else '+='} {prod}")
                    first = False
                lines.append(f"    {out_name} += t")
                if bsign is not None:
                    ts = bsign if p == "re" else -bsign
                    lines.append(f"    {out_name} "
                                 f"{'+' if ts > 0 else '-'}= {tT}")
            continue

        if len(bundle) > 1:
            # --- Cartesian fusion: one matrix product for the group -------
            bu = ck.patterns[bundle[0][0]][0]
            if sesquilinear:
                ucode = _side_factor(bu, "_c", "sesquilinear")
            elif two_sided:
                ucode = _side_factor(bu, "_l", "two-sided")
            else:
                ucode = bu
            for i, (k, _s) in enumerate(bundle):
                lines += _chunked(f"c{i}", ck.patterns[k][2])
            # the fused operand carries the batch axis, so the whole
            # bundle becomes ONE batched matrix product; the grid batch
            # size is the host's knob for the (nx, nbf, ng) intermediate
            bx = "[:, None, :]" if batch else ""
            for i, (k, halve) in enumerate(bundle):
                kufac, kvfac, _ = ck.patterns[k]
                half = "0.5 * " if halve else ""
                vcode = (_side_factor(kvfac, "_r", "two-sided")
                         if two_sided else kvfac)
                op = "d = " if i == 0 else "d += "
                lines.append(f"    {op}{half}c{i}{bx} * {vcode}")
            if batch:
                lines.append(f"    t = np.einsum('ug,xvg->xuv', {ucode}, "
                             f"d, optimize=True)")
                tr = ("np.conjugate(np.transpose(t, (0, 2, 1)))"
                      if sesquilinear else "np.transpose(t, (0, 2, 1))")
            else:
                lines.append(f"    t = {ucode} @ d.T")
                tr = "t.conj().T" if sesquilinear else "t.T"
            lines.append("    out += t")
            if bsign is not None:
                lines.append(f"    out {'+' if bsign > 0 else '-'}= {tr}")
            continue

        k, _halve = bundle[0]
        sign = bsign
        ufac, vfac, monos = ck.patterns[k]


        if part is not None:
            # split-storage lowering: the requested part of the (u-side
            # conjugated) pattern product from real matrix products, with
            # terms touching a statically-zero operand part eliminated
            umask, vmask = _mask(ufac), _mask(vfac)
            terms = {}
            for p, out_name in outs:
                terms[p] = [
                    (s, _side_factor(ufac, "_" + lp, "split"),
                     _side_factor(vfac, "_" + rp, "split"))
                    for s, lp, rp in part_product(p, True)
                    if lp in umask and rp in vmask]
            if not any(terms.values()):
                continue                     # pattern is identically zero
        chunks = [" + ".join(mono_code(coeff, fac) for coeff, fac in
                             monos[i:i + _CHUNK])
                  for i in range(0, len(monos), _CHUNK)]
        lines.append(f"    c = {chunks[0]}")
        lines.extend(f"    c += {p}" for p in chunks[1:])

        if part is not None:
            tT = "np.transpose(t, (0, 2, 1))" if batch else "t.T"
            for p, out_name in outs:
                if not terms[p]:
                    continue
                if sign is None:
                    for s, a, b in terms[p]:
                        lines.append(f"    {out_name} "
                                     f"{'+' if s > 0 else '-'}= {_gemm(a, b)}")
                else:
                    s0, a0, b0 = terms[p][0]
                    lines.append(f"    t = {'-' if s0 < 0 else ''}"
                                 f"{_gemm(a0, b0)}")
                    for s, a, b in terms[p][1:]:
                        lines.append(f"    t {'+' if s > 0 else '-'}= "
                                     f"{_gemm(a, b)}")
                    lines.append(f"    {out_name} += t")
                    # the fused partner adds sign * t^dagger, whose real
                    # part is +t_re^T and imaginary part is -t_im^T
                    ts = sign if p == "re" else -sign
                    lines.append(f"    {out_name} "
                                 f"{'+' if ts > 0 else '-'}= {tT}")
            continue

        if sesquilinear:
            ucode, vcode = _side_factor(ufac, "_c", "sesquilinear"), vfac
        elif two_sided:
            ucode = _side_factor(ufac, "_l", "two-sided")
            vcode = _side_factor(vfac, "_r", "two-sided")
        else:
            ucode, vcode = ufac, vfac
        gemm = _gemm(ucode, vcode)
        if batch:
            tr = ("np.conjugate(np.transpose(t, (0, 2, 1)))"
                  if sesquilinear else "np.transpose(t, (0, 2, 1))")
        else:
            tr = "t.conj().T" if sesquilinear else "t.T"
        if sign is None:
            lines.append(f"    out += {gemm}")
        else:
            lines.append(f"    t = {gemm}")
            lines.append(f"    out += t")
            lines.append(f"    out {'+' if sign > 0 else '-'}= {tr}")

    header = [
        f"def {func_name}({', '.join(params)}):",
        "    # machine-generated by xckernel; do not edit.",
        "    # Copyright (c) 2026 Susi Lehtola.",
        f"    # pattern-collapsed: {len(ck.patterns)} patterns "
        f"from {n_terms} terms",
    ]
    if two_sided:
        header += ["    nl = chi_l.shape[0]",
                   "    nr = chi_r.shape[0]"]
        shape, dtype = "(nl, nr)", ", dtype=np.result_type(chi_l, chi_r)"
    elif part is not None:
        bsrc = next(p for p in params
                    if p.rsplit("_", 1)[0] in _SESQUI_BASES)
        header += [f"    nao = {bsrc}.shape[-2]"]
        shape, dtype = "(nao, nao)", ""
    else:
        header += ["    nao = chi.shape[0]"]
        shape = "(nao, nao)"
        dtype = ", dtype=complex" if sesquilinear else ""
    out_names = [name for _p, name in outs] if part is not None else ["out"]
    if batch:
        if not (pert_scalars or pert_grads):
            raise ValueError("batch=True requires perturbed-field operands")
        nx_src = pert_scalars[0] if pert_scalars \
            else f"grad_rho_{pert_grads[0]}"
        header += [f"    nx = {nx_src}.shape[0]"]
        header += [f"    {name} = np.zeros((nx,) + {shape}{dtype})"
                   if two_sided else
                   f"    {name} = np.zeros((nx, nao, nao){dtype})"
                   for name in out_names]
    else:
        header += [f"    {name} = np.zeros({shape}{dtype})"
                   for name in out_names]
    source = "\n".join(header + lines
                       + [f"    return {', '.join(out_names)}", ""])

    return GeneratedFunction(
        name=func_name, source=source, out_indices=u_lbl + v_lbl,
        libxc_args=libxc_args, uses_lapl_chi=("lapl" in uses),
        uses_grad_rho=("grad" in uses),
        uses_grad_rho_a=("grad_a" in uses), uses_grad_rho_b=("grad_b" in uses),
        pert_grads=pert_grads, pert_scalars=pert_scalars, batch=batch,
        part=part, n_patterns=len(ck.patterns),
        n_products=(source.count(" @ ")
                    + source.count("np.einsum('ug,xvg->xuv'")
                    + source.count("np.einsum('ug,xg,vg->xuv'")))


def compile_function(gen: GeneratedFunction):
    """exec the generated source and return the live callable."""
    import numpy as np
    ns: Dict[str, object] = {"np": np}
    exec(gen.source, ns)
    return ns[gen.name]
