"""Noncollinear (relativistic) response kernels via the locally collinear map.

Two- and four-component relativistic DFT evaluates ordinary collinear
functionals through the locally collinear ansatz: the noncollinear fields

    V = (rho_s, rho_x, rho_y, rho_z [, grad rho_s, grad rho_J])

(charge density, magnetization vector, and their Cartesian gradients) are
mapped onto the collinear Libxc variables

    U = (n+, n-, gamma++, gamma+-, gamma--)

with n+- = (rho_s +- |m|)/2 and the gamma combinations of Scalmani and
Frisch, as employed in the four-component Pauli-quaternion formulation of
Bersson, Kovtun, and Li [Phys. Chem. Chem. Phys. (2026),
doi:10.1039/d6cp02182d].  The functional derivatives with respect to U stay
opaque symbols with the standard polarized Libxc packing (vrho_0, vsigma_2,
v2rhosigma_4, ...); differentiating the map mechanically yields

* the noncollinear xc potential fields dE/dV_i -- the hand-derived
  Z matrices of Bersson et al. (their eqs. 42-47) -- and
* one order up, the noncollinear fxc coefficient matrix

    C_ij = sum_ab f_ab (dU_a/dV_i)(dU_b/dV_j) + sum_a f_a d2U_a/dV_i dV_j,

  the relativistic response kernel, which contracts two perturbed field
  sets as delta1_i C_ij delta2_j per grid point.

The sign factor f_nabla = sgn(grad rho_s . sum_J rho_J grad rho_J) is
piecewise constant and enters as an opaque +-1 field, held constant under
differentiation.  The map is nonsmooth as |m| -> 0: transverse kernel
components carry 1/|m| factors (the locally collinear kernel problem).
Hosts regularize the small-|m| region as in Bersson et al., or switch to
the multicollinear construction of Pu et al. [Phys. Rev. Research 5,
013036 (2023)], whose sphere-averaged integrand is equally generatable.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import sympy as sp

AXES = ("x", "y", "z")

#: collinear Libxc variables in packing order
U_ORDER = ("n_p", "n_m", "g_pp", "g_pm", "g_mm")

#: first functional derivative array element per collinear variable
LIBXC_FIRST = {"n_p": "vrho_0", "n_m": "vrho_1",
               "g_pp": "vsigma_0", "g_pm": "vsigma_1", "g_mm": "vsigma_2"}

#: second derivative array element per unordered collinear-variable pair,
#: standard polarized Libxc packing
LIBXC_SECOND = {
    ("n_p", "n_p"): "v2rho2_0", ("n_p", "n_m"): "v2rho2_1",
    ("n_m", "n_m"): "v2rho2_2",
    ("n_p", "g_pp"): "v2rhosigma_0", ("n_p", "g_pm"): "v2rhosigma_1",
    ("n_p", "g_mm"): "v2rhosigma_2", ("n_m", "g_pp"): "v2rhosigma_3",
    ("n_m", "g_pm"): "v2rhosigma_4", ("n_m", "g_mm"): "v2rhosigma_5",
    ("g_pp", "g_pp"): "v2sigma2_0", ("g_pp", "g_pm"): "v2sigma2_1",
    ("g_pp", "g_mm"): "v2sigma2_2", ("g_pm", "g_pm"): "v2sigma2_3",
    ("g_pm", "g_mm"): "v2sigma2_4", ("g_mm", "g_mm"): "v2sigma2_5",
}

#: the opaque sign factor of the gamma+-+- map
F_NABLA = sp.Symbol("f_nabla")


def _pair_key(a: str, b: str) -> Tuple[str, str]:
    return tuple(sorted((a, b), key=U_ORDER.index))


def nc_fields(family: str) -> List[sp.Symbol]:
    """The ordered noncollinear field symbols of a family."""
    if family not in ("lda", "gga"):
        raise ValueError(f"unsupported family {family!r} (the tau channel "
                         "awaits a noncollinear KED definition)")
    syms = [sp.Symbol("rho_s")]
    syms += [sp.Symbol(f"rho_{J}") for J in AXES]
    if family == "gga":
        syms += [sp.Symbol(f"grad_rho_s_{c}") for c in AXES]
        for J in AXES:
            syms += [sp.Symbol(f"grad_rho_{J}_{c}") for c in AXES]
    return syms


def nc_map(family: str) -> Dict[str, sp.Expr]:
    """The locally collinear map U(V), Bersson et al. eqs. (24)-(29)."""
    rho_s = sp.Symbol("rho_s")
    rho = {J: sp.Symbol(f"rho_{J}") for J in AXES}
    mnorm = sp.sqrt(sum(rho[J]**2 for J in AXES))
    U = {"n_p": rho_s / 2 + mnorm / 2,
         "n_m": rho_s / 2 - mnorm / 2}
    if family == "lda":
        return U
    gs = [sp.Symbol(f"grad_rho_s_{c}") for c in AXES]
    gm = {J: [sp.Symbol(f"grad_rho_{J}_{c}") for c in AXES] for J in AXES}
    gss = sum(g**2 for g in gs)
    gJJ = sum(gm[J][c]**2 for J in AXES for c in range(3))
    sJ = {J: sum(gs[c] * gm[J][c] for c in range(3)) for J in AXES}
    S = sp.sqrt(sum(sJ[J]**2 for J in AXES))
    U["g_pp"] = gss / 4 + gJJ / 4 + F_NABLA * S / 2
    U["g_pm"] = gss / 4 - gJJ / 4
    U["g_mm"] = gss / 4 + gJJ / 4 - F_NABLA * S / 2
    return U


def libxc_args(family: str, order: int) -> List[str]:
    """The Libxc derivative array elements consumed through the given order."""
    names = [LIBXC_FIRST[a] for a in U_ORDER if family == "gga"
             or not a.startswith("g_")]
    if order >= 2:
        names += [LIBXC_SECOND[_pair_key(a, b)]
                  for i, a in enumerate(U_ORDER) for b in U_ORDER[i:]
                  if _pair_key(a, b) in LIBXC_SECOND
                  and (family == "gga" or (not a.startswith("g_")
                                           and not b.startswith("g_")))]
        names = list(dict.fromkeys(names))
    return names


def nc_potential(family: str) -> Tuple[List[sp.Symbol], List[sp.Expr]]:
    """The noncollinear xc potential fields dE/dV_i (the Z matrices of
    Bersson et al.), with the Libxc first-derivative arrays opaque."""
    fields = nc_fields(family)
    U = nc_map(family)
    pot = []
    for v in fields:
        pot.append(sp.Add(*[sp.Symbol(LIBXC_FIRST[a]) * sp.diff(e, v)
                            for a, e in U.items()]))
    return fields, pot


def nc_fxc_matrix(family: str) -> Tuple[List[sp.Symbol],
                                        List[List[sp.Expr]]]:
    """The noncollinear fxc coefficient matrix C_ij over the field
    components, with the Libxc derivative arrays opaque.  Symmetric; the
    full matrix is returned."""
    fields = nc_fields(family)
    U = nc_map(family)
    dU = {a: [sp.diff(e, v) for v in fields] for a, e in U.items()}
    n = len(fields)
    C: List[List[sp.Expr]] = [[sp.Integer(0)] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            e = sp.Integer(0)
            for a in U:
                for b in U:
                    e += (sp.Symbol(LIBXC_SECOND[_pair_key(a, b)])
                          * dU[a][i] * dU[b][j])
                e += (sp.Symbol(LIBXC_FIRST[a])
                      * sp.diff(U[a], fields[i], fields[j]))
            C[i][j] = C[j][i] = e
    return fields, C
