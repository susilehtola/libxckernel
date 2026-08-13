"""DFT ingredients as bilinear forms in the density matrix, and their
derivatives with respect to P_uv.

Every ingredient the exchange-correlation functional depends on is, at a grid
point, one of two things:

* a **primitive** field that is *linear* in the density matrix,

      k(P) = sum_{ab} P_ab * Q_k(chi_a, chi_b),

  with Q_k a fixed bilinear kernel of the two basis functions.  Examples:
  the density rho, each Cartesian component of grad rho, the density Laplacian,
  and the kinetic energy density tau.

* a **derived** field that is an algebraic function of primitive fields, e.g.
  the reduced gradient sigma = grad rho . grad rho.

The derivative that assembles the Fock matrix is

      d k / d P_uv .

For a primitive field this is just the kernel evaluated at the free indices,
Q_k(chi_u, chi_v) -- obtained here by the Kronecker-delta collapse of
d/dP_uv sum_{ab} P_ab Q_k(a,b).  For a derived field it follows by the chain
rule through the primitive fields; SymPy does that composition for us, with each
primitive field carried as an opaque *total-field symbol* so the result stays
O(1) per (u, v) instead of re-expanding the sum over the whole basis.

This is precisely forward-mode automatic differentiation with the seed
direction e_uv in density-matrix space -- the "AD backend for Libxc".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import sympy as sp

from .basis import AXES, HESS_COMPS, Orbital, dot

#: A bilinear kernel Q(a, b): given two orbitals, return the coefficient that
#: multiplies P_ab in a primitive field's microscopic definition.
Kernel = Callable[[Orbital, Orbital], sp.Expr]


@dataclass(frozen=True)
class Primitive:
    """A field linear in P: value symbol + microscopic bilinear kernel."""

    name: str
    symbol: sp.Symbol          # opaque total-field symbol, e.g. rho or grad_rho_x
    kernel: Kernel             # Q(a, b) such that field = sum_ab P_ab Q(a, b)

    def seed(self, u: Orbital, v: Orbital) -> sp.Expr:
        """d field / d P_uv = Q(u, v) (delta-collapse of the P-sum)."""
        return self.kernel(u, v)


# --- primitive kernels -----------------------------------------------------

def _k_rho(a: Orbital, b: Orbital) -> sp.Expr:
    return a.val * b.val


def _k_grad(ax: int) -> Kernel:
    # d/dx (chi_a chi_b) = (dchi_a) chi_b + chi_a (dchi_b)
    def kernel(a: Orbital, b: Orbital) -> sp.Expr:
        return a.grad[ax] * b.val + a.val * b.grad[ax]
    return kernel


def _k_lapl(a: Orbital, b: Orbital) -> sp.Expr:
    # laplacian(chi_a chi_b) = (lapl chi_a) chi_b + 2 grad.grad + chi_a (lapl chi_b)
    return a.lapl * b.val + 2 * dot(a.grad, b.grad) + a.val * b.lapl


def _k_tau(a: Orbital, b: Orbital) -> sp.Expr:
    # tau = 1/2 sum_i grad chi_i . grad chi_i  ->  1/2 grad chi_a . grad chi_b
    return sp.Rational(1, 2) * dot(a.grad, b.grad)


# --- the primitive fields --------------------------------------------------

def _k_jp(ax: int) -> Kernel:
    # paramagnetic current density j_p = Im(psi* grad psi): a REAL vector
    # field contracting the ANTISYMMETRIC (imaginary) part of the density
    # matrix.  Convention: the host passes Im P as a real matrix; the kernel
    # is the antisymmetric bilinear (1/2)(chi_a d chi_b - d chi_a chi_b).
    # Zero for a real (current-free) ground state.
    def kernel(a: Orbital, b: Orbital) -> sp.Expr:
        return sp.Rational(1, 2) * (a.val * b.grad[ax] - a.grad[ax] * b.val)
    return kernel


def _k_hess(i: int, j: int) -> Kernel:
    # density-Hessian component rho_ij = d_i d_j rho: the full second
    # derivative of the bilinear chi_a chi_b (its trace is the Laplacian).
    def kernel(a: Orbital, b: Orbital) -> sp.Expr:
        return (a.hess_ij(i, j) * b.val + a.grad[i] * b.grad[j]
                + a.grad[j] * b.grad[i] + a.val * b.hess_ij(i, j))
    return kernel


RHO = Primitive("rho", sp.Symbol("rho", real=True), _k_rho)
GRAD_RHO = tuple(
    Primitive(f"grad_rho_{ax}", sp.Symbol(f"grad_rho_{ax}", real=True), _k_grad(i))
    for i, ax in enumerate(AXES)
)
LAPL_RHO = Primitive("lapl_rho", sp.Symbol("lapl_rho", real=True), _k_lapl)
TAU = Primitive("tau", sp.Symbol("tau", real=True), _k_tau)
JP = tuple(
    Primitive(f"jp_{ax}", sp.Symbol(f"jp_{ax}", real=True), _k_jp(i))
    for i, ax in enumerate(AXES)
)

# the six independent density-Hessian components, packed xx,xy,xz,yy,yz,zz.
HESS_RHO = tuple(
    Primitive(f"hess_rho_{AXES[i]}{AXES[j]}",
              sp.Symbol(f"hess_rho_{AXES[i]}{AXES[j]}", real=True),
              _k_hess(i, j))
    for (i, j) in HESS_COMPS
)

# 1/rho as a pseudo-primitive: a host-supplied per-point field whose
# P-derivative closes rationally, D(inv_rho) = -inv_rho^2 * D(rho).  The seed
# expression contains field symbols; higher orders differentiate them in turn,
# so the tower stays polynomial in {fields, seeds} at every order.
_INV_RHO_SYMBOL = sp.Symbol("inv_rho", real=True)
INV_RHO = Primitive(
    "inv_rho", _INV_RHO_SYMBOL,
    lambda a, b: -_INV_RHO_SYMBOL**2 * a.val * b.val)

#: All primitives keyed by name, for substitution / operand collection.
PRIMITIVES: Dict[str, Primitive] = {
    p.name: p for p in (RHO, *GRAD_RHO, LAPL_RHO, TAU, *JP, *HESS_RHO,
                        INV_RHO)
}


@dataclass
class Ingredient:
    """A Libxc input variable: its total-field expression in terms of
    primitive symbols, plus its P-derivative seed d(var)/dP_uv."""

    name: str                                  # libxc variable name: rho/sigma/lapl/tau
    value: sp.Expr                             # in terms of primitive symbols
    _seed: Callable[[Orbital, Orbital], sp.Expr]

    def seed(self, u: Orbital, v: Orbital) -> sp.Expr:
        return self._seed(u, v)


def _primitive_ingredient(p: Primitive, libxc_name: str) -> Ingredient:
    # A primitive used directly as a Libxc input variable.  The Libxc variable
    # name may differ from the primitive's internal name (e.g. the density
    # Laplacian primitive is "lapl_rho" but Libxc calls the variable "lapl").
    return Ingredient(libxc_name, p.symbol, p.seed)


# rho, lapl, tau are primitives used directly as libxc variables.
RHO_ING = _primitive_ingredient(RHO, "rho")
LAPL_ING = _primitive_ingredient(LAPL_RHO, "lapl")
TAU_ING = _primitive_ingredient(TAU, "tau")

# sigma = grad rho . grad rho  is derived; its seed follows by the chain rule.
_SIGMA_VALUE = dot(tuple(p.symbol for p in GRAD_RHO),
                   tuple(p.symbol for p in GRAD_RHO))


def _sigma_seed(u: Orbital, v: Orbital) -> sp.Expr:
    # d sigma / dP_uv = sum_i (d sigma / d grad_rho_i)(d grad_rho_i / dP_uv),
    # SymPy provides the outer derivative; the primitives provide the inner seed.
    total = sp.Integer(0)
    for p in GRAD_RHO:
        total += sp.diff(_SIGMA_VALUE, p.symbol) * p.seed(u, v)
    return sp.expand(total)


SIGMA_ING = Ingredient("sigma", _SIGMA_VALUE, _sigma_seed)

# Gauge-corrected kinetic energy density for current-density DFT:
#   tau~ = tau - j_p.j_p / (2 rho)
# (Dobson; Maximoff & Scuseria; Becke).  The functional library is evaluated
# AT tau~, so a standard Libxc meta-GGA becomes gauge-invariant under magnetic
# perturbations; the derivative tower below chains through j_p and 1/rho.
_CTAU_VALUE = TAU.symbol - sp.Rational(1, 2) * INV_RHO.symbol * sum(
    p.symbol**2 for p in JP)


def _ctau_seed(u: Orbital, v: Orbital) -> sp.Expr:
    total = sp.Integer(0)
    for p in (TAU, *JP, INV_RHO):
        total += sp.diff(_CTAU_VALUE, p.symbol) * p.seed(u, v)
    return sp.expand(total)


CTAU_ING = Ingredient("tau", _CTAU_VALUE, _ctau_seed)

# The gradient-projected density Hessian behind the reduced density Hessian
# of local-hybrid calibration functions -- introduced as Z_{sigma,sigmasigma}
# by Maier, Haasler, Arbuznikov & Kaupp, Phys. Chem. Chem. Phys. 18, 21133
# (2016), Eqs. (22)-(23) (calibration-function concept: Arbuznikov & Kaupp,
# J. Chem. Phys. 141, 204101 (2014); renamed eta by Schattenberg & Kaupp,
# J. Phys. Chem. A 125, 2697 (2021), Eq. (10); its dimensionless form is the
# u parameter of GGA exchange potentials since Perdew & Wang 1986):
#   eta = grad rho^T . (grad grad^T rho) . grad rho,
# the raw variable behind the "reduced density Hessian"
# p = eta / (k^2 gamma rho^{5/3}) (the reduction is the functional's
# business, like the reduced gradient s is built from sigma and rho).
# Cubic in P-linear primitives -- one factor deeper than sigma -- so its
# derivative tower exercises second-order seeds against BOTH the gradient
# and the Hessian primitives.
_ETA_VALUE = sum(
    (1 if i == j else 2) * GRAD_RHO[i].symbol * GRAD_RHO[j].symbol * h.symbol
    for h, (i, j) in zip(HESS_RHO, HESS_COMPS))


def _eta_seed(u: Orbital, v: Orbital) -> sp.Expr:
    total = sp.Integer(0)
    for p in (*GRAD_RHO, *HESS_RHO):
        total += sp.diff(_ETA_VALUE, p.symbol) * p.seed(u, v)
    return sp.expand(total)


ETA_ING = Ingredient("eta", _ETA_VALUE, _eta_seed)


#: Libxc input variables keyed by name, for chain-rule seeds at any order.
#: (Default variable set; families may override individual variables with
#: mapped ingredients such as the gauge-corrected tau -- consumers must use
#: the Functional's own ingredient list, not this table, when both exist.)
INGREDIENTS: Dict[str, Ingredient] = {
    "rho": RHO_ING,
    "sigma": SIGMA_ING,
    "lapl": LAPL_ING,
    "tau": TAU_ING,
    "eta": ETA_ING,
}

#: Primitive field keyed by its total-field symbol, for seeding P-atoms that
#: appear inside expressions (notably grad_rho_i inside a sigma seed).
PRIM_BY_SYMBOL: Dict[sp.Symbol, Primitive] = {
    p.symbol: p for p in PRIMITIVES.values()
}


#: Libxc-family -> ordered list of the ingredients it consumes.
FAMILIES: Dict[str, List[Ingredient]] = {
    "lda": [RHO_ING],
    "gga": [RHO_ING, SIGMA_ING],
    "mgga": [RHO_ING, SIGMA_ING, LAPL_ING, TAU_ING],
    # meta-GGA without the density Laplacian (the common case in Libxc):
    "mgga_tau": [RHO_ING, SIGMA_ING, TAU_ING],
    # meta-GGA without tau: deorbitalized functionals (e.g. r2SCAN-L):
    "mgga_lapl": [RHO_ING, SIGMA_ING, LAPL_ING],
    # current-density DFT: a tau-meta-GGA evaluated at the gauge-corrected
    # tau~ = tau - j_p^2/(2 rho); operands gain jp (3,ng) and inv_rho (ng,).
    "cmgga_tau": [RHO_ING, SIGMA_ING, CTAU_ING],
    # local-hybrid calibration-function set (Arbuznikov & Kaupp 2014;
    # Maier et al. 2016): the full meta-GGA variables plus the gradient-projected
    # density Hessian eta; operands gain hess_rho (6,ng) and the
    # second-derivative collocation hess_chi (6,nbf,ng).
    "hmgga": [RHO_ING, SIGMA_ING, LAPL_ING, TAU_ING, ETA_ING],
}
