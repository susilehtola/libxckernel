"""Symbolic basis-function fields on the integration grid.

A single basis function chi_i evaluated at a grid point carries, for our
purposes, a finite set of local values:

    val   = chi_i
    grad  = (d/dx, d/dy, d/dz) chi_i
    lapl  = laplacian chi_i
    hess  = the six independent second derivatives d_i d_j chi_i
            (packed xx, xy, xz, yy, yz, zz; needed by ingredients built on
            the density Hessian, e.g. the local-hybrid calibration variable
            eta = grad rho . (grad grad rho) . grad rho)

Everything the ingredients need (density, gradient, kinetic energy density,
laplacian of the density) is a bilinear form in two such basis functions
contracted with the density matrix P.  We therefore represent a basis function
by a small bundle of SymPy symbols tagged with an index label ('u', 'v', ...).
The label 'u'/'v' are the *free* indices of a Fock matrix element F_uv; 'a'/'b'
are the *summed* indices of the microscopic definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import sympy as sp

#: Cartesian axis labels, fixed order used throughout for gradient vectors.
AXES: Tuple[str, str, str] = ("x", "y", "z")

#: (i, j) axis pairs of the six independent symmetric-tensor components, in
#: the canonical packing xx, xy, xz, yy, yz, zz.
HESS_COMPS: Tuple[Tuple[int, int], ...] = tuple(
    (i, j) for i in range(3) for j in range(i, 3))
#: (i, j) -> packed component index, both orderings.
HESS_INDEX = {(i, j): k for k, (i, j) in enumerate(HESS_COMPS)}
HESS_INDEX.update({(j, i): k for k, (i, j) in enumerate(HESS_COMPS)})


@dataclass(frozen=True)
class Coordinates:
    """An orthogonal curvilinear coordinate system.

    Only the Lame scale factors distinguish one system from another. The
    generator works throughout in PHYSICAL (orthonormal) components,

        g_i = (1/h_i) d_i f,        |grad f|^2 = sum_i g_i^2,

    in which every chain-rule expression is identical to the Cartesian
    one -- the metric enters solely through the ingredient seeds, i.e.
    through how a field is built from the basis functions.

    axes:   component names, in order
    scale:  the Lame factors h_i, as sympy expressions in the grid
            symbols the host supplies; 1 means "no division emitted"
    """

    name: str
    axes: Tuple[str, ...]
    scale: Tuple[Any, ...]
    #: Angular contribution to |grad chi|^2 that survives an angular
    #: average, as a multiple of chi_a chi_b. A spherically averaged
    #: atomic code blocks its density matrix by angular momentum, and the
    #: angular part of the gradient of chi(r) Y_lm then contributes
    #: l(l+1)/r^2 chi_a chi_b to tau within block l -- a term with no
    #: counterpart in a Cartesian seed, and the reason a Cartesian
    #: generator cannot express such a code's kinetic energy density.
    angular: Any = 0

    def __post_init__(self):
        if len(self.axes) != len(self.scale):
            raise ValueError(f"{self.name}: {len(self.axes)} axes but "
                             f"{len(self.scale)} scale factors")

    @property
    def is_cartesian(self) -> bool:
        """True when the seeds carry no metric at all.

        Unit scale factors are not enough: a spherically averaged radial
        system also has h_r = 1, but its tau seed picks up the angular
        term l(l+1)/r^2, which has no Cartesian counterpart.  Both must
        be trivial for a seed to be metric-free.
        """
        return all(h == 1 for h in self.scale) and self.angular == 0


#: The identity case. Every existing emitter uses this, and its seeds
#: must stay byte-identical to the pre-Coordinates ones.
CARTESIAN = Coordinates("cartesian", AXES, (1, 1, 1))

#: Spherical polar, as HelFEM's atomic worker uses it: h = (1, r, r sin(theta)).
#: The radial factor is unity, which is why a radial gradient needs no
#: division at all.
_R = sp.Symbol("r", real=True, positive=True)
_STH = sp.Symbol("sin_theta", real=True)
SPHERICAL = Coordinates("spherical", ("r", "theta", "phi"), (1, _R, _R * _STH))

#: Prolate spheroidal, as HelFEM's diatomic worker uses it. The scale
#: factors are supplied by the host as grid arrays rather than built from
#: mu and nu here, so that the generated code matches whatever convention
#: the host already uses for the focal distance.
_HMU = sp.Symbol("scale_mu", real=True, positive=True)
_HNU = sp.Symbol("scale_nu", real=True, positive=True)
_HPHI = sp.Symbol("scale_phi", real=True, positive=True)
PROLATE = Coordinates("prolate", ("mu", "nu", "phi"), (_HMU, _HNU, _HPHI))

#: Spherically averaged atomic code (HelFEM's sadatom / aij): the density
#: matrix is blocked by angular momentum, the density and its gradient
#: are radial, and each block contributes l(l+1)/r^2 to tau. ``l_factor``
#: is the host-supplied l(l+1) of the block being contracted.
L_FACTOR = sp.Symbol("l_factor", real=True, nonnegative=True)
RADIAL = Coordinates("radial", ("r",), (1,), angular=L_FACTOR / (_R * _R))

#: Diatomic pure-m case (HelFEM's dftgrid_purem): for orbitals
#: R(mu) Y(nu) exp(i m phi) the density is phi-independent, so the
#: quadrature runs over the (mu, nu) plane only, the gradient has no phi
#: component, and the sole surviving azimuthal contribution is the
#: analytic m^2 |psi|^2 / h_phi^2 term in tau. That is the same shape as
#: the l-blocked radial case: an integrated-out angular coordinate
#: leaving a residual operator on the block index.
M_FACTOR = sp.Symbol("m_factor", real=True, nonnegative=True)
PROLATE_PUREM = Coordinates("prolate_purem", ("mu", "nu"), (_HMU, _HNU),
                            angular=M_FACTOR / (_HPHI * _HPHI))


@dataclass(frozen=True)
class Orbital:
    """Local values of one basis function, tagged by an index ``label``.

    Attributes are SymPy symbols named so generated code is readable, e.g.
    ``chi_u``, ``dchi_u_x``, ``lapl_chi_u``, ``hess_chi_u_xy``.
    """

    label: str
    val: sp.Symbol
    grad: Tuple[sp.Symbol, sp.Symbol, sp.Symbol]
    lapl: sp.Symbol
    hess: Tuple[sp.Symbol, ...]

    @classmethod
    def make(cls, label: str, coords=None) -> "Orbital":
        """Local values of a basis function.

        ``coords`` names the derivative components: a spherical orbital
        carries ``dchi_u_r``/``_theta``/``_phi`` so the emitted source
        reads the way the host's own collocation arrays do. The default
        is Cartesian, which reproduces the historical symbols exactly.
        """
        axes = AXES if coords is None else tuple(coords.axes)
        val = sp.Symbol(f"chi_{label}", real=True)
        grad = tuple(sp.Symbol(f"dchi_{label}_{ax}", real=True) for ax in axes)
        lapl = sp.Symbol(f"lapl_chi_{label}", real=True)
        hess = tuple(
            sp.Symbol(f"hess_chi_{label}_{AXES[i]}{AXES[j]}", real=True)
            for (i, j) in HESS_COMPS)
        return cls(label=label, val=val, grad=grad, lapl=lapl, hess=hess)

    def hess_ij(self, i: int, j: int) -> sp.Symbol:
        """Second-derivative symbol d_i d_j chi, from the packed components."""
        return self.hess[HESS_INDEX[(i, j)]]

    @property
    def symbols(self) -> Tuple[sp.Symbol, ...]:
        """All symbols belonging to this orbital, for codegen operand lists."""
        return (self.val, *self.grad, self.lapl, *self.hess)


def dot(a: Tuple[sp.Expr, ...], b: Tuple[sp.Expr, ...]) -> sp.Expr:
    """Euclidean dot product of two 3-vectors of SymPy expressions."""
    return sp.Add(*(ai * bi for ai, bi in zip(a, b)))
