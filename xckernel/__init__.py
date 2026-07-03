"""xckernel: an automatic-differentiation backend for Libxc.

Generate exchange-correlation kernel elements (Fock matrices, orbital Hessians,
arbitrary response quantities) in an LCAO basis, by differentiating the DFT
ingredients with respect to the density matrix and contracting against grid
data and Libxc functional derivatives.
"""

from __future__ import annotations

from .basis import AXES, Orbital, dot
from .ingredients import FAMILIES, Ingredient, Primitive
from .functional import Functional
from .fock import FockIntegrand, fock_integrand
from .deriv import directional_derivative, libxc_deriv_name, libxc_symbol
from .kernel import KernelIntegrand, kernel_integrand, fock, xc_kernel
from .spin_kernel import SpinIntegrand, fock_spin, kernel_spin
from .codegen import GeneratedFunction, generate, compile_function
from .mo import mo_transform, orbital_hessian

__all__ = [
    "AXES",
    "Orbital",
    "dot",
    "FAMILIES",
    "Ingredient",
    "Primitive",
    "Functional",
    "FockIntegrand",
    "fock_integrand",
    "directional_derivative",
    "libxc_deriv_name",
    "libxc_symbol",
    "KernelIntegrand",
    "kernel_integrand",
    "fock",
    "xc_kernel",
    "SpinIntegrand",
    "fock_spin",
    "kernel_spin",
    "GeneratedFunction",
    "generate",
    "compile_function",
    "mo_transform",
    "orbital_hessian",
]
