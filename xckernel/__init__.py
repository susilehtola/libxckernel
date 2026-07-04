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
from .spin_kernel import (SpinIntegrand, SpinResponseIntegrand, fock_spin,
                          kernel_spin, response_fock_spin, response_fock_st)
from .codegen import (CollapsedKernel, GeneratedFunction, collapse,
                      compile_function, generate, generate_collapsed)
from .cbackend import emit_c, scal_order
from .mo import mo_transform, orbital_gradient, orbital_hessian
from .response import (ResponseIntegrand, contracted_derivative, pert_field,
                       perturbed_variable, response_fock)
from .algebra import (orbital_rotation_dm, perturbed_dm_order, project_ov,
                      quadratic_sigma_xc, response_sigma_xc, rpa_sigma,
                      tda_sigma, transition_dm, unit_rotation)

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
    "SpinResponseIntegrand",
    "fock_spin",
    "kernel_spin",
    "response_fock_spin",
    "response_fock_st",
    "CollapsedKernel",
    "GeneratedFunction",
    "collapse",
    "emit_c",
    "scal_order",
    "generate",
    "generate_collapsed",
    "compile_function",
    "mo_transform",
    "orbital_gradient",
    "orbital_hessian",
    "ResponseIntegrand",
    "contracted_derivative",
    "pert_field",
    "perturbed_variable",
    "response_fock",
    "orbital_rotation_dm",
    "perturbed_dm_order",
    "project_ov",
    "quadratic_sigma_xc",
    "response_sigma_xc",
    "rpa_sigma",
    "unit_rotation",
    "tda_sigma",
    "transition_dm",
]
