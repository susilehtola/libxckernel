"""XC kernel elements as repeated density-matrix derivatives of Exc.

    energy    Exc
    fock      F_uv    = D_uv[Exc]
    kernel    g_uvts  = D_ts[F_uv]          (AO-basis f_xc, the TDDFT/CPHF core)
    ...       higher orders by further D's

The energy integrand is w * e, with e the Libxc energy density (per unit volume);
its first density-matrix derivative reproduces exactly the Fock integrand built
in fock.py, and each further derivative is one application of D from deriv.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import sympy as sp

from .deriv import directional_derivative
from .fock import FockIntegrand, fock_integrand
from ..inputs.functional import Functional


@dataclass
class KernelIntegrand:
    """Per-grid integrand of an n-th order density-matrix derivative of Exc,
    with one free orbital-index pair per derivative order."""

    functional: Functional
    index_pairs: List[Tuple[str, str]]   # e.g. [('u','v'), ('t','s')]
    expr: sp.Expr


def kernel_integrand(family: str,
                     index_pairs: List[Tuple[str, str]],
                     coords=None) -> KernelIntegrand:
    """Build the integrand of D_{i_n j_n} ... D_{i_1 j_1}[Exc].

    ``index_pairs`` gives the free index labels for each derivative, outermost
    last.  ``[('u','v')]`` is the Fock matrix; ``[('u','v'),('t','s')]`` is the
    AO-basis XC kernel g_uv,ts.
    """
    if not index_pairs:
        raise ValueError("need at least one index pair (the Fock matrix)")

    func = Functional.of_family(family, coords)
    fi: FockIntegrand = fock_integrand(family, *index_pairs[0], coords=coords)
    expr = fi.expr
    for (u_label, v_label) in index_pairs[1:]:
        expr = directional_derivative(expr, func, u_label, v_label,
                                      coords)

    return KernelIntegrand(functional=func, index_pairs=list(index_pairs),
                           expr=expr)


def fock(family: str, u="u", v="v", coords=None) -> KernelIntegrand:
    """The XC Fock matrix integrand, optionally in a curvilinear system."""
    return kernel_integrand(family, [(u, v)], coords=coords)


def xc_kernel(family: str, u="u", v="v", t="t", s="s") -> KernelIntegrand:
    """AO-basis XC kernel g_uv,ts = d^2 Exc / dP_uv dP_ts."""
    return kernel_integrand(family, [(u, v), (t, s)])
