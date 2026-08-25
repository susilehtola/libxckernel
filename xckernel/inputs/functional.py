"""The exchange-correlation functional as seen from our side of the fence.

Libxc owns the functional.  For a Fock matrix we need its *first* derivatives
with respect to each input variable -- the quantities Libxc returns as ``vrho``,
``vsigma``, ``vlapl``, ``vtau``.  We treat each as an opaque symbol: the AD in
this library differentiates the *ingredients* with respect to P, never the
functional itself.  That division of labour is the whole point -- Libxc supplies
d Exc / d{ingredient}; we supply d{ingredient} / dP.

Symbol naming follows Libxc's C output arrays so generated code maps directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import sympy as sp

from .ingredients import FAMILIES, Ingredient

#: Libxc first-derivative output name for each ingredient variable.
VNAME: Dict[str, str] = {
    "rho": "vrho",
    "sigma": "vsigma",
    "lapl": "vlapl",
    "tau": "vtau",
    # beyond-Libxc extension variable (local-hybrid calibration functions):
    # the gradient-projected density Hessian.  Array names follow the same
    # scheme; the derivative arrays are supplied by the host's functional
    # implementation until a functional library exposes them.
    "eta": "veta",
}


@dataclass(frozen=True)
class Functional:
    """A functional family and the Libxc derivative symbols it exposes."""

    family: str
    ingredients: List[Ingredient]

    @classmethod
    def of_family(cls, family: str, coords=None) -> "Functional":
        """The family's ingredient set, optionally in a curvilinear system.

        ``coords=None`` (or Cartesian) reproduces the historical table
        exactly; any other system rebuilds the metric-carrying seeds and
        refuses the families whose seeds are Cartesian-only.
        """
        from .ingredients import check_coordinates, families_for
        table = families_for(coords)
        if family not in table:
            check_coordinates(family, coords)
            raise ValueError(
                f"unknown family {family!r}; known: {sorted(table)}")
        check_coordinates(family, coords)
        return cls(family=family, ingredients=table[family])

    def vsymbol(self, ingredient: Ingredient) -> sp.Symbol:
        """The opaque Libxc derivative symbol d Exc / d ingredient."""
        return sp.Symbol(VNAME[ingredient.name], real=True)
