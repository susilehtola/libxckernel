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
}


@dataclass(frozen=True)
class Functional:
    """A functional family and the Libxc derivative symbols it exposes."""

    family: str
    ingredients: List[Ingredient]

    @classmethod
    def of_family(cls, family: str) -> "Functional":
        if family not in FAMILIES:
            raise ValueError(
                f"unknown family {family!r}; known: {sorted(FAMILIES)}")
        return cls(family=family, ingredients=FAMILIES[family])

    def vsymbol(self, ingredient: Ingredient) -> sp.Symbol:
        """The opaque Libxc derivative symbol d Exc / d ingredient."""
        return sp.Symbol(VNAME[ingredient.name], real=True)
