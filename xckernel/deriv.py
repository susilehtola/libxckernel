"""Directional differentiation with respect to the density matrix.

Every kernel this library produces is obtained by repeatedly applying the
operator

    D_ts[.] = d[.] / dP_ts

to the exchange-correlation energy:

    F_uv      = D_uv[Exc]                (Fock matrix / potential)
    g_uv,ts   = D_ts[F_uv] = D_ts D_uv[Exc]   (AO-basis XC kernel)
    ...                                   (higher response by further D's)

D_ts is a total derivative: it acts on each *P-dependent atom* of the expression
and sums the results (product rule).  There are exactly two kinds of P-atom:

* **primitive field symbols** (rho, grad_rho_i, lapl_rho, tau).  These are the
  ingredients that are linear in P; their derivative is the ingredient seed
  evaluated at the new free indices (t, s) -- basis data, P-independent, so the
  chain terminates.

* **Libxc derivative symbols** (vrho, vsigma, ..., v2rho2, v2rhosigma, ...).
  Differentiating one bumps its order by one variable and multiplies by that
  variable's seed:

      d v_M / dP_ts = sum_{Y in family} v_{M+Y} * (dY/dP_ts).

  This is the chain rule through Libxc's own derivative tower -- Libxc owns every
  d^n Exc / d{ingredient}^n; we only supply d{ingredient}/dP.

The names v_M follow Libxc's C output arrays (vrho, v2rho2, v2rhosigma, ...) so
generated code maps straight onto a Libxc call with the right do_* flags.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Tuple

import sympy as sp

from .basis import Orbital
from .functional import Functional
from .ingredients import INGREDIENTS, PRIM_BY_SYMBOL

#: Canonical Libxc variable order.  Derivative names concatenate variables in
#: this order, so it must match Libxc exactly.  Extension variables beyond
#: the Libxc set (eta: the gradient-projected density Hessian of local-hybrid
#: calibration functions) are appended AFTER the Libxc variables, so every
#: Libxc-only derivative name is unchanged and extension arrays get
#: self-describing names by the same scheme (veta, v2rhoeta, ...).
VARS: Tuple[str, ...] = ("rho", "sigma", "lapl", "tau", "eta")
_VAR_INDEX = {v: i for i, v in enumerate(VARS)}


def libxc_deriv_name(multiset: Counter) -> str:
    """Libxc output-array name for the derivative d^n Exc / prod(vars).

    order 1 -> 'vrho', 'vsigma', ...
    order n>=2 -> 'v<n>' + concatenated 'var[count-if>1]' in canonical order,
                  e.g. Counter(rho=2) -> 'v2rho2', Counter(rho=1,sigma=1)
                  -> 'v2rhosigma'.
    """
    order = sum(multiset.values())
    if order == 1:
        (var,) = list(multiset.elements())
        return "v" + var
    parts = []
    for var in VARS:
        c = multiset.get(var, 0)
        if c:
            parts.append(var + (str(c) if c > 1 else ""))
    return f"v{order}" + "".join(parts)


def _multisets(order: int) -> Iterable[Counter]:
    """All variable multisets of a given order (combinations with repetition)."""
    from itertools import combinations_with_replacement
    for combo in combinations_with_replacement(VARS, order):
        yield Counter(combo)


#: name -> variable multiset, for every Libxc derivative symbol we may meet.
#: Built up to a generous max order so higher response quantities just work.
_MAX_ORDER = 5
LIBXC_MULTISET: Dict[str, Counter] = {}
for _o in range(1, _MAX_ORDER + 1):
    for _ms in _multisets(_o):
        LIBXC_MULTISET[libxc_deriv_name(_ms)] = _ms


def libxc_symbol(multiset: Counter) -> sp.Symbol:
    return sp.Symbol(libxc_deriv_name(multiset), real=True)


def _atom_derivative(atom: sp.Symbol, func: Functional,
                     u: Orbital, v: Orbital) -> sp.Expr:
    """d(atom)/dP_uv for a single P-dependent symbol; 0 if P-independent."""
    # Primitive field symbol: derivative is its ingredient seed at (u, v).
    prim = PRIM_BY_SYMBOL.get(atom)
    if prim is not None:
        return prim.seed(u, v)

    # Libxc derivative symbol: bump order by each active variable.  The
    # ingredient is looked up on the FUNCTIONAL, not the global table, so a
    # family may map a Libxc variable to a composite ingredient (e.g. the
    # gauge-corrected tau of current-density DFT).
    ms = LIBXC_MULTISET.get(atom.name)
    if ms is not None:
        by_name = {ing.name: ing for ing in func.ingredients}
        total = sp.Integer(0)
        for Y in VARS:
            ing = by_name.get(Y)
            if ing is None:
                continue
            bumped = ms + Counter({Y: 1})
            total += libxc_symbol(bumped) * ing.seed(u, v)
        return total

    # Basis data (chi, dchi, lapl_chi), weight w: independent of P.
    return sp.Integer(0)


def directional_derivative(expr: sp.Expr, func: Functional,
                           u_label: str, v_label: str) -> sp.Expr:
    """Apply D_uv = d/dP_uv to an integrand expression.

    Applied monomial-wise via the fastpoly representation, like every
    other derivative operator of the library."""
    from .fastpoly import from_expr, seeded_derivative, to_expr
    u = Orbital.make(u_label)
    v = Orbital.make(v_label)

    def seed(atom: sp.Symbol):
        d = _atom_derivative(atom, func, u, v)
        return from_expr(d) if d != 0 else None

    return to_expr(seeded_derivative(from_expr(expr), seed))
