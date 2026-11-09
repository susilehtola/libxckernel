"""The product planner shared by the NumPy and C/C++ emitters
(codegen.symmetry_split): for the matrix kernels of every family,

* the planned half products reassemble the kernel exactly: each half
  pattern (a, b) with sign s contributes its monomials at (a, b) and s
  times them at (b, a), and the sum equals the collapsed pattern table
  coefficient for coefficient;
* the plan is never dearer than the plain grouping;
* both emitters pay exactly the planned number of matrix products (the
  GEMMs of the generated C++ headers, single and batched, and the
  matmuls of the generated NumPy function), so neither can drift from
  the shared plan again;
* the counts of the reference kernels are pinned (gga o1: one GEMM).

Symbolic only: nothing is compiled or evaluated."""

from __future__ import annotations

import re

from ..catalog import CatalogEntry, _integrand_for, _kind_kernels, abi_kind
from ..emitters.cbackend import (emit_batch_hpp, emit_kernel_hpp,
                                 emit_tower_hpp)
from ..emitters.codegen import (_gemm_plan, collapse, generate_collapsed,
                                symmetry_split)

FAMILIES = ("lda", "gga", "mgga_tau", "mgga_lapl", "mgga", "cmgga_tau",
            "hmgga")
#: the families whose geometric and GIAO kernels are checked too
KIND_FAMILIES = ("lda", "gga", "mgga_tau")

#: pinned GEMM counts of the restricted matrix kernels (o1 and o2 alike)
PINNED = {"lda": 1, "gga": 1, "mgga_tau": 4, "mgga_lapl": 4, "mgga": 4,
          "cmgga_tau": 4, "hmgga": 4}


def _table(patterns):
    t = {}
    for u, v, monos in patterns:
        for coeff, factors in monos:
            k = (u, v, tuple(sorted(factors)))
            t[k] = t.get(k, 0.0) + coeff
    return {k: c for k, c in t.items() if c != 0.0}


def _reassemble(parts):
    t = {}
    for sub, sign in parts:
        for u, v, monos in sub.patterns:
            for coeff, factors in monos:
                f = tuple(sorted(factors))
                for k, c in (((u, v, f), coeff), ((v, u, f), sign * coeff)):
                    t[k] = t.get(k, 0.0) + c
    return {k: c for k, c in t.items() if c != 0.0}


def _planned(ck, plain_cost=None):
    """(planned products, plain products, parts or None)."""
    plain = len(_gemm_plan(ck.patterns)[1]) if plain_cost is None \
        else plain_cost
    parts = symmetry_split(ck, plain_cost=plain_cost)
    if parts is None:
        return plain, plain, None
    return sum(len(_gemm_plan(sub.patterns)[1]) for sub, _ in parts), \
        plain, parts


def check_block(label, ck, cxx_gemms=None, numpy_products=None,
                batch_gemms=None):
    """Exact reassembly and plan cost of one collapsed matrix block."""
    fails = []
    cost, plain, parts = _planned(ck)
    if parts is not None and _reassemble(parts) != _table(ck.patterns):
        fails.append("half products do not reassemble the kernel")
    if cost > plain:
        fails.append(f"plan {cost} dearer than plain {plain}")
    if cxx_gemms is not None and cxx_gemms != cost:
        fails.append(f"C++ emits {cxx_gemms} GEMMs, plan {cost}")
    if numpy_products is not None:
        ucost, _, _ = _planned(
            ck, plain_cost=len({u for u, _, _ in ck.patterns}))
        if numpy_products != ucost:
            fails.append(f"NumPy emits {numpy_products} products, "
                         f"plan {ucost}")
    if batch_gemms is not None:
        # the stacked GEMM shares the V-side factor
        vcost, _, _ = _planned(
            ck, plain_cost=len({v for _, v, _ in ck.patterns}))
        if batch_gemms != vcost:
            fails.append(f"batched C++ emits {batch_gemms} GEMMs, "
                         f"plan {vcost}")
    print(f"  [{'OK ' if not fails else 'FAIL'}] {label:30s} "
          f"{plain:3d} -> {cost:3d} products"
          + (f"  ({'; '.join(fails)})" if fails else ""))
    return cost, len(fails) > 0


def main():
    checks = failures = 0
    for fam in FAMILIES:
        for spin in ("r", "ua"):
            for order in (1, 2):
                e = CatalogEntry(fam, spin, order)
                ki = _integrand_for(e)
                ck = collapse(ki)
                hpp = emit_kernel_hpp(ck, e.name)
                src = generate_collapsed(ki).source
                batch = (emit_batch_hpp(ck, e.name + "_batch")
                         .count("gemm_rect(nxb") if order == 2 else None)
                cost, bad = check_block(
                    e.name, ck, cxx_gemms=hpp.count("gemm_nt("),
                    numpy_products=len(re.findall(r"@", src)),
                    batch_gemms=batch)
                if spin == "r" and cost != PINNED[fam]:
                    print(f"  [FAIL] {e.name}: {cost} GEMMs, pinned "
                          f"{PINNED[fam]}")
                    bad = True
                checks += 1
                failures += bad
    for fam in KIND_FAMILIES:
        for e in (CatalogEntry(fam, "r", 1, kind="f1"),
                  CatalogEntry(fam, "r", 1, kind="fg"),
                  CatalogEntry(fam, "r", 1, giao=True)):
            blocks, scal_ck, computed = _kind_kernels(e)
            hpp = emit_tower_hpp(blocks, scal_ck, e.name, abi_kind(e),
                                 computed)
            costs = []
            for r, ck in enumerate(blocks):
                cost, bad = check_block(f"{e.name}[{r}]", ck)
                costs.append(cost)
                checks += 1
                failures += bad
            if hpp.count("gemm_nt(") != sum(costs):
                print(f"  [FAIL] {e.name}: C++ emits "
                      f"{hpp.count('gemm_nt(')} GEMMs, plan {sum(costs)}")
                failures += 1
            checks += 1
    status = "OK " if failures == 0 else "FAIL"
    print(f"[{status}] planner_validate: {checks} checks, "
          f"{failures} failures")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
