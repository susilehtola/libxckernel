"""IR compaction: find the compact contraction form of collapsed-kernel
coefficients before emission.

Two structure-aware passes over the per-pattern monomial lists (the same
IR the emitters consume), returning definitions for per-point
intermediates plus rewritten monomials:

* **dot contraction** -- triples of monomials identical up to a Cartesian
  pair g1_i g2_i (same i, equal coefficients) are one dot product:
  they collapse to a single monomial carrying the scalar intermediate
  dot(g1, g2) = sum_i g1_i g2_i. This discovers the sigma-style
  intermediates (gamma_k, gamma_abk, ...) production codes hand-write.

* **common-factor hoisting** -- across a direction-indexed family of
  coefficients c_x, c_y, c_z, sub-sums S appearing as S * v_i in every
  direction (v a registered vector group) are hoisted to one per-point
  intermediate: the "v2_val" idiom. Only sums of at least two terms are
  hoisted.

Monomials are (coeff, ((name, exp), ...)) pairs, as in the collapsed IR.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

Monomial = Tuple[float, Tuple[Tuple[str, int], ...]]

#: vector-group registry: group name -> component operand names (x, y, z)
VECTOR_GROUPS: Dict[str, Tuple[str, str, str]] = {}
for _g in ("grad_rho", "grad_rho_a", "grad_rho_b"):
    VECTOR_GROUPS[_g] = tuple(f"{_g}_{ax}" for ax in "xyz")
    VECTOR_GROUPS[f"{_g}_p1"] = tuple(f"{_g}_p1_{ax}" for ax in "xyz")

_COMPONENT_OF = {comp: (g, i) for g, comps in VECTOR_GROUPS.items()
                 for i, comp in enumerate(comps)}


def _strip_pair(factors, g1: str, g2: str, i: int):
    """Remove one power of g1_i and one of g2_i from a factor tuple, or
    return None if not present."""
    c1, c2 = VECTOR_GROUPS[g1][i], VECTOR_GROUPS[g2][i]
    need = {c1: 1}
    need[c2] = need.get(c2, 0) + 1
    fac = dict(factors)
    for name, cnt in need.items():
        if fac.get(name, 0) < cnt:
            return None
        fac[name] -= cnt
        if fac[name] == 0:
            del fac[name]
    return tuple(sorted(fac.items()))


def contract_dots(monos: Sequence[Monomial]):
    """Collapse isotropic Cartesian pairs into dot-product intermediates.

    Returns (new_monomials, defs) with defs: dot name -> (g1, g2)."""
    monos = list(monos)
    defs: Dict[str, Tuple[str, str]] = {}
    groups = sorted(VECTOR_GROUPS)
    changed = True
    while changed:
        changed = False
        for a in range(len(groups)):
            for b in range(a, len(groups)):
                g1, g2 = groups[a], groups[b]
                # index monomials by (stripped remainder, coeff) per component
                buckets: Dict[Tuple, List[Tuple[int, int]]] = {}
                for im, (coeff, factors) in enumerate(monos):
                    for i in range(3):
                        rest = _strip_pair(factors, g1, g2, i)
                        if rest is not None:
                            buckets.setdefault((rest, round(coeff, 12)), []).append((i, im))
                for (rest, coeff), hits in buckets.items():
                    comps = {}
                    for i, im in hits:
                        comps.setdefault(i, im)
                    if len(comps) < 3:
                        continue
                    used = sorted(set(comps.values()))
                    if len(used) < 3:
                        continue  # same monomial matching several ways
                    name = f"dot_{g1}_{g2}" if g1 <= g2 else f"dot_{g2}_{g1}"
                    defs[name] = (g1, g2)
                    keep = [m for im, m in enumerate(monos) if im not in used]
                    newfac = tuple(sorted(list(rest) + [(name, 1)]))
                    keep.append((coeff, newfac))
                    monos = keep
                    changed = True
                    break
                if changed:
                    break
            if changed:
                break
    return monos, defs


def hoist_common(c_by_i: Dict[int, Sequence[Monomial]]):
    """Hoist direction-shared sub-sums S from c_i = ... + S * v_i.

    Returns (defs, new_c_by_i) with defs: hoist name -> (vector group,
    list of remainder monomials). Only sums of >= 2 terms are hoisted."""
    defs: Dict[str, Tuple[str, List[Monomial]]] = {}
    out = {i: list(c_by_i[i]) for i in range(3)}
    nh = 0
    for g, comps in sorted(VECTOR_GROUPS.items()):
        # candidate remainders per direction
        rem: List[Dict[Tuple, float]] = []
        for i in range(3):
            d: Dict[Tuple, float] = {}
            for coeff, factors in out[i]:
                fac = dict(factors)
                if fac.get(comps[i], 0) == 1 and \
                        not any(fac.get(c, 0) for c in comps if c != comps[i]):
                    del fac[comps[i]]
                    d[tuple(sorted(fac.items()))] = d.get(tuple(sorted(fac.items())), 0.0) + coeff
            rem.append(d)
        shared = [k for k in rem[0]
                  if all(abs(rem[i].get(k, 0.0) - rem[0][k]) < 1e-12 for i in (1, 2))
                  and abs(rem[0][k]) > 0]
        if len(shared) < 2:
            continue
        name = f"hsum_{g}_{nh}"
        nh += 1
        defs[name] = (g, [(rem[0][k], k) for k in sorted(shared)])
        for i in range(3):
            kept = []
            for coeff, factors in out[i]:
                fac = dict(factors)
                if fac.get(comps[i], 0) == 1 and \
                        not any(fac.get(c, 0) for c in comps if c != comps[i]):
                    rest = dict(fac)
                    del rest[comps[i]]
                    if tuple(sorted(rest.items())) in shared:
                        continue
                kept.append((coeff, factors))
            kept.append((1.0, tuple(sorted([(name, 1), (comps[i], 1)]))))
            out[i] = kept
    return defs, out
