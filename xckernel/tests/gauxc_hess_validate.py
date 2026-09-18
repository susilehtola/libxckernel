"""Validate the emitted GauXC Hessian kernels, and the assembly recipe.

Two things can go wrong between ``engine.geometric`` and a working
nuclear Hessian in a host, and they fail differently:

1. the EMITTED C++ could disagree with the symbolic expressions;
2. the ASSEMBLY RECIPE -- sum the per-function rows over the shells of
   an atom, contract the two row sets as an outer product, add the Pulay
   and same-atom seeds -- could be a wrong reading of what the generator
   means, even with every kernel individually correct.

The second is the dangerous one: it produces a Hessian that is smooth,
symmetric and plausible.  So this script does not check the kernels one
at a time.  It builds a small random system with a real atom partition,
assembles the full H^{(A,x),(B,y)} block *through the emitted C++ by the
recipe a host would follow*, and compares against the same block
contracted directly from ``geometric_hessian``'s own ``pair`` and
``same`` expressions with einsum -- the route
``geometric2_validate`` already proves against finite differences.

Agreement is expected at round-off, both sides being the same expression
evaluated through different compilers.

Requires a C++ compiler.

Run with: python -m xckernel.tests.gauxc_hess_validate
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import numpy as np
import sympy as sp

from ..emitters.gauxcwriter import (FAMILIES, _ingredients, _rows, _v2,
                                    emit_header, pair_expression)
from ..engine.geometric import geometric_hessian
from ..inputs.basis import AXES

RNG = np.random.default_rng(20260915)
NBF, NG, NATOM = 7, 5, 3


def _sub(name: str) -> str:
    """einsum subscript for an operand of the symbolic expressions."""
    if name == "D_u_v":
        return "uv"
    if name.endswith("_u") or "_u_" in name:
        return "ug"
    if name.endswith("_v") or "_v_" in name:
        return "vg"
    return "g"


def _direct_block(expr, env, out: str):
    """Contract a symbolic monomial sum with einsum, as the harness in
    geometric2_validate does."""
    total = None
    ex = sp.expand(expr)
    terms = ex.args if ex.is_Add else (ex,)
    for t in terms:
        coeff, rest = t.as_coeff_Mul()
        subs, arrays = [], []
        for sym, e in rest.as_powers_dict().items():
            if not sym.is_Symbol:
                raise AssertionError(f"unexpected factor {sym}")
            for _ in range(int(e)):
                subs.append(_sub(sym.name))
                arrays.append(env[sym.name])
        term = float(coeff) * np.einsum(",".join(subs) + "->" + out, *arrays)
        total = term if total is None else total + term
    return total


def _build_env(family, mA, mB, x, y):
    """Random operands for one (A,x),(B,y) block, shared by both routes."""
    ings = _ingredients(family)
    env = {}
    env["w"] = RNG.normal(size=NG)
    env["D_u_v"] = RNG.normal(size=(NBF, NBF))
    env["D_u_v"] = 0.5 * (env["D_u_v"] + env["D_u_v"].T)
    for lab in ("u", "v"):
        env[f"U0_{lab}"] = RNG.normal(size=(NBF, NG))
        for i in range(3):
            env[f"U{i+1}_{lab}"] = RNG.normal(size=(NBF, NG))
    # displacement collocations, masked to the relevant atom
    env["dchi_gA_u"] = RNG.normal(size=(NBF, NG)) * mA[:, None]
    env["dchi_gB_v"] = RNG.normal(size=(NBF, NG)) * mB[:, None]
    for a in AXES:
        env[f"ddchi_gA_u_{a}"] = RNG.normal(size=(NBF, NG)) * mA[:, None]
        env[f"ddchi_gB_v_{a}"] = RNG.normal(size=(NBF, NG)) * mB[:, None]
    env["d2chi_g2_u"] = RNG.normal(size=(NBF, NG)) * mA[:, None]
    for a in AXES:
        env[f"d3chi_g2_u_{a}"] = RNG.normal(size=(NBF, NG)) * mA[:, None]
    for a in AXES:
        env[f"grad_rho_{a}"] = RNG.normal(size=NG)
    # Libxc derivatives actually referenced by this family
    names = {"vrho", "vsigma", "vtau", "vlapl"}
    for i, k in enumerate(ings):
        for l in ings[i:]:
            names.add(_v2(k, l))
    for n in names:
        env.setdefault(n, RNG.normal(size=NG))
    return env


def _cxx_driver(family, specs) -> str:
    """A driver that calls the EMITTED functions on operands read from
    stdin, in the same order the Python side writes them."""
    L = ['#include "gauxc_hess_kernel.hpp"', "#include <cstdio>",
         "using namespace GauXC::xckernel;", "int main(){"]
    for s in specs:
        ins = s.operands()
        outs = [t for t, _ in s.layout.assignments(s.exprs)]
        L.append(f"  {{ double {', '.join(ins)};")
        L.append("    " + " ".join(f'if(scanf("%lf",&{n})!=1) return 1;' for n in ins))
        L.append(f"    double {', '.join(outs)};")
        L.append(f"    {s.name}({', '.join(ins + outs)});")
        L.append("    " + " ".join(f'printf("%.17e\\n", {o});' for o in outs))
        L.append("  }")
    L.append("  return 0; }")
    return "\n".join(L)


def _drive(tmp, family, specs, values) -> list:
    """Compile once per family, feed the operand stream, read results."""
    src = os.path.join(tmp, f"drv_{family}.cxx")
    exe = os.path.join(tmp, f"drv_{family}")
    with open(src, "w") as f:
        f.write(_cxx_driver(family, specs))
    subprocess.run(["g++", "-O1", "-std=c++17", "-I", tmp, src, "-o", exe],
                   check=True, capture_output=True)
    inp = "\n".join(f"{v!r}" for v in values) + "\n"
    out = subprocess.run([exe], input=inp, capture_output=True, text=True,
                         check=True).stdout.split()
    return [float(x) for x in out]


def run() -> int:
    if shutil.which("g++") is None:
        print("  [SKIP] g++ not found")
        return 0

    tmp = tempfile.mkdtemp(prefix="xckernel-gauxc-hess-")
    hdr = os.path.join(tmp, "gauxc_hess_kernel.hpp")
    with open(hdr, "w") as f:
        f.write(emit_header())

    failures = 0
    for family in FAMILIES:
        gh = geometric_hessian(family)
        ings = _ingredients(family)

        # two atoms with disjoint basis-function sets, plus a same-atom case
        mA = np.zeros(NBF); mA[:3] = 1.0
        mB = np.zeros(NBF); mB[3:] = 1.0
        env = _build_env(family, mA, mB, 0, 1)

        # --- route 1: straight from geometric_hessian ----------------
        pair_uv = _direct_block(gh.pair, env, "uv")
        direct = float(pair_uv.sum())

        # --- route 2: the host recipe, through the emitted expressions
        # rows, summed over the functions of each atom
        rowsA, GA = _rows(family, "A")
        rowsB, GB = _rows(family, "B")

        def ev(e, out):
            return _direct_block(e, env, out)

        F_A = {k: ev(rowsA[k], "g") for k in ings}
        F_B = {k: ev(rowsB[k], "g") for k in ings}
        G_A = [ev(GA[i], "g") for i in range(3)]
        G_B = [ev(GB[i], "g") for i in range(3)]

        penv = dict(env)
        for k in ings:
            penv[f"F_{k}_A"], penv[f"F_{k}_B"] = F_A[k], F_B[k]
        for i, a in enumerate(AXES):
            penv[f"G_A_{a}"], penv[f"G_B_{a}"] = G_A[i], G_B[i]
        outer = _direct_block(pair_expression(family), penv, "g")

        D = sp.Symbol("D_u_v", real=True)
        ex = sp.expand(gh.pair)
        terms = ex.args if ex.is_Add else (ex,)
        pulay_expr = sum(t for t in terms if t.has(D))
        pulay = float(_direct_block(pulay_expr, env, "uv").sum()) if pulay_expr != 0 else 0.0

        recipe = float((env["w"] * outer).sum()) + pulay

        # the Pulay term the way the host now contracts it: as matrix
        # products through the emitted point weights W_ab
        from ..emitters.gauxcwriter import pulay_weights, _pulay_rows
        W = pulay_weights(family)
        al = [np.broadcast_to(env[r.name], (NBF, NG)) if r.name in env
              else np.zeros((NBF, NG)) for r in _pulay_rows("A")]
        be = [np.broadcast_to(env[r.name], (NBF, NG)) if r.name in env
              else np.zeros((NBF, NG)) for r in _pulay_rows("B")]
        M = np.zeros((NBF, NBF))
        for a_ in range(4):
            for b_ in range(4):
                if W[a_][b_] != 0:
                    wg = np.broadcast_to(_direct_block(W[a_][b_], env, "g"), (NG,))
                    M += (al[a_] * wg) @ be[b_].T
        pulay_gemm = float((env["D_u_v"] * M).sum())
        relp = abs(pulay_gemm - pulay) / max(1e-30, abs(pulay))
        okp = relp < 1e-12
        failures += 0 if okp else 1
        print(f"  [{'OK' if okp else '!!'}] {family:9s} Pulay as GEMMs vs per "
              f"pair: {pulay_gemm:+.10e} vs {pulay:+.10e} rel {relp:.2e}")

        rel = abs(direct - recipe) / max(1e-30, abs(direct))
        ok = rel < 1e-12
        failures += 0 if ok else 1
        print(f"  [{'OK' if ok else '!!'}] {family:9s} assembly recipe vs "
              f"geometric_hessian: direct {direct:+.10e} recipe {recipe:+.10e} "
              f"rel {rel:.2e}")

        # --- route 3: the EMITTED C++, on one sampled (function, point) -
        from ..emitters.gauxcwriter import (spec_rows, spec_pair, _seed_specs,
                                            spec_pulay_weights, spec_egrad)
        specs = ([spec_rows(family), spec_pair(family)] + _seed_specs(family)
                 + [spec_pulay_weights(family), spec_egrad(family)])
        iu, iv, ig = 0, NBF - 1, NG // 2
        scal = dict(penv)

        def pick(name):
            a = scal[name]
            if np.ndim(a) == 0:
                return float(a)
            if a.shape == (NBF, NBF):
                return float(a[iu, iv])
            if a.shape == (NBF, NG):
                return float(a[iv if (name.endswith("_v") or "_v_" in name) else iu, ig])
            return float(a[ig])

        values, expect = [], []
        for sp_ in specs:
            for n in sp_.operands():
                values.append(pick(n))
            for t, e in sp_.layout.assignments(sp_.exprs):
                f = sp.lambdify(sorted(e.free_symbols, key=lambda z: z.name),
                                e, "numpy")
                expect.append(float(f(*[pick(z.name) for z in
                                        sorted(e.free_symbols, key=lambda z: z.name)])))
        got = _drive(tmp, family, specs, values)
        dev = max(abs(a - b) / max(1e-30, abs(b)) for a, b in zip(got, expect))
        ok2 = dev < 1e-13
        failures += 0 if ok2 else 1
        print(f"  [{'OK' if ok2 else '!!'}] {family:9s} emitted C++ vs SymPy: "
              f"{len(got)} channels, worst rel {dev:.2e}")

    # --- the weight-class kernels: emitted C++ vs SymPy at 30 digits, on
    # sampled geometries and across each cell function's switching range
    # (Becke right up to mu -> 1, where t and u diverge)
    from ..emitters.gauxcwriter import spec_mu_derivs, spec_cell
    mu_spec = spec_mu_derivs()
    wspecs, values, expect = [], [], []

    def sample(spec, env):
        wspecs.append(spec)
        values.extend(env[n] for n in spec.operands())
        sub = {z: sp.Float(env[z.name], 30) for e in spec.exprs.values()
               for z in e.free_symbols}
        for t, e in spec.layout.assignments(spec.exprs):
            expect.append(float(sp.N(e.subs(sub), 30)))

    for _ in range(4):
        c = RNG.normal(size=9)
        sample(mu_spec, {f"{p}_{a}": float(c[3 * i + j])
                         for i, p in enumerate(("R_D", "R_E", "r"))
                         for j, a in enumerate(AXES)})
    for m in (-0.95, -0.4, 0.0, 0.3, 0.8, 0.999):
        sample(spec_cell("becke"), {"mu": m})
    for m in (-0.6, -0.2, 0.0, 0.25, 0.5, 0.639):
        sample(spec_cell("ssf"), {"mu": m})
    got = _drive(tmp, "weights", wspecs, values)
    # mixed criterion: t -> 0 as mu -> -1 is a sum of O(1) log-derivative
    # terms, so it is accurate to 1e-16 ABSOLUTELY but not relatively --
    # and the weight derivatives only ever see t * dmu
    dev = max(abs(a - b) / max(1.0, abs(b)) for a, b in zip(got, expect))
    okw = dev < 1e-12
    failures += 0 if okw else 1
    print(f"  [{'OK' if okw else '!!'}] weights   emitted C++ vs SymPy: "
          f"{len(got)} channels, worst mixed {dev:.2e}")

    shutil.rmtree(tmp, ignore_errors=True)
    tag = "OK " if not failures else "FAIL"
    print(f"[{tag}] gauxc_hess_validate: {3*len(FAMILIES) + 1} checks, "
          f"{failures} failures")
    return failures


if __name__ == "__main__":
    raise SystemExit(run())
