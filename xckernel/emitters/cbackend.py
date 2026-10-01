"""C backend: emit self-contained, table-driven C kernels.

Design: the largest kernels have >100k scalar monomials per coefficient sum;
emitting them as inline expressions would defeat any compiler, and full CSE is
future work.  Instead each kernel is emitted as **static tables + a small
fixed evaluator**:

  * a flat monomial table per pattern: coefficients, factor-list offsets, and
    factor ids indexing an ordered list of per-point scalar arrays
    (powers > 1 encoded by factor repetition);
  * stage A: one loop over grid points walking the table -> coefficient c(g);
  * stage B: out[u,v] += sum_g U[u,g] * c[g] * V[v,g] as GEMMs.  Patterns
    sharing a basis factor on one side are merged first -- W = sum_p c_p o X_p
    -- so each distinct shared factor costs one GEMM (GGA: 7 patterns -> 4
    GEMMs, meta-GGA: 12 -> 5).  The grid is processed in blocks, bounding
    the scratch to nbf x block.  With XCKERNEL_USE_BLAS the GEMM is the
    Fortran BLAS dgemm_/sgemm_; otherwise (and for precisions BLAS lacks) a
    portable loop nest is used.

ABI (v2, one perturbation-batch entry per call; hosts loop the batch):
the derivative-tower interface of tower.py.

  int <name>(int64_t npts, int64_t nbf,
             const double* chi,        /* (ncomp, nbf, npts) collocation
                                          tower, components to
                                          <name>_chi_order */
             const double* const* scal,/* per-point tower operands, then
                                          the Libxc derivative arrays:
                                          see <name>_scal_names */
             double* out);             /* accumulated (+=) */

The Fock-diagonal (*_o1_diag), gradient-row (*_g1, which also takes
Dchi = D chi) and per-point gradient (*_gg, no collocation) kernels share
it; see ABI_KINDS.  emit_c, the standalone C99 demonstrator used by
cbackend_validate, keeps the engine's own operands (chi, dchi, lapl_chi,
hess_chi and the engine's per-point names).
"""

from __future__ import annotations

import re
from typing import List, Tuple

from .codegen import CollapsedKernel, _classify

#: basis-factor code -> (array expression, needs)
_BASIS = {
    "chi": ("chi", None),
    "dchi[0]": ("dchi + (int64_t)0*nbf*npts", None),
    "dchi[1]": ("dchi + (int64_t)1*nbf*npts", None),
    "dchi[2]": ("dchi + (int64_t)2*nbf*npts", None),
    "lapl_chi": ("lapl_chi", "lapl"),
    **{f"hess_chi[{k}]": (f"hess_chi + (int64_t){k}*nbf*npts", "hess")
       for k in range(6)},
}

#: packed symmetric-tensor components (density Hessian), canonical order.
_H6_COMPS = ("xx", "xy", "xz", "yy", "yz", "zz")


def scal_order(ck: CollapsedKernel) -> List[str]:
    """Ordered scalar-operand names for the `scal` pointer array.

    Vector parameters are flattened to components (grad_rho -> grad_rho_x..z),
    matching the manifest parameter order.
    """
    order: List[str] = []
    for p in ck.params:
        if p in ("chi", "dchi", "lapl_chi", "hess_chi", "Rchi", "Rdchi",
                 "Rlapl_chi"):
            continue
        if p.startswith("hess_rho"):
            # packed symmetric tensor: six components
            for comp in _H6_COMPS:
                order.append(f"{p}_{comp}")
        elif p == "rg":
            # grid coordinates (London-orbital kernels)
            for ax in ("x", "y", "z"):
                order.append(f"rg_{ax}")
        elif p.startswith("jpgrad"):
            # d_d jp_i, i the current component, d the derivative
            for i in ("x", "y", "z"):
                for d in ("x", "y", "z"):
                    order.append(f"{p}_{i}_{d}")
        elif p.startswith("d3rho"):
            # the ten third derivatives of the density
            from .tower import components
            for comp in components(3)[10:]:
                order.append(f"{p}_{comp}")
        elif p.startswith(("grad_rho", "jp", "dgrad_rho", "grad_tau",
                           "grad_lapl_rho")):
            for ax in ("x", "y", "z"):
                # grad_rho -> grad_rho_x; grad_rho_a_p1 -> grad_rho_a_p1_x;
                # dgrad_rho_g (the direction-resolved density-Hessian row of
                # the grid-response class) is a 3-vector too, despite not
                # sharing the grad_rho prefix
                order.append(f"{p}_{ax}")
        else:
            order.append(p)
    return order


def _scal_index(ck: CollapsedKernel) -> dict:
    return {name: i for i, name in enumerate(scal_order(ck))}


def _gemm_plan(patterns) -> Tuple[str, list]:
    """Group patterns into GEMMs sharing one basis factor.

    Returns (side, groups): side "u" merges patterns with a common U factor
    (out += U . W^T, W = sum_p c_p o V_p), side "v" those with a common V
    factor (out += W . V^T, W = sum_p c_p o U_p) -- whichever side has fewer
    distinct factors.  groups is an ordered list of
    (shared_factor, [(pattern_index, other_factor), ...]).
    """
    us = {u for u, _, _ in patterns}
    vs = {v for _, v, _ in patterns}
    side = "u" if len(us) <= len(vs) else "v"
    groups: dict = {}
    for ip, (u, v, _) in enumerate(patterns):
        key, other = (u, v) if side == "u" else (v, u)
        groups.setdefault(key, []).append((ip, other))
    return side, list(groups.items())


def _stage_b_calls(ck: CollapsedKernel, bexpr, stage_a_call, indent="        ",
                   gemm="gemm_nt", acc="accumulate", cast=lambda x: x,
                   out="out"):
    """The per-block body: stage A per pattern, merge, one GEMM per group
    (or, with gemm="rowdot", one row-wise dot product per group).

    bexpr: basis code -> (pointer expression at the block's first point,
    leading dimension)."""
    side, groups = _gemm_plan(ck.patterns)
    lines: List[str] = []
    for shared, members in groups:
        for k, (ip, other) in enumerate(members):
            ptr, ld = bexpr[other]
            lines.append(indent + stage_a_call(ip))
            lines.append(f"{indent}{acc}(bk, nbf, c, {ptr}, {ld}, W, "
                         f"{int(k == 0)});")
        sh, ld = bexpr[shared]
        if side == "u":
            lines.append(f"{indent}{gemm}(nbf, bk, {sh}, {ld}, {cast('W')}, "
                         f"bk, {out});")
        else:
            lines.append(f"{indent}{gemm}(nbf, bk, {cast('W')}, bk, {sh}, "
                         f"{ld}, {out});")
    return lines


#: the fixed C99 evaluator shared by every emit_c translation unit
_C99_EVALUATOR = r"""#ifndef XCKERNEL_GRID_BLOCK
#define XCKERNEL_GRID_BLOCK 1024
#endif
#ifdef XCKERNEL_USE_BLAS
#ifndef XCKERNEL_BLAS_INT
#define XCKERNEL_BLAS_INT int
#endif
void dgemm_(const char* ta, const char* tb, const XCKERNEL_BLAS_INT* m,
            const XCKERNEL_BLAS_INT* n, const XCKERNEL_BLAS_INT* k,
            const double* alpha, const double* a, const XCKERNEL_BLAS_INT* lda,
            const double* b, const XCKERNEL_BLAS_INT* ldb,
            const double* beta, double* c, const XCKERNEL_BLAS_INT* ldc);
#endif

/* stage A on grid points g0 .. g0+bk-1: c[g] = sum_m cf[m] prod_f scal[..] */
static void stage_a(int64_t g0, int64_t bk, int64_t nm,
                    const double* restrict cf,
                    const int32_t* restrict off,
                    const uint16_t* restrict fid,
                    const double* const* restrict scal,
                    double* restrict c) {
    for (int64_t g = 0; g < bk; ++g) {
        double acc = 0.0;
        for (int64_t m = 0; m < nm; ++m) {
            double t = cf[m];
            for (int32_t f = off[m]; f < off[m+1]; ++f)
                t *= scal[fid[f]][g0 + g];
            acc += t;
        }
        c[g] = acc;
    }
}

/* merge one pattern: W(i,g) (=|+=) c(g) X(i,g), X row stride ldx */
static void accumulate(int64_t bk, int64_t n, const double* restrict c,
                       const double* restrict X, int64_t ldx,
                       double* restrict W, int first) {
    for (int64_t i = 0; i < n; ++i) {
        const double* Xi = X + i*ldx;
        double* Wi = W + i*bk;
        if (first)
            for (int64_t g = 0; g < bk; ++g) Wi[g] = c[g] * Xi[g];
        else
            for (int64_t g = 0; g < bk; ++g) Wi[g] += c[g] * Xi[g];
    }
}

/* out(i,j) += sum_g A(i,g) B(j,g); A, B row-major with strides lda, ldb */
static void gemm_nt(int64_t n, int64_t k, const double* A, int64_t lda,
                    const double* B, int64_t ldb, double* out) {
    if (n == 0 || k == 0) return;
#ifdef XCKERNEL_USE_BLAS
    const int64_t imax = sizeof(XCKERNEL_BLAS_INT) >= 8 ? INT64_MAX : INT32_MAX;
    if (n <= imax && k <= imax && lda <= imax && ldb <= imax) {
        /* row-major out = A B^T is column-major out^T = B^T A */
        const XCKERNEL_BLAS_INT bn = (XCKERNEL_BLAS_INT)n,
            bk = (XCKERNEL_BLAS_INT)k, ba = (XCKERNEL_BLAS_INT)lda,
            bb = (XCKERNEL_BLAS_INT)ldb;
        const double one = 1.0;
        dgemm_("T", "N", &bn, &bn, &bk, &one, B, &bb, A, &ba, &one, out, &bn);
        return;
    }
#endif
    for (int64_t i = 0; i < n; ++i) {
        const double* Ai = A + i*lda;
        for (int64_t j = 0; j < n; ++j) {
            const double* Bj = B + j*ldb;
            double s = 0.0;
            for (int64_t g = 0; g < k; ++g) s += Ai[g] * Bj[g];
            out[i*n + j] += s;
        }
    }
}
"""


def emit_c(ck: CollapsedKernel, name: str) -> str:
    """Emit a self-contained C99 translation unit for one kernel."""
    sidx = _scal_index(ck)
    lines: List[str] = [
        "/* machine-generated by xckernel (table-driven pattern-collapsed kernel); do not edit. Copyright (c) 2026 Susi Lehtola. */",
        "#include <stdint.h>",
        "#include <stdlib.h>",
        "",
        f"/* scalar operand order for the `scal` argument: */",
    ]
    names = scal_order(ck)
    lines.append(f"const char* {name}_scal_names[{len(names)}] = {{")
    for n in names:
        lines.append(f'    "{n}",')
    lines += ["};", f"const int {name}_n_scal = {len(names)};", ""]

    # monomial tables per pattern
    pat_meta = []
    for ip, (ufac, vfac, monos) in enumerate(ck.patterns):
        coeffs, offs, fids = [], [0], []
        for coeff, factors in monos:
            coeffs.append(coeff)
            for fname, e in factors:
                fids.extend([sidx[fname]] * e)
            offs.append(len(fids))
        lines.append(f"static const double {name}_c{ip}[] = {{")
        lines.append("    " + ",".join(f"{c!r}" for c in coeffs))
        lines.append("};")
        lines.append(f"static const int32_t {name}_o{ip}[] = {{")
        lines.append("    " + ",".join(str(o) for o in offs))
        lines.append("};")
        lines.append(f"static const uint16_t {name}_f{ip}[] = {{")
        lines.append("    " + (",".join(str(f) for f in fids) or "0"))
        lines.append("};")
        pat_meta.append((ip, ufac, vfac, len(monos)))
    lines.append("")

    # the fixed evaluator (stage A) + merge + GEMM distributor (stage B)
    lines += _C99_EVALUATOR.splitlines()
    lines += [
        f"int {name}(int64_t npts, int64_t nbf,",
        "           const double* chi, const double* dchi,",
        "           const double* lapl_chi, const double* hess_chi,",
        "           const double* const* scal, double* out) {",
        "    const int64_t blk = npts < XCKERNEL_GRID_BLOCK ? npts"
        " : XCKERNEL_GRID_BLOCK;",
        "    double* c = (double*)malloc((size_t)(blk * (1 + nbf) + 1)"
        " * sizeof(double));",
        "    if (!c) return 1;",
        "    double* W = c + blk;",
        "    for (int64_t g0 = 0; g0 < npts; g0 += blk) {",
        "        const int64_t bk = npts - g0 < blk ? npts - g0 : blk;",
    ]
    nms = {ip: nm for ip, _, _, nm in pat_meta}
    lines += _stage_b_calls(
        ck, {k: (f"{v[0]} + g0", "npts") for k, v in _BASIS.items()},
        lambda ip: (f"stage_a(g0, bk, {nms[ip]}, {name}_c{ip}, {name}_o{ip}, "
                    f"{name}_f{ip}, scal, c);"))
    lines.append("    }")
    lines += ["    free(c);", "    return 0;", "}", ""]
    return "\n".join(lines)


# --- C++17 templated emission (the architecture of record) ------------------

_EVALUATOR_HPP = """\
/* libxckernel shared evaluator. Machine-generated by xckernel; do not
 * edit. Copyright (c) 2026 Susi Lehtola. */
#pragma once
#include <cstdint>
#include <limits>

/* Build configuration (BLAS on/off, BLAS integer width), written by CMake.
 * Header-only use without it falls back to the portable loops. */
#if __has_include("xckernel/config.h")
#include "xckernel/config.h"
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
#define XCK_HD __host__ __device__
#else
#define XCK_HD
#endif

/* Grid points per stage-B block: the scratch is (1 + nbf) * block. */
#ifndef XCKERNEL_GRID_BLOCK
#define XCKERNEL_GRID_BLOCK 1024
#endif

#ifdef XCKERNEL_USE_BLAS
#ifndef XCKERNEL_BLAS_INT
#define XCKERNEL_BLAS_INT int
#endif
extern "C" {
void dgemm_(const char* ta, const char* tb, const XCKERNEL_BLAS_INT* m,
            const XCKERNEL_BLAS_INT* n, const XCKERNEL_BLAS_INT* k,
            const double* alpha, const double* a, const XCKERNEL_BLAS_INT* lda,
            const double* b, const XCKERNEL_BLAS_INT* ldb, const double* beta,
            double* c, const XCKERNEL_BLAS_INT* ldc);
void sgemm_(const char* ta, const char* tb, const XCKERNEL_BLAS_INT* m,
            const XCKERNEL_BLAS_INT* n, const XCKERNEL_BLAS_INT* k,
            const float* alpha, const float* a, const XCKERNEL_BLAS_INT* lda,
            const float* b, const XCKERNEL_BLAS_INT* ldb, const float* beta,
            float* c, const XCKERNEL_BLAS_INT* ldc);
}
#endif

namespace xckernel {

constexpr int64_t grid_block = XCKERNEL_GRID_BLOCK;

/* Scratch (in elements of T) a kernel entry point needs for npts points
 * and nbf basis functions: pass a buffer this large as `work`. */
XCK_HD inline int64_t work_size(int64_t npts, int64_t nbf) {
    const int64_t blk = npts < grid_block ? npts : grid_block;
    return blk * (1 + nbf) + 1;
}

/* Stage A: walk a monomial table, producing the per-point coefficient on
 * grid points g0 .. g0+bk-1.
 * Templated on the floating-point type T: instantiate with float, double,
 * long double, __float128, or any type with T*T and T+T. Table
 * coefficients are dyadic rationals, exactly representable in binary
 * floating point at any precision. */
/* Operands split by provenance: `fields` (grid weights and density
 * fields, host-computed, type T) and `xc` (functional-derivative arrays,
 * type Txc -- Libxc computes in double even when the host works at higher
 * precision; Txc -> T conversion happens per access, exact when widening).
 * Factor ids < nfld index `fields`; the rest index `xc`. */
template <typename T, typename Txc = T>
XCK_HD inline void stage_a(int64_t g0, int64_t bk, int64_t nm,
                           const double* cf, const int32_t* off,
                           const uint16_t* fid, int64_t nfld,
                           const T* const* fields, const Txc* const* xc,
                           T* c) {
    for (int64_t g = 0; g < bk; ++g) {
        const int64_t gg = g0 + g;
        T acc = T(0);
        for (int64_t m = 0; m < nm; ++m) {
            T t = T(cf[m]);
            for (int32_t f = off[m]; f < off[m + 1]; ++f) {
                const uint16_t id = fid[f];
                t *= (id < nfld) ? fields[id][gg] : T(xc[id - nfld][gg]);
            }
            acc += t;
        }
        c[g] = acc;
    }
}

/* Merge one pattern into the GEMM operand: W(i,g) = c(g) X(i,g) for the
 * first pattern of a group, += for the rest. X has row stride ldx, W bk. */
template <typename T>
XCK_HD inline void accumulate(int64_t bk, int64_t n, const T* c, const T* X,
                              int64_t ldx, T* W, int first) {
    for (int64_t i = 0; i < n; ++i) {
        const T* Xi = X + i * ldx;
        T* Wi = W + i * bk;
        if (first)
            for (int64_t g = 0; g < bk; ++g) Wi[g] = c[g] * Xi[g];
        else
            for (int64_t g = 0; g < bk; ++g) Wi[g] += c[g] * Xi[g];
    }
}

/* Stage B: out(i,j) += sum_g A(i,g) B(j,g), A and B row-major with row
 * strides lda and ldb. The generic loops work at any precision; float and
 * double go to BLAS when the library is built with it. */
template <typename T>
XCK_HD inline void gemm_nt(int64_t n, int64_t k, const T* A, int64_t lda,
                           const T* B, int64_t ldb, T* out) {
    for (int64_t i = 0; i < n; ++i) {
        const T* Ai = A + i * lda;
        for (int64_t j = 0; j < n; ++j) {
            const T* Bj = B + j * ldb;
            T s = T(0);
            for (int64_t g = 0; g < k; ++g) s += Ai[g] * Bj[g];
            out[i * n + j] += s;
        }
    }
}

/* A fixed linear combination of operands, formed once and then used like
 * any other: dst(i,g) = sum_t wt[t] src[t](i,g), sources with row stride
 * lds, dst with row stride bk. Forms the Laplacians the engine works with
 * from the derivative-tower components the host passes. */
template <typename T>
XCK_HD inline void combine(int64_t bk, int64_t n, int nt, const T* const* src,
                           const double* wt, int64_t lds, T* dst) {
    for (int64_t i = 0; i < n; ++i) {
        T* d = dst + i * bk;
        const T* s0 = src[0] + i * lds;
        for (int64_t g = 0; g < bk; ++g) d[g] = T(wt[0]) * s0[g];
        for (int t = 1; t < nt; ++t) {
            const T* st = src[t] + i * lds;
            for (int64_t g = 0; g < bk; ++g) d[g] += T(wt[t]) * st[g];
        }
    }
}

/* Zero the rows of functions off the atom: X(i,g) = 0 where !mask[i]. */
template <typename T>
XCK_HD inline void mask_rows(int64_t bk, int64_t n, const int8_t* mask,
                             T* X) {
    for (int64_t i = 0; i < n; ++i)
        if (!mask[i])
            for (int64_t g = 0; g < bk; ++g) X[i * bk + g] = T(0);
}

/* Scale each row by its basis function's center coordinate:
 * X(i,g) *= R[i] (the London-orbital operands R_a chi, R_a d_c chi). */
template <typename T>
XCK_HD inline void scale_rows(int64_t bk, int64_t n, const double* R, T* X) {
    for (int64_t i = 0; i < n; ++i) {
        const T r = T(R[i]);
        for (int64_t g = 0; g < bk; ++g) X[i * bk + g] *= r;
    }
}

/* A per-point reduction over the functions on an atom:
 * out(g) = sum_{i: mask[i]} sum_t c[t] A[t](i,g) B[t](i,g), rows of
 * stride npts -- the perturbed fields of a nuclear displacement. */
template <typename T>
XCK_HD inline void masked_colsum(int64_t npts, int64_t n, const int8_t* mask,
                                 int nt, const T* const* A,
                                 const T* const* B, const double* c, T* out) {
    for (int64_t g = 0; g < npts; ++g) out[g] = T(0);
    for (int64_t i = 0; i < n; ++i) {
        if (!mask[i]) continue;
        for (int t = 0; t < nt; ++t) {
            const T* a = A[t] + i * npts;
            const T* b = B[t] + i * npts;
            const T ct = T(c[t]);
            for (int64_t g = 0; g < npts; ++g) out[g] += ct * a[g] * b[g];
        }
    }
}

/* Row-wise contraction: out(i) += sum_g A(i,g) B(i,g) -- the diagonal of
 * gemm_nt, for kernels with one free basis index (Fock diagonals and
 * nuclear-gradient rows). */
template <typename T>
XCK_HD inline void rowdot(int64_t n, int64_t k, const T* A, int64_t lda,
                          const T* B, int64_t ldb, T* out) {
    for (int64_t i = 0; i < n; ++i) {
        const T* Ai = A + i * lda;
        const T* Bi = B + i * ldb;
        T s = T(0);
        for (int64_t g = 0; g < k; ++g) s += Ai[g] * Bi[g];
        out[i] += s;
    }
}

#ifdef XCKERNEL_USE_BLAS
namespace detail {
inline bool blas_fits(int64_t n, int64_t k, int64_t lda, int64_t ldb) {
    constexpr int64_t imax = std::numeric_limits<XCKERNEL_BLAS_INT>::max();
    return n <= imax && k <= imax && lda <= imax && ldb <= imax;
}
} // namespace detail

/* Row-major out = A B^T is column-major out^T = B^T A. */
inline void gemm_nt(int64_t n, int64_t k, const double* A, int64_t lda,
                    const double* B, int64_t ldb, double* out) {
    if (n == 0 || k == 0) return;
    if (!detail::blas_fits(n, k, lda, ldb))
        return gemm_nt<double>(n, k, A, lda, B, ldb, out);
    const XCKERNEL_BLAS_INT bn = n, bk = k, ba = lda, bb = ldb;
    const double one = 1.0;
    dgemm_("T", "N", &bn, &bn, &bk, &one, B, &bb, A, &ba, &one, out, &bn);
}

inline void gemm_nt(int64_t n, int64_t k, const float* A, int64_t lda,
                    const float* B, int64_t ldb, float* out) {
    if (n == 0 || k == 0) return;
    if (!detail::blas_fits(n, k, lda, ldb))
        return gemm_nt<float>(n, k, A, lda, B, ldb, out);
    const XCKERNEL_BLAS_INT bn = n, bk = k, ba = lda, bb = ldb;
    const float one = 1.0f;
    sgemm_("T", "N", &bn, &bn, &bk, &one, B, &bb, A, &ba, &one, out, &bn);
}
#endif

} // namespace xckernel
"""


#: build configuration consumed by evaluator.hpp; filled in by CMake.
_CONFIG_H_IN = """\
/* libxckernel build configuration. Generated by CMake; do not edit. */
#pragma once
#cmakedefine XCKERNEL_USE_BLAS 1
#define XCKERNEL_BLAS_INT @XCKERNEL_BLAS_INT_TYPE@
#define XCKERNEL_GRID_BLOCK @XCKERNEL_GRID_BLOCK@
"""


# --- the derivative-tower kernels ----------------------------------------------
#
# Every compiled kernel takes its collocation as derivative towers
# (tower.py): chi[k, u, g] and, for the gradient rows, Dchi[k, u, g] = (D
# chi)[k, u, g]; per-point fields are tower components by name. The
# engine's operands map onto them one to one or as fixed combinations
# (the Laplacians), which the kernel forms itself.

#: per ABI kind: (has nbf, tower arrays, output shape)
ABI_KINDS = {
    "matrix": (True, ("chi",), "(nbf, nbf)"),
    "diag": (True, ("chi",), "(nbf,)"),
    "g1": (True, ("chi", "Dchi"), "(3, nbf)"),
    # gradient rows of a general (non-symmetric) density matrix M:
    # Dchi = M chi and DTchi = M^T chi
    "g1c": (True, ("chi", "Dchi", "DTchi"), "(3, nbf)"),
    "gg": (False, (), "(3, ng)"),
    # nuclear derivatives of the Fock matrix, one atom per call: the basis
    # class (atom mask; D chi towers to form the perturbed fields) and the
    # grid class (weights restricted to the atom's points by the host)
    "f1": (True, ("chi", "Dchi"), "(3, nbf, nbf)"),
    "f1u": (True, ("chi", "Dchi_a", "Dchi_b"), "(3, nbf, nbf)"),
    "fg": (True, ("chi",), "(3, nbf, nbf)"),
    # ... for a general density matrix M (complex orbitals): the M^T towers
    "f1c": (True, ("chi", "Dchi", "DTchi"), "(3, nbf, nbf)"),
    "f1cu": (True, ("chi", "Dchi_a", "Dchi_b", "DTchi_a", "DTchi_b"),
             "(3, nbf, nbf)"),
    # explicit London-orbital (GIAO) field derivative of the Fock matrix:
    # K^s, s = x, y, z, with dF/dB_s = (i/2c) K^s at a real reference; the
    # kernel forms R_a chi etc. from chi and the basis-function centers
    "giao": (True, ("chi",), "(3, nbf, nbf)"),
}

#: kinds taking the basis-function centers (const double* bf_centers,
#: (3, nbf), after the towers)
CENTER_KINDS = ("giao",)

#: kinds taking the atom mask (const int8_t* atom_mask, after the towers)
MASKED_KINDS = ("f1", "f1u", "f1c", "f1cu")
#: kinds whose output is one nbf x nbf matrix per row block
MATRIX_KINDS = ("matrix", "f1", "f1u", "f1c", "f1cu", "fg", "giao")


def collapse_pointwise(expr, functional) -> CollapsedKernel:
    """Collapse a per-point integrand (no basis factors) into the shared
    table form: a single pattern carried on a dummy basis pair, whose
    collocation factors the pointwise emitter never reads."""
    import sympy as sp

    from ..engine.kernel import KernelIntegrand
    from .codegen import collapse
    pair = sp.Symbol("chi_u", real=True) * sp.Symbol("chi_v", real=True)
    return collapse(KernelIntegrand(functional=functional,
                                    index_pairs=[("u", "v")],
                                    expr=sp.expand(expr * pair)))


def kernel_layout(blocks: List[CollapsedKernel], scal_ck: CollapsedKernel,
                  kind: str = "matrix", computed=None):
    """The tower interface of a kernel (tower.Layout). ``computed``: the
    per-point fields the kernel evaluates itself, name -> per row block
    (channel, [(c, chi axes, Dchi axes)]) (the dF/dX perturbed fields)."""
    from .tower import Layout
    internal = scal_order(scal_ck)
    nf = len(internal) - len(scal_ck.libxc_args)
    codes = set()
    if kind != "gg":
        for ck in blocks:
            for u, v, _ in ck.patterns:
                codes |= {u, v}
    L = Layout(internal[:nf], internal[nf:], codes,
               computed=tuple(computed or ()))
    for n in L.computed:
        for ch, terms in computed[n]:
            for _, a, b in terms:
                arr, b = _row_tower(ch, b)
                L.require("chi", len(a))
                L.require(arr, len(b))
    return L


def _row_tower(ch: str, b: str) -> Tuple[str, str]:
    """(tower array, axes) of a perturbed-field term's contracted factor:
    a 'T' prefix selects the M^T tower of a general density matrix."""
    base = "DTchi" if b.startswith("T") else "Dchi"
    return (f"{base}_{ch}" if ch else base), b.lstrip("T")


def _tables(ck: CollapsedKernel, sidx: dict, tag: str) -> Tuple[list, dict]:
    """constexpr monomial tables of every pattern, named c<tag>_<p>..."""
    lines, nms = [], {}
    for ip, (_, _, monos) in enumerate(ck.patterns):
        coeffs, offs, fids = [], [0], []
        for coeff, factors in monos:
            coeffs.append(coeff)
            for fname, e in factors:
                fids.extend([sidx[fname]] * e)
            offs.append(len(fids))
        lines += [f"static constexpr double c{tag}_{ip}[] = {{",
                  "    " + ",".join(f"{c!r}" for c in coeffs), "};",
                  f"static constexpr int32_t o{tag}_{ip}[] = {{",
                  "    " + ",".join(str(o) for o in offs), "};",
                  f"static constexpr uint16_t f{tag}_{ip}[] = {{",
                  "    " + (",".join(str(f) for f in fids) or "0"), "};"]
        nms[ip] = len(monos)
    return lines, nms


def _combine_call(srcs: List[str], wts: List[int], lds: str, bk: str, n: str,
                  dst: str, indent: str) -> List[str]:
    return [indent + "{",
            indent + "    const T* s_[] = {" + ", ".join(srcs) + "};",
            indent + "    static constexpr double w_[] = {"
            + ", ".join(f"{w}.0" for w in wts) + "};",
            indent + f"    combine<T>({bk}, {n}, {len(srcs)}, s_, w_, {lds}, "
            f"{dst});",
            indent + "}"]


def emit_tower_hpp(blocks: List[CollapsedKernel], scal_ck: CollapsedKernel,
                   name: str, kind: str, computed=None) -> str:
    """Header-only templated kernel on the tower interface.

    kind 'matrix': out (nbf, nbf), one GEMM per pattern group; 'diag'
    (one block) and 'g1' (one block per direction): out[r*nbf + u], one
    row-wise dot product per group; 'gg': out[r*npts + g], pointwise."""
    from .tower import center_axis, comp_index, is_masked
    L = kernel_layout(blocks, scal_ck, kind, computed)
    sidx = _scal_index(scal_ck)
    ns = f"detail_{name}"
    has_nbf, arrays, _ = ABI_KINDS[kind]
    # the table's field operands: passed or formed, and computed ones
    nfld = len(L.internal_fields) + len(L.computed)
    xder = L.derived_basis()
    sder = L.derived_fields()
    cfld = L.computed

    lines = ["/* generated by xckernel; do not edit. */", "#pragma once",
             "#include <cstdint>", "#include <new>",
             '#include "xckernel/evaluator.hpp"', "",
             "namespace xckernel {", f"namespace {ns} {{"]
    nms = []
    for r, ck in enumerate(blocks):
        t, n = _tables(ck, sidx, str(r))
        lines += t
        nms.append(n)
    lines += [f"static constexpr int64_t NFLD = {nfld};",
              f"}} // namespace {ns}", "",
              "/* Scratch (elements of T) the kernel needs: pass at least this",
              " * much as `work`, or nullptr to allocate it internally. */",
              f"inline int64_t {name}_work(int64_t npts, int64_t nbf) {{",
              "    const int64_t blk = npts < grid_block ? npts : grid_block;",
              f"    return blk * (1 + nbf * {1 + len(xder) if has_nbf else 0})"
              f" + {len(sder) + len(cfld)} * npts + 1;",
              "}", ""]

    sig = ["int64_t npts"] + (["int64_t nbf"] if has_nbf else []) \
        + [f"const T* {a}" for a in arrays] \
        + (["const int8_t* atom_mask"] if kind in MASKED_KINDS else []) \
        + (["const double* bf_centers"] if kind in CENTER_KINDS else []) \
        + ["const T* const* fields", "const Txc* const* xc", "T* out",
           "T* work = nullptr"]
    lines += ["/* fields: the per-point tower operands (type T), in the order",
              f" * of {name}_scal_names; xc: the functional-derivative arrays",
              " * (type Txc; Libxc computes in double whatever T is). */",
              "template <typename T, typename Txc = T>",
              f"int {name}_t(" + ",\n        ".join(sig) + ") {",
              "    const int64_t blk = npts < grid_block ? npts : grid_block;",
              "    T* c = work;",
              "    bool own = false;",
              "    if (!c) {",
              f"        c = new (std::nothrow) T[{name}_work(npts, "
              f"{'nbf' if has_nbf else '0'})];",
              "        own = true;",
              "    }",
              "    if (!c) return 1;"]
    if has_nbf:
        lines += ["    T* W = c + blk;", "    const T* Wc = W;"]
        lines += [f"    T* X{j} = W + (int64_t){j + 1}*nbf*blk;"
                  for j in range(len(xder))]
    base = f"blk * (1 + nbf * {1 + len(xder)})" if has_nbf else "blk"
    for j, n in enumerate(sder):
        lines.append(f"    T* S{j} = c + {base} + (int64_t){j}*npts;")
        lines += _combine_call([f"fields[{i}]" for i, _ in L.field_map[n]],
                               [w for _, w in L.field_map[n]],
                               "0", "npts", "1", f"S{j}", "    ")
    for j, n in enumerate(cfld):
        lines.append(f"    T* C{j} = c + {base} + "
                     f"(int64_t){len(sder) + j}*npts;")
    # the engine's per-point operands in table order: passed, formed from
    # the tower, or computed by the kernel (internal_fields keeps the
    # table order of the passed and formed ones; computed ones were split
    # off and are re-inserted at their table positions)
    order = scal_order(scal_ck)[:nfld]
    fi = []
    for n in order:
        if n in cfld:
            fi.append(f"C{cfld.index(n)}")
        elif n in sder:
            fi.append(f"S{sder.index(n)}")
        else:
            fi.append(f"fields[{L.field_map[n][0][0]}]")
    nfi = len(fi)
    lines.append(f"    const T* fi[{nfi}] = {{" + ", ".join(fi) + "};")

    def block_loop(rows):
        out = ["    for (int64_t g0 = 0; g0 < npts; g0 += blk) {",
               "        const int64_t bk = npts - g0 < blk ? npts - g0 : blk;"]
        for code, (arr, combo) in L.basis.items():
            if code in xder:
                j = xder.index(code)
                out.extend(_combine_call(
                    [f"{arr} + (int64_t){comp_index(ax)}*nbf*npts + g0"
                     for ax, _ in combo], [w for _, w in combo],
                    "npts", "bk", "nbf", f"X{j}", "        "))
                if is_masked(code):
                    out.append(f"        mask_rows<T>(bk, nbf, atom_mask, X{j});")
                a = center_axis(code)
                if a is not None:
                    out.append(f"        scale_rows<T>(bk, nbf, bf_centers + "
                               f"(int64_t){a}*nbf, X{j});")
        for r in rows:
            out.extend(row_body(r))
        out.append("    }")
        return out

    bexpr = {}
    for code, (arr, combo) in L.basis.items():
        if code in xder:
            bexpr[code] = (f"X{xder.index(code)}", "bk")
        else:
            k = comp_index(combo[0][0])
            bexpr[code] = (f"{arr} + (int64_t){k}*nbf*npts + g0", "npts")

    def sa(r, ip):
        return (f"stage_a<T, Txc>(g0, bk, {nms[r][ip]}, {ns}::c{r}_{ip}, "
                f"{ns}::o{r}_{ip}, {ns}::f{r}_{ip}, {ns}::NFLD, fi, xc, c);")

    def row_body(r):
        ck = blocks[r]
        if kind == "gg":
            return ["        " + sa(r, 0),
                    f"        for (int64_t g = 0; g < bk; ++g) "
                    f"out[(int64_t){r}*npts + g0 + g] += c[g];"]
        if kind in MATRIX_KINDS:
            return _stage_b_calls(ck, bexpr, lambda ip: sa(r, ip),
                                  acc="accumulate<T>", cast=lambda w: "Wc",
                                  out=f"out + (int64_t){r}*nbf*nbf")
        return _stage_b_calls(ck, bexpr, lambda ip: sa(r, ip),
                              gemm="rowdot", acc="accumulate<T>",
                              cast=lambda w: "Wc",
                              out=f"out + (int64_t){r}*nbf")

    if cfld:
        # one pass per row block: that block's computed fields first
        for r in range(len(blocks)):
            for j, n in enumerate(cfld):
                ch, terms = computed[n][r]
                lines += [
                    "    {",
                    "        const T* A_[] = {" + ", ".join(
                        f"chi + (int64_t){comp_index(a)}*nbf*npts"
                        for _, a, _ in terms) + "};",
                    "        const T* B_[] = {" + ", ".join(
                        "{} + (int64_t){}*nbf*npts".format(
                            _row_tower(ch, b)[0],
                            comp_index(_row_tower(ch, b)[1]))
                        for _, _, b in terms) + "};",
                    "        static constexpr double c_[] = {" + ", ".join(
                        f"{float(c)!r}" for c, _, _ in terms) + "};",
                    f"        masked_colsum<T>(npts, nbf, atom_mask, "
                    f"{len(terms)}, A_, B_, c_, C{j});",
                    "    }"]
            lines += block_loop([r])
    else:
        lines += block_loop(range(len(blocks)))
    lines += ["    if (own) delete[] c;", "    return 0;", "}", "",
              "} // namespace xckernel", ""]
    return "\n".join(lines)


def emit_kernel_hpp(ck: CollapsedKernel, name: str) -> str:
    """A response (matrix) kernel on the tower interface."""
    return emit_tower_hpp([ck], ck, name, "matrix")


def _c_signature(name: str, kind: str, indent: str = "    ") -> str:
    has_nbf, arrays, _ = ABI_KINDS[kind]
    args = ["int64_t npts"] + (["int64_t nbf"] if has_nbf else []) \
        + [f"const double* {a}" for a in arrays] \
        + (["const int8_t* atom_mask"] if kind in MASKED_KINDS else []) \
        + (["const double* bf_centers"] if kind in CENTER_KINDS else []) \
        + ["const double* const* scal", "double* out"]
    return f"int {name}(" + (",\n" + indent).join(args) + ")"


def emit_tower_cpp(blocks: List[CollapsedKernel], scal_ck: CollapsedKernel,
                   name: str, kind: str, computed=None) -> str:
    """The double instantiation, the C ABI wrapper and the operand tables."""
    L = kernel_layout(blocks, scal_ck, kind, computed)
    names = L.scal_names
    has_nbf, arrays, _ = ABI_KINDS[kind]
    nf = len(L.fields)
    call = ["npts"] + (["nbf"] if has_nbf else []) + list(arrays) \
        + (["atom_mask"] if kind in MASKED_KINDS else []) \
        + (["bf_centers"] if kind in CENTER_KINDS else [])
    lines = ["/* generated by xckernel; do not edit. */",
             f'#include "xckernel/kernels/{name}.hpp"', "",
             'extern "C" {', "",
             f"const char* {name}_scal_names[{len(names)}] = {{"]
    lines += [f'    "{n}",' for n in names]
    lines += ["};",
              f"extern const int {name}_n_scal;",
              f"const int {name}_n_scal = {len(names)};",
              f"extern const int {name}_n_fields;",
              f"const int {name}_n_fields = {nf};"]
    for a in arrays:
        lines += [f"extern const int {name}_{a}_order;",
                  f"const int {name}_{a}_order = {L.orders.get(a, 0)};"]
    lines += ["",
              "/* One homogeneous scal list (all double): the per-point tower",
              f" * operands first, the functional-derivative arrays from",
              f" * {name}_n_fields on. */",
              _c_signature(name, kind) + " {",
              f"    return xckernel::{name}_t<double, double>(",
              "        " + ", ".join(call) + ",",
              f"        scal, scal + {nf}, out);",
              "}", "", "} // extern C", ""]
    return "\n".join(lines)


def emit_kernel_cpp(ck: CollapsedKernel, name: str) -> str:
    return emit_tower_cpp([ck], ck, name, "matrix")


# --- package emission (libxckernel) ------------------------------------------

def _unpack(entry) -> Tuple[str, int, str]:
    """(name, order[, kind]) -> (name, order, kind); kind defaults to the
    matrix ABI of the response kernels."""
    if len(entry) == 3:
        return entry
    name, order = entry
    return name, order, "matrix"


#: header comment per non-matrix ABI kind
_KIND_NOTE = {
    "diag": "out (nbf,): the diagonal F_uu of the order-1 kernel",
    "g1": ("out (3, nbf): nuclear-gradient basis-class rows; summed over the "
           "functions on atom A, +dE/dX_{A,d}. Dchi = D chi (tower)"),
    "g1c": ("out (3, nbf): nuclear-gradient basis-class rows for a general "
            "density matrix M; summed over the functions on atom A, "
            "+dE/dX_{A,d}. Dchi = M chi, DTchi = M^T chi (towers)"),
    "gg": ("out (3, ng): nuclear-gradient grid class w * d_d e(r_g); summed "
           "over the points of atom A, its grid-motion term"),
    "f1": ("out (3, nbf, nbf): dF/dX_{A,d}, basis class, for the atom whose "
           "functions atom_mask flags (one call per atom). Dchi = D chi"),
    "f1u": ("out (3, nbf, nbf): dF^s/dX_{A,d}, basis class, for the atom "
            "whose functions atom_mask flags. Dchi_a/b = D^a/b chi"),
    "fg": ("out (3, nbf, nbf): dF/dX_{A,d}, grid class: call with the "
           "weights of atom A's points (w M^A), one call per atom"),
    "giao": ("out (3, nbf, nbf): K^s with dF/dB_s = (i/2c) K^s, the explicit "
             "London-orbital field derivative at a real reference; "
             "bf_centers (3, nbf) holds each basis function's center"),
    "f1c": ("out (3, nbf, nbf): dF/dX_{A,d}, basis class, general density "
            "matrix M: Dchi = M chi, DTchi = M^T chi; one call per atom"),
    "f1cu": ("out (3, nbf, nbf): dF^s/dX_{A,d}, basis class, general "
             "density matrices: Dchi_s = M^s chi, DTchi_s = M^s^T chi"),
}


def emit_header(kernel_names: List[str], version: str) -> str:
    """The public C header for libxckernel."""
    lines = [
        "/* libxckernel: generated exchange-correlation kernel contractions",
        " * (an automatic-differentiation backend for Libxc).",
        f" * Version {version}. Machine-generated by xckernel; do not edit.",
        " * Copyright (c) 2026 Susi Lehtola.",
        " *",
        " * ABI: one perturbation-batch entry per call; `out` is accumulated",
        " * (+=); returns 0 on success. Every spatially varying operand is a",
        " * component of a Cartesian derivative tower, named",
        " *     <base>[_<spin>][_<pert>][_<axes>]   e.g. rho_a_p1_xy",
        " * with the derivative axes sorted ('' = the value). Collocation is",
        " * one array per tower, chi[k][u][g] (and Dchi = D chi for the",
        " * gradient rows), components k in the order 1, x, y, z, xx, xy, xz,",
        " * yy, yz, zz, xxx, xxy, ... (PySCF's eval_ao(deriv=n)); the kernel",
        " * reads components up to <name>_chi_order. `scal` is the ordered",
        " * array of per-point operands (<name>_scal_names, manifest.json):",
        " * tower components (rho_x, rho_xy, tau_p1, ...) first, then the",
        " * Libxc derivative arrays. Laplacians are formed inside the kernel",
        " * from the tower and are never operands.",
        " * Kernels contain XC terms only: Coulomb, HF and range-separated",
        " * exchange are host-owned.",
        " *",
        " * Mixed functionals: libxckernel never evaluates functionals.",
        " * Every kernel is JOINTLY LINEAR in the functional-derivative",
        " * operands (each generated term carries exactly one derivative",
        " * factor; enforced at generation time), so hosts pass their own",
        " * coefficient-mixed (superfunctional) derivative arrays and",
        " * F[sum_i c_i f_i] = sum_i c_i F[f_i] holds exactly. Cross-family",
        " * mixtures: evaluate at the union family; absent derivatives are",
        " * zeros.",
        " */",
        "#ifndef XCKERNEL_H",
        "#define XCKERNEL_H",
        "#include <stdint.h>",
        "#ifdef __cplusplus",
        'extern "C" {',
        "#endif",
        "",
        f'#define XCKERNEL_VERSION "{version}"',
        "",
    ]
    for entry in kernel_names:
        name, order, kind = _unpack(entry)
        if order == 0:
            lines.append(f"double {name}(int64_t npts, const double* w,")
            lines.append(f"              const double* rho, "
                         f"const double* zk);")
            lines.append("")
            continue
        if kind != "matrix":
            lines.append(f"/* {_KIND_NOTE[kind]} */")
        lines.append(_c_signature(name, kind, indent="           ") + ";")
        lines.append(f"extern const char* {name}_scal_names[];")
        lines.append(f"extern const int {name}_n_scal;")
        lines.append(f"extern const int {name}_n_fields;")
        for a in ABI_KINDS[kind][1]:
            lines.append(f"extern const int {name}_{a}_order;")
        lines.append("")
    lines += ["#ifdef __cplusplus", "}", "#endif", "#endif /* XCKERNEL_H */",
              ""]
    return "\n".join(lines)


def _f90_wrap(line: str, limit: int = 132) -> List[str]:
    """Split an over-long free-form line at commas with & continuations.

    The F2018 free-form limit is 132 columns; only F2023 raised it, and
    gfortran versions in the field still hard-error past 132.
    """
    out = []
    cont_indent = line[:len(line) - len(line.lstrip())] + "    "
    while len(line) > limit:
        cut = line.rfind(", ", 0, limit - 1)
        if cut < 0:
            break
        out.append(line[:cut + 1] + " &")
        line = cont_indent + line[cut + 2:]
    out.append(line)
    return out


def emit_f03(kernel_names: List[str], version: str) -> str:
    """The Fortran 2003 ISO_C_BINDING module (the xc_f03 idiom)."""
    lines = [
        "! libxckernel Fortran interface. Machine-generated by xckernel; do not edit.",
        "! Copyright (c) 2026 Susi Lehtola.",
        f"! Version {version}.",
        "! Scalar operands are passed as an array of C pointers in the order",
        "! given by the kernel's manifest entry (see manifest.json).",
        "module xckernel_f03",
        "  use, intrinsic :: iso_c_binding, only: c_int, c_int8_t, "
        "c_int64_t, c_double, c_ptr",
        "  implicit none",
        "  public",
        "  interface",
    ]
    for entry in kernel_names:
        name, order, kind = _unpack(entry)
        if order != 0:
            has_nbf, arrays, _ = ABI_KINDS[kind]
            masked = kind in MASKED_KINDS
            centered = kind in CENTER_KINDS
            args = ["npts"] + (["nbf"] if has_nbf else []) + list(arrays) \
                + (["atom_mask"] if masked else []) \
                + (["bf_centers"] if centered else []) + ["scal", "out"]
            lines += [
                f"    integer(c_int) function {name}({', '.join(args)}) "
                f"bind(C, name='{name}')",
                "      import :: c_int, c_int8_t, c_int64_t, c_double, c_ptr",
                "      integer(c_int64_t), value :: "
                + ", ".join(["npts"] + (["nbf"] if has_nbf else [])),
            ]
            if arrays:
                lines.append("      real(c_double), intent(in) :: "
                             + ", ".join(f"{a}(*)" for a in arrays))
            if masked:
                lines.append("      integer(c_int8_t), intent(in) :: "
                             "atom_mask(*)")
            if centered:
                lines.append("      real(c_double), intent(in) :: "
                             "bf_centers(*)")
            lines += [
                "      type(c_ptr), intent(in) :: scal(*)",
                "      real(c_double), intent(inout) :: out(*)",
                f"    end function {name}",
            ]
            continue
        lines += [
            f"    real(c_double) function {name}(npts, w, rho, zk) "
            f"bind(C, name='{name}')",
            "      import :: c_int64_t, c_double",
            "      integer(c_int64_t), value :: npts",
            "      real(c_double), intent(in) :: w(*), rho(*), zk(*)",
            f"    end function {name}",
        ]
    lines += ["  end interface", "end module xckernel_f03", ""]
    return "\n".join(w for line in lines for w in _f90_wrap(line))


def emit_cmake(kernel_names: List[Tuple[str, int]], version: str) -> str:
    import re as _re
    by_order: dict = {}
    for entry in kernel_names:
        n, order, _ = _unpack(entry)
        by_order.setdefault(order, []).append(n)
    groups = []
    for order in sorted(by_order):
        srcs = "\n        ".join(f"src/{n}.cpp" for n in by_order[order])
        groups.append(
            f"if(XCKERNEL_MAX_ORDER GREATER_EQUAL {order})\n"
            f"    target_sources(xckernel PRIVATE\n        {srcs}\n    )\n"
            f"endif()")
    grouped = "\n".join(groups)
    return f"""\
# libxckernel build. Machine-generated by xckernel; do not edit.
# Copyright (c) 2026 Susi Lehtola.
cmake_minimum_required(VERSION 3.16)
project(xckernel VERSION {version} LANGUAGES CXX)

option(XCKERNEL_FORTRAN "Build the Fortran interface module" ON)
# The response-kernel sources grow steeply with derivative order; exclude
# orders you do not need at configure time (kernels above the limit are
# declared in the header but not compiled).
set(XCKERNEL_MAX_ORDER 4 CACHE STRING
    "Highest derivative order to compile (0-4)")
# Stage B (the basis-pair contraction) is a GEMM; OFF selects portable
# loops, for builds without a BLAS library.
option(XCKERNEL_BLAS "Contract with BLAS dgemm/sgemm" ON)
option(XCKERNEL_BLAS_ILP64 "The BLAS library uses 64-bit integers" OFF)
set(XCKERNEL_GRID_BLOCK 1024 CACHE STRING
    "Grid points per GEMM block (scratch is nbf x block)")

add_library(xckernel)
{grouped}
target_compile_definitions(xckernel PUBLIC
    XCKERNEL_MAX_ORDER=${{XCKERNEL_MAX_ORDER}})
set(XCKERNEL_BLAS_INT_TYPE int)
if(XCKERNEL_BLAS)
    if(XCKERNEL_BLAS_ILP64)
        set(BLA_SIZEOF_INTEGER 8)
        set(XCKERNEL_BLAS_INT_TYPE int64_t)
    endif()
    find_package(BLAS REQUIRED)
    set(XCKERNEL_USE_BLAS 1)
    target_link_libraries(xckernel PUBLIC ${{BLAS_LIBRARIES}})
    if(BLAS_LINKER_FLAGS)
        target_link_options(xckernel PUBLIC ${{BLAS_LINKER_FLAGS}})
    endif()
endif()
configure_file(include/xckernel/config.h.in
    ${{CMAKE_CURRENT_BINARY_DIR}}/include/xckernel/config.h)
set_target_properties(xckernel PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
    VERSION ${{PROJECT_VERSION}}
    SOVERSION ${{PROJECT_VERSION_MAJOR}})
target_include_directories(xckernel PUBLIC
    $<BUILD_INTERFACE:${{CMAKE_CURRENT_SOURCE_DIR}}/include>
    $<BUILD_INTERFACE:${{CMAKE_CURRENT_BINARY_DIR}}/include>
    $<INSTALL_INTERFACE:include>)

if(XCKERNEL_FORTRAN)
    enable_language(Fortran)
    add_library(xckernel_f03 fortran/xckernel_f03.f90)
    target_link_libraries(xckernel_f03 PUBLIC xckernel)
endif()

include(GNUInstallDirs)
install(TARGETS xckernel EXPORT xckernelTargets
    LIBRARY DESTINATION ${{CMAKE_INSTALL_LIBDIR}}
    ARCHIVE DESTINATION ${{CMAKE_INSTALL_LIBDIR}})
install(FILES include/xckernel.h DESTINATION ${{CMAKE_INSTALL_INCLUDEDIR}})
install(DIRECTORY include/xckernel DESTINATION ${{CMAKE_INSTALL_INCLUDEDIR}}
    PATTERN "config.h.in" EXCLUDE)
install(FILES ${{CMAKE_CURRENT_BINARY_DIR}}/include/xckernel/config.h
    DESTINATION ${{CMAKE_INSTALL_INCLUDEDIR}}/xckernel)
install(FILES manifest.json
    DESTINATION ${{CMAKE_INSTALL_DATADIR}}/xckernel)
if(XCKERNEL_FORTRAN)
    install(TARGETS xckernel_f03 EXPORT xckernelTargets
        LIBRARY DESTINATION ${{CMAKE_INSTALL_LIBDIR}}
        ARCHIVE DESTINATION ${{CMAKE_INSTALL_LIBDIR}})
endif()
install(EXPORT xckernelTargets NAMESPACE xckernel::
    DESTINATION ${{CMAKE_INSTALL_LIBDIR}}/cmake/xckernel)
"""


# --- order-0 (energy) kernels -------------------------------------------------

def emit_exc_hpp(name: str) -> str:
    return f"""/* machine-generated by xckernel; do not edit. Copyright (c) 2026 Susi Lehtola. */
#pragma once
#include <cstdint>
#include "xckernel/evaluator.hpp"

namespace xckernel {{

/* Exc = sum_g w_g rho_g zk_g (zk = Libxc energy density per particle).
 * T: host precision; Txc: the Libxc array type (double from Libxc). */
template <typename T, typename Txc = T>
XCK_HD inline T {name}_t(int64_t npts, const T* w, const T* rho,
                         const Txc* zk) {{
    T acc = T(0);
    for (int64_t g = 0; g < npts; ++g) acc += w[g] * rho[g] * T(zk[g]);
    return acc;
}}

}} // namespace xckernel
"""


def emit_exc_cpp(name: str) -> str:
    return f"""/* machine-generated by xckernel; do not edit. Copyright (c) 2026 Susi Lehtola. */
#include "xckernel/kernels/{name}.hpp"

extern "C" {{
double {name}(int64_t npts, const double* w, const double* rho,
              const double* zk) {{
    return xckernel::{name}_t<double, double>(npts, w, rho, zk);
}}
}} // extern C
"""
