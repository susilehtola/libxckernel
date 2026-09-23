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

ABI (v1, one perturbation-batch entry per call; hosts loop the batch):

  int <name>(int64_t npts, int64_t nbf,
             const double* chi,        /* nbf x npts, row-major */
             const double* dchi,       /* 3 x nbf x npts */
             const double* lapl_chi,   /* nbf x npts, may be NULL if unused */
             const double* hess_chi,   /* 6 x nbf x npts, packed
                                          xx,xy,xz,yy,yz,zz; NULL if unused */
             const double* const* scal,/* n_scal arrays (npts,) -- see the
                                          generated <name>_scal_names table */
             double* out);             /* nbf x nbf, accumulated (+=) */

The scalar-operand order is emitted both as a C string table and returned by
scal_order() for programmatic assembly (the ctypes validation uses it).
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
        if p in ("chi", "dchi", "lapl_chi", "hess_chi"):
            continue
        if p.startswith("hess_rho"):
            # packed symmetric tensor: six components
            for comp in _H6_COMPS:
                order.append(f"{p}_{comp}")
        elif p.startswith(("grad_rho", "jp")) or p.startswith("dgrad_rho"):
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
                   gemm="gemm_nt", acc="accumulate", cast=lambda x: x):
    """The per-block body: stage A per pattern, merge, one GEMM per group."""
    side, groups = _gemm_plan(ck.patterns)
    lines: List[str] = []
    for shared, members in groups:
        for k, (ip, other) in enumerate(members):
            lines.append(indent + stage_a_call(ip))
            lines.append(f"{indent}{acc}(bk, nbf, c, {bexpr[other]} + g0, "
                         f"npts, W, {int(k == 0)});")
        sh = f"{bexpr[shared]} + g0"
        if side == "u":
            lines.append(f"{indent}{gemm}(nbf, bk, {sh}, npts, {cast('W')}, "
                         f"bk, out);")
        else:
            lines.append(f"{indent}{gemm}(nbf, bk, {cast('W')}, bk, {sh}, "
                         f"npts, out);")
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
        ck, {k: v[0] for k, v in _BASIS.items()},
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


def emit_kernel_hpp(ck: CollapsedKernel, name: str) -> str:
    """Header-only templated kernel (tables + template entry point)."""
    sidx = _scal_index(ck)
    ns = f"detail_{name}"
    lines = [
        "/* generated by xckernel; do not edit. */",
        "#pragma once",
        "#include <cstdint>",
        '#include "xckernel/evaluator.hpp"',
        "",
        "namespace xckernel {",
        f"namespace {ns} {{",
    ]
    pat_meta = []
    for ip, (ufac, vfac, monos) in enumerate(ck.patterns):
        coeffs, offs, fids = [], [0], []
        for coeff, factors in monos:
            coeffs.append(coeff)
            for fname, e in factors:
                fids.extend([sidx[fname]] * e)
            offs.append(len(fids))
        lines.append(f"static constexpr double c{ip}[] = {{")
        lines.append("    " + ",".join(f"{c!r}" for c in coeffs))
        lines.append("};")
        lines.append(f"static constexpr int32_t o{ip}[] = {{")
        lines.append("    " + ",".join(str(o) for o in offs))
        lines.append("};")
        lines.append(f"static constexpr uint16_t f{ip}[] = {{")
        lines.append("    " + (",".join(str(f) for f in fids) or "0"))
        lines.append("};")
        pat_meta.append((ip, ufac, vfac, len(monos)))
    lines.append(f"static constexpr int64_t NFLD = "
                 f"{len(scal_order(ck)) - len(ck.libxc_args)};")
    lines += [f"}} // namespace {ns}", ""]

    nfld = len(scal_order(ck)) - len(ck.libxc_args)
    lines += [
        "/* fields: host-computed per-point operands (type T); xc: the",
        " * functional-derivative arrays (type Txc; Libxc computes in double",
        " * regardless of T). work: caller scratch of",
        " * xckernel::work_size(npts, nbf) elements or nullptr",
        " * (heap-allocated internally; pass a buffer in device code). */",
        "template <typename T, typename Txc = T>",
        f"int {name}_t(int64_t npts, int64_t nbf,",
        "             const T* chi, const T* dchi, const T* lapl_chi,",
        "             const T* hess_chi,",
        "             const T* const* fields, const Txc* const* xc,",
        "             T* out, T* work = nullptr) {",
        "    const int64_t blk = npts < grid_block ? npts : grid_block;",
        "    T* c = work;",
        "    bool own = false;",
        "    if (!c) {",
        "        c = new (std::nothrow) T[work_size(npts, nbf)];",
        "        own = true;",
        "    }",
        "    if (!c) return 1;",
        "    T* W = c + blk;",
        "    const T* Wc = W;",
        "    for (int64_t g0 = 0; g0 < npts; g0 += blk) {",
        "        const int64_t bk = npts - g0 < blk ? npts - g0 : blk;",
    ]
    _bexpr = {"chi": "chi", "lapl_chi": "lapl_chi",
              "dchi[0]": "dchi + (int64_t)0*nbf*npts",
              "dchi[1]": "dchi + (int64_t)1*nbf*npts",
              "dchi[2]": "dchi + (int64_t)2*nbf*npts",
              **{f"hess_chi[{k}]": f"hess_chi + (int64_t){k}*nbf*npts"
                 for k in range(6)}}
    nms = {ip: nm for ip, _, _, nm in pat_meta}
    lines += _stage_b_calls(
        ck, _bexpr,
        lambda ip: (f"stage_a<T, Txc>(g0, bk, {nms[ip]}, {ns}::c{ip}, "
                    f"{ns}::o{ip}, {ns}::f{ip}, {ns}::NFLD, fields, xc, c);"),
        acc="accumulate<T>", cast=lambda w: "Wc")
    lines.append("    }")
    lines += ["    if (own) delete[] c;", "    return 0;", "}",
              "", "} // namespace xckernel", ""]
    # <new> for std::nothrow
    lines.insert(3, "#include <new>")
    return "\n".join(lines)


def emit_kernel_cpp(ck: CollapsedKernel, name: str) -> str:
    """The double instantiation + C ABI wrapper + operand-name tables."""
    names = scal_order(ck)
    lines = [
        "/* generated by xckernel; do not edit. */",
        f'#include "xckernel/kernels/{name}.hpp"',
        "",
        'extern "C" {',
        "",
        f"const char* {name}_scal_names[{len(names)}] = {{",
    ]
    for n in names:
        lines.append(f'    "{n}",')
    lines += [
        "};",
        f"extern const int {name}_n_scal;",
        f"const int {name}_n_scal = {len(names)};",
        f"extern const int {name}_n_fields;",
        f"const int {name}_n_fields = "
        f"{len(names) - len(ck.libxc_args)};",
        "",
        "/* The C ABI keeps one homogeneous scal list (all double);",
        " * fields come first, functional-derivative arrays last, split at",
        f" * {name}_n_fields. */",
        f"int {name}(int64_t npts, int64_t nbf,",
        "           const double* chi, const double* dchi,",
        "           const double* lapl_chi, const double* hess_chi,",
        "           const double* const* scal, double* out) {",
        f"    return xckernel::{name}_t<double, double>(",
        "        npts, nbf, chi, dchi, lapl_chi, hess_chi,",
        f"        scal, scal + {len(names) - len(ck.libxc_args)}, out);",
        "}",
        "",
        "} // extern C",
        "",
    ]
    return "\n".join(lines)


# --- package emission (libxckernel) ------------------------------------------

_KERNEL_PROTO = ("int {name}(int64_t npts, int64_t nbf,\n"
                 "           const double* chi, const double* dchi,\n"
                 "           const double* lapl_chi, const double* hess_chi,\n"
                 "           const double* const* scal, double* out);")


def emit_header(kernel_names: List[str], version: str) -> str:
    """The public C header for libxckernel."""
    lines = [
        "/* libxckernel: generated exchange-correlation kernel contractions",
        " * (an automatic-differentiation backend for Libxc).",
        f" * Version {version}. Machine-generated by xckernel; do not edit.",
        " * Copyright (c) 2026 Susi Lehtola.",
        " *",
        " * ABI: one perturbation-batch entry per call; `scal` is an ordered",
        " * array of per-point scalar operands (see <name>_scal_names /",
        " * manifest.json); `out` is accumulated (+=). Returns 0 on success.",
        " * lapl_chi is the basis Laplacian collocation and hess_chi the",
        " * packed second-derivative collocation (6 blocks: xx,xy,xz,yy,",
        " * yz,zz); pass NULL when the kernel's family does not use them.",
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
    for name, order in kernel_names:
        if order == 0:
            lines.append(f"double {name}(int64_t npts, const double* w,")
            lines.append(f"              const double* rho, "
                         f"const double* zk);")
            lines.append("")
            continue
        lines.append(_KERNEL_PROTO.format(name=name))
        lines.append(f"extern const char* {name}_scal_names[];")
        lines.append(f"extern const int {name}_n_scal;")
        lines.append(f"extern const int {name}_n_fields;")
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
        "  use, intrinsic :: iso_c_binding, only: c_int, c_int64_t, "
        "c_double, c_ptr",
        "  implicit none",
        "  public",
        "  interface",
    ]
    for name, order in kernel_names:
        if order == 0:
            lines += [
                f"    real(c_double) function {name}(npts, w, rho, zk) "
                f"bind(C, name='{name}')",
                "      import :: c_int64_t, c_double",
                "      integer(c_int64_t), value :: npts",
                "      real(c_double), intent(in) :: w(*), rho(*), zk(*)",
                f"    end function {name}",
            ]
            continue
        lines += [
            f"    integer(c_int) function {name}(npts, nbf, chi, dchi, "
            f"lapl_chi, hess_chi, scal, out) bind(C, name='{name}')",
            "      import :: c_int, c_int64_t, c_double, c_ptr",
            "      integer(c_int64_t), value :: npts, nbf",
            "      real(c_double), intent(in) :: chi(*), dchi(*)",
            "      type(c_ptr), value :: lapl_chi, hess_chi",
            "      type(c_ptr), intent(in) :: scal(*)",
            "      real(c_double), intent(inout) :: out(*)",
            f"    end function {name}",
        ]
    lines += ["  end interface", "end module xckernel_f03", ""]
    return "\n".join(w for line in lines for w in _f90_wrap(line))


def emit_cmake(kernel_names: List[Tuple[str, int]], version: str) -> str:
    import re as _re
    by_order: dict = {}
    for n, order in kernel_names:
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
