# Interfacing plan: emitting the kernel catalog for five hosts

Goal: emit **all XC matrix-element expressions up to 4th derivative order**
for Psi4, OpenMolcas, PySCF, ERKALE, and HelFEM — as runtime NumPy einsums
where the host is Python, and as compiled code elsewhere, with
[Einsums](https://github.com/Einsums/Einsums) (J. Turney) as the common C++
backend and a C ABI for Fortran hosts.

Status: 2026-07-04, complete — incorporates the OpenMolcas source survey and
the Einsums API survey, on top of the six-code survey
(`dedup-analysis.md`) and the validated engine. The pattern-collapse
lowering pass (phase 1) is implemented and verified.

## 1. The deliverable: a finite kernel catalog

"All possible matrix elements up to 4th order" enumerates cleanly. One
generated kernel per point in the product space

| axis | values |
|---|---|
| quantity | energy (order 0), Fock (1), response Fock contraction (2, 3, 4) |
| family | `lda`, `gga`, `mgga_tau`, `mgga` (with ∇²ρ) |
| spin case | unpolarized; unrestricted (per channel); restricted spin-adapted (parity tuple per perturbation, ±1 each) |
| batching | all response kernels batched over perturbed DMs |

Counts per family: unpolarized 5 + unrestricted 2·4 + spin-adapted
(2 + 3 + 4 parity multisets at orders 2–4) ≈ 22; four families ≈ **90
kernels** — a catalog, not an explosion. (mGGA-∇²ρ at orders 3–4 gated on
Libxc `kxc`/`lxc` availability; the manifest records this per functional via
the Libxc flags.)

Each catalog entry carries a machine-readable **manifest** (JSON + generated
C header): operand list in canonical order (weights, collocation blocks,
ground-state fields, perturbed fields per slot, Libxc arrays *by name*),
spin/parity metadata, and what the kernel does NOT include (term-ownership
declaration: XC only — Coulomb/HF/RSH exchange is host-owned, per the
Dalton `addfock` lesson).

Explicit multi-index kernels (`g_uv,ts`, 3- and 4-index beyond) are also
emittable for small-system/debug use, but the catalog's primary form is the
**contracted** one — perturbed DMs in, AO matrices out, O(N²·n_grid) — the
unanimous production interface.

## 2. The key new codegen pass: pattern collapse

The naive per-term emission (fine for NumPy) is wrong for compiled targets:
the order-4 GGA kernel has 862 terms. But every term factorizes as

    (basis factor at u) x (basis factor at v) x (per-point scalar)

and there are only a handful of basis-pair patterns regardless of order:
`chi*chi`, `chi*dchi_c`, `dchi_c*dchi_c'`, `chi*lapl_chi` (+transposes).
The pass groups terms by pattern and factors out the scalar:

    stage A (pointwise): coefficient vectors c_pat(g) = sum of
            (Libxc arrays x field products) -- generated vectorized arithmetic
    stage B (distribute): one GEMM-shaped contraction per pattern,
            out += chi . diag(c) . chi^T  etc.

This reproduces the three-stage architecture every surveyed code converged
on (Dalton `DIST*`, PySCF `_scale_ao/_dot_ao_ao`, Psi4 `phi^T T`, VeloxChem
`mat_G`), turns 862 terms into ~6 GEMMs + one generated scalar loop, and is
exactly the previously-planned "matrix-form lowering". It benefits the NumPy
backend too (fewer, fatter einsums).

**Status: implemented** (`codegen.generate_collapsed`) and verified
bit-identical against the per-term path over 12 kernel classes. Measured
collapse: GGA order-4 862 terms -> 7 patterns; spin-adapted order-3 batched
1034 terms -> 7 patterns; meta-GGA order-3 328 -> 10.

The Einsums survey independently confirms this is the only sane compiled
lowering: Einsums' `einsum` is strictly **binary** (two input tensors per
call), so per-term emission would mean thousands of dispatches with
temporaries, mostly off the BLAS fast path. The collapsed form maps exactly
onto its primitives: stage A = `direct_product`/`axpy` on rank-1 tensors,
stage B = one gemm-dispatched `einsum(Indices{u,v}, &out, Indices{u,g},
tmp, Indices{v,g}, dchi)` per pattern.

## 3. Emission backends

* **B1 — NumPy einsum** (exists today): runtime generation for PySCF and any
  Python host; gains the pattern-collapse pass for speed.
* **B2 — Einsums C++** (new): ahead-of-time emission of one translation unit
  per family (or per kernel), using Einsums tensors for stage B and plain
  vectorized loops (or Einsums `direct_product`/`axpy`) for stage A.
  Survey facts to build on: MIT license; **pin v1.1.x** (v1.1.5 as of
  2026-07; a 2.0-trunk with possible API breaks is in development);
  C++20; deps BLAS/LAPACK + **HDF5 (required)** + fmt/spdlog; CMake
  `find_package(Einsums CONFIG)`; conda-forge package exists; Psi4 lists
  Einsums as an optional build component (precedent: EinHF).
  Two caveats to resolve early: (a) whether `TensorView` can wrap
  externally-owned buffers zero-copy (undocumented publicly — determines
  whether the C ABI layer copies); (b) the einsum dispatcher cannot
  conjugate, only transpose — the complex (HelFEM) instantiation must use
  `gerc`/`true_dot` or pre-conjugated buffers.
  Every kernel is additionally wrapped in a **stable C ABI**:

      int xck_<family>_<spin>_o<order>[_<parities>](
          int64_t npts, int64_t nbf, int64_t nbatch,
          const double* w,                    /* npts */
          const double* chi,                  /* nbf x npts */
          const double* dchi,                 /* 3 x nbf x npts */
          const double* lapl_chi,             /* nbf x npts or NULL */
          const double* const* gs_fields,     /* per manifest */
          const double* const* pert_fields,   /* per slot x batch */
          const double* const* xc_arrays,     /* Libxc outputs, manifest order */
          double* out);                       /* nbatch x nbf x nbf, += */

  Fortran hosts bind this with `ISO_C_BINDING`; C++ hosts may either use the
  ABI or the typed C++ interface. A `std::complex<double>` instantiation of
  the same kernels (templated scalar) serves HelFEM.
* **B3 (roadmap)** — plain C or Fortran source for hosts refusing a C++
  dependency; the pattern-collapsed form makes this nearly mechanical
  (stage B = DGEMM calls).

## 4. Per-host integration

### PySCF (runtime, B1) — effort: small
Already effectively demonstrated by the validation suites. Deliverable: a
`xckernel.hosts.pyscf` module providing (a) `nr_rks_fxc`/`nr_uks_fxc`/
`nr_rks_fxc_st`-signature functions backed by generated kernels, (b) a
`gen_response`-compatible closure. Data marshalling is trivial (`eval_ao`
arrays transposed once). Value beyond parity: orders 3–4 as first-class
`vresp`-style closures, which PySCF itself only has buried in TDDFT
gradients.

### Psi4 (B2 via plugin) — effort: medium
Hook point: `VBase::compute_Vx` (libfock/v.cc:1675) — perturbed AO DMs in,
AO `Vx` out, exactly the catalog's contract. Integration as a Psi4 plugin
(Einsums already has precedent in the Psi4 orbit) that walks Psi4's
`DFTGrid` blocks: per block, take the collocation matrices from
`PointFunctions`, compute perturbed fields from the `Dx` block, call the
kernel, scatter via `functions_local_to_global` (the existing pattern at
v.cc:1957). Immediate value: **meta-GGA Vx** (Psi4 currently throws) and
kxc/lxc contractions Psi4 lacks entirely. Longer-term: upstream into
libfock as an alternative Vx path.

### ERKALE (B2, direct) — effort: medium, biggest capability jump
Ground state: the `increment_lda/gga/mgga_*` templates (dftgrid.h:623-775)
map 1:1 onto stage B; the generated kernels can replace the hand-written
`eval_Fxc` chain-rule blocks per family. Response: ERKALE's Casida is
LDA-only because the GGA/mGGA kernel terms were never hand-derived — the
generated order-2 kernels remove that limitation outright. Two lowering
choices, both supported: AO-route (`Kxc` from AO response matrices +
transform) or the existing MO-on-grid route (`compute_orbs`) with MO-valued
collocation operands (the free-index pair is arbitrary in xckernel).

### HelFEM (B2, complex instantiation) — effort: medium; enables a new module
The out-of-family host. Three marshalling facts from the survey make this
clean:
1. **Per-element dense blocks + scatter index** map directly onto the ABI's
   (`nbf_block` x `npts`) collocation arguments; the host keeps its
   `bf_ind` scatter.
2. **Complex basis**: kernels instantiated for `std::complex<double>`
   (fields and output realized at the end, as HelFEM already does).
3. **Curvilinear metric**: fold the 1/scale factors into the gradient
   collocation operands *before* the kernel call — then the generated
   kernels are metric-blind. (The "gradient direction" index never appears
   uncontracted in any kernel, so pre-scaled `dchi` is exactly equivalent.)
Value: HelFEM currently has NO response module (finite fields only); the
catalog + a small driver gives fully numerical FEM linear response (atomic
polarizabilities at the basis-set limit) and beyond, up to 4th order.

### VeloxChem (B2, C++ integrator internals) — effort: medium-large
VeloxChem's `integrate_{vxc,fxc,kxc,kxclxc}_fock` ladder already *is*
xckernel's contract (DM lists in, AO Focks out, orders 1–4), so the seam is
not the API but the internals. The right injection point is between their
stages: keep the collocation (`SerialDensityGridGenerator`) and the
distribution (`mat_G` + `serialMultABt`), and replace the middle — the
~68k lines of `DensityGridQuad/Cubic` pointwise perturbation products, the
per-property mode strings (`'qrf'`, `'tpa'`, `'thg'`, …16 of them), and the
closed-shell spin-collapse coefficient tables — with generated stage-A
coefficient kernels selected by manifest instead of hand-enumerated modes.
This also removes the known coupling hazard of density-count tables
duplicated between the Python and C++ layers.

What xckernel adds: triplet nonlinear response (currently absent), uniform
meta-GGA coverage across orders, new response properties without new C++
(the mode-string treadmill ends), and 5th order+ if ever wanted. Complex
(CPP) response: VeloxChem hand-unrolls Re/Im (`prod2_r/prod2_i`); generated
kernels can either follow that two-real-contractions pattern or use a
complex instantiation. Caveats: their integrators are MPI/GPU-tuned, so
this is best done with the VeloxChem developers, and a Python-level
prototype (their solver layer is Python; swap in NumPy-backend kernels
behind `_comp_nlr_fock`) can validate the approach before touching C++.

### OpenMolcas (B2 via C ABI + `ISO_C_BINDING`) — reshaped by the survey
**Headline finding: OpenMolcas requests only `exc`/`vxc` from Libxc —
nowhere in the tree is `fxc` or higher called** (only
`xc_f03_*_exc[_vxc]`, nq_util/libxc_interface.F90). MCLR does no on-grid
XC work at all: it consumes precomputed MO-basis potential/generalized-Fock
matrices (`FI_V`, `FA_V`, `ONTOPO/T`, `Fock_PDFT`) stored on the RUNFILE by
the MC-PDFT energy step — a Lagrangian formulation that deliberately avoids
second functional derivatives. Consequences:

1. **There is no existing fxc consumer to hook into.** A drop-in at the
   current boundary would only reproduce `exc`/`vxc` — no added value.
2. **The real near-term value is the MC-PDFT translation layer.**
   `nq_util/translatedens.F90` + `nq_pdft.F90` hand-write the on-top
   translation (ρ,Π) -> (ρ_a,ρ_b): the ratio R = 4Π/ρ², ζ(R) (plain and
   fully-translated quintic-polynomial variants), translated gradients and
   τ/∇²ρ, AND the map's own second derivatives (`d2RdRho2, d2RdRhodPi,
   d2ZdR2`, consumed by calc_pot1.F90). This is precisely xckernel
   ingredient territory: define Π (with ∇Π) and the translated
   spin-density ingredients once, and the Jacobian/Hessian chain rules —
   currently hand-derived Fortran — are generated. Extending to
   higher-order translation derivatives (needed for any future on-grid
   MC-PDFT response) becomes mechanical instead of prohibitive.
3. **An on-grid response path is a net-new opportunity**, not an
   integration: generated fxc+ contractions (including through the
   translation map) would be what an MC-PDFT linear-response / excited-state
   module needs — currently absent because the hand-derivation cost was
   never paid (the same pattern as Psi4's missing MGGA-Vx et al.).

Marshalling facts (all favorable): pure modern Fortran with a strong
`ISO_C_BINDING` culture (108 files; `LibxcInt = c_int`, `LibxcReal =
c_double` type guards); Libxc integrated via CMake external-project
(pinned 7.0.0) — a generated `libxckernel` follows the same pattern; the
grid layer works on desymmetrized full-AO batches with symmetry adaptation
as a separable final step (`SymAdp_Full`), so kernels stay symmetry-blind
and the host keeps its irrep wrapper; DMs arrive triangular/deduplicated
(`DeDe`) — the shim unpacks. τ already uses the Libxc convention.

The active-space gradient projector role is distributed over
`GetPDFTFock`/`fockgen`/`getqaafock` (no single ORBEX); a shared layer-2
projector would serve any future MCSCF response work here.

## 5. Marshalling summary

| host | language | scalar | collocation layout | spin storage | symmetry | backend |
|---|---|---|---|---|---|---|
| PySCF | Python | real | `ao` (npts,nbf) -> transpose | (2,nao,nao) arrays | none (C1 AO) | B1 |
| Psi4 | C++ | real | block `phi` (npts x nlocal) | alternating a/b vectors | SO->AO inside VBase | B2 |
| ERKALE | C++ | real | shell-block `bf` (nbf x npts) | separate Pa, Pb | none | B2 |
| HelFEM | C++ | complex | element block + `bf_ind` | separate Pa, Pb | none (m-blocks) | B2 (complex) |
| OpenMolcas | Fortran | real | batch `Grid_AO`/`TabAO` (nq_util) | nD=1/2 stacked | `SymAdp_Full` host-side | B2 + C ABI |
| VeloxChem | Python + C++ | real (Re/Im split) | `mat_chi` blocks (dft_func) | alpha + collapse tables | none (C1) | B1 proto, then B2 |
| ChronusQ | C++ | real + dcomplex | `double*` point buffers | Pauli spinor (S,Mz,Mx,My) | none | B2 + C++ shim |
| MRChem | C++ | real | Eigen per-node matrices | R + spin variants | none | B2 (Eigen flavor) |
| NWChem | Fortran | real | local batch (nq x nbf) | ipol packing | none (GA host-side) | B2/B3 + C ABI |
| LSDalton | Fortran | real | `GAO(NBLEN,NACTBAST,NTYPSO)` | elms/elmsb | none | B2/B3 + XCFun repack |
| eT | Fortran | real | batched screened blocks | closed-shell only | none | B3 + C ABI |

## 6. Phased work plan

1. **Pattern-collapse lowering pass** in `codegen.py` — **DONE**
   (`generate_collapsed`, verified bit-identical over 12 kernel classes).
2. **Catalog + manifest generator**: enumerate §1, emit manifests; NumPy
   backend catalog complete at this point.
3. **Einsums/C++ backend + C ABI** (real scalar), pinned to Einsums v1.1.x,
   with a standalone CMake mini-library `libxckernel` and a C test driver
   validating against the NumPy backend on identical inputs. First action:
   resolve the TensorView-wraps-external-buffer question (zero-copy ABI).
4. **PySCF host module** (mostly repackaging existing test wiring).
5. **Psi4 plugin prototype**: reproduce `compute_Vx` (LDA/GGA) to machine
   precision, then ship the missing meta-GGA path.
6. **HelFEM**: complex instantiation (mind the no-conjugation einsum caveat
   -> gerc/pre-conjugated buffers) + a linear-response demo module.
7. **ERKALE**: generated ground-state path (validate vs existing), then
   all-family Casida.
8. **OpenMolcas**, reshaped by the survey: (a) generate the MC-PDFT
   translation-layer derivatives (Pi ingredient + translated densities;
   replaces hand-written Jacobian/Hessian Fortran, validated against
   translatedens.F90); (b) optionally, the net-new on-grid response path
   (fxc+ through the translation map) enabling MC-PDFT linear response —
   a new capability, not an integration.

9. **VeloxChem**: Python-level prototype behind `_comp_nlr_fock` (NumPy
   backend), then the C++ integrator-internal replacement of
   DensityGridQuad/Cubic + mode strings, jointly with the VeloxChem
   developers.

Ordering rationale: 1–3 are the shared substrate; 4 is nearly free; 5–7 are
independent of each other (parallelizable); 8–9 are largest, are genuinely
new science-enabling work rather than plumbing, and benefit from the earlier
phases landing first.

## Additional hosts (surveyed 2026-07-04)

### ChronusQ (B2, C++ shim over the C ABI) — effort: medium-to-large
Boundary already matches: `TwoBodyContraction<T>` (raw `X` in, `AX` out) is
literally DM-in/Fock-out, LR-TDDFT contracts fxc on the grid behind
`KohnSham::formFXC` (a 3500-line hand-differentiated header — the prime
replacement target), and the GauXC switch proves ChronusQ accepts an
external pointwise XC backend behind a boolean. In-house path is LDA/GGA
only, fxc-max; **no meta-GGA, no kxc/quadratic response** — all net-new
value. Two ChronusQ-specific requirements for xckernel:
1. **Noncollinear-magnetization ingredient set** (2c/4c): variables
   (ρ, m_x, m_y, m_z, gradients) mapped nonlinearly to Libxc spin variables
   via |m|, K = m/|m| with a small-|m| regularization branch (JCTC 2017, 13,
   2591) — a new ingredient class like MC-PDFT's Π; the AD chain must carry
   the |m|-map derivatives.
2. **Complex (dcomplex) contraction paths** for GIAO/2-component — complex
   *densities* (distinct from HelFEM's complex basis), so the complex
   instantiation serves two hosts.
Seam: the `mkAuxVar -> loadFXCder -> constructZVarsFXC -> formZ_fxc` middle;
collocation (`evalDen`) and AO assembly (`formZ`) stay host-side. Buffers
are plain contiguous `double*`/`dcomplex*` — C-ABI-friendly; a thin C++
header shim fits the `<MatsT>` templating.

### MRChem (C++/Eigen stage-A kernels) — effort: medium; out-of-family #2
Multiwavelet, matrix-free — yet the pointwise seam exists *exactly*:
`Functional::contract_transposed` + the hand-written `xc_mask`/`d_mask`
chain-rule tables (`xc_utils.cpp`), **hard-capped at order 2 with explicit
`MSG_ABORT`s for order > 2**. The crispest confirmation of the thesis in
the whole survey set: MRChem uses XCFun, an AD library that would supply
derivatives at ANY order — but quadratic/cubic response is absent because
the *contraction bookkeeping* was never hand-derived. xckernel generates
precisely that layer. Requirements: a C++/**Eigen** emitter flavor
(per-node dense matrices in/out), outputs in **grad-rho representation**
(fold the vsigma->grad-rho chain into the coefficients; MRCPP applies the
divergence), energy-density convention care (XCFun per-volume vs Libxc
per-particle), LDA/GGA only (mGGA is rejected host-side). Coordinate with
the in-flight `FunctionalBackend` refactor, which targets the same seam.
Payoff: real-space quadratic/cubic response MRChem cannot currently do.

### NWChem (B2/B3 via C ABI at the batch worker) — effort: moderate
Unusually clean: ALL orders route through one grid driver keyed by
`calc_type` (SCF=vxc; CPKS/TDDFT fxc = calc_type 2; TDDFT-gradient kxc =
calc_type 5), with the pointwise contraction + AO assembly in
`xc_tabcd.F`/`xc_3rd_deriv.F` operating on plain local (nq x nbf) arrays —
Global Arrays stays entirely host-side. NWChem also has THREE functional
providers (34 legacy hand-coded fxc files + 18 kxc files; the
**Maxima-generated `nwxc` library** — NWChem already accepted in-house
symbolic codegen for the pointwise tower; and a Libxc F03 interface whose
wrapper is the template for ours). The quantified gap, verbatim from
`xc_3rd_deriv.F`: *"not yet implemented for meta-GGA functionals"* —
meta-GGA TDDFT analytic gradients are blocked on exactly the missing
hand-derived contraction; 4th order absent entirely. Work: map or bypass
the `Amat*/Cmat*/Mmat*` column-packing macros; clone the Libxc-interface
wrapper pattern; new/extended `calc_type` for the added orders.

### LSDalton (B2/B3 at the native worker seam) — effort: medium
Own F90 stack, **XCFun (no Libxc at all)** — the one architectural
mismatch. The hand-written contraction layer is found precisely:
`II_dft_ksm_worker.F90` (6540 lines), with the same chain rule duplicated
per backend (XCFun branch and classic-Dalton-C branch), native response
capped at kxc, and **no meta-GGA response workers at all**. Marshalling is
favorable (dense full DMs, ABI-shaped `GAO(NBLEN,NACTBAST,NTYPSO)` batches,
strong iso_c_binding + CMake external-dependency culture). Design fork to
resolve: retarget xckernel's functional-symbol registry to XCFun's
component packing (a manifest repack shim), or add Libxc to LSDalton.
Note: the OpenRSP/XCint external path already does arbitrary order — the
native-worker seam adds value without competing with XCint.

### eT (B3 Fortran/C ABI; social wedge) — effort: gated on the compiled backend
Correction from the survey: the eT-susi checkout is identical to upstream —
the DFT engine (closed-shell LDA/GGA, Libxc F03, TDDFT with hand-coded fxc
landed 2026-03) is upstream eT-team code; the fork adds only Libxc CMake
discovery. The seam is textbook (`construct_ao_W_xc_and_energy_xc`,
`calculate_td_fxc` — same intermediates xckernel emits), marshalling easy
(dense AO matrices, batched screened GEMM contraction, iso_c_binding via
xc_f03 precedent; Cartesian-then-transform convention to match). What
xckernel adds is everything the young engine lacks: meta-GGA, open shell,
singlet/triplet, orders 3-4. Adoption needs eT DFT authors' buy-in; the
clean wedge is offering net-new capability kernels rather than rewriting
the working LDA/GGA path.

## Cross-cutting findings from the extended survey (11 hosts total)

1. **The gating prerequisite is the compiled backend** (phases 2-3): four
   Fortran hosts (NWChem, LSDalton, eT, OpenMolcas) and three C++ hosts
   (ChronusQ, MRChem-Eigen, plus Psi4/ERKALE/HelFEM/VeloxChem) all wait on
   it. NumPy serves only PySCF and prototyping.
2. **Ingredient extensibility is a first-class feature, not a roadmap
   afterthought**: two hosts need new nonlinearly-mapped ingredient sets
   (OpenMolcas MC-PDFT Pi; ChronusQ noncollinear (rho, m)). Both are maps
   (base fields) -> (Libxc variables) whose derivatives the tower must
   carry — the same mechanism, worth designing once.
3. **Complex contractions serve two distinct needs**: complex basis
   (HelFEM) and complex densities (ChronusQ GIAO/2c).
4. **Functional-derivative provenance must be pluggable**: Libxc (most),
   XCFun (LSDalton mandatory, MRChem optional) — a symbol-registry/manifest
   repack, not a redesign; NWChem's Amat/Cmat/Mmat is a third packing.
5. **The order-cap pattern is universal**: every host stops exactly where
   hand derivation stopped (Psi4 fxc/no-mGGA; ChronusQ fxc; MRChem order 2
   + aborts; NWChem kxc-no-mGGA; LSDalton kxc, no mGGA response; eT fxc
   closed-shell; ERKALE LDA-Casida; HelFEM vxc; OpenMolcas vxc). And two
   communities already accepted generated pointwise code (NWChem's Maxima
   nwxc; XCFun itself) — the missing piece everywhere is the generated
   *contraction*, which is xckernel's exact product.
