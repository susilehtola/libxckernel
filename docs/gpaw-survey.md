# GPAW survey: the plane-wave host demonstration

Surveyed 2026-08-14 on GPAW master (`8fb9ed10`, version 26.7.1b1) at
`/home/work/gpaw`. Scope: plane-wave and finite-difference modes only —
LCAO is excluded from the pilot (and, conveniently, from most of GPAW's
own mGGA support). Companion to the six-code LCAO survey
(`dedup-analysis.md`) and the per-host sections of the historical
`interfacing-plan.md`.

## Executive summary

GPAW is the ideal uniform-grid host for the same reason DIRAC was the
ideal generated-code host: **it already has our architecture, stopped at
first order.** The ground-state XC layer is split into grid logic
(gradients, divergences, PAW radial expansion, stress) and a duck-typed
per-point kernel — `XCKernel.calculate(e_g, n_sg, dedn_sg, sigma_xg,
dedsigma_xg, tau_sg, dedtau_sg)` (`gpaw/xc/kernel.py:39`) — which is
exactly the fields-in/coefficients-out contract the emitters target. The
response side, however, is **adiabatic LDA only, everywhere**: one
`assert fxc in ['ALDA_x', 'ALDA_X', 'ALDA']`
(`gpaw/response/fxc_kernels.py:207`) gates the dielectric/TDDFT kernels,
the Casida codes finite-difference v_xc and would be *silently wrong*
for mGGA (τ is never perturbed), DFPT was deleted in 2025 after its
fxc call had been dead since 2020, and the only GGA-flavoured kernel
(rAPBE) was retired in June 2026. Every derivative past the first is
either absent, hand-capped, or wrong — the exact profile the generator
addresses.

## Ground state: what exists (and what we reuse)

* **Kernel seam.** `XC()` (`gpaw/xc/__init__.py:33`) dispatches on
  `kernel.type` ∈ {LDA, GGA, MGGA}. A generated kernel object with
  `.name`, `.type`, `.calculate(...)` is a complete drop-in; the
  functional wrappers then supply gradients, divergence, PAW radial
  expansion, and stress for free. Conventions to honor: `e_g` is the
  energy density **per volume** (ρ·ε, not Libxc's zk) and is
  overwritten; `dedn_sg` is **accumulated** (+=); `dedsigma_xg`/
  `dedtau_sg` are overwritten; `sigma_xg` may be clobbered by the
  caller afterwards (`gpaw/xc/gga.py:173`). Kernels must be
  shape-agnostic in the trailing dimensions: the same method receives
  3-D `(S, nx, ny, nz)` blocks on grids and 1-D `(S, ng)` arrays on the
  50-point Lebedev × radial PAW grids (`gpaw/xc/gga.py:122`).
* **Gradients are FD stencils even in PW mode.** The density gradient
  is always taken with real-space `Gradient` operators
  (`gpaw/fd_operators.py:284`; default `stencil=2`, O(h⁴)); only
  orbital quantities (τ, the mGGA Hamiltonian term, the KED stress
  tensor) use iG in PW mode. Emitted kernels consume fields, so this
  is transparent — but FD-vs-spectral gradients set the validation
  noise floor.
* **mGGA ground states work in PW and FD** (not LCAO, not GPU): τ via
  `add_ked`, the potential via ∇·(v_τ∇)ψ in
  `gpaw/new/pw/hamiltonian.py:74` / `gpaw/new/fd/hamiltonian.py:34`.
  No Laplacian plumbing exists anywhere (`FunctionalNeedsLaplacianError`,
  `gpaw/xc/libxc.py:85`), so the pilot family set is
  lda/gga/mgga_tau.
* **Analytic XC stress exists in PW mode — including mGGA**, with the
  kinetic-energy-density tensor `taut_swR` (6 components, hess packing;
  `gpaw/new/xc.py:402`) contracted against `dedtaut`: GPAW already
  builds the exact operand our strain seeds need. The GGA anisotropic
  stress term (−2·∂e/∂σ·∂ᵥ₁n·∂ᵥ₂n, `gpaw/xc/gga.py:14`) is term-for-
  term our first-order strain seed of σ. FD mode has **no stress at
  all** (`e_stress = nan`, `gpaw/new/fd/pot_calc.py:143`; legacy raises
  NotImplementedError).
* **Two calculator generations.** The new path (`gpaw/new/` +
  `gpaw/core/` `UGDesc`/`UGArray`/`PWDesc`) is the default for pw/fd;
  legacy code lives in `gpaw/old/`. The response code runs entirely on
  legacy descriptors, bridged by `UGDesc._gd`
  (`gpaw/core/uniform_grid.py:247`). Future-proof integration order:
  (1) a `Functional` subclass in `gpaw/new/xc.py` terms
  (`calculate(nt_sr, taut_sr) -> (exc, vxct_sr, dedtaut_sr)` +
  `_args`/`_stress`), (2) an `XCKernel`-shaped per-point object,
  (3) the response-side protocols below.
* **Parallelism.** Per-point kernel work is embarrassingly parallel
  over the domain decomposition; gradients need halo exchange (handled
  by the existing operators); the new core's FFTs serialize through
  rank 0 — relevant when folding kernels to G space.

## Response: the gap we fill

Confirmed blockers (each with one representative site):

1. ALDA-only gate: `gpaw/response/fxc_kernels.py:207`; non-adiabatic
   names rejected at `gpaw/response/density_kernels.py:35`; spin-paired
   only (`density_kernels.py:19`).
2. The C fxc binding (`calculate_fxc_spinpaired`, `c/xc/libxc.c:480`)
   asserts against the mGGA family; GGA second derivatives are wired in
   C but never passed from Python (a NULL-deref trap, not a feature).
3. The kernel *representation* assumes locality: `FXCKernel` stores
   fxc(G−G′) only (`fxc_kernels.py:16-40`) — structurally incapable of
   GGA/mGGA kernels, whose ∇·(…∇) and τ couplings are not diagonal in r.
   The clean seam is the `PWKernel` ABC (`gpaw/response/dyson.py:31`):
   any object with `get_number_of_plane_waves()` and `_add_to(x_GG)`
   can enter the Dyson equation.
4. The functional-evaluation protocol `add_f(gd, n_sR, f_R)`
   (`gpaw/response/localft.py:96`) passes the density only — no σ, no
   τ, no perturbed fields; inside augmentation spheres it receives a
   *radial* gd and a `MicroSetup` carrying `n_sLg`/`nt_sLg` only.
   `ResponseGroundStateAdapter` exposes no τ at all.
5. Casida codes (`lrtddft`, `lrtddft2`) build fxc by two-point finite
   differences of v_xc against pair densities (numscale 1e-5). Nothing
   validates the functional type: with an mGGA the τ channel is simply
   omitted — **wrong answers with no error**
   (`gpaw/lrtddft/omega_matrix.py:210`,
   `gpaw/lrtddft2/k_matrix.py:417`). The analytic branch
   (`derivativeLevel=2`) calls a method deleted in 2020.
6. DFPT: `gpaw/dfpt/` removed 2025-05-27 (commit `15484b4f50`) after
   ~4.5 years broken; phonons are ASE finite differences. rAPBE — the
   only GGA-ish kernel in the ACFDT/GW path — retired 2026-06-15
   (commit `546a1622fa`).
7. No quadratic response, no analytic excited-state gradients
   (finite-difference forces only), so orders ≥ 3 have no existing
   consumer — net-new capability territory.
8. Real-time TDDFT needs no kernel work at all (it re-evaluates v_xc
   each step): mGGA RT-TDDFT is purely a ground-state-layer question
   and works wherever the mGGA SCF works.

PAW infrastructure we can reuse verbatim: `LocalPAWFTEngine`
(`localft.py:324`) computes augmentation-sphere corrections
generically for any per-point function via real-spherical-harmonic
expansion — only the `add_f` payload (σ, τ, perturbed fields) and the
`MicroSetup` contents need widening. The newer self-enhancement path
(`chiks.py:441`, `matrix_elements.py:197`) folds the kernel into pair
potential matrix elements instead of a K_GG matrix — architecturally
the friendliest place for a semilocal kernel.

## Plan

**Phase 0 — stress regeneration (cross-validation, small).** Emit
GPAW's XC stress contribution (`stress_lda_term` + `stress_gga_term` +
`stress_mgga_term`) from `strain_energy_derivative` and verify
numerical identity against `Functional.stress_contribution` on real
GPAW calculations. Proves the strain seeds against an independent
production implementation, and gives the manuscript a "regenerated a
shipped implementation" data point for the uniform-grid case (the
VeloxChem move, PW edition).

**Phase 1 — Casida mGGA (fills a silent-wrongness hole).** Replace the
finite-difference pair potential in `lrtddft2`/`lrtddft` with generated
analytic fxc contractions including the τ channel: pair densities are
already on the fine grid; pair τ (½∇ψᵢ·∇ψⱼ) comes from the existing
`Gradient` operators; the PAW ± trick is replaced by the generated
radial-grid kernel through the same `calculate_paw_correction`
machinery. Γ-only/molecular, FD mode: modest scope, immediate
correctness payoff, exercises the response emitter end to end.

**Phase 2 — semilocal kernels for the dielectric/TDDFT path (the
flagship).** A new `PWKernel` subclass that represents the generated
GGA/mGGA kernel exactly: per-point coefficient fields (the u/v/τ
channel structure) FFT'd once, assembled into K_GG′ with the G-vector
factors carried by the vector channels; τ vertices require kinetic
pair-density matrix elements, which the `matrix_elements.py` machinery
is built to provide (new matrix-element class). Widen `add_f` and
`MicroSetup` for the PAW corrections. This is the "medium-high" host
work and the paper's headline: mGGA dielectric functions/EELS in a
plane-wave code.

**Phase 3 — strain tier.** With phase 0 done: second strain
derivatives (elastic constants) from `strain_energy_hessian` — no
framework exists in GPAW at all — and FD-mode stress as a by-product
(the seeds don't care that GPAW's FD mode never got a stress
implementation).

Risks: the e_g/zk and accumulate-vs-overwrite conventions (mechanical,
but silent if wrong — validate against GPAW's own kernels first); PAW
completeness for kernel corrections (reuse `LocalPAWFTEngine`, compare
against `LocalGridFTCalculator` all-electron mode); the response code's
legacy-descriptor dependency (target `gd = UGDesc._gd` shapes); FD
noise floors set by stencil order rather than machine precision.
