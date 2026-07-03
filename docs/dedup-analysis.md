# Cross-code deduplication analysis: DFT response stacks

**Codes surveyed:** Dalton (local checkout + public GitLab), PySCF (local checkout),
Psi4 (local checkout), VeloxChem (local checkout + public GitHub), ERKALE (local
checkout; Gaussian AO + Casida TDDFT), HelFEM (local checkout; finite-element
bases — the out-of-family control).
**Method:** parallel code surveys of the full response stacks, from property driver
down to the XC contraction, categorized into ten recurring structures.
**Excluded by design:** quadrature grid generation/partitioning/weights, integral
(J/K) engines, MPI/GPU orchestration, iterative eigensolvers.

Date: 2026-07-04.

## The universal stack

Every code implements the same pipeline, differing only in language, conventions,
and how far up the derivative ladder it goes (Psi4: linear; PySCF: linear + kxc in
TDDFT gradients; VeloxChem: cubic; Dalton: cubic + MCSCF):

```
response vector (kappa; occ x vir, Z/Y pair, irrep/spin labels)
        | pack/unpack                                     [S1]
perturbed density matrices  D^X = [kappa, D], nested commutators   [S2]
        | collocation                                     [S7]
perturbed fields on grid    rho^X(r), grad rho^X(r), ...
        | pointwise chain rule / derivative-variable conversion    [S6]
distribution coefficients   vt0(r), vt_x(r)
        | back-projection onto phi_p phi_q                [S8]
AO Fock-like response matrix
        | MO transform + gradient projection              [S9, S4]
sigma vector / E[n] contraction  (+ orbital-energy and metric terms) [S3]
        with spin adaptation [S5] and permutation bookkeeping [S10]
```

The XC boundary is unanimous: **perturbed AO density matrices in, AO Fock-like
matrices out**; functional derivatives contracted with perturbed densities
*pointwise on the grid*; no code ever materializes an AO kernel tensor with more
than two indices. The orbital/kappa picture stays in the solver — with one
instructive exception (Dalton orders 3-4 evaluates perturbed densities directly
from kappa chains on the grid, but this is an evaluation *strategy* for the same
mathematical object Tr(D^X Omega(r)), and its output is still a Fock-like matrix).

## Deduplication matrix

### S1. Response-vector packing (kappa <-> matrix)

| Code | Implementation |
|---|---|
| Dalton | `GTZYMT` pair-list (`MJWOP`), layout `[Z_conf,Z_orb,Y_conf,Y_orb]`, irrep-blocked, hard `NSIM=1` limit (rsp/rspqrx3.F:2776) |
| PySCF | `zs.reshape(-1,nocc,nvir)`; TDHF `(x,2,no,nv)`; CPHF uses *transposed* `(nvir,nocc)`; RHF norm 1/2 vs UHF norm 1 (tdscf/rhf.py:92,1119; cphf.py:52) |
| Psi4 | occ x vir `Matrix` per irrep; UHF alternating alpha/beta list; `SingleMatPerVector`/`PairedMatPerVector` (scf_products.py:61/88) |
| VeloxChem | `lrvec2mat`/`lrmat2vec` (linearsolver.py:3459/3517); half-size paired storage with gerade/ungerade expansion `(v;v)/(v;-v)`, 1/sqrt(2) normalization + factor-2 half-space inner products; `anti_sym` Omega-metric; Re/Im interleaved columns for complex response |

Same math: antisymmetric generator over occ-vir pairs with (Z,Y) doubling and
(irrep, spin) labels. Different: index order, Z/Y ordering, normalization,
half-size pairing, batching. **Shared library:** one packed-rotation type;
conventions as explicit parameters; batching first-class (Dalton's NSIM=1 vs its
own batched linear response shows the cost of not doing this); VeloxChem's
paired-vector algebra (gerade/ungerade split, Omega metric, paired
preconditioners) is itself a reusable RPA-structure kit.

### S2. Perturbed-density construction

| Code | Implementation |
|---|---|
| Dalton | `DEQ27` one-index transform `D^X_pq = B_tp D_tq - D_pt B_qt` (rsp/deq27.F:23); `commute_d_x` idempotent shortcut + `D^YZ=[[kY,D],kZ]+[[kZ,D],kY]` (quad-faster.c:52-179); dual MO kappa-chain route |
| PySCF | `dm1 = einsum('xov,pv,qo->xpq', z, orbv, orbo*2)` (tdscf/rhf.py:98); relaxed 2nd-order densities in grad/tdrks |
| Psi4 | `Dx = C_occ (C_vir x)^T` via JK C_left/C_right (rhf.cc:481-497) |
| VeloxChem | `commut_mo_density`; `D_bc = [k_b,D_c] + [k_c,D_b]` (quadraticresponsedriver.py) |

Same math: nested commutators of kappa with an idempotent density. **Shared
library:** symbolic generator of n-th order perturbed DMs (and, as a codegen
alternative, Dalton's rank-1 kappa-chain evaluation of the same trace — two
lowerings of one expression).

### S3. Sigma-vector / E[n] assembly

| Code | Implementation |
|---|---|
| Dalton | `RSPOLI` + `FCKOIN` (`[kappa,F]` term) + XC + projector; `T3DRV` = E[3] perms + omega*S[3]; `Q3FOCK` nested transforms; `addfock` ownership flag (rsp/rspoli.F, rspqrx3.F:31-519) |
| PySCF | `v1ao = vresp(dms)` + `e_ia` diagonal; RSH 4-branch exchange bookkeeping copy-pasted ~6x (_response_functions.py) |
| Psi4 | `onel_Hx` = `F.x - x.F`; `twoel_Hx_full` J/K/Vx combos — **assembled three times** (C++ CPHF, Python TDSCF, inline scfgrad) with only scale factors differing |
| VeloxChem | closed-form BCH commutator assembly: `xi(kA,kB,FA,FB) = 1/2([kA,[kB,F0]+2FB] + [kB,[kA,F0]+2FA])` for E[3], `zeta` for E[4] with S[4] = `(2/6)[[k3,[k2,k1]],D^T]` and R[4] = i*gamma S[4]-type terms (nonlinearsolver.py:1391-1435); E[2] sigma = `-[F^kappa + [F0^T,kappa], D]` (linearsolver.py:1836-1866) |

Same math: generalized Hessian action `[kappa,F] + 2e-response + XC-response`
(+ `omega*S[n]` metric terms at higher orders); VeloxChem's `xi`/`zeta` are the
explicit BCH-expansion closed forms of exactly what Dalton assembles via nested
`OITH1`/`Q3FOCK` calls. **Shared library:** one E[n] action template — a small
commutator DSL generating `xi`/`zeta`-type expressions at arbitrary order — with
an explicit *term-ownership spec* (Dalton's `addfock` shows the boundary must be
stated, not assumed).

### S4. Fock-like matrix -> orbital-gradient projection

Dalton `ORBEX` (full active-space algebra, rspqrx3.F:1821, >=3 near-identical
copies: RSPORB, ORBSX); PySCF `einsum('xpq,qo,pv->xov')`; Psi4 `_so_to_mo` =
`Co^T X Cv`; VeloxChem `lrmat2vec`. Same math: `g = <0|[E_kl, K]|0>` projected
onto the rotation pair list, density-weighted. **Shared library:** one projector
parametrized by occupation structure (closed/open/active) — subsumes the TDA
guard (drop Y-side terms) as a flag.

### S5. Spin adaptation

| Code | Implementation |
|---|---|
| Dalton | spin bit per perturbation; `dftpot1/2` expose +/- combinations; bitwise `sY\|sZ`, `sY^sZ` indexing (quad-fast.c:461-478); triplet = flip sign of beta perturbed fields |
| PySCF | `nr_rks_fxc_st` aa+/-ab; general `ud2ts` rotation `[[.5,.5],[.5,-.5]]` per tensor axis (xc_deriv.py:615-653); `fxc*=.5` conventions |
| Psi4 | `singlet` flag => beta kernel components = +/-alpha (v.cc:2444-2450) |
| VeloxChem | **no triplet response machinery at all** in the nonlinear stack; closed-shell collapse via precomputed per-point coefficient tables `rr = v2rho2_aa+v2rho2_ab`, `rrr = v3rho3_aaa+2aab+abb`, ..., `xxx` with up to 9 terms (XCIntegratorForGGA.cpp:2436-2455), recurring across every order/family |

Same math: a tensor-product +/- rotation (up/down -> total/spin-difference basis)
applied per perturbation index; triplet-ness = spin parity of the perturbation
tuple. **Shared library:** generated spin projections at arbitrary order —
PySCF's `ud2ts` is the right primitive, Dalton's bitwise algebra is its order-n
consequence, VeloxChem's collapse tables are its closed-shell specialization
(Faa di Bruno over rho_a=rho_b, sigma_aa=sigma_ab=sigma_bb); all fall out
automatically from a symbolic spin basis change. VeloxChem's missing triplet
support is a gap the generator fills for free.

### S6. Derivative-variable conversion (the strongest AD case)

| Code | Implementation |
|---|---|
| Dalton | functionals in (rho_a, rho_b, \|grad_a\|, \|grad_b\|, grad_ab); response in (R, Z, G); `dftpot0-3ab` conversions with 1/grad^7 binomial resummations (general.c:221-485 — "transform to closed shell, second time. Oh, I love this mess") |
| PySCF | `xc_deriv.transform_xc`: sigma-form -> density-parameter vector `[rho, grad, tau]`; `_stack_fg/_stack_frr/...` symmetric-index defolding; hand-expanded `_rks/_uks_*_wv0/1/2` chain rules, hundreds of lines (numint.py:1551-2000) |
| Psi4 | gamma convention; hand-coded "V/W-term" GGA algebra (v.cc:1820-1948); **MGGA Vx unsupported** — the hand-derived terms were never written |
| VeloxChem | `DensityGridQuad.cpp` (25.7k lines) + `DensityGridCubic.cpp` (42.2k lines) of pointwise perturbation products (`gam`, `gamX..gamZZ`, `pi`, `rt_gam`, `st_gamX`, ...), selected by 16 hand-enumerated mode strings ('qrf','shg','tpa','crf','thg','*_ii',...) encoding only (a) permutation subset, (b) real vs complex, (c) recursion stage; per-mode density-count tables duplicated between Python and C++ (a live coupling hazard, nonlinearsolver.py:536-624) |

Same math: the chain rule between derivative-variable bases and the contraction
of functional derivatives with perturbed fields into per-point distribution
coefficients. This is *exactly* xckernel's derivative tower: every one of these
hand-written blocks (Dalton's `dftpot3ab_`, PySCF's `wv2`, Psi4's missing MGGA
terms, VeloxChem's mode strings) is frozen output of a symbolic differentiation
that the library performs mechanically at any order. The gaps (Psi4 MGGA, PySCF
lxc UKS, Dalton meta-GGA response) are unwritten instances a generator fills for
free.

### S7. Field evaluation from matrices (collocation)

Dalton `getexp_blocked_*` (unsymmetric-matrix expectation over a batch) + the
dual kappa-chain route (`CommData` rank-1 lists, cube-fast.c:84-100); PySCF
`eval_rho/eval_rho1/eval_rho2` (DM vs MO-coeff routes); Psi4 `points.cc` — with
the perturbed-density variant *duplicated inline* in v.cc:1900-1926; VeloxChem
`sdengridgen::serialGenerateDensity*`. Same math: `Tr(D^X Omega(r))` and its
gradient/tau slots. **Shared library:** generated collocation per ingredient —
in xckernel's language this is the contraction of ingredient seeds with D
(the field-folding identity `sum_ts seed_k(t,s) D_ts = k^X(r)`).

### S8. Distribution / back-projection

Dalton `DISTLDAB/DISTGGAB` + response variants with irrep offsets and the
half-triangle traps ("symmetrize *without* 1/2" vs "with 1/2" depending on how
the diagonal was weighted — the most error-prone recurring idiom); PySCF
`_scale_ao/_dot_ao_ao/_tau_dot` + `hermi_sum`; Psi4 `phi^T T` DGEMM + atomic
scatter; VeloxChem `mat_G` + `serialMultABt`. Same math:
`M_pq += sum_g w_g c(g) phi_p(g) phi_q(g)` (+ gradient/tau slots). **Shared
library:** one generated distribute kernel with the symmetrization policy an
explicit parameter, not a convention.

### S9. MO<->AO transforms at the boundary

Commodity two-sided transforms, but with hidden factor conventions (Dalton
`LRAO2MO` multiplies by 2; PySCF folds occupancy 2 into the DM; Psi4 folds -1
into C_right). **Shared library:** the forward (DM build) and backward
(projection) transforms as an adjoint pair with factors explicit.

### S10. Permutation & response-order bookkeeping

Dalton: paired E3(jkl)+E3(jlk) calls; cube-fast's derived coefficient tables
(1/6; -1,+3,+3,+3,-3,-3,-3,+1) for the six triple-commutator permutations;
VeloxChem: the mode-string zoo; PySCF: `wv2 = kxc.rho1.rho1 + fxc.rho2` order
bookkeeping in gradients. Same math: permutation symmetrization over
perturbation labels, plus the (matrix, irrep, spin, frequency) metadata tuple
per perturbation. **Shared library:** an n-th order response context object and
generated permutation-summed contractions — this *replaces* mode strings.

## Additional codes: ERKALE and HelFEM

### ERKALE (Gaussian AO; the fifth in-family witness, plus one outlier)

Ground state conforms exactly to the pattern: `DFTGrid::eval_Fxc` is
AO-DM-in/AO-Fock-out (scf-fock.cpp.in:130-188); `update_density` is S7
(half-transform `Pv = P.bf`, then row reductions, dftgrid.cpp:417/494); the
`increment_lda/gga/mgga_kin/mgga_lapl` template family (dftgrid.h:623-775) is S8
with the symmetrization (`A.B^T + B.A^T`) explicit; the sigma chain rule and
open-shell `2*vsigma_aa*grad_a + vsigma_ab*grad_b` couplings are hand-expanded
(S6, dftgrid.cpp:1721, 1902-1915) exactly like Psi4/HelFEM.

**The Casida outlier (instructive deviation):** `CasidaShell::Kxc`
(casida/casida_grid.cpp:138-200) does NOT build an AO fxc matrix. It evaluates
MO values on the grid (`compute_orbs`: `C^T.bf`) and contracts

    K(ia,jb) += sum_g w_g fxc(g) phi_i phi_a phi_j phi_b

directly into the MO pair-space coupling matrix — fusing the AO->MO transform
into the grid loop. In the library's language this is not a different interface
but a different *lowering*: the free index pair Omega_pq of the contraction is
instantiated with MO orbitals instead of AOs (xckernel's index labels are
already arbitrary, so this is one codegen flag, not a new code path). The cost
of the hand-written version: ERKALE's Casida is **LDA-only** (explicitly
rejected otherwise, casida_grid.cpp:96-99) — the GGA/mGGA response terms were
never hand-derived. Another unpaid-derivation-cost gap, same as Psi4's missing
MGGA-Vx and VeloxChem's missing triplet.

### HelFEM (finite elements; the out-of-family control) — the abstraction holds

HelFEM (radial FEM x spherical harmonics; prolate spheroidal for diatomics)
runs the *identical* pipeline: `update_density` (P -> rho, grad, tau, lapl on
quadrature) -> shared Libxc dispatch (`DFTGridWorkerBase::compute_xc`,
general/dftgrid_common.cpp:95-252) -> `eval_Fxc` with `increment_lda/gga/...`
symmetrized contraction templates (atomic/dftgrid.h:149-239). The
**basis-agnostic invariants confirmed**: (a) field = contract(P, basis-value
products) pointwise; (b) v = Libxc(rho, sigma, lapl, tau); (c) Fock +=
sum w.v.(symmetrized basis outer products); (d) identical spin conventions and
sigma chain rule. The chain-rule contraction exists in three near-duplicate
copies (atomic/diatomic/sadatom shell types) — in-code duplication mirroring
the cross-code duplication.

**What FEM changes — required parameterizations of the contraction backend:**
1. Basis arrives as **per-element dense blocks + a scatter index map**
   (`bf_ind`), not one global (npts x nbf) collocation array.
2. **Complex basis functions** (spherical harmonics / e^{im phi}); results
   realized at the end.
3. **Curvilinear gradient metric**: gradient components are d/d(r,theta,phi)
   with geometry-dependent 1/scale factors (prolate spheroidal for diatomics)
   — the three "gradient directions" are not Cartesian.
4. FEM boundary-node expand/remove bookkeeping (plumbing, host-side).

None of these touch the *mathematics* of S6-S8 — they are attributes of the
collocation operands. HelFEM has **no response module** (properties via finite
fields; only vxc in the hot path) — but its one use of fxc,
`sadatom::TwoDBasis::xc_screening` (sadatom/basis.cpp:1032-1195), hand-expands
the full GGA->local-potential divergence chain rule from
v2rho2/v2rhosigma/v2sigma2 with curvilinear divergence terms — per the survey,
"precisely an AD-through-Libxc expression", i.e. the strongest single-file
evidence that this algebra is codegen-replaceable. A response module for HelFEM
would be a direct consumer of the generated contraction engine.

## What is genuinely not deduplicable

- Quadrature grids (excluded by design) and basis/integral engines.
- Symmetry frameworks (Dalton's irrep machinery has no counterpart in C1-only
  VeloxChem drivers) — though the (irrep) *label* slot costs nothing to carry.
- Parallel/GPU orchestration and I/O.
- Iterative solvers (Davidson/CG) — commodity numerics, albeit duplicated even
  within single codes (Psi4 has two independent solver stacks).

## Direct evidence of the duplication cost

- Psi4 assembles the same orbital-Hessian action **three times** in two
  languages; its `compute_Vx` still lacks MGGA support.
- PySCF's `_uks_gga_wv2` and relatives are hundreds of lines of hand-expanded
  chain rule; the RSH exchange dispatch is copy-pasted ~6 times.
- Dalton's `dftpot3ab_` is ~150 lines of binomial resummation done by hand in
  1/grad^n variables; the in-source comments testify to the pain. NSIM=1 hard
  limits persist in the higher-order stack because batching was never retrofitted.
- VeloxChem needs a new hand-written mode string + DensityGridQuad branch per
  property class: `DensityGridQuad.cpp` + `DensityGridCubic.cpp` total **~68k
  lines of mechanically derivable code**, plus ~120 lines of hand-enumerated
  real/imag operator-parity sign tables per driver (derivable from Hermiticity/
  time-reversal signatures), plus density-count tables that must be kept
  consistent across the Python and C++ layers by hand.
- VeloxChem's nonlinear stack has no triplet support; Psi4's Vx has no MGGA;
  Dalton's meta-GGA response is absent; ERKALE's Casida is LDA-only; HelFEM has
  no response module at all — none of these are physics decisions, they are
  hand-derivation costs that were never paid.
- ERKALE's `increment_*` templates, HelFEM's `increment_*` templates, PySCF's
  `_scale_ao/_dot_ao_ao`, Psi4's collocation DGEMMs, Dalton's `DIST*`, and
  VeloxChem's `mat_G`/`serialMultABt` are six implementations of the same
  distribute kernel; HelFEM additionally carries three internal near-duplicates
  (atomic/diatomic/sadatom).

## Implications for xckernel

Three layers on top of the existing symbolic core, all host-agnostic:

1. **Contraction engine (fuses S6+S7+S8):** runtime-generated response-Fock
   builders at arbitrary derivative order:
   `contract(order, D0, [(D^X, spin, irrep), ...]) -> AO Fock-like matrices`,
   batched over perturbations, with the perturbed-field folding keeping cost at
   O(N^2 * n_grid) per vector. Metadata triples and term-ownership flags are part
   of the signature. The collocation operands must be parameterized (HelFEM
   constraints): real or complex basis blocks, global collocation vs
   per-element blocks + scatter map, Cartesian vs curvilinear gradient metric.
   The free output index pair may be AO or MO (ERKALE-Casida-style direct
   MO-pair contraction is the same generated expression with MO-valued
   operands).
2. **Response algebra (S1-S5, S9, S10):** the "MO picture" — packed rotations,
   perturbed-DM generators, E[n] action template, gradient projector, spin
   projections, permutation bookkeeping — generated with conventions (kappa sign,
   Z/Y order, normalization, factor placement) as explicit parameters. This is
   the layer every code currently hand-duplicates.
3. **Host adapters:** thin mappings to PySCF/Psi4/VeloxChem/Dalton data layouts.

The precedent argument: Libxc deduplicated pointwise functional evaluation and
was adopted by all four codes. Layers 1-2 deduplicate the *contraction and
response algebra* above it, which the survey shows to be the same mathematics
implemented four times — with the hand-written parts (S6 especially) being
precisely frozen symbolic-differentiation output that this library generates.
