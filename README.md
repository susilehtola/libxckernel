# libxckernel

[![CI](https://github.com/susilehtola/libxckernel/actions/workflows/ci.yml/badge.svg)](https://github.com/susilehtola/libxckernel/actions/workflows/ci.yml)

> The project (and the generated C library) is **libxckernel**;
> the Python generator package imports as `xckernel`
> (as `pylibxc` is to `libxc`).

**An automatic-differentiation backend for [Libxc](https://libxc.gitlab.io/):
generate arbitrary exchange–correlation kernel elements for any
discretization — Gaussian LCAO, plane waves, real-space grids, finite
elements, in Cartesian or curvilinear coordinates.**

Density-functional response properties all reduce to derivatives of the XC
energy with respect to the density matrix, contracted with grid data and the
functional derivatives that Libxc provides. Every electronic structure code
hand-derives and hand-writes these contractions — the XC Fock matrix, the
TDDFT kernel, the quadratic-response prefactors — separately, per functional
family, per spin case, per response order. xckernel replaces that hand
derivation with symbolic differentiation and code generation:

* **Libxc owns** the functional-derivative tower
  ∂ⁿExc/∂{ρ, σ, ∇²ρ, τ}ⁿ (`vrho`, `v2rhosigma`, `v3sigma3`, …).
* **xckernel owns** the derivatives of the *ingredients* with respect to the
  density matrix, ρ(r) = Σ\_uv P\_uv χ\_u(r) χ\_v(r) and friends.

Composing the two by the chain rule — mechanically, at any order — yields any
XC kernel element as an Einstein-sum expression over basis-function values,
grid weights, and Libxc derivative arrays. Derivatives are applied
monomial-wise in a fast polynomial representation, and the result is
**pattern-collapsed**: every term factorizes as (basis-pair pattern) ×
(per-point scalar), and only a handful of patterns exist at any order, so
even a 130,566-term kernel lowers to a few GEMMs. Three emitter families
consume the collapsed form: ready-to-run NumPy (`einsum`), a low-level C
library with static coefficient tables, and host-idiom plugins that write a
program's own contraction style (demonstrated on Psi4:
[psi4/psi4#3458](https://github.com/psi4/psi4/pull/3458), and on GPAW:
[gpaw!3425](https://gitlab.com/gpaw/gpaw/-/merge_requests/3425)).

## The derivative tower

Everything is repeated application of one operator, `D_ts = ∂/∂P_ts`:

| quantity | expression | module |
|---|---|---|
| energy | `Exc` | — |
| XC Fock matrix | `F_uv = D_uv[Exc]` | `fock.py` |
| AO XC kernel | `g_uv,ts = D_ts D_uv[Exc]` | `kernel.py` |
| n-th order | `D … D[Exc]` | `kernel.py` |

`D` acts on two kinds of atom: ingredient fields (their derivative is a known
bilinear in the basis functions) and Libxc derivative symbols (their
derivative *bumps the order* — `d vrho = v2rho2·dρ + v2rhosigma·dσ + …`).
The chain terminates in basis data, so any order works (`deriv.py`).

Seeding `D` with a **perturbed-field symbol** instead of an orbital pair —
`Σ_ts (∂k/∂P_ts) D^X_ts = k^X(r)` — turns the same operator into the
**response contraction engine** (`response.py`): perturbed AO density
matrices in, AO Fock-like matrices out, at `O(N²·n_grid)` per perturbation,
never materializing an N⁴ tensor. A survey of six production codes (Dalton,
PySCF, Psi4, VeloxChem, ERKALE, HelFEM — see `docs/dedup-analysis.md`) shows
this is exactly, and unanimously, the interface response solvers want.

On top sits the **response algebra** (`algebra.py`): the "MO picture" every
code duplicates — transition/perturbed density builders (nested commutators
of κ at any order), gradient projections, TDA/RPA σ-vector templates, and an
**arbitrary-order response σ assembly**

```
σ_ia = Σ_{S⊆perturbations} Σ_{partitions π of S}
       Tr[ (g_{1+|π|} : Π_{β∈π} D^β) · ∂^{|S^c|+1}P(κ_{S^c}, K_ia) ]
```

for which linear, quadratic (E[3]) and cubic (E[4]) response are the n = 1,
2, 3 instances of one loop — no per-order hand derivation.

## The kernel catalog

`catalog.py` enumerates, generates, and manifests **407 kernels**: 169
named `xck_<family>_<case>_o<order>[_<parities>]` and 238 Fock-diagonal,
batched and MO-projected response, nuclear-gradient, Fock-derivative and
nuclear-Hessian
kernels (below), spanning seven functional
families — `lda`, `gga`, `mgga_tau` (τ-only), `mgga_lapl` (Laplacian-only),
`mgga` (full), `cmgga_tau` (current-density: the Libxc τ slot is fed the
gauge-corrected τ̃ = τ − j²ₚ/2ρ), and `hmgga` (density-Hessian η of
local-hybrid calibration functions) — in the restricted, unrestricted, and
closed-shell spin-adapted cases (singlet/triplet parity per perturbation)
through fourth derivative order. The six heaviest, the order-4 open-shell
`hmgga` kernels `xck_hmgga_{ua,ub}_o4` and `xck_hmgga_st_o4_*` (up to
~45 min and ~16 GB each to generate), are left out of default builds and
generated on request (`--include-heavy`, `-DXCKERNEL_INCLUDE_HEAVY=ON`).
Of these,
fifteen `xck_<family>_{r,ua,ub}_giao` kernels provide the explicit magnetic-field
derivatives of the Fock matrix with London (GIAO) orbitals, as the real
factor of dF/dB_s = (i/2c) K_s at a real reference (see "London orbitals"
below for their C interface). Every kernel
ships with a machine-readable manifest declaring its operands and shapes,
the Libxc arrays it consumes by name, and its term ownership. Beyond the
catalog: complex orbitals and complex basis functions (sesquilinear
emission, with split Re/Im storage as the recommended layout: the
requested part of the output is accumulated in real arithmetic, at half
the multiplication count when one part suffices, and operands declared
purely real or imaginary have their zero parts eliminated at generation
time), a matrix-free two-sided mode that emits σ-vector contractions
from MO-pair collocation, and nuclear derivatives of the XC contribution
including the full quadrature-grid response (`geometric.py`).

### Fock diagonal and nuclear gradient

The C package also carries kernels with one free basis index, or none:

| entry | `out` | contents |
|---|---|---|
| `xck_<family>_{r,ua,ub}_o1_diag` | `(nbf,)` | the diagonal `F_uu` of `xck_<family>_<spin>_o1` at O(nbf·ng), with the same arguments; all seven families |
| `xck_<family>_{r,ua,ub}_g1` | `(3, nbf)` | nuclear gradient, basis class, as per-function rows |
| `xck_<family>_{r,u}_gg` | `(3, ng)` | nuclear gradient, grid class, per grid point |

The gradient entries cover all seven families. Every class is returned
as **+dE/dX**, and the three classes add up to the full gradient with
grid response:

| class | what moves | how to evaluate dE/dX_{A,d} |
|---|---|---|
| basis | atom A's basis functions | `g1`: sum `out[d, u]` over the functions u on atom A. Inputs: the plain collocation tower `chi` (to third order for the Laplacian and density-Hessian families) and its density contraction `Dchi = D chi`. Unrestricted: call `ua` with Dᵅ and `ub` with Dᵝ, and add. |
| grid | atom A's grid points | `gg`: sum `out[d, g]` over the points g whose parent atom is A. Inputs: the density tower to second order (third for the Laplacian and density-Hessian families) and `tau_x`, `tau_y`, `tau_z` (per channel for `u`). |
| weight | the partition weights | no new kernel: `Σ_g (∂w_g/∂X_{A,d}) e(r_g)`, e.g. `xck_<family>_r_o0` with `w := dw/dX` (total density, Libxc `zk`). The host owns the weight derivative. |

Summed over atoms, the three classes cancel for each direction. The
density matrices must be symmetric, except for `cmgga_tau`, whose
gradient holds for complex orbitals in a real basis. Its density matrix
M is general (symmetric part Re P, antisymmetric part Im P, which carries
the paramagnetic current), and its `g1` rows take `DTchi = Mᵀ chi` next
to `Dchi = M chi`; its `gg` reads the current towers `jpx`, `jpx_y`, ….

### London orbitals

The explicit field derivative of the XC Fock matrix with London (GIAO)
orbitals, `xck_<family>_{r,ua,ub}_giao` for `lda`, `gga`, `mgga_tau`,
`mgga_lapl` and `mgga`, takes the collocation tower `chi`, the centers
of the basis functions `bf_centers (3, nbf)` and the grid coordinates
`rgx`, `rgy`, `rgz` (same origin). The kernel forms the center-scaled
operands `R_a chi`, `R_a ∂_c chi` and `R_a ∇²chi` itself. Output
`(3, nbf, nbf)`: the antisymmetric `K^s` with `dF/dB_s = (i/2c) K^s` at
a real reference, for s = x, y, z.

### Nuclear derivative of the Fock matrix

For CPHF right-hand sides and nuclear Hessians, `dF/dX_{A,d}` splits into
the same three classes, each computed one atom at a time with output
`(3, nbf, nbf)`:

| class | entry | how |
|---|---|---|
| basis | `xck_<family>_{r,ua,ub}_f1` | pass `chi`, `Dchi = D chi` (`Dchi_a` and `Dchi_b` for `ua`/`ub`, since dFᵅ/dX also responds to the β density) and `atom_mask`, an `int8[nbf]` flagging A's functions. The kernel forms the perturbed fields of the displacement itself. |
| grid | `xck_<family>_{r,ua,ub}_fg` | pass the weights of A's points, `w := w·Mᴬ` (zero elsewhere). |
| weight | `xck_<family>_<spin>_o1` | called with `w := dw/dX`. |

Summed over atoms, the three classes vanish. These entries exist for
all seven families and need the response order (`max_order >= 2`). The
density matrices are symmetric, except for `cmgga_tau` with complex
orbitals: its `f1` kernels also take the Mᵀ-contracted towers `DTchi`
(`DTchi_a`, `DTchi_b`).

### Nuclear Hessian

The explicit second derivative of the XC energy at fixed density, with
the full quadrature response:

    d²E/dX_{A,d} dY_{B,e} = BB + BG + GB + GG
                            + Σ_g [ wᴬᴮ e + wᴬ εᴮ + wᴮ εᴬ ]

εᴮ(g) is the derivative of the energy density at point g for atom B, and
wᴬ, wᴬᴮ are the first and second weight derivatives, which the host
supplies. All seven families, restricted and unrestricted (call `ua`
and `ub` and add); generated with `max_order >= 2`.

| class | entry | how |
|---|---|---|
| BB (basis-basis) | `xck_<family>_{r,ua,ub}_h2bb` | one call per atom B: `chi`, `Dchi` (`Dchi_a`, `Dchi_b`), the row channel's `D (nbf, nbf)` and B's `atom_mask`; output rows `(3, 3, nbf)`, summed by the host over the functions of each atom A. The kernel forms `D (mask∘∂χ)` and B's perturbed fields itself. |
| BG (basis A, grid B) | `xck_<family>_{r,ua,ub}_h2bg` | `w := w·Mᴮ`; rows `(3, 3, nbf)` summed over A's functions. GB is its transpose. |
| GG (grid-grid) | `xck_<family>_{r,u}_h2gg` | per point `(3, 3, ng)`; its (A, A) block is the sum over A's points. |
| εᴮ | `xck_<family>_{r,ua,ub}_e1p` + `gg` | `e1p` (B's `atom_mask`, `(3, ng)`) is the basis part; add the `gg` output with `w = 1` at B's points. |

The CPHF part of an analytic Hessian comes from the `f1`/`fg` and `o2`
kernels. The manifest's `hessian_classes` entry restates the assembly.

### Batched linear response

`xck_<family>_{r,ua,ub}_o2_batch` and `xck_<family>_st_o2_{p,m}_batch`
apply the linear response to `nx` perturbations at one ground state, with
output `(nx, nbf, nbf)`. The perturbed operands (`*_p1*`, listed in the
manifest's `batched_operands`) point to `(nx, npts)` arrays, the rest are
shared.

### MO-projected linear response (sigma vectors)

`xck_<family>_st_o2_{p,m}_mo` and `xck_<family>_{ua,ub}_o2_mo` compute the
occupied × virtual block of the response directly, for Casida/TDDFT,
stability analysis and orbital-rotation solvers:

    sigma_x[i,a] = (C_occᵀ F¹[P_x] C_vir)[i,a],   P_x = C_occ X_x C_virᵀ + transpose

for `nx` trial vectors `X (nx, nocc, nvir)` at one ground state. Instead of
`chi`, the kernel takes the MO collocation towers `phi_o = C_occᵀ·chi` and
`phi_v = C_virᵀ·chi` `(ncomp, nocc|nvir, npts)`, which the host forms once
per ground state. It then forms each trial vector's perturbed fields in MO
space itself: one `Z = X·phi_v` GEMM per tower component, then sums over the
occupied orbitals. The output is one rectangular GEMM per pattern group.
Neither direction touches an AO matrix, so the cost per trial vector scales
with nocc·nvir, not nbf². The spin-adapted kernels take the α amplitudes;
the β perturbation is the parity times them. The unrestricted kernels take
both channels' towers and amplitudes (`phi_o_a`, …, `X_a`, `X_b`) and
return the σ of their own channel.

These kernels cover real response only (symmetric P), the case for
real-orbital TDDFT and stability analysis. A purely imaginary (magnetic)
perturbation of `cmgga_tau`, with an antisymmetric P, still goes through
the AO `o2` kernels.

## Discretizations: molecular, periodic, curvilinear

Nothing in the derivative tower assumes Gaussians, or even an atom-centered
expansion. The generated kernels consume collocation data — basis-function
values and their derivatives on whatever grid the host uses — together with
the host's density matrices, and return matrices in the same representation.
Everything specific to the discretization stays on the host side of that
interface.

* **Plane waves and Bloch states.** Expressions are emitted for complex
  basis functions and complex coefficients (sesquilinear emission), so they
  apply as-is to periodic calculations. Demonstrated in GPAW
  ([gpaw!3425](https://gitlab.com/gpaw/gpaw/-/merge_requests/3425)):
  gradient-corrected kernels for the periodic dielectric function,
  represented exactly in the plane-wave basis as
  `K_GG' = Σ_ab fac_a(q+G)* FT[c_ab](G−G') fac_b(q+G')`, which lifts the
  `fxc(G−G')` locality restriction that leaves gradient-corrected kernels
  unrepresentable in the usual local form. Real-space grids and PAW use the
  same kernels through the same emitter; augmentation-sphere corrections
  are host-side.
* **Cell deformation (strain).** `strain.py` carries the master
  transformation law for every ingredient, yielding the XC stress tensor and
  its higher strain derivatives — the periodic counterpart of the nuclear
  derivatives in `geometric.py`, which include the full quadrature-grid
  response.
* **Curvilinear coordinates.** `inputs/basis.py` declares orthogonal
  coordinate systems through their Lamé scale factors `h_i`. Written in
  physical (orthonormal) components `g_i = h_i⁻¹ ∂_i n`, every generated
  expression carries over unchanged: the metric enters *only* through how an
  ingredient is built from the basis functions, so supporting a new
  coordinate system amounts to declaring its scale factors. Spherical and
  prolate spheroidal coordinates are supported, as are the two reductions in
  which an angular coordinate is integrated out analytically, leaving a
  residual operator on a block index of the density matrix
  (`l(l+1)·n_l/r²` for the spherically averaged atom, `m²·n_m/h_φ²` for the
  cylindrically symmetric diatomic) — the four geometries of HelFEM. The
  density Laplacian is the sole ingredient that does not survive the change
  of coordinates: it becomes the Laplace–Beltrami operator, bringing in
  derivatives of the scale factors, and is therefore refused in curvilinear
  coordinates.

## Adding a new ingredient or a new property

Both axes of extension are declarations, not derivations.

**A new ingredient** is defined *once*, through its value and its
density-matrix seed (`inputs/ingredients.py`). The entire tower of matrix
elements and response contractions for every functional built on it then
follows mechanically, to any order. This is how the current density `j_p`,
the gauge-corrected `τ̃ = τ − j_p²/2ρ` (which converts any standard mGGA in
Libxc into a current-corrected functional), and the density-Hessian
calibration variable `η = ∇n·(∇∇ᵀn)·∇n` of local-hybrid functionals were
added — the last is *cubic* in the density matrix, and to our knowledge its
Fock contribution has exactly one published account, stated without
derivation.

**A new property** is a new seed for the same `D` operator:

| property | seed | module |
|---|---|---|
| response to any order | perturbed field `k^X(r)` | `response.py` |
| nuclear derivatives (+ grid response) | displaced basis functions | `geometric.py` |
| cell deformation / XC stress | `r → A r` master law | `strain.py` |
| magnetic field, London (GIAO) orbitals | London phase factor | `london.py` |
| noncollinear / 4-component relativistic | locally collinear map | `noncollinear.py` |

Adding a **host program** is likewise a plugin: an emitter in `emitters/`
writes that program's own contraction idiom, so an integration is generated
files plus `#include`, rather than a merge into hand-written kernel code.
Regenerating is then an overwrite, not a re-derivation.

## The compiled library

The repository ships only the generator and its tests; the compiled
C/C++ library (C ABI + Fortran module, static coefficient tables walked
by a fixed evaluator) is a **generated artifact**. `clib/CMakeLists.txt`
generates and builds it in one go, with the kernel selection as
configure flags:

```sh
cmake -S clib -B build -DXCKERNEL_FAMILIES=lda,gga,mgga_tau -DXCKERNEL_MAX_ORDER=3
cmake --build build
```

### The interface: derivative towers

Every spatially varying operand is a component of a Cartesian derivative
tower, named `<base>[_<spin>][_<pert>][_<axes>]` with the derivative axes
sorted and empty for the value:

| name | meaning |
|---|---|
| `chi` | collocation, one array `chi[k][u][g]` |
| `Dchi` | `D chi`, for the gradient rows (and `DTchi = Dᵀ chi` for a general density matrix) |
| `rho_x`, `rho_xy`, `rho_xyz` | derivatives of the density |
| `rho_a_p1_xx` | ∂²/∂x² of the α perturbed density of perturbation 1 |
| `tau_p1`, `tau_x` | τ fields |
| `jpx_a` | x component of the α paramagnetic current (a vector's component is part of its base, so a trailing axis string always means a derivative) |

The tower components `k` run in the order 1, x, y, z, xx, xy, xz, yy,
yz, zz, xxx, xxy, … , that of PySCF's `eval_ao(deriv=n)`. Each kernel
exports the order it reads (`<name>_chi_order`) and its ordered
per-point operands (`<name>_scal_names`); `manifest.json` lists both.

Laplacians are never operands. The kernel forms `∇²χ`, `∂_d∇²χ`,
`∇²ρ¹` and the like from the tower components, the basis-level ones once
per grid block. Each formed operand takes the place of the single
host-supplied array it replaces, so the number of GEMMs is unchanged.
`manifest.json` records every such definition under `formed_in_kernel`.

Every kernel also exports its ABI kind (`<name>_kind`: `"matrix"`,
`"diag"`, `"g1"`, `"f1u"`, `"h2bb"`, ...), output rank and shape
(`<name>_out_rank`, `<name>_out_shape`), and `xckernel.h` declares an
index of the build, `xckernel_kernels[]`. Each entry holds:
- `name`, `kind`, `order`, `out_rank` and `out_shape`;
- the entry point `fn`, to be cast to the signature of its kind;
- `scal_names`, `n_scal` and `n_fields`;
- the tower orders `chi_order`, `dchi_order` and `dtchi_order`, which
  are −1 for a tower the kind doesn't take.

A host can therefore dispatch from the exported metadata alone, with no
table of its own that could go stale against the kernels it vendors.

**Header-only use.** Each kernel header
`include/xckernel/kernels/<name>.hpp` also works without the compiled
library, at any floating-point type. It declares the template
`xckernel::<name>_t<T, Txc>`, which takes the per-point fields (type `T`)
and the Libxc derivative arrays (type `Txc`) as separate arrays.

The header also states their order. `<name>_field_names[]` and
`<name>_xc_names[]` give the two lists, and `<name>_n_fields` and
`<name>_n_xc` their lengths; all four are `constexpr` in namespace
`xckernel`. The C ABI's `<name>_scal_names` is initialised from these
two lists, so the two can't drift apart.

The operands are positional arrays of identically typed pointers, so
passing them in the wrong order gives wrong numbers with no error. The
header therefore also declares the structs `<name>_fields<T>` and
`<name>_xc<Txc>`, with one named member per operand, and an overload of
`<name>_t` that takes them:

```cpp
xckernel::xck_mgga_r_o1_fields<double> f{};
f.w = w; f.rho_x = gx; f.rho_y = gy; f.rho_z = gz;
xckernel::xck_mgga_r_o1_xc<double> x{};
x.vrho = vrho; x.vsigma = vsigma; x.vlapl = vlapl; x.vtau = vtau;
xckernel::xck_mgga_r_o1_t(npts, nbf, chi, f, x, F);
```

A misspelt or missing member is then a compile error. A host can also
`static_assert` its own packing against the `constexpr` names.

The order itself is fixed: fields in tower order, and the Libxc arrays
grouped by derivative order, alphabetical within each group. Changing it
would silently break every host that packs operands positionally, so any
change will come with a version bump.

Generation takes a small fraction of the time needed to compile the
emitted code. To produce a self-contained source tree for distribution
(no Python required downstream), run the generator directly:

```sh
python3 -m xckernel.catalog libxckernel "lda,gga,mgga_tau,mgga_lapl,mgga,cmgga_tau,hmgga" 4 c
```

A host that needs only some of the kernel kinds selects them with
`--kinds`, which takes the exported kind names (default: `all`):

```sh
python3 -m xckernel.catalog libxckernel lda,gga,mgga_tau,mgga_lapl,mgga,cmgga_tau 2 c \
    --kinds exc,matrix,diag,o2b,mo2,mo2u,g1,g1c,gg
```

Unselected kinds are not generated at all. Everything in the package
follows the selection: the kernel sources, `xckernel.h`, the index, the
CMake and Fortran files, and `manifest.json`, whose `kinds` field records
it. `evaluator.hpp` carries only the helpers the selected kernels call,
and `xckernel.h` records the selection as `XCKERNEL_KINDS`. For the
selection above, generation drops from about 7 min to 9 s and the package
from 19 MB to 4 MB. `catalog_c_validate --kinds …` validates the same
selection, skipping the check groups that need an absent kind.

## What is validated

All checks live in `xckernel/tests/` and compare against PySCF (machine
precision, `~1e-13`–`1e-17`) where PySCF implements the quantity, and against
(Richardson-extrapolated) finite differences where it does not.

| quantity | families | spin | reference | agreement |
|---|---|---|---|---|
| XC Fock `F_uv` | LDA/GGA/mGGA(τ,∇²ρ) | R + U | PySCF `nr_rks`/`nr_uks`; FD | ~1e-15 |
| AO kernel `g_uv,ts` | all four | R + U | PySCF `nr_*_fxc`; FD | ~1e-15 |
| fxc contraction (order 2) | LDA/GGA/mGGA(τ) | R + U + singlet/triplet | PySCF `nr_*_fxc`, `nr_rks_fxc_st` | ~1e-13 |
| kxc contraction (order 3) | LDA/GGA/mGGA(τ) | R + U | FD of Fock | ~1e-5 |
| lxc contraction (order 4) | LDA/GGA | R | FD of Exc | ~1e-5 |
| orbital gradient / Hessian | all four | R | FD under exp(κ) | ~1e-7 |
| TDA σ-vector | LDA/GGA | R | PySCF `TDA.gen_vind` | ~1e-17 |
| RPA supervector σ | LDA/GGA | R | PySCF `gen_tdhf_operation` | ~1e-17 |
| quadratic-response σ (E[3]) | LDA/GGA | R | FD of Exc, both κ signs | ~1e-6 |
| cubic-response σ (E[4]) | LDA/GGA | R | FD of Exc, both κ signs | ~1e-5 |
| geometric gradient + grid response | LDA/GGA/mGGA | R + U | FD of Exc | ~1e-10 |
| C kernels on the tower interface: `o1`, `o2` | all seven; `cmgga_tau` with complex orbitals | R + U | FD of Exc in D; FD of `o1` along D¹ | ~1e-12 |
| C gradient kernels `g1` + `gg` + weight class | all seven; `cmgga_tau` with complex orbitals | R + U | Richardson FD of Exc per class; translational sum rule | ~1e-10; ~1e-16 |
| C nuclear Hessian (all classes, weight terms included) | all seven | R + U | FD of the matching gradient class; FD of the full gradient; translational sum rule | ~1e-11; ~1e-15 |
| C batched response `o2_batch` | all seven | R + U + spin-adapted | nx single `o2` calls | ~1e-13 |
| C MO-projected response `o2_mo` | all seven | U + spin-adapted | C_occᵀ·`o2`[fields of P_x]·C_vir | ~1e-15 |
| C Fock derivative `f1` + `fg` + weight class | all seven; `cmgga_tau` with complex orbitals | R + U | FD of `o1` per class; complete move; translational sum rule | ~1e-10; ~1e-16 |
| C Fock diagonal `o1_diag` | all seven | R + U | diagonal of the `o1` kernel | exact |
| C London-orbital kernels `giao` | LDA/GGA/mGGA(τ,∇²ρ) | R + U | NumPy GIAO kernels (FD-validated in `london_validate`); antisymmetry | ~1e-15 |
| geometric Hessian + grid response | LDA/GGA/mGGA | R + U | FD of gradients | ~1e-9 |
| GauXC Hessian assembly recipe | LDA/GGA/mGGA(tau) | R | contracted `geometric_hessian` | ~1e-16 |
| GauXC Hessian emitted C++ | LDA/GGA/mGGA(tau) | R | SymPy, same operands | exact |
| XC stress (cell strain), 9 components | LDA–mGGA + η | R | Richardson FD of E(A) | ~1e-9 |
| XC stress vs a production implementation | LDA/GGA/mGGA(τ) | R | GPAW's own `_stress` on its arrays | ~1e-15 |
| complex orbitals/basis (sesquilinear) | LDA–mGGA | R | FD in complex P | machine ε |
| split-storage Re/Im parts | LDA–mGGA | R | complex path | machine ε |
| noncollinear/relativistic (4C) vxc + fxc | LDA/GGA | locally collinear | FD in 4C spinor DM | ~1e-8 |
| two-sided (matrix-free) σ | LDA–mGGA | R | AO-route kernels | machine ε |
| current-density (τ̃, jp seeds) | cmgga_tau | R + U + s/t | FD in general M | ~1e-12 |
| density-Hessian (η) | hmgga | R + U + s/t | FD in general M | ~1e-11 |
| curvilinear metric (spherical/prolate, l/m reductions) | all | n/a | Cartesian gradient norm | ~1e-6 |
| curvilinear vxc + fxc (blocked DM) | LDA/GGA/mGGA(τ) | R + U | FD of Exc; FD of vxc | ~1e-8 |
| kxc energy trilinear (grid form) | LDA/GGA | R + U | Richardson triple-cross FD of Exc | ~1e-8 |
| C backend | all | R | NumPy backend | ~1e-16 |
| Fortran backend | LDA/GGA | R + U, real + complex | SymPy reference, through gfortran | ~1e-16 |

Conventions (the κ exponential sign, occupation/factor placement,
singlet/triplet parities, Libxc component packing) are explicit parameters or
documented constants throughout — the six-code survey shows silent convention
assumptions are where cross-code reuse historically dies.

## Quick example

```python
import xckernel as xk

# symbolic integrand of the GGA XC Fock matrix element
fi = xk.fock_integrand("gga")
print(fi.expr)   # chi_u*chi_v*vrho*w + 2*chi_u*dchi_v_x*grad_rho_x*vsigma*w + ...

# generated NumPy source for the linear-response (fxc) contraction,
# batched over perturbed density matrices
gen = xk.generate(xk.response_fock("gga", order=2), "fxc_contract", batch=True)
print(gen.source)          # np.einsum contractions, one AO matrix per DM
fn = xk.compile_function(gen)   # live callable
```

The generated functions take grid collocation data (`chi`, `dchi`, weights),
ground-state and perturbed fields, and the named Libxc derivative arrays that
`pylibxc` returns — see `xckernel/tests/pyscf_demo.py` and
`xckernel/tests/tda_validate.py` for complete wirings into PySCF.

## Layout

```
xckernel/
  inputs/          what goes in: the symbolic input definitions
    basis.py         symbolic basis-function fields (chi, grad, lapl, hess chi)
                     + the orthogonal curvilinear coordinate systems
    ingredients.py   rho, grad rho, lapl rho, tau, jp, hess rho, eta + seeds
    functional.py    families as ingredient sets
    fields.py        numerical collocation helpers (incl. complex P)
  engine/          the derivative tower over the input definitions
    deriv.py         the D operator and the Libxc derivative-name registry
    fastpoly.py      monomial representation; all derivatives applied here
    fock.py          F_uv integrand
    kernel.py        repeated-D kernels (g_uv,ts and higher)
    response.py      contraction engine (perturbed-field seeds), any order
    spin.py          spin-resolved ingredients, seeds, component packing
    spin_kernel.py   open-shell tower, singlet/triplet parities
    geometric.py     nuclear derivatives incl. quadrature-grid response
    gradient.py      the XC nuclear gradient as C-catalog rows/points
    geofock.py       nuclear derivatives of the Fock matrix (dF/dX)
    hessian.py       the explicit XC nuclear Hessian classes
    strain.py        cell-deformation (strain) seeds from the master law
    london.py        explicit magnetic-field derivatives (London orbitals)
    noncollinear.py  locally collinear map: noncollinear/relativistic
                     potential and fxc (4C/2C response kernels)
    mo.py            AO->MO helpers: orbital gradient (kappa sign!) and Hessian
    algebra.py       response algebra: DM builders, projections, sigma templates
  emitters/        what comes out: shared IR + code generators
    codegen.py       pattern collapse + NumPy emission (batched; spin;
                     sesquilinear; two-sided)
    compact.py       IR compaction to the minimal contraction form
    cbackend.py      C library emitter (static tables + fixed evaluator)
    psi4backend.py   Psi4 host-idiom emitter (xcgen include files)
    vlxwriter.py     VeloxChem writer plugin
    gpawwriter.py    GPAW pair-coefficient/stress-field emitter
    helfemwriter.py  HelFEM emitter: potential + kernel channels,
                     curvilinear (radial/spherical/prolate) coordinates,
                     one/two/three gradient components, spin-resolved
    ncwriter.py      noncollinear (relativistic) kernel emitter
    gauxcwriter.py   GauXC emitter: fixed-grid nuclear-Hessian kernels
    octopuswriter.py Octopus Fortran emitter: third-derivative trilinears
    release.py       self-contained C source package assembly
    tower.py         the derivative-tower interface of the C kernels
  catalog.py       the 407-kernel catalog + machine-readable manifests
  runtime.py       compiled-library loader
  tests/           validation suites (see table above)
docs/
  dedup-analysis.md  six-code survey of DFT response stacks and the design
```

## Requirements

`sympy`, `numpy`; `pylibxc` for evaluating anything numerically. The test
suites additionally use `pyscf` (reference values) and `scipy` (`expm`).
Note that `pylibxc` is not installable from PyPI (the `pylibxc2` name
there is an unrelated empty stub); it ships with Libxc itself, e.g. as
the conda-forge `libxc` package or the Fedora `python3-libxc` RPM.

## License

BSD 3-Clause (see `LICENSE`).

## Status and roadmap

Working and validated: everything in the tables above, including the C and
Psi4 emitter backends, complex orbitals, the matrix-free two-sided mode, and
the geometric derivatives with quadrature-grid response. The Psi4
integration (meta-GGA TDDFT/CPKS/stability, GGA and meta-GGA nuclear
Hessians, grid response) is available as
[psi4/psi4#3458](https://github.com/psi4/psi4/pull/3458); its generated
regions are standalone include files, regenerated wholesale with
`python -m xckernel.emitters.psi4backend --emit-dir <psi4>/psi4/src/psi4/libfock/xcgen`.
The GPAW integration (meta-GGA and triplet Casida couplings,
gradient-corrected kernels for the periodic dielectric function, and the XC
stress) is available as
[gpaw!3425](https://gitlab.com/gpaw/gpaw/-/merge_requests/3425), regenerated
with `python -m xckernel.emitters.gpawwriter --emit <gpaw>/gpaw/xckernel_fxc.py`.
A manuscript describing the library is in preparation.

Not yet done: the exact-exchange energy density e_x(r) as a primitive
ingredient (local hybrids; the density-Hessian calibration variable η is
already in); the noncollinear tau channel (the lda/gga noncollinear and
four-component relativistic kernels are in, via the locally collinear
map; the tau channel awaits a noncollinear KED definition, and native
noncollinear functionals await the next major Libxc release); matrix-form
(commutator-algebra) lowering of
the response σ assembly; active-space (MCSCF-type) gradient projector;
spin-basis (`ud2ts`) transformation maps; rank-1 (occupation × orbital)
density-matrix form. Hybrid/range-separated exchange remains host-owned, and
the manifests spell out that term-ownership boundary explicitly.
