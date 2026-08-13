# libxckernel

> The project (and the generated C library) is **libxckernel**;
> the Python generator package imports as `xckernel`
> (as `pylibxc` is to `libxc`).

**An automatic-differentiation backend for [Libxc](https://libxc.gitlab.io/):
generate arbitrary exchange–correlation kernel elements in LCAO basis sets.**

Density-functional response properties all reduce to derivatives of the XC
energy with respect to the density matrix, contracted with grid data and the
functional derivatives that Libxc provides. Every quantum chemistry code
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
[psi4/psi4#3458](https://github.com/psi4/psi4/pull/3458)).

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

`catalog.py` enumerates, generates, and manifests **156 kernels** named
`xck_<family>_<case>_o<order>[_<parities>]`, spanning seven functional
families — `lda`, `gga`, `mgga_tau` (τ-only), `mgga_lapl` (Laplacian-only),
`mgga` (full), `cmgga_tau` (current-density: the Libxc τ slot is fed the
gauge-corrected τ̃ = τ − j²ₚ/2ρ), and `hmgga` (density-Hessian η of
local-hybrid calibration functions) — in the restricted, unrestricted, and
closed-shell spin-adapted cases (singlet/triplet parity per perturbation)
through fourth derivative order (third for the spin-resolved `cmgga_tau` and
for `hmgga`, whose higher orders remain generatable on demand). Fifteen `xck_<family>_{r,ua,ub}_giao` kernels provide the explicit magnetic-field
derivatives of the Fock matrix with London (GIAO) orbitals, as the real
factor of dF/dB_s = (i/2c) K_s at a real reference. Every kernel
ships with a machine-readable manifest declaring its operands and shapes,
the Libxc arrays it consumes by name, and its term ownership. Beyond the
catalog: complex orbitals and complex basis functions (sesquilinear
emission), a matrix-free two-sided mode that emits σ-vector contractions
from MO-pair collocation, and nuclear derivatives of the XC contribution
including the full quadrature-grid response (`geometric.py`).

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

Generation takes a small fraction of the time needed to compile the
emitted code. To produce a self-contained source tree for distribution
(no Python required downstream), run the generator directly:

```sh
python3 -m xckernel.catalog libxckernel "lda,gga,mgga_tau,mgga_lapl,mgga,cmgga_tau,hmgga" 4 c
```

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
| geometric Hessian + grid response | LDA/GGA/mGGA | R + U | FD of gradients | ~1e-9 |
| complex orbitals/basis (sesquilinear) | LDA–mGGA | R | FD in complex P | machine ε |
| two-sided (matrix-free) σ | LDA–mGGA | R | AO-route kernels | machine ε |
| current-density (τ̃, jp seeds) | cmgga_tau | R + U + s/t | FD in general M | ~1e-12 |
| density-Hessian (η) | hmgga | R + U + s/t | FD in general M | ~1e-11 |
| C backend | all | R | NumPy backend | ~1e-16 |

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
`tests/tda_validate.py` for complete wirings into PySCF.

## Layout

```
xckernel/
  basis.py         symbolic basis-function fields (chi, grad, lapl, hess chi)
  ingredients.py   rho, grad rho, lapl rho, tau, jp, hess rho, eta + seeds
  functional.py    families as ingredient sets
  deriv.py         the D operator and the Libxc derivative-name registry
  fastpoly.py      monomial representation; all derivatives applied here
  fock.py          F_uv integrand
  kernel.py        repeated-D kernels (g_uv,ts and higher)
  response.py      contraction engine (perturbed-field seeds), any order
  spin.py          spin-resolved ingredients, seeds, component packing
  spin_kernel.py   open-shell tower, singlet/triplet parities
  geometric.py     nuclear derivatives incl. quadrature-grid response
  codegen.py       pattern collapse + NumPy emission (batched; spin;
                   sesquilinear; two-sided)
  cbackend.py      C library emitter (static tables + fixed evaluator)
  psi4backend.py   Psi4 host-idiom emitter (marked source regions)
  catalog.py       the 156-kernel catalog + machine-readable manifests
  fields.py        numerical collocation helpers (incl. complex P)
  runtime.py       compiled-library loader
  mo.py            AO->MO helpers: orbital gradient (kappa sign!) and Hessian
  algebra.py       response algebra: DM builders, projections, sigma templates
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
[psi4/psi4#3458](https://github.com/psi4/psi4/pull/3458). A manuscript
describing the library is in preparation.

Not yet done: the exact-exchange energy density e_x(r) as a primitive
ingredient (local hybrids; the density-Hessian calibration variable η is
already in); noncollinear spin; matrix-form (commutator-algebra) lowering of
the response σ assembly; active-space (MCSCF-type) gradient projector;
spin-basis (`ud2ts`) transformation maps; rank-1 (occupation × orbital)
density-matrix form. Hybrid/range-separated exchange remains host-owned, and
the manifests spell out that term-ownership boundary explicitly.
