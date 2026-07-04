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
grid weights, and Libxc derivative arrays, which is emitted as ready-to-run
NumPy code (other codegen targets are planned).

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
  basis.py         symbolic basis-function fields (chi, grad chi, lapl chi)
  ingredients.py   rho, grad rho, lapl rho, tau as bilinear forms in P + seeds
  functional.py    Libxc first-derivative symbols
  deriv.py         the D operator and the Libxc derivative-name registry
  fock.py          F_uv integrand
  kernel.py        repeated-D kernels (g_uv,ts and higher)
  response.py      contraction engine (perturbed-field seeds), any order
  spin.py          spin-resolved ingredients and seeds
  spin_kernel.py   open-shell tower, Libxc component packing, singlet/triplet
  codegen.py       einsum code generation (batched; spin; perturbed fields)
  mo.py            AO->MO helpers: orbital gradient (kappa sign!) and Hessian
  algebra.py       response algebra: DM builders, projections, sigma templates
  tests/           validation suites (see table above)
docs/
  dedup-analysis.md  six-code survey of DFT response stacks and the design
```

## Requirements

`sympy`, `numpy`; `pylibxc` for evaluating anything numerically. The test
suites additionally use `pyscf` (reference values) and `scipy` (`expm`).

## License

BSD 3-Clause (see `LICENSE`).

## Status and roadmap

Working and validated: everything in the table. Not yet done: matrix-form
(commutator-algebra) lowering of the response σ assembly for production
efficiency; active-space (MCSCF-type) gradient projector; spin-basis
(`ud2ts`) transformation maps; rank-1 (occupation × orbital) density-matrix
form; C/Fortran codegen targets; hybrid/range-separated bookkeeping (the
exchange part is the host's, but the survey's term-ownership lesson says the
boundary must be spelled out).
