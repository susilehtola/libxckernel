# Changelog

Releases of the generated C catalog (`xckernel.h`, `libxckernel`). Every
release since 0.2.0 is additive to the C ABI v2: existing entries keep
their interface, and new ones add new kernels and ABI kinds. Each kernel
exports its kind and output shape (since 0.4.0), so a host can check what
it links against.

## 0.4.0 (2026-10-05)

New kernels for magnetic properties, faster response, and the nuclear
Hessian.

- **London (GIAO) orbitals:** `xck_<family>_{r,ua,ub}_giao` for `lda`,
  `gga`, `mgga_tau`, `mgga_lapl` and `mgga`, until now NumPy-only.
  - Output: K^s `(3, nbf, nbf)`, with dF/dB_s = (i/2c) K^s at a real
    reference.
  - ABI kind `giao`: `(npts, nbf, chi, bf_centers, scal, out)`.
  - The kernel forms the center-scaled operands from `bf_centers`
    `(3, nbf)`. The grid coordinates are the per-point operands `rgx`,
    `rgy` and `rgz`.
- **Kind and shape exports:** each kernel exports `<name>_kind`,
  `<name>_out_rank` and `<name>_out_shape`. The index `xckernel_kernels[]`
  (`xck_kernel_info`, with `xckernel_n_kernels` entries) lists every built
  kernel. `runtime.Library` reads the kind from the library.
- **Batched linear response:** `xck_<family>_{r,ua,ub}_o2_batch` and
  `xck_<family>_st_o2_{p,m}_batch`.
  - ABI kind `o2b`.
  - Computes `nx` responses at one ground state, with an `int64_t nx`
    after `nbf` and output `(nx, nbf, nbf)`.
  - The perturbed operands (the manifest's `batched_operands`) point to
    `(nx, npts)` arrays.
  - Results are exact, but no faster than `nx` single `o2` calls.
- **Explicit XC nuclear Hessian with grid response:** all seven families,
  restricted and unrestricted (`cmgga_tau` with complex orbitals),
  generated when `max_order >= 2`. The `…c` kinds (`cmgga_tau`, general
  density matrix) also take `DTchi = Mᵀ chi`. The pieces are:

  | kernel | kinds | class | arguments | output |
  |---|---|---|---|---|
  | `h2bb` | `h2bb`, `h2bbu`, `h2bbc`, `h2bbcu` | basis–basis | `chi`, `Dchi` (per channel when unrestricted), the density matrix `D`, atom B's `atom_mask` | rows `(3, 3, nbf)` |
  | `h2bg` | `h2bg`, `h2bgc` | basis–grid, called with `w := w·Mᴮ` | `chi`, `Dchi` | rows `(3, 3, nbf)` |
  | `h2gg` | `h2gg` | grid–grid | per-point operands only | per point `(3, 3, npts)` |
  | `e1p` | `e1p`, `e1pc` | per-point basis part of εᴮ for the weight classes | `chi`, `Dchi`, `atom_mask` | `(3, npts)` |

  - The grid–basis class is the transpose of `h2bg`.
  - The host owns the weight classes, which need dw and d²w. The
    manifest's `hessian_classes` gives the assembly.
- **MO-projected linear response (σ vectors):**
  `xck_<family>_st_o2_{p,m}_mo` (kind `mo2`) and
  `xck_<family>_{ua,ub}_o2_mo` (kind `mo2u`), all seven families.
  - Computes σ_x = C_occᵀ F¹[P_x] C_vir for `nx` trial vectors
    `X (nx, nocc, nvir)`, with P_x = C_occ X_x C_virᵀ + transpose.
  - Takes the MO collocation towers `phi_o = C_occᵀ·chi` and
    `phi_v = C_virᵀ·chi` in place of `chi`.
  - Forms the perturbed fields in MO space, so the cost per trial vector
    scales with nocc·nvir rather than nbf².
  - Real (symmetric-P) response only.

## 0.3.0 (2026-10-01)

Additive to 0.2.0.

- **Nuclear gradient for all seven families:** `hmgga` added, and
  `cmgga_tau` for complex orbitals.
  - The gradient comes in three classes: `g1` rows, `gg` per point, and
    the weight class, which is `o0` with `w := dw/dX`.
  - For a general density matrix, `g1` also takes `DTchi = Mᵀ chi`.
- **dF/dX, one call per atom:** all seven families; `cmgga_tau` also takes
  `DTchi`. It comes in three classes:
  - `xck_<family>_{r,ua,ub}_f1`: the basis class. It takes an `int8` atom
    mask and forms the perturbed fields itself.
  - `_fg`: the grid class, called with `w := w Mᴬ`.
  - The weight class, which is `o1` with `w := dw/dX`.

## 0.2.0 (2026-10-01)

- **C ABI v2:** every compiled kernel takes Cartesian derivative towers.
  - Collocation is `chi[k][u][g]` in `eval_ao` order.
  - Per-point operands are named `<base>[_<spin>][_<pert>][_<axes>]`.
  - Laplacians are formed in the kernel.
- **Fock diagonal:** `xck_<family>_{r,ua,ub}_o1_diag`, all families.
- **XC nuclear gradient:** for `lda`, `gga`, `mgga_tau`, `mgga_lapl` and
  `mgga`.
  - Basis class: `xck_<family>_{r,ua,ub}_g1`.
  - Grid class: `xck_<family>_{r,u}_gg`.
  - Weight class: `o0` with `w := dw/dX`.
