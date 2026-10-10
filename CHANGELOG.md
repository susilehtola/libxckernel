# Changelog

Releases of the generated C catalog (`xckernel.h`, `libxckernel`). Since
0.2.0 every kernel entry point keeps its C ABI v2 interface; releases add
kernels and ABI kinds. Each kernel exports its kind and output shape
(since 0.4.0), so a host can check what it links against. The one
layout change outside the entry points is the index struct
`xck_kernel_info` in 0.5.0, which a host compiled against the header of
the same generated tree picks up automatically.

## Unreleased

- **Imaginary (magnetic) MO-projected response** for `cmgga_tau`:
  `xck_cmgga_tau_st_o2_{p,m}_mo_imag` (kind `mo2i`) and
  `xck_cmgga_tau_{ua,ub}_o2_mo_imag` (kind `mo2iu`).
  - Computes the σ vectors for P_x = C_occ X_x C_virᵀ − C_vir X_xᵀ C_occᵀ,
    the antisymmetric part of the general density matrix.
  - The arguments are the same as for `mo2`/`mo2u`.
  - Only the paramagnetic-current terms remain.
- The MO-projected kernels now drop perturbed fields that vanish
  identically, such as jₚ¹ of a real perturbation in `cmgga_tau`.

## 0.5.0 (2026-10-06)

Generate only the kernel kinds a host uses, dispatch from the index,
and read operand names from the kernel headers.

- **Kind selection:** `python -m xckernel.catalog <out> <families>
  <max_order> c --kinds k1,k2,...` takes the exported ABI kind names
  (`exc`, `matrix`, `diag`, `o2b`, `mo2`, `mo2u`, `g1`, `g1c`, `gg`,
  `f1`, …, `giao`, `h2bb`, …, `e1p`, …). The default is `all`.
  - Unselected kinds are never generated.
  - The sources, `xckernel.h`, the index, `CMakeLists.txt`, the Fortran
    module and `manifest.json` follow the selection.
  - The manifest records the selection as `"kinds"` and the header as
    `XCKERNEL_KINDS`.
  - `evaluator.hpp` carries only the helpers the selected kernels call.
  - For six families at order 2 with
    `exc,matrix,diag,o2b,mo2,mo2u,g1,g1c,gg`: 156 kernels, about 4 MB,
    generated in 9 s, against 273 kernels, 19 MB and about 7 min for
    all kinds.
- **Dispatch metadata in the index:** `xck_kernel_info` gains:
  - `fn`, the entry point as `void (*)(void)`, cast according to `kind`;
  - `scal_names`, `n_scal` and `n_fields`;
  - `chi_order`, `dchi_order` and `dtchi_order`, which are −1 for a
    tower the kind doesn't take. `chi_order` gives the order of
    `phi_o`/`phi_v` for the MO kinds.

  This changes the struct layout. Kernel entry points are unchanged.
- **Operand names in the kernel headers:** for header-only use, each
  kernel header declares, in namespace `xckernel`:
  - the `constexpr` lists `<name>_field_names[]` and `<name>_xc_names[]`,
    split as the template `<name>_t` takes them, and their lengths
    `<name>_n_fields` and `<name>_n_xc`;
  - the structs `<name>_fields<T>` and `<name>_xc<Txc>`, with one named
    member per operand;
  - an overload of `<name>_t` that takes the structs, so a misnamed
    operand is a compile error.

  The C ABI's `scal_names` is initialised from the header's lists. The
  operand order is unchanged, and this is additive.
- **Validation:** `catalog_c_validate --kinds …` validates a selection
  and skips the check groups that need an unselected kind.

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
