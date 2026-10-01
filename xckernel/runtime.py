"""Runtime interface to a compiled libxckernel, with NumPy fallback.

The compiled library is self-describing: each kernel exports its ordered
per-point operand names (``<name>_scal_names``/``<name>_n_scal``) and the
derivative order it reads of each collocation tower
(``<name>_chi_order``), so the generic dispatcher below needs no
per-kernel binding code.

    from xckernel.runtime import Library
    lib = Library("/path/to/libxckernel.so")        # or $XCKERNEL_LIBRARY
    F = lib("xck_gga_r_o2", chi=chi, w=w, rho=rho_tower,
            rho_p1=rho_p1_tower, v2rho2=..., ...)

chi is the collocation tower (ncomp, nbf, ng), components in the order
1, x, y, z, xx, xy, ... (PySCF's eval_ao(deriv=n)); per-point operands
are named tower components (rho_x, rho_p1_xy, tau_p1, ...). A whole
field tower may be passed under its base name, e.g. ``rho=(ncomp, ng)``
supplies rho_x, rho_xy, ... at once.

``get_kernel(name)`` returns a compiled-library callable when a library is
discoverable, else falls back transparently to the NumPy backend (generated
on first use) -- the graceful-degradation contract of the interfacing plan.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from typing import Dict, List, Optional

import numpy as np

from .emitters.tower import components, ncomp

_P = ctypes.POINTER(ctypes.c_double)


def _operands(operands: Dict, ng: int) -> Dict[str, np.ndarray]:
    """Per-point operands by tower name; (ncomp, ng) towers are expanded
    to their components (base name for the value, base_<axes> beyond)."""
    scal: Dict[str, np.ndarray] = {}
    for key, val in operands.items():
        val = np.asarray(val, dtype=np.float64)
        if val.shape == (ng,):
            scal[key] = np.ascontiguousarray(val)
            continue
        n = next((k for k in range(8) if val.shape == (ncomp(k), ng)), None)
        if n is None:
            raise ValueError(f"operand {key!r}: expected ({ng},) or a "
                             f"(ncomp, {ng}) tower, got {val.shape}")
        for i, ax in enumerate(components(n)):
            scal[f"{key}_{ax}" if ax else key] = np.ascontiguousarray(val[i])
    return scal


def _kind(name: str) -> str:
    if name.endswith("_o1_diag"):
        return "diag"
    if name.endswith("_f1"):
        return "f1"
    if name.endswith("_fg"):
        return "fg"
    if name.endswith("_g1"):
        return "g1"
    if name.endswith("_gg"):
        return "gg"
    return "matrix"


class Library:
    """A loaded libxckernel with generic, self-describing dispatch."""

    def __init__(self, path: Optional[str] = None):
        if path is None:
            path = os.environ.get("XCKERNEL_LIBRARY") \
                or ctypes.util.find_library("xckernel")
        if path is None:
            raise OSError("no libxckernel found: pass a path or set "
                          "XCKERNEL_LIBRARY")
        self.path = path
        self._dll = ctypes.CDLL(path)
        self._scal_cache: Dict[str, List[str]] = {}

    def scal_names(self, name: str) -> List[str]:
        """The kernel's ordered scalar-operand names, read from the binary."""
        if name not in self._scal_cache:
            n = ctypes.c_int.in_dll(self._dll, f"{name}_n_scal").value
            arr = (ctypes.c_char_p * n).in_dll(self._dll,
                                               f"{name}_scal_names")
            self._scal_cache[name] = [s.decode() for s in arr]
        return self._scal_cache[name]

    def order(self, name: str, array: str = "chi") -> int:
        """Derivative order the kernel reads of a collocation tower."""
        return ctypes.c_int.in_dll(self._dll, f"{name}_{array}_order").value

    def _has(self, symbol: str) -> bool:
        try:
            ctypes.c_int.in_dll(self._dll, symbol)
            return True
        except ValueError:
            return False

    def __call__(self, name: str, *, w, chi=None, Dchi=None, DTchi=None,
                 Dchi_a=None, Dchi_b=None, atom_mask=None, out=None,
                 **operands) -> np.ndarray:
        """Call a kernel with named operands; returns ``out`` (accumulated
        into when given): (nbf, nbf), (nbf,) for *_o1_diag, (3, nbf) for
        *_g1, (3, ng) for *_gg, (3, nbf, nbf) for *_f1 and *_fg. Gradient
        rows of a general density matrix M take Dchi = M chi and
        DTchi = M^T chi; the dF/dX basis class takes Dchi (or Dchi_a and
        Dchi_b) and the atom mask (nbf,), one atom per call."""
        kind = _kind(name)
        ng = np.asarray(w).shape[0]
        scal = {"w": np.ascontiguousarray(w, dtype=np.float64),
                **_operands(operands, ng)}
        names = self.scal_names(name)
        missing = [n for n in names if n not in scal]
        if missing:
            raise TypeError(f"{name}: missing operands {missing}")
        ptrs = (_P * len(names))(*[scal[n].ctypes.data_as(_P)
                                   for n in names])
        args = [ctypes.c_int64(ng)]
        keep = []                       # the converted towers, alive for the call
        nbf = None
        towers = [] if kind == "gg" else ["chi"]
        if kind == "g1":
            towers.append("Dchi")
            if self._has(f"{name}_DTchi_order"):
                towers.append("DTchi")
        if kind == "f1":
            towers += (["Dchi_a", "Dchi_b"]
                       if self._has(f"{name}_Dchi_a_order") else ["Dchi"])
        given = {"chi": chi, "Dchi": Dchi, "DTchi": DTchi, "Dchi_a": Dchi_a,
                 "Dchi_b": Dchi_b}
        for arr in towers:
            val = given[arr]
            val = np.ascontiguousarray(val, dtype=np.float64)
            if val.ndim == 2:
                val = val[None]
            need = ncomp(self.order(name, arr))
            if val.shape[0] < need or val.shape[2] != ng:
                raise ValueError(f"{name}: {arr} must be a ({need}, nbf, {ng}) "
                                 f"tower, got {val.shape}")
            nbf = val.shape[1]
            if arr == "chi":
                args.append(ctypes.c_int64(nbf))
            keep.append(val)
            args.append(val.ctypes.data_as(_P))
        if kind == "f1":
            mask = np.ascontiguousarray(atom_mask, dtype=np.int8)
            if mask.shape != (nbf,):
                raise ValueError(f"{name}: atom_mask must be ({nbf},) int8")
            keep.append(mask)
            args.append(mask.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)))
        shape = {"matrix": (nbf, nbf), "diag": (nbf,), "g1": (3, nbf),
                 "gg": (3, ng), "f1": (3, nbf, nbf), "fg": (3, nbf, nbf)}[kind]
        out = np.zeros(shape) if out is None else np.ascontiguousarray(out)
        fn = getattr(self._dll, name)
        fn.restype = ctypes.c_int
        rc = fn(*args, ptrs, out.ctypes.data_as(_P))
        if rc != 0:
            raise RuntimeError(f"{name} returned {rc}")
        return out


class _NumpyKernel:
    """Fallback: the NumPy-backend kernel behind the same named interface."""

    def __init__(self, name: str):
        from .catalog import _integrand_for, entries
        entry = next((e for e in entries(include_heavy=True)
                      if e.name == name), None)
        if entry is None or entry.order == 0 or entry.kind or entry.giao:
            raise KeyError(f"no NumPy fallback for kernel {name!r}")
        from .emitters.cbackend import kernel_layout
        from .emitters.codegen import collapse, compile_function, generate_collapsed
        ki = _integrand_for(entry)
        self._ck = collapse(ki)
        self._gen = generate_collapsed(ki, name, batch=False)
        self._fn = compile_function(self._gen)
        self._layout = kernel_layout([self._ck], self._ck)
        self.scal_names = self._layout.scal_names

    def __call__(self, *, w, chi, out=None, **operands):
        from .emitters.tower import internal_collocation, translate_operands
        ng = np.asarray(w).shape[0]
        scal = {"w": np.asarray(w, dtype=np.float64),
                **_operands(operands, ng)}
        chi = np.asarray(chi, dtype=np.float64)
        col = internal_collocation(chi if chi.ndim == 3 else chi[None])
        eng = {**translate_operands(self._layout, scal),
               **{x: scal[x] for x in self._layout.libxc}}
        args = [eng["w"], col["chi"], col.get("dchi")]
        if self._gen.uses_lapl_chi:
            args.append(col["lapl_chi"])
        if "hess_chi" in self._ck.params:
            args.append(col["hess_chi"])
        for p in self._ck.params:
            if p in ("w", "chi", "dchi", "lapl_chi", "hess_chi"):
                continue
            if p.startswith("hess_rho"):
                args.append(np.stack([eng[f"{p}_{c}"] for c in
                                      ("xx", "xy", "xz", "yy", "yz", "zz")]))
            elif p.startswith(("grad_rho", "jp")):
                args.append(np.stack([eng[f"{p}_{ax}"] for ax in "xyz"]))
            else:
                args.append(eng[p])
        res = self._fn(*args)
        if out is not None:
            out += res
            return out
        return res


def get_kernel(name: str, library: Optional[Library] = None):
    """A callable for the named kernel: compiled library when available,
    NumPy backend otherwise. The returned callable takes the same named
    operands either way."""
    if library is None:
        try:
            library = Library()
        except OSError:
            library = None
    if library is not None:
        return lambda **kw: library(name, **kw)
    return _NumpyKernel(name)
