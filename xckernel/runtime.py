"""Runtime interface to a compiled libxckernel, with NumPy fallback.

The compiled library is self-describing: each kernel exports its ordered
scalar-operand name table (``<name>_scal_names``/``<name>_n_scal``), so the
generic dispatcher below needs no per-kernel binding code -- it reads the
operand order from the binary and matches keyword arguments to it.

    from xckernel.runtime import Library
    lib = Library("/path/to/libxckernel.so")        # or $XCKERNEL_LIBRARY
    F = lib("xck_gga_r_o2", chi=chi, dchi=dchi, w=w,
            grad_rho_x=..., ..., rho_p1=..., v2rho2=..., ...)

Vector conveniences: ``grad_rho=(3,ng)`` may be passed instead of the three
components (same for any ``grad_rho_*`` operand family).

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

_P = ctypes.POINTER(ctypes.c_double)

#: packed symmetric-tensor components (density Hessian), canonical order.
_H6_COMPS = ("xx", "xy", "xz", "yy", "yz", "zz")


def _expand_vector(scal, key, val, ng):
    """Expand a (3,ng) vector / (6,ng) packed-tensor operand to components."""
    if val.ndim == 2 and val.shape == (3, ng):
        for i, ax in enumerate("xyz"):
            scal[f"{key}_{ax}"] = np.ascontiguousarray(val[i])
        return True
    if val.ndim == 2 and val.shape == (6, ng):
        for i, comp in enumerate(_H6_COMPS):
            scal[f"{key}_{comp}"] = np.ascontiguousarray(val[i])
        return True
    return False


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

    def __call__(self, name: str, *, chi, dchi, w, lapl_chi=None,
                 hess_chi=None, out=None, **operands) -> np.ndarray:
        """Call a kernel with named operands; returns the (nbf,nbf) matrix.

        chi: (nbf, ng); dchi: (3, nbf, ng); w: (ng,); scalar operands by the
        names in scal_names(name) (vector families may be passed whole, e.g.
        grad_rho=(3,ng), hess_rho=(6,ng)). ``out`` is accumulated into when
        provided.
        """
        chi = np.ascontiguousarray(chi, dtype=np.float64)
        dchi = np.ascontiguousarray(dchi, dtype=np.float64)
        nbf, ng = chi.shape
        if dchi.shape != (3, nbf, ng):
            raise ValueError(f"dchi must be (3,{nbf},{ng})")

        # expand vector conveniences (grad_rho -> grad_rho_x/y/z etc.)
        scal: Dict[str, np.ndarray] = {"w": np.ascontiguousarray(w)}
        for key, val in operands.items():
            val = np.asarray(val, dtype=np.float64)
            if _expand_vector(scal, key, val, ng):
                pass
            elif val.shape == (ng,):
                scal[key] = np.ascontiguousarray(val)
            else:
                raise ValueError(f"operand {key!r}: expected ({ng},), "
                                 f"(3,{ng}) or (6,{ng}), got {val.shape}")

        names = self.scal_names(name)
        missing = [n for n in names if n not in scal]
        if missing:
            raise TypeError(f"{name}: missing operands {missing}")
        ptrs = (_P * len(names))(*[scal[n].ctypes.data_as(_P)
                                   for n in names])

        if out is None:
            out = np.zeros((nbf, nbf))
        else:
            out = np.ascontiguousarray(out)
        if lapl_chi is not None:
            lapl_chi = np.ascontiguousarray(lapl_chi, dtype=np.float64)
            lapl_ptr = lapl_chi.ctypes.data_as(_P)
        else:
            lapl_ptr = None
        if hess_chi is not None:
            hess_chi = np.ascontiguousarray(hess_chi, dtype=np.float64)
            hess_ptr = hess_chi.ctypes.data_as(_P)
        else:
            hess_ptr = None

        fn = getattr(self._dll, name)
        fn.restype = ctypes.c_int
        rc = fn(ctypes.c_int64(ng), ctypes.c_int64(nbf),
                chi.ctypes.data_as(_P), dchi.ctypes.data_as(_P),
                lapl_ptr, hess_ptr, ptrs, out.ctypes.data_as(_P))
        if rc != 0:
            raise RuntimeError(f"{name} returned {rc}")
        return out


class _NumpyKernel:
    """Fallback: the NumPy-backend kernel behind the same named interface."""

    def __init__(self, name: str):
        from .catalog import CatalogEntry, _integrand_for, entries
        entry = next((e for e in entries() if e.name == name), None)
        if entry is None or entry.order == 0:
            raise KeyError(f"unknown kernel {name!r}")
        from .emitters.cbackend import scal_order
        from .emitters.codegen import collapse, compile_function, generate_collapsed
        ki = _integrand_for(entry)
        self._ck = collapse(ki)
        self._gen = generate_collapsed(ki, name, batch=False)
        self._fn = compile_function(self._gen)
        self.scal_names = scal_order(self._ck)

    def __call__(self, *, chi, dchi, w, lapl_chi=None, hess_chi=None,
                 out=None, **operands):
        ng = np.asarray(w).shape[0]
        scal = {"w": np.asarray(w)}
        for key, val in operands.items():
            val = np.asarray(val)
            if not _expand_vector(scal, key, val, ng):
                scal[key] = val
        args = [scal["w"], np.asarray(chi), np.asarray(dchi)]
        if self._gen.uses_lapl_chi:
            args.append(np.asarray(lapl_chi))
        if "hess_chi" in self._ck.params:
            args.append(np.asarray(hess_chi))
        for p in self._ck.params:
            if p in ("w", "chi", "dchi", "lapl_chi", "hess_chi"):
                continue
            if p.startswith("hess_rho"):
                args.append(np.stack([scal[f"{p}_{c}"] for c in _H6_COMPS]))
            elif p.startswith(("grad_rho", "jp")):
                args.append(np.stack([scal[f"{p}_{ax}"] for ax in "xyz"]))
            else:
                args.append(scal[p])
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
