"""Shared emission service for per-grid-point field kernels.

Several writers emit the same shape of artifact: a function that binds a
set of named operand arrays, evaluates a fixed set of named scalar
expressions over them, and returns the results in some container.  Only
three things actually vary between writers:

* **binding** -- how operands reach the body: unpacked from an ``ops``
  dict (the GPAW idiom, where the host owns a bag of arrays) or taken as
  positional parameters (the C/Fortran-friendly idiom);
* **layout** -- how the evaluated channels are named and returned: a flat
  dict of scalar/vector channels, the same per spin channel, or a dict
  keyed by tuples;
* **printer** -- the target language (NumPy today, C99 next).

Everything else -- discovering the operands from the free symbols in a
stable order, common-subexpression elimination, dropping identically zero
channels, and laying out the function -- is common, and lives here.

A writer therefore declares a :class:`FieldKernel` (what to emit) and
never contains printing logic; adding a new writer means writing specs,
not another printer.  ``cse=True`` is a pure optimization of the emitted
code and never changes its meaning, so it can be enabled per writer
without touching the specs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import sympy as sp
from sympy.printing.numpy import NumPyPrinter

#: Rendered as ``(target, expression)`` pairs by a layout.
Assignment = Tuple[str, Any]


# --------------------------------------------------------------------------
# layouts: how evaluated channels are named and handed back
# --------------------------------------------------------------------------

class Layout:
    """Maps an expression mapping onto assignments and a return statement."""

    def prologue(self) -> List[str]:
        return []

    def assignments(self, exprs: Dict) -> List[Assignment]:
        raise NotImplementedError

    def epilogue(self, exprs: Dict) -> List[str]:
        raise NotImplementedError


@dataclass
class ChannelLayout(Layout):
    """Scalar ``u``, vector ``v_<axis>`` and optional ``w_tau`` channels,
    returned as ``{'rho': u, 'grad': [v_x, v_y, v_z][, 'tau': w_tau]}``.

    This is the pair-coefficient shape a Casida host consumes: the XC pair
    potential is ``u - div(v)`` and the tau channel contracts with the
    second pair's kinetic-energy density.
    """

    axes: Sequence[str] = ("x", "y", "z")

    def assignments(self, exprs):
        out = [("u", exprs["rho"])]
        out += [(f"v_{ax}", exprs[f"grad_{ax}"]) for ax in self.axes]
        if "tau" in exprs:
            out.append(("w_tau", exprs["tau"]))
        return out

    def epilogue(self, exprs):
        vec = ", ".join(f"v_{ax}" for ax in self.axes)
        ret = "{'rho': u, 'grad': [" + vec + "]"
        if "tau" in exprs:
            ret += ", 'tau': w_tau"
        return [f"    return {ret}}}"]


@dataclass
class SpinChannelLayout(Layout):
    """:class:`ChannelLayout` per spin channel, returned in one flat dict
    keyed ``rho_<s>`` / ``grad_<s>`` / ``tau_<s>``."""

    axes: Sequence[str] = ("x", "y", "z")
    spins: Sequence[str] = ("a", "b")

    def assignments(self, exprs):
        out = []
        for s in self.spins:
            out.append((f"u_{s}", exprs[f"rho_{s}"]))
            out += [(f"v_{s}_{ax}", exprs[f"grad_{s}_{ax}"])
                    for ax in self.axes]
            if f"tau_{s}" in exprs:
                out.append((f"w_{s}", exprs[f"tau_{s}"]))
        return out

    def epilogue(self, exprs):
        entries = []
        for s in self.spins:
            entries.append(f"'rho_{s}': u_{s}")
            vec = ", ".join(f"v_{s}_{ax}" for ax in self.axes)
            entries.append(f"'grad_{s}': [{vec}]")
            if f"tau_{s}" in exprs:
                entries.append(f"'tau_{s}': w_{s}")
        return ["    return {" + ", ".join(entries) + "}"]


@dataclass
class ExplicitLayout(Layout):
    """Caller-supplied assignment targets and return statement, with an
    optional prologue for shape queries and allocation.

    The escape hatch for writers whose container is neither a channel dict
    nor a keyed mapping -- a preallocated array, say -- where the targets
    are index expressions the writer alone knows how to form.
    """

    targets: Sequence[str]
    ret: str
    pre: Sequence[str] = ()

    def prologue(self):
        return list(self.pre)

    def assignments(self, exprs):
        return list(zip(self.targets, exprs.values()))

    def epilogue(self, exprs):
        return [f"    return {self.ret}"]


@dataclass
class MappingLayout(Layout):
    """A dict keyed by the expression mapping's own keys, accumulated into
    ``<var>`` and returned.  Keys are rendered with ``repr``, so tuples of
    strings and of ints both come out as valid literals."""

    var: str = "c"

    def prologue(self):
        return [f"    {self.var} = {{}}"]

    def assignments(self, exprs):
        return [(f"{self.var}[{k!r}]", e) for k, e in sorted(exprs.items())]

    def epilogue(self, exprs):
        return [f"    return {self.var}"]


# --------------------------------------------------------------------------
# the spec and the emitter
# --------------------------------------------------------------------------

@dataclass
class FieldKernel:
    """What to emit: a named function over named field expressions.

    name:     emitted function name
    exprs:    mapping from channel key to sympy expression
    layout:   how the channels are named and returned
    doc:      docstring lines, verbatim and already indented
    binding:  ``"ops"`` (unpack from a dict) or ``"positional"``
    params:   positional parameter names; defaults to the discovered
              operands, which is what a dict binding always uses
    ops_name: the dict parameter name for the ``"ops"`` binding
    """

    name: str
    exprs: Dict[Any, Any]
    layout: Layout
    doc: Sequence[str] = ()
    binding: str = "ops"
    params: Sequence[str] = None
    ops_name: str = "ops"

    def operands(self) -> List[str]:
        """The operand names the expressions actually reference, in a
        stable order.  Discovered before any CSE, so temporaries never
        leak into the signature."""
        return sorted({s.name for e in self.exprs.values()
                       for s in e.free_symbols})


def emit(spec: FieldKernel, printer=None, cse: bool = False,
         drop_zero: bool = False) -> str:
    """Render one field kernel to source.

    cse:       eliminate common subexpressions across all channels.  A
               pure optimization of the emitted code; the values are
               unchanged, so enabling it never needs a spec change.
    drop_zero: omit channels whose expression is identically zero.
    """
    printer = printer or NumPyPrinter()
    operands = spec.operands()

    if spec.binding == "ops":
        params = [spec.ops_name]
        bind = [f"    {n} = {spec.ops_name}['{n}']" for n in operands]
    elif spec.binding == "positional":
        params = list(spec.params if spec.params is not None else operands)
        bind = []
    else:
        raise ValueError(f"unknown binding {spec.binding!r}")

    body = spec.layout.assignments(spec.exprs)
    targets = [t for t, _e in body]
    exprs = [e for _t, e in body]

    temps: List[Tuple[str, Any]] = []
    if cse:
        temps, exprs = sp.cse(exprs, optimizations="basic",
                              symbols=sp.numbered_symbols("t"))

    lines = [f"def {spec.name}({', '.join(params)}):"]
    lines += list(spec.doc)
    lines += bind
    # allocation/shape prologue first, then the shared temporaries: a
    # temporary may be large, and the container it feeds should exist.
    lines += spec.layout.prologue()
    lines += [f"    {s} = {printer.doprint(e)}" for s, e in temps]
    for target, e in zip(targets, exprs):
        if drop_zero and e == 0:
            continue
        lines.append(f"    {target} = {printer.doprint(e)}")
    lines += spec.layout.epilogue(spec.exprs)
    return "\n".join(lines)


def emit_all(specs: Iterable[FieldKernel], **kwargs) -> List[str]:
    """Render a sequence of field kernels with shared settings."""
    return [emit(s, **kwargs) for s in specs]
