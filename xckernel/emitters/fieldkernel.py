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
from sympy.printing.c import C99CodePrinter
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
class GradientLayout(Layout):
    """Just the vector channel, as ``v_<axis>``.

    A host that already holds vrho and vtau -- they are Libxc outputs,
    not derived quantities -- needs only the gradient coefficient of the
    XC potential built for it, and emitting the identity channels
    alongside would be noise.  Because the gradient channel of a
    semilocal functional depends only on its OWN component (sigma is a
    sum of squares, so d sigma / d g_i = 2 g_i), one single-component
    kernel serves any number of components and any coordinate system:
    the host calls it once per component, exactly as its assembly loop
    already runs.
    """

    axes: Sequence[str] = ("x", "y", "z")

    def assignments(self, exprs):
        return [(f"v_{ax}", exprs[f"grad_{ax}"]) for ax in self.axes]

    def epilogue(self, exprs):
        vec = ", ".join(f"v_{ax}" for ax in self.axes)
        return ["    return [" + vec + "]"]


@dataclass
class SpinGradientLayout(Layout):
    """:class:`GradientLayout` per spin channel, as ``v_<s>_<axis>``."""

    axes: Sequence[str] = ("x", "y", "z")
    spins: Sequence[str] = ("a", "b")

    def assignments(self, exprs):
        return [(f"v_{s}_{ax}", exprs[f"grad_{s}_{ax}"])
                for s in self.spins for ax in self.axes]

    def epilogue(self, exprs):
        entries = ["'grad_%s': [%s]" % (s, ", ".join(f"v_{s}_{ax}"
                                                     for ax in self.axes))
                   for s in self.spins]
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

# --------------------------------------------------------------------------
# C++ emission
# --------------------------------------------------------------------------

class CxxPrinter(C99CodePrinter):
    """C99 with small integer powers written out as products.

    ``pow(x, 2)`` is a library call the compiler has to recognize; the
    expressions here are dense in squares and cubes of grid fields, so
    spelling them out is both faster and easier to read.
    """

    def _print_Pow(self, expr):
        if expr.exp.is_Integer and 1 < int(expr.exp) <= 4:
            b = self.parenthesize(expr.base, 100)
            return "*".join([b] * int(expr.exp))
        return super()._print_Pow(expr)


def emit_cxx(spec: FieldKernel, cse: bool = True, drop_zero: bool = False,
             printer=None, inline: bool = True) -> str:
    """Render one field kernel as a C++ function over scalar fields.

    The emitted function takes the operands by value and writes each
    channel through a reference, so a host evaluates it once per
    quadrature point inside whatever loop it already has:

        xck_name(rho, grad_rho_x, ..., u, v_x, v_y, v_z);

    Everything except the printing shell -- operand discovery, the
    channel layout, common-subexpression elimination -- is shared with
    the NumPy path, so the two emissions cannot drift apart.
    """
    printer = printer or CxxPrinter()
    operands = spec.operands()

    # The layout already names every channel (u, v_x, w_tau, u_a, ...);
    # those names ARE the output references, so the C++ signature cannot
    # drift from the layout the NumPy path uses.
    body = spec.layout.assignments(spec.exprs)
    pairs = [(t, e) for t, e in body if not (drop_zero and e == 0)]
    targets = [t for t, _ in pairs]
    exprs = [e for _, e in pairs]

    temps: List[Tuple[str, Any]] = []
    if cse and exprs:
        temps, exprs = sp.cse(exprs, optimizations="basic",
                              symbols=sp.numbered_symbols("t"))

    args = ([f"double {n}" for n in operands]
            + [f"double & {n}" for n in targets])

    lines = ["/* machine-generated by xckernel; do not edit. */"]
    lines += [f"// {d.strip()}" for d in spec.doc if d.strip()]
    lines.append(("static inline " if inline else "")
                 + f"void {spec.name}({', '.join(args)}) {{")
    for sym, e in temps:
        lines.append(f"  const double {sym} = {printer.doprint(e)};")
    for target, e in zip(targets, exprs):
        lines.append(f"  {target} = {printer.doprint(e)};")
    lines.append("}")
    return "\n".join(lines)
