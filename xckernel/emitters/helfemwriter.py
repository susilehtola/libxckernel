"""Emit the fxc coefficient channels for HelFEM as C++.

HelFEM's radial and finite-element workers assemble the XC matrix from
per-grid-point coefficient fields: a scalar that multiplies the
basis-function pair, a vector that multiplies the pair's gradient, and
(for a tau-meta-GGA) a scalar that multiplies the pair's kinetic-energy
density.  Its ground state already builds those fields; the response
needs the same shapes evaluated from the kernel chain rule instead of
from the first derivatives, which is exactly what
``response.fxc_channels`` produces.

The emitted functions take the operands by value and write the channels
through references, so a host calls one per quadrature point inside the
loop it already has:

    xck_helfem_fxc_gga(rho_p1, grad_rho_p1_r, grad_rho_r,
                       v2rho2, v2rhosigma, v2sigma2, vsigma,
                       u, v_r);

HelFEM's spherically averaged atomic worker carries a single radial
gradient component, so the radial kernels are emitted over one axis; the
diatomic pure-m worker carries two and uses the ``_2d`` variants; the
three-dimensional atomic and diatomic workers carry three and use the
``_3d`` ones.  All three come from the same expressions, so they cannot
drift apart.

The current-corrected tau-meta-GGA (``cmgga_tau``, tau~ = tau -
|j_p|^2 / 2 rho per spin) adds a fourth channel, ``y_<c>``, which
multiplies the pair's paramagnetic current density.  HelFEM's basis
functions are complex, chi = f exp(i m phi), and its density matrices
real symmetric, so the current lives in the basis and a real symmetric
perturbation moves it; the channels come from
``engine.current_channels``, which derives them in that convention (the
catalog's ``cmgga_tau`` kernels assume the opposite one).  Per worker:

* ``_3d`` (atomic and diatomic): three gradient and three current
  components, physical, as for the other ``_3d`` kernels.
* ``_2d`` (diatomic pure-m): gradient (mu, nu), current (phi) only: the
  pure-m current is purely azimuthal, j_phi = sum_m m rho_m / h_phi, and
  the host multiplies y_phi by m / h_phi times the real pair product
  f_u f_v within the m block.
* radial (spherically averaged atomic): none.  The spherically averaged
  density carries no current, j_p = 0 identically, so the functional
  reduces to the tau-meta-GGA and the ``mgga_tau`` kernels apply.

The ground state needs the ``u`` and ``y`` channels (``xck_helfem_vxc_
cmgga_tau_*``): u = vrho + vtau |j|^2 / (2 rho^2) replaces vrho, y =
-vtau j / rho is new, and the gradient channel is the shared
``xck_helfem_vxc_grad``.  The Libxc arrays are evaluated at tau~.

Reproduce with: python -m xckernel.emitters.helfemwriter --emit <file>
"""

from __future__ import annotations

from . import fieldkernel
from .fieldkernel import (ChannelLayout, FieldKernel, GradientLayout,
                          SpinChannelLayout, SpinGradientLayout)
from ..engine.fock import vxc_channels
from ..engine.response import fxc_channels
from ..engine.current_channels import (fxc_current_channels,
                                       vxc_current_channels)
from ..engine.spin_kernel import vxc_channels_spin
from ..inputs.basis import PROLATE_PUREM, RADIAL

#: families HelFEM can assemble today: it has no laplacian response, so
#: the mgga_lapl and full mgga families are deliberately absent.
FAMILIES = ("lda", "gga", "mgga_tau")

#: HelFEM's spherically averaged atomic worker keeps one radial gradient
#: component; the diatomic and 3D workers keep three.  The single
#: component is not a hand-collapse of a Cartesian kernel any more: it
#: is the generator's own RADIAL coordinate system, so the metric that
#: distinguishes the two lives in inputs/basis.py rather than here.
RADIAL_AXES = RADIAL.axes
CARTESIAN_AXES = ("x", "y", "z")

#: HelFEM's diatomic pure-m worker keeps two: the density of an
#: exp(i m phi) orbital is phi-independent, so the azimuthal gradient
#: component vanishes identically and quadrature runs over the (mu, nu)
#: plane.  The channels are written in PHYSICAL components, so the two
#: surviving ones obey the same chain rule as any other pair -- what the
#: reduction removes is a component, not a metric.
PUREM_AXES = PROLATE_PUREM.axes

#: the current-density family, emitted by its own specs (below); its
#: channels need a complex-basis, real-P derivation of their own
CURRENT_FAMILIES = ("cmgga_tau",)

#: current components per worker.  The 3D workers carry all three; the
#: pure-m worker carries the azimuthal one ONLY -- a component its
#: gradient channels do not have, which is why these are not
#: PUREM_AXES.
CARTESIAN_CURRENTS = CARTESIAN_AXES
PUREM_CURRENTS = ("phi",)


def _channel_params(spec_exprs, axes, spins=None, currents=()) -> list:
    """Positional parameter order for a channel kernel.

    The operands a :class:`FieldKernel` discovers come back sorted by
    name, which for a single axis (``_r``) or the Cartesian ones
    (``_x``) happens to put the perturbed gradient before the reference
    one -- ``p1`` sorts before ``r``, ``x``, ``y``, ``z``.  For the
    pure-m axes it does not: ``mu`` sorts before ``p1``, so the same
    spec would emit the two blocks in the opposite order and a call site
    written from one signature would silently mis-bind the other.

    Pin the order instead: perturbed gradient components, reference
    gradient components, the perturbed density and tau, then the Libxc
    arrays.  Every kernel here then has the same shape of signature
    whatever its axes are called.
    """
    ops = sorted({sym.name for e in spec_exprs.values()
                  for sym in e.free_symbols})
    head = []
    for sp_ in (spins or (None,)):
        tag = "" if sp_ is None else f"_{sp_}"
        head += [f"grad_rho{tag}_p1_{ax}" for ax in axes]
        head += [f"grad_rho{tag}_{ax}" for ax in axes]
    for sp_ in (spins or (None,)):
        tag = "" if sp_ is None else f"_{sp_}"
        head.append(f"rho{tag}_p1")
    for sp_ in (spins or (None,)):
        tag = "" if sp_ is None else f"_{sp_}"
        head.append(f"tau{tag}_p1")
    # current-density families: the perturbed current, then the current,
    # then the density (tau~ divides by it), per spin
    for sp_ in (spins or (None,)):
        tag = "" if sp_ is None else f"_{sp_}"
        head += [f"jp{tag}_p1_{ax}" for ax in currents]
    for sp_ in (spins or (None,)):
        tag = "" if sp_ is None else f"_{sp_}"
        head += [f"jp{tag}_{ax}" for ax in currents]
    if currents:
        for sp_ in (spins or (None,)):
            head.append("rho" if sp_ is None else f"rho_{sp_}")
    head = [n for n in head if n in ops]
    return head + [n for n in ops if n not in head]


def spec_radial(family: str) -> FieldKernel:
    """Channel kernel for a worker with one radial gradient component."""
    exprs = fxc_channels(family, coords=RADIAL)
    return FieldKernel(
        name=f"xck_helfem_fxc_{family}",
        exprs=exprs,
        layout=ChannelLayout(axes=RADIAL_AXES),
        doc=(f"{family} fxc coefficient channels, one radial gradient "
             "component.",
             "u multiplies the basis-function pair, v_r its radial "
             "derivative,",
             "w_tau (tau families) the pair's kinetic-energy density.")
    )


def spec_cartesian(family: str) -> FieldKernel:
    """Channel kernel for a worker with three gradient components."""
    return FieldKernel(
        name=f"xck_helfem_fxc_{family}_3d",
        exprs=fxc_channels(family),
        layout=ChannelLayout(axes=CARTESIAN_AXES),
        doc=(f"{family} fxc coefficient channels, three gradient "
             "components.",)
    )


def spec_radial_spin(family: str) -> FieldKernel:
    """Spin-resolved channel kernel, one radial gradient component.

    The polarized Libxc arrays keep their flat component packing, so an
    operand named ``v2sigma2_3`` is row 3 of the caller's v2sigma2.
    """
    from ..engine.spin_kernel import fxc_channels_spin
    exprs = fxc_channels_spin(family, coords=RADIAL)
    return FieldKernel(
        name=f"xck_helfem_fxc_{family}_spin",
        exprs=exprs,
        layout=SpinChannelLayout(axes=RADIAL_AXES),
        doc=(f"Spin-resolved {family} fxc channels, one radial gradient "
             "component.",
             "u_s multiplies the basis-function pair of spin s, v_s_r its "
             "radial derivative,",
             "w_s (tau families) the pair's kinetic-energy density.",
             "Polarized Libxc arrays keep their flat packing "
             "(v2rho2_0 = uu, _1 = ud, _2 = dd, ...).")
    )


def spec_cartesian_spin(family: str) -> FieldKernel:
    """Spin-resolved channel kernel, three gradient components.

    The spin-resolved gradient channel mixes the two spins' gradients --
    sigma_ab = grad rho_a . grad rho_b is not a sum of squares -- so
    unlike the ground-state potential this cannot be applied one
    component at a time from a single-component kernel.  The three
    components have to be emitted together.
    """
    from ..engine.spin_kernel import fxc_channels_spin
    return FieldKernel(
        name=f"xck_helfem_fxc_{family}_3d_spin",
        exprs=fxc_channels_spin(family),
        layout=SpinChannelLayout(axes=CARTESIAN_AXES),
        doc=(f"Spin-resolved {family} fxc channels, three gradient "
             "components.",
             "Polarized Libxc arrays keep their flat packing "
             "(v2rho2_0 = uu, _1 = ud, _2 = dd, ...).")
    )


def spec_purem(family: str) -> FieldKernel:
    """Channel kernel for the diatomic pure-m worker's two components."""
    exprs = fxc_channels(family, coords=PROLATE_PUREM)
    return FieldKernel(
        name=f"xck_helfem_fxc_{family}_2d",
        exprs=exprs,
        params=_channel_params(exprs, PUREM_AXES),
        binding="positional",
        layout=ChannelLayout(axes=PUREM_AXES),
        doc=(f"{family} fxc coefficient channels, two gradient components "
             "(mu, nu).",
             "The pure-m density is phi-independent, so the azimuthal "
             "component is absent",
             "rather than zero: it is never formed.")
    )


def spec_purem_spin(family: str) -> FieldKernel:
    """Spin-resolved pure-m channel kernel, two gradient components."""
    from ..engine.spin_kernel import fxc_channels_spin
    exprs = fxc_channels_spin(family, coords=PROLATE_PUREM)
    return FieldKernel(
        name=f"xck_helfem_fxc_{family}_2d_spin",
        exprs=exprs,
        params=_channel_params(exprs, PUREM_AXES, spins=("a", "b")),
        binding="positional",
        layout=SpinChannelLayout(axes=PUREM_AXES),
        doc=(f"Spin-resolved {family} fxc channels, two gradient "
             "components (mu, nu).",
             "Polarized Libxc arrays keep their flat packing "
             "(v2rho2_0 = uu, _1 = ud, _2 = dd, ...).")
    )


def _vgrad_exprs(family: str, spin: bool):
    """Only the gradient channels: the rho and tau channels of the
    ground-state potential are Libxc outputs the host already holds, so
    emitting them would just copy vrho and vtau through a function."""
    if family in CURRENT_FAMILIES:
        ch = vxc_current_channels(RADIAL_AXES, (), polarized=spin)
        return {k: v for k, v in ch.items() if k.startswith("grad")}
    ch = (vxc_channels_spin if spin else vxc_channels)(family, RADIAL)
    return {k: v for k, v in ch.items() if k.startswith("grad")}


def _shared_vgrad_exprs(spin: bool):
    """The gradient channel shared by every family HelFEM assembles.

    For a GGA and for a tau-meta-GGA alike the gradient coefficient is
    2 vsigma grad(rho): tau does not enter it.  Rather than emit the
    same function body under several names, emit it once -- but verify
    the sharing here, so that adding a family whose gradient channel
    really does differ (a current-density or Hessian functional) fails
    loudly instead of silently getting the wrong kernel.
    """
    base = _vgrad_exprs("gga", spin)
    for fam in FAMILIES + CURRENT_FAMILIES:
        if fam == "lda":
            continue
        other = _vgrad_exprs(fam, spin)
        if other != base:
            raise AssertionError(
                f"the {fam} gradient channel of the XC potential differs "
                f"from the GGA one; emit it separately instead of sharing "
                f"({other} vs {base})")
    return base


def spec_vgrad() -> FieldKernel:
    """Gradient coefficient of the GROUND-STATE XC potential.

    This is the ``2 vsigma grad(rho)`` every host writes by hand -- and
    HelFEM wrote six times over, once per worker and spin channel.  It
    is a chain rule like any other, so it is generated: the emitted
    expression comes from the same ingredient derivatives as the kernel
    above and cannot drift from them.
    """
    return FieldKernel(
        name="xck_helfem_vxc_grad",
        exprs=_shared_vgrad_exprs(spin=False),
        layout=GradientLayout(axes=RADIAL_AXES),
        doc=("Ground-state XC potential, gradient channel (GGA and "
             "tau-meta-GGA alike).",
             "One component: the channel depends only on its own "
             "component, because",
             "sigma is a sum of squares, so the caller applies this once "
             "per component",
             "in any coordinate system -- radial, spherical, prolate "
             "spheroidal or pure-m.")
    )


def spec_vgrad_spin() -> FieldKernel:
    """Spin-resolved gradient coefficient of the ground-state potential."""
    return FieldKernel(
        name="xck_helfem_vxc_grad_spin",
        exprs=_shared_vgrad_exprs(spin=True),
        layout=SpinGradientLayout(axes=RADIAL_AXES),
        doc=("Spin-resolved ground-state XC potential, gradient channel.",
             "Polarized Libxc arrays keep their flat packing "
             "(vsigma_0 = aa, _1 = ab, _2 = bb).")
    )


def _vxc_current_params(exprs, currents, spins) -> list:
    """Ground-state current kernels: density, current, then the Libxc
    arrays (sorted, flat packing)."""
    ops = sorted({sym.name for e in exprs.values() for sym in e.free_symbols})
    head = []
    for sp_ in spins:
        head.append("rho" if sp_ is None else f"rho_{sp_}")
    for sp_ in spins:
        tag = "" if sp_ is None else f"_{sp_}"
        head += [f"jp{tag}_{ax}" for ax in currents]
    head = [n for n in head if n in ops]
    return head + [n for n in ops if n not in head]


def spec_vxc_current(currents, suffix: str, spin: bool) -> FieldKernel:
    """Ground-state u and y channels of the current-corrected
    tau-meta-GGA.  The gradient axes do not enter (the gradient channel
    is the shared xck_helfem_vxc_grad), so only the current's are given."""
    from .fieldkernel import ExplicitLayout
    ch = vxc_current_channels((), currents, polarized=spin)
    spins = ("a", "b") if spin else (None,)
    exprs, targets = {}, []
    for sp_ in spins:
        t = "" if sp_ is None else f"_{sp_}"
        exprs[f"rho{t}"] = ch[f"rho{t}"]
        targets.append(f"u{t}")
        for ax in currents:
            exprs[f"jp{t}_{ax}"] = ch[f"jp{t}_{ax}"]
            targets.append(f"y{t}_{ax}")
    name = f"xck_helfem_vxc_cmgga_tau_{suffix}" + ("_spin" if spin else "")
    comps = ", ".join(currents)
    doc = [f"Ground-state cmgga_tau potential: u replaces vrho, y_<c> "
           f"multiplies the pair's current ({comps}).",
           "tau~ = tau - |j|^2 / (2 rho) per spin; Libxc arrays evaluated "
           "at tau~.",
           "w = vtau and v = xck_helfem_vxc_grad as for the "
           "tau-meta-GGA."]
    if suffix == "2d":
        doc.append("Pure-m: the current is azimuthal only; the host "
                   "multiplies y_phi by m / h_phi f_u f_v.")
    if spin:
        doc.append("Polarized Libxc arrays keep their flat packing "
                   "(vrho_0 = a, _1 = b).")
    return FieldKernel(
        name=name, exprs=exprs,
        params=_vxc_current_params(exprs, currents, spins),
        binding="positional",
        layout=ExplicitLayout(targets=targets,
                              ret="(" + ", ".join(targets) + ")"),
        doc=tuple(doc))


def spec_fxc_current(gaxes, currents, suffix: str, spin: bool) -> FieldKernel:
    """fxc channels of the current-corrected tau-meta-GGA: u, v_<ax>,
    w_tau and the current channel y_<c>, for a perturbation with rho_p1,
    grad_rho_p1, tau_p1 and the perturbed current jp_p1."""
    exprs = fxc_current_channels(gaxes, currents, polarized=spin)
    spins = ("a", "b") if spin else None
    name = f"xck_helfem_fxc_cmgga_tau_{suffix}" + ("_spin" if spin else "")
    doc = [f"cmgga_tau fxc coefficient channels: gradient ({', '.join(gaxes)}),"
           f" current ({', '.join(currents)}).",
           "u, v_<ax>, w as for the tau-meta-GGA; y_<c> multiplies the "
           "pair's current density",
           "Im(chi_u^* d_c chi_v), symmetrized (complex basis, real "
           "symmetric P).",
           "tau~ = tau - |j|^2 / (2 rho) per spin; Libxc arrays evaluated "
           "at tau~."]
    if suffix == "2d":
        doc.append("Pure-m: the current is azimuthal only; the host "
                   "multiplies y_phi by m / h_phi f_u f_v.")
    if spin:
        doc.append("Polarized Libxc arrays keep their flat packing "
                   "(v2rho2_0 = uu, _1 = ud, _2 = dd, ...).")
    layout = (SpinChannelLayout(axes=gaxes, currents=currents) if spin
              else ChannelLayout(axes=gaxes, currents=currents))
    return FieldKernel(
        name=name, exprs=exprs,
        params=_channel_params(exprs, gaxes, spins=spins,
                               currents=currents),
        binding="positional", layout=layout, doc=tuple(doc))


def specs_current():
    """The current-density kernels: ground state and fxc, 3D and pure-m,
    restricted and spin-resolved.  No radial ones: the spherically
    averaged density carries no current."""
    out = []
    for gaxes, cur, sfx in ((CARTESIAN_AXES, CARTESIAN_CURRENTS, "3d"),
                            (PUREM_AXES, PUREM_CURRENTS, "2d")):
        for spin in (False, True):
            out.append(spec_vxc_current(cur, sfx, spin))
            out.append(spec_fxc_current(gaxes, cur, sfx, spin))
    return out


def specs():
    """Every kernel this writer emits, radial first."""
    out = [spec_vgrad(), spec_vgrad_spin()]
    for fam in FAMILIES:
        out.append(spec_radial(fam))
        out.append(spec_radial_spin(fam))
        if fam != "lda":
            out.append(spec_purem(fam))
            out.append(spec_purem_spin(fam))
            out.append(spec_cartesian(fam))
            out.append(spec_cartesian_spin(fam))
    return out + specs_current()


def emit_header(cse: bool = True) -> str:
    """The complete self-contained header HelFEM includes."""
    body = [fieldkernel.emit_cxx(s, cse=cse, drop_zero=True)
            for s in specs()]
    preamble = "\n".join([
        "// Machine-generated by xckernel; do not edit.",
        "// Reproduce with: python -m xckernel.emitters.helfemwriter"
        " --emit <file>",
        "// Copyright (c) 2026 Susi Lehtola.",
        "#pragma once",
        "#include <cmath>",
        "",
        "namespace helfem {",
        "namespace xckernel {",
        "",
    ])
    tail = "\n".join(["", "} // namespace xckernel",
                      "} // namespace helfem", ""])
    return preamble + "\n\n".join(body) + tail


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--emit", metavar="FILE",
                   help="write the header here instead of stdout")
    p.add_argument("--no-cse", action="store_true",
                   help="skip common-subexpression elimination")
    a = p.parse_args(argv)
    src = emit_header(cse=not a.no_cse)
    if a.emit:
        with open(a.emit, "w") as f:
            f.write(src)
        print(f"wrote {a.emit}")
    else:
        print(src)


if __name__ == "__main__":
    main()
