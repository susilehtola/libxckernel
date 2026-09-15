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

Reproduce with: python -m xckernel.emitters.helfemwriter --emit <file>
"""

from __future__ import annotations

from . import fieldkernel
from .fieldkernel import (ChannelLayout, FieldKernel, GradientLayout,
                          SpinChannelLayout, SpinGradientLayout)
from ..engine.fock import vxc_channels
from ..engine.response import fxc_channels
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


def _channel_params(spec_exprs, axes, spins=None) -> list:
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
    for fam in FAMILIES:
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
    return out


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
