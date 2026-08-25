"""Numerical validation of the curvilinear metric data in inputs.basis.

Everything the generator does in a non-Cartesian system rests on one
claim: written in PHYSICAL (orthonormal) components,

    g_i = (1/h_i) d_i f,        |grad f|^2 = sum_i g_i^2,

every xc chain-rule expression is identical to the Cartesian one, so the
metric enters only through the ingredient seeds.  That claim is only as
good as the Lame factors and angular terms recorded in
:mod:`xckernel.inputs.basis`, and those were previously checked only
indirectly, by running a host code.  This script checks them directly
against Cartesian ground truth, with no host and no Libxc:

  * spherical -- h = (1, r, r sin(theta));
  * prolate spheroidal -- the h_mu / h_nu / h_phi the diatomic worker
    supplies as grid arrays;
  * the spherically averaged radial reduction, whose tau carries the
    centrifugal l(l+1)/r^2 term that has no Cartesian counterpart;
  * the pure-m diatomic reduction, whose tau carries m^2/h_phi^2.

The last two are the delicate ones: they are not instances of a
coordinate system but reductions of one, in which an angular coordinate
has been integrated out and left a residual operator on the block index.

Run with: python -m xckernel.tests.curvilinear_validate
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from ..inputs.basis import (L_FACTOR, M_FACTOR, PROLATE, PROLATE_PUREM,
                            RADIAL, SPHERICAL)

RNG = np.random.default_rng(20260826)
#: step for the central differences taken in the curvilinear coordinates
STEP = 1e-5
#: focal half-distance of the prolate spheroidal system used here
FOCAL = 1.3


# --- a smooth test field, and its exact Cartesian gradient -----------------

class Field:
    """A sum of off-centre Gaussians: smooth, anisotropic, and with an
    analytic Cartesian gradient, so it exercises all three components."""

    def __init__(self, n: int = 4):
        self.centres = RNG.normal(scale=0.7, size=(n, 3))
        self.widths = RNG.uniform(0.4, 1.1, size=n)
        self.coeffs = RNG.normal(size=n)

    def value(self, xyz):
        xyz = np.asarray(xyz, dtype=float)
        d = xyz - self.centres
        return float(np.sum(self.coeffs * np.exp(-self.widths
                                                 * np.sum(d * d, axis=-1))))

    def grad(self, xyz):
        xyz = np.asarray(xyz, dtype=float)
        d = xyz - self.centres
        g = self.coeffs * np.exp(-self.widths * np.sum(d * d, axis=-1))
        return np.sum((-2.0 * self.widths * g)[:, None] * d, axis=0)


def _fd(fun, coord, i, h=STEP):
    """Central difference of ``fun`` along curvilinear component ``i``."""
    up, dn = np.array(coord, dtype=float), np.array(coord, dtype=float)
    up[i] += h
    dn[i] -= h
    return (fun(up) - fun(dn)) / (2.0 * h)


# --- coordinate maps ------------------------------------------------------

def spherical_to_xyz(c):
    r, th, ph = c
    return np.array([r * np.sin(th) * np.cos(ph),
                     r * np.sin(th) * np.sin(ph),
                     r * np.cos(th)])


def spherical_scales(c):
    """Evaluated from SPHERICAL.scale itself, not re-derived here, so a
    wrong Lame factor recorded in inputs.basis fails this test."""
    r, th, _ = c
    sub = {sp.Symbol("r", real=True, positive=True): r,
           sp.Symbol("sin_theta", real=True): np.sin(th)}
    return np.array([float(sp.sympify(h).subs(sub))
                     for h in SPHERICAL.scale])


def prolate_to_xyz(c, a=FOCAL):
    mu, nu, ph = c
    rho = a * np.sqrt(max(mu * mu - 1.0, 0.0) * max(1.0 - nu * nu, 0.0))
    return np.array([rho * np.cos(ph), rho * np.sin(ph), a * mu * nu])


#: PROLATE.scale holds opaque host-supplied grid symbols rather than
#: closed forms, because the generated code must match whatever
#: convention the host already uses for the focal distance.  These are
#: the values HelFEM's diatomic worker puts in them; what the test then
#: checks is the CONVENTION -- that those arrays are the Lame factors of
#: the physical components the generator assumes.
def prolate_scales(c, a=FOCAL):
    mu, nu, _ = c
    common = (mu * mu - nu * nu)
    vals = {"scale_mu": a * np.sqrt(common / (mu * mu - 1.0)),
            "scale_nu": a * np.sqrt(common / (1.0 - nu * nu)),
            "scale_phi": a * np.sqrt((mu * mu - 1.0) * (1.0 - nu * nu))}
    return np.array([vals[str(h)] for h in PROLATE.scale])


def xyz_to_prolate(xyz, a=FOCAL):
    """Inverse map, so a field defined on the prolate grid can be
    differentiated in CARTESIAN coordinates for the reference value."""
    x, y, z = xyz
    rp = np.hypot(np.hypot(x, y), z - a)
    rm = np.hypot(np.hypot(x, y), z + a)
    return np.array([(rp + rm) / (2 * a), (rm - rp) / (2 * a),
                     np.arctan2(y, x)])


# --- angular quadrature for the reduced (block-index) systems -------------

def sphere_grid(ntheta=24, nphi=48):
    """Product Gauss-Legendre x uniform grid with weights summing to 4 pi."""
    xs, wx = np.polynomial.legendre.leggauss(ntheta)
    phis = 2.0 * np.pi * (np.arange(nphi) + 0.5) / nphi
    wphi = 2.0 * np.pi / nphi
    dirs, wts = [], []
    for ct, w in zip(xs, wx):
        st = np.sqrt(1.0 - ct * ct)
        for ph in phis:
            dirs.append([st * np.cos(ph), st * np.sin(ph), ct])
            wts.append(w * wphi)
    return np.array(dirs), np.array(wts)


#: real solid harmonics as homogeneous polynomials of the unit vector,
#: for l = 0, 1, 2 -- normalization is fixed numerically below, so only
#: the angular SHAPE matters here.
def _harmonic(l: int, n):
    x, y, z = n[..., 0], n[..., 1], n[..., 2]
    if l == 0:
        return np.ones_like(x)
    if l == 1:
        return z
    if l == 2:
        return 3.0 * z * z - 1.0
    raise ValueError(l)


# --- the checks -----------------------------------------------------------

def _report(checks):
    bad = [c for c in checks if not c[1]]
    for name, ok, err in checks:
        if not ok:
            print(f"[FAIL] {name}: rel dev {err:.3e}")
    tag = "OK " if not bad else "FAIL"
    print(f"[{tag}] curvilinear_validate: {len(checks)} checks, "
          f"{len(bad)} failures")
    return not bad


def check_full_system(name, coords, to_xyz, scales, npts=6, tol=1e-7):
    """|grad f|^2 from the physical components must equal the Cartesian one.

    This is the statement that makes the whole design work: it is what
    lets sigma, and with it every GGA chain rule, cross into a
    curvilinear system unchanged.
    """
    out = []
    f = Field()
    for _ in range(npts):
        if name == "spherical":
            c = np.array([RNG.uniform(0.5, 2.0), RNG.uniform(0.4, 2.6),
                          RNG.uniform(0.0, 6.2)])
        else:
            c = np.array([RNG.uniform(1.15, 2.2), RNG.uniform(-0.8, 0.8),
                          RNG.uniform(0.0, 6.2)])
        h = scales(c)
        g = np.array([_fd(lambda q: f.value(to_xyz(q)), c, i)
                      for i in range(3)]) / h
        ref = f.grad(to_xyz(c))
        got, want = float(g @ g), float(ref @ ref)
        err = abs(got - want) / max(abs(want), 1e-12)
        out.append((f"{name}: |grad|^2 from h = {tuple(coords.axes)}",
                    err < tol, err))
    return out


def check_radial_reduction(npts=4, tol=1e-8):
    """tau of an l-blocked radial code.

    For psi = R(r) Y_lm the angular average of |grad psi|^2 is
    R'(r)^2 + l(l+1) R(r)^2 / r^2: the second term is exactly the
    ``angular`` entry of RADIAL, and it is the piece a Cartesian
    generator cannot produce.
    """
    dirs, wts = sphere_grid()
    out = []
    for l in (0, 1, 2):
        Y = _harmonic(l, dirs)
        Y = Y / np.sqrt(float(wts @ (Y * Y)))          # int |Y|^2 dOmega = 1
        for _ in range(npts):
            alpha = RNG.uniform(0.3, 1.2)
            r = RNG.uniform(0.4, 2.0)

            def R(rr):
                return np.exp(-alpha * rr * rr) * rr ** l

            def dR(rr):
                return np.exp(-alpha * rr * rr) * (
                    l * rr ** (l - 1) - 2 * alpha * rr ** (l + 1))

            # |grad psi|^2 = (dR)^2 |Y|^2 + (R/r)^2 |grad_ang Y|^2, and the
            # angular integral of the second term is l(l+1) by definition of
            # the spherical harmonic -- evaluate it instead by differencing
            # psi in Cartesian space, so nothing about Y is assumed.
            def psi(xyz):
                rr = float(np.linalg.norm(xyz))
                n = np.asarray(xyz) / rr
                yv = _harmonic(l, n[None, :])[0]
                return R(rr) * yv

            norm = np.sqrt(float(wts @ (_harmonic(l, dirs) ** 2)))
            total = 0.0
            for n, w in zip(dirs, wts):
                p = r * n
                g = np.array([_fd(lambda q: psi(q), p, i, 1e-6)
                              for i in range(3)])
                total += w * float(g @ g)
            got = total / (norm * norm)
            ang = float(sp.sympify(RADIAL.angular).subs(
                {L_FACTOR: l * (l + 1),
                 sp.Symbol("r", real=True, positive=True): r}))
            want = dR(r) ** 2 + ang * R(r) ** 2
            err = abs(got - want) / max(abs(want), 1e-12)
            out.append((f"radial: <|grad psi|^2> for l={l} "
                        f"= R'^2 + l(l+1) R^2/r^2", err < 1e-5, err))
    return out


def check_purem_reduction(npts=5, tol=1e-6):
    """tau of the pure-m diatomic reduction.

    For psi = f(mu, nu) exp(i m phi) the azimuthal derivative is analytic
    and contributes m^2 |psi|^2 / h_phi^2 -- the ``angular`` entry of
    PROLATE_PUREM.  Checked against the full complex Cartesian gradient.
    """
    out = []
    for m in (0, 1, 2):
        for _ in range(npts):
            a1, a2 = RNG.uniform(0.3, 1.0, size=2)
            mu0, nu0 = RNG.uniform(1.2, 2.0), RNG.uniform(-0.7, 0.7)
            ph0 = RNG.uniform(0.0, 6.2)

            def f2d(mu, nu):
                return np.exp(-a1 * (mu - 1.4) ** 2 - a2 * nu * nu)

            def psi(xyz):
                mu, nu, ph = xyz_to_prolate(xyz)
                return f2d(mu, nu) * np.exp(1j * m * ph)

            p = prolate_to_xyz(np.array([mu0, nu0, ph0]))
            g = np.array([_fd(psi, p, i, 1e-5) for i in range(3)])
            want = float(np.real(np.vdot(g, g)))

            h = prolate_scales(np.array([mu0, nu0, ph0]))
            dmu = (f2d(mu0 + STEP, nu0) - f2d(mu0 - STEP, nu0)) / (2 * STEP)
            dnu = (f2d(mu0, nu0 + STEP) - f2d(mu0, nu0 - STEP)) / (2 * STEP)
            ang = float(sp.sympify(PROLATE_PUREM.angular).subs(
                {M_FACTOR: m * m,
                 sp.Symbol("scale_phi", real=True, positive=True): h[2]}))
            got = (dmu / h[0]) ** 2 + (dnu / h[1]) ** 2 \
                + ang * f2d(mu0, nu0) ** 2
            err = abs(got - want) / max(abs(want), 1e-12)
            out.append((f"prolate_purem: |grad psi|^2 for m={m} carries "
                        f"m^2 |psi|^2 / h_phi^2", err < 1e-4, err))
    return out


def main():
    checks = []
    checks += check_full_system("spherical", SPHERICAL, spherical_to_xyz,
                                spherical_scales, tol=1e-6)
    checks += check_full_system("prolate", PROLATE, prolate_to_xyz,
                                prolate_scales, tol=1e-6)
    checks += check_radial_reduction()
    checks += check_purem_reduction()
    ok = _report(checks)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
