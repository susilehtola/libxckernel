"""Phase 0 of the GPAW pilot: reproduce GPAW's semilocal XC stress
contribution from xckernel's strain seeds, on GPAW's own arrays.

For each functional family, GPAW's Functional._args gives the exact
field/derivative arrays its _stress contracts; we feed the same arrays
to the symbolic strain_energy_derivative and require elementwise
agreement of the 3x3 XC stress contribution (before GPAW's central
symmetrization and 1/V normalization).

Operand map (spin-paired):
  rho          = nt_sr[0]            zk   = e_r / max(rho, tiny)
  vrho         = vt_sr[0]            sigma = sigma_xr[0]
  vsigma       = dedsigma_xr[0]      grad_rho_i = gradn_svr[0, i]
  tau          = taut_sr[0]          vtau = dedtaut_sr[0]
  tau_tensor_w = interpolate(taut_swR[0, w])
(GPAW's KED tensor carries no 1/2; the convention is declared through
scale_operands({"tau_tensor": 2.0}) rather than hand-scaled arrays.)
"""
try:
    import gpaw  # noqa: F401
except ImportError:
    print("[SKIP] gpaw_stress_validate: GPAW not available")
    raise SystemExit(0)

import numpy as np
import sympy as sp

from ..engine.strain import (scale_operands,  # noqa: E402
                             strain_energy_derivative)
from ..inputs.basis import AXES, HESS_COMPS  # noqa: E402

from ase.build import bulk  # noqa: E402
from gpaw import GPAW, PW  # noqa: E402

TINY = 1e-40


def run(xcname, family, ecut=340, shear=0.02):
    atoms = bulk('Si', 'diamond', a=5.43)
    strain = np.eye(3)
    strain[0, 1] = shear
    strain[1, 2] = -0.6 * shear
    atoms.set_cell(atoms.cell[:] @ strain.T, scale_atoms=True)
    calc = GPAW(mode=PW(ecut), xc=xcname, kpts=(2, 2, 2),
                symmetry='off', txt=f'{family}.txt',
                convergence={'density': 1e-8})
    atoms.calc = calc
    atoms.get_potential_energy()

    dft = calc.dft
    pot_calc = dft.pot_calc
    xc = pot_calc.xc
    interpolate = pot_calc.interpolate

    args, kwargs = xc._args(dft.ibzwfs, dft.density, interpolate)
    xc.xc.kernel.calculate(*[a.data for a in args])
    s_gpaw = np.array(xc._stress(*args, **kwargs))

    # unpack GPAW's arrays into xckernel operand values
    if family == 'lda':
        e_r, nt_sr, vt_sr = args
    elif family == 'gga':
        e_r, nt_sr, vt_sr, sigma_xr, dedsigma_xr = args
    else:
        e_r, nt_sr, vt_sr, sigma_xr, dedsigma_xr, taut_sr, dedtaut_sr = args

    rho = nt_sr.data[0]
    vals = {'rho': rho, 'zk': e_r.data / np.maximum(rho, TINY),
            'vrho': vt_sr.data[0]}
    if family != 'lda':
        gradn_svr = kwargs['gradn_svr']
        vals['sigma'] = sigma_xr.data[0]
        vals['vsigma'] = dedsigma_xr.data[0]
        for i, ax in enumerate(AXES):
            vals[f'grad_rho_{ax}'] = gradn_svr.data[0, i]
    if family == 'mgga_tau':
        vals['tau'] = taut_sr.data[0]
        vals['vtau'] = dedtaut_sr.data[0]
        taut_swR = kwargs['taut_swR']
        for w, (i, j) in enumerate(HESS_COMPS):
            tt_r = interpolate(taut_swR[0, w])
            vals[f'tau_tensor_{AXES[i]}{AXES[j]}'] = tt_r.data

    dv = nt_sr.desc.dv
    s_ours = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            expr = scale_operands(strain_energy_derivative(family, a, b),
                                  {'tau_tensor': 2.0})
            syms = sorted(expr.free_symbols, key=lambda s_: s_.name)
            fn = sp.lambdify(syms, expr, 'numpy')
            per_point = np.broadcast_to(fn(*[vals[s_.name] for s_ in syms]),
                                        rho.shape)
            s_ours[a, b] = dv * float(per_point.sum())

    dev = np.abs(s_ours - s_gpaw).max()
    scale = np.abs(s_gpaw).max()
    print(f'{family:9s} {xcname:6s} XC stress contribution:')
    for row_o, row_g in zip(s_ours, s_gpaw):
        print('   ours ', ' '.join(f'{x:+.10f}' for x in row_o),
              '  gpaw ', ' '.join(f'{x:+.10f}' for x in row_g))
    print(f'  max |ours - gpaw| = {dev:.3e}  (scale {scale:.3e}, '
          f'rel {dev / scale:.3e})')
    return dev / scale


def main():
    worst = 0.0
    for xcname, family in [('LDA', 'lda'), ('PBE', 'gga'),
                           ('TPSS', 'mgga_tau')]:
        worst = max(worst, run(xcname, family))
    print(f'WORST relative deviation: {worst:.3e}')
    return 0 if worst < 1e-10 else 1


if __name__ == '__main__':
    raise SystemExit(main())
