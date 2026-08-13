"""Validate the explicit cell-deformation (strain) seeds on a synthetic
periodic system: plane-wave orbitals on a uniform grid in a triclinic
cell, deformed as r -> A r with coefficients and fractional grid fixed
(G -> A^{-T} G, Omega -> det A Omega).  The analytic strain derivative
assembled from xckernel.engine.strain is compared against Richardson-
extrapolated finite differences of E(A) in all nine deformation
components, for LDA / GGA / mgga_tau / mgga_lapl / mgga through pylibxc
and for the eta (density-Hessian) seed against a synthetic functional.
The antisymmetric part of dE/dA must vanish (rotational invariance);
that sum rule is checked for free."""

from __future__ import annotations

import numpy as np
import sympy as sp

from ..engine.strain import TAU_TENSOR, ZK, strain_energy_derivative
from ..engine.strain import strain_ingredient
from ..inputs.basis import AXES, HESS_COMPS
from ..inputs.ingredients import ETA_ING, GRAD_RHO, HESS_RHO

FAMILIES = {
    "lda": ["LDA_X", "LDA_C_PW"],
    "gga": ["GGA_X_PBE", "GGA_C_PBE"],
    "mgga_tau": ["MGGA_X_R2SCAN", "MGGA_C_R2SCAN"],
    "mgga_lapl": ["MGGA_X_R2SCANL", "MGGA_C_R2SCANL"],
    "mgga": ["MGGA_X_BR89"],
}

N = 12          # grid points per axis
NORB = 3        # synthetic occupied orbitals
H = 2e-3        # deformation-gradient step for finite differences


class PWModel:
    """Synthetic occupied plane-wave orbitals in a deformable cell."""

    def __init__(self, seed=7):
        rng = np.random.default_rng(seed)
        self.cell0 = np.array([[6.0, 0.3, 0.0],
                               [0.0, 5.5, 0.4],
                               [0.2, 0.0, 6.5]])
        # fractional grid, fixed under deformation
        f = np.stack(np.meshgrid(*[np.arange(N) / N] * 3,
                                 indexing="ij"), axis=-1).reshape(-1, 3)
        self.frac = f
        # a handful of low reciprocal-lattice modes per orbital, plus a
        # constant so the density is strictly positive
        self.miller = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                                [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
        nm = len(self.miller)
        self.cos_c = rng.normal(size=(NORB, nm)) * 0.15
        self.sin_c = rng.normal(size=(NORB, nm)) * 0.15
        self.cos_c[:, 0] = rng.uniform(0.8, 1.2, NORB)  # positive baseline
        self.occ = rng.uniform(0.7, 1.0, NORB)
        # a second coefficient set: pair bilinears psi*chi provide synthetic
        # perturbed fields with the exact transformation law of densities
        self.cos_p = rng.normal(size=(NORB, nm)) * 0.2
        self.sin_p = rng.normal(size=(NORB, nm)) * 0.2

    def fields(self, A: np.ndarray):
        """All grid fields in the cell deformed by A (coefficients and
        fractional grid fixed)."""
        cell = self.cell0 @ A.T                       # rows = lattice vectors
        omega = abs(np.linalg.det(cell))
        B = 2 * np.pi * np.linalg.inv(cell).T         # reciprocal rows
        G = self.miller @ B                           # (nm, 3)
        phase = 2 * np.pi * (self.frac @ self.miller.T)   # A-independent
        cph, sph = np.cos(phase), np.sin(phase)       # (ng, nm)
        norm = 1.0 / np.sqrt(omega)

        ng = len(self.frac)
        rho = np.zeros(ng)
        grad = np.zeros((3, ng))
        hess = np.zeros((6, ng))
        tau_t = np.zeros((6, ng))
        lapl = np.zeros(ng)

        for i in range(NORB):
            psi = (cph @ self.cos_c[i] + sph @ self.sin_c[i]) * norm
            dpsi = np.stack([
                (-sph * G[:, c]) @ self.cos_c[i]
                + (cph * G[:, c]) @ self.sin_c[i]
                for c in range(3)]) * norm            # (3, ng)
            d2psi = np.stack([
                (-cph * G[:, a] * G[:, b]) @ self.cos_c[i]
                + (-sph * G[:, a] * G[:, b]) @ self.sin_c[i]
                for (a, b) in HESS_COMPS]) * norm     # (6, ng)
            f = self.occ[i]
            rho += f * psi * psi
            grad += 2 * f * psi * dpsi
            for k, (a, b) in enumerate(HESS_COMPS):
                hess[k] += 2 * f * (psi * d2psi[k] + dpsi[a] * dpsi[b])
                tau_t[k] += 0.5 * f * dpsi[a] * dpsi[b]
        lapl = hess[0] + hess[3] + hess[5]
        tau = tau_t[0] + tau_t[3] + tau_t[5]
        sigma = (grad ** 2).sum(axis=0)
        w = omega / ng
        return {"w": w, "rho": rho, "grad": grad, "sigma": sigma,
                "lapl": lapl, "tau": tau, "hess": hess, "tau_tensor": tau_t}


    def pert_fields(self, A: np.ndarray):
        """Perturbed-density-like pair fields rho_p = sum f psi chi with
        their gradient, tau_p, and tau_p tensor."""
        cell = self.cell0 @ A.T
        omega = abs(np.linalg.det(cell))
        B = 2 * np.pi * np.linalg.inv(cell).T
        G = self.miller @ B
        phase = 2 * np.pi * (self.frac @ self.miller.T)
        cph, sph = np.cos(phase), np.sin(phase)
        norm = 1.0 / np.sqrt(omega)

        ng = len(self.frac)
        rho_p = np.zeros(ng)
        grad_p = np.zeros((3, ng))
        tau_t_p = np.zeros((6, ng))
        for i in range(NORB):
            psi = (cph @ self.cos_c[i] + sph @ self.sin_c[i]) * norm
            chi = (cph @ self.cos_p[i] + sph @ self.sin_p[i]) * norm
            dpsi = np.stack([(-sph * G[:, c]) @ self.cos_c[i]
                             + (cph * G[:, c]) @ self.sin_c[i]
                             for c in range(3)]) * norm
            dchi = np.stack([(-sph * G[:, c]) @ self.cos_p[i]
                             + (cph * G[:, c]) @ self.sin_p[i]
                             for c in range(3)]) * norm
            f = self.occ[i]
            rho_p += f * psi * chi
            grad_p += f * (dpsi * chi + psi * dchi)
            for k, (a, b) in enumerate(HESS_COMPS):
                tau_t_p[k] += 0.25 * f * (dpsi[a] * dchi[b]
                                          + dpsi[b] * dchi[a])
        tau_p = tau_t_p[0] + tau_t_p[3] + tau_t_p[5]
        return {"rho_p": rho_p, "grad_p": grad_p, "tau_p": tau_p,
                "tau_t_p": tau_t_p}


def libxc_eval(func_ids, family, F, orders=(0, 1)):
    """Evaluate pylibxc energy/derivative arrays on a field dict."""
    import pylibxc
    inp = {"rho": F["rho"]}
    if family != "lda":
        inp["sigma"] = F["sigma"]
    if family in ("mgga", "mgga_lapl"):
        inp["lapl"] = F["lapl"]
    if family in ("mgga", "mgga_tau"):
        inp["tau"] = F["tau"]
    if family in ("mgga_lapl",):
        inp["tau"] = F["tau"]     # libxc mgga interface wants tau anyway
    out = {}
    for fid in func_ids:
        f = pylibxc.LibXCFunctional(fid, "unpolarized")
        r = f.compute(inp, do_exc=0 in orders, do_vxc=1 in orders)
        for k, v in r.items():
            out[k] = out.get(k, 0) + v.reshape(v.shape[0], -1).T.squeeze()
    return out


def energy(model, func_ids, family, A):
    F = model.fields(A)
    out = libxc_eval(func_ids, family, F, orders=(0,))
    return F["w"] * float(np.dot(F["rho"], out["zk"]))


def operand_values(F, out):
    """Numeric values for every symbol a strain integrand can contain."""
    vals = {"rho": F["rho"], "sigma": F["sigma"], "lapl_rho": F["lapl"],
            "tau": F["tau"], "zk": out.get("zk", 0)}
    for i, ax in enumerate(AXES):
        vals[f"grad_rho_{ax}"] = F["grad"][i]
    for k, (i, j) in enumerate(HESS_COMPS):
        vals[f"hess_rho_{AXES[i]}{AXES[j]}"] = F["hess"][k]
        vals[f"tau_tensor_{AXES[i]}{AXES[j]}"] = F["tau_tensor"][k]
    for name in ("vrho", "vsigma", "vlapl", "vtau"):
        if name in out:
            vals[name] = out[name]
    return vals


def analytic_dE(model, func_ids, family, a, b):
    F = model.fields(np.eye(3))
    out = libxc_eval(func_ids, family, F, orders=(0, 1))
    expr = strain_energy_derivative(family, a, b)
    vals = operand_values(F, out)
    syms = sorted(expr.free_symbols, key=lambda s: s.name)
    fn = sp.lambdify(syms, expr, "numpy")
    per_point = fn(*[vals[s.name] for s in syms])
    return F["w"] * float(np.sum(per_point * np.ones_like(F["rho"])))


def fd_dE(model, func_ids, family, a, b):
    """5-point Richardson finite difference in deformation component ab."""
    def E(t):
        A = np.eye(3)
        A[a, b] += t
        return energy(model, func_ids, family, A)
    return (8 * (E(H) - E(-H)) - (E(2 * H) - E(-2 * H))) / (12 * H)


def validate_family(model, family, func_ids):
    errs = []
    for a in range(3):
        for b in range(3):
            ana = analytic_dE(model, func_ids, family, a, b)
            fd = fd_dE(model, func_ids, family, a, b)
            errs.append(abs(ana - fd))
    scale = max(1.0, max(abs(fd_dE(model, func_ids, family, i, i))
                         for i in range(3)))
    rel = max(errs) / scale
    ok = rel < 1e-8
    print(f"  [{'OK ' if ok else 'FAIL'}] {family:10s} "
          f"({'+'.join(func_ids)}): max |ana - fd| / scale = {rel:.2e}")
    return 0 if ok else 1


def validate_eta(model):
    """The eta/hess seeds against a synthetic functional e = eta."""
    F0 = model.fields(np.eye(3))

    def eta_of(F):
        g, h = F["grad"], F["hess"]
        tot = np.zeros_like(F["rho"])
        for k, (i, j) in enumerate(HESS_COMPS):
            tot += (1 if i == j else 2) * g[i] * g[j] * h[k]
        return tot

    fails = 0
    for a in range(3):
        for b in range(3):
            expr = strain_ingredient(ETA_ING, a, b)
            vals = operand_values(F0, {})
            syms = sorted(expr.free_symbols, key=lambda s: s.name)
            fn = sp.lambdify(syms, expr, "numpy")
            # volume term of E = sum w * eta: +delta_ab w eta
            vol = (1 if a == b else 0) * eta_of(F0)
            ana = F0["w"] * float(np.sum(
                np.broadcast_to(fn(*[vals[s.name] for s in syms]),
                                F0["rho"].shape) + vol))

            def E(t):
                A = np.eye(3)
                A[a, b] += t
                F = model.fields(A)
                return F["w"] * float(np.sum(eta_of(F)))
            fd = (8 * (E(H) - E(-H)) - (E(2 * H) - E(-2 * H))) / (12 * H)
            if abs(ana - fd) / max(1.0, abs(fd)) > 1e-8:
                fails += 1
    print(f"  [{'OK ' if fails == 0 else 'FAIL'}] eta seed (synthetic "
          f"e = eta): 9 components")
    return fails


def validate_rotation(model, family, func_ids):
    """Antisymmetric part of dE/dA vanishes: rotational invariance."""
    worst = 0.0
    for (a, b) in ((0, 1), (0, 2), (1, 2)):
        d = abs(analytic_dE(model, func_ids, family, a, b)
                - analytic_dE(model, func_ids, family, b, a))
        worst = max(worst, d)
    ok = worst < 1e-10
    print(f"  [{'OK ' if ok else 'FAIL'}] {family:10s} rotational sum "
          f"rule: max |antisym| = {worst:.2e}")
    return 0 if ok else 1


def validate_pert(model):
    """Strain of a response-level integrand: the perturbed-field seeds
    (label carried through, tau_p1 gaining tau_tensor_p1), via the
    monomial strain operator on w (rho_p1 sigma + tau_p1 rho)."""
    from ..engine.fastpoly import from_expr, seeded_derivative, to_expr
    from ..engine.strain import strain_seed_fn
    from ..inputs.functional import Functional

    func = Functional.of_family("mgga_tau")
    sigma_prim = sum(p.symbol ** 2 for p in GRAD_RHO)
    q = sp.Symbol("w", real=True) * (
        sp.Symbol("rho_p1", real=True) * sigma_prim
        + sp.Symbol("tau_p1", real=True) * sp.Symbol("rho", real=True))

    F0 = model.fields(np.eye(3))
    P0 = model.pert_fields(np.eye(3))
    vals = operand_values(F0, {})
    vals["w"] = F0["w"]
    vals["rho_p1"] = P0["rho_p"]
    vals["tau_p1"] = P0["tau_p"]
    for i, ax in enumerate(AXES):
        vals[f"grad_rho_p1_{ax}"] = P0["grad_p"][i]
    for k, (i, j) in enumerate(HESS_COMPS):
        vals[f"tau_tensor_p1_{AXES[i]}{AXES[j]}"] = P0["tau_t_p"][k]

    fails = 0
    for a in range(3):
        for b in range(3):
            dq = to_expr(seeded_derivative(from_expr(q),
                                           strain_seed_fn(func, a, b)))
            syms = sorted(dq.free_symbols, key=lambda s: s.name)
            fn = sp.lambdify(syms, dq, "numpy")
            ana = float(np.sum(fn(*[vals[s.name] for s in syms])))

            def Q(t):
                A = np.eye(3)
                A[a, b] += t
                F = model.fields(A)
                P = model.pert_fields(A)
                return F["w"] * float(np.sum(P["rho_p"] * F["sigma"]
                                             + P["tau_p"] * F["rho"]))
            fd = (8 * (Q(H) - Q(-H)) - (Q(2 * H) - Q(-2 * H))) / (12 * H)
            if abs(ana - fd) / max(1.0, abs(fd)) > 1e-8:
                fails += 1
    print(f"  [{'OK ' if fails == 0 else 'FAIL'}] perturbed-field seeds "
          f"(w (rho_p1 sigma + tau_p1 rho)): 9 components")
    return fails


def main():
    model = PWModel()
    failures = 0
    for family, func_ids in FAMILIES.items():
        failures += validate_family(model, family, func_ids)
        failures += validate_rotation(model, family, func_ids)
    failures += validate_eta(model)
    failures += validate_pert(model)
    status = "OK " if failures == 0 else "FAIL"
    print(f"[{status}] strain_validate: {len(FAMILIES) * 2 + 2} checks, "
          f"{failures} failures")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
