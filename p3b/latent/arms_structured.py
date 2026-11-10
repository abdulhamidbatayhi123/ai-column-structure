"""Structured arms: S and its ablations S-M and S-Pi (formulation 5.3; Plan 2, Task 6).

    dz/dt = P A A^T P grad S^  +  Pi[G_theta(z, u)]
    P = I - B pinv(B),  B = [grad E^, grad N^]  (system B: B = [grad E^])
    Pi[G]: minimum-norm change of G with B^T Pi = (P_E, P_N), then the inequality correction
           Pi <- Pi + lam P grad S^,  lam = max(0, delta) / (|P grad S^|^2 + kappa),
           delta = Phi_S - grad S^ . Pi.

Because P grad S^ is orthogonal to the columns of B, the correction keeps both equalities, and
grad S^ . P grad S^ = |P grad S^|^2, so the residual deficit after it is delta kappa / (|P grad S^|^2
+ kappa). It is counted, with "kappa binds" (|P grad S^|^2 < 10 kappa) and the rank-deficient
steps (fewer than m singular values of B above rank_tol times the largest): defects, never
successes (hypothesis H3a). kappa, rank_tol are arguments (pilot values, PREREG 10).
"""
from __future__ import annotations

import math

import torch
from torch import nn

from p3b.latent.autoencoder import _mlp, small_last, zero_last
from p3b.latent.integrate import RANK_TOL, balance_project, rank_deficient, rk4_step, stage_quadrature


def structure_terms(B, dS, G, A, targets, phi, kappa: float, rank_tol: float = RANK_TOL,
                    metriplectic: bool = True, port_projection: bool = True):
    """The S vector field from its parts. B (n, d, m); dS, G (n, d); A (n, d, d) or None;
    targets (n, m) port fluxes in B's column order; phi (n,) the entropy port flux.

    The columns of B (and the targets with them) are scaled to unit norm first: grad E^ and
    grad N^ differ by about |dH| ~ 5e4 in units, and the relative singular-value tolerance must
    measure their angle, not their units. P, Pi and the deficit are unchanged in exact arithmetic."""
    n, d, m = B.shape
    norms = B.norm(dim=-2, keepdim=True)
    norms = torch.where(norms > 0, norms, torch.ones_like(norms))
    Bs = B / norms
    rank_def = rank_deficient(Bs.transpose(-1, -2), rank_tol)
    Bp = torch.linalg.pinv(Bs, rtol=rank_tol)                    # (n, m, d)
    eye = torch.eye(d, dtype=B.dtype).expand(n, d, d)
    P = eye - Bs @ Bp
    g = (P @ dS.unsqueeze(-1)).squeeze(-1)                       # P grad S
    g2 = (g * g).sum(-1)
    out = torch.zeros_like(dS)
    if metriplectic:
        Atg = (A.transpose(-1, -2) @ g.unsqueeze(-1))
        out = out + (P @ (A @ Atg)).squeeze(-1)
    deficit = torch.zeros(n, dtype=B.dtype)
    if port_projection:
        BT = Bs.transpose(-1, -2)                                 # (n, m, d)
        r = targets / norms.squeeze(-2) - (BT @ G.unsqueeze(-1)).squeeze(-1)
        Pi = G + (torch.linalg.pinv(BT, rtol=rank_tol) @ r.unsqueeze(-1)).squeeze(-1)
        delta = phi - (dS * Pi).sum(-1)
        lam = torch.clamp(delta, min=0.0) / (g2 + kappa)
        Pi = Pi + lam.unsqueeze(-1) * g
        deficit = torch.clamp(delta, min=0.0) * kappa / (g2 + kappa)
        out = out + Pi
    else:
        out = out + G
    info = {"rank_deficient": rank_def, "kappa_binds": g2 < 10 * kappa, "residual_deficit": deficit,
            "P_gradS_sq": g2}
    return out, info


class SDynamics(nn.Module):
    """Arm S (and S-M with metriplectic=False, S-Pi with port_projection=False)."""

    def __init__(self, d: int, k: int, lf, kappa: float, rank_tol: float = RANK_TOL, width: int = 128,
                 metriplectic: bool = True, port_projection: bool = True, time_scale: float = 1.0):
        super().__init__()
        self.d, self.lf, self.kappa, self.rank_tol = d, lf, float(kappa), float(rank_tol)
        # G / tau and A / sqrt(tau g): networks in reference-time units; buffers, so they are saved
        self.register_buffer("time_scale", torch.tensor(float(time_scale), dtype=torch.float64))
        self.register_buffer("grad_scale", torch.tensor(math.sqrt(1e3 * float(kappa)), dtype=torch.float64))
        self.metriplectic, self.port_projection = metriplectic, port_projection
        self.G = zero_last(_mlp([d + k, width, width, width, d]))
        self.A_net = small_last(_mlp([d, width, d * d])) if metriplectic else None

    def forward(self, z, feat, raw):
        t = self.lf.evaluate(z, raw, grads=True)
        B, targets = _physical_b(t, self.lf.system)
        G = self.G(torch.cat([z, feat], dim=-1)) / self.time_scale
        # A / sqrt(tau g): with g = sqrt(1e3 kappa), the pilot median |P grad S^| that kappa comes from,
        # P A A^T P grad S^ has the scale 1/tau whatever the units of entropy (a positive constant: the
        # bracket's symmetry, semidefiniteness and degeneracy are unchanged)
        A = (self.A_net(z).reshape(-1, self.d, self.d) / (self.time_scale * self.grad_scale) ** 0.5
             if self.metriplectic else None)
        out, info = structure_terms(B, t["dS"], G, A, targets, t["Phi_S"], self.kappa, self.rank_tol,
                                    self.metriplectic, self.port_projection)
        info["values"] = t                    # reused as the stage values of the projected step
        return out, info


class SLDynamics(nn.Module):
    """Arm S-L: as S, but the metriplectic term uses learned potentials E_theta, S_theta
    (d -> 64 -> 64 -> 1), degenerate with respect to grad E_theta and grad N^ (system B: grad
    E_theta). The port term is S's own Pi[G_theta] on the physical E^, N^, S^ (PREREG Section 3,
    revised after cs T5 F3), so S and S-L differ in the bracket's potentials alone."""

    def __init__(self, d: int, k: int, lf, kappa: float, rank_tol: float = RANK_TOL, width: int = 128,
                 pot_width: int = 64, time_scale: float = 1.0):
        super().__init__()
        self.d, self.lf, self.kappa, self.rank_tol = d, lf, float(kappa), float(rank_tol)
        self.register_buffer("time_scale", torch.tensor(float(time_scale), dtype=torch.float64))
        self.G = zero_last(_mlp([d + k, width, width, width, d]))
        self.A_net = small_last(_mlp([d, width, d * d]))
        self.E_net = _mlp([d, pot_width, pot_width, 1])
        self.S_net = _mlp([d, pot_width, pot_width, 1])

    def potential_grads(self, z):
        """grad E_theta, grad S_theta; kept in the graph when gradients are enabled (training)."""
        keep = torch.is_grad_enabled()
        with torch.enable_grad():
            zz = z if z.requires_grad else z.detach().requires_grad_(True)
            dE = torch.autograd.grad(self.E_net(zz).sum(), zz, create_graph=keep)[0]
            dS = torch.autograd.grad(self.S_net(zz).sum(), zz, create_graph=keep)[0]
        return (dE, dS) if keep else (dE.detach(), dS.detach())

    def forward(self, z, feat, raw):
        dE_th, dS_th = self.potential_grads(z)
        t = self.lf.evaluate(z, raw, grads=True)
        A_sys = self.lf.system == "A"
        BL = torch.stack([dE_th, t["dN"]], -1) if A_sys else dE_th.unsqueeze(-1)
        A = self.A_net(z).reshape(-1, self.d, self.d) / self.time_scale ** 0.5
        M, info_m = structure_terms(BL, dS_th, torch.zeros_like(z), A, None, None, kappa=1.0,
                                    rank_tol=self.rank_tol, port_projection=False)
        G = self.G(torch.cat([z, feat], dim=-1)) / self.time_scale
        B, targets = _physical_b(t, self.lf.system)
        Pi, info = structure_terms(B, t["dS"], G, None, targets, t["Phi_S"], self.kappa, self.rank_tol,
                                   metriplectic=False)
        info["bracket_rank_deficient"] = info_m["rank_deficient"]
        info["metriplectic"] = M
        info["values"] = t
        return M + Pi, info


def _physical_b(t, system):
    if system == "A":
        return torch.stack([t["dE"], t["dN"]], dim=-1), torch.stack([t["P_E"], t["P_N"]], dim=-1)
    return t["dE"].unsqueeze(-1), t["P_E"].unsqueeze(-1)


def projected_rk4_step(f, z, raw, h, lf, rank_tol: float = RANK_TOL, iters: int = 20, scale=None,
                       stage_values=None):
    """RK4, then the Gauss-Newton projection onto N^ = N^_prev + Q[P_N], E^ = E^_prev + Q[P_E]
    (formulation 5.4), with the port integrals taken on the same four RK4 stages. System B: E only.
    `scale` (batch, m): the physical scale of each balance (PREREG 5), default |target|. The step is
    differentiable when gradients are enabled (see integrate.balance_project). `stage_values`: a
    list that f fills with lf's values at each stage it evaluates (the structured arms do), reused
    instead of evaluating the decoder again at the same four states."""
    z_new, stages = rk4_step(f, z, h)
    if stage_values is not None and len(stage_values) == 4:
        vals = stage_values
    else:
        vals = [lf.evaluate(s_, raw, grads=False) for s_ in stages]
    v0 = vals[0]
    names = ["E", "N"] if lf.system == "A" else ["E"]
    ports = {"E": "P_E", "N": "P_N"}
    Q = {n: stage_quadrature([v[ports[n]] for v in vals], h) for n in names}
    target = torch.stack([v0[n] + Q[n] for n in names], -1)

    def cons(x):
        v = lf.evaluate(x, raw, grads=False)
        return torch.stack([v[n] for n in names], -1)

    z_new, info = balance_project(z_new, cons, target, iters=iters, rank_tol=rank_tol, scale=scale)
    info.update({"Q_" + n: Q[n] for n in names})
    info["Q_S"] = stage_quadrature([v["Phi_S"] for v in vals], h)       # entropy port, same stages
    info["E0"], info["S0"] = v0["E"], v0["S"]
    if "N" in v0:
        info["N0"] = v0["N"]
    return z_new, info


def structured_step(dyn, z, feat, raw, h, rank_tol: float = RANK_TOL, iters: int = 20, scale=None):
    """One projected RK4 step of S, S-M, S-Pi or S-L, keeping the defect flags of all four stages
    (a flag counts if it is raised at any stage) and of the projection: the H3a counts."""
    infos, values = [], []

    def f(x):
        dz, info = dyn(x, feat, raw)
        values.append(info.pop("values"))
        infos.append(info)
        return dz

    z_new, pinfo = projected_rk4_step(f, z, raw, h, dyn.lf, rank_tol, iters, scale, stage_values=values)
    out = {"rank_deficient": torch.stack([i["rank_deficient"] for i in infos]).any(0),
           "kappa_binds": torch.stack([i["kappa_binds"] for i in infos]).any(0),
           "residual_deficit": torch.stack([i["residual_deficit"].detach() for i in infos]).amax(0),
           "projection_rank_deficient": pinfo["rank_deficient"],
           "projection_converged": pinfo["converged"],
           "projection_residual": pinfo["residual"]}
    out.update({k: pinfo[k] for k in ("Q_E", "Q_N", "Q_S", "E0", "N0", "S0") if k in pinfo})
    return z_new, out


class ProjectedRollout(nn.Module):
    """U+proj and U+F+proj: a trained U or U+F rolled out with S's projection after every step.
    It owns no parameters, so nothing is retrained. `dyn(z, feat, raw) -> dz/dt`."""

    def __init__(self, dyn, lf, rank_tol: float = RANK_TOL, iters: int = 20):
        super().__init__()
        self._dyn, self.lf, self.rank_tol, self.iters = dyn, lf, rank_tol, iters

    def step(self, z, feat, raw, h):
        return projected_rk4_step(lambda x: self._dyn(x, feat, raw), z, raw, h, self.lf, self.rank_tol, self.iters)
