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

import torch
from torch import nn

from p3b.latent.autoencoder import _mlp
from p3b.latent.integrate import RANK_TOL, balance_project, rk4_step, stage_quadrature


def structure_terms(B, dS, G, A, targets, phi, kappa: float, rank_tol: float = RANK_TOL,
                    metriplectic: bool = True, port_projection: bool = True):
    """The S vector field from its parts. B (n, d, m); dS, G (n, d); A (n, d, d) or None;
    targets (n, m) port fluxes in B's column order; phi (n,) the entropy port flux."""
    n, d, m = B.shape
    s = torch.linalg.svdvals(B)
    rank_def = (s > rank_tol * s[..., :1]).sum(-1) < m
    Bp = torch.linalg.pinv(B, rtol=rank_tol)                     # (n, m, d)
    eye = torch.eye(d, dtype=B.dtype).expand(n, d, d)
    P = eye - B @ Bp
    g = (P @ dS.unsqueeze(-1)).squeeze(-1)                       # P grad S
    g2 = (g * g).sum(-1)
    out = torch.zeros_like(dS)
    if metriplectic:
        Atg = (A.transpose(-1, -2) @ g.unsqueeze(-1))
        out = out + (P @ (A @ Atg)).squeeze(-1)
    deficit = torch.zeros(n, dtype=B.dtype)
    if port_projection:
        BT = B.transpose(-1, -2)                                  # (n, m, d)
        r = targets - (BT @ G.unsqueeze(-1)).squeeze(-1)
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
                 metriplectic: bool = True, port_projection: bool = True):
        super().__init__()
        self.d, self.lf, self.kappa, self.rank_tol = d, lf, float(kappa), float(rank_tol)
        self.metriplectic, self.port_projection = metriplectic, port_projection
        self.G = _mlp([d + k, width, width, width, d])
        self.A_net = _mlp([d, width, d * d]) if metriplectic else None

    def forward(self, z, feat, raw):
        t = self.lf.evaluate(z, raw, grads=True, create_graph=torch.is_grad_enabled())
        if self.lf.system == "A":
            B = torch.stack([t["dE"], t["dN"]], dim=-1)
            targets = torch.stack([t["P_E"], t["P_N"]], dim=-1)
        else:
            B = t["dE"].unsqueeze(-1)
            targets = t["P_E"].unsqueeze(-1)
        G = self.G(torch.cat([z, feat], dim=-1))
        A = self.A_net(z).reshape(-1, self.d, self.d) if self.metriplectic else None
        return structure_terms(B, t["dS"], G, A, targets, t["Phi_S"], self.kappa, self.rank_tol,
                               self.metriplectic, self.port_projection)


class SLDynamics(nn.Module):
    """Arm S-L: the metriplectic term uses learned potentials E_theta, S_theta (d -> 64 -> 64 -> 1),
    degenerate with respect to grad E_theta and grad N^ (system B: grad E_theta only); the port
    term is projected only on the physical water balance (system B: not at all), because a learned
    energy has no physical port flux. The tLaSDI/GFINN-style control for fixed physical potentials."""

    def __init__(self, d: int, k: int, lf, rank_tol: float = RANK_TOL, width: int = 128, pot_width: int = 64):
        super().__init__()
        self.d, self.lf, self.rank_tol = d, lf, float(rank_tol)
        self.G = _mlp([d + k, width, width, width, d])
        self.A_net = _mlp([d, width, d * d])
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
        dE, dS = self.potential_grads(z)
        A_sys = self.lf.system == "A"
        t = self.lf.evaluate(z, raw, grads=A_sys, create_graph=torch.is_grad_enabled())
        BL = torch.stack([dE, t["dN"]], -1) if A_sys else dE.unsqueeze(-1)
        A = self.A_net(z).reshape(-1, self.d, self.d)
        M, info = structure_terms(BL, dS, torch.zeros_like(z), A, None, None, kappa=0.0,
                                  rank_tol=self.rank_tol, port_projection=False)
        G = self.G(torch.cat([z, feat], dim=-1))
        if A_sys:
            dN = t["dN"].unsqueeze(-2)                                # (n, 1, d)
            r = t["P_N"] - (t["dN"] * G).sum(-1)
            G = G + (torch.linalg.pinv(dN, rtol=self.rank_tol) @ r[:, None, None]).squeeze(-1)
        info["metriplectic"] = M
        return M + G, info


def projected_rk4_step(f, z, raw, h, lf, rank_tol: float = RANK_TOL, iters: int = 20, create_graph: bool = False):
    """RK4, then the Gauss-Newton projection onto N^ = N^_prev + Q[P_N], E^ = E^_prev + Q[P_E]
    (formulation 5.4), with the port integrals taken on the same four RK4 stages. System B: E only."""
    z_new, stages = rk4_step(f, z, h)
    vals = [lf.evaluate(s, raw, grads=False) for s in stages]
    v0 = vals[0]
    names = ["E", "N"] if lf.system == "A" else ["E"]
    ports = {"E": "P_E", "N": "P_N"}
    Q = {n: stage_quadrature([v[ports[n]] for v in vals], h) for n in names}
    target = torch.stack([v0[n] + Q[n] for n in names], -1)

    def cons(x):
        v = lf.evaluate(x, raw, grads=False)
        return torch.stack([v[n] for n in names], -1)

    z_new, info = balance_project(z_new, cons, target, iters=iters, rank_tol=rank_tol, create_graph=create_graph)
    info.update({"Q_" + n: Q[n] for n in names})
    return z_new, info


class ProjectedRollout(nn.Module):
    """U+proj and U+F+proj: a trained U or U+F rolled out with S's projection after every step.
    It owns no parameters, so nothing is retrained. `dyn(z, feat, raw) -> dz/dt`."""

    def __init__(self, dyn, lf, rank_tol: float = RANK_TOL, iters: int = 20):
        super().__init__()
        self._dyn, self.lf, self.rank_tol, self.iters = dyn, lf, rank_tol, iters

    def step(self, z, feat, raw, h):
        return projected_rk4_step(lambda x: self._dyn(x, feat, raw), z, raw, h, self.lf, self.rank_tol, self.iters)
