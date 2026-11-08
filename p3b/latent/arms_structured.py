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
from p3b.latent.integrate import RANK_TOL


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
