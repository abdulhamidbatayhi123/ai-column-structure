"""Unstructured arms U, U+F and P, and the parameter matcher (PREREG 3 and 4; Plan 2, Task 5).

U:   dz/dt = f_theta(z, u), f an MLP (d + k) -> w -> w -> w -> d, SiLU (w = 128 for the reference
     architecture; matched to S's parameter count within +-5 %).
U+F: f also receives E^, N^, S^, P_E, P_N, Phi_S and grad E^, grad N^, grad S^ (B: no N terms),
     with no constraint.
P:   U trained with soft penalties on the water and energy balance residuals and on negative
     total entropy production of the decoded state; one weight, fixed so that the penalty equals
     1 % of the data loss at initialisation. Each term is divided by a reference scale the caller
     supplies (the normalisations of the violation metric, PREREG 5), so the terms are dimensionless.
"""
from __future__ import annotations

import torch
from torch import nn

from p3b.latent.autoencoder import _mlp, zero_last

F64 = torch.float64


class UDynamics(nn.Module):
    """dz/dt = f_theta(z, u) / time_scale: the network works in units of the reference time
    (the training set's median snapshot interval), the integrator in seconds."""

    def __init__(self, d: int, k: int, width: int = 128, time_scale: float = 1.0):
        super().__init__()
        self.net = zero_last(_mlp([d + k, width, width, width, d]))
        self.time_scale = float(time_scale)

    def forward(self, z, feat):
        return self.net(torch.cat([z, feat], dim=-1)) / self.time_scale


class UFDynamics(nn.Module):
    """U with the physical functionals, port fluxes and latent gradients as extra inputs."""

    SCALARS = {"A": ("E", "N", "S", "P_E", "P_N", "Phi_S"), "B": ("E", "S", "P_E", "Phi_S")}
    GRADS = {"A": ("dE", "dN", "dS"), "B": ("dE", "dS")}

    def __init__(self, d: int, k: int, width: int, system: str, time_scale: float = 1.0):
        super().__init__()
        self.system, self.time_scale = system, float(time_scale)
        n_extra = len(self.SCALARS[system]) + len(self.GRADS[system]) * d
        self.net = zero_last(_mlp([d + k + n_extra, width, width, width, d]))

    def forward(self, z, feat, terms: dict):
        """`terms`: LatentFunctionals.evaluate output. Scalars should be standardised by the caller."""
        s = torch.stack([terms[n] for n in self.SCALARS[self.system]], dim=-1)
        g = torch.cat([terms[n] for n in self.GRADS[self.system]], dim=-1)
        return self.net(torch.cat([z, feat, s, g], dim=-1)) / self.time_scale


def count_params(module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def match_width(build, target: int, tol: float = 0.05, lo: int = 4, hi: int = 4096) -> int:
    """The width whose parameter count is closest to `target`; ValueError if not within tol.
    Candidate modules are built on the meta device, so no parameter memory is allocated."""
    _build = build

    def build(w):
        with torch.device("meta"):
            return _build(w)

    best = min(range(lo, hi + 1), key=lambda w: abs(count_params(build(w)) - target)) if hi - lo < 64 else None
    if best is None:  # parameter count grows monotonically with width: bisect, then check neighbours
        a, b = lo, hi
        while b - a > 1:
            m = (a + b) // 2
            if count_params(build(m)) < target:
                a = m
            else:
                b = m
        best = min((a, b), key=lambda w: abs(count_params(build(w)) - target))
    if abs(count_params(build(best)) - target) > tol * target:
        raise ValueError(f"no width within {tol:.0%} of {target} parameters (closest {best})")
    return best


def penalty_weight(data_loss0: float, penalty0: float) -> float:
    """The weight that makes the penalty 1 % of the data loss at initialisation."""
    return 0.01 * data_loss0 / penalty0


def penalty_loss(rN, rE, sigma, scales):
    """Mean-square water (A) and energy residuals and negative production, each over its scale."""
    sN, sE, sS = scales
    p = (rE / sE).pow(2).mean() + (torch.relu(-sigma) / sS).pow(2).mean()
    if rN is not None:
        p = p + (rN / sN).pow(2).mean()
    return p
