"""Arm C+L: the consistent hybrid scheme at 32 cells with learned non-negative multipliers
(PREREG Section 3; formulation 4.2; Plan 2, Task 10).

System A: multipliers on eps D_L,eff and k_z,eff (per face, from the face mean of the local inputs)
and on k_LDF (per cell). System B: on k_f,eff (per face) and the exchange coefficient h_v (per
cell). Each multiplier is the softplus of an MLP 3 -> 32 -> 32 -> 1 (SiLU) of the cell state; the
last layer starts at weight 0 and bias ln(e - 1), so every multiplier starts at exactly 1 and C+L
starts as C. Scaling a non-negative Onsager coefficient by a non-negative factor keeps every block
positive semi-definite, so C+L is consistent by construction (formulation 4.2); `production`
checks it mechanism by mechanism.

The right-hand sides are torch float64 transcriptions of `p3b.column.Column.rhs` (hybrid scheme)
and `p3b.thermal_store.ThermalStore.rhs`, checked against them to 1e-12. Implicit Euler is solved
by Newton; its gradient comes from one Newton correction with a detached Jacobian at the converged
point, y = y* - (I - h J)^-1 (y* - y0 - h f(y*)), which equals y* in value and gives exactly the
implicit-function-theorem gradient with respect to y0 and the multiplier weights.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from p3b.column import C_REF, R, T0
from p3b.latent.autoencoder import _mlp
from p3b.latent.physics import Q_torch
from p3b.thermal_store import T0 as T0_B

F64 = torch.float64
_INV_SOFTPLUS_1 = math.log(math.e - 1.0)


class Multipliers(nn.Module):
    """n_out independent non-negative multipliers of 3 local inputs; `constant` fixes them (tests)."""

    def __init__(self, n_out: int, n_in: int = 3, width: int = 32, constant: float | None = None):
        super().__init__()
        self.n_out, self.constant = n_out, constant
        self.nets = nn.ModuleList(_mlp([n_in, width, width, 1]) for _ in range(n_out))
        for net in self.nets:
            nn.init.zeros_(net[-1].weight)
            nn.init.constant_(net[-1].bias, _INV_SOFTPLUS_1)

    def forward(self, x):
        if self.constant is not None:
            return torch.full(x.shape[:-1] + (self.n_out,), float(self.constant), dtype=F64)
        return torch.cat([nn.functional.softplus(net(x)) for net in self.nets], dim=-1)


def _div(J, N, dz):
    zero = torch.zeros(1, dtype=J.dtype)
    return (torch.cat([J, zero]) * -1 + torch.cat([zero, J])) / dz


class TorchColumnRHS(nn.Module):
    """Column.rhs (system A, hybrid scheme) with multipliers [eps D_L,eff, k_z,eff, k_LDF]."""

    def __init__(self, col, mult: Multipliers | None = None):
        super().__init__()
        self.col, self.mult = col, mult
        p = col.p
        self.N, self.dz, self.v = col.N, col.dz, float(p.v)
        self.eps, self.A, self.Cs, self.c_w, self.dH = col.eps, col.A, col.Cs, col.c_w, col.dH
        self.rhoC, self.h_a, self.iso = col.rhoC, col.h_a, col.iso
        self.D_L, self.k_z, self.k_LDF, self.compensate = float(p.D_L), float(p.k_z), float(p.k_LDF), col.compensate
        self.q_cap = col.iso.q_capacity

    def e_w(self, T):
        return self.c_w * (T - T0)

    def stream_energy(self, c, T):
        return self.rhoC * (T - T0) + c * self.e_w(T)

    def phi_over_T(self, T):
        return self.e_w(T) / T - self.c_w * torch.log(T / T0)

    def state(self, y):
        N = self.N
        n, a, u = y[:N], y[N:2 * N], y[2 * N:]
        T = T0 + (u - a * self.dH - (n + a) * self.e_w(torch.tensor(T0, dtype=F64))) / (self.Cs + (n + a) * self.c_w)
        return n, a, u, n / self.eps, a / self.A, T

    def local_inputs(self, c, q, T):
        return torch.stack([torch.log(torch.clamp(c, min=1e-30)), q / self.q_cap, T / 300.0], -1)

    def multipliers(self, c, q, T):
        if self.mult is None:
            one_f, one_c = torch.ones(self.N - 1, dtype=F64), torch.ones(self.N, dtype=F64)
            return one_f, one_f, one_c
        x = self.local_inputs(c, q, T)
        mf = self.mult(0.5 * (x[1:] + x[:-1]))
        mc = self.mult(x)
        return mf[:, 0], mf[:, 1], mc[:, 2]

    def fluxes(self, y, inp):
        c_in, T_in, T_w = inp[0], inp[1], inp[2]
        n, a, u, c, q, T = self.state(y)
        dz, v = self.dz, self.v
        mD, mk, mL = self.multipliers(c, q, T)
        b = self.iso.b0 * torch.exp(-self.dH / (R * T))
        r_ldf = self.A * self.k_LDF * mL * (Q_torch(self.iso, b * torch.clamp(c, min=0.0)) - q)
        epsD = self.eps * self.D_L
        k_eff = torch.full((self.N - 1,), self.k_z, dtype=F64)
        if self.compensate:
            epsD = max(0.0, epsD - v * dz / 2)
            c_face = 0.5 * (c[1:] + c[:-1])
            k_eff = torch.clamp(self.k_z - (self.rhoC + c_face * self.c_w) * v * dz / 2, min=0.0)
        epsD = epsD * mD
        k_eff = k_eff * mk
        J_n = -epsD * (c[1:] - c[:-1]) / dz
        dinvT = 1.0 / T[1:] - 1.0 / T[:-1]
        dphi = self.phi_over_T(T[1:]) - self.phi_over_T(T[:-1])
        small = dinvT.abs() < 1e-14
        eta = torch.where(small, self.e_w(0.5 * (T[1:] + T[:-1])), dphi / torch.where(small, torch.ones_like(dinvT), dinvT))
        J_u = eta * J_n - k_eff * (T[1:] - T[:-1]) / dz
        g_n = torch.cat([c_in.reshape(1), c])
        g_u = torch.cat([self.stream_energy(c_in, T_in).reshape(1), self.stream_energy(c, T)])
        W = self.h_a * (T - T_w)
        return dict(c=c, q=q, T=T, r_ldf=r_ldf, J_n=J_n, J_u=J_u, g_n=g_n, g_u=g_u, W=W)

    def forward(self, y, inp):
        f = self.fluxes(y, inp)
        dz, v, N = self.dz, self.v, self.N
        dn = v * (f["g_n"][:-1] - f["g_n"][1:]) / dz + _div(f["J_n"], N, dz) - f["r_ldf"]
        du = v * (f["g_u"][:-1] - f["g_u"][1:]) / dz + _div(f["J_u"], N, dz) - f["W"]
        return torch.cat([dn, f["r_ldf"], du])

    def _X(self, q):
        lo, hi = torch.full_like(q, -60.0), torch.full_like(q, 60.0)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            below = Q_torch(self.iso, torch.exp(mid)) < q
            lo, hi = torch.where(below, mid, lo), torch.where(below, hi, mid)
        return torch.exp(0.5 * (lo + hi))

    @torch.no_grad()
    def production(self, y, inp) -> dict:
        """Entropy production per mechanism with the learned multipliers: LDF per cell, dispersion
        plus conduction per face (formulation 4.2); each must be >= 0."""
        f = self.fluxes(y, inp)
        c, q, T = f["c"], f["q"], f["T"]
        wn = -(self.phi_over_T(T) + R * torch.log(c / C_REF))
        wa = -(self.phi_over_T(T) + self.dH / T + R * torch.log(self._X(q) / (self.iso.b0 * C_REF)))
        sig_ldf = f["r_ldf"] * (wa - wn)
        sig_face = (f["J_n"] * (wn[1:] - wn[:-1]) + f["J_u"] * (1 / T[1:] - 1 / T[:-1])) / self.dz
        return {"ldf": sig_ldf, "face": sig_face}


class TorchStoreRHS(nn.Module):
    """ThermalStore.rhs (system B) with multipliers [k_f,eff (per face), h_v (per cell)]."""

    def __init__(self, st, mult: Multipliers | None = None):
        super().__init__()
        self.st, self.mult = st, mult
        self.N, self.dz = st.N, st.dz
        self.Cf, self.Cs, self.rhoc, self.h_v, self.k_s, self.h_a = st.Cf, st.Cs, st.rhoc, st.h_v, st.k_s, st.h_a

    def forward(self, y, inp):
        v, T_in, T_amb = float(inp[0]), inp[1], inp[2]
        N, dz = self.N, self.dz
        Tf, Ts = T0_B + y[:N] / self.Cf, T0_B + y[N:] / self.Cs
        if self.mult is None:
            mk, mh = torch.ones(N - 1, dtype=F64), torch.ones(N, dtype=F64)
        else:
            x = torch.stack([Tf / 300.0, Ts / 300.0, Tf - Ts], -1)
            mk, mh = self.mult(0.5 * (x[1:] + x[:-1]))[:, 0], self.mult(x)[:, 1]
        g_in = (self.rhoc * (T_in - T0_B)).reshape(1)
        g = self.rhoc * (Tf - T0_B)
        g_up = torch.cat([g_in, g[:-1]]) if v >= 0 else torch.cat([g[1:], g_in])
        conv = abs(v) / dz * (g_up - g)
        W = self.h_v * mh * (Tf - Ts)
        Jf = -self.st.k_f_eff(v) * mk * (Tf[1:] - Tf[:-1]) / dz
        Js = -self.k_s * (Ts[1:] - Ts[:-1]) / dz
        Wwall = self.h_a * (Tf - T_amb)
        duf = conv - W + _div(Jf, N, dz) - Wwall
        dus = W + _div(Js, N, dz)
        return torch.cat([duf, dus])


def implicit_euler_step(rhs, y0, inp, h: float, tol: float = 1e-13, max_iter: int = 50):
    """y1 = y0 + h f(y1) by Newton; differentiable in y0 and the multiplier weights (see module doc)."""
    n = y0.numel()
    eye = torch.eye(n, dtype=F64)
    with torch.no_grad():
        y = y0.detach().clone()
        scale = float(y0.abs().max()) + 1e-300
        for _ in range(max_iter):
            F = y - y0.detach() - h * rhs(y, inp)
            if float(F.abs().max()) <= tol * scale:
                break
            J = torch.autograd.functional.jacobian(lambda x: rhs(x, inp), y, vectorize=True)
            y = y - torch.linalg.solve(eye - h * J, F)
    J = torch.autograd.functional.jacobian(lambda x: rhs(x, inp), y, vectorize=True).detach()
    F = y - y0 - h * rhs(y, inp)
    return y - torch.linalg.solve(eye - h * J, F)


class CPlusL(nn.Module):
    def __init__(self, rhs):
        super().__init__()
        self.rhs = rhs

    def rollout(self, y0, inps, dts):
        """States after each implicit-Euler step; inps (steps, 3), dts (steps,)."""
        ys, y = [], y0
        for k in range(inps.shape[0]):
            y = implicit_euler_step(self.rhs, y, inps[k], float(dts[k]))
            ys.append(y)
        return torch.stack(ys)
