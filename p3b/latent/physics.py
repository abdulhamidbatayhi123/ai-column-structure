"""Physical functionals and port fluxes of decoded coarse states, in torch float64 (Plan 2, Task 1).

Formulation Sections 3, 4.4 and 5.2. A decoded state of system A is the channels
(log c, log x, log T) on the coarse cells plus the outlet channels (log c_out, log T_out); q = Q(x),
so no inverse isotherm is needed. Every formula is transcribed from `p3b.column.Column` (energy,
cell entropy, stream energy and entropy, wall exchange) and `p3b.thermal_store.ThermalStore`,
and the tests check them against those to round-off. Leading dimensions are batch dimensions.
"""
from __future__ import annotations

import torch

from p3b.column import C_REF, R, T0
from p3b.thermal_store import T0 as T0_B

F64 = torch.float64


def Q_torch(iso, x):
    out = torch.zeros_like(x)
    for q, a, n in iso.terms:
        y = (x / a) ** n
        out = out + q * y / (1 + y)
    return out


def Pi_torch(iso, x):
    out = torch.zeros_like(x)
    for q, a, n in iso.terms:
        out = out + q / n * torch.log1p((x / a) ** n)
    return out


class ColumnPhysics:
    """E, N, S and the ports of a decoded system-A state on the column `col`'s coarse grid.

    `col` supplies the bed, the isotherm and c_w; the grid spacing is L / n_cells of the channels.
    """

    def __init__(self, col):
        self.col = col
        self.iso = col.iso
        self.eps, self.A, self.Cs, self.c_w = col.eps, col.A, col.Cs, col.c_w
        self.dH, self.rhoC, self.h_a, self.L = col.dH, col.rhoC, col.h_a, col.p.L
        self.b0 = self.iso.b0

    def _dz(self, ch):
        return self.L / ch.shape[-2]

    def e_w(self, T):
        return self.c_w * (T - T0)

    def s_w(self, T):
        return self.c_w * torch.log(T / T0)

    def fields(self, ch):
        return torch.exp(ch[..., 0]), torch.exp(ch[..., 1]), torch.exp(ch[..., 2])

    def conserved(self, c, x, T):
        n = self.eps * c
        a = self.A * Q_torch(self.iso, x)
        u = self.Cs * (T - T0) + (n + a) * self.e_w(T) + a * self.dH
        return n, a, u

    def E(self, ch):
        n, a, u = self.conserved(*self.fields(ch))
        return u.sum(-1) * self._dz(ch)

    def N(self, ch):
        n, a, _ = self.conserved(*self.fields(ch))
        return (n + a).sum(-1) * self._dz(ch)

    def cell_entropy(self, ch):
        c, x, T = self.fields(ch)
        n, a, _ = self.conserved(c, x, T)
        q = a / self.A
        s_gas = n * (self.s_w(T) - R * torch.log(c / C_REF) + R)
        s_ads = q * self.s_w(T) - R * (q * torch.log(x / (self.b0 * C_REF)) - Pi_torch(self.iso, x))
        return self.Cs * torch.log(T / T0) + s_gas + self.A * s_ads

    def S(self, ch):
        return self.cell_entropy(ch).sum(-1) * self._dz(ch)

    def stream_energy(self, c, T):
        return self.rhoC * (T - T0) + c * self.e_w(T)

    def stream_entropy(self, c, T):
        return self.rhoC * torch.log(T / T0) + c * (self.s_w(T) - R * torch.log(c / C_REF) + R)

    def ports(self, ch, out, inp):
        """(P_N, P_E, Phi_S): inlet from the feed inp = (c_in, T_in, T_w, v), outlet from the
        decoded outlet channels, wall from the decoded coarse temperatures (formulation 5.2)."""
        c_in, T_in, T_w, v = inp[..., 0], inp[..., 1], inp[..., 2], inp[..., 3]
        c_out, T_out = torch.exp(out[..., 0]), torch.exp(out[..., 1])
        T = torch.exp(ch[..., 2])
        wall = (self.h_a * (T - T_w[..., None])).sum(-1) * self._dz(ch)
        P_N = v * (c_in - c_out)
        P_E = v * (self.stream_energy(c_in, T_in) - self.stream_energy(c_out, T_out)) - wall
        Phi_S = v * (self.stream_entropy(c_in, T_in) - self.stream_entropy(c_out, T_out)) - wall / T_w
        return P_N, P_E, Phi_S


class StorePhysics:
    """E, S and the ports of a decoded system-B state (channels T_f, T_s; outlet T_out)."""

    def __init__(self, st, T_amb: float):
        self.st, self.T_amb = st, float(T_amb)
        self.Cf, self.Cs, self.rhoc, self.h_a, self.L = st.Cf, st.Cs, st.rhoc, st.h_a, st.L

    def _dz(self, ch):
        return self.L / ch.shape[-2]

    def E(self, ch):
        return (self.Cf * (ch[..., 0] - T0_B) + self.Cs * (ch[..., 1] - T0_B)).sum(-1) * self._dz(ch)

    def S(self, ch):
        return (self.Cf * torch.log(ch[..., 0] / T0_B) + self.Cs * torch.log(ch[..., 1] / T0_B)).sum(-1) \
            * self._dz(ch)

    def ports(self, ch, out, inp):
        """(P_E, Phi_S) with inp = (v, T_in); |v| carries the stream, the outlet is the decoded T_out."""
        v, T_in = inp[..., 0].abs(), inp[..., 1]
        T_out = out[..., 0]
        wall = (self.h_a * (ch[..., 0] - self.T_amb)).sum(-1) * self._dz(ch)
        P_E = v * self.rhoc * (T_in - T_out) - wall
        Phi_S = v * self.rhoc * (torch.log(T_in / T0_B) - torch.log(T_out / T0_B)) - wall / self.T_amb
        return P_E, Phi_S
