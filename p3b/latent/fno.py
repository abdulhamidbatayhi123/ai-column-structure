"""The autoregressive 1-D Fourier neural operator arm (PREREG Section 3; Plan 2, Task 8).

It maps the stored coarse fields and outlet channels at one snapshot to the next, conditioned on
the cycle inputs, with 4 Fourier layers, 16 retained modes and width 64 (the width is then
matched to S's trainable parameter count within +-5 %). The outlet channels and the inputs enter
as constant extra channels; the step length enters as log(dt), because snapshot spacing differs
between half-cycles and trajectories. The update is residual: x_next = x + net(x). The fields
leave through a pointwise head, the outlet channels through a head on the cell mean. Spectral
weights are stored as real tensors (..., 2), so the parameter count counts real numbers.
"""
from __future__ import annotations

import torch
from torch import nn

F64 = torch.float64


class SpectralConv1d(nn.Module):
    def __init__(self, c_in: int, c_out: int, modes: int):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (c_in * c_out)
        self.weight = nn.Parameter(scale * torch.randn(c_in, c_out, modes, 2, dtype=F64))

    def forward(self, x):                                # x (batch, c_in, n)
        xf = torch.fft.rfft(x, dim=-1)
        m = min(self.modes, xf.shape[-1])
        w = torch.view_as_complex(self.weight[:, :, :m].contiguous())
        out = torch.zeros(x.shape[0], w.shape[1], xf.shape[-1], dtype=xf.dtype)
        out[..., :m] = torch.einsum("bim,iom->bom", xf[..., :m], w)
        return torch.fft.irfft(out, n=x.shape[-1], dim=-1)


class FNO1d(nn.Module):
    def __init__(self, n_cells: int, n_ch: int, n_out: int, k: int, modes: int = 16, width: int = 64,
                 layers: int = 4, head: int = 128):
        super().__init__()
        self.n_cells, self.n_ch, self.n_out = n_cells, n_ch, n_out
        self.lift = nn.Linear(n_ch + n_out + k + 2, width, dtype=F64)      # + grid coordinate + log dt
        self.spectral = nn.ModuleList(SpectralConv1d(width, width, modes) for _ in range(layers))
        self.pointwise = nn.ModuleList(nn.Conv1d(width, width, 1, dtype=F64) for _ in range(layers))
        self.field_head = nn.Sequential(nn.Linear(width, head, dtype=F64), nn.GELU(),
                                        nn.Linear(head, n_ch, dtype=F64))
        self.outlet_head = nn.Linear(width, n_out, dtype=F64)
        for head in (self.field_head[-1], self.outlet_head):      # start at the identity map
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, x, feat, dt):
        b, n, c = x.shape[0], self.n_cells, self.n_ch
        fields = x[:, :n * c].reshape(b, n, c)
        outlet = x[:, n * c:]
        grid = torch.linspace(0.0, 1.0, n, dtype=F64).expand(b, n)[..., None]
        const = torch.cat([outlet, feat, torch.log(dt)[:, None]], -1)[:, None, :].expand(b, n, -1)
        h = self.lift(torch.cat([fields, const, grid], -1)).transpose(1, 2)      # (b, width, n)
        for i, (sp, pw) in enumerate(zip(self.spectral, self.pointwise)):
            h = sp(h) + pw(h)
            if i < len(self.spectral) - 1:
                h = nn.functional.gelu(h)
        h = h.transpose(1, 2)                                                   # (b, n, width)
        d_fields = self.field_head(h).reshape(b, n * c)
        d_out = self.outlet_head(h.mean(1))
        return x + torch.cat([d_fields, d_out], -1)
