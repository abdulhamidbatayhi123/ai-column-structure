"""The shared autoencoder, its decoder gate and the latent-dimension rule (PREREG 4 and 5.1).

MLP: encoder n_in -> 256 -> 256 -> d, decoder d -> 256 -> 256 -> n_in, SiLU, on standardised
state vectors (p3b.latent.data). Fallback when no d passes: the same encoder with a fixed 1-D
convolutional decoder (4 conv layers, 32 channels) and a linear outlet head. Training: MSE,
Adam, learning rate 1e-3 with cosine decay to 1e-5 over all steps, batch 256, 60 epochs.
Everything is float64.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from p3b.latent.metrics import field_nrmse

F64 = torch.float64
D_GRID = (16, 24, 32, 48, 64)
GATE = dict(field=0.02, outlet=0.02, balance=0.01)


def _mlp(sizes):
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(a, b, dtype=F64))
        if i < len(sizes) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class MLPAutoencoder(nn.Module):
    def __init__(self, n_in: int, d: int, width: int = 256):
        super().__init__()
        self.encoder = _mlp([n_in, width, width, d])
        self.decoder = _mlp([d, width, width, n_in])

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))


class ConvDecoderAutoencoder(nn.Module):
    """MLP encoder; decoder: linear to 32 x (n_cells/8), three stride-2 transposed convolutions
    (32 channels) and one output convolution, plus a linear head for the outlet channels."""

    def __init__(self, n_cells: int, n_ch: int, n_out: int, d: int, width: int = 256, channels: int = 32):
        super().__init__()
        if n_cells % 8:
            raise ValueError("n_cells must be divisible by 8")
        self.n_cells, self.n_ch, self.n_out, self.c = n_cells, n_ch, n_out, channels
        self.encoder = _mlp([n_cells * n_ch + n_out, width, width, d])
        self.lift = nn.Linear(d, channels * (n_cells // 8), dtype=F64)
        self.convs = nn.Sequential(
            nn.ConvTranspose1d(channels, channels, 4, stride=2, padding=1, dtype=F64), nn.SiLU(),
            nn.ConvTranspose1d(channels, channels, 4, stride=2, padding=1, dtype=F64), nn.SiLU(),
            nn.ConvTranspose1d(channels, channels, 4, stride=2, padding=1, dtype=F64), nn.SiLU(),
            nn.Conv1d(channels, n_ch, 3, padding=1, dtype=F64))
        self.head = nn.Linear(d, n_out, dtype=F64)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        lead = z.shape[:-1]
        h = self.lift(z.reshape(-1, z.shape[-1])).reshape(-1, self.c, self.n_cells // 8)
        f = self.convs(h).permute(0, 2, 1).reshape(-1, self.n_cells * self.n_ch)
        return torch.cat([f, self.head(z.reshape(-1, z.shape[-1]))], dim=-1).reshape(*lead, -1)

    def forward(self, x):
        return self.decode(self.encode(x))


def cosine_lr(step: int, total: int, lr: float, lr_min: float) -> float:
    return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * min(step, total) / total))


def train_autoencoder(model, X, epochs: int = 60, batch: int = 256, lr: float = 1e-3, lr_min: float = 1e-5,
                      seed: int = 0) -> dict:
    """MSE training on standardised state vectors X (n_samples, n). Returns {"loss": per epoch}."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    X = X.reshape(-1, X.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    steps_per_epoch = math.ceil(X.shape[0] / batch)
    total, step, hist = epochs * steps_per_epoch, 0, []
    for _ in range(epochs):
        perm = torch.randperm(X.shape[0], generator=gen)
        tot = 0.0
        for i in range(steps_per_epoch):
            xb = X[perm[i * batch:(i + 1) * batch]]
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, total, lr, lr_min)
            loss = ((model(xb) - xb) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * xb.shape[0]
            step += 1
        hist.append(tot / X.shape[0])
    return {"loss": hist}


def _temps_low_high(meta):
    T = [inp[1] for _, _, inp in meta["schedule"]]
    return min(T), max(T)


@torch.no_grad()
def decoder_gate(model, ds, std, phys) -> dict:
    """PREREG 5.1 on a validation Dataset: mean field nRMSE, outlet RMSEs, balance errors.

    Balances are the largest |N^ - N| / W_ref and |E^ - E| / (|dH| W_ref + int |wall| dt) over all
    snapshots (B: |E^ - E| / (|Q_ref| + int |wall loss| dt)); each must be <= 0.01."""
    A = ds.system == "A"
    n_ch, n_out = (3, 2) if A else (2, 1)
    nc = ds.n_cells
    nr, oc, ot, bN, bE = [], [], [], [], []
    for j in range(ds.X.shape[0]):
        lo, hi = _temps_low_high(ds.meta[j])
        for k in range(ds.X.shape[1]):
            X = ds.X[j, k]
            R = std.decode(model.decode(model.encode(std.encode(X))))
            nr.append(float(field_nrmse(R, X, nc, n_ch)))
            ch_r = R[:, :nc * n_ch].reshape(-1, nc, n_ch)
            out_r, out_t = R[:, nc * n_ch:], X[:, nc * n_ch:]
            W = abs(float(ds.W_ref[j, k]))
            wall = float(ds.wall_abs[j, k])
            if A:
                c_in = ds.raw_inputs[j, k, :, 0]
                oc.append(((torch.exp(out_r[:, 0]) - torch.exp(out_t[:, 0])) / c_in).pow(2))
                ot.append(((torch.exp(out_r[:, 1]) - torch.exp(out_t[:, 1])) / (hi - lo)).pow(2))
                bN.append(float((phys.N(ch_r) - ds.N_true[j, k]).abs().max()) / W)
                bE.append(float((phys.E(ch_r) - ds.E_true[j, k]).abs().max()) / (abs(phys.dH) * W + wall))
            else:
                ot.append(((out_r[:, 0] - out_t[:, 0]) / (hi - lo)).pow(2))
                bE.append(float((phys.E(ch_r) - ds.E_true[j, k]).abs().max()) / (W + wall))
    res = dict(field_nrmse=float(np.mean(nr)),
               outlet_c=float(torch.sqrt(torch.cat(oc).mean())) if oc else None,
               outlet_T=float(torch.sqrt(torch.cat(ot).mean())),
               balance_N=max(bN) if bN else None, balance_E=max(bE))
    res["passed"] = bool(res["field_nrmse"] <= GATE["field"]
                         and (res["outlet_c"] is None or res["outlet_c"] <= GATE["outlet"])
                         and res["outlet_T"] <= GATE["outlet"]
                         and (res["balance_N"] is None or res["balance_N"] <= GATE["balance"])
                         and res["balance_E"] <= GATE["balance"])
    return res


def choose_d(passed: dict):
    """The smallest d whose full-data autoencoders all passed the gate; None if none passed."""
    ok = [d for d, p in passed.items() if p]
    return min(ok) if ok else None
