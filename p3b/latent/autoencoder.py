"""The shared autoencoder, its decoder gate and the latent-dimension rule (PREREG 4 and 5.1).

MLP: encoder n_in -> 256 -> 256 -> d, decoder d -> 256 -> 256 -> n_in, SiLU, on standardised
state vectors (p3b.latent.data). Fallback when no d passes: the same encoder with a fixed 1-D
convolutional decoder (4 conv layers, 32 channels) and a linear outlet head. Training: MSE,
Adam, learning rate 1e-3 with cosine decay to 1e-5 over all steps, batch 256, 60 epochs.
Everything is float64.
"""
from __future__ import annotations

import math

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


def zero_last(net: nn.Module) -> nn.Module:
    """Zero the last Linear layer: a state-update network starts at a zero increment (every
    dynamics arm starts from rest: U at dz/dt = 0, S at the physics-minimal field, FNO at the
    identity), the same choice for every arm."""
    last = [m for m in net.modules() if isinstance(m, nn.Linear)][-1]
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    return net


def small_last(net: nn.Module, scale: float = 1e-2) -> nn.Module:
    """Scale the last Linear layer down (bias zero): for S's A(z), whose term A A^T vanishes
    quadratically, so it cannot start at exactly zero (its gradient would vanish too)."""
    last = [m for m in net.modules() if isinstance(m, nn.Linear)][-1]
    with torch.no_grad():
        last.weight.mul_(scale)
        last.bias.zero_()
    return net


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


BALANCE_WEIGHT = 1.0    # freeze decision 1b (author, 2026-11-12): MSE + 1 x (the gate's balance errors)^2


def balance_errors(ds, std, phys):
    """f(rec, idx) -> (eN, eE): the decoder gate's balance errors (PREREG 5.1) of standardised
    reconstructions rec of the flattened snapshots idx of ds (in std.encode(ds.X).reshape(-1, n) order):
    (N^ - N) / W_ref and (E^ - E) / (|dH| W_ref + int |wall| dt); B: eN None and (E^ - E) / (|Q_ref| +
    int |wall loss| dt), as decoder_gate normalises them."""
    A = ds.system == "A"
    n_ch = 3 if A else 2
    nc = ds.n_cells
    T, C, S = ds.X.shape[:3]
    W = ds.W_ref.abs().unsqueeze(-1).expand(T, C, S).reshape(-1)
    wall = ds.wall_abs.unsqueeze(-1).expand(T, C, S).reshape(-1)
    Et = ds.E_true.reshape(-1)
    Nt = ds.N_true.reshape(-1) if A else None
    denom_E = (abs(phys.dH) * W + wall) if A else (W + wall)

    def f(rec, idx):
        ch = std.decode(rec)[:, :nc * n_ch].reshape(-1, nc, n_ch)
        eE = (phys.E(ch) - Et[idx]) / denom_E[idx]
        eN = (phys.N(ch) - Nt[idx]) / W[idx] if A else None
        return eN, eE
    return f


def fit_autoencoder(model, ds, std, phys, epochs: int = 60, seed: int = 0, weight: float = BALANCE_WEIGHT,
                    **kw) -> dict:
    """Every autoencoder of the study and the pilot is trained here: MSE on the standardised states plus
    `weight` x the mean squared balance errors (balance_errors), the decoder trained for what the gate
    checks (freeze decision 1b). weight 0 is the plain MSE training."""
    errs = balance_errors(ds, std, phys) if weight else None

    def penalty(rec, idx):
        eN, eE = errs(rec, idx)
        return eE.pow(2).mean() + (eN.pow(2).mean() if eN is not None else 0.0)
    return train_autoencoder(model, std.encode(ds.X), epochs=epochs, seed=seed,
                             penalty=penalty if weight else None, weight=weight, **kw)


def train_autoencoder(model, X, epochs: int = 60, batch: int = 256, lr: float = 1e-3, lr_min: float = 1e-5,
                      seed: int = 0, penalty=None, weight: float = 0.0) -> dict:
    """MSE training on standardised state vectors X (n_samples, n), plus weight x penalty(rec, idx) when a
    penalty is given (idx: the batch's rows of the flattened X). Returns {"loss": per epoch}."""
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
            idx = perm[i * batch:(i + 1) * batch]
            xb = X[idx]
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, total, lr, lr_min)
            rec = model(xb)
            loss = ((rec - xb) ** 2).mean()
            if penalty is not None and weight:
                loss = loss + weight * penalty(rec, idx)
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
    """PREREG 5.1 on a validation Dataset: field nRMSE, outlet RMSEs, balance errors.

    The field nRMSE is Paper 1's metric on the physical fields (A: c, q, T; B: T_f, T_s), each
    field's mean over the validation trajectory-cycles, and the gate takes the largest field
    (Paper 1 forbids pooling across fields); the same on the stored channels is reported as
    field_nrmse_channels.

    Balances are the largest |N^ - N| / W_ref and |E^ - E| / (|dH| W_ref + int |wall| dt) over all
    snapshots (B: |E^ - E| / (|Q_ref| + int |wall loss| dt)); each must be <= 0.01."""
    A = ds.system == "A"
    n_ch, n_out = (3, 2) if A else (2, 1)
    nc = ds.n_cells
    nr, nr_ch, oc, ot, bN, bE = [], [], [], [], [], []
    for j in range(ds.X.shape[0]):
        lo, hi = _temps_low_high(ds.meta[j])
        for k in range(ds.X.shape[1]):
            X = ds.X[j, k]
            R = std.decode(model.decode(model.encode(std.encode(X))))
            nr_ch.append(field_nrmse(R, X, nc, n_ch, per_field=True))
            ch_r = R[:, :nc * n_ch].reshape(-1, nc, n_ch)
            ch_t = X[:, :nc * n_ch].reshape(-1, nc, n_ch)
            nr.append(field_nrmse(phys.physical_fields(ch_r).reshape(len(R), -1),
                                  phys.physical_fields(ch_t).reshape(len(X), -1), nc, n_ch, per_field=True))
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
    per_field = torch.stack(nr).mean(0)
    res = dict(field_nrmse=float(per_field.max()), field_nrmse_per_field=per_field.tolist(),
               field_nrmse_channels=float(torch.stack(nr_ch).mean(0).max()),
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


def choose_decoder(mlp: dict, conv: dict | None = None):
    """(decoder, d): the MLP's smallest passing d; if none passes, the convolutional decoder's on the same
    grid (PREREG 4); (None, None) if neither passes."""
    d = choose_d(mlp)
    if d is not None:
        return "mlp", d
    d = choose_d(conv or {})
    return ("conv", d) if d is not None else (None, None)


def full_data_choice(metrics: dict):
    """(decoder, d) from the full-data gate (PREREG 4, 5.1): metrics = {kind: {d: [gate dict per seed]}}.
    A d passes when it was scored for at least one seed and every seed passed (and passed_long where the
    long-validation check was recorded); the smallest passing MLP d, else the smallest passing conv d,
    else (None, None): not testable."""
    def passes(seeds):
        return bool(seeds) and all(s["passed"] and s.get("passed_long", True) for s in seeds)
    return choose_decoder({int(d): passes(s) for d, s in (metrics.get("mlp") or {}).items()},
                          {int(d): passes(s) for d, s in (metrics.get("conv") or {}).items()})


def gate_ratio(m: dict) -> float:
    """The worst decoder-gate criterion as a multiple of its threshold (<= 1 passes); None criteria
    (system B has no outlet_c and no balance_N) are skipped."""
    pairs = [(m.get("field_nrmse"), GATE["field"]), (m.get("outlet_c"), GATE["outlet"]),
             (m.get("outlet_T"), GATE["outlet"]), (m.get("balance_N"), GATE["balance"]),
             (m.get("balance_E"), GATE["balance"])]
    return max(v / t for v, t in pairs if v is not None)


def _passed(gate: dict) -> dict:
    return {int(k): bool(v["passed"] if isinstance(v, dict) else v) for k, v in gate.items()}


def pilot_working_decoder(mlp: dict, conv: dict | None = None):
    """The (decoder, d) the pilot's own runs use. The pilot does not set d (PREREG 10); d is set at full
    data (PREREG 4). Rule: the smallest passing d, MLP then conv; if none passes, the MLP d whose worst
    criterion is nearest its threshold (ties: the smaller d). Gates are metric dicts or booleans."""
    dec, d = choose_decoder(_passed(mlp), _passed(conv or {}))
    if d is not None:
        return dec, d
    if not mlp or not all(isinstance(v, dict) for v in mlp.values()):
        raise ValueError("no pilot d passes and the gate has no metrics to rank the d grid")
    return "mlp", min((gate_ratio(v), int(k)) for k, v in mlp.items())[1]


def make_autoencoder(kind: str, n_cells: int, n_ch: int, n_out: int, d: int, width: int = 256):
    """The shared autoencoder of the recorded kind: "mlp" (pre-registered) or "conv" (the fallback)."""
    n_in = n_cells * n_ch + n_out
    if kind == "mlp":
        return MLPAutoencoder(n_in, d, width=width)
    if kind == "conv":
        return ConvDecoderAutoencoder(n_cells, n_ch, n_out, d, width=width)
    raise ValueError(f"unknown decoder {kind!r}")
