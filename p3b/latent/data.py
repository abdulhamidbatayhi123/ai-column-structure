"""Training data from Plan 1's stored trajectories (Plan 2, Task 2).

A state vector is the stored coarse channels, flattened cell-major ((cell, channel) order), followed
by the outlet channels: system A 128 x (log c, log x, log T) + (log c_out, log T_out) = 386; system
B 128 x (T_f, T_s) + T_out = 257. Each snapshot carries the inputs of the interval that starts at
it (the half-cycle containing t_i; inputs are piecewise constant per interval):
  * raw inputs (what the physics needs): A (c_in, T_in, T_w, v); B (signed v, T_in);
  * model features (PREREG Section 4): A (log c_in, T_in, T_w, v); B (T_in, T_w, |v|, direction),
    with T_w the ambient of the wall loss.
Everything is float64 (Plan 2 Global Constraints).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from p3b.gates import cycle_release, load_meta

F64 = torch.float64


@dataclass
class Dataset:
    X: torch.Tensor            # (traj, cycle, snap, n_state)
    raw_inputs: torch.Tensor   # (traj, cycle, snap, 4 | 2)
    features: torch.Tensor     # (traj, cycle, snap, 4)
    t: torch.Tensor            # (traj, cycle, snap), absolute times
    meta: list
    system: str
    n_cells: int
    paths: list = None
    N_true: torch.Tensor = None    # (traj, cycle, snap) reference water inventory, A only (B: None)
    E_true: torch.Tensor = None    # (traj, cycle, snap) reference energy
    W_ref: torch.Tensor = None     # (traj, cycle) release per cycle (A water, B energy; gates.cycle_release)
    wall_abs: torch.Tensor = None  # (traj, cycle) integral of |wall heat| over the cycle


def _inputs(meta: dict, n_cycles: int, n_snap: int):
    sched, half = meta["schedule"], n_snap // 2
    if meta["system"] == "A":
        rows = [list(inp) + [meta["point"]["v"]] for _, _, inp in sched]
        raw = np.array([rows[0]] * half + [rows[1]] * half, dtype=float)
        feat = raw.copy()
        feat[:, 0] = np.log(raw[:, 0])
    else:
        from p3b.simulate_b import T_AMB
        rows = [[inp[0], inp[1]] for _, _, inp in sched]
        raw = np.array([rows[0]] * half + [rows[1]] * half, dtype=float)
        feat = np.stack([raw[:, 1], np.full(n_snap, T_AMB), np.abs(raw[:, 0]), np.sign(raw[:, 0])], axis=-1)
    return np.broadcast_to(raw, (n_cycles,) + raw.shape), np.broadcast_to(feat, (n_cycles,) + feat.shape)


def load_set(paths) -> Dataset:
    """Load equally shaped npz files of one system into float64 tensors."""
    X, raw, feat, t, metas, Nt, Et, W, wall = [], [], [], [], [], [], [], [], []
    paths = [str(p) for p in paths]
    for p in paths:
        d = np.load(p, allow_pickle=False)
        meta = load_meta(d)
        ch, out = d["channels"].astype(float), d["outlet"].astype(float)
        nc, ns = ch.shape[:2]
        X.append(np.concatenate([ch.reshape(nc, ns, -1), out.reshape(nc, ns, -1)], axis=-1))
        r, f = _inputs(meta, nc, ns)
        raw.append(r)
        feat.append(f)
        t.append(d["t"].astype(float))
        metas.append(meta)
        dz = meta["L"] / meta["N_c"]
        cb = d["conserved_blocks"]
        if meta["system"] == "A":
            Nt.append((cb[..., 0] + cb[..., 1]).sum(-1) * dz)
            Et.append(cb[..., 2].sum(-1) * dz)
            col = 4
        else:
            Et.append((cb[..., 0] + cb[..., 1]).sum(-1) * dz)
            col = 2
        W.append(cycle_release(d))
        wc = d["ports_cum"][..., col]
        nxt = np.append(wc[1:, 0], d["ports_end"][col])
        seq = np.concatenate([wc, nxt[:, None]], axis=1)
        wall.append(np.abs(np.diff(seq, axis=1)).sum(axis=1))
    tt = lambda a: torch.tensor(np.stack(a), dtype=F64)  # noqa: E731
    return Dataset(tt(X), tt(raw), tt(feat), tt(t), metas, metas[0]["system"], int(metas[0]["N_c"]),
                   paths, tt(Nt) if Nt else None, tt(Et), tt(W), tt(wall))


class Standardiser:
    """One mean and SD per channel type (not per cell, so no position is baked in), per outlet
    channel and per feature; fitted on the training set only."""

    def __init__(self, mean, sd, f_mean=None, f_sd=None):
        self.mean, self.sd, self.f_mean, self.f_sd = mean, sd, f_mean, f_sd

    @staticmethod
    def _safe(sd):
        return torch.where(sd > 0, sd, torch.ones_like(sd))

    @classmethod
    def fit(cls, X, n_cells: int, n_ch: int, n_out: int, features=None) -> "Standardiser":
        flat = X.reshape(-1, X.shape[-1])
        fields = flat[:, :n_cells * n_ch].reshape(-1, n_ch)
        outl = flat[:, n_cells * n_ch:]
        m = torch.cat([fields.mean(0).repeat(n_cells), outl.mean(0)])
        s = torch.cat([fields.std(0).repeat(n_cells), outl.std(0)])
        fm = fs = None
        if features is not None:
            F = features.reshape(-1, features.shape[-1])
            fm, fs = F.mean(0), F.std(0)
        return cls(m, cls._safe(s), fm, None if fs is None else cls._safe(fs))

    def encode(self, X):
        return (X - self.mean) / self.sd

    def decode(self, Z):
        return Z * self.sd + self.mean

    def encode_features(self, F):
        return (F - self.f_mean) / self.f_sd


def windows(ds: Dataset, n_windows: int = 8, n_steps: int = 20, rng=None, starts=None):
    """Random training windows over the flattened (cycle, snapshot) sequence of each trajectory.

    Returns (x0, X, F, R, dt): the start state (w, n), the next n_steps states (w, n_steps, n), the
    features and raw inputs of each step's interval (w, n_steps, k) and its length dt (w, n_steps).
    `starts`, a list of (trajectory, flat index), overrides the random choice (tests).
    """
    n_traj, n_cyc, n_snap, n = ds.X.shape
    T = n_cyc * n_snap
    Xf = ds.X.reshape(n_traj, T, n)
    Ff = ds.features.reshape(n_traj, T, -1)
    Rf = ds.raw_inputs.reshape(n_traj, T, -1)
    tf = ds.t.reshape(n_traj, T)
    if starts is None:
        rng = rng if rng is not None else np.random.default_rng()
        starts = [(int(rng.integers(n_traj)), int(rng.integers(T - n_steps))) for _ in range(n_windows)]
    idx = [(j, s) for j, s in starts]
    x0 = torch.stack([Xf[j, s] for j, s in idx])
    X = torch.stack([Xf[j, s + 1:s + 1 + n_steps] for j, s in idx])
    F = torch.stack([Ff[j, s:s + n_steps] for j, s in idx])
    R_ = torch.stack([Rf[j, s:s + n_steps] for j, s in idx])
    dt = torch.stack([tf[j, s + 1:s + 1 + n_steps] - tf[j, s:s + n_steps] for j, s in idx])
    return x0, X, F, R_, dt
