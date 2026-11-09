"""The pilot runner (Plan 5): the inputs of scripts/pilot_rules.py, computed from the pilot data
only, written as JSON, never printed. This module holds the data-only inputs (Task 1); the
training-based inputs are added by Tasks 2 to 5.

- reference_noise: the reference floors of the violation metric (p3b.eval.violations.reference_floor),
  the reference's model-style production rates (the negative part sets tau_S, plan 4 review), and the
  pilot-wide tau_ref: 10 x the 99.9th percentile of the negative part of the fine-level entropy
  statistic pooled over the pilot trajectories, floored at 10 x rtol x the median entropy turnover
  rate (PREREG 5.3, p3b.gates.entropy_gate per trajectory);
- css_cycles: the cyclic-steady-state cycle of each pilot trajectory (p3b.gates.css_cycle); a
  trajectory that does not reach it within its stored cycles is recorded as n_cycles + 1;
- decoder_gate_grid: the decoder gate (PREREG 5.1) over the d grid on pilot data.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch


def _physics(ds):
    from p3b.latent.physics import ColumnPhysics, StorePhysics
    if ds.system == "A":
        from p3b.simulate_a import setup_a
        return ColumnPhysics(setup_a(ds.meta[0]["point"], ds.meta[0]["material"], ds.n_cells)[0])
    from p3b.simulate_b import T_AMB, make_store
    return StorePhysics(make_store(ds.n_cells), T_AMB)


def reference_noise(paths) -> dict:
    from p3b import gates
    from p3b.eval.violations import reference_floor, trapezoid_residuals
    from p3b.latent.data import load_set
    from p3b.simulate_a import RTOL
    ds = load_set(paths)
    phys = _physics(ds)
    floors = reference_floor(ds, phys)
    prod = []
    for j in range(ds.X.shape[0]):
        X = ds.X[j].reshape(-1, ds.X.shape[-1])
        R = ds.raw_inputs[j].reshape(X.shape[0], -1)
        t = ds.t[j].reshape(-1)
        r = trapezoid_residuals(X, R, t, phys, ds.system, ds.n_cells)
        prod.append((r["S"] / (t[1:] - t[:-1])).numpy())
    neg, turnover = [], []
    for p in paths:
        meta = gates.load_meta(p)
        e = gates._entropy_intervals(p, gates.model_from_meta(meta))
        if e["fine_rate"] is not None:
            f = e["fine_rate"][np.isfinite(e["fine_rate"])]
            neg.append(np.maximum(0.0, -f))
        turnover.append(e["turnover_rate"])
    neg = np.concatenate(neg) if neg else np.zeros(1)
    tau_ref = max(10.0 * float(np.percentile(neg, 99.9)), 10.0 * RTOL * float(np.median(np.concatenate(turnover))))
    return {"floors": {k: v for k, v in floors.items() if k in ("N", "E")},
            "ref_production": np.concatenate(prod).tolist(), "tau_ref": tau_ref}


def css_cycles(paths) -> list:
    from p3b import gates
    out = []
    for p in paths:
        c = gates.css_cycle(p)
        out.append(int(c) if c is not None else int(np.load(p)["ports_cum"].shape[0]) + 1)
    return out


def decoder_gate_grid(train_paths, val_paths, d_grid=(16, 24, 32, 48, 64), epochs: int = 60, width: int = 256,
                      seed: int = 0) -> dict:
    from p3b.latent.autoencoder import MLPAutoencoder, decoder_gate, train_autoencoder
    from p3b.latent.data import Standardiser, load_set
    ds, val = load_set(train_paths), load_set(val_paths)
    n_ch, n_out = (3, 2) if ds.system == "A" else (2, 1)
    std = Standardiser.fit(ds.X, ds.n_cells, n_ch, n_out, features=ds.features)
    phys = _physics(val)
    out = {}
    for d in d_grid:
        torch.manual_seed(seed)
        ae = MLPAutoencoder(ds.X.shape[-1], d, width=width)
        train_autoencoder(ae, std.encode(ds.X), epochs=epochs, seed=seed)
        out[int(d)] = bool(decoder_gate(ae, val, std, phys)["passed"])
    return out


def write_data_inputs(dirs_by_key: dict, out_dir, d_grid=(16, 24, 32, 48, 64), epochs: int = 60,
                      width: int = 256) -> list:
    """For each pilot key's directory (train/ and validation/ subdirectories), write css.json,
    decoder_gate.json and reference_noise.json (the data part of threshold_inputs.json)."""
    import glob
    os.makedirs(out_dir, exist_ok=True)
    css, gate, noise = {}, {}, {}
    for key, d in dirs_by_key.items():
        train = sorted(glob.glob(os.path.join(str(d), "train", "*.npz")))
        val = sorted(glob.glob(os.path.join(str(d), "validation", "*.npz")))
        css[key] = css_cycles(val)
        gate[key] = {str(k): v for k, v in decoder_gate_grid(train, val, d_grid, epochs, width).items()}
        noise[key] = reference_noise(val)
    written = []
    for name, obj in (("css.json", css), ("decoder_gate.json", gate), ("reference_noise.json", noise)):
        path = os.path.join(str(out_dir), name)
        with open(path, "w") as f:
            json.dump(obj, f)
        written.append(path)
    return written
