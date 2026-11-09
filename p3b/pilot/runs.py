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


# ---- Task 2: learning rates and iteration curves ----------------------------------------------------
@torch.no_grad()
def kappa_from_data(lf, ae, std, ds, n: int = 256, seed: int = 0) -> float:
    """kappa = 1e-3 x the median |P grad S^|^2 over encoded pilot snapshots (PREREG Section 4). It
    needs no trained dynamics, so S's learning-rate sweep can use it."""
    from p3b.latent.arms_structured import structure_terms
    rng = np.random.default_rng(seed)
    X = ds.X.reshape(-1, ds.X.shape[-1])
    R = ds.raw_inputs.reshape(X.shape[0], -1)
    idx = torch.as_tensor(rng.choice(X.shape[0], size=min(n, X.shape[0]), replace=False))
    z = ae.encode(std.encode(X[idx]))
    t = lf.evaluate(z, R[idx], grads=True, create_graph=False)
    B = torch.stack([t["dE"], t["dN"]], -1) if lf.system == "A" else t["dE"].unsqueeze(-1)
    _, info = structure_terms(B, t["dS"], torch.zeros_like(z), None, None, None, kappa=1.0,
                              metriplectic=False, port_projection=False)
    return 1e-3 * float(info["P_gradS_sq"].median())


def validation_loss(arm, val, h_steps: int, n_cycles: int = 1) -> float:
    """MSE of the decoded standardised channels over the first n_cycles of every validation
    trajectory (the training loss's metric on a rollout); NaN if any rollout failed."""
    from p3b.eval.rollout import rollout
    losses = []
    for j in range(val.X.shape[0]):
        r = rollout(arm, val, j, n_cycles, h_steps)
        if r.failed_cycle is not None:
            return float("nan")
        d = arm.std.encode(r.X) - arm.std.encode(val.X[j, :n_cycles])
        losses.append(float((d ** 2).mean()))
    return float(np.mean(losses))


def train_curve(arm, ds, val, lr: float, iterations: int, every: int, h_steps: int, seed: int,
                n_windows: int = 8, n_steps: int = 20) -> list:
    """Train with one cosine schedule over `iterations`, pausing every `every` iterations (a
    checkpoint) to record the pooled validation loss: [(iteration, loss), ...]."""
    import tempfile

    from p3b.train.trainer import TrainConfig, train
    curve = []
    with tempfile.TemporaryDirectory() as tmp:
        ck = os.path.join(tmp, "ck.pt")
        done = 0
        while done < iterations:
            step = min(every, iterations - done)
            train(arm, ds, TrainConfig(lr=lr, iterations=iterations, seed=seed, h_steps=h_steps,
                                       n_windows=n_windows, n_steps=n_steps, checkpoint=ck, every=every,
                                       stop_after=step))
            done += step
            curve.append((done, validation_loss(arm, val, h_steps)))
    return curve


def lr_sweep(make_arm, ds, val, rates=(3e-4, 1e-3, 3e-3), iterations: int = 2000, h_steps: int = 1, seed: int = 0,
             n_windows: int = 8, n_steps: int = 20) -> dict:
    """{lr: validation rollout loss over one cycle after `iterations`} for a fresh arm per rate."""
    return {lr: train_curve(make_arm(), ds, val, lr, iterations, iterations, h_steps, seed,
                            n_windows, n_steps)[-1][1] for lr in rates}


# ---- Task 3: the step size h, S's defects ------------------------------------------------------------
RK4_STABLE = 2.5     # h rho(J) below RK4's real-axis stability limit (2.785) with a margin


def spectral_radius(f, z, iters: int = 30, seed: int = 0) -> float:
    """The largest |eigenvalue| of the Jacobian of the vector field f at z (one state, shape (1, d)),
    by power iteration on Jacobian-vector products. The products are central differences, because S's
    field calls reverse-mode autograd internally (its latent gradients), which forward mode cannot wrap."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(z.shape, dtype=z.dtype, generator=g)
    v = v / v.norm()
    rho = 0.0
    eps = 1e-6 * max(1.0, float(z.abs().max()))
    for _ in range(iters):
        with torch.no_grad():
            Jv = (f(z + eps * v) - f(z - eps * v)) / (2 * eps)
        n = float(Jv.norm())
        if not np.isfinite(n) or n == 0.0:
            return n
        rho, v = n, Jv / n
    return rho


def h_check(arm, val, tau_S: float, steps=(1, 2, 4), n_states: int = 8) -> dict:
    """For each h (RK4 steps per snapshot interval): S's entropy-violation rate on the validation
    rollouts (production per second below -tau_S, per step) and the near-dry stiffness check
    (h rho(J) < 2.5 at the encoded validation states with the smallest concentration, where
    -mu_v/T, and so the metriplectic term, is stiffest; formulation 5.4). Also the projection
    defects per step of every validation trajectory at the coarsest h."""
    from p3b.eval.rollout import rollout
    std, ae = arm.std, arm.ae
    X = val.X.reshape(-1, val.X.shape[-1])
    R = val.raw_inputs.reshape(X.shape[0], -1)
    F = std.encode_features(val.features.reshape(X.shape[0], -1))
    t = val.t.reshape(-1)
    dt = float(torch.median(t[1:] - t[:-1]))
    n_ch = 3 if val.system == "A" else 2
    dryness = X[:, 0:val.n_cells * n_ch:n_ch].min(dim=1).values          # smallest log c (A) or T_f (B)
    pick = torch.argsort(dryness)[:n_states]
    with torch.no_grad():
        Z = ae.encode(std.encode(X[pick]))
    out, defects = {"h": {}}, []
    for k in steps:
        viol = total = 0
        for j in range(val.X.shape[0]):
            r = rollout(arm, val, j, val.X.shape[1], k, track=True)
            if "S" in r.residuals:
                n = len(r.residuals["S"])
                tt = val.t[j].reshape(-1)
                d_t = torch.cat([tt[1:] - tt[:-1], tt[-1:] - tt[-2:-1]])[:n]
                viol += int(((r.residuals["S"] / d_t) < -tau_S).sum())
                total += n * k
            if k == min(steps):
                bad = sum(r.flags.get(key, 0) for key in ("rank_deficient", "kappa_binds", "projection_rank_deficient"))
                defects.append(bad / max(1, val.X.shape[1] * val.X.shape[2] * k))
        rhos = []
        for i in range(Z.shape[0]):
            f = lambda z: arm.dyn(z, F[pick[i]:pick[i] + 1], R[pick[i]:pick[i] + 1])[0]  # noqa: E731
            rhos.append(spectral_radius(f, Z[i:i + 1]))
        h_rho = (dt / k) * max(rhos)
        out["h"][k] = {"viol_rate": viol / max(1, total), "stiff_ok": bool(np.isfinite(h_rho) and h_rho < RK4_STABLE),
                       "h_rho": float(h_rho)}
    out["defects_per_step"] = defects
    return out
