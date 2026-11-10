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
import math
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
def kappa_from_data(lf, ae, std, ds, n: int = 256, seed: int = 0) -> float:
    """kappa = 1e-3 x the median |P grad S^|^2 over encoded pilot snapshots (PREREG Section 4). It
    needs no trained dynamics, so S's learning-rate sweep can use it."""
    return 1e-3 * float(np.median(g2_from_data(lf, ae, std, ds, n, seed)))


@torch.no_grad()
def g2_from_data(lf, ae, std, ds, n: int = 256, seed: int = 0) -> list:
    """|P grad S^|^2 at n encoded pilot snapshots (the raw values p3b.pilot.rules.thresholds takes)."""
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
    return [float(v) for v in info["P_gradS_sq"].reshape(-1)]


def validation_loss(arm, val, h_steps: int, n_cycles: int = 1) -> float:
    """MSE of the decoded standardised channels over the first n_cycles of every validation
    trajectory (the training loss's metric on a rollout); NaN if any rollout failed."""
    from p3b.eval import rollout as ro
    losses = []
    for j, r in enumerate(ro.rollouts(arm, val, n_cycles, h_steps)):
        if r.failed_cycle is not None:
            return float("nan")
        d = arm.std.encode(r.X) - arm.std.encode(val.X[j, :n_cycles])
        losses.append(float((d ** 2).mean()))
    return float(np.mean(losses))


def train_curve(arm, ds, val, lr: float, iterations: int, every: int, h_steps: int, seed: int,
                n_windows: int = 8, n_steps: int = 20) -> list:
    """Train with one cosine schedule over `iterations`, pausing every `every` iterations (a
    checkpoint) to record the pooled validation loss: [(iteration, loss), ...]. A checkpoint after more
    than SKIP_LIMIT of the iterations so far were skipped (diverged windows leave a barely trained,
    finite model) records NaN: not finished (plan 5 review I5)."""
    import tempfile

    from p3b.train import trainer
    curve = []
    with tempfile.TemporaryDirectory() as tmp:
        ck = os.path.join(tmp, "ck.pt")
        done = 0
        while done < iterations:
            step = min(every, iterations - done)
            hist = trainer.train(arm, ds, trainer.TrainConfig(lr=lr, iterations=iterations, seed=seed,
                                                              h_steps=h_steps, n_windows=n_windows,
                                                              n_steps=n_steps, checkpoint=ck, every=every,
                                                              stop_after=step))
            done += step
            loss = float("nan") if too_many_skipped(hist, done) else validation_loss(arm, val, h_steps)
            curve.append((done, loss))
    return curve


SKIP_LIMIT = 0.05      # PREREG 4: a run with more than 5 % of its iterations skipped is a deviation


def too_many_skipped(hist: dict, iterations: int) -> bool:
    return (hist.get("skipped", 0) + hist.get("skipped_grad", 0)) > SKIP_LIMIT * iterations


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


def snapshot_intervals(val) -> torch.Tensor:
    """The interval after each snapshot, flattened like val.X.reshape(-1, n) (trajectory, cycle,
    snapshot); a trajectory's last snapshot takes the interval before it. The two half-cycles have
    different lengths, so the intervals differ within and across trajectories."""
    out = []
    for j in range(val.t.shape[0]):
        tt = val.t[j].reshape(-1)
        out.append(torch.cat([tt[1:] - tt[:-1], tt[-1:] - tt[-2:-1]]))
    return torch.cat(out)


def h_check(arm, val, tau_S: float, steps=(1, 2, 4), n_states: int = 8) -> dict:
    """For each h (k RK4 steps per snapshot interval):
    - viol_rate: the share of snapshot intervals in which S's production per second falls below
      -tau_S (residuals and flags are per interval at every k, so the rate is per interval; plan 5
      review I2). A rollout that fails sets the rate to 1 (inadmissible) and is counted in "failed";
    - the near-dry stiffness check: h rho(J) < 2.5 with each picked state's own interval divided by k
      (review I3), at the encoded validation states with the smallest concentration, where -mu_v/T,
      and so the metriplectic term, is stiffest (formulation 5.4);
    - defects_per_step[k]: per validation trajectory, the share of intervals with any projection
      defect flag (counted once per interval), NaN for a failed rollout (review I4)."""
    from p3b.eval import rollout as ro
    std, ae = arm.std, arm.ae
    X = val.X.reshape(-1, val.X.shape[-1])
    R = val.raw_inputs.reshape(X.shape[0], -1)
    F = std.encode_features(val.features.reshape(X.shape[0], -1))
    dts = snapshot_intervals(val)
    n_ch = 3 if val.system == "A" else 2
    dryness = X[:, 0:val.n_cells * n_ch:n_ch].min(dim=1).values          # smallest log c (A) or T_f (B)
    pick = torch.argsort(dryness)[:n_states]
    with torch.no_grad():
        Z = ae.encode(std.encode(X[pick]))
    rhos = []
    for i in range(Z.shape[0]):
        f = lambda z: arm.dyn(z, F[pick[i]:pick[i] + 1], R[pick[i]:pick[i] + 1])[0]  # noqa: E731
        rhos.append(spectral_radius(f, Z[i:i + 1]))
    out = {"h": {}, "defects_per_step": {}}
    n_int = val.X.shape[1] * val.X.shape[2]
    for k in steps:
        viol = total = n_failed = 0
        defects = []
        for j, r in enumerate(ro.rollouts(arm, val, val.X.shape[1], k, track=True)):
            failed = r.failed_cycle is not None
            n_failed += int(failed)
            if "S" in r.residuals:
                n = len(r.residuals["S"])
                tt = val.t[j].reshape(-1)
                d_t = torch.cat([tt[1:] - tt[:-1], tt[-1:] - tt[-2:-1]])[:n]
                viol += int(((r.residuals["S"] / d_t) < -tau_S).sum())
                total += n
            defects.append(float("nan") if failed else r.flags.get("any_defect", 0) / n_int)
        h_rho = max(float(dts[pick[i]]) / k * rhos[i] for i in range(len(rhos)))
        out["h"][k] = {"viol_rate": 1.0 if n_failed else viol / max(1, total), "failed": n_failed,
                       "stiff_ok": bool(np.isfinite(h_rho) and h_rho < RK4_STABLE), "h_rho": float(h_rho)}
        out["defects_per_step"][k] = defects
    return out


# ---- Task 4: C+L's substeps K, wall-clock per iteration ----------------------------------------------
def cplusl_K(val_paths, material: str, n_coarse: int = 32, Ks=(1, 2, 4, 8)) -> dict:
    """{K: k_discrepancy, the largest |W(C+L) - W(C)| / |W_ref| over the pilot trajectories and
    cycles}, with C+L's multipliers all fixed at 1 and C (BDF) on the same n_coarse cells (PREREG
    Section 10): "reproduces C's eps_W within 1 %" read as the two releases within 1 % of the reference
    release (p3b.pilot.rules.choose_K). A run of either that fails makes the value infinite."""
    import tempfile

    from p3b.eval.evaluate import evaluate_files
    from p3b.gates import load_meta
    from p3b.latent.data import load_set
    from p3b.solvers.c_plus_l import Multipliers
    from p3b.solvers.coarse import run_c
    from p3b.train import cplusl
    ds = load_set(val_paths)
    phys = _physics(ds)
    nc, ns, n_cycles = ds.n_cells, ds.X.shape[2], ds.X.shape[1]
    points = [load_meta(p)["point"] for p in val_paths]
    unit = Multipliers(n_out=3, constant=1.0)
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        c_files = []
        for j, pt in enumerate(points):
            p = os.path.join(tmp, f"C_{j}.npz")
            c_files.append(p if run_c(ds.system, pt, material, N_c=n_coarse, path=p, n_snap=ns, n_target=nc).ok
                           else None)
        for K in Ks:
            files = []
            for j, pt in enumerate(points):
                p = os.path.join(tmp, f"CL_{K}_{j}.npz")
                try:
                    cplusl.rollout_cplusl_to_file(unit, pt, material, n_coarse, nc, ns, int(K), p)
                    files.append(p)
                except RuntimeError:                             # Newton failure
                    files.append(None)
            if None in files or None in c_files:
                out[int(K)] = float("inf")
                continue
            out[int(K)] = k_discrepancy(evaluate_files({"C": c_files, "C+L": files}, ds, phys, n_cycles, seed=0))
    return out


def k_discrepancy(rows) -> float:
    """The largest |W(C+L) - W(C)| / |W_ref| over trajectories and cycles: the released water of the
    two compared directly, so opposite-signed errors cannot cancel (plan 5 review I1); infinite if a
    row of either failed (its errors substitute reference values for non-finite snapshots)."""
    by = {(r["arm"], r["trajectory"]): r for r in rows}
    worst = 0.0
    for (arm, j), r in by.items():
        if arm != "C+L":
            continue
        c = by[("C", j)]
        if r["failed_cycle"] is not None or c["failed_cycle"] is not None:
            return math.inf
        d = np.abs(np.asarray(r["W_pred"]) - np.asarray(c["W_pred"])) / np.abs(np.asarray(r["W_ref"]))
        if not np.isfinite(d).all():
            return math.inf
        worst = max(worst, float(d.max()))
    return worst


def wall_per_iter(make_arms: dict, ds, iterations: int = 50, warmup: int = 5, h_steps: int = 1, lr: float = 1e-3,
                  seed: int = 0, n_windows: int = 8, n_steps: int = 20) -> dict:
    """{arm: training seconds per iteration} for a fresh arm per maker, timed after `warmup`
    iterations, on the machine and threads the study trains on (Kaggle CPU, one thread per run). U and S set
    U-long's iteration count (p3b.pilot.rules.iterations_ulong)."""
    import time

    from p3b.train.trainer import TrainConfig, train
    out = {}
    for name, make in make_arms.items():
        arm = make()
        if warmup:
            train(arm, ds, TrainConfig(lr=lr, iterations=warmup, seed=seed, h_steps=h_steps,
                                       n_windows=n_windows, n_steps=n_steps))
        t0 = time.perf_counter()
        hist = train(arm, ds, TrainConfig(lr=lr, iterations=iterations, seed=seed, h_steps=h_steps,
                                          n_windows=n_windows, n_steps=n_steps))
        out[name] = (time.perf_counter() - t0) / iterations
        if hist["skipped"] + hist["skipped_grad"] > iterations / 2:   # diverged windows cost almost nothing
            raise RuntimeError(f"{name}: most timed iterations were skipped; the wall-clock is not a training cost")
    return out


# ---- Task 5: the pilot SDs, the positive control -----------------------------------------------------
# The statistic of each hypothesis whose SD the power simulation and the H4 margin need (PREREG
# Sections 6 and 9), on pilot_val (16 trajectories, 60 cycles). In the pilot, H2 and H4 share their
# models: pilot_train holds 32 trajectories, so full pilot data is size 32 (the SD at 32 is the larger,
# so H4's margin is conservative). H5 has no pilot OOD set; its SD is taken at cycle 20 of pilot_val.
HYP_STAT = {"H1": ("eps", 50), "H5": ("eps", 20), "H2": ("e", 6), "H4": ("e", 6)}


def hypothesis_error(row: dict, hyp: str, stat: dict | None = None) -> float:
    """eps_W at cycle c (H1, H5) or the mean e over cycles 1..c (H2, H4), from a capped evaluation row;
    stat overrides HYP_STAT's cycles (tests on short data only)."""
    key, c = (stat or HYP_STAT)[hyp]
    vals = row[key]
    if len(vals) < c:
        raise ValueError(f"{hyp} needs {c} cycles, the row has {len(vals)}")
    return float(vals[c - 1]) if key == "eps" else float(np.mean(vals[:c]))


def _log_ratios(arm_rows: dict, base_rows: dict, hyp: str, stat: dict | None = None) -> list:
    if set(arm_rows) != set(base_rows):
        raise ValueError(f"seeds differ: {sorted(arm_rows)} against {sorted(base_rows)}")
    seeds = sorted(base_rows)
    by = {s: ({r["trajectory"]: r for r in arm_rows[s]}, {r["trajectory"]: r for r in base_rows[s]}) for s in seeds}
    trajs = sorted(by[seeds[0]][1])
    v = np.array([[math.log(hypothesis_error(by[s][0][t], hyp, stat) / hypothesis_error(by[s][1][t], hyp, stat))
                   for s in seeds]
                  for t in trajs])
    if not np.isfinite(v).all():
        raise ValueError(f"{hyp}: a non-finite log error ratio")
    return v.tolist()


def sds_values(u_rows: dict, s_rows: dict, stat: dict | None = None) -> dict:
    """{hypothesis: [[log(err_S / err_U) per seed] per trajectory]} from {seed: evaluation rows} of U
    and S (seed k of S paired with seed k of U), the input of p3b.pilot.rules.upper80_sd."""
    return {h: _log_ratios(s_rows, u_rows, h, stat) for h in HYP_STAT}


def positive_control(full_rows: dict, quarter_rows: dict, hyp: str = "H2", alpha: float = 0.05 / 3,
                     draws: int = 10_000, seed: int = 0, stat: dict | None = None) -> dict:
    """U at full iterations against U at a quarter of them through the full directional rule (PREREG
    Section 9) on the H2 statistic at H2's Holm worst-case level 0.05/3, the level at which the rule's
    power is judged (plan 5 review M1). Only the word is written (blinding)."""
    from p3b.pilot.power import directional_rule
    word = directional_rule(_log_ratios(full_rows, quarter_rows, hyp, stat), alpha, draws, seed)
    return {"word": word, "hypothesis": hyp, "alpha": alpha}


def two_seed_rows(make_arm, ds, val, lr: float, iterations: int, h_steps: int, seeds=(0, 1), n_cycles=None,
                  floors=None, tau_S=None, n_windows: int = 8, n_steps: int = 20) -> dict:
    """({seed: p3b.eval.evaluate rows on val}, {seed: {"iterations", "skipped", "skipped_grad"}}) for
    a fresh arm per seed trained at the pilot's lr and iterations; the rows never leave the pilot
    results directory (blinding); the training counts let the collector stop on an unfinished run
    (review I5)."""
    from p3b.eval import evaluate as ev
    from p3b.train import trainer
    phys = _physics(val)
    n_cycles = n_cycles or val.X.shape[1]
    floors = floors or {"N": float("inf"), "E": float("inf")}
    tau_S = float("inf") if tau_S is None else tau_S
    out, info = {}, {}
    for s in seeds:
        torch.manual_seed(s)
        arm = make_arm()
        hist = trainer.train(arm, ds, trainer.TrainConfig(lr=lr, iterations=iterations, seed=s, h_steps=h_steps,
                                                          n_windows=n_windows, n_steps=n_steps))
        info[int(s)] = {"iterations": int(iterations), "skipped": int(hist.get("skipped", 0)),
                        "skipped_grad": int(hist.get("skipped_grad", 0))}
        out[int(s)] = ev.evaluate(arm, val, phys, n_cycles, h_steps, floors, tau_S, s)
    return out, info


# ---- Task 6 building block: C+L's curve and learning-rate sweep -------------------------------------
def cplusl_curve(ds, val, material: str, lr: float, iterations: int, every: int, K: int, seed: int,
                 n_coarse: int = 32, n_windows: int = 8, n_steps: int = 20) -> list:
    """C+L trained with one cosine schedule over `iterations`; its deterministic validation loss
    (p3b.train.cplusl.cplusl_val_loss) every `every` iterations: [(iteration, loss), ...]."""
    from p3b.solvers.c_plus_l import Multipliers
    from p3b.train import cplusl
    torch.manual_seed(seed)
    mult = Multipliers(n_out=3)
    curve = []

    def record(it, m, hist):
        loss = float("nan") if too_many_skipped(hist, it) else cplusl.cplusl_val_loss(m, val, material, n_coarse, K)
        curve.append((it, loss))
    cplusl.train_cplusl(mult, ds, material, n_coarse, lr, iterations, seed, n_windows=n_windows, n_steps=n_steps,
                        K=K, every=every, callback=record)
    return curve


def cplusl_lr_sweep(ds, val, material: str, K: int, rates=(3e-4, 1e-3, 3e-3), iterations: int = 2000, seed: int = 0,
                    n_coarse: int = 32, n_windows: int = 8, n_steps: int = 20) -> dict:
    """{lr: C+L's validation loss after `iterations`} for fresh multipliers per rate."""
    return {lr: cplusl_curve(ds, val, material, lr, iterations, iterations, K, seed, n_coarse, n_windows,
                             n_steps)[-1][1] for lr in rates}
