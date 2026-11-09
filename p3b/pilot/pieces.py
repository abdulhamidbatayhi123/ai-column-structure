"""The pilot's Kaggle pieces (Plan 5 Task 6): one function per job of kaggle/pilot_jobs.py. Each reads
the pilot data under data_root/<key>/{train,validation} and the chosen values of the earlier phases
(stage: p3b.pilot.collect.stage), and writes one piece file into out. Pieces with losses or errors
are read only by p3b.pilot.collect, which writes chosen values only; nothing here prints a loss.

q holds overrides for tests on tiny data (grids, iterations, widths); the study runs with q = {}.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import torch

from p3b.pilot import runs

KEYS = {"A_MOF-303": ("A", "MOF-303"), "A_Al-fumarate": ("A", "Al-fumarate"), "B": ("B", None)}
PRIMARY_KEY = "A_MOF-303"
LATENT_ARMS = ("U", "U+F", "P", "S", "S-L", "S-M", "S-Pi", "FNO")
LR_ARMS = LATENT_ARMS + ("C+L",)
CURVE_MAX, CURVE_EVERY = 16_000, 1_000          # PREREG Section 10: every 1,000 iterations up to 16,000
LR_ITERATIONS = 2_000
TWO_SEEDS = (0, 1)


def files(data_root, key: str, split: str) -> list:
    from p3b.gates import load_meta
    fs = glob.glob(os.path.join(str(data_root), key, split, "*.npz"))
    return sorted(fs, key=lambda p: int(load_meta(p)["point"]["index"]))


def long_files(data_root, key: str) -> list:
    """Every trajectory of the key with the key's full cycle count (CSS and reference noise need the
    long ones: system A primary's pilot_val; all 8 of A_replication and B)."""
    fs = files(data_root, key, "train") + files(data_root, key, "validation")
    n = {p: int(np.load(p)["ports_cum"].shape[0]) for p in fs}
    top = max(n.values())
    return [p for p in fs if n[p] == top]


def _write(out, name: str, obj) -> str:
    os.makedirs(str(out), exist_ok=True)
    path = os.path.join(str(out), name)
    tmp = path + ".partial"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)
    return path


# ---- phase 0 -------------------------------------------------------------------------------------------
def piece_data(data_root, key: str, out, q: dict | None = None) -> str:
    """CSS per long trajectory, the decoder gate over the d grid, the reference noise."""
    q = q or {}
    lf = long_files(data_root, key)
    gate = runs.decoder_gate_grid(files(data_root, key, "train"), files(data_root, key, "validation"),
                                  q.get("d_grid", (16, 24, 32, 48, 64)), q.get("ae_epochs", 60), q.get("ae_width", 256))
    return _write(out, f"data_{key}.json", {"css": runs.css_cycles(lf), "decoder_gate": {str(k): v for k, v in gate.items()},
                                            "reference_noise": runs.reference_noise(lf)})


def piece_cplk(data_root, key: str, out, q: dict | None = None) -> str:
    q = q or {}
    sys_, mat = KEYS[key]
    eps = runs.cplusl_K(files(data_root, key, "validation"), mat, n_coarse=q.get("n_coarse", 32),
                        Ks=q.get("Ks", (1, 2, 4, 8)))
    return _write(out, f"cplusl_K_{key}.json", {str(k): v for k, v in eps.items()})


# ---- arms ------------------------------------------------------------------------------------------------
def _cell(arm: str, key: str, seed: int):
    from p3b.train.cells import Cell
    sys_, mat = KEYS[key]
    return Cell(arm, sys_, mat, "pilot", seed)


def _std(ds):
    from p3b.latent.data import Standardiser
    n_ch, n_out = (3, 2) if ds.system == "A" else (2, 1)
    return Standardiser.fit(ds.X, ds.n_cells, n_ch, n_out, features=ds.features)


def piece_ae(data_root, key: str, seed: int, stage: dict, out, q: dict | None = None) -> str:
    """The pilot autoencoder of (key, seed) at the chosen d, saved under the study's naming
    (ae_<system>_<material>_pilot_s<seed>.pt) so p3b.train.cells._build finds it."""
    from p3b.latent.autoencoder import MLPAutoencoder, train_autoencoder
    from p3b.latent.data import load_set
    q = q or {}
    ds = load_set(files(data_root, key, "train"))
    std = _std(ds)
    torch.manual_seed(seed)
    ae = MLPAutoencoder(ds.X.shape[-1], int(stage["d"][key]), width=q.get("ae_width", 256))
    train_autoencoder(ae, std.encode(ds.X), epochs=q.get("ae_epochs", 60), seed=seed)
    c = _cell("U", key, seed)
    os.makedirs(str(out), exist_ok=True)
    path = os.path.join(str(out), f"ae_{c.system}_{c.material}_{c.size}_s{seed}.pt")
    torch.save(ae.state_dict(), path)
    return path


def _kappa(key, seed, stage, ds, out, q):
    from p3b.latent.autoencoder import MLPAutoencoder
    from p3b.latent.functionals import LatentFunctionals
    from p3b.train.cells import _load_ae, _physics
    c = _cell("S", key, seed)
    std = _std(ds)
    ae = MLPAutoencoder(ds.X.shape[-1], int(stage["d"][key]), width=q.get("ae_width", 256))
    _load_ae(c, ae, str(out), _pilot(stage, key, None), {**q, "require_ae": True}, std, ds)
    lf = LatentFunctionals(ae.decode, std, _physics(c.system, c.material, ds), c.system, ds.n_cells)
    return runs.kappa_from_data(lf, ae, std, ds), runs.g2_from_data(lf, ae, std, ds)


def _pilot(stage, key, kappa):
    return {"d": stage["d"][key], "kappa": kappa, "ae_input_root": stage.get("ae_input_root", "/kaggle/input")}


def make_arm(arm: str, key: str, seed: int, stage: dict, ds, out, q: dict | None = None):
    """A fresh pilot arm built exactly as the study builds its cells (p3b.train.cells._build): matched
    widths, the shared autoencoder of (key, seed), S's kappa from the encoded pilot data."""
    from p3b.train.cells import _build
    q = q or {}
    kappa = _kappa(key, seed, stage, ds, out, q)[0] if arm in ("S", "S-L", "S-M", "S-Pi") else None
    built, _, _, _ = _build(_cell(arm, key, seed), _pilot(stage, key, kappa), {**q, "require_ae": True}, ds, str(out))
    return built


# ---- phase 1 and 2 ---------------------------------------------------------------------------------------
def piece_lr(arm: str, data_root, stage: dict, out, q: dict | None = None) -> str:
    from p3b.latent.data import load_set
    q = q or {}
    ds = load_set(files(data_root, PRIMARY_KEY, "train"))
    val = load_set(files(data_root, PRIMARY_KEY, "validation"))
    rates = q.get("lr_grid", (3e-4, 1e-3, 3e-3))
    its = q.get("lr_iterations", LR_ITERATIONS)
    nw, ns = q.get("n_windows", 8), q.get("n_steps", 20)
    if arm == "C+L":
        res = runs.cplusl_lr_sweep(ds, val, KEYS[PRIMARY_KEY][1], int(stage["K"][PRIMARY_KEY]), rates, its,
                                   n_coarse=q.get("n_coarse", 32), n_windows=nw, n_steps=ns)
    else:
        res = runs.lr_sweep(lambda: make_arm(arm, PRIMARY_KEY, 0, stage, ds, out, q), ds, val, rates, its,
                            h_steps=1, seed=0, n_windows=nw, n_steps=ns)
    return _write(out, f"lr_{_tag(arm)}.json", {str(k): v for k, v in res.items()})


def piece_curve(arm: str, data_root, stage: dict, out, q: dict | None = None) -> str:
    from p3b.latent.data import load_set
    q = q or {}
    ds = load_set(files(data_root, PRIMARY_KEY, "train"))
    val = load_set(files(data_root, PRIMARY_KEY, "validation"))
    its, every = q.get("curve_iterations", CURVE_MAX), q.get("curve_every", CURVE_EVERY)
    nw, ns = q.get("n_windows", 8), q.get("n_steps", 20)
    lr = stage["lr"][arm]
    if arm == "C+L":
        curve = runs.cplusl_curve(ds, val, KEYS[PRIMARY_KEY][1], lr, its, every, int(stage["K"][PRIMARY_KEY]), 0,
                                  n_coarse=q.get("n_coarse", 32), n_windows=nw, n_steps=ns)
    else:
        curve = runs.train_curve(make_arm(arm, PRIMARY_KEY, 0, stage, ds, out, q), ds, val, lr, its, every, 1, 0,
                                 nw, ns)
    return _write(out, f"curve_{_tag(arm)}.json", [list(x) for x in curve])


def _tag(arm: str) -> str:
    return arm.replace("+", "p").replace("-", "")


# ---- phase 3 ---------------------------------------------------------------------------------------------
def piece_s(key: str, data_root, stage: dict, out, q: dict | None = None) -> str:
    """S trained at its pilot lr and iterations with one RK4 step per snapshot interval on the key's
    pilot training data; the h check on its validation data with the reference's tau_S; the raw
    |P grad S|^2 values and the projection defects per step (threshold inputs)."""
    from p3b.latent.data import load_set
    from p3b.train.trainer import TrainConfig, train
    q = q or {}
    ds = load_set(files(data_root, key, "train"))
    val = load_set(files(data_root, key, "validation"))
    arm = make_arm("S", key, 0, stage, ds, out, q)
    train(arm, ds, TrainConfig(lr=stage["lr"]["S"], iterations=int(q.get("iterations", stage["iterations"])), seed=0,
                               h_steps=1, n_windows=q.get("n_windows", 8), n_steps=q.get("n_steps", 20)))
    hc = runs.h_check(arm, val, float(stage["tau_S"][key]), steps=q.get("h_steps_grid", (1, 2, 4)))
    g2 = _kappa(key, 0, stage, ds, out, q)[1]
    return _write(out, f"s_{key}.json", {"h_check": {str(k): v for k, v in hc["h"].items()}, "g2": g2,
                                         "defects_per_step": hc["defects_per_step"]})


def piece_wall(data_root, stage: dict, out, q: dict | None = None) -> str:
    from p3b.latent.data import load_set
    q = q or {}
    ds = load_set(files(data_root, PRIMARY_KEY, "train"))
    makers = {a: (lambda a=a: make_arm(a, PRIMARY_KEY, 0, stage, ds, out, q)) for a in ("U", "S")}
    res = runs.wall_per_iter(makers, ds, iterations=q.get("wall_iterations", 50), warmup=q.get("wall_warmup", 5),
                             n_windows=q.get("n_windows", 8), n_steps=q.get("n_steps", 20))
    return _write(out, "wall_per_iter.json", res)


# ---- phase 4 ---------------------------------------------------------------------------------------------
def piece_two(arm: str, seed: int, data_root, stage: dict, out, q: dict | None = None, quarter: bool = False) -> str:
    """One seed of U or S (or U at a quarter of the iterations: the positive control) on system A
    primary's pilot data at the chosen lr, iterations and h; evaluation rows on pilot_val."""
    from p3b.latent.data import load_set
    q = q or {}
    ds = load_set(files(data_root, PRIMARY_KEY, "train"))
    val = load_set(files(data_root, PRIMARY_KEY, "validation"))
    its = int(q.get("iterations", stage["iterations"]))
    its = max(1, its // 4) if quarter else its
    rows = runs.two_seed_rows(lambda: make_arm(arm, PRIMARY_KEY, seed, stage, ds, out, q), ds, val,
                              stage["lr"][arm], its, int(stage["h_steps"][PRIMARY_KEY]), seeds=(seed,),
                              n_windows=q.get("n_windows", 8), n_steps=q.get("n_steps", 20))[seed]
    name = f"rows_{'Uq' if quarter else _tag(arm)}_s{seed}.json"
    return _write(out, name, rows)
