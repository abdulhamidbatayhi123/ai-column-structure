"""Run one study job inside a Kaggle kernel (Plan 4, Task 4): an autoencoder job (--ae NAME) or a
cell (--cell NAME), by the names of kaggle/train_jobs.py. Data: the kit's data/study/<system>_<material>
directory (train/ and test/ subdirectories of stored trajectories); pilot values: the kit's
results/pilot_values.json. Outputs go to --out (the kernel's output directory).

    python kaggle/train_chunk.py --cell p3b-s-a-m303-full-s0 --out OUT
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]
sys.path.insert(0, ROOT)
if os.path.isdir(os.path.join(ROOT, "paper1")):            # inside the kit
    os.environ.setdefault("P3B_PAPER1_DIR", os.path.join(ROOT, "paper1"))

from p3b.pilot.rules import pilot_key  # noqa: E402
from p3b.train.cells import (SOLVER_ARMS, _pool_size, _train_files, ae_job_name, cells, evaluate_cell,  # noqa: E402
                             job_name, train_cell)


def data_dir(c) -> str:
    return os.path.join(ROOT, "data", "study", pilot_key(c.system, c.material))


def train_ae(c, out: str, pilot: dict) -> str:
    import torch
    from p3b.latent.autoencoder import MLPAutoencoder, train_autoencoder
    from p3b.latent.data import Standardiser, load_set
    files, _ = _train_files(data_dir(c), c.size, _pool_size(c, {}))
    ds = load_set(files)
    n_ch, n_out = (3, 2) if c.system == "A" else (2, 1)
    std = Standardiser.fit(ds.X, ds.n_cells, n_ch, n_out, features=ds.features)
    torch.manual_seed(c.seed)
    ae = MLPAutoencoder(ds.X.shape[-1], int(pilot["d"][pilot_key(c.system, c.material)]))
    train_autoencoder(ae, std.encode(ds.X), seed=c.seed)
    path = os.path.join(out, f"ae_{c.system}_{c.material}_{c.size}_s{c.seed}.pt")
    torch.save(ae.state_dict(), path)
    return path


def cell_pilot(c, pilot: dict) -> dict:
    """The pilot values one cell needs (per system and material where PREREG Section 10 says so)."""
    key = pilot_key(c.system, c.material)
    th = pilot["thresholds"][key]
    out = {"lr": pilot["lr"], "iterations": pilot["iterations"], "h_steps": pilot["h_steps"][key],
           "d": pilot["d"][key], "kappa": th["kappa"], "tau_S": th["tau_S"], "K": pilot["K"].get(key),
           "floors": {k[len("violation_"):]: v / 10.0 for k, v in th.items() if k.startswith("violation_")}}
    if c.arm == "U-long":
        if "iterations_ulong" not in pilot:
            raise KeyError("iterations_ulong (S's measured wall-clock) is required for U-long")
        out["iterations_ulong"] = pilot["iterations_ulong"]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ae")
    g.add_argument("--cell")
    g.add_argument("--evaluate", nargs=2, metavar=("CELL", "SET"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate", default=os.path.join(ROOT, "results", "h3a_gate.json"))
    ap.add_argument("--pilot", default=os.path.join(ROOT, "results", "pilot_values.json"))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    with open(a.pilot) as f:
        pilot = json.load(f)
    if a.ae:
        c = next(c for c in cells() if c.arm not in SOLVER_ARMS and ae_job_name(c) == a.ae)
        print(train_ae(c, a.out, pilot), flush=True)
        return
    if a.evaluate:
        name, set_name = a.evaluate
        c = next(c for c in cells() if job_name(c) == name)
        with open(a.gate) as f:
            gate = json.load(f)
        print(evaluate_cell(c, set_name, data_dir(c), a.out, cell_pilot(c, pilot), gate), flush=True)
        return
    c = next(c for c in cells() if job_name(c) == a.cell)
    print(train_cell(c, data_dir(c), a.out, cell_pilot(c, pilot)), flush=True)


if __name__ == "__main__":
    main()
