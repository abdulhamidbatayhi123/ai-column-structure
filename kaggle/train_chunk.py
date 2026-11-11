"""Run one study job inside a Kaggle kernel (Plan 4, Task 4): an autoencoder job (--ae NAME) or a
cell (--cell NAME), by the names of kaggle/train_jobs.py. Data: the kit's data/study/<system>_<material>
directory (train/ and test/ subdirectories of stored trajectories); pilot values: the kit's
results/pilot_values.json; d and the decoder: the kit's results/decoder_choice.json, from the full-data
decoder gate (--gate-ae jobs, then scripts/decoder_choice.py; PREREG 4 and 5.1), never from the pilot.
Outputs go to --out (the kernel's output directory).

    python kaggle/train_chunk.py --cell p3b-s-a-m303-full-s0 --out OUT [--hours 11] [--resume-from DIR]

A cell still training after --hours checkpoints and writes OUT/RESUME.json; a later kernel given that
output as input continues it with --resume-from.
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
                             gate_ae, gate_specs, job_name, train_cell)


def data_dir(c) -> str:
    return os.path.join(ROOT, "data", "study", pilot_key(c.system, c.material))


def decoder_for(c, choice: dict) -> tuple:
    """(decoder, d) of the cell's system and material from decoder_choice.json."""
    key = pilot_key(c.system, c.material)
    if key not in choice:
        raise KeyError(f"{key} is not in decoder_choice.json: run the full-data decoder gate first")
    v = choice[key]
    if v.get("not_testable"):
        raise RuntimeError(f"{key}: no decoder passes the full-data gate, so it is not testable (PREREG 4)")
    if v.get("needs_conv"):
        raise RuntimeError(f"{key}: no MLP d passed; run the conv gate wave before any cell (PREREG 4)")
    return v["decoder"], int(v["d"])


def train_ae(c, out: str, choice: dict) -> str:
    import torch
    from p3b.latent.autoencoder import make_autoencoder, train_autoencoder
    from p3b.latent.data import Standardiser, load_set
    files, _ = _train_files(data_dir(c), c.size, _pool_size(c, {}))
    ds = load_set(files)
    n_ch, n_out = (3, 2) if c.system == "A" else (2, 1)
    std = Standardiser.fit(ds.X, ds.n_cells, n_ch, n_out, features=ds.features)
    torch.manual_seed(c.seed)
    kind, d = decoder_for(c, choice)
    ae = make_autoencoder(kind, ds.n_cells, n_ch, n_out, d)
    train_autoencoder(ae, std.encode(ds.X), seed=c.seed)
    path = os.path.join(out, f"ae_{c.system}_{c.material}_{c.size}_s{c.seed}.pt")
    torch.save(ae.state_dict(), path)
    return path


def cell_pilot(c, pilot: dict, choice: dict | None = None) -> dict:
    """The pilot values one cell needs (per system and material where PREREG Section 10 says so), with d
    and the decoder from the full-data gate's choice (learned arms only)."""
    key = pilot_key(c.system, c.material)
    th = pilot["thresholds"][key]
    kind, d = decoder_for(c, choice or {}) if c.arm not in SOLVER_ARMS else (None, None)
    out = {"lr": pilot["lr"], "iterations": pilot["iterations"], "h_steps": pilot["h_steps"][key],
           "d": d, "decoder": kind, "kappa": th["kappa"], "tau_S": th["tau_S"], "K": pilot["K"].get(key),
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
    g.add_argument("--gate-ae", help="a full-data decoder-gate autoencoder (kaggle/train_jobs.py gate_jobs)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gate", default=os.path.join(ROOT, "results", "h3a_gate.json"))
    ap.add_argument("--pilot", default=os.path.join(ROOT, "results", "pilot_values.json"))
    ap.add_argument("--decoder-choice", default=os.path.join(ROOT, "results", "decoder_choice.json"))
    ap.add_argument("--resume-from", default=None, help="an earlier session's output (Kaggle kernel input)")
    ap.add_argument("--hours", type=float, default=11.0,
                    help="pause with a checkpoint and RESUME.json after so many hours (Kaggle sessions end at 12)")
    a = ap.parse_args(argv)
    import time
    deadline = time.monotonic() + a.hours * 3600
    os.makedirs(a.out, exist_ok=True)
    with open(a.pilot) as f:
        pilot = json.load(f)
    if a.gate_ae:
        spec = next(s for kind in ("mlp", "conv") for s in gate_specs(cells(), kind) if s["name"] == a.gate_ae)
        key = pilot_key(spec["system"], spec["material"])
        c = next(c for c in cells() if (c.system, c.material) == (spec["system"], spec["material"]))
        print(gate_ae(spec, data_dir(c), a.out, css_cycle=int(pilot["css"][key])), flush=True)
        return
    with open(a.decoder_choice) as f:
        choice = json.load(f)
    if a.ae:
        c = next(c for c in cells() if c.arm not in SOLVER_ARMS and ae_job_name(c) == a.ae)
        print(train_ae(c, a.out, choice), flush=True)
        return
    if a.evaluate:
        name, set_name = a.evaluate
        c = next(c for c in cells() if job_name(c) == name)
        with open(a.gate) as f:
            gate = json.load(f)
        print(evaluate_cell(c, set_name, data_dir(c), a.out, cell_pilot(c, pilot, choice), gate), flush=True)
        return
    c = next(c for c in cells() if job_name(c) == a.cell)
    print(train_cell(c, data_dir(c), a.out, cell_pilot(c, pilot, choice), deadline=deadline,
                     resume_from=a.resume_from),
          flush=True)


if __name__ == "__main__":
    main()
