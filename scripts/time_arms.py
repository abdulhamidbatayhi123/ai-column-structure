"""Training and rollout cost of every latent arm where the study runs (Kaggle CPU), before the pilot.

The laptop's timings are unreliable under load (docs/compute_3B.md); this measures, at the study's
stored resolution (128 cells, 200 snapshots per cycle, d = 16): seconds per training iteration
(8 windows x 20 snapshot intervals) of every latent arm at h = 1 and 2 RK4 steps per interval, the
skipped iterations, and S's seconds per rollout interval, each on 4 threads and on 1 thread (the job
layout: one 4-thread job or four 1-thread jobs per session). It generates its own small data set
(3 trajectories, 2 cycles); it reports costs only, never an error value.

    python scripts/time_arms.py --out OUT/time_arms.json [--quick]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]
sys.path.insert(0, ROOT)
if os.path.isdir(os.path.join(ROOT, "paper1")):            # inside the kit
    os.environ.setdefault("P3B_PAPER1_DIR", os.path.join(ROOT, "paper1"))

import torch  # noqa: E402

from p3b.eval.rollout import rollout  # noqa: E402
from p3b.generate import generate_trajectory  # noqa: E402
from p3b.latent.data import load_set  # noqa: E402
from p3b.pilot import pieces  # noqa: E402
from p3b.provenance import git_commit  # noqa: E402
from p3b.train.trainer import TrainConfig, train  # noqa: E402

KEY = pieces.PRIMARY_KEY


def make_data(root: str, n_z: int, n_c: int, n_snap: int) -> dict:
    out = {}
    for split, n in (("train", 2), ("validation", 1)):
        d = os.path.join(root, KEY, split)
        os.makedirs(d, exist_ok=True)
        for i in range(n):
            pt = dict(RH_ads=0.32 + 0.06 * i + (0.03 if split == "validation" else 0.0), T_ads=303.15,
                      T_des=353.15, v=0.10, alpha=0.8, beta=0.6, index=i, n_cycles=2)
            t0 = time.process_time()
            ok = generate_trajectory("A", pt, "MOF-303", N_z=n_z, N_c=n_c, path=os.path.join(d, f"t{i}.npz"),
                                     n_snap=n_snap).ok
            if not ok:
                raise RuntimeError(f"timing data: trajectory {split} {i} failed")
            out[f"{split}_{i}_cpu_s"] = time.process_time() - t0
    return out


def time_arms(root: str, q: dict, arms, threads=(4, 1), steps=(1, 2), iterations: int = 10, warmup: int = 2) -> dict:
    ds = load_set(pieces.files(root, KEY, "train"))
    val = load_set(pieces.files(root, KEY, "validation"))
    out_dir = tempfile.mkdtemp()
    stage = {"d": {KEY: int(q.get("d", 16))}}
    pieces.piece_ae(root, KEY, 0, stage, out_dir, q)
    res = {"train": {}, "rollout_S": {}}
    for th in threads:
        torch.set_num_threads(th)
        for k in steps:
            for arm_name in arms:
                arm = pieces.make_arm(arm_name, KEY, 0, stage, ds, out_dir, q)
                cfg = dict(lr=1e-3, seed=0, h_steps=k, n_windows=q.get("n_windows", 8), n_steps=q.get("n_steps", 20))
                train(arm, ds, TrainConfig(iterations=warmup, **cfg))
                w0, c0 = time.perf_counter(), time.process_time()
                h = train(arm, ds, TrainConfig(iterations=iterations, **cfg))
                res["train"][f"{arm_name}|h{k}|t{th}"] = {
                    "wall_s_per_iteration": (time.perf_counter() - w0) / iterations,
                    "cpu_s_per_iteration": (time.process_time() - c0) / iterations,
                    "skipped": h["skipped"] + h["skipped_grad"], "iterations": iterations}
                print(f"timed {arm_name} h{k} t{th}", flush=True)
            arm = pieces.make_arm("S", KEY, 0, stage, ds, out_dir, q)
            w0 = time.perf_counter()
            r = rollout(arm, val, 0, 1, k)
            n_int = val.X.shape[2]
            res["rollout_S"][f"h{k}|t{th}"] = {"wall_s_per_interval": (time.perf_counter() - w0) / n_int,
                                               "failed_cycle": r.failed_cycle}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true", help="tiny settings for tests only")
    a = ap.parse_args(argv)
    if a.quick:
        n_z, n_c, n_snap, q, arms, its = 32, 8, 40, {"d": 4, "ae_epochs": 1, "ae_width": 8, "width": 8, "n_windows": 1,
                                                        "n_steps": 2}, ("U", "S"), 1
    else:
        n_z, n_c, n_snap, q, arms, its = 512, 128, 200, {"d": 16}, pieces.LATENT_ARMS, 10
    data = tempfile.mkdtemp()
    gen = make_data(data, n_z, n_c, n_snap)
    res = time_arms(data, q, arms, iterations=its, warmup=1 if a.quick else 2)
    res.update({"generation_cpu_s": gen, "settings": {"N_z": n_z, "N_c": n_c, "n_snap": n_snap, "q": q},
                "commit": open(os.path.join(ROOT, "kit_commit.txt")).read().strip()
                if os.path.exists(os.path.join(ROOT, "kit_commit.txt")) else git_commit(),
                "torch": torch.__version__, "cpu_count": os.cpu_count()})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(os.path.basename(a.out), flush=True)


if __name__ == "__main__":
    main()
