"""Collecting the pilot's pieces (Plan 5 Task 6).

- stage_data: the downloaded generation outputs laid out as data_root/<key>/{train,validation}
  (system A primary: pilot_train and pilot_val; A_replication and B: design indices 0-5 train, 6-7
  validation);
- stage: after phase n, the chosen values the next phase needs (d, K, tau_S; lr; iterations;
  h_steps), by the rule functions of p3b.pilot.rules. Chosen values only: no loss or error;
- final: the ten input files of scripts/pilot_rules.py from the pieces.
"""
from __future__ import annotations

import glob
import json
import os
import shutil

from p3b.pilot import rules, runs
from p3b.pilot.pieces import KEYS, LR_ARMS, PRIMARY_KEY, TWO_SEEDS, _tag

SPLIT_TRAIN = 6                                   # A_replication and B: indices 0-5 train, 6-7 validation
_DESIGN_TO_KEY = {"A_primary": "A_MOF-303", "A_replication": "A_Al-fumarate", "B": "B"}


def stage_data(gen_dirs, data_root) -> dict:
    """Copy every pilot npz under gen_dirs into data_root/<key>/<split>/; returns counts per key and split."""
    from p3b.gates import load_meta
    counts = {}
    for d in gen_dirs:
        for p in glob.glob(os.path.join(str(d), "**", "*.npz"), recursive=True):
            name = os.path.basename(p)
            if "_smoke" in name:
                continue
            design = load_meta(p).get("design_set", "")
            system, _, set_name = design.rpartition("_pilot")
            if not system:
                continue
            key = _DESIGN_TO_KEY[system]
            if system == "A_primary":
                split = {"_train": "train", "_val": "validation"}[set_name]
            else:
                split = "train" if int(load_meta(p)["point"]["index"]) < SPLIT_TRAIN else "validation"
            dst = os.path.join(str(data_root), key, split)
            os.makedirs(dst, exist_ok=True)
            shutil.copy2(p, os.path.join(dst, name))
            counts[(key, split)] = counts.get((key, split), 0) + 1
    return {f"{k}/{s}": n for (k, s), n in sorted(counts.items())}


def _load(pieces_dir, name):
    with open(os.path.join(str(pieces_dir), name)) as f:
        return json.load(f)


def stage(n: int, pieces_dir, previous: dict | None = None, keys=tuple(KEYS)) -> dict:
    """The stage after phase n (0 to 3), built on the previous stage."""
    from p3b.latent.autoencoder import choose_d
    s = dict(previous or {})
    if n == 0:
        s["d"], s["tau_S"] = {}, {}
        for key in keys:
            data = _load(pieces_dir, f"data_{key}.json")
            s["d"][key] = choose_d({int(k): bool(v) for k, v in data["decoder_gate"].items()})
            s["tau_S"][key] = rules.tau_S_from(data["reference_noise"]["ref_production"])
        k = _load(pieces_dir, f"cplusl_K_{PRIMARY_KEY}.json")
        s["K"] = {PRIMARY_KEY: rules.choose_K({int(a): v for a, v in k.items()})}
        missing = [key for key in keys if s["d"][key] is None]
        if missing or s["K"][PRIMARY_KEY] is None:
            raise RuntimeError(f"no admissible d for {missing} or no K: the pilot stops here (PREREG 5.1, 10)")
    elif n == 1:
        s["lr"] = rules.choose_lr({arm: {float(k): v for k, v in _load(pieces_dir, f"lr_{_tag(arm)}.json").items()}
                                   for arm in LR_ARMS})
        if any(v is None for v in s["lr"].values()):
            raise RuntimeError("an arm finished none of its learning-rate runs: the pilot stops here")
    elif n == 2:
        s["iterations"] = rules.choose_iterations({arm: [tuple(x) for x in _load(pieces_dir, f"curve_{_tag(arm)}.json")]
                                                   for arm in LR_ARMS})
        if s["iterations"] is None:
            raise RuntimeError("no checkpoint where every arm's improvement fell below 2 %: the pilot stops here")
    elif n == 3:
        s["h_steps"] = {}
        for key in keys:
            hc = _load(pieces_dir, f"s_{key}.json")["h_check"]
            s["h_steps"][key] = rules.choose_h_steps({int(k): v["viol_rate"] for k, v in hc.items()},
                                                     {int(k): v["stiff_ok"] for k, v in hc.items()})
        if s["h_steps"][PRIMARY_KEY] is None:
            raise RuntimeError("no admissible h for system A primary: the pilot stops here")
    else:
        raise ValueError(f"stage {n}")
    return s


def final(pieces_dir, inputs_dir, keys=tuple(KEYS), hyp_stat: dict | None = None, draws: int = 10_000) -> list:
    """The ten input files of scripts/pilot_rules.py, written into inputs_dir; returns their paths."""
    data = {key: _load(pieces_dir, f"data_{key}.json") for key in keys}
    s_ = {key: _load(pieces_dir, f"s_{key}.json") for key in keys}
    u, s, uq = ({sd: _load(pieces_dir, f"rows_{a}_s{sd}.json") for sd in TWO_SEEDS} for a in ("U", "S", "Uq"))
    sds = runs.sds_values(u, s, stat=hyp_stat)
    pc = runs.positive_control(u, uq, draws=draws, stat=hyp_stat)
    out = {
        "lr_runs.json": {arm: _load(pieces_dir, f"lr_{_tag(arm)}.json") for arm in LR_ARMS},
        "curves.json": {arm: _load(pieces_dir, f"curve_{_tag(arm)}.json") for arm in LR_ARMS},
        "h_check.json": {key: s_[key]["h_check"] for key in keys},
        "threshold_inputs.json": {key: {"floors": data[key]["reference_noise"]["floors"],
                                        "ref_production": data[key]["reference_noise"]["ref_production"],
                                        "tau_ref": data[key]["reference_noise"]["tau_ref"],
                                        "g2": s_[key]["g2"], "defects_per_step": s_[key]["defects_per_step"]}
                                  for key in keys},
        "cplusl_K.json": {PRIMARY_KEY: _load(pieces_dir, f"cplusl_K_{PRIMARY_KEY}.json")},
        "sds.json": sds,
        "css.json": {key: data[key]["css"] for key in keys},
        "decoder_gate.json": {key: data[key]["decoder_gate"] for key in keys},
        "positive_control.json": pc,
        "wall_per_iter.json": _load(pieces_dir, "wall_per_iter.json"),
    }
    os.makedirs(str(inputs_dir), exist_ok=True)
    paths = []
    for name, obj in out.items():
        p = os.path.join(str(inputs_dir), name)
        with open(p, "w") as f:
            json.dump(obj, f)
        paths.append(p)
    return paths


def collect_ae(pieces_dir, dst) -> list:
    """The phase-1 autoencoders (ae_*_pilot_s*.pt) from the downloaded pieces into dst
    (results/pilot_ae/, which kaggle/make_kit.py puts in the kit for phases 2 to 4)."""
    from pathlib import Path
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    out = []
    for p in sorted(Path(pieces_dir).rglob("ae_*_pilot_s*.pt")):
        shutil.copy2(p, dst / p.name)
        out.append(dst / p.name)
    return out
