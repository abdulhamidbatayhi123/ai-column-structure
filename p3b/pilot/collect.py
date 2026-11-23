"""Collecting the pilot's pieces (Plan 5 Task 6; revised after the plan 5 review).

- stage_data: the downloaded generation outputs laid out as data_root/<key>/{train,validation}
  (system A primary: pilot_train and pilot_val; A_replication and B: design indices 0-5 train, 6-7
  validation); a file name met twice is an error;
- load_piece: a piece found anywhere under the downloads (the queue nests them per kernel), the one
  whose recorded stage hash is the stage it was meant to run under; stale pieces of an earlier stage
  are never used (review I6, I7). Given the data root, a piece whose recorded data files differ from
  the current ones (or are gone) is refused too (T8 RS-4: regenerated pilot data);
- stage: after phase n, the chosen values the next phase needs (d, K, tau_S; lr; iterations;
  h_steps), by the rule functions of p3b.pilot.rules; chosen values only, and the values of later
  phases are dropped (review M2); tagged with PILOT_REVISION, without which kaggle/pilot_jobs.py and
  kaggle/pilot_chunk.py refuse the file (T8 fix review, R1-4);
- final: the ten input files of scripts/pilot_rules.py from the pieces.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
from pathlib import Path

from p3b.pilot import rules, runs
from p3b.pilot.pieces import KEYS, LR_ARMS, PILOT_REVISION, PRIMARY_KEY, TWO_SEEDS, _tag, restrict, stage_sha  # noqa: F401

SPLIT_TRAIN = 6                                   # A_replication and B: indices 0-5 train, 6-7 validation
_DESIGN_TO_KEY = {"A_primary": rules.pilot_key("A", "MOF-303"), "A_replication": rules.pilot_key("A", "Al-fumarate"),
                  "B": rules.pilot_key("B", None)}


def _phase(name: str) -> int:
    for prefix, p in (("data_", 0), ("cplusl_K_", 0), ("lr_", 1), ("curve_", 2), ("s_", 3), ("wall_", 3),
                      ("rows_", 4)):
        if name.startswith(prefix):
            return p
    raise ValueError(f"unknown piece {name!r}")


def stage_data(gen_dirs, data_root, link: bool = False) -> dict:
    """Copy every pilot npz under gen_dirs into data_root/<key>/<split>/; returns counts per key and split."""
    from p3b.gates import load_meta
    counts, seen = {}, {}
    for d in gen_dirs:
        for p in glob.glob(os.path.join(str(d), "**", "*.npz"), recursive=True):
            name = os.path.basename(p)
            if "_smoke" in name:
                continue
            design = load_meta(p).get("design_set", "")
            system, _, set_name = design.rpartition("_pilot")
            if not system:
                continue
            if name in seen:
                raise RuntimeError(f"duplicate pilot file {name}: {seen[name]} and {p}")
            seen[name] = p
            key = _DESIGN_TO_KEY[system]
            if system == "A_primary":
                split = {"_train": "train", "_val": "validation"}[set_name]
            else:
                split = "train" if int(load_meta(p)["point"]["index"]) < SPLIT_TRAIN else "validation"
            dst = os.path.join(str(data_root), key, split)
            os.makedirs(dst, exist_ok=True)
            target = os.path.join(dst, name)
            if link:
                try:
                    os.symlink(os.path.abspath(p), target)
                except OSError:                       # no symlink right (e.g. Windows without developer mode)
                    shutil.copy2(p, target)
            else:
                shutil.copy2(p, target)
            counts[(key, split)] = counts.get((key, split), 0) + 1
    return {f"{k}/{s}": n for (k, s), n in sorted(counts.items())}


_SHA_CACHE: dict = {}


def _current_sha(data_root, name: str):
    """SHA-256 of the data file `name` under data_root (searched recursively), None if it is not there
    exactly once; cached by path, size and modification time."""
    from p3b.pilot.pieces import _sha_file
    hits = glob.glob(os.path.join(str(data_root), "**", name), recursive=True)
    if len(hits) != 1:
        return None
    st = os.stat(hits[0])
    key = (os.path.abspath(hits[0]), st.st_size, st.st_mtime_ns)
    if key not in _SHA_CACHE:
        _SHA_CACHE[key] = _sha_file(hits[0])
    return _SHA_CACHE[key]


def _stale_data(obj: dict, data_root) -> list:
    """The data files a piece recorded whose current SHA-256 under data_root differs (or that are gone)."""
    used = (obj.get("meta", {}).get("used") or {}).get("data") or {}
    return sorted(f for f, sha in used.items() if _current_sha(data_root, f) != sha)


def _env_of(path, top) -> dict | None:
    """The versions recorded by the nearest ENV.json at or above `path`'s folder, up to `top`; None if none."""
    d, top = Path(path).resolve().parent, Path(top).resolve()
    while True:
        e = d / "ENV.json"
        if e.is_file():
            with open(e) as f:
                return json.load(f).get("got")
        if d == top or d == d.parent:
            return None
        d = d.parent


def _pinned(path, top, env_pins) -> bool:
    return env_pins is None or _env_of(path, top) == env_pins


def load_piece(pieces_dir, name: str, stage: dict, data_root=None, env_pins=None):
    """The value of piece `name` run under `stage` (already restricted to the piece's phase), searched
    recursively; FileNotFoundError if absent, RuntimeError if no copy or more than one copy matches.
    With `data_root` (the current data/pilot), a copy whose recorded data files differ from the current
    ones does not match. With `env_pins` ({package: version}), only a copy whose nearest ENV.json records
    exactly those versions matches (PREREG 11, the pinned environment)."""
    found = sorted(Path(pieces_dir).rglob(name))
    if not found:
        raise FileNotFoundError(f"piece {name} not found under {pieces_dir}")
    want = stage_sha(stage)
    match, stale, other_env = [], [], 0
    for p in found:
        if not _pinned(p, pieces_dir, env_pins):
            other_env += 1
            continue
        with open(p) as f:
            obj = json.load(f)
        if obj.get("meta", {}).get("stage_sha256") != want:
            continue
        bad = _stale_data(obj, data_root) if data_root is not None else []
        if bad:
            stale.append(f"{p} ({len(bad)} data files differ, e.g. {bad[0]})")
            continue
        match.append(obj["value"])
    if len(match) != 1:
        raise RuntimeError(f"piece {name}: {len(match)} of {len(found)} copies ran under the current stage "
                           f"(sha256 {want[:12]}) and data; rerun or remove the stale ones"
                           + (f"; other data: {'; '.join(stale)}" if stale else "")
                           + (f"; {other_env} ran outside the pinned environment {env_pins}" if other_env else ""))
    return match[0]


def _load(pieces_dir, name, stage, data_root=None, env_pins=None):
    return load_piece(pieces_dir, name, restrict(stage, _phase(name)), data_root, env_pins)


def stage(n: int, pieces_dir, previous: dict | None = None, keys=tuple(KEYS), data_root=None,
          env_pins=None) -> dict:
    """The stage after phase n (0 to 3): the previous stage's values of phases before n, plus phase n's.
    `data_root`: the current pilot data, against which every piece's data files are checked."""
    from p3b.latent.autoencoder import pilot_working_decoder
    s = restrict(previous or {}, n)
    if n == 0:
        s["d"], s["decoder"], s["tau_S"] = {}, {}, {}
        for key in keys:
            data = _load(pieces_dir, f"data_{key}.json", s, data_root, env_pins)
            # the pilot's working d (PREREG 10 does not set d; the full-data gate does, PREREG 4)
            s["decoder"][key], s["d"][key] = pilot_working_decoder(data["decoder_gate"],
                                                                   data.get("decoder_gate_conv", {}))
            s["tau_S"][key] = rules.tau_S_from(data["reference_noise"]["ref_production"])
        k = _load(pieces_dir, f"cplusl_K_{PRIMARY_KEY}.json", s, data_root, env_pins)
        s["K"] = {PRIMARY_KEY: rules.choose_K({int(a): v for a, v in k.items()})}
        if s["K"][PRIMARY_KEY] is None:
            raise RuntimeError("no admissible K: the pilot stops here (PREREG 10)")
    elif n == 1:
        s["lr"] = rules.choose_lr({arm: {float(k): v for k, v in
                                         _load(pieces_dir, f"lr_{_tag(arm)}.json", s, data_root, env_pins).items()}
                                   for arm in LR_ARMS})
        if any(v is None for v in s["lr"].values()):
            raise RuntimeError("an arm finished none of its learning-rate runs: the pilot stops here")
    elif n == 2:
        s["iterations"] = rules.choose_iterations(
            {arm: [tuple(x) for x in _load(pieces_dir, f"curve_{_tag(arm)}.json", s, data_root, env_pins)] for arm in LR_ARMS})
        if s["iterations"] is None:
            raise RuntimeError("no checkpoint where every arm's improvement fell below 2 %: the pilot stops here")
    elif n == 3:
        s["h_steps"] = {}
        for key in keys:
            hc = _load(pieces_dir, f"s_{key}.json", s, data_root, env_pins)["h_check"]
            s["h_steps"][key] = rules.choose_h_steps({int(k): v["viol_rate"] for k, v in hc.items()},
                                                     {int(k): v["stiff_ok"] for k, v in hc.items()})
        missing = [key for key in keys if s["h_steps"][key] is None]
        if missing:                                     # review M3: every system and material needs its h
            raise RuntimeError(f"no admissible h for {missing}: the pilot stops here")
    else:
        raise ValueError(f"stage {n}")
    s["_revision"] = PILOT_REVISION     # pilot_jobs.py and pilot_chunk.py refuse a stage file of another revision
    return s


def final(pieces_dir, inputs_dir, stage: dict, keys=tuple(KEYS), hyp_stat: dict | None = None,
          draws: int = 10_000, data_root=None, env_pins=None) -> list:
    """The ten input files of scripts/pilot_rules.py, written into inputs_dir; returns their paths.
    The defects are those of the chosen h of each key (review I4); a two-seed run with more than 5 %
    of its iterations skipped stops the pilot here (review I5). `data_root` as for stage."""
    data = {key: _load(pieces_dir, f"data_{key}.json", stage, data_root, env_pins) for key in keys}
    s_ = {key: _load(pieces_dir, f"s_{key}.json", stage, data_root, env_pins) for key in keys}
    two = {(a, sd): _load(pieces_dir, f"rows_{a}_s{sd}.json", stage, data_root, env_pins)
           for a in ("U", "S", "Uq") for sd in TWO_SEEDS}
    for (a, sd), piece in two.items():
        t = piece["train"]
        if runs.too_many_skipped(t, t["iterations"]):
            raise RuntimeError(f"{a} seed {sd}: more than 5 % of its training iterations were skipped "
                               f"({t['skipped']} + {t['skipped_grad']} of {t['iterations']}); the pilot stops here")
    u, s, uq = ({sd: two[(a, sd)]["rows"] for sd in TWO_SEEDS} for a in ("U", "S", "Uq"))
    sds = runs.sds_values(u, s, stat=hyp_stat)
    pc = runs.positive_control(u, uq, draws=draws, stat=hyp_stat)
    defects = {}
    for key in keys:
        h = stage["h_steps"].get(key)
        d = s_[key]["defects_per_step"].get(str(h)) if h is not None else None
        if d is None or any(x != x for x in d):
            raise RuntimeError(f"{key}: no defect counts at the chosen h ({h}); the pilot stops here")
        defects[key] = d
    out = {
        "lr_runs.json": {arm: _load(pieces_dir, f"lr_{_tag(arm)}.json", stage, data_root, env_pins) for arm in LR_ARMS},
        "curves.json": {arm: _load(pieces_dir, f"curve_{_tag(arm)}.json", stage, data_root, env_pins) for arm in LR_ARMS},
        "h_check.json": {key: s_[key]["h_check"] for key in keys},
        "threshold_inputs.json": {key: {"floors": data[key]["reference_noise"]["floors"],
                                        "ref_production": data[key]["reference_noise"]["ref_production"],
                                        "tau_ref": data[key]["reference_noise"]["tau_ref"],
                                        "g2": s_[key]["g2"], "defects_per_step": defects[key]}
                                  for key in keys},
        "cplusl_K.json": {PRIMARY_KEY: _load(pieces_dir, f"cplusl_K_{PRIMARY_KEY}.json", stage, data_root, env_pins)},
        "sds.json": sds,
        "css.json": {key: data[key]["css"] for key in keys},
        "decoder_gate.json": {key: data[key]["decoder_gate"] for key in keys},
        "decoder_gate_conv.json": {key: data[key].get("decoder_gate_conv", {}) for key in keys},
        "positive_control.json": pc,
        "wall_per_iter.json": _load(pieces_dir, "wall_per_iter.json", stage, data_root, env_pins),
    }
    os.makedirs(str(inputs_dir), exist_ok=True)
    paths = []
    for name, obj in out.items():
        p = os.path.join(str(inputs_dir), name)
        with open(p, "w") as f:
            json.dump(obj, f)
        paths.append(p)
    with open(os.path.join(str(inputs_dir), "pilot_notes.json"), "w") as f:   # beside the ten, not hashed by the rules
        json.dump(notes(u, s, hyp_stat), f, indent=1)
    return paths


CAVEATS = {
    "H5": "the H5 SD is taken in distribution, at cycle 20 of pilot_val: the pilot has no OOD set",
    "H1": "the H1 SD is taken at cycle 50; if the data-cost contingency shortens the long sets, H1's cycle "
          "changes and this SD is an approximation",
}


def notes(u_rows: dict, s_rows: dict, hyp_stat: dict | None = None) -> dict:
    """Beside sds.json (plan 5 review M8, M9): per hypothesis, how many (trajectory, seed) pairs had a
    capped value (an arm failed by the hypothesis's cycle; both failed gives a log ratio of exactly 0,
    which shrinks the SD), and the caveats of the H5 and H1 SDs."""
    stat = hyp_stat or runs.HYP_STAT
    capped = {}
    for h, (_, c) in stat.items():
        either = both = pairs = 0
        for sd in u_rows:
            su = {r["trajectory"]: r for r in s_rows[sd]}
            for r in u_rows[sd]:
                fu = r.get("failed_cycle") is not None and r["failed_cycle"] <= c
                fs_ = su[r["trajectory"]].get("failed_cycle")
                fs_ = fs_ is not None and fs_ <= c
                pairs += 1
                either += int(fu or fs_)
                both += int(fu and fs_)
        capped[h] = {"either": either, "both": both, "pairs": pairs}
    return {"capped_pairs": capped, "caveats": CAVEATS}


def collect_ae(pieces_dir, dst, env_pins=None) -> list:
    """The phase-1 autoencoders (ae_*_pilot_s*.pt) from the downloaded pieces into dst
    (results/pilot_ae/, which kaggle/make_kit.py puts in the kit for phases 2 to 4)."""
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    from p3b.pilot.pieces import _sha_file
    by_name = {}
    for p in sorted(Path(pieces_dir).rglob("ae_*_pilot_s*.pt")):
        if not _pinned(p, pieces_dir, env_pins):          # PREREG 11: only the pinned environment's autoencoders
            continue
        by_name.setdefault(p.name, set()).add((_sha_file(p), p))
    out = []
    for name, copies in sorted(by_name.items()):
        if len({sha for sha, _ in copies}) > 1:              # a stale autoencoder of an earlier stage
            raise RuntimeError(f"autoencoder {name}: {len(copies)} different downloads; remove the stale ones")
        shutil.copy2(next(iter(copies))[1], dst / name)
        out.append(dst / name)
    return out
