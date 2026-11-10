"""The pilot's Kaggle jobs by phase (Plan 5 Task 6; the table in docs/superpowers/plans/
2026-11-09-p3b-pilot-runner.md). Each phase is its own jobs file for kaggle/run_queue.py; the kit is
rebuilt with results/pilot_stage.json between phases (scripts/pilot_collect.py stage N). Kernel
names end in the first 6 hex digits of the SHA-256 of the stage the phase runs under, so a phase
rerun after a stage change pushes new kernels instead of reusing finished ones (plan 5 review I6).

    python kaggle/pilot_jobs.py PHASE OUT.json [results/pilot_stage.json]
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from p3b.pilot.pieces import KEYS, LR_ARMS, PRIMARY_KEY, TWO_SEEDS, _tag, restrict, stage_sha  # noqa: E402

KIT = "abdulhamidbatayhi/p3b-reference-kit"


def _name(piece: str, suffix: str) -> str:
    return "p3b-pilot-" + piece.lower().replace("_", "").replace("+", "p") + "-" + suffix


def _job(piece: str, expect: str, suffix: str, depends_on=None, threads: int = 1) -> dict:
    """One pilot piece per kernel, on one thread (Kaggle, results/timing_arms.json: 4 threads are no faster
    for the arms' small tensors and slower for the structured ones); autoencoders keep 4 (batched MLPs)."""
    name = _name(piece, suffix)
    job = {"name": name, "system": "pilot", "set": piece, "threads": threads,
           "commands": [["kaggle/pilot_chunk.py", "--piece", piece, "--out", "OUT"]], "expect": [expect]}
    if depends_on:
        job["depends_on"] = depends_on
    return job


def _ae(key: str, seed: int, suffix: str) -> dict:
    sys_, mat = KEYS[key]
    return _job(f"ae-{key}-s{seed}", f"ae_{sys_}_{mat}_pilot_s{seed}.pt", suffix, threads=4)


def phase_jobs(phase: int, stage: dict | None = None, kit_slug: str = KIT) -> dict:
    x = stage_sha(restrict(stage or {}, phase))[:6]
    jobs = []
    if phase == 0:
        jobs += [_job(f"data-{k}", f"data_{k}.json", x) for k in KEYS]
        jobs.append(_job(f"cplk-{PRIMARY_KEY}", f"cplusl_K_{PRIMARY_KEY}.json", x))
    elif phase == 1:
        aes = [_ae(k, 0, x) for k in KEYS] + [_ae(PRIMARY_KEY, s, x) for s in TWO_SEEDS if s != 0]
        primary = _ae(PRIMARY_KEY, 0, x)["name"]
        jobs += aes + [_job(f"lr-{_tag(a)}", f"lr_{_tag(a)}.json", x, None if a == "C+L" else [primary])
                       for a in LR_ARMS]
    elif phase == 2:
        jobs += [_job(f"curve-{_tag(a)}", f"curve_{_tag(a)}.json", x) for a in LR_ARMS]
    elif phase == 3:
        jobs += [_job(f"s-{k}", f"s_{k}.json", x) for k in KEYS] + [_job("wall", "wall_per_iter.json", x)]
    elif phase == 4:
        jobs += [_job(f"two-{a}-s{s}", f"rows_{a}_s{s}.json", x) for a in ("U", "S") for s in TWO_SEEDS]
        jobs += [_job(f"pcq-s{s}", f"rows_Uq_s{s}.json", x) for s in TWO_SEEDS]
    else:
        raise ValueError(f"phase {phase}")
    return {"kit_slug": kit_slug, "nz": 0, "jobs": jobs}


if __name__ == "__main__":
    stage_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ROOT, "results", "pilot_stage.json")
    st = {}
    if os.path.exists(stage_path):
        with open(stage_path) as f:
            st = json.load(f)
    with open(sys.argv[2], "w") as f:
        json.dump(phase_jobs(int(sys.argv[1]), st), f, indent=1)
