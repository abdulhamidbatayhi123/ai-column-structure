"""The pilot's Kaggle jobs by phase (Plan 5 Task 6; the table in docs/superpowers/plans/
2026-11-09-p3b-pilot-runner.md). Each phase is its own jobs file for kaggle/run_queue.py; the kit is
rebuilt with results/pilot_stage.json between phases (scripts/pilot_collect.py stage N).

    python kaggle/pilot_jobs.py PHASE OUT.json
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from p3b.pilot.pieces import KEYS, LR_ARMS, PRIMARY_KEY, TWO_SEEDS, _tag  # noqa: E402

KIT = "abdulhamidbatayhi/p3b-reference-kit"


def _job(piece: str, expect: str, depends_on=None) -> dict:
    name = "p3b-pilot-" + piece.lower().replace("_", "").replace("+", "p")
    job = {"name": name, "system": "pilot", "set": piece, "threads": 4,
           "commands": [["kaggle/pilot_chunk.py", "--piece", piece, "--out", "OUT"]], "expect": [expect]}
    if depends_on:
        job["depends_on"] = depends_on
    return job


def _ae(key: str, seed: int) -> dict:
    sys_, mat = KEYS[key]
    return _job(f"ae-{key}-s{seed}", f"ae_{sys_}_{mat}_pilot_s{seed}.pt")


def phase_jobs(phase: int, kit_slug: str = KIT) -> dict:
    jobs = []
    if phase == 0:
        jobs += [_job(f"data-{k}", f"data_{k}.json") for k in KEYS]
        jobs.append(_job(f"cplk-{PRIMARY_KEY}", f"cplusl_K_{PRIMARY_KEY}.json"))
    elif phase == 1:
        aes = [_ae(k, 0) for k in KEYS] + [_ae(PRIMARY_KEY, s) for s in TWO_SEEDS if s != 0]
        primary = _ae(PRIMARY_KEY, 0)["name"]
        jobs += aes + [_job(f"lr-{_tag(a)}", f"lr_{_tag(a)}.json", None if a == "C+L" else [primary])
                       for a in LR_ARMS]
    elif phase == 2:
        jobs += [_job(f"curve-{_tag(a)}", f"curve_{_tag(a)}.json") for a in LR_ARMS]
    elif phase == 3:
        jobs += [_job(f"s-{k}", f"s_{k}.json") for k in KEYS] + [_job("wall", "wall_per_iter.json")]
    elif phase == 4:
        jobs += [_job(f"two-{a}-s{s}", f"rows_{a}_s{s}.json") for a in ("U", "S") for s in TWO_SEEDS]
        jobs += [_job(f"pcq-s{s}", f"rows_Uq_s{s}.json") for s in TWO_SEEDS]
    else:
        raise ValueError(f"phase {phase}")
    return {"kit_slug": kit_slug, "nz": 0, "jobs": jobs}


if __name__ == "__main__":
    with open(sys.argv[2], "w") as f:
        json.dump(phase_jobs(int(sys.argv[1])), f, indent=1)
