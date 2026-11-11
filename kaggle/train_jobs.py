"""The study's Kaggle jobs (Plan 4, Task 4): one autoencoder job per (system, material, size, seed)
shared by the learned cells, then one job per cell, depending on its autoencoder job (its outputs
are attached as kernel inputs). Solver cells (C, MUSCL-C, C+L) need no autoencoder.

    python kaggle/train_jobs.py OUT.json          # writes a jobs file for kaggle/run_queue.py
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from p3b.train.cells import GATED_SETS, LEARNED_WITH_AE, SOLVER_ARMS, ae_job_name, cells, gate_specs, job_name  # noqa: E402

KIT = "abdulhamidbatayhi/p3b-reference-kit"


def build_jobs(cs, kit_slug: str = KIT) -> dict:
    ae_jobs, jobs = {}, []
    for c in cs:
        if c.arm not in LEARNED_WITH_AE:
            continue
        n = ae_job_name(c)
        if n not in ae_jobs:
            ae_jobs[n] = {"name": n, "system": "train", "set": n,
                          "commands": [["kaggle/train_chunk.py", "--ae", n, "--out", "OUT"]], "threads": 4,
                          "expect": [f"ae_{c.system}_{c.material}_{c.size}_s{c.seed}.pt"]}
    for c in cs:
        n = job_name(c)
        job = {"name": n, "system": "train", "set": n,
               "commands": [["kaggle/train_chunk.py", "--cell", n, "--out", "OUT"]], "expect": [f"{n}.json"],
               "threads": 4, "resumable": True}           # past --hours: RESUME.json, run_queue continues it
        if c.arm in LEARNED_WITH_AE:
            job["depends_on"] = [ae_job_name(c)]
        jobs.append(job)
    return {"kit_slug": kit_slug, "nz": 0, "jobs": list(ae_jobs.values()) + jobs}


def gate_jobs(cs, kind: str = "mlp", keys=None, kit_slug: str = KIT) -> dict:
    """The full-data decoder gate (PREREG 4, 5.1), run before any cell: one job per gate autoencoder of
    one decoder kind; keys (pilot keys, e.g. "B") restricts it, as the conv wave runs only for keys where
    no MLP d passed (scripts/decoder_choice.py reports them)."""
    from p3b.pilot.rules import pilot_key
    jobs = [{"name": s["name"], "system": "train", "set": s["name"], "threads": 4,
             "commands": [["kaggle/train_chunk.py", "--gate-ae", s["name"], "--out", "OUT"]],
             "expect": [f"gate_{s['name']}.json"]}
            for s in gate_specs(cs, kind) if keys is None or pilot_key(s["system"], s["material"]) in keys]
    return {"kit_slug": kit_slug, "nz": 0, "jobs": jobs}


def evaluation_jobs(cs, sets=GATED_SETS, kit_slug: str = KIT) -> dict:
    """The second phase, built only after H3a passed (results/h3a_gate.json in the kit): one job per
    cell and test set, depending on the cell's training job (its weights) and, for latent arms, its
    autoencoder job."""
    jobs = []
    for c in cs:
        for set_name in sets:
            n = f"{job_name(c)}-{set_name.replace('_', '')}"
            deps = [job_name(c)] + ([ae_job_name(c)] if c.arm in LEARNED_WITH_AE else [])
            jobs.append({"name": n, "system": "evaluate", "set": n, "threads": 4, "depends_on": deps,
                         "commands": [["kaggle/train_chunk.py", "--evaluate", job_name(c), set_name, "--out", "OUT"]],
                         "expect": [f"{job_name(c)}_{set_name}.json"]})
    return {"kit_slug": kit_slug, "nz": 0, "jobs": jobs}


if __name__ == "__main__":
    with open(sys.argv[1], "w") as f:
        json.dump(build_jobs(cells()), f, indent=1)
