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
from p3b.train.cells import SOLVER_ARMS, ae_job_name, cells, job_name  # noqa: E402

KIT = "abdulhamidbatayhi/p3b-reference-kit"


def build_jobs(cs, kit_slug: str = KIT) -> dict:
    ae_jobs, jobs = {}, []
    for c in cs:
        if c.arm in SOLVER_ARMS:
            continue
        n = ae_job_name(c)
        if n not in ae_jobs:
            ae_jobs[n] = {"name": n, "system": "train", "set": n,
                          "commands": [["kaggle/train_chunk.py", "--ae", n, "--out", "OUT"]],
                          "expect": [f"ae_{c.system}_{c.material}_{c.size}_s{c.seed}.pt"]}
    for c in cs:
        n = job_name(c)
        job = {"name": n, "system": "train", "set": n,
               "commands": [["kaggle/train_chunk.py", "--cell", n, "--out", "OUT"]], "expect": [f"{n}.json"]}
        if c.arm not in SOLVER_ARMS:
            job["depends_on"] = [ae_job_name(c)]
        jobs.append(job)
    return {"kit_slug": kit_slug, "nz": 0, "jobs": list(ae_jobs.values()) + jobs}


if __name__ == "__main__":
    with open(sys.argv[1], "w") as f:
        json.dump(build_jobs(cells()), f, indent=1)
