"""Merge the Task 10 Kaggle timing outputs into results/timing_reference.json.

System B and the Paper 1 limit at 2048 cells come from the p3b-timing-b-v1 kernel's JSON files. The
system A kernel (p3b-timing-a) was stopped at Kaggle's 12-hour limit before it wrote its JSON
(scripts/time_reference.py writes at the end), so its records are parsed from the kernel's execution
log, where every record was printed as it finished; the record of the MOF-303 centre point at 2048
cells never finished (a solver stall, see verify/stall_diagnostic.py).

    python scripts/merge_timing.py
"""
import glob
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
T = os.path.join(ROOT, "data", "reference", "timing")


def a_records_from_log(path):
    raw = open(path, encoding="utf-8", errors="replace").read()
    entries = json.loads(raw[raw.find("["):])
    text = "".join(e.get("data", "") for e in entries)
    recs, grid = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        (grid if "N_coarse" in r else recs).append(r)
    last = entries[-1].get("time") if entries else None
    return recs, grid, last


def main():
    out = {"source": "Task 10 Kaggle kernels p3b-timing-a (log) and p3b-timing-b-v1 (JSON)", "records": [],
           "grid_convergence": [], "notes": []}
    for f in sorted(glob.glob(os.path.join(T, "p3b-timing-b-v1", "**", "out", "timing_B_*.json"), recursive=True)):
        d = json.load(open(f))
        out["records"] += d["records"]
        out["grid_convergence"] += d["grid_convergence"]
    log = glob.glob(os.path.join(T, "p3b-timing-a", "p3b-timing-a.log"))[0]
    recs, grid, last = a_records_from_log(log)
    for r in recs:
        r["point_name"] = "centre" if r["point"]["v"] < 0.2 else "corner"
    out["records"] += recs
    out["grid_convergence"] += grid
    out["notes"].append(f"p3b-timing-a: stopped at Kaggle's 12 h limit; its log ends at {last:.0f} s of kernel time; "
                        "MOF-303 centre at 2048 cells never finished (solver stall), and its 1024-cell cycle 1 took "
                        "3,753 CPU-s with 642,420 evaluations against 9 s and 2,338 at 512 cells")
    v1 = glob.glob(os.path.join(T, "p3b-timing-b-v1", "**", "out", "results_v1_2048.json"), recursive=True)[0]
    out["v1_2048"] = json.load(open(v1))
    path = os.path.join(ROOT, "results", "timing_reference.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(len(out["records"]), "records,", len(out["grid_convergence"]), "grid-convergence pairs ->", path)


if __name__ == "__main__":
    main()
