"""Generate one chunk of a pre-registered design set: one npz per trajectory plus a report.

    python kaggle/generate_chunk.py --system A_primary --set train --chunk 3 --of 16 --nz 2048 --out OUT

The material follows from the system (A_primary: MOF-303; A_replication: Al-fumarate; B: none);
--material, if given, must agree. Failures are recorded in chunk_<K>_report.json, never hidden.
--cycles overrides the set's cycle count for smoke tests and timing only; the report records it.
Paper 1's files are found beside the kit (paper1/) or through P3B_PAPER1_DIR.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if os.path.isdir(os.path.join(ROOT, "paper1")):            # inside the kit
    os.environ.setdefault("P3B_PAPER1_DIR", os.path.join(ROOT, "paper1"))
# the script's own directory must not shadow standard-library or installed modules
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != HERE]
sys.path.insert(0, ROOT)

MATERIAL = {"A_primary": "MOF-303", "A_replication": "Al-fumarate", "B": None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=sorted(MATERIAL))
    ap.add_argument("--set", required=True)
    ap.add_argument("--material", default=None)
    ap.add_argument("--chunk", type=int, required=True)
    ap.add_argument("--of", type=int, required=True)
    ap.add_argument("--nz", type=int, required=True)
    ap.add_argument("--nc", type=int, default=128)
    ap.add_argument("--cycles", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    material = MATERIAL[a.system]
    if a.material is not None and a.material != material:
        sys.exit(f"--material {a.material} does not match {a.system}, which uses {material}")

    from kaggle.make_kit import chunk_indices
    from p3b.generate import generate_trajectory
    from p3b.provenance import git_commit

    with open(os.path.join(ROOT, "data", "design", f"{a.system}_{a.set}.json")) as f:
        design = json.load(f)
    points = design["points"]
    mine = chunk_indices(len(points), a.of)[a.chunk]
    os.makedirs(a.out, exist_ok=True)
    kit_commit = None
    if os.path.exists(os.path.join(ROOT, "kit_commit.txt")):
        kit_commit = open(os.path.join(ROOT, "kit_commit.txt")).read().strip()
    report = {"system": a.system, "set": a.set, "material": material, "chunk": a.chunk, "of": a.of,
              "N_z": a.nz, "N_c": a.nc, "cycles_override": a.cycles, "design_seed": design["seed"],
              "design_commit": design["commit"], "kit_commit": kit_commit, "code_commit": git_commit(),
              "trajectories": []}
    t0 = time.time()
    for i in mine:
        point = dict(points[i])
        if a.cycles is not None:
            point["n_cycles"] = a.cycles
        if point["n_cycles"] is None:
            sys.exit(f"{a.system}_{a.set} has no cycle count (scenarios are run by the H6 driver)")
        sysletter = "B" if a.system == "B" else "A"
        path = os.path.join(a.out, f"{a.system}_{a.set}_{i:04d}.npz")
        res = generate_trajectory(sysletter, point, material, a.nz, a.nc, path=path)
        report["trajectories"].append({"index": i, "ok": res.ok, "failed_cycle": res.failed_cycle,
                                       "message": res.message, "cpu_s": res.cpu_s,
                                       "closure_max": max(res.closure) if res.closure else None,
                                       "nfev": res.nfev, "file": os.path.basename(path) if res.ok else None})
        print(json.dumps(report["trajectories"][-1]), flush=True)
    report["wall_s"] = time.time() - t0
    report["ok"] = all(t["ok"] for t in report["trajectories"])
    with open(os.path.join(a.out, f"chunk_{a.chunk}_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
