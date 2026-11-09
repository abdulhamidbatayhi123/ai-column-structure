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
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--of", type=int, default=None)
    ap.add_argument("--indices", default=None, help="explicit trajectory indices, e.g. 0,1,2 (instead of --chunk/--of)")
    ap.add_argument("--report", default=None, help="report tag (default: the chunk number)")
    ap.add_argument("--nz", type=int, required=True)
    ap.add_argument("--nc", type=int, default=128)
    ap.add_argument("--cycles", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    material = MATERIAL[a.system]
    if a.material is not None and a.material != material:
        sys.exit(f"--material {a.material} does not match {a.system}, which uses {material}")

    from p3b import generate as gen
    from p3b.design import chunk_indices
    from p3b.provenance import git_commit

    with open(os.path.join(ROOT, "data", "design", f"{a.system}_{a.set}.json")) as f:
        design = json.load(f)
    points = design["points"]
    if a.indices is not None:
        mine = [int(x) for x in a.indices.split(",")]
        if not all(0 <= i < len(points) for i in mine):
            sys.exit(f"indices outside 0..{len(points) - 1}")
    elif a.chunk is not None and a.of is not None:
        mine = chunk_indices(len(points), a.of)[a.chunk]
    else:
        sys.exit("give --chunk and --of, or --indices")
    tag = a.report if a.report is not None else str(a.chunk)
    os.makedirs(a.out, exist_ok=True)
    kit_commit = None
    if os.path.exists(os.path.join(ROOT, "kit_commit.txt")):
        kit_commit = open(os.path.join(ROOT, "kit_commit.txt")).read().strip()
    report = {"system": a.system, "set": a.set, "material": material, "chunk": a.chunk, "of": a.of,
              "indices": mine,
              "N_z": a.nz, "N_c": a.nc, "cycles_override": a.cycles, "design_seed": design["seed"],
              "design_commit": design["commit"], "kit_commit": kit_commit, "code_commit": git_commit(),
              "trajectories": []}
    report_path = os.path.join(a.out, f"chunk_{tag}_report.json")

    def write_report():   # after every trajectory, atomically: a killed session keeps its record
        report["wall_s"] = time.time() - t0
        report["ok"] = len(report["trajectories"]) == len(mine) and all(t["ok"] for t in report["trajectories"])
        with open(report_path + ".partial", "w") as f:
            json.dump(report, f, indent=1)
        os.replace(report_path + ".partial", report_path)

    t0 = time.time()
    sysletter = "B" if a.system == "B" else "A"
    for i in mine:
        point = dict(points[i])
        if a.cycles is not None:
            point["n_cycles"] = a.cycles
        if point["n_cycles"] is None:
            sys.exit(f"{a.system}_{a.set} has no cycle count (scenarios are run by the H6 driver)")
        smoke = "_smoke" if a.cycles is not None else ""      # smoke outputs never carry production names
        path = os.path.join(a.out, f"{a.system}_{a.set}_{i:04d}{smoke}.npz")
        try:
            res = gen.generate_trajectory(sysletter, point, material, a.nz, a.nc, path=path,
                                          extra_meta={"cycles_override": a.cycles, "design_seed": design["seed"],
                                                      "design_set": f"{a.system}_{a.set}"})
            entry = {"index": i, "ok": res.ok, "failed_cycle": res.failed_cycle, "message": res.message,
                     "cpu_s": res.cpu_s, "closure_max": max(res.closure) if res.closure else None,
                     "nfev": res.nfev, "file": os.path.basename(path) if res.ok else None}
        except Exception as exc:   # anything outside the trajectory: recorded, the batch goes on
            entry = {"index": i, "ok": False, "failed_cycle": None, "message": f"{type(exc).__name__}: {exc}",
                     "cpu_s": None, "closure_max": None, "nfev": None, "file": None}
        report["trajectories"].append(entry)
        print(json.dumps(entry), flush=True)
        write_report()
    write_report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
