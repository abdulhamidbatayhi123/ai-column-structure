"""The costs the compute ceiling needs that no earlier kernel measured (T8 DOC-2, DOC-7, R1-6), on Kaggle CPU,
one thread, with the code after R3. It reports costs only (seconds), never an error value.

- every latent arm, seconds per training iteration at d = 64 (the largest d of the gate's grid, so an upper
  bound for the study's d), h = 1 and 2 (scripts/time_arms.py's procedure);
- S's rollout step at d = 64, batch 1 and 48;
- arm C on 16, 32, 64 and 128 cells and MUSCL-C on 32 and 64, CPU seconds per cycle, one test-set point
  per system key (p3b.solvers.coarse.run_c);
- the reference after R3 at the registered grids (A 2048, B 1024 cells), CPU seconds per cycle;
- one autoencoder epoch at the full-data snapshot count (256 trajectories x 6 cycles x 200 snapshots,
  the generated trajectories tiled: the cost does not depend on the values), balance weight 1 (R1) and 0.

    python scripts/time_ceiling.py --out OUT/timing_ceiling.json [--quick]
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
import os
import sys
import tempfile
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]
sys.path.insert(0, ROOT)
if os.path.isdir(os.path.join(ROOT, "paper1")):            # inside a kit
    os.environ.setdefault("P3B_PAPER1_DIR", os.path.join(ROOT, "paper1"))

import torch  # noqa: E402

from p3b.design import make_set  # noqa: E402
from p3b.generate import generate_trajectory  # noqa: E402
from p3b.latent.data import load_set  # noqa: E402
from p3b.pilot import pieces, runs  # noqa: E402

_spec = importlib.util.spec_from_file_location("time_arms", os.path.join(ROOT, "scripts", "time_arms.py"))
time_arms = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(time_arms)

KEYS = {"A_MOF-303": ("A", "MOF-303", "A_primary"), "A_Al-fumarate": ("A", "Al-fumarate", "A_replication"),
        "B": ("B", None, "B")}
FULL = {"n_z": 512, "n_c": 128, "n_snap": 200, "q": {"d": 64}, "arms": tuple(pieces.LATENT_ARMS), "iterations": 10,
        "warmup": 2, "keys": tuple(KEYS), "grids_c": (16, 32, 64, 128), "grids_muscl": (32, 64), "cycles": 2,
        "ref_nz": {"A": 2048, "B": 1024}, "ae_traj": 256, "ae_cycles": 6, "batch": (1, 48)}
QUICK = {"n_z": 32, "n_c": 8, "n_snap": 40, "q": {"d": 4, "ae_epochs": 1, "ae_width": 8, "width": 8, "n_windows": 1,
                                                  "n_steps": 2},
         "arms": ("U", "S"), "iterations": 1, "warmup": 1, "keys": ("A_MOF-303", "B"), "grids_c": (4, 8),
         "grids_muscl": (8,), "cycles": 2, "ref_nz": {"A": 32, "B": 32}, "ae_traj": 3, "ae_cycles": 3,
         "batch": (1, 48)}


def _point(key: str, cycles: int) -> dict:
    return {**make_set(KEYS[key][2], "long_test", 0)[0], "n_cycles": cycles}


def time_solvers(s: dict) -> dict:
    from p3b.solvers.coarse import run_c
    out = {"C": {}, "MUSCL-C": {}}
    for key in s["keys"]:
        system, material, _ = KEYS[key]
        pt = _point(key, s["cycles"])
        for arm, grids in (("C", s["grids_c"]), ("MUSCL-C", s["grids_muscl"])):
            for g in grids:
                r = run_c(system, pt, material, N_c=g, path=None, n_snap=s["n_snap"], n_target=s["n_c"],
                          muscl=arm == "MUSCL-C")
                out[arm][f"{key}|{g}"] = {"ok": bool(r.ok), "cycle_cpu_s": [float(x) for x in r.cycle_cpu_s],
                                          "nfev": [int(x) for x in r.nfev], "message": r.message}
                print(f"timed {arm} {key} {g}: {out[arm][f'{key}|{g}']['cycle_cpu_s']}", flush=True)
    return out


def time_reference(s: dict) -> dict:
    out = {}
    for key in s["keys"]:
        system, material, _ = KEYS[key]
        r = generate_trajectory(system, _point(key, s["cycles"]), material, N_z=s["ref_nz"][system], N_c=s["n_c"],
                                path=None, n_snap=s["n_snap"])
        out[key] = {"ok": bool(r.ok), "N_z": s["ref_nz"][system], "cycle_cpu_s": [float(x) for x in r.cycle_cpu_s],
                    "nfev": [int(x) for x in r.nfev], "message": r.message}
        print(f"timed reference {key}: {out[key]['cycle_cpu_s']}", flush=True)
    return out


def _tile(ds, n_traj: int, n_cycles: int):
    rt, rc = math.ceil(n_traj / ds.X.shape[0]), math.ceil(n_cycles / ds.X.shape[1])
    kw = {f.name: getattr(ds, f.name) for f in dataclasses.fields(ds)}
    for name in ("X", "raw_inputs", "features", "t", "N_true", "E_true", "W_ref", "wall_abs"):
        v = kw[name]
        if v is not None:
            reps = [rt, rc] + [1] * (v.dim() - 2)
            kw[name] = v.repeat(*reps)[:n_traj, :n_cycles]
    kw["meta"] = (list(ds.meta) * rt)[:n_traj]
    return type(ds)(**kw)


def _b_data(root: str, s: dict) -> list:
    d = os.path.join(root, "B", "train")
    os.makedirs(d, exist_ok=True)
    paths = []
    for i, pt in enumerate(make_set("B", "long_test", 0)[:2]):
        p = os.path.join(d, f"t{i}.npz")
        if not generate_trajectory("B", {**pt, "n_cycles": s["cycles"]}, None, N_z=s["n_z"], N_c=s["n_c"], path=p,
                                   n_snap=s["n_snap"]).ok:
            raise RuntimeError(f"timing data: B trajectory {i} failed")
        paths.append(p)
    return paths


def time_ae(root: str, s: dict) -> dict:
    from p3b.latent.autoencoder import fit_autoencoder
    torch.set_num_threads(1)
    out = {}
    sets = {"A": load_set(pieces.files(root, time_arms.KEY, "train")), "B": load_set(_b_data(root, s))}
    for system, small in sets.items():
        ds = _tile(small, s["ae_traj"], s["ae_cycles"])
        std = pieces._std(ds)
        phys = runs._physics(ds)
        key = time_arms.KEY if system == "A" else "B"
        for w in (1, 0):
            torch.manual_seed(0)
            ae = pieces._ae(ds, {"d": {key: int(s["q"]["d"])}}, key, s["q"])
            w0, c0 = time.perf_counter(), time.process_time()
            fit_autoencoder(ae, ds, std, phys, epochs=1, seed=0, weight=float(w))
            out[f"{system}|w{w}"] = {"epoch_s": time.perf_counter() - w0, "cpu_s": time.process_time() - c0,
                                     "n_snapshots": int(ds.X.shape[0] * ds.X.shape[1] * ds.X.shape[2]),
                                     "d": int(s["q"]["d"])}
            print(f"timed autoencoder epoch {system} w{w}: {out[f'{system}|w{w}']['epoch_s']:.1f} s", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true", help="tiny settings for tests only")
    a = ap.parse_args(argv)
    s = QUICK if a.quick else FULL
    torch.set_num_threads(1)
    data = tempfile.mkdtemp()
    gen = time_arms.make_data(data, s["n_z"], s["n_c"], s["n_snap"])
    res = {}
    t = time_arms.time_arms(data, s["q"], s["arms"], threads=(1,), steps=(1, 2), iterations=s["iterations"],
                            warmup=s["warmup"])
    res["train"] = t["train"]
    res["step_S"] = time_arms.time_batched(data, s["q"], list(s["batch"]))
    res.update(time_solvers(s))
    res["reference"] = time_reference(s)
    res["ae"] = time_ae(data, s)
    commit = os.path.join(ROOT, "kit_commit.txt")
    res.update({"generation_cpu_s": gen, "torch": torch.__version__, "cpu_count": os.cpu_count(),
                "settings": {k: (list(v) if isinstance(v, tuple) else v) for k, v in s.items()},
                "commit": open(commit).read().strip() if os.path.exists(commit) else os.environ.get("P3B_COMMIT", "")})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    main()
