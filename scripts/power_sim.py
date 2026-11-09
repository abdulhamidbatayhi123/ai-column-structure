"""Power of the full decision rules from the pilot's upper-80 % SDs (PREREG Section 9).

Committed and hashed before it is run. Reads results/pilot_values.json (scripts/pilot_rules.py)
for the SDs, prints the word fractions per hypothesis at the declared minimum effect (ratio 0.8)
and at no effect, and writes results/power.json with this script's SHA-256.

    python scripts/power_sim.py [--reps 1000] [--draws 10000]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from p3b.pilot.power import simulate  # noqa: E402

HYPOTHESES = {"H1": dict(n_traj=48, alpha_h=0.05 / 3), "H2": dict(n_traj=48, alpha_h=0.05 / 3),
              "H5": dict(n_traj=24, alpha_h=0.05)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--values", default=os.path.join(ROOT, "results", "pilot_values.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "power.json"))
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--draws", type=int, default=10_000)
    a = ap.parse_args(argv)
    with open(a.values) as f:
        sds = json.load(f)["sds"]
    out = {"script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(), "power": {}}
    for h, cfg in HYPOTHESES.items():
        sd_t, sd_s = sds[h]["sd_traj_upper80"], sds[h]["sd_seed_upper80"]
        out["power"][h] = {str(r): simulate(math.log(r), sd_t, sd_s, n_traj=cfg["n_traj"], reps=a.reps,
                                            draws=a.draws, seed=1000 + i, alpha_h=cfg["alpha_h"])
                           for i, r in enumerate((0.8, 1.0))}
        print(h, json.dumps(out["power"][h]))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
