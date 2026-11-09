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
from p3b.pilot.power import simulate, simulate_tost  # noqa: E402
from p3b.pilot.rules import h4_margin  # noqa: E402

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
    missing = [h for h in list(HYPOTHESES) + ["H4"] if h not in sds]
    if missing:
        raise KeyError(f"pilot SDs missing for {missing}: run scripts/pilot_rules.py on complete pilot results")
    for h, cfg in HYPOTHESES.items():
        sd_t, sd_s = sds[h]["sd_traj_upper80"], sds[h]["sd_seed_upper80"]
        out["power"][h] = {str(r): simulate(math.log(r), sd_t, sd_s, n_traj=cfg["n_traj"], reps=a.reps,
                                            draws=a.draws, seed=1000 + i, alpha_h=cfg["alpha_h"])
                           for i, r in enumerate((0.8, 1.0))}
        print(h, json.dumps(out["power"][h]))
    # seed-sharing sensitivity of H1's power (the default 1.0 is the conservative case)
    sd_t, sd_s = sds["H1"]["sd_traj_upper80"], sds["H1"]["sd_seed_upper80"]
    out["H1_seed_shared_sensitivity"] = {str(f): simulate(math.log(0.8), sd_t, sd_s, reps=a.reps, draws=a.draws,
                                                          seed=2000, seed_shared=f)["power"] for f in (0.0, 0.5, 1.0)}
    # H4: TOST power at no true difference, Holm worst case, and the margin rule (PREREG Section 6)
    sd_t, sd_s = sds["H4"]["sd_traj_upper80"], sds["H4"]["sd_seed_upper80"]
    p4 = {m: simulate_tost(0.0, sd_t, sd_s, m, reps=a.reps, seed=3000) for m in (1.10, 1.15, 1.20)}
    out["H4_tost_power"] = {str(m): v for m, v in p4.items()}
    out["h4_margin"] = h4_margin(p4)
    print("H4", json.dumps(out["H4_tost_power"]), "margin", out["h4_margin"])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
