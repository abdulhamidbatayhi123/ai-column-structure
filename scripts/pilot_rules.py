"""The blinded pilot rules (PREREG Section 10). Committed and hashed before it is run.

Reads the pilot's results files from one directory and writes results/pilot_values.json with only
the chosen values, this script's SHA-256 and the SHA-256 of every input file. It prints the chosen
values and nothing else: no error value of any arm is ever printed or written (blinding).

Inputs (JSON, written by the pilot runs):
  lr_runs.json            {arm: {lr: validation rollout loss over one cycle after 2,000 iterations}}
  curves.json             {arm: [[iteration, pooled validation loss], ...]}
  h_check.json            {system_material: {h: {"viol_rate", "stiff_ok"}}}
  threshold_inputs.json   {system_material: {"floors", "neg_production", "g2", "defects_per_step", "tau_ref"}}
  cplusl_K.json           {system_material: {K: epsilon_W of C+L with unit multipliers against C}}
  sds.json                {hypothesis: [[log ratio per seed] per trajectory]}
  css.json                {system_material: [CSS cycle per pilot trajectory]}
  decoder_gate.json       {system_material: {d: passed}}
  positive_control.json   {"word": the full rule's word for U full against U quarter}

    python scripts/pilot_rules.py <pilot_results_dir> [results/pilot_values.json]
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from p3b.latent.autoencoder import choose_d  # noqa: E402
from p3b.pilot import rules  # noqa: E402

FILES = ("lr_runs.json", "curves.json", "h_check.json", "threshold_inputs.json", "cplusl_K.json",
         "sds.json", "css.json", "decoder_gate.json", "positive_control.json")


def _sha(path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main(results_dir: str, out_path: str) -> dict:
    data, hashes = {}, {}
    for name in FILES:
        p = os.path.join(results_dir, name)
        with open(p) as f:
            data[name] = json.load(f)
        hashes[name] = _sha(p)
    lr = rules.choose_lr({arm: {float(k): v for k, v in by.items()} for arm, by in data["lr_runs.json"].items()})
    iters = rules.choose_iterations({arm: [tuple(x) for x in c] for arm, c in data["curves.json"].items()})
    h = {key: rules.choose_h({float(k): v["viol_rate"] for k, v in hs.items()},
                             {float(k): v["stiff_ok"] for k, v in hs.items()})
         for key, hs in data["h_check.json"].items()}
    th = {key: rules.thresholds(t["floors"], t["neg_production"], t["g2"], t["defects_per_step"], t["tau_ref"])
          for key, t in data["threshold_inputs.json"].items()}
    K = {key: rules.choose_K({int(k): v for k, v in e.items()}) for key, e in data["cplusl_K.json"].items()}
    sds = {}
    for hyp, vals in data["sds.json"].items():
        s_t, s_s = rules.upper80_sd(vals)
        sds[hyp] = {"sd_traj_upper80": s_t, "sd_seed_upper80": s_s}
    css = {key: int(max(c)) for key, c in data["css.json"].items()}
    d = {key: choose_d({int(k): bool(v) for k, v in g.items()}) for key, g in data["decoder_gate.json"].items()}
    out = {"lr": lr, "iterations": iters, "h": h, "thresholds": th, "K": K, "sds": sds, "css": css, "d": d,
           "positive_control": data["positive_control.json"]["word"],
           "script_sha256": _sha(__file__), "inputs_sha256": hashes}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(json.dumps({k: out[k] for k in ("lr", "iterations", "h", "K", "css", "d", "positive_control")},
                     sort_keys=True))
    return out


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "results", "pilot_values.json"))
