"""The latent dimension d and the decoder from the full-data decoder gate (PREREG Section 4, 5.1).

Reads the gate jobs' gate_<name>.json files (kaggle/train_jobs.py gate_jobs; any depth under the
directory) and writes results/decoder_choice.json:
  {key: {"decoder": "mlp" | "conv", "d": d}       the smallest d at which every seed passed
       | {"needs_conv": true}                      no MLP d passed and the conv wave has not run
       | {"not_testable": true}}                   neither decoder passes at any d
plus this script's SHA-256 and every input file's. Every (d, seed) of a decoder kind that ran must be
present; a missing file stops the script (no choice from part of the grid).

    python scripts/decoder_choice.py <gate_dir> [results/decoder_choice.json]
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from p3b.latent.autoencoder import full_data_choice  # noqa: E402
from p3b.pilot.rules import pilot_key  # noqa: E402
from p3b.train.cells import cells, gate_specs  # noqa: E402


def _sha(path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main(gate_dir: str, out_path: str, keys=None) -> dict:
    found = {}
    for p in glob.glob(os.path.join(gate_dir, "**", "gate_*.json"), recursive=True):
        found.setdefault(os.path.basename(p), p)
    cs = cells()
    all_keys = sorted({pilot_key(s["system"], s["material"]) for s in gate_specs(cs, "mlp")})
    out, hashes = {}, {}
    for key in keys or all_keys:
        metrics = {}
        for kind in ("mlp", "conv"):
            specs = [s for s in gate_specs(cs, kind) if pilot_key(s["system"], s["material"]) == key]
            files = {s["name"]: found.get(f"gate_{s['name']}.json") for s in specs}
            if kind == "conv" and not any(files.values()):
                continue                                    # the conv wave has not run for this key
            missing = [n for n, p in files.items() if p is None]
            if missing:
                raise FileNotFoundError(f"{key} {kind}: {len(missing)} gate results missing, e.g. {missing[0]}")
            by_d = {}
            for s in specs:
                p = files[s["name"]]
                with open(p) as f:
                    r = json.load(f)
                if (r["kind"], int(r["d"]), int(r["seed"])) != (kind, s["d"], s["seed"]):
                    raise ValueError(f"{p} does not hold {s['name']}")
                by_d.setdefault(s["d"], []).append({k: r[k] for k in ("passed", "passed_long") if k in r})
                hashes[os.path.relpath(p, gate_dir).replace(os.sep, "/")] = _sha(p)
            metrics[kind] = by_d
        dec, d = full_data_choice(metrics)
        if d is not None:
            out[key] = {"decoder": dec, "d": d}
        elif "conv" not in metrics:
            out[key] = {"needs_conv": True}
        else:
            out[key] = {"not_testable": True}
    out["inputs_sha256"] = hashes
    out["script_sha256"] = _sha(__file__)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(json.dumps({k: v for k, v in out.items() if not k.endswith("sha256")}, sort_keys=True))
    return out


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "results", "decoder_choice.json"))
