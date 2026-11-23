"""Collect the pilot's pieces (Plan 5 Task 6; p3b.pilot.collect). Prints counts and chosen values only.

    python scripts/pilot_collect.py data GEN_DIR [GEN_DIR ...] DATA_ROOT      # lay out the pilot data
    python scripts/pilot_collect.py stage N PIECES_DIR [results/pilot_stage.json]
    python scripts/pilot_collect.py ae PIECES_DIR [results/pilot_ae]          # after phase 1, before the kit
    python scripts/pilot_collect.py final PIECES_DIR INPUTS_DIR [results/pilot_stage.json]   # then pilot_rules.py

stage and final refuse any piece whose recorded data files differ from the current data/pilot (or
P3B_PILOT_DATA when set): a piece computed on earlier pilot data is never collected (T8 RS-4). The
stage file is written with the pilot revision; stage N > 0 and final refuse a stage file of another
revision (T8 fix review, R1-4).
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from p3b import envcheck  # noqa: E402
from p3b.pilot import collect  # noqa: E402
from p3b.pilot.pieces import load_stage  # noqa: E402

# every piece and autoencoder collected must have run under requirements.txt's pins (PREREG 11)
ENV_PINS = envcheck.pins_from_requirements(os.path.join(ROOT, "requirements.txt"))
DATA_ROOT = os.environ.get("P3B_PILOT_DATA", os.path.join(ROOT, "data", "pilot"))


def main(argv):
    cmd = argv[0]
    if cmd == "data":
        print(json.dumps(collect.stage_data(argv[1:-1], argv[-1]), indent=1))
    elif cmd == "stage":
        n, pieces_dir = int(argv[1]), argv[2]
        path = argv[3] if len(argv) > 3 else os.path.join(ROOT, "results", "pilot_stage.json")
        # after phase n > 0 the previous stage must be of the current pilot revision (T8 fix review, R1-4)
        prev = load_stage(path, n, required=True)
        s = collect.stage(n, pieces_dir, prev, data_root=DATA_ROOT, env_pins=ENV_PINS)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(s, f, indent=1, sort_keys=True)
        print(json.dumps(s, sort_keys=True))
    elif cmd == "ae":
        dst = argv[2] if len(argv) > 2 else os.path.join(ROOT, "results", "pilot_ae")
        for p in collect.collect_ae(argv[1], dst, env_pins=ENV_PINS):
            print(p.name)
    elif cmd == "final":
        path = argv[3] if len(argv) > 3 else os.path.join(ROOT, "results", "pilot_stage.json")
        st = load_stage(path, 4, required=True)
        for p in collect.final(argv[1], argv[2], st, data_root=DATA_ROOT, env_pins=ENV_PINS):
            print(os.path.basename(p))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
