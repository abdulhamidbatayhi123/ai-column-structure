"""Collect the pilot's pieces (Plan 5 Task 6; p3b.pilot.collect). Prints counts and chosen values only.

    python scripts/pilot_collect.py data GEN_DIR [GEN_DIR ...] DATA_ROOT      # lay out the pilot data
    python scripts/pilot_collect.py stage N PIECES_DIR [results/pilot_stage.json]
    python scripts/pilot_collect.py ae PIECES_DIR [results/pilot_ae]          # after phase 1, before the kit
    python scripts/pilot_collect.py final PIECES_DIR INPUTS_DIR               # then scripts/pilot_rules.py INPUTS_DIR
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from p3b.pilot import collect  # noqa: E402


def main(argv):
    cmd = argv[0]
    if cmd == "data":
        print(json.dumps(collect.stage_data(argv[1:-1], argv[-1]), indent=1))
    elif cmd == "stage":
        n, pieces_dir = int(argv[1]), argv[2]
        path = argv[3] if len(argv) > 3 else os.path.join(ROOT, "results", "pilot_stage.json")
        prev = {}
        if n > 0 and os.path.exists(path):
            with open(path) as f:
                prev = json.load(f)
        s = collect.stage(n, pieces_dir, prev)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(s, f, indent=1, sort_keys=True)
        print(json.dumps(s, sort_keys=True))
    elif cmd == "ae":
        dst = argv[2] if len(argv) > 2 else os.path.join(ROOT, "results", "pilot_ae")
        for p in collect.collect_ae(argv[1], dst):
            print(p.name)
    elif cmd == "final":
        for p in collect.final(argv[1], argv[2]):
            print(os.path.basename(p))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
