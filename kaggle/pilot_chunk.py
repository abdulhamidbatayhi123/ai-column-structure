"""Run one pilot piece inside a Kaggle kernel (Plan 5 Task 6), by the names of kaggle/pilot_jobs.py:

    data-<key> | cplk-<key> | ae-<key>-s<seed> | lr-<arm> | curve-<arm> | s-<key> | wall | two-<arm>-s<seed> | pcq-s<seed>

Data: the kit's data/pilot/<key>/{train,validation}; chosen values of earlier phases: the kit's
results/pilot_stage.json (scripts/pilot_collect.py stage N). Outputs go to --out. It prints the piece
file's name, never a loss (blinding).

    python kaggle/pilot_chunk.py --piece lr-S --out OUT
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]
sys.path.insert(0, ROOT)
if os.path.isdir(os.path.join(ROOT, "paper1")):            # inside the kit
    os.environ.setdefault("P3B_PAPER1_DIR", os.path.join(ROOT, "paper1"))

from p3b.pilot import pieces  # noqa: E402

ARMS = {pieces._tag(a): a for a in pieces.LR_ARMS}


def run_piece(name: str, data: str, stage: dict, out: str, q: dict, deadline=None, resume_from=None) -> str:
    kind, _, rest = name.partition("-")
    if kind == "data":
        return pieces.piece_data(data, rest, out, q, stage)
    if kind == "cplk":
        return pieces.piece_cplk(data, rest, out, q, stage)
    if kind == "ae":
        key, _, seed = rest.rpartition("-s")
        return pieces.piece_ae(data, key, int(seed), stage, out, q)
    if kind == "lr":
        return pieces.piece_lr(ARMS[rest], data, stage, out, q, deadline=deadline, resume_from=resume_from)
    if kind == "curve":
        return pieces.piece_curve(ARMS[rest], data, stage, out, q, deadline=deadline, resume_from=resume_from)
    if kind == "s":
        return pieces.piece_s(rest, data, stage, out, q)
    if kind == "wall":
        return pieces.piece_wall(data, stage, out, q)
    if kind == "two":
        arm, _, seed = rest.rpartition("-s")
        return pieces.piece_two(ARMS[arm], int(seed), data, stage, out, q)
    if kind == "pcq":
        return pieces.piece_two("U", int(rest.lstrip("s")), data, stage, out, q, quarter=True)
    raise ValueError(f"unknown piece {name!r}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--piece", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "pilot"))
    ap.add_argument("--stage", default=os.path.join(ROOT, "results", "pilot_stage.json"))
    ap.add_argument("--quick", default=None, help="JSON overrides for tests on tiny data only")
    ap.add_argument("--hours", type=float, default=11.0,
                    help="resumable pieces (lr-*, curve-*) checkpoint after so many hours (Kaggle sessions end at 12)")
    ap.add_argument("--resume-from", default=None, help="an earlier session's output (Kaggle kernel input)")
    a = ap.parse_args(argv)
    import time
    deadline = time.monotonic() + a.hours * 3600
    stage = {}
    if os.path.exists(a.stage):
        with open(a.stage) as f:
            stage = json.load(f)
    q = json.loads(a.quick) if a.quick else {}
    if not os.path.isdir(a.data):
        raise SystemExit(f"no pilot data at {a.data}: stage it with scripts/pilot_collect.py data and rebuild the kit")
    print(os.path.basename(run_piece(a.piece, a.data, stage, a.out, q, deadline, a.resume_from)), flush=True)


if __name__ == "__main__":
    main()
