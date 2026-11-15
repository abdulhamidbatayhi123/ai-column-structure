"""What a continued Kaggle session must share with the session it continues (T8 review RS-5).

A session that stops writes RESUME.json beside its checkpoint, with a fingerprint of the run: the
stage it ran under, the kit commit, the configuration (overrides, iteration count, pilot values) and
the inputs (data files, autoencoder). The next session builds the same fingerprint from its own
inputs and refuses to continue if any part differs, so a chain never finishes under code, stage or
data other than those it started with (PREREG 12: the frozen commit in every results file).
"""
from __future__ import annotations

import json
import os


def plain(x):
    """x as it reads back from JSON (tuples become lists), so stored and current fingerprints compare."""
    return json.loads(json.dumps(x, sort_keys=True, default=str))


def check_resume(where: str, fingerprint: dict) -> dict:
    """The RESUME.json in directory `where` (the earlier session's output holding the checkpoint), after
    checking that its fingerprint equals `fingerprint`; RuntimeError otherwise."""
    path = os.path.join(str(where), "RESUME.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no RESUME.json beside the checkpoint in {where}: its run fingerprint is unknown")
    with open(path) as f:
        info = json.load(f)
    stored, now = info.get("fingerprint") or {}, plain(fingerprint)
    diff = sorted(k for k in set(stored) | set(now) if stored.get(k) != now.get(k))
    if diff:
        raise RuntimeError(f"{path}: the run fingerprint differs in {diff} (the stage, kit, configuration or "
                           f"inputs changed while the chain was in flight); start the job afresh")
    return info
