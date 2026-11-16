"""The running total the compute ceiling counts (PREREG Section 11; T8 DOC-1).

Counted quantity: for every kernel version pushed for the study from the freeze on (training, autoencoder,
gate, evaluation, reference data and solver kernels, continuations `<name>-r<n>`, and kernels that failed,
timed out or were cancelled), the time stamp of the last line of its Kaggle log, which Kaggle writes in
seconds since the session started. Pushes are read from kaggle/registry.txt ("<date> <time> <name> pushed",
written by kaggle/run_queue.py); logs from the downloaded kernel outputs (`<name>/<name>.log`). A pushed
kernel without a readable log (never downloaded, cancelled before output) counts as a full 12-hour session.
The pilot and the data regeneration before the freeze are not counted.

    python scripts/ceiling_ledger.py --since "YYYY-MM-DD HH:MM:SS" [--registry kaggle/registry.txt]
        [--logs data/reference] [--plan]

With --plan it also prints the reduction steps (p3b.train.cells.ceiling_status), which needs torch.
"""
from __future__ import annotations

import argparse
import json
import os
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MISSING_SESSION_H = 12.0                 # Kaggle's session limit: an unreadable session is charged in full
_PUSHED = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (\S+) pushed$")
_TIME = re.compile(r'"time"\s*:\s*([0-9.eE+-]+)')


def session_seconds(log_path) -> float:
    """The last time stamp of a Kaggle log (a JSON list of {"stream_name", "time", "data"}); a truncated
    file gives the last complete "time" field."""
    text = open(log_path, encoding="utf-8", errors="replace").read()
    try:
        return float(json.loads(text)[-1]["time"])
    except (ValueError, IndexError, KeyError, TypeError):
        found = _TIME.findall(text)
        if not found:
            raise ValueError(f"no time stamp in {log_path}")
        return float(found[-1])


def _logs_by_name(logs_root) -> dict:
    out = {}
    for d, _, fs in os.walk(str(logs_root)):
        name = os.path.basename(d)
        if f"{name}.log" in fs:
            out.setdefault(name, os.path.join(d, f"{name}.log"))
    return out


def ledger(registry, logs_root, since: str) -> dict:
    pushes = []
    for line in open(registry, encoding="utf-8", errors="replace"):
        m = _PUSHED.match(line.strip())
        if m and m.group(1) >= since:
            pushes.append((m.group(1), m.group(2)))
    logs = _logs_by_name(logs_root)
    last = {name: i for i, (_, name) in enumerate(pushes)}     # only a kernel's latest version has a log
    kernels, missing, total = [], [], 0.0
    for i, (when, name) in enumerate(pushes):
        s = None
        if last[name] == i and name in logs:
            try:
                s = session_seconds(logs[name])
            except (OSError, ValueError):
                s = None
        h = MISSING_SESSION_H if s is None else s / 3600
        if s is None and name not in missing:
            missing.append(name)
        kernels.append({"name": name, "pushed": when, "session_s": s, "charged_h": h})
        total += h
    return {"since": since, "kernels": kernels, "missing": missing, "total_h": total}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", required=True, help='the freeze time, "YYYY-MM-DD HH:MM:SS" (registry clock)')
    ap.add_argument("--registry", default=os.path.join(ROOT, "kaggle", "registry.txt"))
    ap.add_argument("--logs", default=os.path.join(ROOT, "data", "reference"))
    ap.add_argument("--plan", action="store_true")
    a = ap.parse_args(argv)
    led = ledger(a.registry, a.logs, a.since)
    print(f"kernels pushed since {a.since}: {len(led['kernels'])}; missing logs (charged "
          f"{MISSING_SESSION_H:.0f} h each): {len(led['missing'])}")
    print(f"running total: {led['total_h']:.1f} Kaggle session hours")
    if a.plan:
        import sys
        sys.path.insert(0, ROOT)
        from p3b.train.cells import CEILING_H
        print(f"ceiling {CEILING_H:.0f} h; remaining {CEILING_H - led['total_h']:.1f} h "
              "(pass the projected costs of the remaining cells to p3b.train.cells.ceiling_status)")
    return led


if __name__ == "__main__":
    main()
