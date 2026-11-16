"""scripts/ceiling_ledger.py (T8 DOC-1): the running total the ceiling counts, from Kaggle's own logs.

The counted quantity (PREREG Section 11): for every kernel version pushed for the study from the freeze
on, the time stamp of the last line of its Kaggle log (seconds since the session started). A pushed
kernel whose log is missing or unreadable counts as a full 12-hour session."""
import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("ceiling_ledger", ROOT / "scripts" / "ceiling_ledger.py")
cl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cl)


def _log(path, times):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([{"stream_name": "stdout", "time": t, "data": "x\n"} for t in times]))


def test_session_seconds_is_the_last_time_stamp(tmp_path):
    p = tmp_path / "k.log"
    _log(p, [120.5, 900.0, 3600.25])
    assert cl.session_seconds(p) == pytest.approx(3600.25)


def test_a_truncated_log_still_gives_its_last_time_stamp(tmp_path):
    p = tmp_path / "k.log"
    p.write_text('[{"stream_name":"stdout","time":10.0,"data":"a"}\n,{"stream_name":"stderr","time":4321.5,"da')
    assert cl.session_seconds(p) == pytest.approx(4321.5)


def test_ledger_counts_pushes_from_the_freeze_on_and_charges_missing_logs_a_full_session(tmp_path):
    reg = tmp_path / "registry.txt"
    reg.write_text("\n".join([
        "2026-12-01 09:00:00 p3b-old-pilot pushed",                       # before the freeze: not counted
        "2026-12-02 10:00:00 p3b-u-a-m303-full-s0 push not accepted yet: Maximum batch CPU session count",
        "2026-12-02 10:05:00 p3b-u-a-m303-full-s0 pushed",
        "2026-12-02 10:06:00 p3b-s-a-m303-full-s0 pushed",
        "2026-12-02 22:10:00 p3b-s-a-m303-full-s0-r1 pushed",             # a continuation counts
        "2026-12-03 08:00:00 p3b-c-a-m303-full-s0 pushed",                # no log downloaded (failed, cancelled)
    ]) + "\n")
    logs = tmp_path / "data"
    _log(logs / "train" / "x" / "p3b-u-a-m303-full-s0" / "p3b-u-a-m303-full-s0.log", [100.0, 7200.0])
    _log(logs / "train" / "y" / "p3b-s-a-m303-full-s0" / "p3b-s-a-m303-full-s0.log", [50.0, 39600.0])
    _log(logs / "train" / "y" / "p3b-s-a-m303-full-s0-r1" / "p3b-s-a-m303-full-s0-r1.log", [80.0, 18000.0])
    _log(logs / "pilot" / "p3b-old-pilot" / "p3b-old-pilot.log", [1.0, 40000.0])
    led = cl.ledger(reg, logs, since="2026-12-02 00:00:00")
    assert [k["name"] for k in led["kernels"]] == ["p3b-u-a-m303-full-s0", "p3b-s-a-m303-full-s0",
                                                   "p3b-s-a-m303-full-s0-r1", "p3b-c-a-m303-full-s0"]
    assert led["missing"] == ["p3b-c-a-m303-full-s0"]
    assert led["total_h"] == pytest.approx((7200 + 39600 + 18000) / 3600 + cl.MISSING_SESSION_H)
    assert cl.MISSING_SESSION_H == 12.0


def test_a_kernel_pushed_twice_counts_twice_and_only_its_latest_log_is_known(tmp_path):
    reg = tmp_path / "registry.txt"
    reg.write_text("2026-12-02 10:00:00 p3b-x pushed\n2026-12-02 12:00:00 p3b-x pushed\n")
    _log(tmp_path / "d" / "p3b-x" / "p3b-x.log", [3600.0])
    led = cl.ledger(reg, tmp_path / "d", since="2026-12-01 00:00:00")
    assert led["total_h"] == pytest.approx(1.0 + cl.MISSING_SESSION_H)
    assert led["missing"] == ["p3b-x"]
