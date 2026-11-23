"""The pilot's choices come only from pieces run under the pinned environment (PREREG 11; 2026-11-23). Phase 0
ran twice under the same stage (torch 2.11 of the Kaggle image, then the pinned 2.12): with env_pins, a copy
counts only if the nearest ENV.json above it records exactly those versions; scripts/pilot_collect.py always
passes requirements.txt's pins."""
import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PINS = {"numpy": "2.4.6", "scipy": "1.17.1", "torch": "2.12.0+cpu"}


def _piece(base, kernel, value, got=None, name="data_B.json"):
    from p3b.pilot.pieces import stage_sha
    out = base / kernel / "out"
    out.mkdir(parents=True)
    (out / name).write_text(json.dumps({"meta": {"stage_sha256": stage_sha({})}, "value": value}))
    if got is not None:
        (out / "ENV.json").write_text(json.dumps({"kernel": "p3b-env-x", "wanted": PINS, "got": got}))
    return out


def test_only_the_pinned_copy_counts(tmp_path):
    from p3b.pilot import collect
    _piece(tmp_path, "k-unpinned", "old")                                        # no ENV.json (image torch)
    _piece(tmp_path, "k-pinned", "new", got=PINS)
    assert collect.load_piece(tmp_path, "data_B.json", {}, env_pins=PINS) == "new"
    with pytest.raises(RuntimeError):                                            # without pins: two copies
        collect.load_piece(tmp_path, "data_B.json", {})


def test_a_copy_under_other_versions_never_counts(tmp_path):
    from p3b.pilot import collect
    _piece(tmp_path, "k-other", "x", got={**PINS, "torch": "2.11.0+cpu"})
    with pytest.raises(RuntimeError, match="environment"):
        collect.load_piece(tmp_path, "data_B.json", {}, env_pins=PINS)


def test_autoencoders_only_from_pinned_kernels(tmp_path):
    from p3b.pilot import collect
    a = _piece(tmp_path, "ae-unpinned", 0, name="unused.json")
    b = _piece(tmp_path, "ae-pinned", 0, got=PINS, name="unused.json")
    (a / "ae_B_none_pilot_s0.pt").write_bytes(b"old")
    (b / "ae_B_none_pilot_s0.pt").write_bytes(b"new")
    out = collect.collect_ae(tmp_path, tmp_path / "dst", env_pins=PINS)
    assert [p.name for p in out] == ["ae_B_none_pilot_s0.pt"] and out[0].read_bytes() == b"new"


def test_the_cli_always_passes_the_pinned_versions(tmp_path, monkeypatch):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import pilot_collect
    seen = {}
    monkeypatch.setattr(pilot_collect.collect, "stage", lambda *a, **k: seen.setdefault("stage", k) or {})
    monkeypatch.setattr(pilot_collect.collect, "collect_ae", lambda *a, **k: seen.setdefault("ae", k) and [])
    monkeypatch.setattr(pilot_collect.collect, "final", lambda *a, **k: seen.setdefault("final", k) and [])
    monkeypatch.setattr(pilot_collect, "load_stage", lambda *a, **k: {})
    pilot_collect.main(["stage", "0", str(tmp_path), str(tmp_path / "st.json")])
    pilot_collect.main(["ae", str(tmp_path), str(tmp_path / "ae")])
    pilot_collect.main(["final", str(tmp_path), str(tmp_path), str(tmp_path / "st.json")])
    assert seen["stage"]["env_pins"] == seen["ae"]["env_pins"] == seen["final"]["env_pins"] == PINS
