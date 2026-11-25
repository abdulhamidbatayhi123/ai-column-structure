"""Pre-freeze revision R5 (2026-11-25): the pilot's step size before h is chosen.

The pilot chooses h in phase 3, but phases 1 to 3 trained at h = 1 RK4 step per snapshot interval (the learning-rate
sweep, the curves, and S's own training before its h check). At h = 1 S's training skips more than 5 % of its windows
and its validation rollouts fail (projection SVD fails to converge); at h = 2 and 4 neither happens (diagnostic kernel
p3b-diag-s-h, 2026-11-25: skipped 18/0/0 of 200, failed rollouts 5/0/0 of 16). R2 had recorded the same for the timing
runs. So every structured arm finished none of its learning-rate runs and the pilot stopped at stage 1.

Now every training run before h is chosen uses the provisional PILOT_H_STEPS = 2 (the value the ceiling was sized on,
R2); the h check itself still evaluates 1, 2 and 4. The wall-clock per iteration (U-long's iteration count) is measured
at every h of the grid, and U-long uses the primary key's chosen h."""
import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_the_provisional_pilot_h_is_2_and_the_revision_r5():
    from p3b.pilot import pieces
    assert pieces.PILOT_H_STEPS == 2 and pieces.PILOT_REVISION == "r5"


@pytest.fixture
def stub(monkeypatch, tmp_path):
    from p3b.pilot import pieces, runs
    seen = {}
    monkeypatch.setattr(pieces, "_primary", lambda root: ("ds", "val", {"data": {}}))
    monkeypatch.setattr(pieces, "_session_in", lambda *a, **k: None)
    monkeypatch.setattr(pieces, "_prev_autoencoders", lambda *a, **k: None)
    monkeypatch.setattr(pieces, "_arm_checked", lambda *a, **k: "arm")
    monkeypatch.setattr(pieces, "_write", lambda out, name, value, *a, **k: seen.setdefault("written", (name, value)))
    return pieces, runs, seen


def test_the_learning_rate_sweep_trains_at_the_pilot_h(stub, monkeypatch, tmp_path):
    pieces, runs, seen = stub
    monkeypatch.setattr(runs, "lr_sweep", lambda *a, **k: seen.setdefault("h", k["h_steps"]) and {3e-4: 1.0})
    pieces.piece_lr("S", "root", {"lr": {}}, str(tmp_path))
    assert seen["h"] == 2


def test_the_curves_train_at_the_pilot_h(stub, monkeypatch, tmp_path):
    pieces, runs, seen = stub
    monkeypatch.setattr(runs, "train_curve", lambda arm, ds, val, lr, its, every, h, *a, **k:
                        seen.setdefault("h", h) and [(1, 1.0)])
    pieces.piece_curve("S", "root", {"lr": {"S": 1e-3}}, str(tmp_path))
    assert seen["h"] == 2


def test_s_trains_at_the_pilot_h_before_its_h_check(monkeypatch, tmp_path):
    from p3b.pilot import pieces
    from p3b.train import trainer
    seen = {}
    monkeypatch.setattr(pieces, "files", lambda *a, **k: [])
    monkeypatch.setattr("p3b.latent.data.load_set", lambda paths: "ds")
    monkeypatch.setattr(pieces, "_train_in_sessions", lambda *a, **k: str(tmp_path / "ck.pt"))
    monkeypatch.setattr(pieces, "_prev_autoencoders", lambda *a, **k: None)
    monkeypatch.setattr(pieces, "_arm_checked", lambda *a, **k: "arm")

    class Stop(Exception):
        pass

    def train(arm, ds, cfg):
        seen["h"] = cfg.h_steps
        raise Stop
    monkeypatch.setattr(trainer, "train", train)
    with pytest.raises(Stop):
        pieces.piece_s("B", "root", {"lr": {"S": 1e-3}, "iterations": 10, "tau_S": {"B": 1.0}}, str(tmp_path))
    assert seen["h"] == 2


def test_the_wall_clock_is_measured_at_every_h_of_the_grid(monkeypatch, tmp_path):
    from p3b.pilot import pieces, runs
    calls = []
    monkeypatch.setattr(pieces, "files", lambda *a, **k: [])
    monkeypatch.setattr("p3b.latent.data.load_set", lambda paths: "ds")
    monkeypatch.setattr(runs, "wall_per_iter", lambda makers, ds, **k: calls.append(k["h_steps"]) or {"S": 2.0, "U": 1.0})
    written = {}
    monkeypatch.setattr(pieces, "_write", lambda out, name, value, *a, **k: written.setdefault("v", value))
    pieces.piece_wall("root", {}, str(tmp_path))
    assert set(written["v"]) == {"1", "2", "4"} and written["v"]["2"] == {"S": 2.0, "U": 1.0}
    assert {1, 2, 4} <= set(calls)


def test_u_long_uses_the_wall_clock_at_the_chosen_h():
    from p3b.pilot import rules
    wall = {"1": {"S": 9.0, "U": 1.0}, "2": {"S": 4.0, "U": 1.0}, "4": {"S": 3.0, "U": 1.0}}
    assert rules.iterations_ulong_at(1000, wall, 2) == 4000
    assert rules.iterations_ulong_at(1000, wall, 4) == 3000
