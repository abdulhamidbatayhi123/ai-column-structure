"""Pilot sweeps across Kaggle sessions: the curve and learning-rate pieces stop past a deadline and
resume from their checkpoint and state; the resumed result equals the one done in one go."""
import time

import numpy as np

from test_pilot_runs import _setup, pilot_dir  # noqa: F401  (fixture)

from p3b.pilot import runs


def _make(ds, ae, std, lf):
    from p3b.latent.arms_unstructured import UDynamics
    from p3b.train.trainer import Arm, time_scale
    import torch
    torch.manual_seed(0)
    return Arm("U", UDynamics(4, 4, 8, time_scale=time_scale(ds)), ae, std, lf)


def _eq(a, b):
    return np.array_equal(np.array(a, dtype=float), np.array(b, dtype=float), equal_nan=True)


def test_train_curve_resumed_in_sessions_equals_the_curve_in_one_go(pilot_dir, tmp_path):  # noqa: F811
    ds, val, std, ae, lf = _setup(pilot_dir)
    kw = dict(lr=1e-3, iterations=4, every=2, h_steps=1, seed=0, n_windows=1, n_steps=2)
    straight = runs.train_curve(_make(ds, ae, std, lf), ds, val, **kw)
    state, sessions = None, 0
    while True:
        out = runs.train_curve(_make(ds, ae, std, lf), ds, val, **kw, checkpoint=str(tmp_path / "ck.pt"),
                               deadline=time.monotonic() - 1, state=state)
        sessions += 1
        if not isinstance(out, dict):
            break
        state = out["state"]
    assert sessions == 3                       # at least one iteration per session; a point reached is recorded
    assert [i for i, _ in out] == [2, 4] and _eq([v for _, v in out], [v for _, v in straight])


def test_lr_sweep_resumed_in_sessions_equals_the_sweep_in_one_go(pilot_dir, tmp_path):  # noqa: F811
    ds, val, std, ae, lf = _setup(pilot_dir)
    kw = dict(rates=(3e-4, 1e-3), iterations=2, h_steps=1, seed=0, n_windows=1, n_steps=2)
    straight = runs.lr_sweep(lambda: _make(ds, ae, std, lf), ds, val, **kw)
    state, sessions = None, 0
    while True:
        out = runs.lr_sweep(lambda: _make(ds, ae, std, lf), ds, val, **kw, checkpoint_dir=str(tmp_path),
                            deadline=time.monotonic() - 1, state=state)
        sessions += 1
        if "stopped_at" not in out:
            break
        state = out["state"]
    assert sessions == 3
    assert set(out) == set(straight) and _eq([out[k] for k in sorted(out)], [straight[k] for k in sorted(straight)])
