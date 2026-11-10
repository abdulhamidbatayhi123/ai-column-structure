"""Resuming across Kaggle sessions (2026-11-10): a run that would pass the 12 h session limit pauses at a
deadline with a checkpoint and a RESUME.json, and a later session continues it to the same weights.

Measured on Kaggle: S at h = 2 and 8,000 iterations takes about 14 h at four runs per session
(docs/compute_3B.md), so confirmatory cells need this.
"""
import json
import time

import pytest
import torch
from torch import nn

from p3b.generate import generate_trajectory
from p3b.latent.arms_unstructured import UDynamics
from p3b.latent.data import Standardiser, load_set
from p3b.train.trainer import Arm, TrainConfig, train

POINT_A = dict(RH_ads=0.40, T_ads=303.15, T_des=353.15, v=0.10, alpha=0.8, beta=0.6, index=0, n_cycles=1)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("res")
    for split, pts in (("train", [POINT_A, dict(POINT_A, RH_ads=0.3, index=1)]),
                       ("validation", [dict(POINT_A, index=2)])):
        (d / split).mkdir(parents=True)
        for pt in pts:
            assert generate_trajectory("A", pt, "MOF-303", N_z=32, N_c=8, path=d / split / f"{pt['index']}.npz",
                                       n_snap=40).ok
    return d


class I(nn.Module):
    def encode(self, x):
        return x

    def decode(self, z):
        return z


def _run(ds, ck, deadline=None):
    std = Standardiser.fit(ds.X, 8, 3, 2, features=ds.features)
    torch.manual_seed(0)
    dyn = UDynamics(ds.X.shape[-1], ds.features.shape[-1], 8, time_scale=100.0)
    nn.init.normal_(dyn.net[-1].weight, std=0.01)
    h = train(Arm("U", dyn, I(), std, None), ds,
              TrainConfig(lr=1e-3, iterations=4, seed=0, n_windows=1, n_steps=2, checkpoint=ck, every=1,
                          deadline=deadline))
    return dyn, h


def test_a_passed_deadline_pauses_after_one_iteration_and_resuming_gives_the_same_weights(data, tmp_path):
    ds = load_set(sorted((data / "train").glob("*.npz")))
    full, h_full = _run(ds, None)
    ck = str(tmp_path / "ck.pt")
    _, h1 = _run(ds, ck, deadline=time.monotonic())        # already passed: one iteration, then pause
    assert h1["stopped_at"] == 1
    resumed, h2 = _run(ds, ck)
    assert h2["resumed_from"] == 1 and "stopped_at" not in h2
    assert all(torch.allclose(a, b) for a, b in zip(full.state_dict().values(), resumed.state_dict().values()))
    assert len(h2["loss"]) == len(h_full["loss"]) == 4


def test_cpu_time_accumulates_over_sessions(data, tmp_path):
    ds = load_set(sorted((data / "train").glob("*.npz")))
    ck = str(tmp_path / "ck.pt")
    _, h1 = _run(ds, ck, deadline=time.monotonic())
    _, h2 = _run(ds, ck)
    assert h2["wall_s"] > h1["wall_s"] > 0


def test_train_cell_pauses_with_resume_json_and_continues_from_another_directory(data, tmp_path):
    from p3b.train.cells import Cell, job_name, train_cell
    pilot = {"lr": {"U": 1e-3}, "iterations": 3, "h_steps": 1, "d": 6, "kappa": 1e-3,
             "floors": {"N": 1e-6, "E": 1e-6}, "tau_S": 1e-6}
    q = dict(ae_epochs=2, ae_width=16, width=16, n_windows=2, n_steps=3, n_cycles=1, pool_size=2)
    cell = Cell("U", "A", "MOF-303", "full", 0, ())
    first = tmp_path / "s1"
    p1 = train_cell(cell, str(data), str(first), pilot, quick=q, deadline=time.monotonic())
    assert p1.name == "RESUME.json"
    r = json.loads(p1.read_text())
    assert r["job"] == job_name(cell) and r["stopped_at"] == 1
    assert not (first / (job_name(cell) + ".json")).exists()
    nested = tmp_path / "input" / "prev"                    # Kaggle mounts a kernel's output one level down
    nested.parent.mkdir()
    first.rename(nested)
    second = tmp_path / "s2"
    p2 = train_cell(cell, str(data), str(second), pilot, quick=q, resume_from=str(tmp_path / "input"))
    res = json.loads(p2.read_text())
    assert res["meta"]["resumed_from"] == 1 and res["meta"]["iterations"] == 3
    ae = "ae_A_MOF-303_full_s0.pt"
    assert (second / ae).read_bytes() == (nested / ae).read_bytes()      # the same autoencoder, not a new one


def test_train_chunk_passes_a_deadline_and_the_resume_directory(tmp_path, monkeypatch):
    import importlib.util
    import pathlib
    import sys
    monkeypatch.setattr(sys, "path", list(sys.path))       # train_chunk edits sys.path at import; keep it local
    spec = importlib.util.spec_from_file_location(
        "train_chunk", pathlib.Path(__file__).resolve().parent.parent / "kaggle" / "train_chunk.py")
    tc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tc)
    seen = {}

    def fake(c, data_dir, out, pilot, deadline=None, resume_from=None):
        seen.update(deadline=deadline, resume_from=resume_from)
        return "x"
    monkeypatch.setattr(tc, "train_cell", fake)
    monkeypatch.setattr(tc, "cell_pilot", lambda c, pilot: {})
    pilot = tmp_path / "p.json"
    pilot.write_text("{}")
    name = tc.job_name(next(c for c in tc.cells() if c.arm == "U"))
    t0 = time.monotonic()
    tc.main(["--cell", name, "--out", str(tmp_path / "o"), "--pilot", str(pilot), "--resume-from", "/kaggle/input",
             "--hours", "2"])
    assert seen["resume_from"] == "/kaggle/input"
    assert t0 + 2 * 3600 - 5 <= seen["deadline"] <= time.monotonic() + 2 * 3600
