"""C+L training across Kaggle sessions: a run stopped and resumed from its checkpoint equals the run
done in one go, bit for bit (weights, losses, skipped counts, the pilot curve)."""
import numpy as np
import pytest
import torch

from p3b.generate import generate_trajectory
from p3b.latent.data import load_set
from p3b.solvers.c_plus_l import Multipliers
from p3b.train.cplusl import train_cplusl

NC = 8
POINTS = [dict(RH_ads=0.40, T_ads=303.15, T_des=353.15, v=0.10, alpha=0.8, beta=0.6, index=0, n_cycles=1),
          dict(RH_ads=0.55, T_ads=308.15, T_des=358.15, v=0.12, alpha=0.8, beta=0.6, index=1, n_cycles=1)]


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    d = tmp_path_factory.mktemp("clr")
    paths = []
    for pt in POINTS:
        p = d / f"A{pt['index']}.npz"
        assert generate_trajectory("A", pt, "MOF-303", N_z=32, N_c=NC, path=p, n_snap=20).ok
        paths.append(p)
    return load_set(paths)


def _fresh(seed=3):
    torch.manual_seed(seed)
    return Multipliers(n_out=3)


KW = dict(material="MOF-303", n_coarse=4, lr=1e-3, iterations=4, seed=3, n_windows=2, n_steps=2, K=1)


def test_a_resumed_run_equals_the_run_in_one_go(ds, tmp_path):
    m1 = _fresh()
    h1 = train_cplusl(m1, ds, **KW)
    ck = tmp_path / "cl_ckpt.pt"
    sessions = 0
    while True:
        m2 = _fresh()                                 # a new session builds the model afresh
        h2 = train_cplusl(m2, ds, **KW, checkpoint=str(ck), stop_after=1)
        sessions += 1
        if "stopped_at" not in h2:
            break
    assert sessions == 4                              # four one-iteration sessions; the fourth finishes
    assert np.array_equal(np.array(h2["loss"]), np.array(h1["loss"]), equal_nan=True)
    assert (h2["skipped"], h2["skipped_grad"]) == (h1["skipped"], h1["skipped_grad"])
    for a, b in zip(m1.state_dict().values(), m2.state_dict().values()):
        assert torch.equal(a, b)
    assert h2["resumed_from"] == 3


def test_a_deadline_stops_after_at_least_one_iteration_and_checkpoints(ds, tmp_path):
    import time
    ck = tmp_path / "ck.pt"
    h = train_cplusl(_fresh(), ds, **KW, checkpoint=str(ck), deadline=time.monotonic() - 1)
    assert h["stopped_at"] == 1 and ck.exists() and len(h["loss"]) == 1


def test_the_pilot_curve_survives_a_resume(ds, tmp_path):
    from p3b.pilot import runs
    kw = dict(material="MOF-303", lr=1e-3, iterations=4, every=2, K=4, seed=3, n_coarse=8, n_windows=2, n_steps=2)
    straight = runs.cplusl_curve(ds, ds, **kw)
    ck = tmp_path / "curve_ckpt.pt"
    while True:
        out = runs.cplusl_curve(ds, ds, **kw, checkpoint=str(ck), stop_after=1)
        if not isinstance(out, dict):
            break
        assert "stopped_at" in out
    assert [i for i, _ in out] == [2, 4] == [i for i, _ in straight]
    assert np.isfinite([v for _, v in straight]).all()
    assert np.array_equal(np.array([v for _, v in out]), np.array([v for _, v in straight]), equal_nan=True)


# ---- the pilot piece and its Kaggle entry point span sessions ------------------------------------
@pytest.fixture(scope="module")
def pilot_root(tmp_path_factory, ds):
    import shutil
    root = tmp_path_factory.mktemp("pilotroot")
    for split in ("train", "validation"):
        d = root / "A_MOF-303" / split
        d.mkdir(parents=True)
        for p in ds.paths:
            shutil.copy(p, d / os.path.basename(str(p)))
    return root


import os  # noqa: E402

STAGE = {"lr": {"C+L": 1e-3}, "K": {"A_MOF-303": 4}}
Q = {"curve_iterations": 4, "curve_every": 2, "n_coarse": 8, "n_windows": 2, "n_steps": 2}


def test_the_cplusl_curve_piece_resumes_across_sessions(pilot_root, tmp_path):
    import json
    import time
    from p3b.pilot import pieces
    straight = pieces.piece_curve("C+L", pilot_root, STAGE, tmp_path / "straight", Q)
    prev, sessions = None, 0
    while True:
        out = tmp_path / f"s{sessions}"
        res = pieces.piece_curve("C+L", pilot_root, STAGE, out, Q, deadline=time.monotonic() - 1,
                                 resume_from=str(prev) if prev else None)
        sessions += 1
        if os.path.basename(res) != "RESUME.json":
            break
        info = json.loads(open(res).read())
        assert info["piece"] == "curve-CpL" and info["stopped_at"] == sessions
        prev = out
    assert sessions == 4
    a, b = json.load(open(straight))["value"], json.load(open(res))["value"]
    assert np.array_equal(np.array(a), np.array(b), equal_nan=True)


def test_pilot_chunk_passes_hours_and_resume_to_the_curve_piece(monkeypatch, tmp_path):
    import importlib.util
    import pathlib
    import sys
    import time
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec = importlib.util.spec_from_file_location(
        "pilot_chunk", pathlib.Path(__file__).resolve().parent.parent / "kaggle" / "pilot_chunk.py")
    pc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pc)
    seen = {}

    def fake(arm, data, stage, out, q, deadline=None, resume_from=None):
        seen.update(arm=arm, deadline=deadline, resume_from=resume_from)
        return str(tmp_path / "curve_CpL.json")
    monkeypatch.setattr(pc.pieces, "piece_curve", fake)
    (tmp_path / "data").mkdir()
    t0 = time.monotonic()
    pc.main(["--piece", "curve-CpL", "--out", str(tmp_path / "o"), "--data", str(tmp_path / "data"),
             "--stage", str(tmp_path / "none.json"), "--hours", "2", "--resume-from", "/kaggle/input"])
    assert seen["arm"] == "C+L" and seen["resume_from"] == "/kaggle/input"
    assert t0 + 2 * 3600 - 5 <= seen["deadline"] <= time.monotonic() + 2 * 3600


# ---- the study's C+L cells span sessions too ------------------------------------------------------
@pytest.fixture(scope="module")
def study(tmp_path_factory):
    d = tmp_path_factory.mktemp("clst")
    for split, pts in (("train", POINTS), ("validation", [dict(POINTS[0], index=5)])):
        (d / split).mkdir()
        for pt in pts:
            assert generate_trajectory("A", pt, "MOF-303", N_z=32, N_c=NC, path=d / split / f"t{pt['index']}.npz",
                                       n_snap=20).ok
    return d


CP = {"lr": {"C+L": 1e-3}, "iterations": 3, "K": 4, "floors": {"N": 1e-6, "E": 1e-6}, "tau_S": 1e-6}
CQ = dict(n_coarse=8, n_windows=2, n_steps=2, n_cycles=1, pool_size=2)


def test_a_cplusl_cell_resumed_across_sessions_equals_the_cell_in_one_go(study, tmp_path):
    import json
    import time
    from p3b.train.cells import Cell, train_cell
    cell = Cell("C+L", "A", "MOF-303", "full", 0)
    straight = json.loads(train_cell(cell, str(study), str(tmp_path / "one"), CP, quick=CQ).read_text())
    prev, sessions = None, 0
    while True:
        out = tmp_path / f"s{sessions}"
        res = train_cell(cell, str(study), str(out), CP, quick=CQ, deadline=time.monotonic() - 1,
                         resume_from=str(prev) if prev else None)
        sessions += 1
        if res.name != "RESUME.json":
            break
        assert json.loads(res.read_text())["stopped_at"] == sessions
        prev = out
    assert sessions == 3
    got = json.loads(res.read_text())
    assert got["meta"]["weights"]["sha256"] == straight["meta"]["weights"]["sha256"]
    assert got["meta"]["resumed_from"] == 2


def test_the_cplusl_lr_piece_resumes_across_sessions(pilot_root, tmp_path):
    import json
    import time
    from p3b.pilot import pieces
    q = {**Q, "lr_grid": (3e-4, 1e-3), "lr_iterations": 2}
    straight = pieces.piece_lr("C+L", pilot_root, STAGE, tmp_path / "straight", q)
    prev, sessions = None, 0
    while True:
        out = tmp_path / f"s{sessions}"
        res = pieces.piece_lr("C+L", pilot_root, STAGE, out, q, deadline=time.monotonic() - 1,
                              resume_from=str(prev) if prev else None)
        sessions += 1
        if os.path.basename(res) != "RESUME.json":
            break
        assert json.loads(open(res).read())["piece"] == "lr-CpL"
        prev = out
    assert sessions == 3
    a, b = json.load(open(straight))["value"], json.load(open(res))["value"]
    assert set(a) == set(b) and np.array_equal(np.array([a[k] for k in sorted(a)]), np.array([b[k] for k in sorted(b)]),
                                               equal_nan=True)
