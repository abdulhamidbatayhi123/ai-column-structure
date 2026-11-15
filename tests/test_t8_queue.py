"""T8 review, session resume and the Kaggle queue (findings RS-1 to RS-8, with R1-4): one test or more
per finding, each failing on the code before the fix for the reason the finding states.

All runs are tiny (8 cells, 20 snapshots, one or two iterations per session) and use a deadline that
has already passed, so every session trains exactly one iteration and stops."""
import importlib.util
import json
import os
import pathlib
import shutil
import time
import types

import numpy as np
import pytest
import torch

from test_cplusl_resume import Q as CLQ  # noqa: F401
from test_cplusl_resume import STAGE as CLSTAGE
from test_cplusl_resume import ds, pilot_root  # noqa: F401  (fixtures)
from test_pilot_resume import _make
from test_pilot_runs import _setup, pilot_dir  # noqa: F401  (fixture)
from test_review_fixes_plan4 import study  # noqa: F401  (fixture)

from p3b.pilot import collect, pieces, runs

ROOT = pathlib.Path(__file__).resolve().parent.parent
KEY = "A_MOF-303"
OWNER = "abdulhamidbatayhi"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "kaggle" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- the fake Kaggle CLI of the queue tests ----------------------------------------------------------
@pytest.fixture
def rq(tmp_path, monkeypatch):
    m = _load("run_queue")
    monkeypatch.setattr(m, "BUILD", str(tmp_path / "build"))
    monkeypatch.setattr(m, "REG", str(tmp_path / "registry.txt"))
    monkeypatch.setattr(m, "ROOT", str(tmp_path))
    monkeypatch.setattr(m.time, "sleep", lambda s: None)
    monkeypatch.setenv("KAGGLE_CLI", "kaggle")
    return m


def _kaggle(rq, monkeypatch, outputs, complete=(), limit=80):
    """Kaggle as a script: kernels in `complete` (and every pushed kernel) are COMPLETE; a kernel's output
    is outputs[name] ({file: text}). More than `limit` CLI calls means the queue never stops."""
    state = {n: "COMPLETE" for n in complete}
    seen = {"pushes": [], "sources": {}, "calls": 0}

    def fake_run(cmd, capture_output, text, timeout):
        seen["calls"] += 1
        if seen["calls"] > limit:
            raise RuntimeError("the queue did not stop")
        verb, ref = cmd[2], cmd[3] if len(cmd) > 3 else ""
        if verb == "status":
            s = state.get(ref.split("/")[1])
            out = f'{ref} has status "KernelWorkerStatus.{s}"' if s else "404 Not Found"
        elif verb == "output":
            name = ref.split("/")[1]
            d = os.path.join(cmd[cmd.index("-p") + 1], "out")
            os.makedirs(d, exist_ok=True)
            for f, body in outputs[name].items():
                with open(os.path.join(d, f), "w") as fh:
                    fh.write(body)
            out = "downloaded"
        else:                                                   # push
            kdir = cmd[cmd.index("-p") + 1]
            name = os.path.basename(kdir)
            seen["pushes"].append(name)
            with open(os.path.join(kdir, "kernel-metadata.json")) as fh:
                seen["sources"][name] = json.load(fh)["kernel_sources"]
            state[name] = "COMPLETE"
            out = "Kernel version 1 successfully pushed"
        return types.SimpleNamespace(stdout=out, stderr="")
    monkeypatch.setattr(rq.subprocess, "run", fake_run)
    return seen


BASE = "p3b-pilot-lr-s-x"


def _spec(resumable=True, dependent=False):
    jobs = [{"name": BASE, "system": "pilot", "set": "lr-S", "threads": 1,
             "commands": [["kaggle/pilot_chunk.py", "--piece", "lr-S", "--out", "OUT"]], "expect": ["lr_S.json"]}]
    if resumable:
        jobs[0]["resumable"] = True
    if dependent:
        jobs.append({"name": "p3b-after-x", "system": "pilot", "set": "after", "threads": 1, "depends_on": [BASE],
                     "commands": [["kaggle/pilot_chunk.py", "--piece", "wall", "--out", "OUT"]],
                     "expect": ["wall_per_iter.json"]})
    return {"kit_slug": "o/p3b-reference-kit", "nz": 0, "jobs": jobs}


def _resume(stopped_at, progress=None):
    return {"RESUME.json": json.dumps({"stopped_at": stopped_at, "progress": progress or [stopped_at]})}


def _main(rq, tmp_path, spec):
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps(spec))
    rq.main(str(jobs))
    return open(rq.REG).read()


# ---- RS-1: latent lr sweeps keep their checkpoints across sessions; stuck chains stop -----------------
@pytest.fixture(scope="module")
def ae_root(tmp_path_factory, pilot_root):  # noqa: F811
    """A pilot autoencoder job's output, mounted as Kaggle mounts kernel inputs."""
    root = tmp_path_factory.mktemp("aeroot")
    pieces.piece_ae(pilot_root, KEY, 0, {"d": {KEY: 4}}, root / "ae-job" / "out", {"ae_epochs": 1, "ae_width": 8})
    return root


def _stage(ae_root):
    return {"d": {KEY: 4}, "K": {KEY: 4}, "tau_S": {KEY: 1e-6}, "lr": {"U": 1e-3, "S": 1e-3, "C+L": 1e-3},
            "iterations": 2, "h_steps": {KEY: 1}, "ae_input_root": str(ae_root)}


LQ = {"ae_width": 8, "width": 8, "lr_grid": (3e-4, 1e-3), "lr_iterations": 2, "n_windows": 1, "n_steps": 2,
      "iterations": 2}


def _sessions(run, tmp_path, limit=6):
    """Run a piece session after session (each in a new output directory, the previous one as input)
    until it writes its result; returns (result path, sessions)."""
    prev = None
    for n in range(1, limit + 1):
        out = tmp_path / f"s{n}"
        res = run(out, deadline=time.monotonic() - 1, resume_from=str(prev) if prev else None)
        if os.path.basename(str(res)) != "RESUME.json":
            return res, n
        prev = out
    return res, limit


def test_rs1_a_latent_lr_piece_resumes_with_a_fresh_output_each_session(pilot_root, ae_root, tmp_path):  # noqa: F811
    stage = _stage(ae_root)
    straight = json.load(open(pieces.piece_lr("U", pilot_root, stage, tmp_path / "straight", LQ)))["value"]
    res, n = _sessions(lambda out, **kw: pieces.piece_lr("U", pilot_root, stage, out, LQ, **kw), tmp_path)
    assert os.path.basename(res) == "lr_U.json" and n == 3        # two rates x two iterations, one per session
    got = json.load(open(res))["value"]
    assert set(got) == set(straight)
    assert np.array_equal(np.array([got[k] for k in sorted(got)]), np.array([straight[k] for k in sorted(straight)]),
                          equal_nan=True)
    assert any(f.startswith("lr_U_") and f.endswith("_ckpt.pt") for f in os.listdir(tmp_path / "s1"))


def test_rs1_train_curve_refuses_a_state_its_checkpoint_does_not_hold(pilot_dir, tmp_path):  # noqa: F811
    ds_, val, std, ae, lf = _setup(pilot_dir)
    kw = dict(lr=1e-3, iterations=4, every=2, h_steps=1, seed=0, n_windows=1, n_steps=2)
    ck = str(tmp_path / "ck.pt")
    with pytest.raises(FileNotFoundError, match="checkpoint"):         # a state at it 1 but no checkpoint
        runs.train_curve(_make(ds_, ae, std, lf), ds_, val, **kw, checkpoint=ck, state={"curve": [], "it": 1})
    out = runs.train_curve(_make(ds_, ae, std, lf), ds_, val, **kw, checkpoint=ck, deadline=time.monotonic() - 1)
    assert out["state"]["it"] == 1
    with pytest.raises(RuntimeError, match="checkpoint"):              # the checkpoint holds it 1, not 2
        runs.train_curve(_make(ds_, ae, std, lf), ds_, val, **kw, checkpoint=ck, state={"curve": [], "it": 2})


def test_rs1_the_queue_stops_a_chain_whose_resume_point_does_not_advance(rq, tmp_path, monkeypatch):
    # the base stops at rate 0, iteration 5; r1 at rate 0, iteration 3: no progress, so no r2
    outputs = {BASE: _resume(5, [0, 5]), f"{BASE}-r1": _resume(3, [0, 3]), f"{BASE}-r2": _resume(5, [0, 5]),
               f"{BASE}-r3": _resume(3, [0, 3])}
    seen = _kaggle(rq, monkeypatch, outputs, complete=(BASE,))
    reg = _main(rq, tmp_path, _spec())
    assert seen["pushes"] == [f"{BASE}-r1"]
    assert sum(f"{BASE}-r1" in line and "needs attention" in line for line in reg.splitlines()) == 1
    assert "ALL_OUTPUTS_DOWNLOADED" not in reg


def test_rs1_the_queue_caps_the_continuations_of_a_chain(rq, tmp_path, monkeypatch):
    monkeypatch.setattr(rq, "MAX_CONTINUATIONS", 2)
    outputs = {BASE: _resume(1), **{f"{BASE}-r{n}": _resume(n + 1) for n in range(1, 6)}}
    seen = _kaggle(rq, monkeypatch, outputs, complete=(BASE,))
    reg = _main(rq, tmp_path, _spec())
    assert seen["pushes"] == [f"{BASE}-r1", f"{BASE}-r2"]
    assert sum("needs attention" in line for line in reg.splitlines()) == 1
    assert "ALL_OUTPUTS_DOWNLOADED" not in reg


# ---- cells: the study's tiny cell set-up --------------------------------------------------------------
CP = {"lr": {"U": 1e-3, "C+L": 1e-3}, "iterations": 3, "h_steps": 1, "d": 4, "kappa": 1e-3, "K": 4,
      "floors": {"N": 1e-6, "E": 1e-6}, "tau_S": 1e-6}
Q = dict(ae_epochs=1, ae_width=8, width=8, n_windows=1, n_steps=2, n_cycles=1, pool_size=2, n_coarse=8)
AE = "ae_A_MOF-303_full_s0.pt"


def _cell(arm="U"):
    from p3b.train.cells import Cell
    return Cell(arm, "A", "MOF-303", "full", 0, ())


# ---- RS-2: a third session with the autoencoder job mounted beside the previous output ----------------
def test_rs2_a_cell_resumed_twice_with_its_autoencoder_job_mounted_beside_each_output(study, tmp_path):  # noqa: F811
    from p3b.train.cells import _sha, train_cell
    cell = _cell()
    pre = tmp_path / "pre"                                   # makes the autoencoder file of the "ae job"
    train_cell(cell, str(study), str(pre), CP, quick=Q, deadline=time.monotonic() - 1)
    res, prev = None, None
    for n in (1, 2, 3):
        inp = tmp_path / f"in{n}"
        (inp / "aejob").mkdir(parents=True)
        shutil.copy(pre / AE, inp / "aejob" / AE)
        if prev is not None:
            shutil.copytree(prev, inp / "prev")
        out = tmp_path / f"s{n}"
        res = train_cell(cell, str(study), str(out), dict(CP, ae_input_root=str(inp)), quick=Q,
                         deadline=time.monotonic() - 1, resume_from=str(inp) if prev is not None else None)
        if n < 3:
            assert res.name == "RESUME.json"
        prev = out
    got = json.loads(res.read_text())["meta"]
    assert got["resumed_from"] == 2 and got["autoencoder"]["sha256"] == _sha(pre / AE)


# ---- RS-3: dependents mount the final kernel; evaluation finds the weights under its inputs -----------
def test_rs3_a_dependent_mounts_the_kernel_that_produced_the_output(rq, tmp_path, monkeypatch):
    outputs = {BASE: _resume(1), f"{BASE}-r1": _resume(2), f"{BASE}-r2": {"lr_S.json": "{}"},
               "p3b-after-x": {"wall_per_iter.json": "{}"}}
    seen = _kaggle(rq, monkeypatch, outputs, complete=(BASE,))
    reg = _main(rq, tmp_path, _spec(dependent=True))
    assert seen["pushes"] == [f"{BASE}-r1", f"{BASE}-r2", "p3b-after-x"]
    assert seen["sources"]["p3b-after-x"] == [f"{OWNER}/{BASE}-r2"]
    assert "ALL_OUTPUTS_DOWNLOADED" in reg


def test_rs3_evaluation_reads_the_weights_from_the_mounted_training_output(study, tmp_path):  # noqa: F811
    from p3b.train.cells import _weights_file, evaluate_cell, train_cell
    cell = _cell()
    inp = tmp_path / "input"
    pilot = dict(CP, iterations=1, ae_input_root=str(inp))
    trained = json.loads(train_cell(cell, str(study), str(inp / "train-kernel"), pilot, quick=Q).read_text())
    gate = {KEY: "pass"}
    out = evaluate_cell(cell, "test", str(study), str(tmp_path / "eval"), pilot, gate, quick=Q, weights_from=str(inp))
    assert json.loads(out.read_text())["meta"]["weights"]["sha256"] == trained["meta"]["weights"]["sha256"]
    w = inp / "train-kernel" / _weights_file(cell)
    sd = torch.load(w)
    torch.save({k: v + 1.0 for k, v in sd.items()}, w)                  # not the weights training recorded
    with pytest.raises(RuntimeError, match="sha256"):
        evaluate_cell(cell, "test", str(study), str(tmp_path / "eval2"), pilot, gate, quick=Q, weights_from=str(inp))
    torch.save(sd, w)
    (inp / "other").mkdir()
    shutil.copy(w, inp / "other" / w.name)                              # two copies: which one is meant?
    with pytest.raises(RuntimeError, match="2"):
        evaluate_cell(cell, "test", str(study), str(tmp_path / "eval3"), pilot, gate, quick=Q, weights_from=str(inp))


def test_rs3_a_cplusl_evaluation_reads_the_multipliers_from_the_mounted_training_output(study, tmp_path):  # noqa: F811
    from p3b.train.cells import evaluate_cell, train_cell
    cell = _cell("C+L")
    inp = tmp_path / "input"
    pilot = dict(CP, iterations=1)
    trained = json.loads(train_cell(cell, str(study), str(inp / "train-kernel"), pilot, quick=Q).read_text())
    out = evaluate_cell(cell, "test", str(study), str(tmp_path / "eval"), pilot, {KEY: "pass"}, quick=Q,
                        weights_from=str(inp))
    assert json.loads(out.read_text())["meta"]["weights"]["sha256"] == trained["meta"]["weights"]["sha256"]


def test_rs3_train_chunk_evaluates_with_the_weights_under_the_kernel_inputs(tmp_path, monkeypatch):
    import sys
    monkeypatch.setattr(sys, "path", list(sys.path))
    tc = _load("train_chunk")
    seen = {}

    def fake(c, set_name, data_dir, out, pilot, gate, quick=None, weights_from=None):
        seen.update(set_name=set_name, weights_from=weights_from)
        return "x"
    monkeypatch.setattr(tc, "evaluate_cell", fake)
    monkeypatch.setattr(tc, "cell_pilot", lambda c, pilot, choice=None: {})
    for name, body in (("p.json", "{}"), ("choice.json", "{}"), ("gate.json", "{}")):
        (tmp_path / name).write_text(body)
    cell = tc.job_name(next(c for c in tc.cells() if c.arm == "U"))
    tc.main(["--evaluate", cell, "test", "--out", str(tmp_path / "o"), "--pilot", str(tmp_path / "p.json"),
             "--decoder-choice", str(tmp_path / "choice.json"), "--gate", str(tmp_path / "gate.json")])
    assert seen == {"set_name": "test", "weights_from": "/kaggle/input"}


# ---- RS-4 / R1-4: a pilot revision tag in the stage hash and names; the collector checks the data -----
def test_rs4_the_pilot_revision_enters_the_stage_hash_and_every_kernel_name(monkeypatch):
    import hashlib
    assert pieces.stage_sha({}) != hashlib.sha256(b"{}").hexdigest()        # phase 0's old suffix 44136f
    pj = _load("pilot_jobs")
    for ph in range(5):
        for j in pj.phase_jobs(ph)["jobs"]:
            assert f"-{pieces.PILOT_REVISION}-" in j["name"] and not j["name"].endswith("44136f")
    before = pieces.stage_sha({})
    monkeypatch.setattr(pieces, "PILOT_REVISION", "r-next")
    assert pieces.stage_sha({}) != before


def test_rs4_the_collector_refuses_a_piece_whose_data_files_changed(tmp_path):
    data = tmp_path / "data" / KEY / "train"
    data.mkdir(parents=True)
    f = data / "t0.npz"
    f.write_bytes(b"before")
    pieces._write(tmp_path / "pieces" / "k", f"data_{KEY}.json", {"x": 1}, stage={},
                  used={"data": pieces._data_used([str(f)])})
    assert collect.load_piece(tmp_path / "pieces", f"data_{KEY}.json", {}, data_root=tmp_path / "data") == {"x": 1}
    f.write_bytes(b"regenerated")                                         # the data were regenerated (R3)
    with pytest.raises(RuntimeError, match="data"):
        collect.load_piece(tmp_path / "pieces", f"data_{KEY}.json", {}, data_root=tmp_path / "data")
    f.unlink()                                                            # or are not there at all
    with pytest.raises(RuntimeError, match="data"):
        collect.load_piece(tmp_path / "pieces", f"data_{KEY}.json", {}, data_root=tmp_path / "data")


# ---- RS-5: a continuation refuses another stage, kit or configuration ---------------------------------
def test_rs5_a_cell_refuses_to_resume_under_another_configuration(study, tmp_path):  # noqa: F811
    from p3b.train.cells import train_cell
    cell = _cell()
    p1 = train_cell(cell, str(study), str(tmp_path / "s1"), CP, quick=Q, deadline=time.monotonic() - 1)
    assert "fingerprint" in json.loads(p1.read_text())
    with pytest.raises(RuntimeError, match="fingerprint"):
        train_cell(cell, str(study), str(tmp_path / "s2"), dict(CP, iterations=5), quick=Q,
                   resume_from=str(tmp_path / "s1"))


def test_rs5_a_pilot_piece_refuses_to_resume_under_another_kit_or_q(pilot_root, tmp_path, monkeypatch):  # noqa: F811
    q = {**CLQ, "lr_grid": (3e-4, 1e-3), "lr_iterations": 2}
    p1 = pieces.piece_lr("C+L", pilot_root, CLSTAGE, tmp_path / "s1", q, deadline=time.monotonic() - 1)
    assert "fingerprint" in json.load(open(p1))
    with pytest.raises(RuntimeError, match="fingerprint"):
        pieces.piece_lr("C+L", pilot_root, CLSTAGE, tmp_path / "s2", {**q, "lr_iterations": 3},
                        deadline=time.monotonic() - 1, resume_from=str(tmp_path / "s1"))
    real = pieces._kit_commit()
    monkeypatch.setattr(pieces, "_kit_commit", lambda: f"not-{real}")
    with pytest.raises(RuntimeError, match="fingerprint"):
        pieces.piece_lr("C+L", pilot_root, CLSTAGE, tmp_path / "s3", q, deadline=time.monotonic() - 1,
                        resume_from=str(tmp_path / "s1"))


def test_rs5_the_cplusl_curve_piece_refuses_another_iteration_count(pilot_root, tmp_path):  # noqa: F811
    p1 = pieces.piece_curve("C+L", pilot_root, CLSTAGE, tmp_path / "s1", CLQ, deadline=time.monotonic() - 1)
    assert "fingerprint" in json.load(open(p1))
    with pytest.raises(RuntimeError, match="fingerprint"):
        pieces.piece_curve("C+L", pilot_root, CLSTAGE, tmp_path / "s2", {**CLQ, "curve_iterations": 6},
                           deadline=time.monotonic() - 1, resume_from=str(tmp_path / "s1"))


# ---- RS-6: a continuation without its checkpoint is an error, not a restart ---------------------------
@pytest.mark.parametrize("arm", ["U", "C+L"])
def test_rs6_a_continuation_without_its_checkpoint_raises(study, tmp_path, arm):  # noqa: F811
    from p3b.train.cells import train_cell
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="ckpt"):
        train_cell(_cell(arm), str(study), str(tmp_path / "s2"), CP, quick=Q, deadline=time.monotonic() - 1,
                   resume_from=str(tmp_path / "empty"))
    assert not (tmp_path / "s2" / "RESUME.json").exists()


# ---- RS-7: no stale phase-1 jobs file; a COMPLETE kernel with nothing usable is reported once ---------
def test_rs7_the_stale_pilot_jobs_files_are_not_in_the_repository():
    # jobs_pilot1.json: pre-R1 names (c21879), no resumable flags; jobs_pilot0.json: pre-revision names
    # (44136f), RS-4 ruling. A phase's file is generated just before queueing, so only git tracking counts.
    import subprocess
    try:
        tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "kaggle/jobs_pilot0.json",
                                  "kaggle/jobs_pilot1.json"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    assert tracked.strip() == ""


def test_rs7_a_complete_kernel_with_neither_output_nor_usable_resume_is_reported_once(rq, tmp_path, monkeypatch):
    seen = _kaggle(rq, monkeypatch, {BASE: _resume(4)}, complete=(BASE,))     # not resumable, yet RESUME.json
    reg = _main(rq, tmp_path, _spec(resumable=False))
    assert seen["pushes"] == []
    assert sum(BASE in line and "needs attention" in line for line in reg.splitlines()) == 1
    assert "ALL_OUTPUTS_DOWNLOADED" not in reg


# ---- RS-8: the s-<key>, two-* and pcq-* pieces span sessions like the sweeps --------------------------
def test_rs8_the_s_and_two_seed_pieces_are_resumable_jobs():
    pj = _load("pilot_jobs")
    p3 = {j["set"]: j for j in pj.phase_jobs(3)["jobs"]}
    assert all(p3[f"s-{k}"].get("resumable") for k in pieces.KEYS) and not p3["wall"].get("resumable")
    p4 = pj.phase_jobs(4)["jobs"]
    assert p4 and all(j.get("resumable") for j in p4)


def test_rs8_pilot_chunk_passes_the_deadline_and_resume_to_the_s_and_two_seed_pieces(tmp_path, monkeypatch):
    import sys
    monkeypatch.setattr(sys, "path", list(sys.path))
    pc = _load("pilot_chunk")
    seen = []

    def fake_s(key, data, stage, out, q, deadline=None, resume_from=None):
        seen.append(("s", key, deadline is not None, resume_from))
        return str(tmp_path / "s.json")

    def fake_two(arm, seed, data, stage, out, q, quarter=False, deadline=None, resume_from=None):
        seen.append(("two", arm, seed, quarter, deadline is not None, resume_from))
        return str(tmp_path / "rows.json")
    monkeypatch.setattr(pc.pieces, "piece_s", fake_s)
    monkeypatch.setattr(pc.pieces, "piece_two", fake_two)
    (tmp_path / "data").mkdir()
    for piece in (f"s-{KEY}", "two-S-s1", "pcq-s0"):
        pc.main(["--piece", piece, "--out", str(tmp_path / "o"), "--data", str(tmp_path / "data"),
                 "--stage", str(tmp_path / "none.json"), "--resume-from", "/kaggle/input"])
    assert seen == [("s", KEY, True, "/kaggle/input"), ("two", "S", 1, False, True, "/kaggle/input"),
                    ("two", "U", 0, True, True, "/kaggle/input")]


def _rows(path):
    v = json.load(open(path))["value"]
    return json.dumps({"rows": [{k: x for k, x in r.items() if k != "cpu_s"} for r in v["rows"]],
                       "train": v["train"]}, sort_keys=True)


def test_rs8_piece_two_resumes_across_sessions_to_the_same_rows(pilot_root, ae_root, tmp_path):  # noqa: F811
    stage = _stage(ae_root)
    straight = pieces.piece_two("U", 0, pilot_root, stage, tmp_path / "straight", LQ)
    res, n = _sessions(lambda out, **kw: pieces.piece_two("U", 0, pilot_root, stage, out, LQ, **kw), tmp_path)
    assert os.path.basename(res) == "rows_U_s0.json" and n == 2
    assert _rows(res) == _rows(straight)


def test_rs8_piece_s_resumes_across_sessions(pilot_root, ae_root, tmp_path, monkeypatch):  # noqa: F811
    # the session plumbing only: U stands in for S (an S of d = 4 on 20 snapshots diverges), the h check
    # and kappa are stubs
    orig = pieces.make_arm
    monkeypatch.setattr(pieces, "make_arm", lambda arm, *a, **k: orig("U", *a, **k))
    monkeypatch.setattr(pieces, "_kappa", lambda *a, **k: (1.0, [1.0]))
    monkeypatch.setattr(runs, "h_check", lambda arm, val, tau, steps=(1,): {"h": {1: {"viol_rate": 0.0}},
                                                                            "defects_per_step": {1: [0.0]}})
    stage = _stage(ae_root)
    straight = json.load(open(pieces.piece_s(KEY, pilot_root, stage, tmp_path / "straight", LQ)))["value"]
    res, n = _sessions(lambda out, **kw: pieces.piece_s(KEY, pilot_root, stage, out, LQ, **kw), tmp_path)
    assert os.path.basename(res) == f"s_{KEY}.json" and n == 2
    assert json.load(open(tmp_path / "s1" / "RESUME.json"))["piece"] == f"s-{KEY}"
    assert json.load(open(res))["value"] == straight
