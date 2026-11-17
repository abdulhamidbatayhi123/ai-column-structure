"""T8 fix review (T8_files/fix_review_queue.json): one test or more per issue, each failing on the
branch before the fix for the reason the issue states.

- R1-4 (important): the stale pre-R1/pre-R3 results/pilot_stage.json was committed and copied into the
  kit; a stage file now carries PILOT_REVISION, pilot_jobs.py and pilot_chunk.py refuse one of another
  revision, and pilot_chunk hands every piece the stage of its own phase (a phase-0 piece records
  stage_sha({}) whatever stage the kit carries);
- the queue abandoned a COMPLETE kernel after two failed (429) downloads;
- RS-5 for pilot pieces: a continuation under another autoencoder resumed without error;
- a dependent outside its producer's jobs file was never pushed; evaluation_jobs now carries the
  training jobs, the queue refuses a dependency outside the file, and the replay finds the producer
  among a finished chain's continuations.
"""
import json
import os
import runpy
import subprocess
import sys
import time
import types

import pytest

from test_cplusl_resume import ds, pilot_root  # noqa: F401  (fixtures; pilot_root needs ds)
from test_pilot_collect import _phase0
from test_t8_queue import BASE, LQ, OWNER, ROOT, _kaggle, _load, _main, _sessions, _spec, _stage, ae_root, rq  # noqa: F401

from p3b.pilot import collect, pieces

KEY = pieces.PRIMARY_KEY


# ---- R1-4: the stage file carries the pilot revision ---------------------------------------------------
def test_r14_a_committed_pilot_stage_file_carries_the_current_revision():
    # the pre-R1/pre-R3 file (d 64/64/16, tau_S of the old data) had no revision and went into every kit
    try:
        tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "results/pilot_stage.json"],
                                 capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    if tracked:
        with open(ROOT / "results" / "pilot_stage.json") as f:
            assert json.load(f).get("_revision") == pieces.PILOT_REVISION


def test_r14_collect_stage_records_the_pilot_revision(tmp_path):
    _phase0(tmp_path)
    s = collect.stage(0, tmp_path, keys=(KEY,))
    assert s["_revision"] == pieces.PILOT_REVISION
    assert collect.stage_sha(s) == collect.stage_sha({k: v for k, v in s.items() if k != "_revision"})


def _stage_file(tmp_path, name, **extra):
    st = {"d": {KEY: 24}, "decoder": {KEY: "mlp"}, "K": {KEY: 2}, "tau_S": {KEY: 1.0}, **extra}
    p = tmp_path / name
    p.write_text(json.dumps(st))
    return p


def _pilot_jobs_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", ["pilot_jobs.py", *map(str, argv)])
    runpy.run_path(str(ROOT / "kaggle" / "pilot_jobs.py"), run_name="__main__")


def test_r14_pilot_jobs_refuses_a_stage_without_the_current_revision(tmp_path, monkeypatch):
    stale = _stage_file(tmp_path, "stale.json")                                    # no revision at all
    other = _stage_file(tmp_path, "other.json", _revision="r2")
    for p in (stale, other):
        with pytest.raises((RuntimeError, SystemExit), match="revision"):
            _pilot_jobs_main(monkeypatch, 1, tmp_path / "jobs1.json", p)
        assert not (tmp_path / "jobs1.json").exists()
    good = _stage_file(tmp_path, "good.json", _revision=pieces.PILOT_REVISION)
    _pilot_jobs_main(monkeypatch, 1, tmp_path / "jobs1.json", good)
    assert json.loads((tmp_path / "jobs1.json").read_text())["jobs"]
    _pilot_jobs_main(monkeypatch, 0, tmp_path / "jobs0.json", stale)              # phase 0 needs no stage
    names = [j["name"] for j in json.loads((tmp_path / "jobs0.json").read_text())["jobs"]]
    assert all(n.endswith(pieces.stage_sha({})[:6]) for n in names)


def _chunk(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    return _load("pilot_chunk")


def test_r14_pilot_chunk_refuses_a_stage_of_another_revision(tmp_path, monkeypatch):
    pc = _chunk(monkeypatch)
    ran = []
    monkeypatch.setattr(pc.pieces, "piece_lr", lambda *a, **k: ran.append(1) or str(tmp_path / "lr_S.json"))
    (tmp_path / "data").mkdir()
    for p in (_stage_file(tmp_path, "stale.json"), _stage_file(tmp_path, "other.json", _revision="r2")):
        with pytest.raises((RuntimeError, SystemExit), match="revision"):
            pc.main(["--piece", "lr-S", "--out", str(tmp_path / "o"), "--data", str(tmp_path / "data"),
                     "--stage", str(p)])
    assert ran == []


def test_r14_a_phase0_piece_run_with_a_non_empty_kit_stage_is_collected(tmp_path, monkeypatch):
    # the kit carries a stage file (here a stale one); phase 0 must record stage_sha({}) all the same
    pc = _chunk(monkeypatch)
    src = tmp_path / "src"
    _phase0(src)
    values = {n: json.load(open(src / n))["value"] for n in (f"data_{KEY}.json", f"cplusl_K_{KEY}.json")}

    def fake(name):
        return lambda data, key, out, q=None, stage=None: pieces._write(out, name, values[name], stage, q)
    monkeypatch.setattr(pc.pieces, "piece_data", fake(f"data_{KEY}.json"))
    monkeypatch.setattr(pc.pieces, "piece_cplk", fake(f"cplusl_K_{KEY}.json"))
    (tmp_path / "data").mkdir()
    kit_stage = _stage_file(tmp_path, "kit_stage.json")
    for piece in (f"data-{KEY}", f"cplk-{KEY}"):
        pc.main(["--piece", piece, "--out", str(tmp_path / "pieces" / piece), "--data", str(tmp_path / "data"),
                 "--stage", str(kit_stage)])
    s = collect.stage(0, tmp_path / "pieces", keys=(KEY,))
    assert s["d"] == {KEY: 24}


def test_r14_pilot_chunk_hands_a_piece_the_stage_of_its_own_phase(tmp_path, monkeypatch):
    # a stage file of a later collection (lr chosen) must not change what a phase-1 piece records
    pc = _chunk(monkeypatch)
    seen = {}

    def fake_lr(arm, data, stage, out, q, deadline=None, resume_from=None):
        seen["stage"] = stage
        return str(tmp_path / "lr_S.json")
    monkeypatch.setattr(pc.pieces, "piece_lr", fake_lr)
    (tmp_path / "data").mkdir()
    st = _stage_file(tmp_path, "s.json", lr={"S": 1e-3}, _revision=pieces.PILOT_REVISION)
    pc.main(["--piece", "lr-S", "--out", str(tmp_path / "o"), "--data", str(tmp_path / "data"), "--stage", str(st)])
    assert "lr" not in seen["stage"] and seen["stage"]["d"] == {KEY: 24}
    assert pieces.stage_sha(seen["stage"]) == pieces.stage_sha(pieces.restrict(json.loads(st.read_text()), 1))


# ---- the queue retries a failed download of a COMPLETE kernel ------------------------------------------
def test_a_complete_kernel_whose_download_fails_twice_is_retried_not_abandoned(rq, tmp_path, monkeypatch):  # noqa: F811
    _kaggle(rq, monkeypatch, {BASE: {"lr_S.json": "{}"}}, complete=(BASE,))
    inner = rq.subprocess.run
    fails = {"n": 0}

    def flaky(cmd, **kw):
        if cmd[2] == "output" and fails["n"] < 2:                # a 429 storm: nothing downloaded
            fails["n"] += 1
            return types.SimpleNamespace(stdout="", stderr="429 Client Error: Too Many Requests for url")
        return inner(cmd, **kw)
    monkeypatch.setattr(rq.subprocess, "run", flaky)
    reg = _main(rq, tmp_path, _spec(resumable=False))
    assert fails["n"] == 2
    assert "needs attention" not in reg and "ALL_OUTPUTS_DOWNLOADED" in reg


# ---- RS-5 for pilot pieces: the autoencoder is part of what a continuation must share --------------------
@pytest.fixture(scope="module")
def ae_root2(tmp_path_factory, pilot_root):  # noqa: F811
    """Another autoencoder under the same file name (the kit rebuilt with a re-collected pilot_ae)."""
    root = tmp_path_factory.mktemp("aeroot2")
    pieces.piece_ae(pilot_root, KEY, 0, {"d": {KEY: 4}}, root / "ae-job" / "out", {"ae_epochs": 2, "ae_width": 8})
    return root


def _other_ae(ae_root, ae_root2):  # noqa: F811
    from p3b.train.cells import _sha
    name = "ae_A_MOF-303_pilot_s0.pt"
    a, b = (_sha(str(r / "ae-job" / "out" / name)) for r in (ae_root, ae_root2))
    assert a != b
    return a


@pytest.mark.parametrize("piece", ["two", "lr", "curve"])
def test_rs5_a_pilot_piece_refuses_to_resume_under_another_autoencoder(pilot_root, ae_root, ae_root2, tmp_path,  # noqa: F811
                                                                       piece):
    sha = _other_ae(ae_root, ae_root2)

    def run(stage, out, **kw):
        if piece == "two":
            return pieces.piece_two("U", 0, pilot_root, stage, out, LQ, **kw)
        if piece == "lr":
            return pieces.piece_lr("U", pilot_root, stage, out, LQ, **kw)
        return pieces.piece_curve("U", pilot_root, stage, out, {**LQ, "curve_iterations": 4, "curve_every": 2}, **kw)
    stage = _stage(ae_root)
    p1 = run(stage, tmp_path / "s1", deadline=time.monotonic() - 1)
    assert os.path.basename(p1) == "RESUME.json"
    assert sha in json.load(open(p1))["autoencoders"].values()
    with pytest.raises(RuntimeError, match="autoencoder"):
        run({**stage, "ae_input_root": str(ae_root2)}, tmp_path / "s2", deadline=time.monotonic() - 1,
            resume_from=str(tmp_path / "s1"))
    res, _ = _sessions(lambda out, **kw: run(stage, out, **kw), tmp_path / "same")   # the same one continues
    assert os.path.basename(res) != "RESUME.json"


# ---- dependents and their producers: one jobs file, replayed across continuations ----------------------
def _train_jobs(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    return _load("train_jobs")


def test_evaluation_jobs_carry_the_jobs_they_depend_on(monkeypatch):
    from p3b.train.cells import cells
    tj = _train_jobs(monkeypatch)
    cs = [c for c in cells() if c.arm in ("U", "C+L", "FNO")][:6]
    spec = tj.evaluation_jobs(cs, kit_slug="o/kit")
    names = {j["name"] for j in spec["jobs"]}
    deps = {d for j in spec["jobs"] for d in j.get("depends_on", ())}
    assert deps and deps <= names
    assert any(j["system"] == "evaluate" for j in spec["jobs"])


def test_the_queue_refuses_a_dependency_outside_its_jobs_file(rq, tmp_path):  # noqa: F811
    spec = {"kit_slug": "o/kit", "nz": 0, "jobs": [
        {"name": "p3b-eval-x", "system": "evaluate", "set": "e", "commands": [["b.py"]], "expect": ["b.json"],
         "depends_on": ["p3b-u-x"]}]}
    with pytest.raises(ValueError, match="p3b-u-x"):
        rq.write_kernels(spec)


def test_the_replay_finds_a_finished_chain_s_producer_among_its_continuations(rq, tmp_path, monkeypatch):  # noqa: F811
    T, E = "p3b-u-x", "p3b-u-x-test"
    spec = {"kit_slug": "o/p3b-reference-kit", "nz": 0, "jobs": [
        {"name": T, "system": "train", "set": T, "threads": 4, "resumable": True,
         "commands": [["kaggle/train_chunk.py", "--cell", T, "--out", "OUT"]], "expect": [f"{T}.json"]},
        {"name": E, "system": "evaluate", "set": E, "threads": 4, "depends_on": [T],
         "commands": [["kaggle/train_chunk.py", "--evaluate", T, "test", "--out", "OUT"]], "expect": [f"{E}.json"]}]}
    for name, f, body in ((T, "RESUME.json", '{"stopped_at": 1}'), (f"{T}-r1", "RESUME.json", '{"stopped_at": 2}'),
                          (f"{T}-r2", f"{T}.json", "{}")):                   # an earlier run finished the chain
        d = tmp_path / "data" / "reference" / "train" / T / name / "out"
        d.mkdir(parents=True)
        (d / f).write_text(body)
    with open(rq.REG, "w") as f:
        f.writelines(f"2026-11-16 00:00:0{i} {n} pushed\n" for i, n in enumerate((T, f"{T}-r1", f"{T}-r2")))
    seen = _kaggle(rq, monkeypatch, {E: {f"{E}.json": "{}"}})
    inner, polled = rq.subprocess.run, []

    def spy(cmd, **kw):
        polled.append(cmd[3] if len(cmd) > 3 else "")
        return inner(cmd, **kw)
    monkeypatch.setattr(rq.subprocess, "run", spy)
    reg = _main(rq, tmp_path, spec)
    assert seen["pushes"] == [E] and seen["sources"][E] == [f"{OWNER}/{T}-r2"]
    assert not any(r.split("/")[-1].startswith(T) and not r.endswith(E) for r in polled)   # no re-polling
    assert "ALL_OUTPUTS_DOWNLOADED" in reg
