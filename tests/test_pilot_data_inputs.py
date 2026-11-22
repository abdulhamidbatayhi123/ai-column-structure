"""The pilot data reach the pilot kernels as the outputs of the kernels that generated them (2026-11-22), not
inside the kit: the kit stays small (code and stage only), so a phase's kit rebuild no longer uploads 2 GB, and
the data files are exactly the generated ones.
- run_queue: a jobs file's "kernel_inputs" (finished kernels) are mounted by every kernel and continuation;
- pilot_chunk --data-from DIR lays the mounted npz out by key and split (symlinks), as scripts/pilot_collect.py data;
- pilot_jobs: every phase mounts the nine R3 data kernels and passes --data-from /kaggle/input;
- make_kit no longer copies data/pilot."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kaggle"))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def rq(tmp_path, monkeypatch):
    import run_queue
    monkeypatch.setattr(run_queue, "BUILD", str(tmp_path / "build"))
    monkeypatch.setattr(run_queue, "REG", str(tmp_path / "registry.txt"))
    monkeypatch.setattr(run_queue, "ROOT", str(tmp_path))
    monkeypatch.setattr(run_queue.time, "sleep", lambda s: None)
    return run_queue


SPEC = {"kit_slug": "o/p3b-reference-kit", "nz": 0, "kernel_inputs": ["p3b-ref-b-pilot-0-of-1-r3"], "jobs": [
    {"name": "p3b-pilot-curve-cpl-x", "system": "pilot", "set": "curve-CpL", "threads": 1, "resumable": True,
     "commands": [["kaggle/pilot_chunk.py", "--piece", "curve-CpL", "--out", "OUT"]], "expect": ["curve_CpL.json"]}]}


def _sources(k):
    return json.load(open(os.path.join(k["dir"], "kernel-metadata.json")))["kernel_sources"]


def test_every_kernel_and_continuation_mounts_the_jobs_file_inputs(rq):
    k, = rq.write_kernels(SPEC)
    assert "abdulhamidbatayhi/p3b-ref-b-pilot-0-of-1-r3" in _sources(k)
    c = rq.continuation(k, SPEC["kit_slug"])
    assert set(_sources(c)) == {"abdulhamidbatayhi/p3b-ref-b-pilot-0-of-1-r3", "abdulhamidbatayhi/p3b-pilot-curve-cpl-x"}
    rq.point_sources(k, SPEC["kit_slug"], {})
    assert "abdulhamidbatayhi/p3b-ref-b-pilot-0-of-1-r3" in _sources(k)


def test_pilot_jobs_mount_the_nine_r3_data_kernels_and_read_them():
    import pilot_jobs
    names = pilot_jobs.data_kernels()
    with open(os.path.join(ROOT, "kaggle", "jobs_pilot_r3.json")) as f:
        spec = json.load(f)
    assert len(names) == sum(j["of"] for j in spec["jobs"]) == 9
    assert "p3b-ref-a-primary-pilot-train-0-of-2-r3" in names and "p3b-ref-b-pilot-0-of-1-r3" in names
    for phase in range(5):
        st = {"_revision": pilot_jobs.PILOT_REVISION} if phase else None
        js = pilot_jobs.phase_jobs(phase, st)
        assert js["kernel_inputs"] == names
        for j in js["jobs"]:
            assert j["commands"][0][-2:] == ["--data-from", "/kaggle/input"], j["name"]


def _npz(path, design, index):
    import numpy as np
    meta = json.dumps({"design_set": design, "point": {"index": index}})
    np.savez(path, meta=np.array(meta), X=np.zeros(2))


def test_pilot_chunk_lays_out_mounted_data_by_key_and_split(tmp_path, monkeypatch):
    import pilot_chunk
    from p3b.gates import load_meta
    mnt = tmp_path / "input" / "p3b-ref-b-pilot-0-of-1-r3"
    mnt.mkdir(parents=True)
    _npz(mnt / "B_pilot_0000.npz", "B_pilot", 0)
    _npz(mnt / "B_pilot_0007.npz", "B_pilot", 7)
    assert load_meta(mnt / "B_pilot_0000.npz")["design_set"] == "B_pilot"
    root = pilot_chunk.stage_mounted(str(tmp_path / "input"), str(tmp_path / "laid"))
    assert os.path.exists(os.path.join(root, "B", "train", "B_pilot_0000.npz"))
    assert os.path.exists(os.path.join(root, "B", "validation", "B_pilot_0007.npz"))


def test_the_kit_no_longer_carries_the_pilot_data():
    src = open(os.path.join(ROOT, "kaggle", "make_kit.py"), encoding="utf-8").read()
    assert "data/pilot" not in src.split("def make_kit", 1)[1].split("p1 = out")[0]
