"""The Kaggle queue continues a resumable job across sessions: a kernel that ends with RESUME.json
instead of its expected output gets a continuation kernel (<name>-r<n>) with the earlier output
attached and --resume-from /kaggle/input, and jobs that depend on it wait for the logical job."""
import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kaggle"))


@pytest.fixture
def rq(tmp_path, monkeypatch):
    import run_queue
    monkeypatch.setattr(run_queue, "BUILD", str(tmp_path / "build"))
    monkeypatch.setattr(run_queue, "REG", str(tmp_path / "registry.txt"))
    monkeypatch.setattr(run_queue, "ROOT", str(tmp_path))
    monkeypatch.setattr(run_queue.time, "sleep", lambda s: None)
    monkeypatch.setenv("KAGGLE_CLI", "kaggle")
    return run_queue


SPEC = {"kit_slug": "o/p3b-reference-kit", "nz": 0, "jobs": [
    {"name": "p3b-pilot-curve-cpl-x", "system": "pilot", "set": "curve-CpL", "threads": 1, "resumable": True,
     "commands": [["kaggle/pilot_chunk.py", "--piece", "curve-CpL", "--out", "OUT"]], "expect": ["curve_CpL.json"]},
    {"name": "p3b-pilot-after-x", "system": "pilot", "set": "after", "threads": 1, "depends_on": ["p3b-pilot-curve-cpl-x"],
     "commands": [["kaggle/pilot_chunk.py", "--piece", "wall", "--out", "OUT"]], "expect": ["wall_per_iter.json"]}]}


def test_continuation_kernel_resumes_from_the_earlier_output(rq):
    k = next(k for k in rq.write_kernels(SPEC) if k["name"] == "p3b-pilot-curve-cpl-x")
    c = rq.continuation(k, SPEC["kit_slug"])
    assert c["name"] == "p3b-pilot-curve-cpl-x-r1" and c["base"] == "p3b-pilot-curve-cpl-x" and c["expect"] == k["expect"]
    meta = json.load(open(os.path.join(c["dir"], "kernel-metadata.json")))
    assert meta["kernel_sources"] == ["abdulhamidbatayhi/p3b-pilot-curve-cpl-x"]
    script = open(os.path.join(c["dir"], "run.py")).read()
    assert "'--resume-from', '/kaggle/input'" in script
    c2 = rq.continuation(c, SPEC["kit_slug"])
    assert c2["name"] == "p3b-pilot-curve-cpl-x-r2"
    assert json.load(open(os.path.join(c2["dir"], "kernel-metadata.json")))["kernel_sources"] == [
        "abdulhamidbatayhi/p3b-pilot-curve-cpl-x-r1"]


def test_the_queue_pushes_continuations_until_the_expected_output_arrives(rq, tmp_path, monkeypatch):
    # Kaggle as a script: the base kernel and r1 end with RESUME.json, r2 with the curve; the
    # dependent job is pushed only after the curve is downloaded
    state = {"p3b-pilot-curve-cpl-x": "COMPLETE"}
    outputs = {"p3b-pilot-curve-cpl-x": "RESUME.json", "p3b-pilot-curve-cpl-x-r1": "RESUME.json",
               "p3b-pilot-curve-cpl-x-r2": "curve_CpL.json", "p3b-pilot-after-x": "wall_per_iter.json"}
    pushes = []

    def fake_run(cmd, capture_output, text, timeout):
        verb, ref = cmd[2], cmd[3] if len(cmd) > 3 else ""
        if verb == "status":
            name = ref.split("/")[1]
            s = state.get(name)
            out = f'{ref} has status "KernelWorkerStatus.{s}"' if s else "404 Not Found"
        elif verb == "output":
            name = ref.split("/")[1]
            d = cmd[cmd.index("-p") + 1]
            os.makedirs(os.path.join(d, "out"), exist_ok=True)
            open(os.path.join(d, "out", outputs[name]), "w").write("{}")
            out = "downloaded"
        elif verb == "push":
            name = os.path.basename(cmd[cmd.index("-p") + 1])
            pushes.append(name)
            state[name] = "COMPLETE"
            out = "Kernel version 1 successfully pushed"
        return types.SimpleNamespace(stdout=out, stderr="")
    monkeypatch.setattr(rq.subprocess, "run", fake_run)
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps(SPEC))
    rq.main(str(jobs))
    assert pushes == ["p3b-pilot-curve-cpl-x-r1", "p3b-pilot-curve-cpl-x-r2", "p3b-pilot-after-x"]
    assert "ALL_OUTPUTS_DOWNLOADED" in open(rq.REG).read()


def test_the_cplusl_curve_and_the_study_cells_are_resumable_jobs():
    import importlib
    pj = importlib.import_module("pilot_jobs")
    phase2 = {j["name"]: j for j in pj.phase_jobs(2)["jobs"]}
    curve = [j for n, j in phase2.items() if "curve-cpl" in n]
    assert curve and all(j.get("resumable") for j in curve)
    assert not any(j.get("resumable") for n, j in phase2.items() if "curve-cpl" not in n)
    tj = importlib.import_module("train_jobs")
    from p3b.train.cells import cells
    jobs = tj.build_jobs(cells())["jobs"]
    cell_jobs = [j for j in jobs if j["commands"][0][1] == "--cell"]
    assert cell_jobs and all(j.get("resumable") for j in cell_jobs)
    assert not any(j.get("resumable") for j in jobs if j["commands"][0][1] == "--ae")
