"""Four one-thread jobs per Kaggle session (PREREG Section 11, freeze decision 2 option (i)).

A jobs file with "pack": 4 has its command jobs packed, four to a kernel, by dependency level: each member
keeps its own commands, writing into OUT/<member>, and runs as its own process; the kernel expects the
union of the members' files. A packed kernel that ends with some members unfinished continues only those
that left a RESUME.json, from it; dependants wait for (and mount the kernels of) the member jobs they need."""
import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kaggle"))


@pytest.fixture
def rq(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_queue_pack", os.path.join(os.path.dirname(__file__), "..", "kaggle", "run_queue.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "BUILD", str(tmp_path / "build"))
    monkeypatch.setattr(mod, "REG", str(tmp_path / "registry.txt"))
    monkeypatch.setattr(mod, "ROOT", str(tmp_path))
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    monkeypatch.setenv("KAGGLE_CLI", "kaggle")
    return mod


def _ae(i):
    return {"name": f"p3b-ae-{i}", "system": "train", "set": f"p3b-ae-{i}", "threads": 1,
            "commands": [["kaggle/train_chunk.py", "--ae", f"p3b-ae-{i}", "--out", "OUT"]], "expect": [f"ae_{i}.pt"]}


def _cell(i):
    return {"name": f"p3b-u-{i}", "system": "train", "set": f"p3b-u-{i}", "threads": 1, "resumable": True,
            "depends_on": [f"p3b-ae-{i}"],
            "commands": [["kaggle/train_chunk.py", "--cell", f"p3b-u-{i}", "--out", "OUT"]], "expect": [f"p3b-u-{i}.json"]}


SPEC = {"kit_slug": "o/p3b-reference-kit", "nz": 0, "pack": 4, "jobs": [_ae(i) for i in range(5)] + [_cell(i) for i in range(5)]}


def test_jobs_are_packed_four_to_a_kernel_by_dependency_level(rq):
    packed = rq.pack_jobs(SPEC["jobs"], 4)
    assert [[m["name"] for m in p["members"]] for p in packed] == [
        [f"p3b-ae-{i}" for i in range(4)], ["p3b-ae-4"], [f"p3b-u-{i}" for i in range(4)], ["p3b-u-4"]]
    p = packed[2]
    assert p["commands"][1] == ["kaggle/train_chunk.py", "--cell", "p3b-u-1", "--out", "OUT/p3b-u-1"]
    assert p["expect"] == [f"p3b-u-{i}/p3b-u-{i}.json" for i in range(4)]
    assert p["depends_on"] == [f"p3b-ae-{i}" for i in range(4)] and p["resumable"] and p["threads"] == 1
    assert len({q["name"] for q in packed}) == 4 and all(len(q["name"]) <= 46 for q in packed)
    assert not packed[0]["resumable"]


def test_the_queue_runs_packs_continues_only_unfinished_members_and_mounts_member_producers(rq, tmp_path, monkeypatch):
    packed = rq.pack_jobs(SPEC["jobs"], 4)
    ae4, ae1, cell4, cell1 = (p["name"] for p in packed)
    files = {
        ae4: [f"p3b-ae-{i}/ae_{i}.pt" for i in range(4)],
        ae1: ["p3b-ae-4/ae_4.pt"],
        cell4: ["p3b-u-0/p3b-u-0.json", "p3b-u-1/p3b-u-1.json", "p3b-u-2/RESUME.json", "p3b-u-3/RESUME.json"],
        cell4 + "-r1": ["p3b-u-2/p3b-u-2.json", "p3b-u-3/p3b-u-3.json"],
        cell1: ["p3b-u-4/p3b-u-4.json"],
    }
    state, pushes = {}, []

    def fake_run(cmd, capture_output, text, timeout):
        verb, ref = cmd[2], cmd[3] if len(cmd) > 3 else ""
        if verb == "status":
            s = state.get(ref.split("/")[1])
            out = f'{ref} has status "KernelWorkerStatus.{s}"' if s else "404 Not Found"
        elif verb == "output":
            d = cmd[cmd.index("-p") + 1]
            for rel in files[ref.split("/")[1]]:
                p = os.path.join(d, "out", rel)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                open(p, "w").write(json.dumps({"progress": [5]}) if rel.endswith("RESUME.json") else "{}")
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
    assert pushes[:2] == [ae4, ae1] and set(pushes[2:4]) == {cell4, cell1} and pushes[4:] == [cell4 + "-r1"]
    assert "ALL_OUTPUTS_DOWNLOADED" in open(rq.REG).read()
    script = open(os.path.join(rq.BUILD, cell4 + "-r1", "run.py")).read()
    assert "p3b-u-2" in script and "p3b-u-3" in script and "p3b-u-0" not in script and "p3b-u-1" not in script
    assert script.count("'--resume-from', '/kaggle/input'") == 2
    meta = json.load(open(os.path.join(rq.BUILD, cell4 + "-r1", "kernel-metadata.json")))
    assert set(meta["kernel_sources"]) == {f"abdulhamidbatayhi/{ae4}", f"abdulhamidbatayhi/{cell4}"}
    meta1 = json.load(open(os.path.join(rq.BUILD, cell1, "kernel-metadata.json")))
    assert meta1["kernel_sources"] == [f"abdulhamidbatayhi/{ae1}"]


def test_a_member_that_ends_without_output_or_resume_needs_attention_and_the_rest_go_on(rq, tmp_path, monkeypatch):
    spec = {"kit_slug": "o/k", "nz": 0, "pack": 4, "jobs": [_ae(0), _ae(1)]}
    (p,) = rq.pack_jobs(spec["jobs"], 4)
    state, pushes = {}, []

    def fake_run(cmd, capture_output, text, timeout):
        verb, ref = cmd[2], cmd[3] if len(cmd) > 3 else ""
        if verb == "status":
            s = state.get(ref.split("/")[1])
            out = f'{ref} has status "KernelWorkerStatus.{s}"' if s else "404 Not Found"
        elif verb == "output":
            d = cmd[cmd.index("-p") + 1]
            os.makedirs(os.path.join(d, "out", "p3b-ae-0"), exist_ok=True)
            open(os.path.join(d, "out", "p3b-ae-0", "ae_0.pt"), "w").write("x")
            out = "downloaded"
        else:
            name = os.path.basename(cmd[cmd.index("-p") + 1])
            pushes.append(name)
            state[name] = "COMPLETE"
            out = "Kernel version 1 successfully pushed"
        return types.SimpleNamespace(stdout=out, stderr="")
    monkeypatch.setattr(rq.subprocess, "run", fake_run)
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps(spec))
    rq.main(str(jobs))
    reg = open(rq.REG).read()
    assert pushes == [p["name"]] and "p3b-ae-1" in reg and "needs attention" in reg and "QUEUE_STOPPED" in reg
