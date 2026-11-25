"""Claude Science's T10 pre-freeze review (hub REVIEW_T10_postT8.md, 2026-11-24), the findings that change code:
- L4: at v = 0 the outflow end is z = L, as in the reference and the solvers (v >= 0);
- L3: a B file whose stored direction port disagrees with the inputs' flow direction is refused when loaded
  (R4 assumes every reversal falls on a snapshot);
- constructed cases asked for with the review: a mirrored column gives identical ports, and at zero flow the port
  terms do not depend on which end is taken;
- M2: the reference-generation kernels run the pinned environment too, when the jobs file carries one;
- M1: a kernel that ends with status ERROR (or is cancelled) no longer stalls the queue: what it wrote is
  collected (finished members, resumable sessions), the rest is flagged once, and the queue stops."""
import json
import os
import sys
import types

import numpy as np
import pytest
import torch

from p3b.generate import generate_trajectory

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kaggle"))
F64 = torch.float64


def test_l4_outflow_end_at_zero_flow_is_z_L():
    from p3b.latent.data import outflow_temperature
    out = torch.tensor([[300.0, 340.0]], dtype=F64)
    assert float(outflow_temperature(out, torch.tensor([[0.0, 320.0]], dtype=F64))) == 340.0
    assert float(outflow_temperature(out, torch.tensor([[-1e-3, 320.0]], dtype=F64))) == 300.0


@pytest.fixture(scope="module")
def b_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("t10b") / "b.npz"
    pt = dict(T_hot=353.15, T_cold=298.15, v=2.0e-3, alpha=0.7, beta=0.8, index=0, n_cycles=2)
    assert generate_trajectory("B", pt, None, N_z=32, N_c=8, path=p, n_snap=20).ok
    return p


def test_l3_a_direction_port_that_disagrees_with_the_inputs_is_refused(b_path, tmp_path):
    from p3b.latent.data import load_set
    load_set([b_path])                                            # consistent: loads
    d = dict(np.load(b_path))
    d["ports_cum"] = d["ports_cum"].copy()
    d["ports_cum"][0, 3, 3] *= -1.0                                # one snapshot whose direction does not match
    bad = tmp_path / "bad.npz"
    np.savez(bad, **d)
    with pytest.raises(ValueError, match="direction"):
        load_set([bad])


def _store_phys(b_path):
    from p3b.latent.data import load_set
    from p3b.pilot.runs import _physics
    ds = load_set([b_path])
    return ds, _physics(ds)


def test_a_mirrored_column_gives_identical_ports(b_path):
    ds, phys = _store_phys(b_path)
    nc = ds.n_cells
    X = ds.X[0, 0, 3]
    ch = X[:nc * 2].reshape(1, nc, 2)
    out = X[nc * 2:].reshape(1, 2)
    for v, T_in in ((2e-3, 353.15), (-2e-3, 298.15)):
        a = phys.ports(ch, out, torch.tensor([[v, T_in]], dtype=F64))
        b = phys.ports(ch.flip(1), out.flip(-1), torch.tensor([[-v, T_in]], dtype=F64))
        for x, y in zip(a, b):
            assert torch.allclose(x, y, rtol=1e-12, atol=0)


def test_at_zero_flow_the_ports_do_not_depend_on_the_end(b_path):
    ds, phys = _store_phys(b_path)
    nc = ds.n_cells
    ch = ds.X[0, 0, 3, :nc * 2].reshape(1, nc, 2)
    inp = torch.tensor([[0.0, 320.0]], dtype=F64)
    a = phys.ports(ch, torch.tensor([[300.0, 340.0]], dtype=F64), inp)
    b = phys.ports(ch, torch.tensor([[340.0, 300.0]], dtype=F64), inp)
    for x, y in zip(a, b):
        assert torch.allclose(x, y, rtol=1e-12, atol=0)


# ---- the queue --------------------------------------------------------------------------------------------
@pytest.fixture
def rq(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_queue_t10", os.path.join(os.path.dirname(__file__), "..", "kaggle", "run_queue.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "BUILD", str(tmp_path / "build"))
    monkeypatch.setattr(mod, "REG", str(tmp_path / "registry.txt"))
    monkeypatch.setattr(mod, "ROOT", str(tmp_path))
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    monkeypatch.setenv("KAGGLE_CLI", "kaggle")
    return mod


def test_m2_reference_kernels_run_the_pinned_environment(rq, tmp_path):
    os.makedirs(tmp_path / "data" / "design")
    (tmp_path / "data" / "design" / "B_pilot.json").write_text(json.dumps({"points": [{}, {}]}))
    spec = {"kit_slug": "o/p3b-reference-kit", "nz": 0, "env": {"kernel": "p3b-env-x", "pins": {"torch": "2.12.0+cpu"}},
            "jobs": [{"system": "B", "set": "pilot", "of": 1}]}
    k, = rq.write_kernels(spec)
    run = open(os.path.join(k["dir"], "run.py")).read()
    meta = json.load(open(os.path.join(k["dir"], "kernel-metadata.json")))
    assert "envcheck" in run and "2.12.0+cpu" in run and run.index("envcheck") < run.index("subprocess.Popen")
    assert "abdulhamidbatayhi/p3b-env-x" in meta["kernel_sources"]
    compile(run, "run.py", "exec")


def _ae(i, resumable=False):
    return {"name": f"p3b-ae-{i}", "system": "train", "set": f"p3b-ae-{i}", "threads": 1, "resumable": resumable,
            "commands": [["kaggle/train_chunk.py", "--ae", f"p3b-ae-{i}", "--out", "OUT"]], "expect": [f"ae_{i}.pt"]}


def _fake(rq, monkeypatch, state_after_push, write_output, limit=60):
    """subprocess.run stand-in: pushes succeed and the kernel then reports state_after_push(name); outputs are
    written by write_output(name, dir). More than `limit` status polls means the queue would stall: fail."""
    state, pushes, polls = {}, [], [0]

    def fake_run(cmd, capture_output, text, timeout):
        verb, ref = cmd[2], cmd[3] if len(cmd) > 3 else ""
        if verb == "status":
            polls[0] += 1
            if polls[0] > limit:
                raise AssertionError("the queue keeps polling a kernel that can never finish")
            s = state.get(ref.split("/")[1])
            out = f'{ref} has status "KernelWorkerStatus.{s}"' if s else "404 Not Found"
        elif verb == "output":
            write_output(ref.split("/")[1], cmd[cmd.index("-p") + 1])
            out = "Output file downloaded"
        else:
            name = os.path.basename(cmd[cmd.index("-p") + 1])
            pushes.append(name)
            state[name] = state_after_push(name)
            out = "Kernel version 1 successfully pushed"
        return types.SimpleNamespace(stdout=out, stderr="")
    monkeypatch.setattr(rq.subprocess, "run", fake_run)
    return pushes


def test_m1_a_crashed_member_of_a_pack_is_flagged_and_the_finished_one_collected(rq, tmp_path, monkeypatch):
    spec = {"kit_slug": "o/k", "nz": 0, "pack": 4, "jobs": [_ae(0), _ae(1)]}

    def write(name, d):
        os.makedirs(os.path.join(d, "out", "p3b-ae-0"), exist_ok=True)
        open(os.path.join(d, "out", "p3b-ae-0", "ae_0.pt"), "w").write("x")      # member 1 crashed: nothing
    _fake(rq, monkeypatch, lambda n: "ERROR", write)
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps(spec))
    rq.main(str(jobs))
    reg = open(rq.REG).read()
    assert "p3b-ae-1" in reg and "needs attention" in reg and "QUEUE_STOPPED" in reg
    assert os.path.exists(tmp_path / "data" / "reference")                       # the output was downloaded


def test_m1_a_failed_single_kernel_with_nothing_stops_the_queue(rq, tmp_path, monkeypatch):
    spec = {"kit_slug": "o/k", "nz": 0, "jobs": [_ae(0)]}
    _fake(rq, monkeypatch, lambda n: "ERROR", lambda name, d: os.makedirs(d, exist_ok=True))
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps(spec))
    rq.main(str(jobs))
    reg = open(rq.REG).read()
    assert "status ERROR" in reg and "QUEUE_STOPPED" in reg


def test_m1_a_failed_session_that_left_resume_continues(rq, tmp_path, monkeypatch):
    spec = {"kit_slug": "o/k", "nz": 0, "jobs": [_ae(0, resumable=True)]}

    def write(name, d):
        os.makedirs(os.path.join(d, "out"), exist_ok=True)
        if name.endswith("-r1"):
            open(os.path.join(d, "out", "ae_0.pt"), "w").write("x")
        else:
            open(os.path.join(d, "out", "RESUME.json"), "w").write(json.dumps({"progress": [1]}))
    pushes = _fake(rq, monkeypatch, lambda n: "COMPLETE" if n.endswith("-r1") else "ERROR", write)
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps(spec))
    rq.main(str(jobs))
    assert pushes == ["p3b-ae-0", "p3b-ae-0-r1"]
    assert "ALL_OUTPUTS_DOWNLOADED" in open(rq.REG).read()
