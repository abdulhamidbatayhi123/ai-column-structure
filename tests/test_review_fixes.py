"""Final-review fix pass (plan 1): C1, C2, I1, I2, I3 and two minors promoted to Important."""
import glob
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from p3b import _paper1, gates
from p3b import generate as gen
from p3b import store
from kaggle.make_kit import make_kit

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FAST_A = dict(RH_ads=0.40, T_ads=300.0, T_des=348.0, v=0.30, alpha=1.2, beta=0.75, index=0, n_cycles=2)
POINT_B = dict(T_hot=363.0, T_cold=293.0, v=3.0e-3, alpha=0.9, beta=1.0, index=0, n_cycles=2)


# ---- C1 and I1: the kit runs on its own, with provenance ------------------------------------

@pytest.fixture(scope="module")
def kit(tmp_path_factory):
    return make_kit(tmp_path_factory.mktemp("k") / "kit", "abc1234", paper1_src=_paper1.PAPER1_DIR)


def test_chunk_runs_from_a_built_kit_outside_the_repo(kit, tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "P3B_PAPER1_DIR"}
    r = subprocess.run([sys.executable, str(kit / "kaggle" / "generate_chunk.py"), "--system", "A_primary",
                        "--set", "long_validation", "--chunk", "0", "--of", "2", "--nz", "64", "--nc", "32",
                        "--cycles", "1", "--out", str(tmp_path / "out")], capture_output=True, text=True,
                       cwd=str(tmp_path), env=env)
    assert r.returncode == 0, r.stderr[-3000:]
    files = sorted((tmp_path / "out").glob("*.npz"))
    assert len(files) == 2
    assert gates.load_meta(files[0])["commit"] == "kit:abc1234"
    rep = json.loads((tmp_path / "out" / "chunk_0_report.json").read_text())
    assert rep["code_commit"] == "kit:abc1234"


def test_meta_commit_is_not_overridden(tmp_path):
    path = tmp_path / "a.npz"
    res = gen.generate_trajectory("A", dict(FAST_A, n_cycles=1, v=0.10), "MOF-303", 32, 16, path=path,
                                  extra_meta={"commit": "given"})
    assert res.ok and gates.load_meta(path)["commit"] == "given"


# ---- C2: exact entropy ports; the gate uses tau_ref ---------------------------------------

@pytest.fixture(scope="module")
def fast_a(tmp_path_factory):
    d = tmp_path_factory.mktemp("fa")
    out = {}
    for nc in (64, 16):
        p = d / f"fast_{nc}.npz"
        res = gen.generate_trajectory("A", FAST_A, "MOF-303", 64, nc, path=p)
        assert res.ok, res.message
        out[nc] = p
    return out


def test_entropy_ports_are_stored(fast_a):
    d = np.load(fast_a[64])
    assert d["ports_cum"].shape[-1] == 7 and d["ports_end"].shape == (7,)


def test_entropy_gate_passes_exact_solves_at_the_fast_corner(fast_a):
    """The reviewer's counter-example: margin -0.017 by trapezoid quadrature, at any coarsening."""
    for nc, path in fast_a.items():
        g = gates.entropy_gate(path, gates.model_from_meta(gates.load_meta(path)))
        assert g["passed"], (nc, g)
        assert np.isfinite(g["min_rate"]) and g["tau_ref"] > 0


def test_no_coarsening_production_matches_the_fine_production(fast_a):
    g = gates.entropy_gate(fast_a[64], gates.model_from_meta(gates.load_meta(fast_a[64])))
    assert g["min_rate"] >= -g["tau_ref"]


def test_system_b_entropy_ports(tmp_path):
    p = tmp_path / "b.npz"
    assert gen.generate_trajectory("B", POINT_B, None, 64, 16, path=p).ok
    d = np.load(p)
    assert d["ports_cum"].shape[-1] == 6
    assert gates.entropy_gate(p, gates.model_from_meta(gates.load_meta(p)))["passed"]


# ---- I2: fine-level gates are computed at every snapshot and stored ----------------------------

def test_fine_rate_identities_and_production_are_stored_and_pass(fast_a):
    d = np.load(fast_a[16])
    assert d["rate_residual"].shape == (2, 200, 3)            # water, energy, entropy
    assert d["fine_min_production"].shape == (2, 200)
    assert gates.rate_identity_max(fast_a[16]) <= 1e-12
    assert gates.fine_production_min(fast_a[16]) >= -1e-12
    assert gates.closure_ok(fast_a[16])


# ---- I3: failures never crash a batch or leave a partial file ------------------------------

def test_setup_failure_is_returned_as_data(tmp_path):
    res = gen.generate_trajectory("A", dict(FAST_A, v=0.0), "MOF-303", 32, 16, path=tmp_path / "x.npz")
    assert res.ok is False and res.message and not (tmp_path / "x.npz").exists()


def test_mid_trajectory_failure_writes_no_file(tmp_path, monkeypatch):
    orig = store.Coarsener.add_cycle

    def boom(self, k, *a, **kw):
        if k == 1:
            raise MemoryError("forced in cycle 2")
        return orig(self, k, *a, **kw)
    monkeypatch.setattr(store.Coarsener, "add_cycle", boom)
    res = gen.generate_trajectory("A", dict(FAST_A, v=0.10), "MOF-303", 32, 16, path=tmp_path / "y.npz")
    assert res.ok is False and res.failed_cycle == 1 and "forced" in res.message
    assert not glob.glob(str(tmp_path / "y*"))


def test_chunk_report_survives_an_exception(tmp_path, monkeypatch):
    import kaggle.generate_chunk as gc

    def fake(system, point, material, N_z, N_c=128, path=None, n_snap=200, extra_meta=None):
        if point["index"] == 1:
            raise RuntimeError("disk full")
        return gen.TrajectoryResult(True, None, "ok", 0.1, [0.0])
    monkeypatch.setattr(gen, "generate_trajectory", fake)
    gc.main(["--system", "A_primary", "--set", "long_validation", "--chunk", "0", "--of", "2", "--nz", "64",
             "--nc", "32", "--cycles", "1", "--out", str(tmp_path)])
    rep = json.loads((tmp_path / "chunk_0_report.json").read_text())
    assert [t["ok"] for t in rep["trajectories"]] == [True, False]
    assert "disk full" in rep["trajectories"][1]["message"] and rep["ok"] is False


# ---- promoted minors ------------------------------------------------------------------

def test_subchunks_never_exceed_the_set(tmp_path, monkeypatch):
    from kaggle import run_queue as q
    monkeypatch.setattr(q, "BUILD", str(tmp_path))
    ks = q.write_kernels({"kit_slug": "o/k", "nz": 1024, "jobs": [{"system": "B", "set": "pilot", "of": 3}]})
    covered = []
    for k in ks:
        run_py = open(os.path.join(k["dir"], "run.py")).read()
        args = eval(run_py.split("args = ", 1)[1].split("\n", 1)[0])
        for a in args:
            idx = [int(x) for x in a[a.index("--indices") + 1].split(",")]
            assert idx
            covered += idx
        assert k["expected_reports"] == len(args) <= 4
    assert sorted(covered) == list(range(8))


def test_gate_rejects_smoke_outputs(tmp_path):
    p = tmp_path / "s.npz"
    gen.generate_trajectory("A", dict(FAST_A, n_cycles=1, v=0.10), "MOF-303", 32, 16, path=p,
                            extra_meta={"cycles_override": 1})
    ok, why = gates.production_ready(p)
    assert ok is False and "override" in why
    q = tmp_path / "t.npz"
    gen.generate_trajectory("A", dict(FAST_A, n_cycles=1, v=0.10), "MOF-303", 32, 16, path=q)
    assert gates.production_ready(q) == (True, "")
