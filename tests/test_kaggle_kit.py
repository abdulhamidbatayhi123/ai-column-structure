"""Kaggle kit and resumable queue (plan 1, Task 9). No network: Paper 1's files come from the sibling checkout."""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from p3b import _paper1
from kaggle.make_kit import chunk_indices, make_kit
from kaggle import run_queue as q

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_chunks_partition():
    ch = chunk_indices(48, 5)
    flat = [i for c in ch for i in c]
    assert sorted(flat) == list(range(48)) and len(flat) == len(set(flat))
    sizes = [len(c) for c in ch]
    assert max(sizes) - min(sizes) <= 1 and len(ch) == 5
    with pytest.raises(ValueError):
        chunk_indices(3, 5)


def test_kit_manifest(tmp_path):
    kit = make_kit(tmp_path / "kit", "abc1234", paper1_src=_paper1.PAPER1_DIR)
    assert (kit / "kit_commit.txt").read_text().strip() == "abc1234"
    manifest = json.loads((kit / "MANIFEST.json").read_text())
    for need in ("p3b/column.py", "p3b/generate.py", "kaggle/generate_chunk.py", "scripts/time_reference.py",
                 "verify/materials/fit_isotherms.json", "verify/v1_large_grid.py", "data/design/A_primary_train.json",
                 "paper1/solver_fd.py", "paper1/isotherm.py", "paper1/gen_parametric_dataset.py"):
        assert need in manifest["files"], need
    for rel, sha in manifest["files"].items():
        assert hashlib.sha256((kit / rel).read_bytes()).hexdigest() == sha, rel
    assert manifest["paper1_source"]
    assert not any("__pycache__" in rel or rel.endswith(".pyc") for rel in manifest["files"])


def test_generate_chunk_small(tmp_path):
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, "kaggle/generate_chunk.py", "--system", "A_primary", "--set", "long_validation",
                        "--material", "MOF-303", "--chunk", "0", "--of", "2", "--nz", "64", "--nc", "32",
                        "--cycles", "1", "--out", str(out)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-2000:]
    npz = sorted(out.glob("*.npz"))
    assert [p.name for p in npz] == ["A_primary_long_validation_0000_smoke.npz",
                                     "A_primary_long_validation_0001_smoke.npz"]
    rep = json.loads((out / "chunk_0_report.json").read_text())
    assert rep["ok"] is True and len(rep["trajectories"]) == 2 and rep["cycles_override"] == 1
    assert all(t["ok"] for t in rep["trajectories"])
    d = np.load(npz[0])
    assert d["channels"].shape == (1, 200, 32, 3)


def test_material_must_match_system(tmp_path):
    r = subprocess.run([sys.executable, "kaggle/generate_chunk.py", "--system", "A_replication", "--set", "pilot",
                        "--material", "MOF-303", "--chunk", "0", "--of", "8", "--nz", "64", "--nc", "32",
                        "--out", str(tmp_path)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and "Al-fumarate" in r.stderr


@pytest.mark.parametrize("text,expected", [
    ('status "KernelWorkerStatus.COMPLETE"', "COMPLETE"),
    ('status "KernelWorkerStatus.RUNNING"', "RUNNING"),
    ("404 Client Error: Not Found for url", "ABSENT"),
    ("429 Client Error: Too Many Requests", "UNKNOWN"),
    ("", "UNKNOWN"),
])
def test_status_parsing_never_guesses_absent(text, expected):
    assert q.parse_status(text) == expected


def test_command_job_kernel(tmp_path, monkeypatch):
    monkeypatch.setattr(q, "BUILD", str(tmp_path))
    spec = {"kit_slug": "o/kit", "nz": 0, "jobs": [
        {"name": "p3b-timing-a", "commands": [["scripts/time_reference.py", "--system", "A", "--out", "OUT/a.json"],
                                              ["verify/v1_large_grid.py", "--n", "2048", "--out", "OUT/v1.json"]],
         "expect": ["a.json", "v1.json"]}]}
    (k,) = q.write_kernels(spec)
    assert k["name"] == "p3b-timing-a" and k["expect"] == ["a.json", "v1.json"]
    run_py = (tmp_path / "p3b-timing-a" / "run.py").read_text()
    assert "time_reference.py" in run_py and "v1_large_grid.py" in run_py and "/kaggle/working/out" in run_py
    meta = json.loads((tmp_path / "p3b-timing-a" / "kernel-metadata.json").read_text())
    assert meta["is_private"] is True and meta["enable_internet"] is False and meta["dataset_sources"] == ["o/kit"]
