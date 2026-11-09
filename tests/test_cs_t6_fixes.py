"""Fixes for the open findings of Claude Science's code review T6 (I1, I2, I3, I6, I7).

I1: the column has no silent default isotherm.
I2: Paper 1's files are checked against release v2.0.1 by hash, and the hashes are recorded.
I3: the committed design files reproduce from their recorded seeds, and record the library versions.
I6: the numerical and physical constants a trajectory was made with are in its meta, and the gates
    refuse to rebuild the model from constants that differ.
I7: the cumulative wall heat per block is stored in float64.
"""
import glob
import json
import os
import shutil

import numpy as np
import pytest

from p3b import _paper1, gates, materials
from p3b.column import Column
from p3b.design import make_set
from p3b.generate import generate_trajectory

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
POINT_A = dict(RH_ads=0.40, T_ads=303.15, T_des=353.15, v=0.10, alpha=0.8, beta=0.6, index=0, n_cycles=1)
POINT_B = dict(T_hot=353.15, T_cold=298.15, v=2.0e-3, alpha=0.7, beta=0.8, index=0, n_cycles=1)


@pytest.fixture(scope="module")
def npz_a(tmp_path_factory):
    path = tmp_path_factory.mktemp("t6") / "A.npz"
    res = generate_trajectory("A", POINT_A, "MOF-303", N_z=32, N_c=8, path=path)
    assert res.ok, res.message
    return path


@pytest.fixture(scope="module")
def npz_b(tmp_path_factory):
    path = tmp_path_factory.mktemp("t6") / "B.npz"
    res = generate_trajectory("B", POINT_B, None, N_z=32, N_c=8, path=path)
    assert res.ok, res.message
    return path


def test_column_has_no_default_isotherm():
    phys = materials.bed_physics(0.1, 303.15)
    with pytest.raises(TypeError):
        Column(phys, 8, 33.6)


def test_paper1_files_match_release_hashes():
    assert set(_paper1.PAPER1_SHA256) == {"isotherm.py", "solver_fd.py", "gen_parametric_dataset.py"}
    assert _paper1.paper1_hashes(_paper1.PAPER1_DIR) == _paper1.PAPER1_SHA256


def test_paper1_check_rejects_a_changed_file(tmp_path):
    for name in _paper1.PAPER1_SHA256:
        shutil.copy(os.path.join(_paper1.PAPER1_DIR, name), tmp_path / name)
    with open(tmp_path / "solver_fd.py", "a") as f:
        f.write("\n# changed\n")
    with pytest.raises(ImportError, match="solver_fd.py"):
        _paper1.check_paper1(str(tmp_path))


@pytest.mark.parametrize("path", sorted(glob.glob(os.path.join(ROOT, "data", "design", "*.json"))))
def test_design_file_reproduces_from_its_seed(path):
    with open(path) as f:
        rec = json.load(f)
    assert {"numpy", "scipy", "python"} <= set(rec["versions"])
    assert make_set(rec["system"], rec["set"], rec["seed"]) == rec["points"]


def test_design_script_refuses_to_overwrite(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("design_sets", os.path.join(ROOT, "scripts", "design_sets.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(str(tmp_path))
    with pytest.raises(FileExistsError):
        mod.main(str(tmp_path))


def test_meta_records_the_constants(npz_a, npz_b):
    for path in (npz_a, npz_b):
        meta = gates.load_meta(path)
        assert meta["constants"] == gates.current_constants(meta["system"])
        assert meta["paper1_sha256"] == _paper1.PAPER1_SHA256
    assert {"K_E0", "DELTA_AX", "C_W", "RTOL", "ATOL", "PREWET", "COMPENSATE", "BED"} <= set(
        gates.current_constants("A"))
    assert {"D_P_B", "T_AMB", "RTOL", "ATOL"} <= set(gates.current_constants("B"))


def test_model_from_meta_refuses_changed_constants(npz_a, monkeypatch):
    meta = gates.load_meta(npz_a)
    gates.model_from_meta(meta)
    monkeypatch.setattr(materials, "DELTA_AX", 0.7)
    with pytest.raises(ValueError, match="DELTA_AX"):
        gates.model_from_meta(meta)


def test_wall_heat_is_float64(npz_a, npz_b):
    for path in (npz_a, npz_b):
        assert np.load(path)["wall_heat_blocks"].dtype == np.float64
