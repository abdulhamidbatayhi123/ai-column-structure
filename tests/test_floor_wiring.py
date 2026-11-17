"""The decoder-only floor (PREREG 6, R1) in the confirmatory pipeline: one floor job per A-primary study
autoencoder (every size) on each of its test sets (long test, OOD), after H3a, as scripts/budget.py prices it;
p3b.train.cells.floor_cell loads the study autoencoder (never trains one), builds the projected arms'
LatentFunctionals, and writes both floors (p3b.eval.floor) to <autoencoder job>_floor_<set>.json."""
import importlib.util
import json
import pathlib
import shutil
import sys

import pytest

from test_review_fixes_plan4 import QUICK, _cp, study  # noqa: F401  (fixture)

from p3b.train.cells import LEARNED_WITH_AE, Cell, ae_job_name, cells, sensitivity_cells

ROOT = pathlib.Path(__file__).resolve().parent.parent
KEY = "A_MOF-303"
SETS = ("long_test", "ood")


def _load(name, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec = importlib.util.spec_from_file_location(name, ROOT / "kaggle" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _primary_aes():
    return sorted({ae_job_name(c) for c in cells()
                   if c.arm in LEARNED_WITH_AE and (c.system, c.material) == ("A", "MOF-303")})


def test_one_floor_job_per_a_primary_study_autoencoder_and_test_set(monkeypatch):
    tj = _load("train_jobs", monkeypatch)
    jobs = tj.evaluation_jobs(cells() + sensitivity_cells(), kit_slug="o/kit")["jobs"]
    floor = {j["name"]: j for j in jobs if "--floor" in j["commands"][0]}
    want = {f"{ae}-floor-{s.replace('_', '')}": (ae, s) for ae in _primary_aes() for s in SETS}
    assert set(floor) == set(want)
    names = {j["name"] for j in jobs}
    for n, (ae, s) in want.items():
        j = floor[n]
        assert j["commands"] == [["kaggle/train_chunk.py", "--floor", ae, s, "--out", "OUT"]]
        assert j["depends_on"] == [ae] and ae in names            # its producer is in the same jobs file
        assert j["expect"] == [f"{ae}_floor_{s}.json"] and j["threads"] == 1 and j["system"] == "evaluate"
        assert len(n) <= 50


def test_the_budget_prices_exactly_the_floor_jobs(monkeypatch):
    tj = _load("train_jobs", monkeypatch)
    spec = importlib.util.spec_from_file_location("budget", ROOT / "scripts" / "budget.py")
    budget = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(budget)
    n_jobs = sum("--floor" in j["commands"][0] for j in tj.evaluation_jobs(cells(), kit_slug="o/kit")["jobs"])
    n_budget = sum(s["kind"] == "study" and s["key"] == budget.P_KEY for s in budget.ae_specs()) * len(budget.GATED)
    assert n_jobs == n_budget == len(_primary_aes()) * len(SETS)


@pytest.fixture(scope="module")
def fstudy(study, tmp_path_factory):  # noqa: F811
    """The quick study data with its test trajectory as the long test."""
    d = tmp_path_factory.mktemp("fst")
    shutil.copytree(study / "train", d / "train")
    shutil.copytree(study / "test", d / "long_test")
    return d


@pytest.fixture(scope="module")
def ae_input(fstudy, tmp_path_factory):
    """A study autoencoder as its job would leave it (trained here by a quick U cell)."""
    from p3b.train.cells import train_cell
    inp = tmp_path_factory.mktemp("inp")
    train_cell(Cell("U", "A", "MOF-303", "full", 0), str(fstudy), str(inp / "ae-kernel"), _cp(), quick=QUICK)
    return inp


def test_floor_cell_writes_both_floors_for_the_study_autoencoder(fstudy, ae_input, tmp_path):
    from p3b.train.cells import _ae_file, _sha, floor_cell
    cell = Cell("S", "A", "MOF-303", "full", 0)
    pilot = dict(_cp(), ae_input_root=str(ae_input))
    out = floor_cell(cell, "long_test", str(fstudy), str(tmp_path / "out"), pilot, {KEY: "pass"}, quick=QUICK)
    assert out.name == f"{ae_job_name(cell)}_floor_long_test.json"
    doc = json.loads(out.read_text())
    meta, rows = doc["meta"], doc["rows"]
    assert meta["kind"] == "decoder_floor" and meta["set"] == "long_test"
    assert (meta["system"], meta["material"], meta["size"], meta["seed"]) == ("A", "MOF-303", "full", 0)
    ae_path = next(ae_input.rglob(_ae_file(cell)))
    assert meta["autoencoder"]["sha256"] == _sha(ae_path) and meta["pilot"]["h_steps"] == 1
    assert {r["floor"] for r in rows} == {"U", "projected"}
    n_traj = len(list((fstudy / "long_test").glob("*.npz")))
    for kind in ("U", "projected"):
        mine = [r for r in rows if r["floor"] == kind]
        assert len(mine) == n_traj
        assert all({"e", "eps", "violations", "W_pred", "W_ref", "design_index"} <= set(r) for r in mine)
    assert any(r["violations"]["E"] is not None for r in rows if r["floor"] == "projected")


def test_floor_cell_needs_the_h3a_gate_and_the_study_autoencoder(fstudy, ae_input, tmp_path):
    from p3b.train.cells import floor_cell
    cell = Cell("S", "A", "MOF-303", "full", 0)
    with pytest.raises(PermissionError, match="H3a"):
        floor_cell(cell, "long_test", str(fstudy), str(tmp_path / "a"), dict(_cp(), ae_input_root=str(ae_input)),
                   {KEY: "defect"}, quick=QUICK)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="autoencoder"):          # never trains one, even in quick runs
        floor_cell(cell, "long_test", str(fstudy), str(tmp_path / "b"), dict(_cp(), ae_input_root=str(empty)),
                   {KEY: "pass"}, quick=QUICK)


def test_train_chunk_floor_entry_point(tmp_path, monkeypatch):
    tc = _load("train_chunk", monkeypatch)
    seen = {}
    monkeypatch.setattr(tc, "floor_cell", lambda c, s, d, o, p, g: seen.update(cell=c, set=s, pilot=p, gate=g)
                        or pathlib.Path(o) / "x.json")
    th = {"kappa": 1e-3, "tau_S": 2e-6, "violation_N": 1e-5, "violation_E": 1e-5}
    pilot = {"lr": {"S": 1e-3}, "iterations": 10, "h_steps": {KEY: 2}, "K": {}, "thresholds": {KEY: th}}
    files = {"pilot": pilot, "choice": {KEY: {"decoder": "mlp", "d": 16}}, "gate": {KEY: "pass"}}
    for k, v in files.items():
        (tmp_path / f"{k}.json").write_text(json.dumps(v))
    ae = _primary_aes()[0]
    tc.main(["--floor", ae, "ood", "--out", str(tmp_path / "out"), "--pilot", str(tmp_path / "pilot.json"),
             "--decoder-choice", str(tmp_path / "choice.json"), "--gate", str(tmp_path / "gate.json")])
    assert ae_job_name(seen["cell"]) == ae and seen["set"] == "ood" and seen["gate"] == {KEY: "pass"}
    assert seen["pilot"]["h_steps"] == 2 and seen["pilot"]["d"] == 16 and seen["pilot"]["tau_S"] == 2e-6
