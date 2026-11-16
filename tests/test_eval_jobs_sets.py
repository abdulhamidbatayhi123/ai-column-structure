"""Evaluation jobs only for sets that exist (T8 ceiling pass, 2026-11-17): GATED_SETS named a "test" set that no
system has (p3b.design.SET_SIZES), and the OOD set exists for A-primary only, so evaluation_jobs made one empty
kernel per cell for "test" (and for OOD outside A-primary), each paying a Kaggle session start-up."""
import importlib.util
import pathlib
import sys

from p3b.design import SET_SIZES
from p3b.train.cells import DESIGN_NAME, cells, job_name, sensitivity_cells

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _train_jobs(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec = importlib.util.spec_from_file_location("train_jobs", ROOT / "kaggle" / "train_jobs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_evaluation_job_names_a_set_its_system_has(monkeypatch):
    tj = _train_jobs(monkeypatch)
    cs = cells() + sensitivity_cells()
    ev = [j for j in tj.evaluation_jobs(cs, kit_slug="o/kit")["jobs"] if j["system"] == "evaluate"]
    by_name = {job_name(c): c for c in cs}
    seen = set()
    for j in ev:
        cell_name, set_name = j["commands"][0][2], j["commands"][0][3]
        c = by_name[cell_name]
        assert (DESIGN_NAME[(c.system, c.material)], set_name) in SET_SIZES, (cell_name, set_name)
        seen.add((cell_name, set_name))
    for c in cs:                                   # every cell keeps its long test; OOD only where it exists
        assert (job_name(c), "long_test") in seen
        assert ((job_name(c), "ood") in seen) == (DESIGN_NAME[(c.system, c.material)] == "A_primary")
