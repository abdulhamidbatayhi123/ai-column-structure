"""Plan 4, Task 4b: solver-arm cells, nested training subsets, shared autoencoders as Kaggle kernel
dependencies."""
import json
import sys

import pytest

from p3b.train.cells import Cell, ae_job_name, cells, job_name, train_cell

POINT_A = dict(RH_ads=0.40, T_ads=303.15, T_des=353.15, v=0.10, alpha=0.8, beta=0.6, index=0, n_cycles=1)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    from p3b.generate import generate_trajectory
    d = tmp_path_factory.mktemp("c4b")
    for split, pts in (("train", [dict(POINT_A, index=i, RH_ads=0.3 + 0.05 * i) for i in range(3)]),
                       ("validation", [dict(POINT_A, index=9)])):
        (d / split).mkdir()
        for pt in pts:
            assert generate_trajectory("A", pt, "MOF-303", N_z=32, N_c=8, path=d / split / f"t{pt['index']}.npz",
                                       n_snap=20).ok
    return d


def test_queue_dependencies_set_kernel_sources_and_gate_the_push(tmp_path, monkeypatch):
    sys.path.insert(0, "kaggle")
    import run_queue
    monkeypatch.setattr(run_queue, "BUILD", str(tmp_path))
    spec = {"kit_slug": "o/kit", "nz": 0, "jobs": [
        {"name": "p3b-ae-x", "commands": [["a.py"]], "expect": ["a.pt"]},
        {"name": "p3b-u-x", "commands": [["b.py"]], "expect": ["b.json"], "depends_on": ["p3b-ae-x"]}]}
    ks = run_queue.write_kernels(spec)
    meta = json.loads((tmp_path / "p3b-u-x" / "kernel-metadata.json").read_text())
    assert meta["kernel_sources"] == [f"{run_queue.OWNER}/p3b-ae-x"]
    k = [k for k in ks if k["name"] == "p3b-u-x"][0]
    assert not run_queue.ready(k, done=set()) and run_queue.ready(k, done={"p3b-ae-x"})


def test_a_size_cell_trains_on_the_first_trajectories_of_the_pool(data, tmp_path):
    pilot = {"lr": {"U": 1e-3}, "iterations": 1, "h_steps": 1, "d": 4, "kappa": 1e-3,
             "floors": {"N": 1e-6, "E": 1e-6}, "tau_S": 1e-6}
    out = train_cell(Cell("U", "A", "MOF-303", 2, 0), str(data), str(tmp_path), pilot,
                     quick=dict(ae_epochs=1, ae_width=8, width=8, n_windows=1, n_steps=2, n_cycles=1, pool_size=3))
    meta = json.loads(out.read_text())["meta"]
    assert meta["train_indices"] == [0, 1]


def test_solver_cells_score_every_grid(data, tmp_path):
    pilot = {"floors": {"N": 1e-6, "E": 1e-6}, "tau_S": 1e-6}
    out = train_cell(Cell("C", "A", "MOF-303", "full", 0), str(data), str(tmp_path), pilot, quick=dict(grids=[4, 8]))
    rows = json.loads(out.read_text())["rows"]
    assert {r["arm"] for r in rows} == {"C@4", "C@8"}


def test_jobs_put_each_autoencoder_before_the_cells_that_share_it():
    sys.path.insert(0, "kaggle")
    import train_jobs
    spec = train_jobs.build_jobs(cells(), kit_slug="o/kit")
    names = [j["name"] for j in spec["jobs"]]
    assert len(names) == len(set(names))
    pos = {n: i for i, n in enumerate(names)}
    for c in cells():
        j = spec["jobs"][pos[job_name(c)]]
        if c.arm in ("C", "MUSCL-C", "FNO"):
            assert not j.get("depends_on")
        else:
            dep = ae_job_name(c) if c.arm != "C+L" else None
            if dep:
                assert j["depends_on"] == [dep] and pos[dep] < pos[job_name(c)]
