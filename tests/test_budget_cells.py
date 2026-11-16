"""The budget's job counts (scripts/budget.py, pure Python) agree with the code that makes the jobs
(p3b.train.cells, kaggle/train_jobs.py), so the registered need counts what will run."""
import importlib.util
import pathlib

from p3b.train.cells import LEARNED_WITH_AE, ae_job_name, cells, gate_specs, sensitivity_cells

ROOT = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("budget", ROOT / "scripts" / "budget.py")
budget = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget)


def test_budget_counts_match_the_cells():
    cs = cells()
    specs = budget.ae_specs()
    study = {ae_job_name(c) for c in cs if c.arm in LEARNED_WITH_AE}
    assert sum(s["kind"] == "study" for s in specs) == len(study)
    assert sum(s["kind"] == "gate" for s in specs) == len(gate_specs(cs, "mlp"))
    # the sensitivity cells of one kind and seed share one autoencoder (U, U+F and S alike)
    assert sum(s["kind"] == "sensitivity" for s in specs) == len({(c.arm.split("|")[1], c.seed) for c in sensitivity_cells()})
    runs = budget.training_runs()
    for arm, n in runs.items():
        n_cells = sum(c.arm == arm for c in cs) + sum(c.arm.split("|")[0] == arm for c in sensitivity_cells())
        assert n == n_cells, arm
    assert budget.n_cells() == len(cs) + len(sensitivity_cells())
