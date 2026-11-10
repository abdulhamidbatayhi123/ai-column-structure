"""scripts/budget.py: the compute budget from measured per-iteration costs and the design table."""
import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("budget", ROOT / "scripts" / "budget.py")
budget = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget)

ARMS = ("U", "U+F", "P", "S", "S-L", "S-M", "S-Pi", "FNO")


def flat_timing(s1=1.0, s2=2.0):
    """Every arm 1 s per iteration at h = 1 and 2 s at h = 2, one thread."""
    return {"train": {f"{a}|h{k}|t1": {"wall_s_per_iteration": s1 if k == 1 else s2} for a in ARMS for k in (1, 2)}}


def test_training_runs_per_arm_match_the_design_table():
    # PREREG design table: A-primary full / 32 / learning curve 16, 64, 128; A-replication full / 32;
    # B full / 32; plus the sensitivities (next latent dimension, degraded data: U, U+F, S, 3 seeds each)
    assert budget.training_runs() == {"U": 41, "U+F": 32, "P": 25, "S": 41, "S-L": 16, "S-M": 5, "S-Pi": 5,
                                      "FNO": 6, "U-long": 3, "C+L": 6}


def test_run_hours_scale_with_iterations_and_h_and_extrapolate_h4_linearly():
    t = flat_timing()
    assert budget.run_hours(t, "S", iterations=3600, h=1) == pytest.approx(1.0)
    assert budget.run_hours(t, "S", iterations=3600, h=2) == pytest.approx(2.0)
    assert budget.run_hours(t, "S", iterations=3600, h=4) == pytest.approx(4.0)   # a + b h through h1, h2
    assert budget.run_hours(t, "U-long", iterations=3600, h=2) == budget.run_hours(t, "S", 3600, 2)


def test_training_total_is_runs_times_hours_with_c_plus_l_flagged_as_unmeasured():
    b = budget.training_budget(flat_timing(), iterations=3600, h=1, c_plus_l_hours=1.0)
    assert b["total"] == pytest.approx(sum(budget.training_runs().values()))
    assert "C+L" in b["unmeasured"]


def test_reduction_order_savings_are_listed_in_the_preregistered_order():
    b = budget.training_budget(flat_timing(), iterations=3600, h=1, c_plus_l_hours=1.0)
    names = [r["step"] for r in b["reductions"]]
    assert names == ["degraded-data sensitivity", "U-long", "S-Pi", "latent-dimension sensitivity", "MUSCL-C",
                     "learning-curve sizes 16 and 128 to 2 seeds", "System B S-L"]
    saved = {r["step"]: r["hours"] for r in b["reductions"]}
    assert saved["degraded-data sensitivity"] == pytest.approx(9.0)
    assert saved["learning-curve sizes 16 and 128 to 2 seeds"] == pytest.approx(6.0)   # U, P, S x 2 sizes
    assert saved["System B S-L"] == pytest.approx(6.0)
