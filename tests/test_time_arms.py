"""scripts/time_arms.py: the arms' costs measured on Kaggle before the pilot (quick settings here)."""
import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_time_arms_quick_run_reports_costs_per_arm_step_and_thread(tmp_path):
    spec = importlib.util.spec_from_file_location("time_arms", ROOT / "scripts" / "time_arms.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    out = tmp_path / "t.json"
    m.main(["--out", str(out), "--quick"])
    r = json.loads(out.read_text())
    assert set(r["train"]) == {f"{a}|h{k}|t{t}" for a in ("U", "S") for k in (1, 2) for t in (4, 1)}
    assert all(v["wall_s_per_iteration"] > 0 and v["iterations"] == 1 for v in r["train"].values())
    assert set(r["rollout_S"]) == {f"h{k}|t{t}" for k in (1, 2) for t in (4, 1)}
    assert r["settings"]["n_snap"] == 40 and "commit" in r
