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


def test_time_arms_concurrency_mode_runs_n_one_thread_workers_at_once(tmp_path):
    """Four 1-thread jobs per session only pay if they do not slow each other: measure n workers at once."""
    spec = importlib.util.spec_from_file_location("time_arms", ROOT / "scripts" / "time_arms.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    out = tmp_path / "c.json"
    m.main(["--out", str(out), "--quick", "--concurrency", "1,2"])
    r = json.loads(out.read_text())
    assert set(r["concurrent"]) == {f"{a}|h2|n{n}" for a in ("U", "S") for n in (1, 2)}
    for key, v in r["concurrent"].items():
        n = int(key.rsplit("n", 1)[1])
        assert len(v["wall_s_per_iteration"]) == n and all(w > 0 for w in v["wall_s_per_iteration"])
    assert "train" not in r


def test_time_arms_batch_mode_times_s_rollout_step_per_batch_size(tmp_path):
    """Evaluation rolls out many test trajectories: S's step cost at batch 1 and at the test-set batch."""
    spec = importlib.util.spec_from_file_location("time_arms", ROOT / "scripts" / "time_arms.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    out = tmp_path / "b.json"
    m.main(["--out", str(out), "--quick", "--batch", "1,4"])
    r = json.loads(out.read_text())
    assert set(r["step_S"]) == {f"h{k}|b{b}" for k in (1, 2) for b in (1, 4)}
    # an untrained tiny model may diverge; then the failure is recorded, never a fake time
    assert all(v["threads"] == 1 and (v["wall_s_per_interval"] > 0 or v["failed_at"] is not None)
               for v in r["step_S"].values())


def test_time_batched_success_path_times_every_interval_of_a_stable_arm(tmp_path):
    """The quick S model diverges (as in rollout()); U does not, so the timed path itself is checked here."""
    spec = importlib.util.spec_from_file_location("time_arms", ROOT / "scripts" / "time_arms.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.make_data(str(tmp_path), 32, 8, 40)
    q = {"d": 4, "ae_epochs": 1, "ae_width": 8, "width": 8, "n_windows": 1, "n_steps": 2}
    r = m.time_batched(str(tmp_path), q, [1, 4], steps=(2,), intervals=5, arm_name="U")
    for b in (1, 4):
        v = r[f"h2|b{b}"]
        assert v["failed_at"] is None and v["intervals"] == 5 and v["wall_s_per_interval"] > 0
