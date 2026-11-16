"""scripts/time_ceiling.py (T8 DOC-2, DOC-7, R1-6): the costs the ceiling needs that no earlier kernel
measured. It reports costs only (seconds), never an error value."""
import importlib.util
import json
import math
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("time_ceiling", ROOT / "scripts" / "time_ceiling.py")
tc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tc)


def test_quick_run_reports_every_cost_block_and_no_error_values(tmp_path):
    out = tmp_path / "t.json"
    res = tc.main(["--quick", "--out", str(out)])
    assert json.loads(out.read_text()) == json.loads(json.dumps(res))
    # latent arms at the study's largest d, one thread, h = 1 and 2
    assert res["settings"]["q"]["d"] == tc.QUICK["q"]["d"]
    for arm in tc.QUICK["arms"]:
        for h in (1, 2):
            r = res["train"][f"{arm}|h{h}|t1"]
            assert r["wall_s_per_iteration"] > 0 and r["cpu_s_per_iteration"] > 0
    assert set(res["step_S"]) >= {"h1|b1", "h2|b1", "h1|b48", "h2|b48"}
    # arm C and MUSCL-C: CPU seconds per cycle on each grid, per system key
    for key in tc.QUICK["keys"]:
        for g in tc.QUICK["grids_c"]:
            r = res["C"][f"{key}|{g}"]
            assert r["ok"] and len(r["cycle_cpu_s"]) == tc.QUICK["cycles"] and min(r["cycle_cpu_s"]) > 0
        for g in tc.QUICK["grids_muscl"]:
            assert res["MUSCL-C"][f"{key}|{g}"]["ok"]
    # one autoencoder epoch at the full-data snapshot count, balance weight 1 (R1) and 0 (plain MSE)
    for key in ("A", "B"):
        for w in (1, 0):
            r = res["ae"][f"{key}|w{w}"]
            assert r["epoch_s"] > 0 and r["n_snapshots"] == tc.QUICK["ae_traj"] * tc.QUICK["ae_cycles"] * tc.QUICK["n_snap"]
    # the reference after R3, CPU seconds per cycle at the registered grids
    for key in tc.QUICK["keys"]:
        assert all(s > 0 for s in res["reference"][key]["cycle_cpu_s"])
    flat = json.dumps(res)
    for word in ("nrmse", "eps_W", "balance_N", "field_err"):
        assert word not in flat
    assert all(math.isfinite(v["wall_s_per_iteration"]) for v in res["train"].values())


def test_quick_run_times_the_batched_rollout_step_of_every_arm(tmp_path):
    res = tc.main(["--quick", "--out", str(tmp_path / "t.json")])
    for arm in tc.QUICK["arms"]:
        r = res["step"][arm]
        assert ("error" in r) or all(r[f"h{h}|b48"]["wall_s_per_interval"] > 0 for h in (1, 2))
    assert "error" not in res["step"]["U"]
