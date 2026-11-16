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
    # B full / 32; plus the sensitivities (next latent dimension, degraded data: U, U+F, S, 3 seeds each;
    # autoencoder loss, R1: U and S, 3 seeds each)
    assert budget.training_runs() == {"U": 44, "U+F": 32, "P": 25, "S": 44, "S-L": 16, "S-M": 5, "S-Pi": 5,
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
    assert names == ["degraded-data sensitivity", "U-long", "S-Pi", "latent-dimension sensitivity",
                     "autoencoder-loss sensitivity", "MUSCL-C", "learning-curve sizes 16 and 128 to 2 seeds",
                     "System B S-L"]
    saved = {r["step"]: r["hours"] for r in b["reductions"]}
    assert saved["degraded-data sensitivity"] == pytest.approx(9.0)
    assert saved["autoencoder-loss sensitivity"] == pytest.approx(6.0)                  # U, S x 3 seeds
    assert saved["learning-curve sizes 16 and 128 to 2 seeds"] == pytest.approx(6.0)   # U, P, S x 2 sizes
    assert saved["System B S-L"] == pytest.approx(6.0)


def test_evaluation_hours_from_the_batched_step_cost():
    # 120 projected-arm rollouts of a 48-trajectory test set, 60 cycles x 200 snapshot intervals
    assert budget.evaluation_hours(0.3) == pytest.approx(120 * 60 * 200 * 0.3 / 3600)
    t2 = {"step_S": {"h2|b48": {"wall_s_per_interval": 0.18}, "h2|b1": {"wall_s_per_interval": 0.075}}}
    assert budget.evaluation_hours_from(t2, h=2) == pytest.approx(budget.evaluation_hours(0.18))
    assert budget.evaluation_hours_from(t2, h=2, batched=False) == pytest.approx(48 * budget.evaluation_hours(0.075))


def test_session_throughput_from_the_concurrency_timing():
    t2 = {"concurrent": {"S|h2|n1": {"wall_s_per_iteration": [4.0]},
                         "S|h2|n4": {"wall_s_per_iteration": [8.0, 8.0, 8.0, 8.0]}}}
    assert budget.session_throughput(t2, "S", 4) == pytest.approx(2.0)   # 4 jobs, each twice as slow


# ---- the whole need from a script (T8 DOC-2, DOC-7, R1-6) ---------------------------------------------
def flat_inputs(s=1.0, step=0.1, c=2.0, ae_epoch=60.0, ref_a=30.0, ref_b=2.0):
    """Every latent arm s s per iteration (h = 1 and 2 alike), S's batched step `step` s per interval,
    C and MUSCL-C c CPU-s per cycle on every grid, one full-data autoencoder epoch ae_epoch s."""
    arms = ARMS
    keys = ("A_MOF-303", "A_Al-fumarate", "B")
    return {
        "train": {f"{a}|h{k}|t1": {"wall_s_per_iteration": s} for a in arms for k in (1, 2)},
        "step_S": {f"h{k}|b48": {"wall_s_per_interval": step} for k in (1, 2)},
        "C": {f"{key}|{g}": {"ok": True, "cycle_cpu_s": [c, c]} for key in keys for g in (16, 32, 64, 128)},
        "MUSCL-C": {f"{key}|{g}": {"ok": True, "cycle_cpu_s": [c, c]} for key in keys for g in (32, 64)},
        "ae": {f"{sy}|w{w}": {"epoch_s": ae_epoch} for sy in ("A", "B") for w in (0, 1)},
        "reference": {"A_MOF-303": {"cycle_cpu_s": [ref_a]}, "A_Al-fumarate": {"cycle_cpu_s": [ref_a]},
                      "B": {"cycle_cpu_s": [ref_b]}},
        "cl_train": {"K1": {"wall_s_per_iter": 3.0}, "K4": {"wall_s_per_iter": 9.0}},
        "cl_eval": {"n_traj": 16, "cycles": 2, "K1": {"wall_s": 32.0}, "K8": {"wall_s": 32.0 * 8}},
    }


def test_autoencoder_jobs_of_the_study_the_gate_wave_and_the_sensitivities():
    specs = budget.ae_specs()
    kind = lambda k: [s for s in specs if s["kind"] == k]  # noqa: E731
    assert len(kind("study")) == 35                      # one per (system, material, size, seed) of the learned cells
    assert len(kind("gate")) == 65                       # (5 + 3 + 5) full-data seeds x 5 d (cells.gate_specs)
    assert len(kind("sensitivity")) == 9                 # degraded data, next d, plain MSE (R1): 3 each
    assert sum(s["weight"] == 0 for s in specs) == 3     # only R1's sensitivity trains with plain MSE
    assert all(s["weight"] == 1 for s in kind("study") + kind("gate"))


def test_autoencoder_hours_scale_with_the_pool_size():
    t = flat_inputs(ae_epoch=60.0)                       # 60 epochs x 60 s = 1 h at full data (256 trajectories)
    full = {"kind": "study", "system": "A", "size": "full", "weight": 1}
    assert budget.ae_hours(t, [full]) == pytest.approx(1.0)
    assert budget.ae_hours(t, [dict(full, size=32)]) == pytest.approx(32 / 256)


def test_c_plus_l_training_hours_come_from_the_measured_k():
    t = flat_inputs()
    assert budget.cplusl_run_hours(t, iterations=3600, K=1) == pytest.approx(3.0)
    assert budget.cplusl_run_hours(t, iterations=3600, K=4) == pytest.approx(9.0)
    with pytest.raises(KeyError):
        budget.cplusl_run_hours(t, iterations=3600, K=2)  # not measured: never guessed


def test_latent_evaluation_of_one_cell_counts_batches_cycles_and_intervals():
    t = flat_inputs(step=0.1)
    # B: validation 32 x 6, long validation 4 x 60, long test 48 x 60, each one batch of <= 48 trajectories
    expect = (6 + 60 + 60) * 200 * 0.1 / 3600
    assert budget.latent_eval_hours(t, "B", h=1, n_arms=1) == pytest.approx(expect)
    # A-primary adds the OOD set (24 x 30); U+proj doubles the rollouts of a U cell
    expect_a = (6 + 60 + 60 + 30) * 200 * 0.1 / 3600
    assert budget.latent_eval_hours(t, "A_MOF-303", h=1, n_arms=2) == pytest.approx(2 * expect_a)


def test_solver_evaluation_uses_the_slowest_measured_cycle():
    t = flat_inputs(c=2.0)
    t["C"]["B|16"]["cycle_cpu_s"] = [2.0, 5.0]
    tc_b = 32 * 6 + 4 * 60 + 48 * 60
    assert budget.solver_eval_hours(t, "C", "B") == pytest.approx(tc_b * (5.0 + 2.0 * 3) / 3600)
    assert budget.solver_eval_hours(t, "MUSCL-C", "A_MOF-303") == pytest.approx(
        (tc_b + 24 * 30) * 2.0 * 2 / 3600)


def test_need_lists_every_block_and_sums_them_for_each_h():
    t = flat_inputs()
    for h in (1, 2, 4):
        n = budget.need(t, iterations=3600, h=h, K=4, n_css=13)
        assert set(n["blocks"]) == {"training, latent arms", "training, C+L", "autoencoders", "evaluation, latent arms",
                                    "evaluation, solver arms", "H6 searches and reference checks",
                                    "reference data (confirmatory sets)"}
        assert n["total_run_hours"] == pytest.approx(sum(b["hours"] for b in n["blocks"].values()))
        assert n["sessions"] >= n["kernels"] > 0
    assert budget.need(t, 3600, 4, 4, 13)["total_run_hours"] == pytest.approx(
        budget.need(t, 3600, 2, 4, 13)["total_run_hours"])   # flat timing: h does not change the cost here


def test_session_hours_add_the_start_up_of_every_session_a_job_needs():
    # a 23-hour job needs 3 sessions of at most 11 h of work; each pays the start-up
    assert budget.job_sessions(23.0) == 3 and budget.job_sessions(0.5) == 1
    assert budget.session_hours([23.0, 0.5], startup_s=360.0) == pytest.approx(23.5 + 4 * 0.1)


def test_each_arm_is_evaluated_at_its_own_measured_batched_step_when_there_is_one():
    t = flat_inputs(step=0.1)
    t["step"] = {"U": {"h1|b48": {"wall_s_per_interval": 0.01}, "h2|b48": {"wall_s_per_interval": 0.02}}}
    expect_u = (6 + 60 + 60) * 200 * 0.01 / 3600
    assert budget.latent_eval_hours(t, "B", h=1, n_arms=1, arm="U") == pytest.approx(expect_u)
    assert budget.latent_eval_hours(t, "B", h=1, n_arms=1, arm="P") == pytest.approx(expect_u * 10)  # S's step
    assert budget.latent_eval_hours(t, "B", h=4, n_arms=1, arm="U") == pytest.approx(expect_u * 4)   # a + b h


def test_the_reference_cost_is_the_slowest_of_the_r3_probe_and_the_earlier_records(tmp_path):
    """The R3 probe times one point per key; the earlier exact-Jacobian records (timing_reference.json,
    8 points for A) stay as a floor, so the data block is never priced on one fast point."""
    import json as js
    t = flat_inputs(ref_a=10.0, ref_b=0.4)
    for f, obj in {"timing_arms.json": {"train": t["train"]},
                   "timing_concurrency_batch.json": {"step_S": t["step_S"], "concurrent": {}},
                   "timing_cplusl_train.json": t["cl_train"], "timing_cplusl_eval.json": t["cl_eval"],
                   "timing_ceiling.json": {**{k: t[k] for k in ("train", "step_S", "C", "MUSCL-C", "ae", "reference")},
                                           "step": {}, "settings": {"q": {"d": 64}}},
                   "timing_reference.json": {"records": [
                       {"system": "A", "N_z": 2048, "ok": True, "cpu_s": 26.0, "jacobian": "exact"},
                       {"system": "B", "N_z": 1024, "ok": True, "cpu_s": 1.5},
                       {"system": "B", "N_z": 512, "ok": True, "cpu_s": 9.0}],
                       "data_cost_decision": {"system_A_2048_exact_max": 26.0, "N_z_B": 1024}}}.items():
        (tmp_path / f).write_text(js.dumps(obj))
    got = budget.load_inputs(str(tmp_path))
    assert max(got["reference"]["A_MOF-303"]["cycle_cpu_s"]) == 26.0
    assert max(got["reference"]["B"]["cycle_cpu_s"]) == 1.5          # only the registered grid (1024 cells)
