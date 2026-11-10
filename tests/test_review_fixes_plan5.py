"""Plan 5 whole-plan review (opus, 2026-11-09): one test per fixed finding, on stubs (fast)."""
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from p3b.pilot import collect, pieces, runs

F64 = torch.float64
KEY = pieces.PRIMARY_KEY


# ---- stubs -------------------------------------------------------------------------------------------
class _Id:
    def encode(self, x):
        return x

    def decode(self, x):
        return x

    def encode_features(self, x):
        return x


def _val(n_traj=2, n_cyc=1, ns=6, nc=2, dts=None):
    n = 3 * nc + 2
    X = torch.zeros(n_traj, n_cyc, ns, n, dtype=F64)
    X[..., 0:3 * nc:3] = torch.arange(n_traj * n_cyc * ns, dtype=F64).reshape(n_traj, n_cyc, ns, 1)
    if dts is None:
        dts = [10.0] * (n_cyc * ns)
    t = torch.cumsum(torch.tensor([0.0] + list(dts[:-1]), dtype=F64), 0).reshape(n_cyc, ns)
    return SimpleNamespace(X=X, system="A", n_cells=nc, features=torch.zeros(n_traj, n_cyc, ns, 2, dtype=F64),
                           raw_inputs=torch.zeros(n_traj, n_cyc, ns, 3, dtype=F64),
                           t=t.expand(n_traj, n_cyc, ns).clone())


def _arm(rate=1.0):
    return SimpleNamespace(std=_Id(), ae=_Id(), kind="S", dyn=lambda z, F, R: -rate * z)


def _result(n_int, s_res, failed=None, flags=None):
    from p3b.eval.rollout import RolloutResult
    return RolloutResult(X=torch.zeros(1), X_end=torch.zeros(1), finite=torch.tensor([failed is None]),
                         failed_cycle=failed, cpu_s=0.0, flags=flags or {},
                         residuals={"S": torch.tensor(s_res[:n_int], dtype=F64)})


# ---- I1: K compares the released water of C+L and C directly -----------------------------------------
def test_I1_k_discrepancy_does_not_let_opposite_errors_cancel():
    rows = [{"arm": "C", "trajectory": 0, "W_pred": [1.015], "W_ref": [1.0], "failed_cycle": None},
            {"arm": "C+L", "trajectory": 0, "W_pred": [0.985], "W_ref": [1.0], "failed_cycle": None}]
    assert runs.k_discrepancy(rows) == pytest.approx(0.03)


def test_I1_k_discrepancy_is_infinite_for_a_failed_row():
    rows = [{"arm": "C", "trajectory": 0, "W_pred": [1.0], "W_ref": [1.0], "failed_cycle": None},
            {"arm": "C+L", "trajectory": 0, "W_pred": [1.0], "W_ref": [1.0], "failed_cycle": 1}]
    assert runs.k_discrepancy(rows) == math.inf


# ---- I2: the violation rate is per snapshot interval at every k ----------------------------------------
def test_I2_violation_rate_does_not_shrink_with_more_steps(monkeypatch):
    from p3b.eval import rollout as ro
    val = _val()
    res = [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]            # one violating interval of 6, the same at every k
    monkeypatch.setattr(ro, "rollout_batch", lambda arm, ds, js, n, k, track=False: [_result(6, res) for _ in js])
    out = runs.h_check(_arm(), val, tau_S=1e-3, steps=(1, 2, 4), n_states=2)
    rates = {k: v["viol_rate"] for k, v in out["h"].items()}
    assert rates[1] == rates[2] == rates[4] == pytest.approx(1 / 6)


# ---- I3: the stiffness check uses each picked state's own interval -------------------------------------
def test_I3_stiffness_uses_the_picked_snapshots_own_interval(monkeypatch):
    from p3b.eval import rollout as ro
    # the driest snapshot (index 0) sits before a long interval (30 s); the median interval is 10 s
    val = _val(n_traj=1, ns=6, dts=[30.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    monkeypatch.setattr(ro, "rollout_batch", lambda arm, ds, js, n, k, track=False: [_result(6, [0.0] * 6) for _ in js])
    out = runs.h_check(_arm(rate=0.1), val, tau_S=1.0, steps=(1,), n_states=1)
    assert out["h"][1]["h_rho"] == pytest.approx(3.0, rel=1e-4)       # 30 s x rho 0.1, not 10 s x 0.1
    assert out["h"][1]["stiff_ok"] is False


# ---- I4: defects per k, once per interval, failed rollouts NaN; final takes the chosen h -----------------
def test_I4_defects_are_recorded_per_k_once_per_interval(monkeypatch):
    from p3b.eval import rollout as ro

    def fake(arm, ds, j, n, k, track=False):
        if k == 1:
            return _result(6, [0.0] * 6, failed=1)
        return _result(6, [0.0] * 6, flags={"any_defect": 3, "rank_deficient": 3, "kappa_binds": 3})
    monkeypatch.setattr(ro, "rollout_batch", lambda arm_, ds_, js, n_, k_, track=False, _f=fake: [_f(arm_, ds_, j, n_, k_, track) for j in js])
    out = runs.h_check(_arm(), _val(), tau_S=1.0, steps=(1, 2), n_states=2)
    d = out["defects_per_step"]
    assert set(d) == {1, 2}
    assert all(math.isnan(x) for x in d[1])                          # failed rollouts: not a low count
    assert d[2] == [pytest.approx(0.5), pytest.approx(0.5)]          # 3 defect intervals of 6, counted once


def test_I4_rollout_counts_an_interval_with_any_flag_once():
    from p3b.eval.rollout import rollout
    val = _val(n_traj=1, ns=4)

    class Arm:
        std, ae, kind, lf = _Id(), _Id(), "S", None

        def step(self, z, F, R, dt, h_steps, track=False):
            return z, {"rank_deficient": torch.tensor([True]), "kappa_binds": torch.tensor([True]),
                       "projection_rank_deficient": torch.tensor([False])}
    r = rollout(Arm(), val, 0, 1, 1)
    assert r.flags["any_defect"] == 4 and r.flags["rank_deficient"] == 4


# ---- I5: runs with more than 5 % skipped windows are not finished -------------------------------------
def test_I5_train_curve_marks_a_mostly_skipped_run_unfinished(monkeypatch):
    from p3b.train import trainer
    monkeypatch.setattr(trainer, "train", lambda arm, ds, cfg: {"skipped": 9, "skipped_grad": 0})
    monkeypatch.setattr(runs, "validation_loss", lambda arm, val, h: 0.5)
    curve = runs.train_curve(None, None, None, lr=1e-3, iterations=20, every=10, h_steps=1, seed=0)
    assert [it for it, _ in curve] == [10, 20] and all(math.isnan(v) for _, v in curve)


def test_I5_cplusl_curve_marks_a_mostly_skipped_run_unfinished(monkeypatch):
    from p3b.train import cplusl

    def fake_train(mult, ds, material, n_coarse, lr, iterations, seed, every=0, callback=None, **kw):
        callback(10, mult, {"skipped": 3, "skipped_grad": 0})
        return {}
    monkeypatch.setattr(cplusl, "train_cplusl", fake_train)
    monkeypatch.setattr(cplusl, "cplusl_val_loss", lambda *a, **k: 0.5)
    curve = runs.cplusl_curve(None, None, "MOF-303", lr=1e-3, iterations=10, every=10, K=1, seed=0)
    assert curve[0][0] == 10 and math.isnan(curve[0][1])


def test_I5_two_seed_rows_report_their_training(monkeypatch):
    from p3b.eval import evaluate as ev
    from p3b.train import trainer
    monkeypatch.setattr(trainer, "train", lambda arm, ds, cfg: {"skipped": 1, "skipped_grad": 2})
    monkeypatch.setattr(ev, "evaluate", lambda *a, **k: [{"trajectory": 0}])
    monkeypatch.setattr(runs, "_physics", lambda ds: None)
    rows, train = runs.two_seed_rows(lambda: None, None, _val(), lr=1e-3, iterations=100, h_steps=1, seeds=(0,))
    assert rows == {0: [{"trajectory": 0}]}
    assert train == {0: {"iterations": 100, "skipped": 1, "skipped_grad": 2}}


def test_I5_final_stops_on_an_sd_run_with_too_many_skipped(tmp_path):
    stage = _full_stage()
    _write_all(tmp_path, stage, rows_skipped=10)                      # 10 of 100 iterations: 10 % > 5 %
    with pytest.raises(RuntimeError, match="skipped"):
        collect.final(tmp_path, tmp_path / "inputs", stage, keys=(KEY,), hyp_stat=_STAT, draws=50)


# ---- I6: pieces carry provenance; the collector checks the stage they ran under ------------------------
def test_I6_piece_files_carry_their_stage_hash_and_provenance(tmp_path):
    stage = {"d": {KEY: 16}}
    p = pieces._write(tmp_path, "x.json", {"a": 1}, stage=stage, q={"n": 1}, used={"data": "abc"})
    obj = json.loads(open(p).read())
    assert obj["value"] == {"a": 1}
    m = obj["meta"]
    assert m["stage_sha256"] == collect.stage_sha(stage) and m["q"] == {"n": 1} and m["used"] == {"data": "abc"}
    assert "torch" in m and "kit_commit" in m


def test_I6_collector_refuses_a_piece_from_another_stage(tmp_path):
    pieces._write(tmp_path, "lr_U.json", {"0.001": 1.0}, stage={"d": {KEY: 24}})
    with pytest.raises(RuntimeError, match="stage"):
        collect.load_piece(tmp_path, "lr_U.json", {"d": {KEY: 16}})


def test_I6_kernel_names_change_with_the_stage():
    import importlib.util
    import pathlib
    p = pathlib.Path(__file__).resolve().parent.parent / "kaggle" / "pilot_jobs.py"
    spec = importlib.util.spec_from_file_location("pilot_jobs", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    a = [j["name"] for j in m.phase_jobs(1, {"d": {KEY: 16}, "K": {KEY: 1}, "tau_S": {KEY: 0.0}})["jobs"]]
    b = [j["name"] for j in m.phase_jobs(1, {"d": {KEY: 24}, "K": {KEY: 1}, "tau_S": {KEY: 0.0}})["jobs"]]
    assert set(a).isdisjoint(b)


# ---- I7: pieces are found in the queue's nested download folders ----------------------------------------
def test_I7_load_piece_finds_nested_files_and_picks_the_current_stage(tmp_path):
    old, new = {"d": {KEY: 16}}, {"d": {KEY: 24}}
    pieces._write(tmp_path / "out" / "p3b-pilot-lr-u-aaaaaa" / "k", "lr_U.json", {"0.001": 9.0}, stage=old)
    pieces._write(tmp_path / "out" / "p3b-pilot-lr-u-bbbbbb" / "k", "lr_U.json", {"0.001": 1.0}, stage=new)
    assert collect.load_piece(tmp_path, "lr_U.json", new) == {"0.001": 1.0}
    with pytest.raises(FileNotFoundError):
        collect.load_piece(tmp_path, "lr_S.json", new)


# ---- M1, M2, M4, M13 ------------------------------------------------------------------------------------
def test_M1_positive_control_runs_at_the_holm_worst_case_level():
    rows = {k: [{"trajectory": j, "e": [1.0] * 6, "eps": [1.0] * 60} for j in range(4)] for k in (0, 1)}
    assert runs.positive_control(rows, rows, draws=20)["alpha"] == pytest.approx(0.05 / 3)


def test_M2_restaging_drops_the_values_of_later_phases(tmp_path):
    from p3b.pilot.pieces import LR_ARMS, _tag
    stage = {"d": {KEY: 24}, "K": {KEY: 2}, "tau_S": {KEY: 0.0}, "lr": {"U": 3e-3}, "iterations": 4000,
             "h_steps": {KEY: 1}}
    prev0 = collect.restrict(stage, 1)
    for a in LR_ARMS:
        pieces._write(tmp_path, f"lr_{_tag(a)}.json", {"0.001": 1.0}, stage=prev0)
    s = collect.stage(1, tmp_path, stage, keys=(KEY,))
    assert "iterations" not in s and "h_steps" not in s and s["d"] == {KEY: 24}


def test_M4_stage_data_refuses_duplicate_names(tmp_path):
    import shutil
    src = tmp_path / "a" / "A_primary_pilot_val_0000.npz"
    src.parent.mkdir(parents=True)
    meta = json.dumps({"design_set": "A_primary_pilot_val", "point": {"index": 0}})
    np.savez(src, meta=np.array(meta))
    (tmp_path / "b").mkdir()
    shutil.copy2(src, tmp_path / "b" / src.name)
    with pytest.raises(RuntimeError, match="duplicate"):
        collect.stage_data([tmp_path / "a", tmp_path / "b"], tmp_path / "data")


def test_M13_wall_piece_times_every_latent_arm(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(pieces, "files", lambda *a: [])
    import p3b.latent.data as data
    monkeypatch.setattr(data, "load_set", lambda f: None)
    monkeypatch.setattr(runs, "wall_per_iter", lambda makers, ds, **k: seen.setdefault("arms", sorted(makers)) and
                        {a: 1.0 for a in makers})
    pieces.piece_wall(tmp_path, {}, tmp_path, {})
    assert seen["arms"] == sorted(pieces.LATENT_ARMS)


# ---- helpers for final() --------------------------------------------------------------------------------
_STAT = {"H1": ("eps", 2), "H5": ("eps", 1), "H2": ("e", 2), "H4": ("e", 2)}


def _full_stage():
    from p3b.pilot.pieces import LR_ARMS
    return {"d": {KEY: 16}, "K": {KEY: 1}, "tau_S": {KEY: 0.0}, "lr": {a: 1e-3 for a in LR_ARMS},
            "iterations": 100, "h_steps": {KEY: 2}}


def _write_all(d, stage, rows_skipped=0):
    from p3b.pilot.pieces import LR_ARMS, TWO_SEEDS, _tag
    w = pieces._write
    w(d, f"data_{KEY}.json", {"css": [3], "decoder_gate": {"16": True},
                              "reference_noise": {"floors": {"N": 1e-6, "E": 1e-7}, "tau_ref": 1e-9,
                                                  "ref_production": [-1e-7, 1e-6]}}, stage=collect.restrict(stage, 0))
    w(d, f"cplusl_K_{KEY}.json", {"1": 0.0}, stage=collect.restrict(stage, 0))
    for a in LR_ARMS:
        w(d, f"lr_{_tag(a)}.json", {"0.001": 1.0}, stage=collect.restrict(stage, 1))
        w(d, f"curve_{_tag(a)}.json", [[1000, 1.0], [2000, 0.999]], stage=collect.restrict(stage, 2))
    w(d, f"s_{KEY}.json", {"h_check": {"2": {"viol_rate": 0.0, "stiff_ok": True, "h_rho": 1.0, "failed": 0}},
                           "g2": [1.0], "defects_per_step": {"1": [float("nan")], "2": [0.0, 1e-5]}},
      stage=collect.restrict(stage, 3))
    w(d, "wall_per_iter.json", {"U": 1.0, "S": 2.0}, stage=collect.restrict(stage, 3))
    rng = np.random.default_rng(0)
    for a in ("U", "S", "Uq"):
        for s in TWO_SEEDS:
            rows = [{"trajectory": j, "e": list(np.exp(rng.normal(size=2))), "eps": list(np.exp(rng.normal(size=2)))}
                    for j in range(4)]
            w(d, f"rows_{a}_s{s}.json", {"rows": rows, "train": {"iterations": 100, "skipped": rows_skipped,
                                                                  "skipped_grad": 0}}, stage=stage)


def test_final_takes_the_defects_of_the_chosen_h_and_writes_the_ten_inputs(tmp_path):
    stage = _full_stage()
    _write_all(tmp_path, stage)
    paths = collect.final(tmp_path, tmp_path / "inputs", stage, keys=(KEY,), hyp_stat=_STAT, draws=50)
    assert len(paths) == 10
    th = json.loads((tmp_path / "inputs" / "threshold_inputs.json").read_text())
    assert th[KEY]["defects_per_step"] == [0.0, 1e-5]                  # h_steps 2, not the failed h_steps 1


# ---- M8, M9: capped pairs and caveats beside the SDs ------------------------------------------------------
def test_M8_M9_notes_count_capped_pairs_and_state_the_caveats(tmp_path):
    stage = _full_stage()
    _write_all(tmp_path, stage)
    piece = json.loads((tmp_path / "rows_U_s0.json").read_text())
    piece["value"]["rows"][0]["failed_cycle"] = 1                    # U seed 0 failed on trajectory 0
    (tmp_path / "rows_U_s0.json").write_text(json.dumps(piece))
    collect.final(tmp_path, tmp_path / "inputs", stage, keys=(KEY,), hyp_stat=_STAT, draws=50)
    notes = json.loads((tmp_path / "inputs" / "pilot_notes.json").read_text())
    assert notes["capped_pairs"]["H1"] == {"either": 1, "both": 0, "pairs": 8}
    assert "H5" in notes["caveats"] and "H1" in notes["caveats"]


def test_M3_stage3_stops_when_any_key_has_no_admissible_h(tmp_path):
    stage = collect.restrict(_full_stage(), 3)
    ok = {"1": {"viol_rate": 0.0, "stiff_ok": True, "h_rho": 1.0, "failed": 0}}
    bad = {"1": {"viol_rate": 1.0, "stiff_ok": True, "h_rho": 1.0, "failed": 1}}
    for key, hc in ((KEY, ok), ("B", bad)):
        pieces._write(tmp_path, f"s_{key}.json", {"h_check": hc, "g2": [1.0], "defects_per_step": {}}, stage=stage)
    with pytest.raises(RuntimeError, match="B"):
        collect.stage(3, tmp_path, stage, keys=(KEY, "B"))


def test_wall_piece_also_times_one_thread_for_the_job_layout(monkeypatch, tmp_path):
    """Small tensors may gain nothing from 4 intra-op threads; one-thread timings decide whether a
    Kaggle session runs one 4-thread job or four 1-thread jobs (recorded in the piece's meta)."""
    calls = []
    monkeypatch.setattr(pieces, "files", lambda *a: [])
    import p3b.latent.data as data
    monkeypatch.setattr(data, "load_set", lambda f: None)

    def fake(makers, ds, **k):
        calls.append(torch.get_num_threads())
        return {a: float(torch.get_num_threads()) for a in makers}
    monkeypatch.setattr(runs, "wall_per_iter", fake)
    torch.set_num_threads(4)
    p = pieces.piece_wall(tmp_path, {}, tmp_path, {})
    obj = json.loads(open(p).read())
    assert calls == [4, 1] and torch.get_num_threads() == 4
    assert all(v == 4.0 for v in obj["value"].values())
    assert all(v == 1.0 for v in obj["meta"]["used"]["timing_1_thread"].values())


# ---- 2026-11-10: pilot kernels on one thread (Kaggle: 4 threads no faster; S-L's curve near 12 h on 4) ----
def test_pilot_jobs_run_on_one_thread():
    import importlib.util
    import pathlib
    spec = importlib.util.spec_from_file_location(
        "pilot_jobs", pathlib.Path(__file__).resolve().parent.parent / "kaggle" / "pilot_jobs.py")
    pj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pj)
    for phase in range(5):
        for j in pj.phase_jobs(phase, {})["jobs"]:
            assert j["threads"] == (4 if j["set"].startswith("ae-") else 1), j["name"]
