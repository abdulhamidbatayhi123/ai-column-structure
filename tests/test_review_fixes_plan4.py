"""Fixes from the whole-branch review of Plan 4 (one test per finding)."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "kaggle"))


# ---- Critical 4: kernels run the kit read-only and write only their outputs -------------------------
def test_kernel_runs_read_only_from_the_kit_and_writes_only_outputs(tmp_path):
    import run_queue
    kit = tmp_path / "input" / "p3b-reference-kit"
    (kit / "kaggle").mkdir(parents=True)
    (kit / "kaggle" / "generate_chunk.py").write_text("")
    (kit / "data").mkdir()
    (kit / "data" / "big.npz").write_text("x" * 1000)
    (kit / "w.py").write_text("import sys, os\nopen(os.path.join(sys.argv[1], 'r.json'), 'w').write('{}')\n")
    script = run_queue.RUN_CMDS_PY.format(cmds=[["w.py", "OUT"]], kit_name="p3b-reference-kit", threads=4)
    base = str(tmp_path).replace("\\", "/")
    script = script.replace("/kaggle/input", base + "/input").replace("/kaggle/working", base + "/working")
    (tmp_path / "run.py").write_text(script)
    subprocess.run([sys.executable, str(tmp_path / "run.py")], check=True)
    out = tmp_path / "working" / "out"
    assert (out / "r.json").exists()
    written = [p for p in (tmp_path / "working").rglob("*") if p.is_file()]
    assert written == [out / "r.json"]                                # no kit copy, no data


# ---- Critical 1: the pilot output drives the cells (h_steps, one key function) -----------------------
def test_pilot_output_feeds_cell_pilot(tmp_path):
    from p3b.pilot.rules import choose_h_steps, pilot_key
    assert pilot_key("B", None) == "B" and pilot_key("A", "MOF-303") == "A_MOF-303"
    rates = {1: 3e-4, 2: 5e-5, 4: 1e-5}                                  # steps per snapshot interval
    assert choose_h_steps(rates, {1: True, 2: True, 4: True}) == 2         # the largest h that passes
    assert choose_h_steps(rates, {1: True, 2: False, 4: True}) == 4
    assert choose_h_steps({1: 1.0, 2: 1.0, 4: 1.0}, {1: True, 2: True, 4: True}) is None


# ---- Critical 2: U-long needs its own iteration count ------------------------------------------------
def test_u_long_iterations_follow_s_wall_clock_and_are_required():
    from p3b.pilot.rules import iterations_ulong
    assert iterations_ulong(8000, wall_per_iter={"S": 2.0, "U": 0.5}) == 32000
    import train_chunk
    from p3b.train.cells import Cell
    pilot = _pilot_values()
    pilot.pop("iterations_ulong")
    with pytest.raises(KeyError, match="iterations_ulong"):
        train_chunk.cell_pilot(Cell("U-long", "A", "MOF-303", "full", 0), pilot)


def _pilot_values():
    return {"lr": {"U": 1e-3, "U-long": 1e-3, "S": 1e-3, "C+L": 1e-3}, "iterations": 2, "iterations_ulong": 4,
            "h_steps": {"A_MOF-303": 1, "B": 1}, "d": {"A_MOF-303": 4, "B": 4}, "K": {"A_MOF-303": 1},
            "thresholds": {"A_MOF-303": {"violation_N": 1e-5, "violation_E": 1e-5, "tau_S": 1e-6, "kappa": 1e-3,
                                         "defect_bound": 1e-3, "tau_ref": 1e-3}},
            "widths": {}}


# ---- Critical 3: weights are saved; evaluation per named set; test sets gated on H3a ------------------
POINT_A = dict(RH_ads=0.40, T_ads=303.15, T_des=353.15, v=0.10, alpha=0.8, beta=0.6, index=0, n_cycles=1)


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    from p3b.generate import generate_trajectory
    d = tmp_path_factory.mktemp("st")
    for split, pts in (("train", [dict(POINT_A, index=i, RH_ads=0.3 + 0.05 * i) for i in range(2)]),
                       ("validation", [dict(POINT_A, index=5)]), ("test", [dict(POINT_A, index=9)])):
        (d / split).mkdir()
        for pt in pts:
            assert generate_trajectory("A", pt, "MOF-303", N_z=32, N_c=8, path=d / split / f"t{pt['index']}.npz",
                                       n_snap=20).ok
    return d


QUICK = dict(ae_epochs=1, ae_width=8, width=8, n_windows=1, n_steps=2, n_cycles=1, pool_size=2)


def _cp():
    return {"lr": {"U": 1e-3}, "iterations": 1, "h_steps": 1, "d": 4, "kappa": 1e-3,
            "floors": {"N": 1e-6, "E": 1e-6}, "tau_S": 1e-6}


def test_training_saves_weights_and_evaluates_only_validation(study, tmp_path):
    from p3b.train.cells import Cell, train_cell
    out = train_cell(Cell("U", "A", "MOF-303", "full", 0), str(study), str(tmp_path), _cp(), quick=QUICK)
    res = json.loads(out.read_text())
    assert {r["set"] for r in res["rows"]} == {"validation"}
    w = res["meta"]["weights"]
    assert os.path.exists(os.path.join(tmp_path, w["file"])) and len(w["sha256"]) == 64
    assert res["meta"]["autoencoder"]["sha256"]


def test_test_set_evaluation_requires_the_h3a_gate(study, tmp_path):
    from p3b.train.cells import Cell, evaluate_cell, train_cell
    c = Cell("U", "A", "MOF-303", "full", 0)
    train_cell(c, str(study), str(tmp_path), _cp(), quick=QUICK)
    with pytest.raises(PermissionError, match="H3a"):
        evaluate_cell(c, "test", str(study), str(tmp_path), _cp(), gate={"A_MOF-303": "defect"}, quick=QUICK)
    out = evaluate_cell(c, "test", str(study), str(tmp_path), _cp(), gate={"A_MOF-303": "pass"}, quick=QUICK)
    rows = json.loads(out.read_text())["rows"]
    assert {r["set"] for r in rows} == {"test"} and rows[0]["design_index"] == 9


# ---- Important 9: a missing shared autoencoder is an error ---------------------------------------
def test_missing_autoencoder_raises_outside_quick(study, tmp_path):
    from p3b.train.cells import Cell, train_cell
    with pytest.raises(FileNotFoundError, match="autoencoder"):
        train_cell(Cell("U", "A", "MOF-303", "full", 1), str(study), str(tmp_path), _cp(),
                   quick=dict(QUICK, require_ae=True))


# ---- Important 15: training subsets are validated -------------------------------------------------
def test_training_subset_is_checked_against_the_pool(study, tmp_path):
    from p3b.train.cells import Cell, train_cell
    with pytest.raises(ValueError, match="pool"):
        train_cell(Cell("U", "A", "MOF-303", "full", 0), str(study), str(tmp_path), _cp(),
                   quick=dict(QUICK, pool_size=256))


# ---- Important 8: parameter matching is wired --------------------------------------------------------
def test_widths_match_s_within_five_percent():
    from p3b.latent.arms_unstructured import count_params
    from p3b.train.cells import matched_widths
    w = matched_widths(d=32, k=4, system="A", n_cells=128, n_ch=3, n_out=2)
    assert set(w) >= {"U", "U+F", "P", "FNO", "S"}
    target = w["_S_params"]
    from p3b.latent.arms_unstructured import UDynamics
    assert abs(count_params(UDynamics(32, 4, w["U"])) - target) <= 0.05 * target


# ---- Important 5: tau_S from the reference data, never from S ----------------------------------------
def test_tau_s_comes_from_reference_noise_and_raises_on_empty():
    from p3b.pilot.rules import thresholds
    th = thresholds(floors={"N": 1e-6, "E": 1e-6}, ref_production=np.array([0.5, -1e-6, 2.0, -3e-6]),
                    g2=np.ones(10), defects_per_step=np.zeros(5), tau_ref=1e-3)
    assert th["tau_S"] == pytest.approx(10 * np.percentile([1e-6, 3e-6], 99.9))   # negative part only
    with pytest.raises(ValueError):
        thresholds(floors={"N": 1, "E": 1}, ref_production=np.array([]), g2=np.ones(3),
                   defects_per_step=np.zeros(2), tau_ref=1.0)


# ---- Important 6 and 7: H4 power, conservative seed sharing, one power number ----------------------
def test_h4_tost_power_and_margin():
    from p3b.pilot.power import simulate_tost
    p_small = simulate_tost(0.0, sd_traj=0.3, sd_seed=0.1, m=1.10, reps=200, seed=0)
    p_big = simulate_tost(0.0, sd_traj=0.3, sd_seed=0.1, m=1.20, reps=200, seed=0)
    assert 0 <= p_small <= p_big <= 1


def test_power_simulation_defaults_and_definition():
    import inspect
    from p3b.pilot import power
    assert inspect.signature(power.simulate).parameters["seed_shared"].default == 1.0
    res = power.simulate(np.log(0.8), 0.3, 0.1, reps=20, draws=100, seed=1, alpha_h=0.05)
    assert res["power"] == pytest.approx(res["SUPPORTED"] + res["SMALL"])
    from p3b.pilot.rules import upper80_sd
    with pytest.raises(ValueError, match="seed"):
        upper80_sd(np.zeros((10, 1)))


# ---- Important 10: only divergence ends a rollout; other errors surface ----------------------------
def test_rollout_does_not_swallow_programming_errors(study):
    from torch import nn

    from p3b.eval.rollout import rollout
    from p3b.latent.data import Standardiser, load_set
    from p3b.train.trainer import Arm
    ds = load_set(sorted((study / "train").glob("*.npz")))

    class Bug(nn.Module):
        def forward(self, z, feat):
            raise RuntimeError("shape mismatch")

    class I(nn.Module):
        def encode(self, x):
            return x

        def decode(self, z):
            return z

    with pytest.raises(RuntimeError, match="shape"):
        rollout(Arm("U", Bug(), I(), Standardiser.fit(ds.X, 8, 3, 2, features=ds.features), None), ds, 0, 1, 1)


# ---- Important 12: both skip counts recorded ----------------------------------------------------------
def test_meta_records_both_skip_counts(study, tmp_path):
    from p3b.train.cells import Cell, train_cell
    res = json.loads(train_cell(Cell("U", "A", "MOF-303", "full", 0), str(study), str(tmp_path), _cp(),
                                quick=QUICK).read_text())
    assert {"train_skipped", "train_skipped_grad"} <= set(res["meta"])


# ---- Important 13: the ceiling model includes the sensitivities and fixed costs -----------------------
def test_ceiling_counts_sensitivities_and_fixed_costs():
    from p3b.train.cells import cells, plan_reductions, sensitivity_cells
    cs = cells() + sensitivity_cells()
    cost = {c: 1.0 for c in cs}
    n_sens = len(sensitivity_cells())
    steps = plan_reductions(spent_h=700.0 - len(cs) + 1, remaining=cs, cost_h=cost, ceiling_h=700.0)
    assert steps == ["degraded-data sensitivity"]                       # step 1 frees enough
    steps = plan_reductions(spent_h=0.0, remaining=cs, cost_h=cost, ceiling_h=len(cs) - 1, fixed_h=n_sens)
    assert len(steps) >= 1


# ---- Important 14: training checkpoints survive a session limit -----------------------------------------
def test_training_resumes_from_a_checkpoint(study, tmp_path):
    from torch import nn

    from p3b.latent.arms_unstructured import UDynamics
    from p3b.latent.data import Standardiser, load_set
    from p3b.train.trainer import Arm, TrainConfig, train
    ds = load_set(sorted((study / "train").glob("*.npz")))
    std = Standardiser.fit(ds.X, 8, 3, 2, features=ds.features)

    class I(nn.Module):
        def encode(self, x):
            return x

        def decode(self, z):
            return z

    def run(ck, stop_after=None):
        torch.manual_seed(0)
        dyn = UDynamics(26, 4, 8, time_scale=100.0)
        nn.init.normal_(dyn.net[-1].weight, std=0.01)
        h = train(Arm("U", dyn, I(), std, None), ds,
                  TrainConfig(lr=1e-3, iterations=4, seed=0, n_windows=1, n_steps=2, checkpoint=ck, every=1,
                              stop_after=stop_after))
        return dyn, h

    full, _ = run(None)
    ck = str(tmp_path / "ck.pt")
    run(ck, stop_after=2)                                               # a session limit after 2 iterations
    resumed, h = run(ck)
    assert h["resumed_from"] == 2
    assert all(torch.allclose(a, b) for a, b in zip(full.state_dict().values(), resumed.state_dict().values()))
