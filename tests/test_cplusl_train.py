"""Plan 3 review I6 (C+L part) and Plan 4 Task 4b: training C+L on the same coarse data, rolling it
out like C, and evaluating stored solver-arm files against the reference."""
import numpy as np
import pytest
import torch

from p3b.eval.evaluate import evaluate_files
from p3b.generate import generate_trajectory
from p3b.latent.data import load_set
from p3b.latent.physics import ColumnPhysics
from p3b.simulate_a import setup_a
from p3b.solvers.c_plus_l import Multipliers
from p3b.solvers.coarse import run_c
from p3b.train.cplusl import coarse_states, cplusl_loss, rollout_cplusl_to_file, train_cplusl

NC = 8
POINTS = [dict(RH_ads=0.40, T_ads=303.15, T_des=353.15, v=0.10, alpha=0.8, beta=0.6, index=0, n_cycles=1),
          dict(RH_ads=0.30, T_ads=298.15, T_des=343.15, v=0.15, alpha=0.9, beta=0.7, index=1, n_cycles=1)]


@pytest.fixture(scope="module")
def paths(tmp_path_factory):
    d = tmp_path_factory.mktemp("cl")
    out = []
    for pt in POINTS:
        p = d / f"A{pt['index']}.npz"
        assert generate_trajectory("A", pt, "MOF-303", N_z=32, N_c=NC, path=p, n_snap=20).ok
        out.append(p)
    return out


def test_coarse_states_are_block_averages_of_the_stored_conserved_blocks(paths):
    Y = coarse_states(paths, n_coarse=4)
    d = np.load(paths[0])
    cb = d["conserved_blocks"]                                        # (cycles, snaps, NC, 3)
    np.testing.assert_allclose(Y[0][0, 3, 0:4].numpy(), cb[0, 3, :, 0].reshape(4, 2).mean(1))
    assert Y[0].shape == (1, 20, 12)                                   # (cycles, snaps, 3 x 4)


def test_one_training_step_descends(paths):
    torch.manual_seed(0)
    mult = Multipliers(n_out=3)
    for net in mult.nets:
        torch.nn.init.normal_(net[-1].weight, std=0.3)
    ds = load_set(paths)
    rng = np.random.default_rng(3)
    loss0 = cplusl_loss(mult, ds, "MOF-303", n_coarse=4, n_windows=2, n_steps=2, K=1, rng=np.random.default_rng(3))
    h = train_cplusl(mult, ds, "MOF-303", n_coarse=4, lr=1e-4, iterations=1, seed=3, n_windows=2, n_steps=2, K=1)
    assert h["loss"][0] == pytest.approx(float(loss0), rel=1e-10)
    loss1 = cplusl_loss(mult, ds, "MOF-303", n_coarse=4, n_windows=2, n_steps=2, K=1, rng=np.random.default_rng(3))
    assert float(loss1) < float(loss0)


def test_unit_multipliers_rollout_matches_c_and_files_evaluate(paths, tmp_path):
    ds = load_set(paths[:1])
    phys = ColumnPhysics(setup_a(POINTS[0], "MOF-303", NC)[0])
    mult = Multipliers(n_out=3, constant=1.0)
    p_cl = tmp_path / "cl.npz"
    rollout_cplusl_to_file(mult, POINTS[0], "MOF-303", n_coarse=4, n_target=NC, n_snap=20, K=8, path=p_cl)
    p_c = tmp_path / "c.npz"
    assert run_c("A", POINTS[0], "MOF-303", N_c=4, path=p_c, n_snap=20, n_target=NC).ok
    rows = evaluate_files({"C+L": [p_cl], "C": [p_c]}, ds, phys, n_cycles=1, seed=0)
    by = {r["arm"]: r for r in rows}
    assert set(by) == {"C+L", "C"}
    # implicit Euler with 8 substeps against BDF at rtol 1e-6: close, not equal
    assert abs(by["C+L"]["W_pred"][0] - by["C"]["W_pred"][0]) < 0.05 * abs(by["C"]["W_pred"][0])
    assert by["C"]["violations_method"] == "trapezoid on own snapshots"


def test_train_cplusl_calls_back_at_each_checkpoint(paths):
    ds = load_set(paths)
    seen = []
    train_cplusl(Multipliers(n_out=3), ds, "MOF-303", n_coarse=4, lr=1e-3, iterations=4, seed=0, n_windows=1,
                 n_steps=2, every=2, callback=lambda it, mult, hist: seen.append(it))
    assert seen == [2, 4]


def test_evaluation_rollouts_run_without_autograd(paths, tmp_path, monkeypatch):
    # forward-only paths need no graph and, with modified Newton, no Jacobian at the converged point
    import torch

    from p3b.train import cplusl
    seen = []
    real = cplusl.implicit_euler_step

    def spy(*a, **k):
        seen.append(torch.is_grad_enabled())
        return real(*a, **k)
    monkeypatch.setattr(cplusl, "implicit_euler_step", spy)
    mult = Multipliers(n_out=3, constant=1.0)
    rollout_cplusl_to_file(mult, POINTS[0], "MOF-303", n_coarse=4, n_target=NC, n_snap=20, K=8, path=tmp_path / "c.npz")
    cplusl.cplusl_val_loss(mult, load_set(paths[:1]), "MOF-303", n_coarse=4, K=1)
    assert seen and not any(seen)
