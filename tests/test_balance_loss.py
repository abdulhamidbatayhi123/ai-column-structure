"""Freeze decision 1b (author, 2026-11-12): every autoencoder is trained on MSE + BALANCE_WEIGHT x (the
decoder gate's own balance errors)^2, so it is trained for what the gate checks (PREREG 5.1)."""
import glob
import os

import pytest
import torch

from p3b.generate import generate_trajectory
from p3b.latent.autoencoder import (BALANCE_WEIGHT, MLPAutoencoder, balance_errors, decoder_gate,
                                    fit_autoencoder, train_autoencoder)
from p3b.latent.data import Standardiser, load_set
from p3b.pilot.runs import _physics

POINT = dict(RH_ads=0.40, T_ads=303.15, T_des=353.15, v=0.10, alpha=0.8, beta=0.6, index=0, n_cycles=2)


@pytest.fixture(scope="module", params=["A", "B"])
def data(request, tmp_path_factory):
    d = tmp_path_factory.mktemp("bal" + request.param)
    paths = []
    for i in range(2):
        p = d / f"t{i}.npz"
        if request.param == "A":
            ok = generate_trajectory("A", dict(POINT, index=i, RH_ads=0.35 + 0.1 * i), "MOF-303", N_z=32, N_c=8,
                                     path=p, n_snap=20).ok
        else:
            pt = dict(T_hot=353.15, T_cold=298.15, v=2.0e-3 * (1 + 0.5 * i), alpha=0.7, beta=0.8, index=i, n_cycles=2)
            ok = generate_trajectory("B", pt, None, N_z=32, N_c=8, path=p, n_snap=20).ok
        assert ok
        paths.append(p)
    ds = load_set(paths)
    n_ch, n_out = (3, 2) if ds.system == "A" else (2, 1)
    std = Standardiser.fit(ds.X, ds.n_cells, n_ch, n_out, features=ds.features)
    return ds, std, _physics(ds)


def _ae(ds, seed=0):
    torch.manual_seed(seed)
    return MLPAutoencoder(ds.X.shape[-1], 4, width=8)


def test_the_decision_weight_is_one():
    assert BALANCE_WEIGHT == 1.0


def test_weight_zero_is_the_plain_mse_training(data):
    ds, std, phys = data
    a, b = _ae(ds), _ae(ds)
    train_autoencoder(a, std.encode(ds.X), epochs=2, seed=0)
    fit_autoencoder(b, ds, std, phys, epochs=2, seed=0, weight=0.0)
    for x, y in zip(a.state_dict().values(), b.state_dict().values()):
        assert torch.equal(x, y)


def test_the_penalty_measures_the_gates_own_balance_errors(data):
    # the largest per-snapshot error the penalty sees equals the decoder gate's balance criterion
    ds, std, phys = data
    ae = _ae(ds)
    train_autoencoder(ae, std.encode(ds.X), epochs=1, seed=0)
    X = std.encode(ds.X).reshape(-1, ds.X.shape[-1])
    idx = torch.arange(X.shape[0])
    with torch.no_grad():
        eN, eE = balance_errors(ds, std, phys)(ae(X), idx)
    g = decoder_gate(ae, ds, std, phys)
    assert float(eE.abs().max()) == pytest.approx(g["balance_E"], rel=1e-9)
    if ds.system == "A":
        assert float(eN.abs().max()) == pytest.approx(g["balance_N"], rel=1e-9)
    else:
        assert eN is None and g["balance_N"] is None


def test_the_balance_term_changes_the_training(data):
    ds, std, phys = data
    a, b = _ae(ds), _ae(ds)
    fit_autoencoder(a, ds, std, phys, epochs=2, seed=0, weight=0.0)
    fit_autoencoder(b, ds, std, phys, epochs=2, seed=0)
    assert any(not torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


def test_every_autoencoder_is_trained_through_fit_autoencoder():
    # pilot pieces, the pilot gate grid, the full-data gate, the study's AE jobs, the quick fallback
    root = os.path.join(os.path.dirname(__file__), "..")
    srcs = [os.path.join(root, f) for f in ("p3b/pilot/pieces.py", "p3b/pilot/runs.py", "p3b/train/cells.py",
                                             "kaggle/train_chunk.py")]
    for s in srcs:
        text = open(s, encoding="utf-8").read()
        assert "train_autoencoder(" not in text, s
        assert "fit_autoencoder(" in text, s
