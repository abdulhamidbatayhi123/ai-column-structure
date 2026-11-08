"""Plan 2, Task 3: autoencoder, decoder gate (PREREG 5.1) and the latent-dimension rule."""
import numpy as np
import pytest
import torch

from p3b.generate import generate_trajectory
from p3b.latent.autoencoder import (ConvDecoderAutoencoder, MLPAutoencoder, choose_d, cosine_lr,
                                    decoder_gate, train_autoencoder)
from p3b.latent.data import Standardiser, load_set
from p3b.latent.metrics import field_nrmse
from p3b.latent.physics import ColumnPhysics, StorePhysics
from p3b.simulate_a import setup_a
from p3b.simulate_b import T_AMB, make_store

POINT_A = dict(RH_ads=0.40, T_ads=303.15, T_des=353.15, v=0.10, alpha=0.8, beta=0.6, index=0, n_cycles=2)
POINT_B = dict(T_hot=353.15, T_cold=298.15, v=2.0e-3, alpha=0.7, beta=0.8, index=0, n_cycles=2)
NC, NS = 8, 40


def test_layer_sizes():
    a = MLPAutoencoder(386, 32)
    sizes = [m.weight.shape for m in a.modules() if isinstance(m, torch.nn.Linear)]
    assert sizes == [(256, 386), (256, 256), (32, 256), (256, 32), (256, 256), (386, 256)]
    b = MLPAutoencoder(257, 16)
    assert [m.weight.shape for m in b.modules() if isinstance(m, torch.nn.Linear)][-1] == (257, 256)
    assert all(p.dtype == torch.float64 for p in a.parameters())


def test_conv_decoder_shapes():
    m = ConvDecoderAutoencoder(n_cells=128, n_ch=3, n_out=2, d=24)
    x = torch.randn(5, 386, dtype=torch.float64)
    assert m.encode(x).shape == (5, 24) and m(x).shape == (5, 386)
    convs = [c for c in m.modules() if isinstance(c, (torch.nn.Conv1d, torch.nn.ConvTranspose1d))]
    assert len(convs) == 4 and all(c.out_channels == 32 for c in convs[:3])


def test_cosine_schedule():
    assert cosine_lr(0, 100, 1e-3, 1e-5) == pytest.approx(1e-3)
    assert cosine_lr(100, 100, 1e-3, 1e-5) == pytest.approx(1e-5)
    assert cosine_lr(50, 100, 1e-3, 1e-5) == pytest.approx(0.5 * (1e-3 + 1e-5))


def test_training_lowers_the_loss():
    torch.manual_seed(0)
    z = torch.randn(512, 3, dtype=torch.float64)
    X = torch.tanh(z @ torch.randn(3, 26, dtype=torch.float64))
    hist = train_autoencoder(MLPAutoencoder(26, 4), X, epochs=15, batch=64, seed=0)
    assert hist["loss"][-1] < 0.3 * hist["loss"][0]


def test_field_nrmse():
    rng = np.random.default_rng(0)
    true = torch.tensor(rng.normal(size=(4, NC * 3 + 2)), dtype=torch.float64)
    assert float(field_nrmse(true, true, NC, 3).max()) == 0.0
    ch = true[:, :NC * 3].reshape(4, NC, 3)
    rng_ = ch.amax(dim=(0, 1)) - ch.amin(dim=(0, 1))
    pred = true.clone()
    pred[:, :NC * 3] += (0.01 * rng_).repeat(NC)
    assert float(field_nrmse(pred, true, NC, 3)) == pytest.approx(0.01, rel=1e-12)


@pytest.fixture(scope="module", params=["A", "B"])
def gate_case(request, tmp_path_factory):
    d = tmp_path_factory.mktemp("gate")
    if request.param == "A":
        p = d / "A.npz"
        assert generate_trajectory("A", POINT_A, "MOF-303", N_z=32, N_c=NC, path=p, n_snap=NS).ok
        ds = load_set([p])
        phys = ColumnPhysics(setup_a(POINT_A, "MOF-303", NC)[0])
        std = Standardiser.fit(ds.X, NC, 3, 2)
    else:
        p = d / "B.npz"
        assert generate_trajectory("B", POINT_B, None, N_z=32, N_c=NC, path=p, n_snap=NS).ok
        ds = load_set([p])
        phys = StorePhysics(make_store(NC), T_AMB)
        std = Standardiser.fit(ds.X, NC, 2, 1)
    return ds, phys, std


class _Oracle(torch.nn.Module):
    def encode(self, x):
        return x

    def decode(self, z):
        return z


class _Constant(torch.nn.Module):
    def encode(self, x):
        return x[..., :1]

    def decode(self, z):
        return torch.zeros(z.shape[:-1] + (self.n,), dtype=torch.float64)


def test_gate_passes_exact_reconstruction(gate_case):
    ds, phys, std = gate_case
    g = decoder_gate(_Oracle(), ds, std, phys)
    assert g["passed"] and g["field_nrmse"] <= 1e-12
    assert g["balance_E"] <= 1e-4 and (g["balance_N"] is None or g["balance_N"] <= 1e-4)   # float32 storage


def test_gate_fails_a_constant_decoder(gate_case):
    ds, phys, std = gate_case
    m = _Constant()
    m.n = ds.X.shape[-1]
    g = decoder_gate(m, ds, std, phys)
    assert not g["passed"] and g["field_nrmse"] > 0.02


def test_choose_d():
    assert choose_d({16: False, 24: True, 32: True, 48: True, 64: True}) == 24
    assert choose_d({16: False, 24: False, 32: False, 48: False, 64: False}) is None
    assert choose_d({64: True, 16: True}) == 16
