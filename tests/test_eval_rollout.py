"""Plan 3, Task 4: long rollouts of the learned arms from the encoded initial state."""
import pytest
import torch
from torch import nn

from p3b.eval.rollout import rollout
from p3b.generate import generate_trajectory
from p3b.latent.data import Standardiser, load_set
from p3b.train.trainer import Arm

F64 = torch.float64
NC, NS = 8, 40
POINT_A = dict(RH_ads=0.40, T_ads=303.15, T_des=353.15, v=0.10, alpha=0.8, beta=0.6, index=0, n_cycles=2)


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    p = tmp_path_factory.mktemp("ro") / "A.npz"
    assert generate_trajectory("A", POINT_A, "MOF-303", N_z=32, N_c=NC, path=p, n_snap=NS).ok
    return load_set([p])


class Identity(nn.Module):
    def encode(self, x):
        return x

    def decode(self, z):
        return z


class Recorder(nn.Module):
    """dz/dt = 0, recording the features it is called with."""

    def __init__(self):
        super().__init__()
        self.seen = []
        self.p = nn.Parameter(torch.zeros(1, dtype=F64))

    def forward(self, z, feat):
        self.seen.append(feat.clone())
        return torch.zeros_like(z)


class Exploder(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1, dtype=F64))

    def forward(self, z, feat):
        return 1e300 * (z.abs() + 1.0)


def test_zero_dynamics_reproduce_the_initial_state(ds):
    std = Standardiser.fit(ds.X, NC, 3, 2, features=ds.features)
    arm = Arm("U", Recorder(), Identity(), std, None)
    r = rollout(arm, ds, 0, n_cycles=2, h_steps=1)
    assert r.X.shape == (2, NS, ds.X.shape[-1]) and r.X_end.shape == (ds.X.shape[-1],)
    assert torch.allclose(r.X, ds.X[0, 0, 0].expand_as(r.X), rtol=1e-12)
    assert bool(r.finite.all()) and r.failed_cycle is None and r.cpu_s >= 0


def test_inputs_switch_at_the_half_cycle(ds):
    std = Standardiser.fit(ds.X, NC, 3, 2, features=ds.features)
    rec = Recorder()
    rollout(Arm("U", rec, Identity(), std, None), ds, 0, n_cycles=1, h_steps=1)
    T_in = torch.stack([f[0, 1] for f in rec.seen[::4]])          # one RK4 stage per call, 4 per step
    assert T_in[NS // 2 - 1] != T_in[NS // 2]                      # adsorption -> desorption
    assert len(rec.seen) == 4 * NS                                 # every interval, the last one closing


def test_a_diverging_model_stops_and_reports(ds):
    std = Standardiser.fit(ds.X, NC, 3, 2, features=ds.features)
    r = rollout(Arm("U", Exploder(), Identity(), std, None), ds, 0, n_cycles=2, h_steps=1)
    assert r.failed_cycle == 1 and not bool(r.finite[0])
    assert torch.isnan(r.X[1]).all()                               # nothing after the failure
