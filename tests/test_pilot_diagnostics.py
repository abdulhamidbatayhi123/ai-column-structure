"""Plan 3 review: the pilot diagnostic of S's stiffness on encoded data."""
import numpy as np
import pytest
import torch
from torch import nn

from p3b.generate import generate_trajectory
from p3b.latent.data import Standardiser, load_set
from p3b.latent.functionals import LatentFunctionals
from p3b.latent.physics import ColumnPhysics
from p3b.pilot.diagnostics import s_stiffness
from p3b.simulate_a import setup_a

NC = 8
POINT_A = dict(RH_ads=0.40, T_ads=303.15, T_des=353.15, v=0.10, alpha=0.8, beta=0.6, index=0, n_cycles=1)


class Identity(nn.Module):
    def encode(self, x):
        return x

    def decode(self, z):
        return z


def test_stiffness_diagnostic_on_an_exact_autoencoder(tmp_path):
    p = tmp_path / "A.npz"
    assert generate_trajectory("A", POINT_A, "MOF-303", N_z=32, N_c=NC, path=p, n_snap=40).ok
    ds = load_set([p])
    std = Standardiser.fit(ds.X, NC, 3, 2, features=ds.features)
    lf = LatentFunctionals(Identity().decode, std, ColumnPhysics(setup_a(POINT_A, "MOF-303", NC)[0]), "A", NC)
    out = s_stiffness(lf, Identity(), std, ds, h_over_dt=1.0)
    assert 0.0 <= out["share_violating"] <= 1.0
    assert out["n"] == ds.X.shape[1] * ds.X.shape[2] - 1
    assert all(np.isfinite(out[k]) for k in ("correction_median", "correction_p99"))
    # an exact autoencoder of entropy-consistent data: the encoded motion meets S's inequality almost always
    assert out["share_violating"] < 0.2
