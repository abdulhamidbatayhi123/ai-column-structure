"""Pre-freeze revision R4 (2026-11-23, after an adversarial review of the first R4): system B's outlet is carried as
BOTH end temperatures (the stored fine-cell T_f at z = 0 and z = L, `ends`), and the ports take the outflow end by
the interval's flow direction.

B reverses its flow at every half-cycle. The single outlet channel was the temperature at the current outflow end,
so it changed end across the last interval of each half-cycle: the residual over that interval mixed the old
direction's inputs with the other end's temperature (B's energy residual 99.9th percentile 255,000 with those
intervals, 280.5 without), and S's projection and P's penalty were driven by that wrong balance at every reversal
while U was not. Both end temperatures are continuous in time, so with them every residual is defined and nothing
has to be left out. At every snapshot, the end chosen by the snapshot's own direction is the old stored outlet."""
import glob
import os

import numpy as np
import pytest
import torch

from p3b.generate import generate_trajectory

PT = dict(T_hot=353.15, T_cold=298.15, v=2.0e-3, alpha=0.7, beta=0.8, index=0, n_cycles=3)


@pytest.fixture(scope="module")
def b_path(tmp_path_factory):
    p = tmp_path_factory.mktemp("r4ends") / "b.npz"
    assert generate_trajectory("B", PT, None, N_z=64, N_c=16, path=p, n_snap=40).ok
    return p


def test_layout_gives_two_outlet_channels_for_both_systems():
    from p3b.latent.data import layout
    assert layout("A") == (3, 2) and layout("B") == (2, 2)


def test_b_outlet_channels_are_the_stored_ends_and_the_outflow_end_is_the_old_outlet(b_path):
    from p3b.latent.data import load_set, outflow_temperature
    ds = load_set([b_path])
    d = np.load(b_path)
    X = ds.X[0].reshape(-1, ds.X.shape[-1])
    assert X.shape[-1] == ds.n_cells * 2 + 2
    assert torch.allclose(X[:, -2:], torch.tensor(d["ends"].reshape(-1, 2)))
    raw = ds.raw_inputs[0].reshape(X.shape[0], -1)
    old = torch.tensor(d["outlet"].reshape(-1), dtype=torch.float64)         # stored as float32; ends as float64
    assert torch.allclose(outflow_temperature(X[:, -2:], raw), old, rtol=1e-6, atol=0)


def test_store_ports_take_the_outflow_end_by_direction(b_path):
    from p3b.latent.data import load_set
    from p3b.pilot.runs import _physics
    ds = load_set([b_path])
    phys = _physics(ds)
    ch = ds.X[0, 0, :1, :ds.n_cells * 2].reshape(1, ds.n_cells, 2)
    out = torch.tensor([[300.0, 340.0]], dtype=torch.float64)            # T_f(z=0), T_f(z=L)
    fwd = phys.ports(ch, out, torch.tensor([[+2e-3, 353.15]], dtype=torch.float64))
    bwd = phys.ports(ch, out, torch.tensor([[-2e-3, 298.15]], dtype=torch.float64))
    same_fwd = phys.ports(ch, torch.tensor([[999.0, 340.0]], dtype=torch.float64),
                          torch.tensor([[+2e-3, 353.15]], dtype=torch.float64))
    same_bwd = phys.ports(ch, torch.tensor([[300.0, 999.0]], dtype=torch.float64),
                          torch.tensor([[-2e-3, 298.15]], dtype=torch.float64))
    assert torch.allclose(fwd[0], same_fwd[0]) and torch.allclose(bwd[0], same_bwd[0])   # the inflow end is unused
    assert not torch.allclose(fwd[0], phys.ports(ch, torch.tensor([[300.0, 330.0]], dtype=torch.float64),
                                                 torch.tensor([[+2e-3, 353.15]], dtype=torch.float64))[0])


def test_no_reversal_artifact_in_the_residuals(b_path):
    from p3b.eval.violations import trapezoid_residuals
    from p3b.latent.data import load_set
    from p3b.pilot.runs import _physics
    ds = load_set([b_path])
    phys = _physics(ds)
    X = ds.X[0].reshape(-1, ds.X.shape[-1])
    raw = ds.raw_inputs[0].reshape(X.shape[0], -1)
    t = ds.t[0].reshape(-1)
    r = trapezoid_residuals(X, raw, t, phys, "B", ds.n_cells)
    ns = ds.X.shape[2]
    pos = np.arange(X.shape[0] - 1) % ns
    rev = np.isin(pos, [ns // 2 - 1, ns - 1])
    for q in ("E", "S"):
        x = np.abs((r[q] / (t[1:] - t[:-1])).numpy())
        assert x[rev].max() < 3 * x[~rev].max(), q
    assert ((r["S"] / (t[1:] - t[:-1])).numpy()[rev] > -1e-9 * np.abs(r["S"].numpy()).max()).all()


def test_release_is_unchanged(b_path):
    from p3b.eval.metrics import release
    from p3b.latent.data import load_set
    from p3b.pilot.runs import _physics
    ds = load_set([b_path])
    phys = _physics(ds)
    d = np.load(b_path)
    old = torch.tensor(d["outlet"], dtype=torch.float64).reshape(ds.X.shape[1], ds.X.shape[2], 1).expand(-1, -1, 2)
    new = release(ds.X[0][..., -2:], ds.raw_inputs[0], ds.t[0], "B", phys.rhoc)
    assert torch.allclose(new, release(old, ds.raw_inputs[0], ds.t[0], "B", phys.rhoc), rtol=1e-5)   # float32 outlet


def test_the_gate_judges_the_outflow_end(b_path):
    from p3b.latent.autoencoder import decoder_gate
    from p3b.latent.data import load_set
    from p3b.pilot.runs import _physics
    ds = load_set([b_path])
    phys = _physics(ds)
    ns, nc = ds.X.shape[2], ds.n_cells
    v_first = float(ds.raw_inputs[0, 0, 0, 0])            # the direction of each cycle's first half
    v_row = torch.tensor([v_first] * (ns // 2) + [-v_first] * (ns // 2))

    class Identity:
        encode = decode = staticmethod(lambda x: x)

    class ShiftOneEnd(torch.nn.Module):                   # reconstruction exact, except one end channel + 5 K
        def __init__(self, shift_outflow: bool):
            super().__init__()
            self.shift_outflow = shift_outflow

        def encode(self, x):
            return x

        def decode(self, z):
            out = z.clone()
            outflow_col = torch.where(v_row > 0, 1, 0)       # v > 0: outflow at z = L (column 1)
            col = outflow_col if self.shift_outflow else 1 - outflow_col
            out[torch.arange(ns), nc * 2 + col] += 5.0
            return out

    assert decoder_gate(ShiftOneEnd(False), ds, Identity, phys)["outlet_T"] < 1e-12   # inflow end: not judged
    assert decoder_gate(ShiftOneEnd(True), ds, Identity, phys)["outlet_T"] > 1e-3
