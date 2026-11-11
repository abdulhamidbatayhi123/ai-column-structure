"""The decoder fallback (PREREG Section 4, latent dimension d): "If none passes, a fixed 1-D convolutional
decoder (4 layers, 32 channels) is tried on the same grid before 'not testable' is declared."

Found missing on 2026-11-10 when the pilot's MLP decoder gate failed for MOF-303 at every d: the pilot and
the study built only the MLP autoencoder. The choice is recorded per system and material ("decoder") and
every autoencoder of that key is built with it.
"""
import json

import pytest
import torch

from p3b.latent.autoencoder import ConvDecoderAutoencoder, MLPAutoencoder, make_autoencoder
from p3b.pilot import collect
from p3b.pilot.pieces import PRIMARY_KEY

KEY = PRIMARY_KEY


def test_make_autoencoder_builds_the_recorded_kind():
    a = make_autoencoder("mlp", n_cells=16, n_ch=3, n_out=2, d=8)
    c = make_autoencoder("conv", n_cells=16, n_ch=3, n_out=2, d=8)
    assert isinstance(a, MLPAutoencoder) and isinstance(c, ConvDecoderAutoencoder)
    x = torch.randn(5, 16 * 3 + 2, dtype=torch.float64)
    for m in (a, c):
        assert m.decode(m.encode(x)).shape == x.shape
    with pytest.raises(ValueError):
        make_autoencoder("transformer", n_cells=16, n_ch=3, n_out=2, d=8)


def _w(d, name, obj):
    from p3b.pilot.pieces import _write
    _write(d, name, obj, stage={})


def _phase0(d, gate, gate_conv=None):
    v = {"css": [3, 4], "decoder_gate": gate,
         "reference_noise": {"floors": {"N": 1e-6, "E": 1e-7}, "tau_ref": 1e-9,
                             "ref_production": [-2e-6, 1e-5, -1e-7, 3e-6]}}
    if gate_conv is not None:
        v["decoder_gate_conv"] = gate_conv
    _w(d, f"data_{KEY}.json", v)
    _w(d, f"cplusl_K_{KEY}.json", {"1": 0.03, "2": 0.004, "4": 0.001, "8": 0.0})


def test_stage0_keeps_the_mlp_when_a_d_passes(tmp_path):
    _phase0(tmp_path, {"16": False, "24": True})
    s = collect.stage(0, tmp_path, keys=(KEY,))
    assert s["d"] == {KEY: 24} and s["decoder"] == {KEY: "mlp"}


def test_stage0_falls_back_to_the_conv_decoder_on_the_same_grid(tmp_path):
    _phase0(tmp_path, {"16": False, "24": False}, gate_conv={"16": False, "24": True})
    s = collect.stage(0, tmp_path, keys=(KEY,))
    assert s["d"] == {KEY: 24} and s["decoder"] == {KEY: "conv"}


def test_stage0_without_gate_metrics_cannot_rank_when_neither_passes(tmp_path):
    _phase0(tmp_path, {"16": False}, gate_conv={"16": False})
    with pytest.raises(ValueError, match="no metrics"):            # metric gates: tests/test_pilot_gate_report.py
        collect.stage(0, tmp_path, keys=(KEY,))


def test_the_decoder_is_a_phase_0_value_that_later_phases_inherit():
    from p3b.pilot.pieces import restrict, stage_sha
    s = {"d": {KEY: 24}, "decoder": {KEY: "conv"}, "K": {KEY: 2}, "tau_S": {KEY: 1e-6}}
    assert restrict(s, 1)["decoder"] == {KEY: "conv"}
    assert stage_sha(s) != stage_sha({**s, "decoder": {KEY: "mlp"}})


def test_piece_data_runs_the_conv_grid_only_when_no_mlp_d_passes(tmp_path, monkeypatch):
    from p3b.pilot import pieces, runs
    calls = []

    def fake_grid(tr, va, d_grid, epochs, width, seed=0, kind="mlp"):
        calls.append(kind)
        return {d: {"passed": kind == "conv" and d == 32} for d in d_grid}
    monkeypatch.setattr(runs, "decoder_gate_grid", fake_grid)
    monkeypatch.setattr(runs, "css_cycles", lambda lf: [3])
    monkeypatch.setattr(runs, "reference_noise", lambda lf: {"ref_production": [0.0]})
    monkeypatch.setattr(pieces, "long_files", lambda root, key: [])
    monkeypatch.setattr(pieces, "files", lambda root, key, split: [])
    monkeypatch.setattr(pieces, "_data_used", lambda *a: {})
    p = pieces.piece_data("root", KEY, str(tmp_path), {})
    v = json.loads(open(p).read())["value"]
    assert calls == ["mlp", "conv"]
    assert v["decoder_gate_conv"]["32"]["passed"] is True and not any(g["passed"] for g in v["decoder_gate"].values())


def test_study_cells_build_the_pilot_decoder(tmp_path):
    from p3b.train.cells import _autoencoder_for
    ae = _autoencoder_for({"decoder": "conv"}, n_state=16 * 3 + 2, d=8, n_cells=16, system="A", q={})
    assert isinstance(ae, ConvDecoderAutoencoder)
    ae = _autoencoder_for({}, n_state=16 * 3 + 2, d=8, n_cells=16, system="A", q={})
    assert isinstance(ae, MLPAutoencoder)                       # default: the pre-registered MLP
