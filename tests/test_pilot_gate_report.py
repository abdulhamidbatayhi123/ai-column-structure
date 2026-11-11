"""The pilot's decoder gate is a risk report, not the choice of d (PREREG 10: the pilot sets "exactly the
following, and nothing else", and d is not in the list; PREREG 4: d is the smallest that passes the decoder
gate at full data).

Found on 2026-11-11 when the pilot gate failed at every d for all three keys: collect.stage(0) stopped the
pilot ("no admissible d"), although the pilot trains on 32 (MOF-303) or 6 trajectories and the gate that
sets d uses the 256-trajectory autoencoders. The pilot now records the gate's metrics and runs its own
phases at a working d by a fixed rule; d itself is set after the freeze by the full-data gate.
"""
import json

import pytest

from p3b.latent.autoencoder import GATE, gate_ratio, pilot_working_decoder
from p3b.pilot import collect
from p3b.pilot.pieces import PRIMARY_KEY

KEY = PRIMARY_KEY


def _m(field=0.01, oc=0.01, ot=0.01, bN=0.005, bE=0.005):
    passed = field <= GATE["field"] and oc <= GATE["outlet"] and ot <= GATE["outlet"] \
        and bN <= GATE["balance"] and bE <= GATE["balance"]
    return {"field_nrmse": field, "outlet_c": oc, "outlet_T": ot, "balance_N": bN, "balance_E": bE,
            "passed": passed}


def test_gate_ratio_is_the_worst_criterion_over_its_threshold():
    assert gate_ratio(_m(bN=0.1)) == pytest.approx(10.0)
    assert gate_ratio(_m(field=0.03)) == pytest.approx(1.5)
    assert gate_ratio({**_m(), "outlet_c": None, "balance_N": None}) == pytest.approx(0.5)   # system B


def test_working_decoder_is_the_smallest_passing_d_when_one_passes():
    assert pilot_working_decoder({"16": _m(bN=0.1), "24": _m(), "32": _m()}) == ("mlp", 24)
    assert pilot_working_decoder({"16": _m(bN=0.1)}, {"16": _m(bN=0.2), "32": _m()}) == ("conv", 32)
    assert pilot_working_decoder({"16": False, "24": True}) == ("mlp", 24)          # boolean-only gates


def test_working_decoder_is_the_mlp_d_nearest_to_passing_when_none_passes():
    mlp = {"16": _m(bN=0.10), "32": _m(bN=0.07), "64": _m(bN=0.08)}
    conv = {"16": _m(bN=0.05), "32": _m(bN=0.06)}          # nearer, but the conv decoder is only a fallback
    assert pilot_working_decoder(mlp, conv) == ("mlp", 32)
    assert pilot_working_decoder({"16": _m(bN=0.07), "32": _m(bN=0.07)}) == ("mlp", 16)     # tie: smaller d
    with pytest.raises(ValueError):
        pilot_working_decoder({"16": False})              # nothing to rank


def _phase0(d, gate, gate_conv=None):
    from p3b.pilot.pieces import _write
    v = {"css": [3, 4], "decoder_gate": gate,
         "reference_noise": {"floors": {"N": 1e-6, "E": 1e-7}, "tau_ref": 1e-9,
                             "ref_production": [-2e-6, 1e-5, -1e-7, 3e-6]}}
    if gate_conv is not None:
        v["decoder_gate_conv"] = gate_conv
    _write(d, f"data_{KEY}.json", v, stage={})
    _write(d, f"cplusl_K_{KEY}.json", {"1": 0.03, "2": 0.004, "4": 0.001, "8": 0.0}, stage={})


def test_stage0_continues_at_the_working_d_when_no_pilot_d_passes(tmp_path):
    _phase0(tmp_path, {"16": _m(bN=0.10), "32": _m(bN=0.07)}, gate_conv={"16": _m(bN=0.2), "32": _m(bN=0.3)})
    s = collect.stage(0, tmp_path, keys=(KEY,))
    assert s["d"] == {KEY: 32} and s["decoder"] == {KEY: "mlp"}


def test_piece_data_keeps_the_gate_metrics(tmp_path, monkeypatch):
    from p3b.pilot import pieces, runs
    calls = []

    def fake_grid(tr, va, d_grid, epochs, width, seed=0, kind="mlp"):
        calls.append(kind)
        return {d: _m(bN=0.1) for d in d_grid}
    monkeypatch.setattr(runs, "decoder_gate_grid", fake_grid)
    monkeypatch.setattr(runs, "css_cycles", lambda lf: [3])
    monkeypatch.setattr(runs, "reference_noise", lambda lf: {"ref_production": [0.0]})
    monkeypatch.setattr(pieces, "long_files", lambda root, key: [])
    monkeypatch.setattr(pieces, "files", lambda root, key, split: [])
    monkeypatch.setattr(pieces, "_data_used", lambda *a: {})
    v = json.loads(open(pieces.piece_data("root", KEY, str(tmp_path), {"d_grid": (16, 32)})).read())["value"]
    assert calls == ["mlp", "conv"]
    assert v["decoder_gate"]["32"]["balance_N"] == 0.1 and v["decoder_gate_conv"]["16"]["passed"] is False


def test_pilot_values_report_the_gate_and_set_no_d(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("pilot_rules", "scripts/pilot_rules.py")
    pr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pr)
    spec2 = importlib.util.spec_from_file_location("tps", "tests/test_pilot_script.py")
    tps = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(tps)
    tps._inputs(tmp_path)
    (tmp_path / "decoder_gate.json").write_text(json.dumps({KEY: {"16": _m(bN=0.1), "32": _m(bN=0.07)}}))
    (tmp_path / "decoder_gate_conv.json").write_text(json.dumps({KEY: {"16": _m(bN=0.2)}}))
    v = pr.main(str(tmp_path), str(tmp_path / "pilot_values.json"))
    assert "d" not in v and "decoder" not in v
    rep = v["decoder_gate_pilot"][KEY]
    assert rep["working"] == ["mlp", 32]
    assert rep["passed"] == {"mlp": {"16": False, "32": False}, "conv": {"16": False}}
    assert rep["worst_ratio"]["mlp"]["32"] == pytest.approx(7.0)
