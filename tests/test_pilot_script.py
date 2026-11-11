"""Plan 4, Task 2: the blinded pilot script (PREREG Section 10)."""
import importlib.util
import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KEYS = {"lr", "iterations", "iterations_ulong", "h_steps", "thresholds", "K", "sds", "css", "decoder_gate_pilot",
        "positive_control", "script_sha256", "inputs_sha256"}


def _load():
    spec = importlib.util.spec_from_file_location("pilot_rules", os.path.join(ROOT, "scripts", "pilot_rules.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write(d, name, obj):
    with open(os.path.join(d, name), "w") as f:
        json.dump(obj, f)


def _inputs(d):
    rng = np.random.default_rng(0)
    lr = {"U": {"0.0003": 0.4123457, "0.001": 0.3123457, "0.003": 0.5123457},
          "S": {"0.0003": 0.2987654, "0.001": 0.3987654, "0.003": 0.6987654}}
    _write(d, "lr_runs.json", lr)
    _write(d, "curves.json", {"U": [[1000, 0.9111], [2000, 0.8111], [3000, 0.8011]],
                              "S": [[1000, 0.7222], [2000, 0.7122], [3000, 0.7102]]})
    _write(d, "h_check.json", {"A_MOF-303": {"1": {"viol_rate": 5e-5, "stiff_ok": True},
                                             "2": {"viol_rate": 1e-5, "stiff_ok": True}}})
    _write(d, "wall_per_iter.json", {"S": 2.0, "U": 0.5})
    _write(d, "threshold_inputs.json", {"A_MOF-303": {"floors": {"N": 1e-7, "E": 2e-7},
                                                      "ref_production": (-rng.exponential(1e-4, 100)).tolist(),
                                                      "g2": rng.lognormal(0, 1, 100).tolist(),
                                                      "defects_per_step": (rng.poisson(1, 50) / 1e4).tolist(),
                                                      "tau_ref": 1e-3}})
    _write(d, "cplusl_K.json", {"A_MOF-303": {"1": 0.03, "2": 0.008, "4": 0.004, "8": 0.002}})
    _write(d, "sds.json", {"H1": rng.normal(0, 0.3, (16, 3)).tolist(), "H2": rng.normal(0, 0.3, (16, 3)).tolist(),
                           "H5": rng.normal(0, 0.3, (16, 3)).tolist()})
    _write(d, "css.json", {"A_MOF-303": [4, 5, 7, 6], "B": [3, 3, 4]})
    _write(d, "decoder_gate.json", {"A_MOF-303": {"16": False, "24": True, "32": True}})
    _write(d, "positive_control.json", {"word": "SUPPORTED"})
    return lr


def test_output_has_exactly_the_section_10_keys(tmp_path, capsys):
    _inputs(tmp_path)
    out = tmp_path / "pilot_values.json"
    _load().main(str(tmp_path), str(out))
    v = json.loads(out.read_text())
    assert set(v) == KEYS
    assert v["lr"] == {"U": 0.001, "S": 0.0003} and v["iterations"] == 3000
    assert v["h_steps"] == {"A_MOF-303": 1} and v["K"] == {"A_MOF-303": 2}
    assert v["decoder_gate_pilot"]["A_MOF-303"]["working"] == ["mlp", 24]
    assert v["iterations_ulong"] == 4 * v["iterations"]
    assert v["css"] == {"A_MOF-303": 7, "B": 4}


def test_no_per_arm_loss_is_printed_or_written(tmp_path, capsys):
    lr = _inputs(tmp_path)
    out = tmp_path / "pilot_values.json"
    _load().main(str(tmp_path), str(out))
    printed = capsys.readouterr().out + out.read_text()
    losses = [v for by in lr.values() for v in by.values()] + [0.9111, 0.8111, 0.8011, 0.7222, 0.7122, 0.7102]
    for x in losses:
        assert repr(x) not in printed and f"{x:.4f}" not in printed


def test_rerun_is_identical(tmp_path):
    _inputs(tmp_path)
    m = _load()
    m.main(str(tmp_path), str(tmp_path / "a.json"))
    m.main(str(tmp_path), str(tmp_path / "b.json"))
    assert (tmp_path / "a.json").read_text() == (tmp_path / "b.json").read_text()
