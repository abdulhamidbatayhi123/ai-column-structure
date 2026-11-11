"""Plan 5 Task 6: the collector's stages on synthetic piece files (fast), and the job phases."""
import json

import pytest

from p3b.pilot import collect
from p3b.pilot.pieces import LR_ARMS, PRIMARY_KEY, _tag

KEY = PRIMARY_KEY


def _w(d, name, obj, stage=None):
    from p3b.pilot.pieces import _write
    _write(d, name, obj, stage=stage or {})


def _phase0(d, gate=None, K=None):
    _w(d, f"data_{KEY}.json", {"css": [3, 4], "decoder_gate": gate or {"16": False, "24": True, "32": True},
                               "reference_noise": {"floors": {"N": 1e-6, "E": 1e-7}, "tau_ref": 1e-9,
                                                   "ref_production": [-2e-6, 1e-5, -1e-7, 3e-6]}})
    _w(d, f"cplusl_K_{KEY}.json", K or {"1": 0.03, "2": 0.004, "4": 0.001, "8": 0.0})


def test_stage0_chooses_d_K_and_tau_S(tmp_path):
    from p3b.pilot.rules import tau_S_from
    _phase0(tmp_path)
    s = collect.stage(0, tmp_path, keys=(KEY,))
    assert s["d"] == {KEY: 24} and s["K"] == {KEY: 2}
    assert s["tau_S"][KEY] == tau_S_from([-2e-6, 1e-5, -1e-7, 3e-6])


def test_stage0_cannot_rank_boolean_gates_when_no_d_passes(tmp_path):
    _phase0(tmp_path, gate={"16": False, "24": False})
    with pytest.raises(ValueError):
        collect.stage(0, tmp_path, keys=(KEY,))


def test_stage1_to_3_add_lr_iterations_and_h(tmp_path):
    s0 = {"d": {KEY: 24}}
    for a in LR_ARMS:
        _w(tmp_path, f"lr_{_tag(a)}.json", {"0.0003": 2.0, "0.001": 1.0, "0.003": float("nan")}, s0)
    s = collect.stage(1, tmp_path, s0, keys=(KEY,))
    assert set(s["lr"]) == set(LR_ARMS) and all(v == 1e-3 for v in s["lr"].values()) and s["d"] == {KEY: 24}
    for a in LR_ARMS:
        _w(tmp_path, f"curve_{_tag(a)}.json", [[1000, 1.0], [2000, 0.5], [3000, 0.495]], s)
    s = collect.stage(2, tmp_path, s, keys=(KEY,))
    assert s["iterations"] == 3000
    _w(tmp_path, f"s_{KEY}.json", {"h_check": {"1": {"viol_rate": 0.0, "stiff_ok": True, "h_rho": 1.0},
                                               "2": {"viol_rate": 0.0, "stiff_ok": True, "h_rho": 0.5}},
                                   "g2": [1.0], "defects_per_step": {"1": [0.0], "2": [0.0]}}, s)
    s = collect.stage(3, tmp_path, s, keys=(KEY,))
    assert s["h_steps"] == {KEY: 1}


def test_stage_values_are_chosen_values_only(tmp_path):
    """Blinding: no loss or error enters a stage file."""
    _phase0(tmp_path)
    s = collect.stage(0, tmp_path, keys=(KEY,))
    assert set(s) == {"d", "decoder", "K", "tau_S"}


def test_phase_jobs_cover_every_piece_once():
    import importlib.util
    import pathlib
    p = pathlib.Path(__file__).resolve().parent.parent / "kaggle" / "pilot_jobs.py"
    spec = importlib.util.spec_from_file_location("pilot_jobs", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    names = [j["name"] for ph in range(5) for j in m.phase_jobs(ph)["jobs"]]
    assert len(names) == len(set(names)) == 4 + 13 + 9 + 4 + 6
    for ph in range(5):
        jobs = m.phase_jobs(ph)["jobs"]
        within = {j["name"] for j in jobs}
        assert all(d in within for j in jobs for d in j.get("depends_on", []))   # no dependency across phases
        assert all(j["threads"] == (4 if j["set"].startswith("ae-") else 1) and j["expect"] for j in jobs)


def test_collect_ae_copies_the_pilot_autoencoders(tmp_path):
    src = tmp_path / "pieces" / "kernel-a"
    src.mkdir(parents=True)
    (src / "ae_A_MOF-303_pilot_s0.pt").write_bytes(b"w")
    (src / "lr_U.json").write_text("{}")
    got = collect.collect_ae(tmp_path / "pieces", tmp_path / "ae")
    assert [p.name for p in got] == ["ae_A_MOF-303_pilot_s0.pt"] and (tmp_path / "ae" / "ae_A_MOF-303_pilot_s0.pt").exists()
