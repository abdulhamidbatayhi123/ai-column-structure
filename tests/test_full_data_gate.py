"""The full-data decoder gate sets d (PREREG 4: "the smallest in {16, 24, 32, 48, 64} that passes the
decoder gate (Section 5.1) at full data"; 5.1: "All seeds of the full-data autoencoders must meet" it).
Plan: docs/superpowers/plans/2026-11-11-p3b-full-data-decoder-gate.md, Task 1."""
from p3b.latent.autoencoder import full_data_choice

P, F = {"passed": True}, {"passed": False}


def test_a_d_passes_only_when_every_seed_passes():
    assert full_data_choice({"mlp": {16: [P, F, P], 24: [P, P, P]}}) == ("mlp", 24)


def test_long_validation_enters_when_recorded():
    m = {"mlp": {16: [{"passed": True, "passed_long": False}], 24: [{"passed": True, "passed_long": True}]}}
    assert full_data_choice(m) == ("mlp", 24)


def test_mlp_before_conv_and_conv_as_the_fallback():
    assert full_data_choice({"mlp": {32: [P]}, "conv": {16: [P]}}) == ("mlp", 32)
    assert full_data_choice({"mlp": {16: [F], 32: [F]}, "conv": {16: [F], 24: [P]}}) == ("conv", 24)


def test_not_testable_when_nothing_passes_or_nothing_ran():
    assert full_data_choice({"mlp": {16: [F]}, "conv": {16: [F]}}) == (None, None)
    assert full_data_choice({"mlp": {16: []}}) == (None, None)            # no seed scored is no pass
    assert full_data_choice({}) == (None, None)


def test_string_keys_from_json_are_accepted():
    assert full_data_choice({"mlp": {"48": [P], "16": [F]}}) == ("mlp", 48)


# ---- Tasks 2 and 3: gate jobs, the gate run, the choice file ------------------------------------------
import json  # noqa: E402

import pytest  # noqa: E402

from p3b.train.cells import D_GRID, LEARNED_WITH_AE, cells, gate_specs  # noqa: E402


def _full_seeds(system, material):
    return sorted({c.seed for c in cells() if c.size == "full" and c.arm in LEARNED_WITH_AE
                   and (c.system, c.material) == (system, material)})


def test_gate_specs_cover_grid_seeds_and_keys_with_unique_names():
    sp = gate_specs(cells(), "mlp")
    names = [s["name"] for s in sp]
    assert len(names) == len(set(names))
    keys = {(c.system, c.material) for c in cells() if c.size == "full" and c.arm in LEARNED_WITH_AE}
    for sysmat in keys:
        mine = [s for s in sp if (s["system"], s["material"]) == sysmat]
        assert sorted({s["d"] for s in mine}) == list(D_GRID)
        assert sorted({s["seed"] for s in mine}) == _full_seeds(*sysmat)
        assert len(mine) == len(D_GRID) * len(_full_seeds(*sysmat))
    assert all(s["kind"] == "mlp" for s in sp)
    assert {s["name"] for s in gate_specs(cells(), "conv")}.isdisjoint(names)


def test_gate_jobs_one_per_spec_filtered_by_key():
    from kaggle.train_jobs import gate_jobs
    j = gate_jobs(cells(), "conv", keys=("B",))["jobs"]
    sp = [s for s in gate_specs(cells(), "conv") if s["system"] == "B"]
    assert [x["name"] for x in j] == [s["name"] for s in sp]
    assert all(x["expect"] == [f"gate_{x['name']}.json"] and x["threads"] == 4 for x in j)
    assert j[0]["commands"] == [["kaggle/train_chunk.py", "--gate-ae", j[0]["name"], "--out", "OUT"]]


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    from p3b.generate import generate_trajectory
    pa = dict(RH_ads=0.40, T_ads=303.15, T_des=353.15, v=0.10, alpha=0.8, beta=0.6, index=0, n_cycles=1)
    d = tmp_path_factory.mktemp("study")
    for split, pts in (("train", [pa, dict(pa, RH_ads=0.3, index=1)]), ("validation", [dict(pa, index=2)]),
                       ("long_validation", [dict(pa, index=3, n_cycles=2)])):
        (d / split).mkdir(parents=True)
        for pt in pts:
            assert generate_trajectory("A", pt, "MOF-303", N_z=32, N_c=8, path=d / split / f"{pt['index']}.npz",
                                       n_snap=40).ok
    return d


def _spec(kind="mlp", d=4):
    return {"name": f"p3b-gate-a-m303-{kind}-d{d}-s0", "system": "A", "material": "MOF-303", "kind": kind,
            "d": d, "seed": 0}


def test_gate_ae_scores_validation_and_long_validation_only_after_cycle_6(study, tmp_path):
    from p3b.train.cells import gate_ae
    q = {"ae_epochs": 1, "ae_width": 8, "pool_size": 2, "long_from": 1}
    r = json.loads(open(gate_ae(_spec(), str(study), str(tmp_path), css_cycle=7, q=q)).read())
    assert r["gate"]["passed"] in (True, False) and "balance_N" in r["gate"]
    assert r["gate_long"] is not None and r["passed_long"] == r["gate_long"]["passed"]
    assert r["passed"] == r["gate"]["passed"] and r["kind"] == "mlp" and r["d"] == 4
    r6 = json.loads(open(gate_ae(_spec(d=5), str(study), str(tmp_path), css_cycle=6, q=q)).read())
    assert r6["gate_long"] is None and "passed_long" not in r6


def _gate_file(dirp, spec, passed):
    p = dirp / f"gate_{spec['name']}.json"
    p.write_text(json.dumps({**spec, "key": "A_MOF-303", "passed": passed, "gate": {"passed": passed}}))


def _load_choice_script():
    import importlib.util
    s = importlib.util.spec_from_file_location("decoder_choice", "scripts/decoder_choice.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def test_decoder_choice_needs_every_seed_and_records_the_choice(tmp_path):
    dc = _load_choice_script()
    specs = [s for s in gate_specs(cells(), "mlp") if s["system"] == "A" and s["material"] == "MOF-303"]
    for s in specs:
        _gate_file(tmp_path, s, passed=s["d"] >= 32)
    out = dc.main(str(tmp_path), str(tmp_path / "decoder_choice.json"), keys=("A_MOF-303",))
    assert out["A_MOF-303"] == {"decoder": "mlp", "d": 32}
    assert len(out["inputs_sha256"]) == len(specs) and "script_sha256" in out
    (tmp_path / f"gate_{specs[0]['name']}.json").unlink()          # a missing seed: refuse, do not guess
    with pytest.raises(FileNotFoundError):
        dc.main(str(tmp_path), str(tmp_path / "c2.json"), keys=("A_MOF-303",))


def test_decoder_choice_reports_keys_that_need_the_conv_wave_and_not_testable(tmp_path):
    dc = _load_choice_script()
    for s in gate_specs(cells(), "mlp"):
        if s["system"] == "B":
            _gate_file(tmp_path, s, passed=False)
    out = dc.main(str(tmp_path), str(tmp_path / "c.json"), keys=("B",))
    assert out["B"] == {"needs_conv": True}
    for s in gate_specs(cells(), "conv"):
        if s["system"] == "B":
            _gate_file(tmp_path, s, passed=False)
    out = dc.main(str(tmp_path), str(tmp_path / "c.json"), keys=("B",))
    assert out["B"] == {"not_testable": True}


def test_train_chunk_takes_d_and_decoder_from_the_choice_never_from_the_pilot(monkeypatch):
    import importlib.util
    import sys
    monkeypatch.setattr(sys, "path", list(sys.path))
    s = importlib.util.spec_from_file_location("train_chunk", "kaggle/train_chunk.py")
    tc = importlib.util.module_from_spec(s)
    s.loader.exec_module(tc)
    c = next(c for c in cells() if c.arm == "S" and c.system == "A" and c.material == "MOF-303")
    pilot = {"lr": {"S": 1e-3}, "iterations": 10, "h_steps": {"A_MOF-303": 1}, "K": {},
             "thresholds": {"A_MOF-303": {"kappa": 1.0, "tau_S": 1e-6}}}
    p = tc.cell_pilot(c, pilot, {"A_MOF-303": {"decoder": "conv", "d": 48}})
    assert p["d"] == 48 and p["decoder"] == "conv"
    with pytest.raises(KeyError, match="decoder_choice"):
        tc.cell_pilot(c, pilot, {})
    with pytest.raises(RuntimeError, match="not testable"):
        tc.cell_pilot(c, pilot, {"A_MOF-303": {"not_testable": True}})
    with pytest.raises(RuntimeError, match="conv"):
        tc.cell_pilot(c, pilot, {"A_MOF-303": {"needs_conv": True}})
