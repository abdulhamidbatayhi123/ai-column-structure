"""Plan 5 Task 6: an end-to-end dry run of the pilot on small generated data. Every piece runs through
kaggle/pilot_chunk.py's dispatcher into its own folder (as the queue downloads them), the collector
finds them by stage hash, assembles the ten inputs, and scripts/pilot_rules.py reads them. The data
are small (16 cells, 2 cycles) but at the study's 200 snapshots per cycle with d = 8, where S trains
and rolls out (at d = 4 and 40 snapshots it diverges). Small models cannot meet every real rule
(decoder gate, 2 % plateau, h check), so d, K, iterations and h are set by hand; stage 1 (the
learning rates) runs the real rule. The rules are tested in test_pilot_collect.py and
test_pilot_rules.py."""
import importlib.util
import pathlib

import pytest

from p3b.generate import generate_trajectory
from p3b.pilot import collect, rules
from p3b.pilot.pieces import LR_ARMS, PRIMARY_KEY, TWO_SEEDS, _tag

ROOT = pathlib.Path(__file__).resolve().parent.parent
KEY = PRIMARY_KEY
Q = {"d_grid": [8], "ae_epochs": 60, "ae_width": 64, "width": 16, "n_coarse": 16, "Ks": [1], "lr_grid": [1e-3, 3e-3],
     "lr_iterations": 2, "curve_iterations": 4, "curve_every": 2, "n_windows": 1, "n_steps": 2, "iterations": 2,
     "wall_iterations": 2, "wall_warmup": 1, "h_steps_grid": [1, 2]}
STAT = {"H1": ("eps", 2), "H5": ("eps", 1), "H2": ("e", 2), "H4": ("e", 2)}


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def gen(tmp_path_factory):
    d = tmp_path_factory.mktemp("gen")
    for split, n in (("train", 2), ("val", 2)):
        for i in range(n):
            pt = dict(RH_ads=0.35 + 0.05 * i, T_ads=303.15, T_des=353.15, v=0.10, alpha=0.8, beta=0.6, index=i,
                      n_cycles=2)
            assert generate_trajectory("A", pt, "MOF-303", N_z=64, N_c=16, path=d / f"A_primary_pilot_{split}_{i:04d}.npz",
                                       n_snap=200, extra_meta={"design_set": f"A_primary_pilot_{split}"}).ok
    return d


def test_pilot_dry_run_end_to_end(gen, tmp_path):
    chunk = _load("kaggle/pilot_chunk.py", "pilot_chunk")
    data, pieces, inputs = tmp_path / "data", tmp_path / "pieces", tmp_path / "inputs"
    assert collect.stage_data([gen], data) == {f"{KEY}/train": 2, f"{KEY}/validation": 2}

    def run(name, stage):                      # each piece in its own folder, like the queue's downloads
        return chunk.run_piece(name, str(data), stage, str(pieces / name / "out"), Q)

    run(f"data-{KEY}", {})
    run(f"cplk-{KEY}", {})
    noise = collect.load_piece(pieces, f"data_{KEY}.json", {})["reference_noise"]
    stage = {"d": {KEY: 8}, "K": {KEY: 1}, "tau_S": {KEY: rules.tau_S_from(noise["ref_production"])},
             "ae_input_root": str(pieces)}
    for s in TWO_SEEDS:
        run(f"ae-{KEY}-s{s}", stage)
    for a in LR_ARMS:
        run(f"lr-{_tag(a)}", stage)
    stage = {**collect.stage(1, pieces, stage, keys=(KEY,)), "ae_input_root": str(pieces)}   # the real rule
    for a in LR_ARMS:
        run(f"curve-{_tag(a)}", stage)
    stage["iterations"] = 2
    run(f"s-{KEY}", stage)
    run("wall", stage)
    stage["h_steps"] = {KEY: 1}
    for a in ("U", "S"):
        for s in TWO_SEEDS:
            run(f"two-{a}-s{s}", stage)
    for s in TWO_SEEDS:
        run(f"pcq-s{s}", stage)

    paths = collect.final(pieces, inputs, stage, keys=(KEY,), hyp_stat=STAT, draws=200)
    assert len(paths) == 11                     # ten Section 10 inputs + decoder_gate_conv.json (PREREG 4)
    pr = _load("scripts/pilot_rules.py", "pilot_rules")
    out = pr.main(str(inputs), str(tmp_path / "pilot_values.json"))
    # tiny models may leave a rule without an admissible value (None), which the rules report as such
    assert set(out["sds"]) == {"H1", "H2", "H4", "H5"} and set(out["K"]) == {KEY} and set(out["lr"]) == set(LR_ARMS)
    assert out["lr"]["U"] in (1e-3, 3e-3) and out["lr"]["S"] in (1e-3, 3e-3)       # S's runs finished
    assert out["positive_control"] in ("SUPPORTED", "SMALL", "REFUTED", "REFUTED (REVERSED)", "INCONCLUSIVE")
    assert set(out["inputs_sha256"]) == set(pr.FILES) | set(pr.OPTIONAL)   # the conv grid ran (no d passed)
    assert all(v > 0 for v in out["sds"]["H1"].values())
