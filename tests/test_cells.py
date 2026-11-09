"""Plan 4, Task 4: the study cells, job names, the compute ledger and one end-to-end cell run."""
import json
import re

import pytest

from p3b.train.cells import (REDUCTION_ORDER, Cell, apply_reductions, cells, job_name, plan_reductions,
                             run_cell)

POINT_A = dict(RH_ads=0.40, T_ads=303.15, T_des=353.15, v=0.10, alpha=0.8, beta=0.6, index=0, n_cycles=2)


def test_cell_counts_match_the_prereg_design_table():
    cs = cells()
    count = lambda arm, **kw: sum(1 for c in cs if c.arm == arm and all(getattr(c, k) == v for k, v in kw.items()))  # noqa: E731
    assert count("U", system="A", material="MOF-303", size="full") == 5
    assert count("U", system="A", material="MOF-303", size=16) == 3
    assert count("U", system="A", material="Al-fumarate", size=32) == 3
    assert count("U", system="B", size="full") == 5
    assert count("U+F") == 5 + 5 + 3 + 3 + 5 + 5
    assert count("U-long") == 3 and count("S-M") == 5 and count("S-Pi") == 5
    assert count("S-L") == 5 + 5 + 3 + 3
    assert count("P") == 5 + 5 + 9 + 3 + 3
    assert count("FNO") == 6 and count("C+L") == 6 and count("C") == 3 and count("MUSCL-C") == 1
    assert count("U+proj") == 0 and count("U+F+proj") == 0                    # evaluated with U / U+F
    u_full = [c for c in cs if c.arm == "U" and c.system == "A" and c.material == "MOF-303" and c.size == "full"]
    assert all("U+proj" in c.evaluate_also for c in u_full)
    assert all("U+F+proj" in c.evaluate_also for c in cs if c.arm == "U+F")


def test_job_names_are_stable_unique_and_kaggle_safe():
    names = [job_name(c) for c in cells()]
    assert len(names) == len(set(names))
    assert all(re.fullmatch(r"[a-z0-9-]{5,50}", n) for n in names)
    assert job_name(Cell("S-Pi", "A", "MOF-303", "full", 2)) == "p3b-spi-a-m303-full-s2"


def test_reductions_apply_in_order_only_while_over_the_ceiling():
    cs = cells()
    cost = {c: 1.0 for c in cs}
    steps = plan_reductions(spent_h=600.0, remaining=cs, cost_h=cost, ceiling_h=700.0)
    assert steps == REDUCTION_ORDER[:len(steps)] and len(steps) >= 1
    kept = apply_reductions(cs, steps)
    assert 600.0 + sum(cost[c] for c in kept) <= 700.0 or steps == REDUCTION_ORDER
    assert plan_reductions(spent_h=0.0, remaining=cs, cost_h={c: 0.1 for c in cs}, ceiling_h=700.0) == []
    core = lambda c: c.arm in ("U", "U+F", "S", "C", "C+L") and c.size in ("full", 32)  # noqa: E731
    assert sum(map(core, apply_reductions(cs, REDUCTION_ORDER))) == sum(map(core, cs))   # never reduced
    assert all(c.seed < 2 for c in apply_reductions(cs, REDUCTION_ORDER) if c.size in (16, 128))


def test_rollout_records_a_projection_failure_instead_of_raising(tmp_path):
    import torch
    from p3b.eval.rollout import rollout

    class Boom:
        kind, std = "U+proj", None

        def step(self, *a, **k):
            raise torch.linalg.LinAlgError("non-finite")

    from p3b.generate import generate_trajectory
    from p3b.latent.data import Standardiser, load_set
    p = tmp_path / "A.npz"
    assert generate_trajectory("A", POINT_A, "MOF-303", N_z=32, N_c=8, path=p, n_snap=40).ok
    ds = load_set([p])
    b = Boom()
    b.std = Standardiser.fit(ds.X, 8, 3, 2, features=ds.features)

    class AE:
        encode = staticmethod(lambda x: x)
        decode = staticmethod(lambda z: z)
    b.ae = AE()
    r = rollout(b, ds, 0, n_cycles=1, h_steps=1)
    assert r.failed_cycle == 1


def test_one_tiny_cell_runs_end_to_end(tmp_path):
    from p3b.generate import generate_trajectory
    data = tmp_path / "data"
    for split, pts in (("train", [POINT_A, dict(POINT_A, RH_ads=0.3, index=1)]), ("test", [dict(POINT_A, index=2)])):
        (data / split).mkdir(parents=True)
        for pt in pts:
            assert generate_trajectory("A", pt, "MOF-303", N_z=32, N_c=8, path=data / split / f"{pt['index']}.npz",
                                       n_snap=40).ok
    pilot = {"lr": {"U": 1e-3}, "iterations": 2, "h_steps": 1, "d": 6, "kappa": 1e-3,
             "floors": {"N": 1e-6, "E": 1e-6}, "tau_S": 1e-6}
    cell = Cell("U", "A", "MOF-303", "full", 0, ("U+proj",))
    out = run_cell(cell, str(data), str(tmp_path / "out"), pilot, quick=dict(ae_epochs=2, ae_width=16, width=16,
                                                                            n_windows=2, n_steps=3, n_cycles=1))
    res = json.loads(out.read_text())
    assert res["meta"]["arm"] == "U" and res["meta"]["params"] > 0
    assert {r["arm"] for r in res["rows"]} == {"U", "U+proj"} and len(res["rows"]) == 2
    assert (tmp_path / "out" / "ae_A_MOF-303_full_s0.pt").exists()            # the shared autoencoder
