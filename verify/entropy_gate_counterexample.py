"""The final code review's entropy-gate counter-example, through every reference gate (plan 1 fix pass).

System A, MOF-303, the fast corner the reviewer used (RH 0.4, T_ads 300 K, T_des 348 K, v 0.30 m/s,
alpha 1.2, beta 0.75), 2 cycles at N_z = 256, coarsened to 16 and 256 cells; and system B at the fast
corner of its box, N_z = 256 onto 32 cells. With the trapezoid outlet quadrature the entropy margin
was -0.017 at every coarsening; with exact entropy ports it must pass.

    python verify/entropy_gate_counterexample.py   -> verify/results_entropy_gate_counterexample.json
"""
import json
import os
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from p3b import gates  # noqa: E402
from p3b.generate import generate_trajectory  # noqa: E402
from p3b.provenance import git_commit  # noqa: E402

A = dict(RH_ads=0.40, T_ads=300.0, T_des=348.0, v=0.30, alpha=1.2, beta=0.75, index=0, n_cycles=2)
B = dict(T_hot=363.0, T_cold=293.0, v=3.0e-3, alpha=0.9, beta=1.0, index=0, n_cycles=2)
out = {"commit": git_commit(), "cases": []}
with tempfile.TemporaryDirectory() as tmp:
    for system, point, material, nz, nc in (("A", A, "MOF-303", 256, 16), ("A", A, "MOF-303", 256, 256),
                                            ("B", B, None, 256, 32)):
        path = os.path.join(tmp, f"{system}_{nc}.npz")
        res = generate_trajectory(system, point, material, nz, nc, path=path)
        rec = {"system": system, "material": material, "point": point, "N_z": nz, "N_c": nc, "ok": res.ok}
        if res.ok:
            eg = gates.entropy_gate(path, gates.model_from_meta(gates.load_meta(path)))
            rec.update(entropy_gate=eg, rate_identity_max=gates.rate_identity_max(path),
                       fine_production_min=gates.fine_production_min(path),
                       stored_closure=gates.stored_closure(path), closure_max=gates.load_meta(path)["closure_max"],
                       floor_share=gates.floor_share(path))
        out["cases"].append(rec)
        print(json.dumps(rec))
with open(os.path.join(ROOT, "verify", "results_entropy_gate_counterexample.json"), "w") as f:
    json.dump(out, f, indent=1)
