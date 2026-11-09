"""Where the hybrid-scheme clip bites, with the flow-dependent axial conductivities (cs T3; plan 1 Task 7b).

Upwind convection carries a numerical conduction (rho C) v dz/2 (system A: rho C = rho_g C_pg + c c_w,
taken at the largest c of the training box; system B: rho_f c_f |v|). The hybrid scheme removes it from
the physical coefficient; where it exceeds the physical coefficient the clip at zero bites and the
excess stays. For each grid this script reports the share of the velocity box where it bites and the
largest ratio numerical/physical, for the thermal and (A) the mass equation.

    python verify/clip_shares.py      -> verify/clip_shares.json
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from p3b import materials as m  # noqa: E402
from p3b.design import BOXES  # noqa: E402
from p3b.provenance import git_commit  # noqa: E402
from p3b.simulate_a import C_W  # noqa: E402
from p3b.simulate_b import make_store  # noqa: E402

out = {"commit": git_commit(), "K_E0": m.K_E0, "DELTA_AX": m.DELTA_AX, "A": [], "B": []}
c_max = m.c_in_from_rh(BOXES["A"]["RH_ads"][1], BOXES["A"]["T_ads"][1])
out["A_c_max"] = c_max
vA = np.linspace(*BOXES["A"]["v"], 2701)
for N in (16, 32, 64, 128, 256, 512, 1024, 2048):
    rT, rM = [], []
    for v in vA:
        p = m.bed_physics(v, 303.15)
        dz = p.L / N
        rT.append((p.rho_g * p.C_pg + c_max * C_W) * v * dz / 2 / p.k_z)
        rM.append(v * dz / 2 / (p.eps_t * p.D_L))
    rT, rM = np.array(rT), np.array(rM)
    out["A"].append({"N": N, "thermal_clip_share": float(np.mean(rT > 1)), "thermal_max_ratio": float(rT.max()),
                     "mass_clip_share": float(np.mean(rM > 1)), "mass_max_ratio": float(rM.max())})
vB = np.linspace(*BOXES["B"]["v"], 2501)
for N in (16, 32, 64, 128, 256, 512, 1024):
    st = make_store(N)
    r = np.array([st.rhoc * v * st.dz / 2 / st.k_f_phys(v) for v in vB])
    out["B"].append({"N": N, "D_P_B": st.d_p, "thermal_clip_share": float(np.mean(r > 1)),
                     "thermal_max_ratio": float(r.max())})
with open(os.path.join(ROOT, "verify", "clip_shares.json"), "w") as f:
    json.dump(out, f, indent=1)
for sysname in ("A", "B"):
    for rec in out[sysname]:
        print(sysname, rec)
