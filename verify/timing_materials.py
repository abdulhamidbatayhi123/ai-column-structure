"""CPU time of one temperature-swing cycle with the 3B material isotherms (three-term fits).

The steps of the real isotherms are steep (n about 9 and 19, docs/materials_3B.md), much
steeper than the n = 3 and 6 cases in timing_scaling.py, so the reference cost is measured
again here: 128, 256 and 512 cells, single thread, adsorption 0.8 t_st at 303 K and RH 0.40,
desorption 0.6 t_st at 353 K. The LDF coefficient follows from the pellet by Paper 1's
Glueckauf relation with the material's own isotherm. Writes verify/timing_materials.json.
"""
import json
import os
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "ai-mof-cof-dynamics")))
from consistent_column import Column  # noqa: E402
from isotherms import from_fit, p_sat, R  # noqa: E402
from gen_parametric_dataset import build_physics, D_M_WATER, TORTUOSITY, EPS_PELLET  # noqa: E402

FITS = json.load(open(os.path.join(HERE, "materials", "fit_isotherms.json")))
DH_C = {"MOF-303": -51.3e3, "Al-fumarate": -48.5e3}
BED = dict(q_max=24.5, delta_H=-52000.0, step_rh=0.12, isotherm_n=3.0, henry_fraction=0.17,
           d_p=3.25e-3, rho_p=1050.0, eps_t=0.39)   # bed and transport properties only
T_ADS, T_DES, RH = 303.15, 353.15, 0.40

out = []
for name in ("MOF-303", "Al-fumarate"):
    iso = from_fit(FITS[name], DH_C[name])
    p = build_physics(BED, dict(T_in=T_ADS, rh_feed=RH, v=0.10))
    c_in = RH * p_sat(T_ADS) / (R * T_ADS)
    q_in = float(iso.q_star(np.array([c_in]), np.array([T_ADS]))[0])
    K = q_in / c_in
    p.k_LDF = 15.0 * EPS_PELLET * (D_M_WATER / TORTUOSITY) / ((p.d_p / 2) ** 2 * p.rho_p * K)
    cap = (1 - p.eps_t) * p.rho_p * q_in + p.eps_t * c_in
    t_st = cap * p.L / (p.v * c_in)
    for N in (128, 256, 512):
        col = Column(p, N, 33.6, iso=iso)
        y = col.initial_state(T_ADS)
        rec = {"material": name, "N": N, "k_LDF": p.k_LDF, "t_st_h": t_st / 3600, "q_feed": q_in}
        for ph, dur, inp in (("ads", 0.8 * t_st, (c_in, T_ADS, T_ADS)), ("des", 0.6 * t_st, (c_in, T_DES, T_DES))):
            t0 = time.process_time()
            sol = solve_ivp(col.rhs, (0, dur), y, method="BDF", args=(inp,), t_eval=np.linspace(0, dur, 201),
                            jac_sparsity=col.sparsity(), rtol=1e-6, atol=1e-10)
            rec[ph + "_cpu_s"] = round(time.process_time() - t0, 2)
            rec[ph + "_nfev"] = int(sol.nfev)
            rec[ph + "_ok"] = bool(sol.success)
            y = sol.y[:, -1]
        out.append(rec)
        print(rec, flush=True)
json.dump(out, open(os.path.join(HERE, "timing_materials.json"), "w"), indent=1)
