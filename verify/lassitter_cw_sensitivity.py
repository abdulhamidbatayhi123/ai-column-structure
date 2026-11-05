"""Does the water's heat capacity (c_w) change the comparison with the only measured MOF-303 bed?

Lassitter et al. (2024) measured outlet humidity of a 6.35 mm MOF-303 bed (Paper 1 compares
its solver with their Fig. 10 in compare_lassitter.py). This script solves their primary
condition with Paper 1's non-isothermal coupling (h_w = 10, C_ps = 1000) using the consistent
column at c_w = 0 and at c_w = 33.6, and reports the largest difference in outlet humidity and
the largest temperature rise. Design verification only; nothing is fitted.
Writes verify/lassitter_cw_sensitivity.json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
P1 = os.path.abspath(os.path.join(HERE, "..", "..", "ai-mof-cof-dynamics"))
sys.path.insert(0, P1)
from consistent_column import Column  # noqa: E402
import compare_lassitter as cl  # noqa: E402  (Paper 1, read-only)

os.chdir(P1)  # compare_lassitter resolves its data paths relative to Paper 1
p = cl.physics(0.20, 3e-3, 10.0, 1000.0)
c_in = cl.rh_to_conc(cl.RH_IN, cl.T_K)
t_end = cl.HORIZON_MIN * 60.0
out = {"condition": "Lassitter 2024 primary (k 0.20 1/s, d_p 3 mm), h_w 10, C_ps 1000", "N": 400}
curves = {}
for cw in (0.0, 33.6):
    col = Column(p, 400, cw)
    t, y = col.integrate(col.initial_state(cl.T_K), (0.0, t_end), (c_in, cl.T_K, cl.T_K), 601,
                         rtol=1e-7, atol=1e-11)
    n, a, u = col.split(y)
    curves[cw] = (n[-1] / col.eps / c_in, col.temperature(n, a, u))
dc = np.max(np.abs(curves[0.0][0] - curves[33.6][0]))
out["max_abs_diff_outlet_c_over_cin"] = float(dc)
out["max_T_rise_K_cw0"] = float(np.max(curves[0.0][1]) - cl.T_K)
out["max_T_rise_K_cw33.6"] = float(np.max(curves[33.6][1]) - cl.T_K)
print(json.dumps(out, indent=2))
json.dump(out, open(os.path.join(HERE, "lassitter_cw_sensitivity.json"), "w"), indent=2)
