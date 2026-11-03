"""Numerical checks of formulation_3B.md on small grids (design verification only).

V1  c_w = 0 reproduces Paper 1's solver_fd.py (Section 3.4 claim).
V2  a temperature-swing cycle with c_w = 33.6: water and energy rate identities, the
    entropy bookkeeping identity, and the sign of every production term (Section 4).
V3  concavity of the gas-stream entropy and of the cell entropy (Sections 3.3, 4.3).
V4  coarse-graining: do 2000->128-style block averages keep the balances and the
    entropy inequality, with port fluxes taken from the coarse cell or the fine outlet?

Prints one JSON document; writes it to verify/results_run_checks.json.
Nothing here is study data: one MOF-303-like material, one condition, small grids.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from consistent_column import Column  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "ai-mof-cof-dynamics")))
from gen_parametric_dataset import build_physics, rh_to_conc  # noqa: E402  (read-only)
from solver_fd import generate_breakthrough_data  # noqa: E402  (read-only)

MAT = dict(q_max=24.5, delta_H=-52000.0, step_rh=0.12, isotherm_n=3.0,
           henry_fraction=0.17, d_p=3.25e-3, rho_p=1050.0, eps_t=0.39)
COND = dict(T_in=303.15, rh_feed=0.40, v=0.10)
T_DES = 353.15
C_W = 33.6

out = {"material": MAT, "condition": COND, "T_des": T_DES, "c_w": C_W}
phys = build_physics(MAT, COND)
c_in = rh_to_conc(COND["rh_feed"], COND["T_in"])
t_st = phys.stoichiometric_time(c_in, COND["T_in"])
out["c_in"] = c_in
out["t_st_s"] = t_st
out["k_LDF"] = phys.k_LDF


def rel(x, scale):
    return float(np.max(np.abs(x)) / max(scale, 1e-300))


# ---------------------------------------------------------------- V1 Paper 1 limit
t0 = time.time()
N1 = 200
t_end = 1.0 * t_st
_, tp, yp = generate_breakthrough_data(phys, N_z=N1, t_final=t_end, c_in=c_in,
                                       n_snapshots=41, rtol=1e-9, atol=1e-13, verbose=False)
col0 = Column(phys, N1, c_w=0.0)
y0 = col0.initial_state(phys.T_w)
tm, ym = col0.integrate(y0, (0.0, t_end), (c_in, phys.T_in, phys.T_w), 41, rtol=1e-9, atol=1e-13)
n, a, u = col0.split(ym)
c_m, q_m, T_m = n / col0.eps, a / col0.A, col0.temperature(n, a, u)
c_p, q_p, T_p = yp[:N1], yp[N1:2 * N1], yp[2 * N1:]
out["V1_paper1_limit"] = {
    "N": N1, "horizon_t_st": 1.0,
    "max_rel_diff_c": rel(c_m - c_p, c_in),
    "max_rel_diff_q": rel(q_m - q_p, phys.q_max),
    "max_abs_diff_T_K": float(np.max(np.abs(T_m - T_p))),
    "T_rise_K": float(T_p.max() - phys.T_w),
    "wall_s": round(time.time() - t0, 1),
}

# ---------------------------------------------------------------- V2 cycle, c_w > 0
t0 = time.time()
N2 = 120
col = Column(phys, N2, c_w=C_W)
t_A, t_D = 0.8 * t_st, 0.6 * t_st
phases = [("ads", t_A, (c_in, COND["T_in"], COND["T_in"])),
          ("des", t_D, (c_in, T_DES, T_DES))] * 2
y = col.initial_state(COND["T_in"])
rate = {"water": 0.0, "energy": 0.0, "entropy_identity": 0.0}
min_sig = {"ldf": 0.0, "face": 0.0, "adv": 0.0, "wall": 0.0}
integ = []
for name, dur, inp in phases:
    ts, ys = col.integrate(y, (0.0, dur), inp, 401)
    Ntot, Etot, portN, portE = [], [], [], []
    for k in range(ys.shape[1]):
        b = col.budget(ys[:, k], inp)
        sN = abs(b["portN"]) + abs(b["dN"]) + 1e-30
        sE = abs(b["portE"]) + abs(b["dE"]) + 1e-30
        f_k = col.fluxes(ys[:, k], inp)
        d_k = col.rhs(0.0, ys[:, k], inp)
        dn_k, da_k, du_k = col.split(d_k)
        sS = np.sum(np.abs(col.mu_v_over_T(f_k["c"], f_k["T"]) * dn_k)
                    + np.abs(col.mu_a_over_T(f_k["q"], f_k["T"]) * da_k)
                    + np.abs(du_k / f_k["T"])) * col.dz + 1e-30   # sum |w . xdot|
        rate["water"] = max(rate["water"], abs(b["dN"] - b["portN"]) / sN)
        rate["energy"] = max(rate["energy"], abs(b["dE"] - b["portE"]) / sE)
        rate["entropy_identity"] = max(rate["entropy_identity"],
                                       abs(b["dS"] - b["portS"] - b["sigma_total"]) / sS)
        scale_sig = abs(b["sigma_total"]) / col.N / col.dz + 1e-30
        for key, arr in (("ldf", b["sig_ldf"]), ("face", b["sig_face"]),
                         ("adv", b["sig_adv"]), ("wall", b["sig_wall"])):
            min_sig[key] = min(min_sig[key], float(np.min(arr)) / scale_sig)
        nn, aa, uu = col.split(ys[:, k])
        Ntot.append(np.sum(nn + aa) * col.dz)
        Etot.append(np.sum(uu) * col.dz)
        portN.append(b["portN"])
        portE.append(b["portE"])
    Ntot, Etot = np.array(Ntot), np.array(Etot)
    gN = np.trapezoid(portN, ts)
    gE = np.trapezoid(portE, ts)
    integ.append({"phase": name,
                  "water_closure_rel": float(abs(Ntot[-1] - Ntot[0] - gN) / (abs(gN) + 1e-30)),
                  "energy_closure_rel": float(abs(Etot[-1] - Etot[0] - gE) / (abs(gE) + 1e-30)),
                  "water_change_mol_m2": float(Ntot[-1] - Ntot[0])})
    y = ys[:, -1]
out["V2_cycle"] = {
    "N": N2, "phases": [p[0] for p in phases], "t_A_over_t_st": 0.8, "t_D_over_t_st": 0.6,
    "max_rel_rate_residual": rate,
    "min_production_over_mean_production": min_sig,
    "integrated_closure_trapezoid_401_points": integ,
    "wall_s": round(time.time() - t0, 1),
}

# ---------------------------------------------------------------- V3 concavity
rng = np.random.default_rng(0)
h_fd = 1e-4


def hessian(f, x):
    """Hessian in relative coordinates x_i (1 + xi_i): D H D with D = diag(|x|). By Sylvester's
    law of inertia its eigenvalue signs are those of H, and the scaling removes the eight
    orders of magnitude between n, a and u."""
    d = len(x)
    H = np.zeros((d, d))
    step = 1e-3 * np.maximum(np.abs(x), 1e-12)
    for i in range(d):
        for j in range(d):
            ei = np.zeros(d)
            ej = np.zeros(d)
            ei[i] = step[i]
            ej[j] = step[j]
            H[i, j] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / 4e-6
    return H


worst_stream, worst_cell = -np.inf, -np.inf
for _ in range(200):
    cc = 10 ** rng.uniform(-3, 0.5)
    TT = rng.uniform(285, 390)
    gu = col.stream_energy(cc, TT)

    def s_stream(xv):
        c_, g_ = xv
        T_ = T0_ = None
        # invert g_u = rhoC (T - T0) + c e_w(T) for T (linear in T)
        T_ = (g_ + col.rhoC * 298.15 + c_ * col.c_w * 298.15) / (col.rhoC + c_ * col.c_w)
        return col.stream_entropy(c_, T_)

    H = hessian(s_stream, np.array([cc, gu]))
    ev = np.linalg.eigvalsh(0.5 * (H + H.T))
    worst_stream = max(worst_stream, ev.max() / np.abs(ev).max())

    qq = rng.uniform(0.02, 0.97) * phys.q_max
    nn_ = cc * col.eps
    aa_ = qq * col.A
    uu_ = col.energy(nn_, aa_, TT)
    H = hessian(lambda xv: col.cell_entropy(*xv), np.array([nn_, aa_, uu_]))
    ev = np.linalg.eigvalsh(0.5 * (H + H.T))
    worst_cell = max(worst_cell, ev.max() / np.abs(ev).max())
out["V3_concavity"] = {
    "samples": 200,
    "max_eig_over_max_abs_eig_stream_entropy": float(worst_stream),
    "max_eig_over_max_abs_eig_cell_entropy": float(worst_cell),
    "note": "largest eigenvalue / largest |eigenvalue| of the relative-coordinate Hessian; <= ~1e-4 means negative semidefinite up to finite-difference error",
}

# ---------------------------------------------------------------- V4 coarse-graining
t0 = time.time()
NF, NC = 400, 40
fine = Column(phys, NF, c_w=C_W)
coarse = Column(phys, NC, c_w=C_W)
blk = NF // NC
y = fine.initial_state(COND["T_in"])
res = {"sigma_c_min_over_scale_coarse_outlet": 0.0, "sigma_c_min_over_scale_fine_outlet": 0.0,
       "water_rate_rel_coarse_outlet": 0.0, "water_rate_rel_fine_outlet": 0.0,
       "energy_rate_rel_coarse_ports": 0.0, "energy_rate_rel_fine_ports": 0.0,
       "fraction_times_sigma_c_negative_fine_ports": 0.0}
neg, total = 0, 0
for name, dur, inp in phases[:2]:
    c_feed, T_feed, T_w = inp
    ts, ys = fine.integrate(y, (0.0, dur), inp, 301)
    for k in range(ys.shape[1]):
        yk = ys[:, k]
        dk = fine.rhs(0.0, yk, inp)
        n, a, u = fine.split(yk)
        dn, da, du = fine.split(dk)
        avg = lambda z: z.reshape(NC, blk).mean(axis=1)  # noqa: E731
        nc, ac, uc = avg(n), avg(a), avg(u)
        dnc, dac, duc = avg(dn), avg(da), avg(du)
        Tc = coarse.temperature(nc, ac, uc)
        cc_, qc = nc / coarse.eps, ac / coarse.A
        wn = -coarse.mu_v_over_T(cc_, Tc)
        wa = -coarse.mu_a_over_T(qc, Tc)
        wu = 1.0 / Tc
        dSc = np.sum(wn * dnc + wa * dac + wu * duc) * coarse.dz
        dNc = np.sum(dnc + dac) * coarse.dz
        dEc = np.sum(duc) * coarse.dz
        Tf = fine.temperature(n, a, u)
        cf = n / fine.eps
        v = phys.v
        sin = coarse.stream_entropy(c_feed, T_feed)
        ein = coarse.stream_energy(c_feed, T_feed)
        # (a) ports evaluated on the coarse state
        Wc = coarse.h_a * (Tc - T_w)
        phiS_a = v * (sin - coarse.stream_entropy(cc_[-1], Tc[-1])) - np.sum(Wc) / T_w * coarse.dz
        portN_a = v * (c_feed - cc_[-1])
        portE_a = v * (ein - coarse.stream_energy(cc_[-1], Tc[-1])) - np.sum(Wc) * coarse.dz
        # (b) ports evaluated on the fine outlet cell and the fine wall flux
        Wf = fine.h_a * (Tf - T_w)
        phiS_b = v * (sin - fine.stream_entropy(cf[-1], Tf[-1])) - np.sum(Wf) / T_w * fine.dz
        portN_b = v * (c_feed - cf[-1])
        portE_b = v * (ein - fine.stream_energy(cf[-1], Tf[-1])) - np.sum(Wf) * fine.dz
        scale = abs(dSc) + abs(phiS_b) + 1e-30
        sa = (dSc - phiS_a) / scale
        sb = (dSc - phiS_b) / scale
        res["sigma_c_min_over_scale_coarse_outlet"] = min(res["sigma_c_min_over_scale_coarse_outlet"], sa)
        res["sigma_c_min_over_scale_fine_outlet"] = min(res["sigma_c_min_over_scale_fine_outlet"], sb)
        sN = abs(dNc) + abs(portN_b) + 1e-30
        sE = abs(dEc) + abs(portE_b) + 1e-30
        res["water_rate_rel_coarse_outlet"] = max(res["water_rate_rel_coarse_outlet"], abs(dNc - portN_a) / sN)
        res["water_rate_rel_fine_outlet"] = max(res["water_rate_rel_fine_outlet"], abs(dNc - portN_b) / sN)
        res["energy_rate_rel_coarse_ports"] = max(res["energy_rate_rel_coarse_ports"], abs(dEc - portE_a) / sE)
        res["energy_rate_rel_fine_ports"] = max(res["energy_rate_rel_fine_ports"], abs(dEc - portE_b) / sE)
        total += 1
        neg += sb < 0
    y = ys[:, -1]
res["fraction_times_sigma_c_negative_fine_ports"] = neg / total
res.update({"N_fine": NF, "N_coarse": NC, "snapshots": total, "wall_s": round(time.time() - t0, 1)})
out["V4_coarse_graining"] = res

txt = json.dumps(out, indent=2, default=float)
print(txt)
with open(os.path.join(HERE, "results_run_checks.json"), "w") as fh:
    fh.write(txt)
