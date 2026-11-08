"""System A reference cycles: temperature-swing adsorption on the consistent column.

One trajectory is a sequence of cycles at one design point (PREREG_P3B_draft.md, Section 4):
adsorption for alpha*t_st at (c_in, T_ads) with the wall at T_ads, then desorption for
beta*alpha*t_st with the same absolute humidity c_in heated to T_des, wall at T_des. The bed
starts pre-wetted: c_0 = 1e-4 c_in in equilibrium at T_ads.

The column state (n, a, u) per cell is augmented with the cumulative wall heat of each cell and
four port integrals per unit cross-section, integrated in the same BDF solve so the integrated
closure is limited only by the solver tolerance. The N_PORTS reported ports are:
    0 water in, 1 water out (mol m^-2), 2 energy in, 3 energy out by convection, 4 heat to the wall (J m^-2).
They are reset to zero at the start of each trajectory and are cumulative over it.

Each half-cycle stores 100 snapshots uniformly on [start, end), so a cycle has 200 snapshots and
snapshot 0 of a cycle is its start state. A solver failure is returned as data, never raised.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import bmat, csr_matrix, diags, lil_matrix

from p3b import materials as mat
from p3b.column import Column

C_W = 33.6           # J mol^-1 K^-1, molar heat capacity of water in both phases
RTOL, ATOL = 1e-6, 1e-10
PREWET = 1e-4
N_PORTS = 5
PORT_NAMES = ("water_in", "water_out", "energy_in", "energy_out", "wall_heat")


@dataclass
class TrajectoryResult:
    ok: bool
    failed_cycle: int | None
    message: str
    cpu_s: float
    closure: list[float] = field(default_factory=list)
    final_ports: list[float] | None = None   # the ports at the end of the last cycle


def cycle_schedule(point: dict, iso, phys) -> list[tuple[str, float, tuple[float, float, float]]]:
    """[(phase, duration_s, (c_in, T_in, T_wall))] for the two half-cycles of one cycle."""
    T_ads, T_des = float(point["T_ads"]), float(point["T_des"])
    c_in = mat.c_in_from_rh(point["RH_ads"], T_ads)
    t_a = float(point["alpha"]) * mat.t_stoich(iso, phys, c_in, T_ads)
    return [("ads", t_a, (c_in, T_ads, T_ads)),
            ("des", float(point["beta"]) * t_a, (c_in, T_des, T_des))]


def setup_a(point: dict, material: str, N_z: int):
    """(Column with the material's isotherm and Glueckauf k_LDF at the feed, cycle schedule)."""
    iso = mat.load_isotherm(material)
    phys = mat.bed_physics(point["v"], point["T_ads"])
    schedule = cycle_schedule(point, iso, phys)
    phys.k_LDF = mat.k_ldf(iso, phys, schedule[0][2][0], float(point["T_ads"]))
    return Column(phys, N_z, C_W, iso=iso), schedule


def prewet_state(col: Column, c_in: float, T: float) -> np.ndarray:
    c0 = np.full(col.N, PREWET * c_in)
    Tv = np.full(col.N, T)
    n = col.eps * c0
    a = col.A * col.q_star(c0, Tv)
    return np.concatenate([n, a, col.energy(n, a, Tv)])


def augmented_sparsity(N: int) -> csr_matrix:
    """Jacobian pattern of the augmented state [n, a, u, wall heat per cell (N), 4 port integrals].

    The wall heat is integrated per cell (each row depends on its own cell only), not as one
    total, so no row is dense and colored finite differences stay cheap.
    """
    n3 = 3 * N
    S = lil_matrix((4 * N + 4, 4 * N + 4))
    band = diags([1.0, 1.0, 1.0], [-1, 0, 1], shape=(N, N))
    S[:n3, :n3] = bmat([[band, band, band]] * 3)
    eye = diags([1.0], [0], shape=(N, N))
    S[n3:4 * N, :n3] = bmat([[eye, eye, eye]])
    P = 4 * N
    S[P + 1, N - 1] = 1.0                               # water out: c_N = n_N / eps
    for j in (N - 1, 2 * N - 1, 3 * N - 1):
        S[P + 3, j] = 1.0                               # energy out: c_N, T_N
    return csr_matrix(S)


class _Augmented:
    """Column right-hand side plus cumulative wall heat per cell and the port integrals."""

    def __init__(self, col: Column):
        self.col = col
        self.n3 = 3 * col.N
        self.P = 4 * col.N
        self.sparsity = augmented_sparsity(col.N)

    def rhs(self, t, y, inputs):
        col = self.col
        c_in, T_in, T_w = inputs
        f = col.fluxes(y[:self.n3], inputs)
        T, c = f["T"], f["c"]
        v = col.p.v
        ports = [v * c_in, v * c[-1], v * col.stream_energy(c_in, T_in), v * col.stream_energy(c[-1], T[-1])]
        return np.concatenate([col.rhs(t, y[:self.n3], inputs), f["W"] * col.dz, ports])

    def ports(self, Y) -> np.ndarray:
        """(n, N_PORTS) cumulative ports from augmented states Y (n, 4N+4); wall = sum of cells."""
        return np.column_stack([Y[:, self.P:self.P + 4], Y[:, self.n3:self.P].sum(axis=1)])

    def totals(self, y) -> tuple[float, float]:
        n, a, u = self.col.split(y[:self.n3])
        return float(np.sum(n + a) * self.col.dz), float(np.sum(u) * self.col.dz)


def half_cycle_closure(aug: _Augmented, y0, y1) -> float:
    """Max over water and energy of |change in content - net port inflow| / scale."""
    N0, E0 = aug.totals(y0)
    N1, E1 = aug.totals(y1)
    dp = (aug.ports(y1[None, :]) - aug.ports(y0[None, :]))[0]
    rw = abs((N1 - N0) - (dp[0] - dp[1])) / max(abs(dp[0]), abs(dp[1]), abs(N1 - N0), 1e-300)
    re = abs((E1 - E0) - (dp[2] - dp[3] - dp[4])) / max(abs(dp[2]), abs(dp[3]), abs(dp[4]), abs(E1 - E0), 1e-300)
    return float(max(rw, re))


def simulate_trajectory_a(point: dict, material: str, N_z: int, n_snap: int = 200,
                          on_cycle: Callable | None = None) -> TrajectoryResult:
    """Run point["n_cycles"] cycles; call on_cycle(k, t, y_fine, ports) once per cycle.

    t: (n_snap,) absolute times (s); y_fine: (n_snap, 4 N_z) float64: the column states (n, a, u)
    and the cumulative wall heat of each cell (J m^-2); ports: (n_snap, N_PORTS) cumulative.
    """
    if n_snap % 2:
        raise ValueError("n_snap must be even (half per half-cycle)")
    t_cpu = time.process_time()
    closure: list[float] = []
    k = 0
    try:
        col, schedule = setup_a(point, material, N_z)
        c_in = schedule[0][2][0]
        aug = _Augmented(col)
        y = np.concatenate([prewet_state(col, c_in, float(point["T_ads"])), np.zeros(N_z + 4)])
        t_abs = 0.0
        half = n_snap // 2
        for k in range(int(point["n_cycles"])):
            ts, ys = [], []
            for phase, dur, inputs in schedule:
                if not (np.isfinite(dur) and dur > 0):
                    raise ValueError(f"non-positive or non-finite {phase} duration {dur}")
                t_eval = np.linspace(0.0, dur, half + 1)
                sol = solve_ivp(aug.rhs, (0.0, dur), y, method="BDF", args=(inputs,), t_eval=t_eval,
                                jac_sparsity=aug.sparsity, rtol=RTOL, atol=ATOL)
                if not sol.success or sol.y.shape[1] != half + 1 or not np.all(np.isfinite(sol.y)):
                    raise RuntimeError(f"cycle {k} {phase}: {sol.message}")
                closure.append(half_cycle_closure(aug, y, sol.y[:, -1]))
                ts.append(t_abs + sol.t[:-1])
                ys.append(sol.y[:, :-1].T)
                y = sol.y[:, -1]
                t_abs += dur
            if on_cycle is not None:
                Y = np.vstack(ys)
                on_cycle(k, np.concatenate(ts), Y[:, :aug.P], aug.ports(Y))
    except Exception as exc:  # recorded, never raised: the generator logs and moves on
        return TrajectoryResult(False, k, f"{type(exc).__name__}: {exc}",
                                time.process_time() - t_cpu, closure)
    return TrajectoryResult(True, None, "ok", time.process_time() - t_cpu, closure,
                            aug.ports(y[None, :])[0].tolist())
