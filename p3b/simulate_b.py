"""System B reference cycles: the two-temperature thermocline store with flow reversal.

One trajectory is a sequence of cycles at one design point (PREREG_P3B_draft.md, Section 4):
charge with hot fluid at T_hot entering at z = 0 (v > 0) for alpha*t_front, then discharge with
cold fluid at T_cold entering at z = L (v < 0, same |v|) for beta*alpha*t_front, where
t_front = L (C_f + C_s) / (rho_f c_f |v|). The bed starts uniform at T_cold. The wall loses heat
to an ambient bath at T_AMB.

As in system A (p3b.simulate_a), the state (u_f, u_s) per cell is augmented with the cumulative
wall heat of each cell and two port integrals, so the Jacobian stays banded and the integrated
closure is solver-limited. The N_PORTS reported ports, per unit cross-section, are:
    0 energy in, 1 energy out (advective, J m^-2), 2 heat to the wall (J m^-2), all cumulative
    over the trajectory; 3 flow direction at the snapshot (+1 charge, -1 discharge); 4 entropy in,
    5 entropy out with the stream (J K^-1 m^-2, cumulative; exact, for the entropy gate).
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import bmat, csr_matrix, diags, lil_matrix

from p3b.simulate_a import ATOL, RTOL, TrajectoryResult
from p3b.thermal_store import T0, ThermalStore

T_AMB = 293.15       # K, ambient bath of the wall loss (constant)
# Rock size for the flow-dependent fluid dispersion; provisional (about 5 cm is what the default
# h_v = 2e4 W m^-3 K^-1 implies for a rock bed; to be verified, cs task). The reference uses the
# hybrid scheme, as in system A.
D_P_B = 0.05


def make_store(N_z: int) -> ThermalStore:
    """The reference thermocline store: flow-dependent fluid dispersion, hybrid scheme."""
    return ThermalStore(N_z, d_p=D_P_B, compensate=True)
N_PORTS = 6
PORT_NAMES = ("energy_in", "energy_out", "wall_heat", "direction", "entropy_in", "entropy_out")


def cycle_schedule(point: dict, st: ThermalStore) -> list[tuple[str, float, tuple[float, float, float]]]:
    """[(phase, duration_s, (v, T_in, T_amb))] for the charge and the discharge of one cycle."""
    v = float(point["v"])
    t_front = st.L * (st.Cf + st.Cs) / (st.rhoc * v)
    t_c = float(point["alpha"]) * t_front
    return [("charge", t_c, (v, float(point["T_hot"]), T_AMB)),
            ("discharge", float(point["beta"]) * t_c, (-v, float(point["T_cold"]), T_AMB))]


def augmented_sparsity(N: int, band: int = 1) -> csr_matrix:
    """Jacobian pattern of [u_f, u_s, wall heat per cell (N), energy in, energy out, entropy in, entropy out]."""
    S = lil_matrix((3 * N + 4, 3 * N + 4))
    offs = list(range(-band, band + 1))
    blk = diags([1.0] * len(offs), offs, shape=(N, N))
    S[:2 * N, :2 * N] = bmat([[blk, blk], [blk, blk]])
    S[2 * N:3 * N, :N] = diags([1.0], [0], shape=(N, N))     # wall loss from the fluid of each cell
    S[3 * N + 1, 0] = 1.0                                     # energy out: T_f at z = 0 (discharge)
    S[3 * N + 1, N - 1] = 1.0                                 # or at z = L (charge)
    S[3 * N + 3, 0] = 1.0                                     # entropy out, likewise
    S[3 * N + 3, N - 1] = 1.0
    return csr_matrix(S)


class _Augmented:
    def __init__(self, st: ThermalStore):
        self.st = st
        N = st.N
        self.n2, self.P = 2 * N, 3 * N
        self.sparsity = augmented_sparsity(N, getattr(st, "BAND", 1))

    def rhs(self, t, y, inputs):
        st = self.st
        v, T_in, _ = inputs
        f = st.fluxes(y[:self.n2], inputs)
        T_out = f["Tf"][-1] if v >= 0 else f["Tf"][0]
        ports = [abs(v) * st.rhoc * (T_in - T0), abs(v) * st.rhoc * (T_out - T0),
                 abs(v) * st.stream_entropy(T_in), abs(v) * st.stream_entropy(T_out)]
        return np.concatenate([st.rhs(t, y[:self.n2], inputs), f["Wwall"] * st.dz, ports])

    def ports(self, Y, direction: float) -> np.ndarray:
        return np.column_stack([Y[:, self.P], Y[:, self.P + 1], Y[:, self.n2:self.P].sum(axis=1),
                                np.full(Y.shape[0], direction), Y[:, self.P + 2], Y[:, self.P + 3]])

    def energy(self, y) -> float:
        return float(np.sum(y[:self.n2]) * self.st.dz)


def half_cycle_closure(aug: _Augmented, y0, y1) -> float:
    dE = aug.energy(y1) - aug.energy(y0)
    d = (aug.ports(y1[None, :], 0.0) - aug.ports(y0[None, :], 0.0))[0]
    return float(abs(dE - (d[0] - d[1] - d[2])) / max(abs(d[0]), abs(d[1]), abs(d[2]), abs(dE), 1e-300))


def simulate_trajectory_b(point: dict, N_z: int, n_snap: int = 200,
                          on_cycle: Callable | None = None, store_factory=None) -> TrajectoryResult:
    """Run point["n_cycles"] cycles; call on_cycle(k, t, y_fine, ports) once per cycle.

    t: (n_snap,) absolute times (s); y_fine: (n_snap, 3 N_z) float64: u_f, u_s and the
    cumulative wall heat of each cell (J m^-2); ports: (n_snap, N_PORTS).
    """
    if n_snap % 2:
        raise ValueError("n_snap must be even (half per half-cycle)")
    t_cpu = time.process_time()
    closure: list[float] = []
    nfev: list[int] = []
    cyc_cpu: list[float] = []
    k = 0
    try:
        st = (store_factory or make_store)(N_z)
        aug = _Augmented(st)
        schedule = cycle_schedule(point, st)
        y = np.concatenate([st.initial_state(float(point["T_cold"])), np.zeros(N_z + 4)])
        t_abs = 0.0
        half = n_snap // 2
        for k in range(int(point["n_cycles"])):
            ts, ps, ys = [], [], []
            t_cyc, nf = time.process_time(), 0
            for phase, dur, inputs in schedule:
                if not (np.isfinite(dur) and dur > 0):
                    raise ValueError(f"non-positive or non-finite {phase} duration {dur}")
                sol = solve_ivp(aug.rhs, (0.0, dur), y, method="BDF", args=(inputs,),
                                t_eval=np.linspace(0.0, dur, half + 1), jac_sparsity=aug.sparsity,
                                rtol=RTOL, atol=ATOL)
                if not sol.success or sol.y.shape[1] != half + 1 or not np.all(np.isfinite(sol.y)):
                    raise RuntimeError(f"cycle {k} {phase}: {sol.message}")
                closure.append(half_cycle_closure(aug, y, sol.y[:, -1]))
                nf += int(sol.nfev)
                Y = sol.y[:, :-1].T
                ts.append(t_abs + sol.t[:-1])
                ys.append(Y[:, :aug.P])
                ps.append(aug.ports(Y, 1.0 if inputs[0] > 0 else -1.0))
                y = sol.y[:, -1]
                t_abs += dur
            nfev.append(nf)
            cyc_cpu.append(time.process_time() - t_cyc)
            if on_cycle is not None:
                on_cycle(k, np.concatenate(ts), np.vstack(ys), np.vstack(ps))
    except Exception as exc:  # recorded, never raised
        return TrajectoryResult(False, k, f"{type(exc).__name__}: {exc}", time.process_time() - t_cpu, closure,
                                None, nfev, cyc_cpu)
    return TrajectoryResult(True, None, "ok", time.process_time() - t_cpu, closure,
                            aug.ports(y[None, :], -1.0)[0].tolist(), nfev, cyc_cpu)
