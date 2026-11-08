"""Coarse storage of reference trajectories (formulation_3B.md, Section 5.2; PREREG Section 4).

The fine solution is never stored. Per cycle, `Coarsener.add_cycle` keeps:
  * block averages of the conserved variables over N_z/N_c fine cells (float64, for the gates);
  * the model's input channels from those block averages (float32):
      system A (log c, log x, log T), with c and q floored at 1e-8 c_in and 1e-8 q_capacity
      before the log; system B (T_f, T_s);
  * the fine outlet state (float32): A (log c_out, log T_out) at z = L; B T_f at the current
    outlet (z = L when charging, z = 0 when discharging, from the direction port);
  * the exact cumulative wall heat of each block (sum of the fine per-cell integrals);
  * system B only, `ends`: the fine T_f at z = 0 and z = L (float64, for the gates: at a flow
    reversal the outlet of an interval is at the end set by that interval's direction);
  * the cumulative ports and the snapshot times.
`floored_cells` counts fine cell-snapshots below a floor (A: c or q), out of `checked_cells`.
"""
from __future__ import annotations

import json

import numpy as np

from p3b.provenance import git_commit
from p3b.thermal_store import T0 as T0_B

FLOOR = 1e-8


class Coarsener:
    def __init__(self, system: str, N_z: int, N_c: int = 128, model=None, c_in: float | None = None):
        if system not in ("A", "B"):
            raise ValueError(f"unknown system {system!r}")
        if N_c <= 0 or N_z % N_c:
            raise ValueError(f"N_z = {N_z} is not an integer multiple of N_c = {N_c}")
        if model is None or getattr(model, "N", N_z) != N_z:
            raise ValueError("model must be the fine Column / ThermalStore with N = N_z")
        if system == "A" and not (c_in and c_in > 0):
            raise ValueError("system A needs c_in for the concentration floor")
        self.system, self.N_z, self.N_c, self.model = system, N_z, N_c, model
        self.r = N_z // N_c
        self.nvar = 3 if system == "A" else 2
        if system == "A":
            self.c_floor = FLOOR * c_in
            self.q_floor = FLOOR * model.iso.q_capacity
        self.t, self.channels, self.outlet, self.wall, self.ports, self.conserved = [], [], [], [], [], []
        self.ends: list[np.ndarray] = []   # system B only: fine T_f at z = 0 and z = L, float64
        self.cycles: list[int] = []
        self.floored_cells = 0
        self.checked_cells = 0

    def _blocks(self, F):
        """(ns, N_z) -> (ns, N_c) block averages."""
        return F.reshape(F.shape[0], self.N_c, self.r).mean(axis=2)

    def add_cycle(self, cycle_index: int, t, y_fine, ports) -> None:
        y_fine = np.asarray(y_fine, dtype=float)
        N = self.N_z
        ns = y_fine.shape[0]
        if y_fine.shape[1] != (self.nvar + 1) * N:
            raise ValueError(f"y_fine has {y_fine.shape[1]} columns, expected {(self.nvar + 1) * N}")
        fine = [y_fine[:, j * N:(j + 1) * N] for j in range(self.nvar)]
        cons = np.stack([self._blocks(F) for F in fine], axis=-1)
        wall = y_fine[:, self.nvar * N:].reshape(ns, self.N_c, self.r).sum(axis=2)
        m = self.model
        if self.system == "A":
            n, a, u = fine
            low = (n / m.eps < self.c_floor) | (a / m.A < self.q_floor)
            self.floored_cells += int(low.sum())
            nb, ab, ub = cons[..., 0], cons[..., 1], cons[..., 2]
            Tb = m.temperature(nb, ab, ub)
            cb = np.maximum(nb / m.eps, self.c_floor)
            xb = m.iso.X(np.maximum(ab / m.A, self.q_floor))
            ch = np.stack([np.log(cb), np.log(xb), np.log(Tb)], axis=-1)
            T_out = m.temperature(n[:, -1], a[:, -1], u[:, -1])
            out = np.stack([np.log(np.maximum(n[:, -1] / m.eps, self.c_floor)), np.log(T_out)], axis=-1)
        else:
            Tf, _ = _temps_b(m, fine)
            ch = np.stack(_temps_b(m, [cons[..., 0], cons[..., 1]]), axis=-1)
            direction = np.asarray(ports)[:, 3]
            out = np.where(direction > 0, Tf[:, -1], Tf[:, 0])[:, None]
            self.ends.append(np.stack([Tf[:, 0], Tf[:, -1]], axis=-1))
        self.checked_cells += ns * N
        self.cycles.append(int(cycle_index))
        self.t.append(np.asarray(t, dtype=float).copy())
        self.channels.append(ch.astype(np.float32))
        self.outlet.append(out.astype(np.float32))
        self.wall.append(wall.astype(np.float32))
        self.ports.append(np.asarray(ports, dtype=float).copy())
        self.conserved.append(cons)

    def to_npz(self, path, meta: dict, ports_end=None) -> None:
        """Write the npz; ports_end (the ports at the end of the last cycle) completes the last cycle."""
        L = self.model.p.L if self.system == "A" else self.model.L
        meta = dict(meta, commit=git_commit(), N_z=self.N_z, N_c=self.N_c, system=self.system,
                    floor=FLOOR, cycles=self.cycles, L=float(L))
        np.savez_compressed(
            path, channels=np.stack(self.channels), outlet=np.stack(self.outlet),
            wall_heat_blocks=np.stack(self.wall), ports_cum=np.stack(self.ports),
            conserved_blocks=np.stack(self.conserved), t=np.stack(self.t),
            **({"ends": np.stack(self.ends)} if self.ends else {}),
            **({"ports_end": np.asarray(ports_end, dtype=float)} if ports_end is not None else {}),
            floored_cells=np.int64(self.floored_cells), checked_cells=np.int64(self.checked_cells),
            meta=np.array(json.dumps(meta, default=_jsonable)))


def _jsonable(x):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(f"not JSON serialisable: {type(x).__name__}")


def _temps_b(st, fields):
    """(u_f, u_s) arrays -> (T_f, T_s), ThermalStore's linear energy relation."""
    return T0_B + fields[0] / st.Cf, T0_B + fields[1] / st.Cs
