"""Solver arms C and MUSCL-C (PREREG Section 3; Plan 2, Task 9).

C: the reference simulator itself (hybrid scheme, rtol 1e-6, atol 1e-10) on 16, 32, 64 or 128
cells. Its states are mapped to the 128 target cells by piecewise-constant injection of the
conserved variables (densities repeated; the extensive per-cell wall heat split equally) and
then stored by the same Coarsener as the reference, so its files have the reference's format and
every metric reads them unchanged. Its outlet is its own last cell, its ports its own exact
cumulative ports.

MUSCL-C: C with a van Leer limited reconstruction of the convected quantities at the faces, 32
and 64 cells, physical (uncompensated) dispersion: a second-order scheme has no leading
numerical diffusion to compensate. Descriptive, labelled "not proven consistent": the Bregman
proof covers first-order upwind plus non-negative diffusion, not limited reconstructions. The
first interior face downstream of the inlet is plain upwind (no second upstream neighbour); the
inlet face is the feed and the outlet face the last cell, so the port fluxes are unchanged.
"""
from __future__ import annotations

import os
import time

import numpy as np

from p3b.column import Column
from p3b.generate import _paper1_meta
from p3b.simulate_a import TrajectoryResult, setup_a, simulate_trajectory_a
from p3b.simulate_b import cycle_schedule as schedule_b
from p3b.simulate_b import make_store, simulate_trajectory_b
from p3b.store import Coarsener
from p3b.thermal_store import T0 as T0_B
from p3b.thermal_store import ThermalStore


def inject(y, nvar: int, nc: int, r: int) -> np.ndarray:
    """(ns, (nvar+1) nc) -> (ns, (nvar+1) nc r): densities repeated, wall heat split equally."""
    y = np.asarray(y, dtype=float)
    parts = [np.repeat(y[:, j * nc:(j + 1) * nc], r, axis=1) for j in range(nvar)]
    parts.append(np.repeat(y[:, nvar * nc:(nvar + 1) * nc], r, axis=1) / r)
    return np.concatenate(parts, axis=1)


def van_leer_faces(q, q_in):
    """Face values 0..N for flow from face 0 to face N: feed at face 0, last cell at face N,
    plain upwind at face 1, van Leer limited upwind-biased reconstruction elsewhere."""
    q = np.asarray(q, dtype=float)
    f = np.empty(q.size + 1)
    f[0], f[1], f[-1] = q_in, q[0], q[-1]
    i = np.arange(1, q.size - 1)
    dl, dr = q[i] - q[i - 1], q[i + 1] - q[i]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(dl != 0, dr / dl, 0.0)
    phi = (r + np.abs(r)) / (1 + np.abs(r))
    f[2:-1] = q[i] + 0.5 * phi * dl
    return f


class MusclColumn(Column):
    BAND = 2
    consistent = False

    def __init__(self, phys, N, c_w, *, iso, compensate=False):
        super().__init__(phys, N, c_w, iso=iso, compensate=False)

    @classmethod
    def from_column(cls, col: Column) -> "MusclColumn":
        return cls(col.p, col.N, col.c_w, iso=col.iso)

    def face_values(self, c, c_in=None):
        return van_leer_faces(c, c[0] if c_in is None else c_in)

    def fluxes(self, y, inputs):
        f = super().fluxes(y, inputs)
        c_in, T_in, _ = inputs
        cf = van_leer_faces(f["c"], c_in)
        Tf = van_leer_faces(f["T"], T_in)
        f["g_n"] = cf
        f["g_u"] = self.stream_energy(cf, Tf)
        return f


class MusclStore(ThermalStore):
    BAND = 2
    consistent = False

    @classmethod
    def from_store(cls, st: ThermalStore) -> "MusclStore":
        p = dict(st.params, compensate=False)
        return cls(st.N, **p)

    def fluxes(self, y, inputs):
        f = super().fluxes(y, inputs)
        v, T_in, _ = inputs
        g, g_in = f["g"], self.rhoc * (T_in - T0_B)
        if v >= 0:
            F = van_leer_faces(g, g_in)
            f["conv"] = abs(v) / self.dz * (F[:-1] - F[1:])
        else:
            F = van_leer_faces(g[::-1], g_in)[::-1]          # flow from face N to face 0
            f["conv"] = abs(v) / self.dz * (F[1:] - F[:-1])
        return f


def run_c(system: str, point: dict, material, N_c: int, path=None, n_snap: int = 200, n_target: int = 128,
          muscl: bool = False) -> TrajectoryResult:
    """Run arm C (or MUSCL-C) on N_c cells and store it at the n_target cells of the reference."""
    if n_target % N_c:
        raise ValueError("n_target must be a multiple of N_c")
    r = n_target // N_c
    t0 = time.process_time()
    try:
        if system == "A":
            model, schedule = setup_a(point, material, n_target)
            cz = Coarsener("A", n_target, n_target, model=model, c_in=schedule[0][2][0])
            nvar = 3
            on = lambda k, t, y, p: cz.add_cycle(k, t, inject(y, nvar, N_c, r), p)  # noqa: E731
            res = simulate_trajectory_a(point, material, N_c, n_snap=n_snap, on_cycle=on,
                                        column_cls=MusclColumn if muscl else Column)
        else:
            model = make_store(n_target)
            schedule = schedule_b(point, model)
            cz = Coarsener("B", n_target, n_target, model=model)
            nvar = 2
            on = lambda k, t, y, p: cz.add_cycle(k, t, inject(y, nvar, N_c, r), p)  # noqa: E731
            factory = (lambda n: MusclStore.from_store(make_store(n))) if muscl else None
            res = simulate_trajectory_b(point, N_c, n_snap=n_snap, on_cycle=on, store_factory=factory)
        if res.ok and path is not None:
            meta = dict(material=material, point=point, schedule=schedule, n_snap=n_snap, cpu_s=res.cpu_s,
                        arm="MUSCL-C" if muscl else "C", N_solver=N_c, consistent=not muscl,
                        closure_max=max(res.closure), nfev=res.nfev, cycle_cpu_s=res.cycle_cpu_s,
                        **_paper1_meta(system))
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            cz.to_npz(path, meta=meta, ports_end=res.final_ports)
        return res
    except Exception as exc:  # recorded as data, like the reference generator
        return TrajectoryResult(False, None, f"{type(exc).__name__}: {exc}", time.process_time() - t0,
                                [], None, [], [])
