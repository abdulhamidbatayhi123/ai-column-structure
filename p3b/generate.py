"""One reference trajectory: simulate on the fine grid, coarsen per cycle, write one npz.

The npz meta (JSON) records the commit, the system, the material, the design point, the cycle
schedule (phase, duration, inputs) and the bed length, which is everything the gates need to
rebuild the model and the port inputs. The fine-level gate quantities (rate identities, fine
production, fine entropy) are computed at every snapshot before the fine state is discarded.

Failures are returned as data, never raised: a failed trajectory writes no file, and the npz is
written atomically, so a file with the final name is always complete.
"""
from __future__ import annotations

import os
import time

from p3b._paper1 import PAPER1_SHA256
from p3b.gates import current_constants
from p3b.simulate_a import TrajectoryResult, setup_a, simulate_trajectory_a
from p3b.simulate_b import cycle_schedule as schedule_b
from p3b.simulate_b import make_store, simulate_trajectory_b
from p3b.store import Coarsener


def _paper1_meta(system: str) -> dict:
    """The provenance every stored trajectory carries: generation constants and Paper 1's hashes."""
    return dict(constants=current_constants(system), paper1_sha256=dict(PAPER1_SHA256))


def generate_trajectory(system: str, point: dict, material: str | None, N_z: int, N_c: int = 128,
                        path=None, n_snap: int = 200, extra_meta: dict | None = None) -> TrajectoryResult:
    """Run one trajectory and, if it succeeds and `path` is given, write its coarse npz."""
    t0 = time.process_time()
    try:
        if system == "A":
            col, schedule = setup_a(point, material, N_z)
            cz = Coarsener("A", N_z, N_c, model=col, c_in=schedule[0][2][0], schedule=schedule)
            res = simulate_trajectory_a(point, material, N_z, n_snap=n_snap, on_cycle=cz.add_cycle)
        elif system == "B":
            st = make_store(N_z)
            schedule = schedule_b(point, st)
            cz = Coarsener("B", N_z, N_c, model=st, schedule=schedule)
            res = simulate_trajectory_b(point, N_z, n_snap=n_snap, on_cycle=cz.add_cycle)
        else:
            raise ValueError(f"unknown system {system!r}")
        if res.ok and path is not None:
            meta = dict(material=material, point=point, schedule=schedule, n_snap=n_snap, cpu_s=res.cpu_s,
                        closure_max=max(res.closure), nfev=res.nfev, cycle_cpu_s=res.cycle_cpu_s,
                        cycles_override=None)
            meta.update(_paper1_meta(system))
            meta.update(extra_meta or {})
            cz.to_npz(path, meta=meta, ports_end=res.final_ports)
        return res
    except Exception as exc:  # setup, coarsening or writing failed: recorded, never raised
        if path is not None and os.path.exists(os.fspath(path) + ".partial"):
            os.remove(os.fspath(path) + ".partial")
        return TrajectoryResult(False, None, f"{type(exc).__name__}: {exc}", time.process_time() - t0, [])
