"""One reference trajectory: simulate on the fine grid, coarsen per cycle, write one npz.

The npz meta (JSON) records the commit, the system, the material, the design point, the cycle
schedule (phase, duration, inputs) and the bed length, which is everything the gates need to
rebuild the model and the port inputs.
"""
from __future__ import annotations

from p3b.simulate_a import TrajectoryResult, setup_a, simulate_trajectory_a
from p3b.simulate_b import cycle_schedule as schedule_b
from p3b.simulate_b import make_store, simulate_trajectory_b
from p3b.store import Coarsener


def generate_trajectory(system: str, point: dict, material: str | None, N_z: int, N_c: int = 128,
                        path=None, n_snap: int = 200) -> TrajectoryResult:
    """Run one trajectory and, if it succeeds and `path` is given, write its coarse npz."""
    if system == "A":
        col, schedule = setup_a(point, material, N_z)
        cz = Coarsener("A", N_z, N_c, model=col, c_in=schedule[0][2][0])
        res = simulate_trajectory_a(point, material, N_z, n_snap=n_snap, on_cycle=cz.add_cycle)
    elif system == "B":
        st = make_store(N_z)
        schedule = schedule_b(point, st)
        cz = Coarsener("B", N_z, N_c, model=st)
        res = simulate_trajectory_b(point, N_z, n_snap=n_snap, on_cycle=cz.add_cycle)
    else:
        raise ValueError(f"unknown system {system!r}")
    if res.ok and path is not None:
        cz.to_npz(path, meta=dict(material=material, point=point, schedule=schedule, n_snap=n_snap,
                                  cpu_s=res.cpu_s, closure_max=max(res.closure)),
                  ports_end=res.final_ports)
    return res
