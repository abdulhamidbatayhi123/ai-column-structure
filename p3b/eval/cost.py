"""Cost of every arm (PREREG Section 5; Plan 3, Task 5).

CPU time (process_time, all threads of the process) on an idle Kaggle CPU session with 4 threads,
the same session type for every arm, never the laptop:
- latency: CPU-seconds per simulated cycle at batch 1;
- throughput: trajectory-cycles per CPU-second at batch 48. Learned arms batch 48 trajectories in
  one tensor; the solver arms (C, C+L, MUSCL-C) run one trajectory per process, so 48 processes
  are run and the summed CPU time of all of them is charged (parallel_throughput).
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor

import torch

THREADS = 4


def _fix_threads():
    torch.set_num_threads(THREADS)


def latency(run_one, n_cycles: int, clock=time.process_time) -> float:
    """run_one() simulates one trajectory for n_cycles at batch 1."""
    _fix_threads()
    t0 = clock()
    run_one()
    return (clock() - t0) / n_cycles


def throughput(run_batch, batch: int, n_cycles: int, clock=time.process_time) -> float:
    """run_batch() simulates `batch` trajectories for n_cycles in one call."""
    _fix_threads()
    t0 = clock()
    run_batch()
    return batch * n_cycles / (clock() - t0)


def _timed(fn):
    torch.set_num_threads(1)
    t0 = time.process_time()
    fn()
    return time.process_time() - t0


def parallel_throughput(run_one, n_procs: int, n_cycles: int, workers: int = THREADS):
    """(trajectory-cycles per CPU-second, summed CPU-seconds) for n_procs independent single-
    trajectory runs on `workers` processes; run_one must be picklable (a module-level function)."""
    with ProcessPoolExecutor(max_workers=workers) as pool:
        cpu = sum(pool.map(_timed, [run_one] * n_procs))
    return n_procs * n_cycles / cpu, cpu
