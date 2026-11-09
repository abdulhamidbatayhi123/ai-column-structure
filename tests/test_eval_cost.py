"""Plan 3, Task 5: cost (PREREG Section 5): CPU time, latency at batch 1, throughput at batch 48."""
import time

import pytest
import torch

from p3b.eval.cost import latency, parallel_throughput, throughput


class FakeClock:
    def __init__(self, per_call):
        self.t, self.per_call = 0.0, per_call

    def __call__(self):
        self.t += self.per_call
        return self.t


def test_latency_is_cpu_seconds_per_simulated_cycle():
    calls = []
    lat = latency(lambda: calls.append(1), n_cycles=4, clock=FakeClock(2.0))
    assert lat == pytest.approx(2.0 / 4) and calls == [1]


def test_throughput_is_trajectory_cycles_per_cpu_second():
    thr = throughput(lambda: None, batch=48, n_cycles=10, clock=FakeClock(3.0))
    assert thr == pytest.approx(48 * 10 / 3.0)


def _busy():
    s = 0.0
    for i in range(200_000):
        s += i * 1e-9
    return s


def test_parallel_throughput_charges_the_summed_cpu_of_all_processes():
    thr, cpu = parallel_throughput(_busy, n_procs=4, n_cycles=1, workers=2)
    assert thr == pytest.approx(4 / cpu, rel=1e-12) and cpu > 0


def test_threads_are_fixed_to_four():
    latency(lambda: None, n_cycles=1, clock=FakeClock(1.0))
    assert torch.get_num_threads() == 4


def test_default_clock_is_process_time_not_wall_time():
    import inspect
    from p3b.eval import cost
    assert inspect.signature(cost.latency).parameters["clock"].default is time.process_time
