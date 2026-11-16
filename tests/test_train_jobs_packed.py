"""The confirmatory jobs run on one thread each, packed four to a Kaggle session (PREREG Section 11,
freeze decision 2 option (i)): one job given all four threads is no faster than on one
(results/timing_arms.json), while four one-thread jobs give 1.99 x the throughput
(results/timing_concurrency_batch.json)."""
import importlib.util
import os

from p3b.train.cells import cells

spec = importlib.util.spec_from_file_location("train_jobs", os.path.join(os.path.dirname(__file__), "..", "kaggle", "train_jobs.py"))
tj = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tj)


def test_every_confirmatory_jobs_file_is_one_thread_and_packed_four_per_session():
    cs = cells()
    for spec_ in (tj.build_jobs(cs), tj.gate_jobs(cs), tj.evaluation_jobs(cs)):
        assert spec_["pack"] == 4
        assert spec_["jobs"] and all(j["threads"] == 1 for j in spec_["jobs"])
