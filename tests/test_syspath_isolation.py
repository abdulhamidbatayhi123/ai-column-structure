"""kaggle/pilot_chunk.py, train_chunk.py and generate_chunk.py drop kaggle/ from sys.path when imported (so the
Kaggle CLI's own `kaggle` package is never shadowed on Kaggle). A test that imports one must not leave later tests
without kaggle/ on sys.path: that made later plain imports fail by test order (env_wheels in the full CI run at
bd2ed5f; run_queue, pilot_jobs, pilot_chunk after test_pilot_e2e). tests/conftest.py restores sys.path after
every test. The two tests below run in file order."""
import importlib
import os
import sys

KAGGLE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "kaggle"))
sys.path.insert(0, KAGGLE)              # at collection, as every test module that imports kaggle/ scripts does


def test_a_importing_a_chunk_script_strips_kaggle_from_sys_path():
    sys.modules.pop("pilot_chunk", None)
    importlib.import_module("pilot_chunk")
    assert KAGGLE not in [os.path.abspath(p or ".") for p in sys.path]


def test_b_the_next_test_still_has_kaggle_on_sys_path():
    assert KAGGLE in [os.path.abspath(p or ".") for p in sys.path]
