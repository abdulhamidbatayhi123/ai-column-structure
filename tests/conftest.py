"""Make the repository root importable (package p3b) without installing into a shared venv."""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture(autouse=True)
def _restore_sys_path():
    """Give every test back the sys.path it started with: the chunk scripts (kaggle/pilot_chunk.py,
    train_chunk.py, generate_chunk.py) drop kaggle/ from sys.path when imported, which broke later tests'
    plain imports by test order (tests/test_syspath_isolation.py)."""
    saved = list(sys.path)
    yield
    sys.path[:] = saved
