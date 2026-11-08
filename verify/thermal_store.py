"""Re-export shim: the module moved to the package p3b (plan 1, Task 1)."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from p3b.thermal_store import *  # noqa: E402,F401,F403
