"""Make the repository root importable (package p3b) without installing into a shared venv."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
