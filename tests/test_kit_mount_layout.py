"""The kernel scripts find the kit wherever Kaggle mounts it (2026-11-22). A kernel with only the kit dataset sees
/kaggle/input/<slug>/; a kernel that also mounts other kernels' outputs (pilot data, continuations) sees
/kaggle/input/datasets/<owner>/<slug>/ and /kaggle/input/notebooks/<owner>/<slug>/ (checked on Kaggle with the
diagnostic kernel p3b-diag-mounts). Pilot phase 0 at 08e4b93 failed on all four kernels because the locator only
knew the flat layout."""
import glob
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kaggle"))


def _locator(template: str, root: str) -> str:
    import run_queue
    src = getattr(run_queue, template)
    line = next(l for l in src.splitlines() if l.startswith("hits = "))
    line = line.replace("{kit_name}", "p3b-reference-kit")
    assert '"/kaggle/input/' in line
    return line.replace('"/kaggle/input/', '"' + root.replace(os.sep, "/") + "/")


def _kit(base):
    k = base / "kaggle"
    k.mkdir(parents=True)
    (k / "generate_chunk.py").write_text("")
    return base


@pytest.mark.parametrize("template", ["RUN_PY", "RUN_CMDS_PY"])
@pytest.mark.parametrize("layout", [("p3b-reference-kit",), ("datasets", "abdulhamidbatayhi", "p3b-reference-kit")])
def test_kit_found_in_flat_and_nested_mount_layouts(tmp_path, template, layout):
    kit = _kit(tmp_path.joinpath(*layout))
    # another kernel's output beside it, as in the nested layout
    (tmp_path / "notebooks" / "abdulhamidbatayhi" / "p3b-ref-b-pilot-0-of-1-r3" / "out").mkdir(parents=True)
    ns = {"glob": glob}
    exec(_locator(template, str(tmp_path)), ns)
    assert len(ns["hits"]) == 1
    assert os.path.samefile(os.path.dirname(os.path.dirname(ns["hits"][0])), kit)
