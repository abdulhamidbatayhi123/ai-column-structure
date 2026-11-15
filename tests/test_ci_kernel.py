"""kaggle/ci.py runs the suite (or a subset) on Kaggle at a pushed commit, from clean public clones: the
kernel it writes clones this repository at that commit and Paper 1 at v2.0.1 beside it, installs the
pinned requirements, and starts pytest.log with the versions used."""
import importlib.util
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _ci():
    spec = importlib.util.spec_from_file_location("ci", os.path.join(ROOT, "kaggle", "ci.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_kernel_clones_the_commit_and_paper1_and_runs_the_given_tests(tmp_path):
    ci = _ci()
    d = ci.write_kernel("0123456789abcdef", ["tests/test_a.py", "-k", "x and not y"], str(tmp_path))
    meta = json.load(open(os.path.join(d, "kernel-metadata.json")))
    assert meta["id"].startswith("abdulhamidbatayhi/p3b-ci-0123456-") and meta["enable_internet"] is True
    assert meta["is_private"] is True and meta["enable_gpu"] is False and len(meta["id"].split("/")[1]) <= 50
    script = open(os.path.join(d, "run.py")).read()
    assert "ai-column-structure.git" in script and "git checkout -q 0123456789abcdef" in script
    assert "ai-mof-cof-dynamics.git" in script and "git checkout -q v2.0.1" in script
    assert "pip install -q -r requirements.txt" in script
    assert "tests/test_a.py -k 'x and not y'" in script          # arguments shell-quoted
    assert script.index("pip freeze") < script.index("-m pytest")   # versions first in pytest.log


def test_push_waits_for_a_free_kaggle_slot(tmp_path, monkeypatch):
    ci = _ci()
    replies = iter(["Kernel push error: Maximum batch CPU session count of 5 reached."] * 2
                   + ["Kernel version 1 successfully pushed."])
    calls = []
    monkeypatch.setattr(ci, "_cli", lambda *a, **k: (calls.append(a), next(replies))[1])
    monkeypatch.setattr(ci.time, "sleep", lambda s: None)
    monkeypatch.setattr(ci, "HERE", str(tmp_path))
    assert ci.push("abcdef0123", []) == "p3b-ci-abcdef0-full"
    assert len(calls) == 3 and all(c[0] == "kernels" and c[1] == "push" for c in calls)


def test_the_full_suite_and_a_subset_get_different_kernel_names(tmp_path):
    ci = _ci()
    full = json.load(open(os.path.join(ci.write_kernel("abcdef0123", [], str(tmp_path / "a")), "kernel-metadata.json")))
    sub = json.load(open(os.path.join(ci.write_kernel("abcdef0123", ["tests/test_a.py"], str(tmp_path / "b")),
                                      "kernel-metadata.json")))
    assert full["id"].endswith("-full") and full["id"] != sub["id"]
