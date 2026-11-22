"""Production kernels run the pinned environment of requirements.txt, not whatever Kaggle's image carries
(2026-11-22: the image had torch 2.11.0 while CI pins 2.12.0+cpu, and the image can change during the
confirmatory runs). A wheel kernel (internet on) downloads the pinned wheels once; every pilot and study kernel
mounts its output, installs offline (p3b.envcheck.ensure), writes ENV.json and refuses to run on any mismatch."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "kaggle"))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PINS = {"numpy": "2.4.6", "scipy": "1.17.1", "torch": "2.12.0+cpu"}


def test_pins_come_from_requirements_without_test_tools():
    from p3b import envcheck
    assert envcheck.pins_from_requirements(os.path.join(ROOT, "requirements.txt")) == PINS


def test_env_kernel_name_is_stable_and_follows_the_pins():
    from p3b import envcheck
    n = envcheck.env_kernel_name(PINS)
    assert n.startswith("p3b-env-") and n == envcheck.env_kernel_name(dict(PINS))
    assert n != envcheck.env_kernel_name({**PINS, "torch": "2.11.0+cpu"})


def _wheels(tmp_path, name):
    d = tmp_path / "input" / "notebooks" / "abdulhamidbatayhi" / name / "wheels"
    d.mkdir(parents=True)
    (d / "torch-2.12.0+cpu-cp313-cp313-manylinux_2_28_x86_64.whl").write_text("")
    return d


def test_ensure_refuses_without_wheels(tmp_path):
    from p3b import envcheck
    with pytest.raises(SystemExit, match="wheels not found"):
        envcheck.ensure({"kernel": "p3b-env-x", "pins": PINS}, str(tmp_path / "input"), str(tmp_path / "out"),
                        install=lambda *a: None, version=lambda p: PINS[p])


def test_ensure_installs_offline_records_and_passes(tmp_path):
    from p3b import envcheck
    d = _wheels(tmp_path, "p3b-env-x")
    seen = []
    got = envcheck.ensure({"kernel": "p3b-env-x", "pins": PINS}, str(tmp_path / "input"), str(tmp_path / "out"),
                          install=lambda wd, specs: seen.append((wd, specs)), version=lambda p: PINS[p])
    assert got == PINS
    assert seen == [(str(d), ["numpy==2.4.6", "scipy==1.17.1", "torch==2.12.0+cpu"])]
    rec = json.load(open(tmp_path / "out" / "ENV.json"))
    assert rec["wanted"] == PINS and rec["got"] == PINS and rec["kernel"] == "p3b-env-x" and "python" in rec


def test_ensure_refuses_a_mismatch_but_records_it(tmp_path):
    from p3b import envcheck
    _wheels(tmp_path, "p3b-env-x")
    with pytest.raises(SystemExit, match="environment mismatch"):
        envcheck.ensure({"kernel": "p3b-env-x", "pins": PINS}, str(tmp_path / "input"), str(tmp_path / "out"),
                        install=lambda *a: None, version=lambda p: "2.11.0+cpu" if p == "torch" else PINS[p])
    assert json.load(open(tmp_path / "out" / "ENV.json"))["got"]["torch"] == "2.11.0+cpu"


def test_wheel_kernel_downloads_the_pins_with_internet(tmp_path):
    import env_wheels
    from p3b import envcheck
    d = env_wheels.write(str(tmp_path))
    meta = json.load(open(os.path.join(d, "kernel-metadata.json")))
    assert meta["enable_internet"] is True and meta["id"].endswith("/" + envcheck.env_kernel_name(PINS))
    src = open(os.path.join(d, "run.py")).read()
    assert "download" in src and "--only-binary=:all:" in src and all(f"{p}=={v}" in src for p, v in PINS.items())


SPEC = {"kit_slug": "o/p3b-reference-kit", "nz": 0, "env": {"kernel": "p3b-env-x", "pins": PINS}, "jobs": [
    {"name": "p3b-pilot-curve-cpl-x", "system": "pilot", "set": "curve-CpL", "threads": 1, "resumable": True,
     "commands": [["kaggle/pilot_chunk.py", "--piece", "curve-CpL", "--out", "OUT"]], "expect": ["curve_CpL.json"]}]}


@pytest.fixture
def rq(tmp_path, monkeypatch):
    import run_queue
    monkeypatch.setattr(run_queue, "BUILD", str(tmp_path / "build"))
    monkeypatch.setattr(run_queue, "REG", str(tmp_path / "registry.txt"))
    monkeypatch.setattr(run_queue, "ROOT", str(tmp_path))
    return run_queue


def _files(k):
    return (json.load(open(os.path.join(k["dir"], "kernel-metadata.json")))["kernel_sources"],
            open(os.path.join(k["dir"], "run.py")).read())


def test_every_kernel_and_continuation_mounts_and_checks_the_env(rq):
    k, = rq.write_kernels(SPEC)
    for kk in (k, rq.continuation(k, SPEC["kit_slug"])):
        srcs, run = _files(kk)
        assert "abdulhamidbatayhi/p3b-env-x" in srcs
        assert "envcheck" in run and "torch" in run and "2.12.0+cpu" in run
        assert run.index("envcheck") < run.index("subprocess.Popen")
    compile(_files(k)[1], "run.py", "exec")


def test_packed_kernels_check_the_env(rq):
    jobs = [{**SPEC["jobs"][0], "name": f"p3b-pilot-curve-cpl-{i}", "expect": [f"c{i}.json"]} for i in range(2)]
    ks = rq.write_kernels({**SPEC, "pack": 2, "jobs": jobs})
    assert all("envcheck" in _files(k)[1] and "abdulhamidbatayhi/p3b-env-x" in _files(k)[0] for k in ks)


def test_pilot_and_study_specs_carry_the_env_and_names_stay_short():
    import pilot_jobs
    import train_jobs
    from p3b import envcheck
    env = {"kernel": envcheck.env_kernel_name(PINS), "pins": PINS}
    for phase in range(5):
        st = {"_revision": pilot_jobs.PILOT_REVISION} if phase else None
        js = pilot_jobs.phase_jobs(phase, st)
        assert js["env"] == env
        for j in js["jobs"]:
            assert j["name"].endswith("-e" + env["kernel"][-4:]), j["name"]
            assert len(j["name"] + "-r10") <= 50, j["name"]
    from p3b.train.cells import cells
    cs = cells()
    for spec in (train_jobs.build_jobs(cs), train_jobs.gate_jobs(cs), train_jobs.evaluation_jobs(cs)):
        assert spec["env"] == env
