"""The pinned runtime environment of every production kernel (2026-11-22).

Kaggle's image decides the library versions a kernel gets, and it can change between kernels (it had torch
2.11.0 while requirements.txt and CI pin 2.12.0+cpu). So the pinned versions are made the runtime: a wheel
kernel (kaggle/env_wheels.py, internet on) downloads them once; every pilot and study kernel mounts its output,
installs them offline before any computation starts (`ensure`), writes ENV.json into its outputs and refuses to
run on any mismatch. This module imports nothing heavy: it runs before torch is installed."""
import glob
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys

TEST_ONLY = {"pytest"}


def pins_from_requirements(path: str) -> dict:
    """{package: version} for the exact pins of requirements.txt, test tools left out."""
    pins = {}
    with open(path) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if not line or line.startswith("-") or "==" not in line:
                continue
            name, ver = (s.strip() for s in line.split("==", 1))
            if name.lower() not in TEST_ONLY:
                pins[name.lower()] = ver
    return dict(sorted(pins.items()))


def env_kernel_name(pins: dict) -> str:
    """The wheel kernel's name: it changes whenever a pin does, so a new environment never reuses old wheels."""
    h = hashlib.sha256(json.dumps(dict(sorted(pins.items())), sort_keys=True).encode()).hexdigest()
    return "p3b-env-" + h[:8]


def env_spec(requirements: str) -> dict:
    pins = pins_from_requirements(requirements)
    return {"kernel": env_kernel_name(pins), "pins": pins}


def _pip_install(wheel_dir: str, specs: list) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", wheel_dir,
                           *specs])


def _pip_freeze() -> list:
    r = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    return r.stdout.splitlines()


def ensure(env: dict, input_root: str, out: str, install=_pip_install, version=importlib.metadata.version,
           freeze=_pip_freeze) -> dict:
    """Install env["pins"] offline from the wheel kernel env["kernel"] mounted under input_root, record what
    the kernel runs in out/ENV.json and stop (SystemExit) unless every pin is what runs."""
    wheels = sorted(glob.glob(os.path.join(input_root, "**", env["kernel"], "**", "*.whl"), recursive=True))
    if not wheels:
        raise SystemExit(f"environment wheels not found: mount the output of {env['kernel']} (kaggle/env_wheels.py)")
    pins = env["pins"]
    install(os.path.dirname(wheels[0]), [f"{p}=={v}" for p, v in pins.items()])
    got = {}
    for p in pins:
        try:
            got[p] = version(p)
        except importlib.metadata.PackageNotFoundError:
            got[p] = None
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "ENV.json"), "w") as f:
        json.dump({"kernel": env["kernel"], "wanted": pins, "got": got, "python": sys.version,
                   "freeze": freeze()}, f, indent=1)
    bad = {p: {"wanted": v, "got": got[p]} for p, v in pins.items() if got[p] != v}
    if bad:
        raise SystemExit(f"environment mismatch: {bad}")
    return got
