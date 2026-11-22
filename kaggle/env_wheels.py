"""Write the wheel kernel of the pinned environment (p3b/envcheck.py): a Kaggle kernel with internet that
downloads the wheels of requirements.txt's pins (and their dependencies) for Kaggle's own Python into its
output, which every production kernel then mounts and installs offline. Push it once per set of pins:

    python kaggle/env_wheels.py [BUILD_DIR]      then   kaggle kernels push -p BUILD_DIR/<name>
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from p3b import envcheck  # noqa: E402

OWNER = "abdulhamidbatayhi"
INDEX = "https://download.pytorch.org/whl/cpu"

RUN = r'''import json, subprocess, sys
pins = {pins!r}
specs = {specs!r}
subprocess.check_call([sys.executable, "-m", "pip", "download", "-d", "/kaggle/working/wheels",
                       "--only-binary=:all:", "--extra-index-url", "{index}", *specs])
with open("/kaggle/working/PINS.json", "w") as f:
    json.dump({{"pins": pins, "python": sys.version}}, f, indent=1)
print("\n".join(specs))
'''


def write(build: str = os.path.join(ROOT, "kaggle", "build")) -> str:
    env = envcheck.env_spec(os.path.join(ROOT, "requirements.txt"))
    d = os.path.join(build, env["kernel"])
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "run.py"), "w") as f:
        f.write(RUN.format(pins=env["pins"], specs=[f"{p}=={v}" for p, v in env["pins"].items()], index=INDEX))
    with open(os.path.join(d, "kernel-metadata.json"), "w") as f:
        json.dump({"id": f"{OWNER}/{env['kernel']}", "title": env["kernel"], "code_file": "run.py",
                   "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": False,
                   "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": []},
                  f, indent=1)
    return d


if __name__ == "__main__":
    print(write(*sys.argv[1:2]))
