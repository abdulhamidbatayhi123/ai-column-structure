"""Run the test suite, or a subset, on Kaggle at a pushed commit.

For machines with too little free memory to import torch: the kernel clones this repository at the
commit and Paper 1 (ai-mof-cof-dynamics) at release v2.0.1 beside it, where p3b/_paper1.py looks for
it, installs the pinned requirements and runs pytest. pytest.log starts with the Python, torch,
numpy, scipy and pytest versions and the commit that ran.

    python kaggle/ci.py push <commit> [pytest args ...]   push kernel p3b-ci-<commit7>-<label>
    python kaggle/ci.py fetch <kernel-name>               wait for it, download, print the tail

The commit must be on GitHub (both repositories are public). The kaggle CLI is found on PATH or
through KAGGLE_CLI; its calls are retried, since the API sometimes drops TLS connections.
"""
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OWNER = "abdulhamidbatayhi"
REPO = "https://github.com/abdulhamidbatayhi123/ai-column-structure.git"
PAPER1 = "https://github.com/abdulhamidbatayhi123/ai-mof-cof-dynamics.git"
PAPER1_REF = "v2.0.1"

RUN = '''"""Paper 3B tests on Kaggle at {commit}: {label}."""
import os
import subprocess
import sys

W, T, PY = "/kaggle/working", "/tmp/w", sys.executable
os.makedirs(T, exist_ok=True)
log = open(os.path.join(W, "steps.log"), "a")


def sh(cmd, cwd=T):
    log.write("+ " + cmd + "\\n")
    log.flush()
    r = subprocess.run(cmd, shell=True, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    log.write("  -> %d\\n" % r.returncode)
    log.flush()
    return r.returncode


sh("git clone -q {repo} ai-column-structure")
sh("git checkout -q {commit}", cwd=T + "/ai-column-structure")
sh("git clone -q {paper1} ai-mof-cof-dynamics")
sh("git checkout -q {paper1_ref}", cwd=T + "/ai-mof-cof-dynamics")
pinned = sh(PY + " -m pip install -q -r requirements.txt", cwd=T + "/ai-column-structure")
rc = sh("{{ " + PY + " --version; " + PY + " -m pip freeze | grep -i -E '^(torch|numpy|scipy|pytest)=='; "
        "git rev-parse HEAD; echo pip_pinned_rc=%d; }} > %s/pytest.log 2>&1; " % (pinned, W)
        + PY + " -m pytest -q -p no:cacheprovider {args} >> %s/pytest.log 2>&1" % W,
        cwd=T + "/ai-column-structure")
with open(os.path.join(W, "rc.txt"), "w") as f:
    f.write("pytest %d\\npip_pinned %d\\ncommit {commit}\\n" % (rc, pinned))
'''


def kernel_name(commit: str, args: list) -> str:
    label = hashlib.sha1(" ".join(args).encode()).hexdigest()[:6] if args else "full"
    return f"p3b-ci-{commit[:7]}-{label}"


def write_kernel(commit: str, args: list, build: str) -> str:
    """The kernel directory (run.py and kernel-metadata.json) for `pytest args` at `commit`."""
    name = kernel_name(commit, args)
    d = os.path.join(build, name)
    os.makedirs(d, exist_ok=True)
    label = " ".join(args) or "full suite"
    with open(os.path.join(d, "run.py"), "w") as f:
        f.write(RUN.format(commit=commit, label=label.replace('"', "'"), repo=REPO, paper1=PAPER1,
                           paper1_ref=PAPER1_REF, args=shlex.join(args)))
    meta = {"id": f"{OWNER}/{name}", "title": name, "code_file": "run.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_internet": True,
            "dataset_sources": [], "competition_sources": [], "kernel_sources": []}
    with open(os.path.join(d, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return d


def _cli(*args, tries: int = 6) -> str:
    exe = os.environ.get("KAGGLE_CLI") or shutil.which("kaggle")
    if not exe:
        sys.exit("kaggle CLI not found: set KAGGLE_CLI")
    out = ""
    for i in range(tries):
        r = subprocess.run([exe, *args], capture_output=True, text=True)
        out = r.stdout + r.stderr
        if "SSLError" not in out and "Max retries" not in out:
            return out
        time.sleep(20 * (i + 1))
    return out


def push(commit: str, args: list) -> str:
    d = write_kernel(commit, args, os.path.join(HERE, "build", "ci"))
    print(_cli("kernels", "push", "-p", d).strip().splitlines()[-1])
    return os.path.basename(d)


def fetch(name: str, poll: int = 60) -> int:
    ref = f"{OWNER}/{name}"
    while True:
        s = _cli("kernels", "status", ref)
        if "COMPLETE" in s or "ERROR" in s or "CANCEL" in s:
            break
        time.sleep(poll)
    out = os.path.join(HERE, "build", "ci", name, "out")
    os.makedirs(out, exist_ok=True)
    _cli("kernels", "output", ref, "-p", out)
    for f in ("rc.txt", "pytest.log"):
        p = os.path.join(out, f)
        if os.path.exists(p):
            lines = open(p, encoding="utf-8", errors="replace").read().splitlines()
            print(f"== {f} ==")
            print("\n".join(lines[:8] + ["..."] + lines[-25:] if len(lines) > 40 else lines))
    print("status:", s.strip().splitlines()[-1])
    return 0 if "COMPLETE" in s else 1


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "push":
        print(push(sys.argv[2], sys.argv[3:]))
    elif len(sys.argv) == 3 and sys.argv[1] == "fetch":
        sys.exit(fetch(sys.argv[2]))
    else:
        sys.exit(__doc__)
