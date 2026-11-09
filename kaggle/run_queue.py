"""Resumable Kaggle queue for reference-data generation (Paper 2's kaggle_b90 pattern).

    python kaggle/run_queue.py JOBS.json

JOBS.json: {"kit_slug": "abdulhamidbatayhi/p3b-reference-kit", "nz": 2048, "jobs": [
            {"system": "A_primary", "set": "train", "of": 16}, ...]}

For every job and chunk it writes a private kernel directory under kaggle/build/<name>/, pushes the
kernels that are absent on Kaggle while CPU slots are free (at most 5 running), downloads each
finished kernel's output into data/reference/<system>/<set>/, and ends with ALL_OUTPUTS_DOWNLOADED.
Rules carried from Paper 2:
  * only an explicit "not found" counts as absent; every other poll result is UNKNOWN and retried
    later (a 429 storm once made a queue resubmit finished kernels);
  * every push is logged in kaggle/registry.txt and never repeated;
  * no credentials in the repository: the kaggle CLI reads the user's own ~/.kaggle/kaggle.json.
The kaggle CLI is found on PATH, or through the KAGGLE_CLI environment variable.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
from p3b.design import chunk_indices  # noqa: E402


def chunk_indices_list(items: list, n: int) -> list[list]:
    """Split a list into n contiguous parts whose sizes differ by at most one."""
    return [[items[i] for i in part] for part in chunk_indices(len(items), n)]
BUILD = os.path.join(HERE, "build")
REG = os.path.join(HERE, "registry.txt")
OWNER = "abdulhamidbatayhi"
MAX_RUNNING = 5

RUN_PY = r'''import glob, os, shutil, subprocess, sys
kit = "/kaggle/working/kit"
if not os.path.exists(kit):
    src = [os.path.dirname(os.path.dirname(p)) for p in glob.glob("/kaggle/input/**/kaggle/generate_chunk.py", recursive=True)][0]
    shutil.copytree(src, kit)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
out = "/kaggle/working/out"
args = {args!r}
procs = [subprocess.Popen([sys.executable, os.path.join(kit, "kaggle", "generate_chunk.py"), *a, "--out", out])
         for a in args]
sys.exit(max(p.wait() for p in procs))
'''


def kaggle_cli() -> str:
    exe = os.environ.get("KAGGLE_CLI") or shutil.which("kaggle")
    if not exe:
        sys.exit("kaggle CLI not found: install it (pip install kaggle) or set KAGGLE_CLI")
    return exe


def parse_status(text: str) -> str:
    m = re.search(r'KernelWorkerStatus\.(\w+)', text)
    if m:
        return m.group(1)
    if "404" in text or "Permission 'kernels.get' was denied" in text:
        return "ABSENT"
    return "UNKNOWN"


def kernel_name(job: dict, chunk: int) -> str:
    return f"p3b-ref-{job['system']}-{job['set']}-{chunk}-of-{job['of']}".lower().replace("_", "-")


RUN_CMDS_PY = r'''import glob, os, subprocess, sys
# the kit is used read-only where Kaggle mounts it; nothing of it is copied into the outputs
hits = glob.glob("/kaggle/input/{kit_name}/**/kaggle/generate_chunk.py", recursive=True)
kit = os.path.dirname(os.path.dirname(hits[0]))
os.environ["OMP_NUM_THREADS"] = "{threads}"
os.environ["OPENBLAS_NUM_THREADS"] = "{threads}"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["P3B_PAPER1_DIR"] = os.path.join(kit, "paper1")
out = "/kaggle/working/out"
os.makedirs(out, exist_ok=True)


def m(x):
    return out if x == "OUT" else (out + x[3:] if x.startswith("OUT/") else x)


cmds = {cmds!r}
procs = [subprocess.Popen([sys.executable, os.path.join(kit, c[0]), *[m(x) for x in c[1:]]], cwd=out) for c in cmds]
sys.exit(max(p.wait() for p in procs))
'''


def _metadata(name: str, kit_slug: str, depends_on=()) -> dict:
    """Kernel metadata; the outputs of the kernels in `depends_on` are attached as inputs."""
    return {"id": f"{OWNER}/{name}", "title": name, "code_file": "run.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_internet": False,
            "dataset_sources": [kit_slug], "competition_sources": [],
            "kernel_sources": [f"{OWNER}/{d}" for d in depends_on]}


def ready(k: dict, done: set) -> bool:
    """A kernel is pushed only when every kernel it depends on has its outputs downloaded."""
    return all(d in done for d in k.get("depends_on", ()))


def write_kernels(spec: dict) -> list[dict]:
    """Chunk jobs: one kernel per chunk, split over the 4 CPU cores of a Kaggle session. Command jobs
    ({"name", "commands", "expect"}): one kernel running each command as its own process; arguments
    starting with OUT/ point into the kernel's output directory."""
    kernels = []
    for job in spec["jobs"]:
        if "commands" in job:
            d = os.path.join(BUILD, job["name"])
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "run.py"), "w") as f:
                f.write(RUN_CMDS_PY.format(cmds=job["commands"], kit_name=spec["kit_slug"].split("/")[-1],
                                           threads=job.get("threads", 1)))
            with open(os.path.join(d, "kernel-metadata.json"), "w") as f:
                json.dump(_metadata(job["name"], spec["kit_slug"], job.get("depends_on", ())), f, indent=1)
            kernels.append({"name": job["name"], "dir": d, "system": job.get("system", "timing"),
                            "set": job.get("set", job["name"]), "expect": job["expect"],
                            "depends_on": list(job.get("depends_on", ()))})
            continue
        with open(os.path.join(ROOT, "data", "design", f"{job['system']}_{job['set']}.json")) as f:
            n_traj = len(json.load(f)["points"])
        if job["of"] > n_traj:
            raise ValueError(f"{job['system']}_{job['set']}: {job['of']} chunks for {n_traj} trajectories")
        for chunk, members in enumerate(chunk_indices(n_traj, job["of"])):
            name = kernel_name(job, chunk)
            d = os.path.join(BUILD, name)
            os.makedirs(d, exist_ok=True)
            # sub-chunks of this chunk, one per core, never more than its trajectories; each sub-chunk
            # is addressed as a chunk of the whole set (generate_chunk's --chunk/--of) by index ranges
            sub = min(job.get("cores", 4), len(members))
            args = [["--system", job["system"], "--set", job["set"], "--indices",
                     ",".join(str(i) for i in part), "--report", f"{chunk}_{s}", "--nz", str(spec["nz"])]
                    for s, part in enumerate(chunk_indices_list(members, sub))]
            with open(os.path.join(d, "run.py"), "w") as f:
                f.write(RUN_PY.format(args=args))
            with open(os.path.join(d, "kernel-metadata.json"), "w") as f:
                json.dump(_metadata(name, spec["kit_slug"]), f, indent=1)
            kernels.append({"name": name, "dir": d, "system": job["system"], "set": job["set"], "chunk": chunk,
                            "expected_reports": len(args)})
    return kernels


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(REG, "a") as f:
        f.write(line + "\n")


def main(jobs_file: str) -> None:
    spec = json.load(open(jobs_file))
    kernels = write_kernels(spec)
    exe = kaggle_cli()

    def run(*a):
        r = subprocess.run([exe, *a], capture_output=True, text=True, timeout=600)
        return (r.stdout + r.stderr).strip()

    pushed = set(re.findall(r"^\S+ \S+ (\S+) pushed$", open(REG).read(), re.M)) if os.path.exists(REG) else set()

    def out_dir(k):
        return os.path.join(ROOT, "data", "reference", k["system"], k["set"], k["name"])

    def is_done(k):
        d = out_dir(k)
        if not os.path.isdir(d):
            return False
        names = [f for _, _, fs in os.walk(d) for f in fs]
        if "expect" in k:
            return all(e in names for e in k["expect"])
        return sum(f.endswith("_report.json") for f in names) >= k["expected_reports"]

    done = {k["name"] for k in kernels if is_done(k)}
    while len(done) < len(kernels):
        status = {k["name"]: parse_status(run("kernels", "status", f"{OWNER}/{k['name']}"))
                  for k in kernels if k["name"] not in done}
        busy = sum(s in ("RUNNING", "QUEUED") for s in status.values())
        for k in kernels:
            s = status.get(k["name"])
            if s is None:
                continue
            if s == "COMPLETE":
                os.makedirs(out_dir(k), exist_ok=True)
                run("kernels", "output", f"{OWNER}/{k['name']}", "-p", out_dir(k))
                if is_done(k):
                    done.add(k["name"])
                    log(f"{k['name']} output downloaded")
            elif s in ("ERROR", "CANCEL_ACKNOWLEDGED"):
                log(f"{k['name']} status {s}: needs attention")
            elif s == "ABSENT" and k["name"] not in pushed and busy < MAX_RUNNING and ready(k, done):
                out = run("kernels", "push", "-p", k["dir"])
                if "successfully pushed" in out.lower():
                    pushed.add(k["name"])
                    busy += 1
                    log(f"{k['name']} pushed")
                else:
                    log(f"{k['name']} push not accepted yet: {out[-160:]}")
        print(time.strftime("%H:%M:%S"), f"done {len(done)}/{len(kernels)}", flush=True)
        if len(done) < len(kernels):
            time.sleep(300)
    log("ALL_OUTPUTS_DOWNLOADED")


if __name__ == "__main__":
    main(sys.argv[1])
