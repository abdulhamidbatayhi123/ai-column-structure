# The ceiling's unmeasured costs (results/timing_ceiling.json; T8 DOC-2, DOC-7, R1-6): Kaggle kernel
# p3b-timing-ceiling. It clones this repository at COMMIT and Paper 1 at v2.0.1 beside it (as kaggle/ci.py
# does; no shared kit), installs the pinned requirements and runs scripts/time_ceiling.py on one thread.
# Costs only, never an error value.
import os
import subprocess
import sys

COMMIT = "__COMMIT__"
W, T, PY = "/kaggle/working", "/tmp/w", sys.executable
os.makedirs(T, exist_ok=True)
log = open(os.path.join(W, "steps.log"), "a")


def sh(cmd, cwd=T, env=None):
    log.write("+ " + cmd + "\n")
    log.flush()
    r = subprocess.run(cmd, shell=True, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, env=env)
    log.write("  -> %d\n" % r.returncode)
    log.flush()
    return r.returncode


sh("git clone -q https://github.com/abdulhamidbatayhi123/ai-column-structure.git ai-column-structure")
sh("git checkout -q " + COMMIT, cwd=T + "/ai-column-structure")
sh("git clone -q https://github.com/abdulhamidbatayhi123/ai-mof-cof-dynamics.git ai-mof-cof-dynamics")
sh("git checkout -q v2.0.1", cwd=T + "/ai-mof-cof-dynamics")
pinned = sh(PY + " -m pip install -q -r requirements.txt", cwd=T + "/ai-column-structure")
env = dict(os.environ, P3B_COMMIT=COMMIT, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
rc = sh(PY + " scripts/time_ceiling.py --out " + W + "/out/timing_ceiling.json", cwd=T + "/ai-column-structure", env=env)
with open(os.path.join(W, "rc.txt"), "w") as f:
    f.write("time_ceiling %d\npip_pinned %d\ncommit %s\n" % (rc, pinned, COMMIT))
print("DONE", rc, flush=True)
