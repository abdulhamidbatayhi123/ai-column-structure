# C+L timing at kit 212ba17 (batched Newton): 16 MOF-303 validation trajectories, 2 cycles, K = 1 and 8, one thread.
import glob, json, os, sys, tempfile, time
hits = glob.glob("/kaggle/input/p3b-reference-kit/**/kaggle/generate_chunk.py", recursive=True)
kit = os.path.dirname(os.path.dirname(hits[0]))
sys.path.insert(0, kit)
os.environ["P3B_PAPER1_DIR"] = os.path.join(kit, "paper1")
import torch
torch.set_num_threads(1)
from p3b.gates import load_meta
from p3b.latent.data import load_set
from p3b.pilot.pieces import files
from p3b.solvers.c_plus_l import Multipliers
from p3b.train.cplusl import rollout_cplusl_to_files
print("kit", open(os.path.join(kit, "kit_commit.txt")).read().strip(), flush=True)
va = files(os.path.join(kit, "data", "pilot"), "A_MOF-303", "validation")
ds = load_set(va[:1])
nc, ns = ds.n_cells, ds.X.shape[2]
pts = [dict(load_meta(p)["point"], n_cycles=2) for p in va]
tmp = tempfile.mkdtemp()
out = {"n_traj": len(pts), "cycles": 2}
for K in (1, 8):
    t = time.time(); c = time.process_time()
    rollout_cplusl_to_files(Multipliers(n_out=3, constant=1.0), pts, "MOF-303", 32, nc, ns, K,
                            [os.path.join(tmp, f"cl{K}_{j}.npz") for j in range(len(pts))])
    out[f"K{K}"] = {"wall_s": time.time() - t, "cpu_s": time.process_time() - c}
    print(K, out[f"K{K}"], flush=True)
os.makedirs("/kaggle/working/out", exist_ok=True)
json.dump(out, open("/kaggle/working/out/cl_timing.json", "w"), indent=1)
print("DONE", flush=True)
