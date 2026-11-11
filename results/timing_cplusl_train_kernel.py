# C+L training cost at kit 212ba17: train_cplusl (8 windows x 20 steps, the study defaults) for 20 iterations
# on the 32 MOF-303 pilot training trajectories at 32 cells, K = 1 and 2, one thread.
import glob, json, os, sys, time
hits = glob.glob("/kaggle/input/p3b-reference-kit/**/kaggle/generate_chunk.py", recursive=True)
kit = os.path.dirname(os.path.dirname(hits[0]))
sys.path.insert(0, kit)
os.environ["P3B_PAPER1_DIR"] = os.path.join(kit, "paper1")
import torch
torch.set_num_threads(1)
from p3b.latent.data import load_set
from p3b.pilot.pieces import files
from p3b.solvers.c_plus_l import Multipliers
from p3b.train.cplusl import train_cplusl
kit_commit = open(os.path.join(kit, "kit_commit.txt")).read().strip()
print("kit", kit_commit, flush=True)
ds = load_set(files(os.path.join(kit, "data", "pilot"), "A_MOF-303", "train"))
out = {"kit": kit_commit, "n_traj": int(ds.X.shape[0]), "iterations": 20}
for K in (1, 2):
    torch.manual_seed(0)
    t = time.time(); c = time.process_time()
    h = train_cplusl(Multipliers(n_out=3), ds, "MOF-303", 32, lr=1e-3, iterations=20, seed=0, K=K)
    out[f"K{K}"] = {"wall_s_per_iter": (time.time() - t) / 20, "cpu_s_per_iter": (time.process_time() - c) / 20,
                    "skipped": h["skipped"], "skipped_grad": h["skipped_grad"]}
    print(K, out[f"K{K}"], flush=True)
os.makedirs("/kaggle/working/out", exist_ok=True)
json.dump(out, open("/kaggle/working/out/cl_train_timing.json", "w"), indent=1)
print("DONE", flush=True)
