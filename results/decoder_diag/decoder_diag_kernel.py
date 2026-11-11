# Decoder-gate diagnostic on the pilot data (no arm is trained or evaluated): the gate's metrics per latent
# dimension, at the pre-registered 60 epochs and at 300, and the pre-registered convolutional fallback.
import glob, json, os, sys, time
hits = glob.glob("/kaggle/input/p3b-reference-kit/**/kaggle/generate_chunk.py", recursive=True)
kit = os.path.dirname(os.path.dirname(hits[0]))
sys.path.insert(0, kit)
os.environ["P3B_PAPER1_DIR"] = os.path.join(kit, "paper1")
import torch
torch.set_num_threads(4)
from p3b.latent.autoencoder import ConvDecoderAutoencoder, MLPAutoencoder, decoder_gate, train_autoencoder
from p3b.latent.data import Standardiser, load_set
from p3b.pilot import runs
from p3b.pilot.pieces import files
out = "/kaggle/working/out"
os.makedirs(out, exist_ok=True)
res = {"kit_commit": open(os.path.join(kit, "kit_commit.txt")).read().strip(), "rows": []}
root = os.path.join(kit, "data", "pilot")
GRID = [("mlp", 16, 60), ("mlp", 64, 60), ("mlp", 16, 300), ("mlp", 64, 300), ("conv", 16, 60), ("conv", 64, 60),
        ("mlp", 32, 300), ("conv", 32, 60)]
for key in ("A_MOF-303", "A_Al-fumarate", "B"):
    ds, val = load_set(files(root, key, "train")), load_set(files(root, key, "validation"))
    n_ch, n_out = (3, 2) if ds.system == "A" else (2, 1)
    std = Standardiser.fit(ds.X, ds.n_cells, n_ch, n_out, features=ds.features)
    phys = runs._physics(val)
    for kind, d, ep in GRID:
        torch.manual_seed(0)
        ae = (MLPAutoencoder(ds.X.shape[-1], d) if kind == "mlp"
              else ConvDecoderAutoencoder(ds.n_cells, n_ch, n_out, d))
        t0 = time.time()
        h = train_autoencoder(ae, std.encode(ds.X), epochs=ep, seed=0)
        g = decoder_gate(ae, val, std, phys)
        row = {"key": key, "decoder": kind, "d": d, "epochs": ep, "train_s": time.time() - t0,
               "final_train_loss": h["loss"][-1], "n_train_traj": int(ds.X.shape[0]), **g}
        res["rows"].append(row)
        print(json.dumps(row), flush=True)
        with open(os.path.join(out, "decoder_diag.json"), "w") as f:
            json.dump(res, f, indent=1)
