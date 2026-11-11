# Decoder-gate diagnostic 3 (follows diag2: larger balance weights, the lambda=1 learning curve, d=64)
# Decoder-gate diagnostic 2 on the pilot data (no arm is trained or evaluated).
# (1) Learning curve: does the balance error fall with training trajectories (8, 16, 32 for MOF-303)?
# (2) Per-snapshot balance errors: percentiles, signed bias, share above 0.01, where the maximum sits.
# (3) Exploratory only, never part of the design: the same MLP with a balance term in the loss.
import glob, json, math, os, sys, time
hits = glob.glob("/kaggle/input/p3b-reference-kit/**/kaggle/generate_chunk.py", recursive=True)
kit = os.path.dirname(os.path.dirname(hits[0]))
sys.path.insert(0, kit)
os.environ["P3B_PAPER1_DIR"] = os.path.join(kit, "paper1")
import numpy as np
import torch
torch.set_num_threads(4)
from p3b.latent.autoencoder import MLPAutoencoder, cosine_lr, decoder_gate, train_autoencoder
from p3b.latent.data import Standardiser, load_set
from p3b.pilot import runs
from p3b.pilot.pieces import files

out = "/kaggle/working/out"
os.makedirs(out, exist_ok=True)
res = {"kit_commit": open(os.path.join(kit, "kit_commit.txt")).read().strip(), "rows": []}
root = os.path.join(kit, "data", "pilot")


def save():
    with open(os.path.join(out, "decoder_diag3.json"), "w") as f:
        json.dump(res, f, indent=1)


@torch.no_grad()
def per_snapshot(model, ds, std, phys):
    """Signed balance errors per snapshot, normalised as in decoder_gate."""
    A = ds.system == "A"
    n_ch = 3 if A else 2
    nc = ds.n_cells
    rows = {"N": [], "E": [], "N_gas": [], "N_ads": [], "loc": []}
    for j in range(ds.X.shape[0]):
        for k in range(ds.X.shape[1]):
            X = ds.X[j, k]
            R = std.decode(model.decode(model.encode(std.encode(X))))
            ch_r = R[:, :nc * n_ch].reshape(-1, nc, n_ch)
            ch_t = X[:, :nc * n_ch].reshape(-1, nc, n_ch)
            W = abs(float(ds.W_ref[j, k]))
            wall = float(ds.wall_abs[j, k])
            if A:
                e = (phys.N(ch_r) - ds.N_true[j, k]) / W
                rows["N"].append(e.numpy())
                nr, ar, _ = phys.conserved(*phys.fields(ch_r))
                nt, at, _ = phys.conserved(*phys.fields(ch_t))
                dz = phys._dz(ch_r)
                rows["N_gas"].append(((nr - nt).sum(-1) * dz / W).numpy())
                rows["N_ads"].append(((ar - at).sum(-1) * dz / W).numpy())
                rows["E"].append(((phys.E(ch_r) - ds.E_true[j, k]) / (abs(phys.dH) * W + wall)).numpy())
            else:
                rows["E"].append(((phys.E(ch_r) - ds.E_true[j, k]) / (W + wall)).numpy())
            rows["loc"].append((j, k, X.shape[0]))
    summ = {}
    for name in ("N", "E", "N_gas", "N_ads"):
        if not rows[name]:
            continue
        a = np.concatenate(rows[name])
        ab = np.abs(a)
        i = int(np.argmax(np.concatenate([np.abs(r) for r in rows[name]])))
        # where the max sits: trajectory, cycle, fraction of the cycle
        cum = 0
        where = None
        for (j, k, n), r in zip(rows["loc"], rows[name]):
            if i < cum + n:
                where = {"traj": j, "cycle": k, "frac": (i - cum) / n}
                break
            cum += n
        per_cycle = {}
        for (j, k, n), r in zip(rows["loc"], rows[name]):
            per_cycle.setdefault(k, []).append(float(np.abs(r).max()))
        summ[name] = {"p50": float(np.percentile(ab, 50)), "p90": float(np.percentile(ab, 90)),
                      "p99": float(np.percentile(ab, 99)), "max": float(ab.max()),
                      "signed_mean": float(a.mean()), "share_gt_0.01": float((ab > 0.01).mean()),
                      "argmax": where,
                      "max_by_cycle": {str(k): float(np.max(v)) for k, v in sorted(per_cycle.items())}}
    return summ


def train_balance(model, ds, std, phys, lam, epochs=60, batch=256, lr=1e-3, lr_min=1e-5, seed=0):
    """Exploratory: MSE + lam * (balance errors normalised as in the gate)^2."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    A = ds.system == "A"
    n_ch = 3 if A else 2
    nc = ds.n_cells
    Xs = std.encode(ds.X)
    T, C, S, n = Xs.shape
    Xf = Xs.reshape(-1, n)
    Xraw = ds.X.reshape(-1, n)
    W = ds.W_ref.abs().unsqueeze(-1).expand(T, C, S).reshape(-1)
    wall = ds.wall_abs.unsqueeze(-1).expand(T, C, S).reshape(-1)
    Et = ds.E_true.reshape(-1)
    Nt = ds.N_true.reshape(-1) if A else None
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    spe = math.ceil(Xf.shape[0] / batch)
    total, step = epochs * spe, 0
    for _ in range(epochs):
        perm = torch.randperm(Xf.shape[0], generator=gen)
        for i in range(spe):
            idx = perm[i * batch:(i + 1) * batch]
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, total, lr, lr_min)
            rec = model(Xf[idx])
            loss = ((rec - Xf[idx]) ** 2).mean()
            R = std.decode(rec)
            ch = R[:, :nc * n_ch].reshape(-1, nc, n_ch)
            if A:
                eN = (phys.N(ch) - Nt[idx]) / W[idx]
                eE = (phys.E(ch) - Et[idx]) / (abs(phys.dH) * W[idx] + wall[idx])
                loss = loss + lam * (eN.pow(2).mean() + eE.pow(2).mean())
            else:
                eE = (phys.E(ch) - Et[idx]) / (W[idx] + wall[idx])
                loss = loss + lam * eE.pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1


def subset(ds, m):
    import dataclasses
    kw = {}
    for f in dataclasses.fields(ds):
        v = getattr(ds, f.name)
        if isinstance(v, torch.Tensor):
            v = v[:m]
        elif f.name in ("meta", "paths") and v is not None:
            v = v[:m]
        kw[f.name] = v
    return type(ds)(**kw)


def record(tag, key, ae, val, std, phys, extra, t0):
    g = decoder_gate(ae, val, std, phys)
    row = {"tag": tag, "key": key, **extra, "train_s": time.time() - t0, **g,
           "snapshots": per_snapshot(ae, val, std, phys)}
    res["rows"].append(row)
    print(json.dumps({k: v for k, v in row.items() if k != "snapshots"}), flush=True)
    save()



def run(tag, key, ds, val, std, phys, lam, d, n):
    torch.manual_seed(0)
    ae = MLPAutoencoder(ds.X.shape[-1], d)
    t0 = time.time()
    if lam == 0:
        train_autoencoder(ae, std.encode(ds.X), epochs=60, seed=0)
    else:
        train_balance(ae, ds, std, phys, lam)
    record(tag, key, ae, val, std, phys, {"n_train_traj": n, "d": d, "loss": "mse" if lam == 0 else f"mse+{lam}*balance"}, t0)


for key in ("A_MOF-303", "B"):
    full, val = load_set(files(root, key, "train")), load_set(files(root, key, "validation"))
    n_ch, n_out = (3, 2) if full.system == "A" else (2, 1)
    phys = runs._physics(val)
    N = int(full.X.shape[0])
    std = Standardiser.fit(full.X, full.n_cells, n_ch, n_out, features=full.features)
    for lam in (3.0, 10.0, 30.0):
        run("lambda", key, full, val, std, phys, lam, 32, N)
    if key == "A_MOF-303":
        for m in (8, 16):
            ds = subset(full, m)
            run("curve_lambda1", key, ds, val, Standardiser.fit(ds.X, ds.n_cells, n_ch, n_out, features=ds.features),
                phys, 1.0, 32, m)
        run("d64", key, full, val, std, phys, 3.0, 64, N)
print("DONE", flush=True)
