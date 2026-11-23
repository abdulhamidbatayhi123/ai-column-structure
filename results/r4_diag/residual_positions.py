"""R4 diagnosis (2026-11-23): model-style trapezoid residuals per snapshot interval on every pilot file, with the
share of the extremes and of the negative values that fall on the last interval of each half-cycle (B's flow
reversals). Run from the repository root:  python results/r4_diag/residual_positions.py OUT.json
before_single_outlet.json was produced with B's single outlet channel (main 12a9667), after_two_end_outlet.json
with both end temperatures (R4)."""
import sys, glob, json, numpy as np
sys.path.insert(0, __import__("os").path.abspath(__import__("os").path.join(__import__("os").path.dirname(__file__), "..", "..")))
from p3b.latent.data import load_set
from p3b.pilot.runs import _physics
from p3b.eval.violations import trapezoid_residuals
out = {}
for key in ("A_MOF-303", "A_Al-fumarate", "B"):
    files = sorted(glob.glob(f"data/pilot/{key}/*/*.npz"))
    acc = {}
    for f in files:
        ds = load_set([f]); phys = _physics(ds); ns = ds.X.shape[2]
        X = ds.X[0].reshape(-1, ds.X.shape[-1]); R = ds.raw_inputs[0].reshape(X.shape[0], -1); t = ds.t[0].reshape(-1)
        r = trapezoid_residuals(X, R, t, phys, ds.system, ds.n_cells)
        pos = np.arange(X.shape[0] - 1) % ns
        for q, v in r.items():
            x = (v / (t[1:] - t[:-1])).numpy()
            a = acc.setdefault(q, {"x": [], "pos": []}); a["x"].append(x); a["pos"].append(pos)
    res = {}
    for q, a in acc.items():
        x, pos = np.concatenate(a["x"]), np.concatenate(a["pos"])
        sw = np.isin(pos, [ns // 2 - 1, ns - 1])
        thr = np.percentile(np.abs(x), 99.9)
        top = np.abs(x) >= thr
        neg = x < 0
        res[q] = {"n": int(x.size), "p99.9|x| all": float(thr), "p99.9|x| non-switch": float(np.percentile(np.abs(x[~sw]), 99.9)),
                  "top0.1% on switch": float(sw[top].mean()),
                  "n_neg": int(neg.sum()), "neg on switch": int((neg & sw).sum()),
                  "neg positions (non-switch, most common)": [int(p) for p in np.bincount(pos[neg & ~sw], minlength=ns).argsort()[::-1][:5]] if (neg & ~sw).any() else [],
                  "min non-switch": float(x[~sw].min()), "min switch": float(x[sw].min())}
    out[key] = res
    print(key, json.dumps(res, indent=1))
json.dump(out, open(sys.argv[1], "w"), indent=1)
