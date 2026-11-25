import glob, json, os, sys, time
hits = glob.glob("/kaggle/input/**/p3b-reference-kit/**/kaggle/generate_chunk.py", recursive=True)
kit = os.path.dirname(os.path.dirname(hits[0]))
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["P3B_PAPER1_DIR"] = os.path.join(kit, "paper1")
sys.path.insert(0, kit); sys.path.insert(0, os.path.join(kit, "kaggle"))
out = "/kaggle/working"
from p3b import envcheck
envcheck.ensure({'kernel': 'p3b-env-fa681caa', 'pins': {'numpy': '2.4.6', 'scipy': '1.17.1', 'torch': '2.12.0+cpu'}}, "/kaggle/input", out)
import numpy as np
import pilot_chunk
data = pilot_chunk.stage_mounted("/kaggle/input", "/tmp/pd")
from p3b.pilot import pieces, runs
from p3b.train import trainer
from p3b.eval import rollout as ro
stage = {'K': {'A_MOF-303': 4}, '_revision': 'r4', 'd': {'A_Al-fumarate': 48, 'A_MOF-303': 32, 'B': 16}, 'decoder': {'A_Al-fumarate': 'mlp', 'A_MOF-303': 'mlp', 'B': 'conv'}, 'tau_S': {'A_Al-fumarate': 2.698299414224527, 'A_MOF-303': 0.0003275014853817415, 'B': 0.027308662178948485}}
ds, val, used = pieces._primary(data)
res = {}
for h in (1, 2, 4):
    t0 = time.time()
    arm = pieces.make_arm("S", pieces.PRIMARY_KEY, 0, stage, ds, os.path.join(out, "o%d" % h), {}, used)
    hist = trainer.train(arm, ds, trainer.TrainConfig(lr=3e-4, iterations=200, seed=0, h_steps=h, n_windows=8, n_steps=20,
                         checkpoint="/tmp/ck%d.pt" % h, every=200, stop_after=200))
    fails = [r.failed_cycle for r in ro.rollouts(arm, val, 1, h)]
    vl = runs.validation_loss(arm, val, h)
    res[h] = {"skipped": int(hist.get("skipped", 0)), "skipped_grad": int(hist.get("skipped_grad", 0)),
              "too_many_skipped": bool(runs.too_many_skipped(hist, 200)),
              "val_rollouts_failed": sum(f is not None for f in fails), "val_rollouts": len(fails),
              "fail_reasons": sorted({str(r.error)[:80] for r in ro.rollouts(arm, val, 1, h) if r.failed_cycle is not None}),
              "val_loss_finite": bool(np.isfinite(vl)), "minutes": round((time.time() - t0) / 60, 1)}
    print(h, res[h], flush=True)
json.dump(res, open(os.path.join(out, "hdiag.json"), "w"), indent=1)
