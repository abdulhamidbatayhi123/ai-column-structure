"""Pilot diagnostic of arm S's stiffness (plan 3 review; formulation 5.3).

With the equalities fixed, grad S^ = B c + g (g = P grad S^, B = [grad E^, grad N^]), so every
admissible latent velocity has g . zdot >= delta0 = Phi_S - c . (P_E, P_N). For the encoded data,
g . zdot_enc = delta0 + sigma >= delta0 holds whenever the decoded functionals are consistent with
the data; S's correction then asks for no more motion than the data make. The diagnostic reports,
on consecutive encoded snapshots of the pilot data (finite differences):
  - share_violating: the share of snapshots with g . zdot_enc < delta0 (the encoded data violate
    S's inequality: a decoder inconsistency that makes S stiff);
  - the distribution of the correction size h delta0 / |g| (h = h_over_dt x the snapshot interval),
    which the pilot's h rule weighs.
"""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def s_stiffness(lf, ae, std, ds, h_over_dt: float = 1.0) -> dict:
    share, corr, n = 0, [], 0
    for j in range(ds.X.shape[0]):
        X = ds.X[j].reshape(-1, ds.X.shape[-1])
        R = ds.raw_inputs[j].reshape(X.shape[0], -1)
        t = ds.t[j].reshape(-1)
        z = ae.encode(std.encode(X))
        zdot = (z[1:] - z[:-1]) / (t[1:] - t[:-1])[:, None]
        v = lf.evaluate(z[:-1], R[:-1], grads=True, create_graph=False)
        if lf.system == "A":
            B = torch.stack([v["dE"], v["dN"]], -1)
            ports = torch.stack([v["P_E"], v["P_N"]], -1)
        else:
            B = v["dE"].unsqueeze(-1)
            ports = v["P_E"].unsqueeze(-1)
        c = (torch.linalg.pinv(B) @ v["dS"].unsqueeze(-1)).squeeze(-1)
        g = v["dS"] - (B @ c.unsqueeze(-1)).squeeze(-1)
        delta0 = v["Phi_S"] - (c * ports).sum(-1)
        share += int(((g * zdot).sum(-1) < delta0).sum())
        h = h_over_dt * (t[1:] - t[:-1])
        corr.append((h * delta0.clamp(min=0) / g.norm(dim=-1).clamp(min=1e-300)).numpy())
        n += z.shape[0] - 1
    corr = np.concatenate(corr)
    return {"n": n, "share_violating": share / n, "correction_median": float(np.median(corr)),
            "correction_p99": float(np.percentile(corr, 99))}
