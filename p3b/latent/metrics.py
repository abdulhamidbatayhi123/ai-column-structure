"""Metrics shared by the decoder gate and the evaluation (PREREG Section 5).

field_nrmse is Paper 1's normalised RMSE (`metrics.per_variable_nrmse`: the RMSE of a field over
one sample, divided by that field's range in the reference sample), applied to each stored
channel on the coarse cells and averaged over the channels. One sample is one trajectory-cycle:
`pred` and `true` hold its snapshots, (snap, n_state), unstandardised, fields first.
"""
from __future__ import annotations

import torch


def field_nrmse(pred, true, n_cells: int, n_ch: int, per_field: bool = False):
    """RMSE / range per field over the snapshots and cells of one sample; the mean over the fields,
    or the vector of fields if `per_field` (the gate takes each field separately, because Paper 1
    forbids pooling across fields of very different variance)."""
    p = pred[..., :n_cells * n_ch].reshape(-1, n_cells, n_ch)
    t = true[..., :n_cells * n_ch].reshape(-1, n_cells, n_ch)
    rng = t.amax(dim=(0, 1)) - t.amin(dim=(0, 1))
    rmse = torch.sqrt(((p - t) ** 2).mean(dim=(0, 1)))
    v = rmse / torch.where(rng > 0, rng, torch.ones_like(rng))
    return v if per_field else v.mean()
