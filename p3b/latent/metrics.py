"""Metrics shared by the decoder gate and the evaluation (PREREG Section 5).

field_nrmse is Paper 1's normalised RMSE (`metrics.per_variable_nrmse`: the RMSE of a field over
one sample, divided by that field's range in the reference sample), applied to each stored
channel on the coarse cells and averaged over the channels. One sample is one trajectory-cycle:
`pred` and `true` hold its snapshots, (snap, n_state), unstandardised, fields first.
"""
from __future__ import annotations

import torch


def field_nrmse(pred, true, n_cells: int, n_ch: int):
    """Mean over the field channels of RMSE / range, over the snapshots and cells of one sample."""
    p = pred[..., :n_cells * n_ch].reshape(-1, n_cells, n_ch)
    t = true[..., :n_cells * n_ch].reshape(-1, n_cells, n_ch)
    rng = t.amax(dim=(0, 1)) - t.amin(dim=(0, 1))
    rmse = torch.sqrt(((p - t) ** 2).mean(dim=(0, 1)))
    return (rmse / torch.where(rng > 0, rng, torch.ones_like(rng))).mean()
