"""Exact Jacobians of the reference simulators' augmented systems (Task 10 stall fix).

scipy's BDF estimates the Jacobian by finite differences on the sparsity pattern. On MOF-303 at the
box centre that estimate failed as the front crossed the isotherm step: at 1024 cells cycle 1 took
642,420 evaluations (3,753 CPU-s) and 2048 cells gave no result in 11 h on Kaggle, with num_jac
overflow warnings; with the exact Jacobian the same half-cycle took 1,487 evaluations and 45 CPU-s
(verify/stall_exact_jacobian.py). The exact Jacobian comes from the torch transcriptions of the
right-hand sides (p3b.solvers.c_plus_l, checked against numpy to 1e-12) by forward-mode products,
coloured by variable and cell mod 3, since every output row depends on at most cells i-1, i, i+1:
- system A, state [n, a, u (3N) | wall heat per cell (N) | 6 cumulative ports]: physical rows depend
  on the neighbouring cells, wall row i on cell i, the outlet ports on the last cell;
- system B, state [u_f, u_s (2N) | wall heat per cell (N) | 4 cumulative ports]: likewise, the
  outlet ports on the outlet cell of the current flow direction.
The integral states feed nothing, so their columns are zero.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import csr_matrix

from p3b.column import C_REF, R, T0
from p3b.solvers.c_plus_l import TorchColumnRHS, TorchStoreRHS
from p3b.thermal_store import T0 as T0_B

F64 = torch.float64


def colored_rows(f, y, n_cells: int, n_var: int, row_cell, n_rows: int, n_cols: int) -> csr_matrix:
    """Sparse (n_rows, n_cols) Jacobian of f at y, where output row r depends only on cells
    row_cell[r] - 1, row_cell[r], row_cell[r] + 1 of every variable (columns beyond n_var n_cells, the
    integral states, are zero): 3 n_var forward-mode products, at most 3 n_var entries per row."""
    n = y.numel()
    rows_all, cols_all, vals_all = [], [], []
    rows = torch.arange(n_rows)
    rc = torch.as_tensor(row_cell)
    seeds = torch.zeros(3 * n_var, n, dtype=F64)
    for var in range(n_var):
        for k in range(3):
            seeds[3 * var + k, var * n_cells + torch.arange(k, n_cells, 3)] = 1.0
    JV = torch.func.vmap(lambda s: torch.func.jvp(f, (y,), (s,))[1])(seeds)   # one batched pass
    for var in range(n_var):
        for k in range(3):
            Jv = JV[3 * var + k]
            for off in (-1, 0, 1):
                cell = rc + off
                ok = (cell >= 0) & (cell < n_cells) & (cell % 3 == k)
                rows_all.append(rows[ok])
                cols_all.append(var * n_cells + cell[ok])
                vals_all.append(Jv[ok].detach())
    r = torch.cat(rows_all).numpy()
    c = torch.cat(cols_all).numpy()
    val = torch.cat(vals_all).numpy()
    return csr_matrix((val, (r, c)), shape=(n_rows, n_cols))


def augmented_jacobian_a(col, y, inputs, trhs: TorchColumnRHS | None = None) -> csr_matrix:
    """Exact Jacobian of p3b.simulate_a._Augmented.rhs at the augmented state y."""
    N = col.N
    trhs = trhs or TorchColumnRHS(col)
    inp = torch.tensor(inputs, dtype=F64)
    c_in, T_in = inp[0], inp[1]
    v, dz = float(col.p.v), col.dz

    def s_stream(c, T):
        return trhs.rhoC * torch.log(T / T0) + c * (trhs.c_w * torch.log(T / T0) - R * torch.log(c / C_REF) + R)

    def f(yp):
        fl = trhs.fluxes(yp, inp)
        c, T = fl["c"], fl["T"]
        c_out = torch.clamp(c[-1], min=1e-30)
        ports = torch.stack([v * c_in, v * c[-1], v * trhs.stream_energy(c_in, T_in),
                             v * trhs.stream_energy(c[-1], T[-1]), v * s_stream(c_in, T_in), v * s_stream(c_out, T[-1])])
        return torch.cat([trhs(yp, inp), fl["W"] * dz, ports])

    yp = torch.tensor(np.asarray(y[:3 * N], dtype=float), dtype=F64)
    row_cell = np.concatenate([np.tile(np.arange(N), 3), np.arange(N), np.full(6, N - 1)])
    return colored_rows(f, yp, N, 3, row_cell, 4 * N + 6, 4 * N + 6)


def augmented_jacobian_b(st, y, inputs, trhs: TorchStoreRHS | None = None) -> csr_matrix:
    """Exact Jacobian of p3b.simulate_b._Augmented.rhs at the augmented state y."""
    N = st.N
    trhs = trhs or TorchStoreRHS(st)
    v, T_in, T_amb = float(inputs[0]), float(inputs[1]), float(inputs[2])
    inp = torch.tensor(inputs, dtype=F64)
    out_cell = N - 1 if v >= 0 else 0

    def f(yp):
        Tf = T0_B + yp[:N] / st.Cf
        T_out = Tf[out_cell]
        Tin = torch.tensor(T_in, dtype=F64)
        ports = torch.stack([abs(v) * st.rhoc * (Tin - T0_B), abs(v) * st.rhoc * (T_out - T0_B),
                             abs(v) * st.rhoc * torch.log(Tin / T0_B), abs(v) * st.rhoc * torch.log(T_out / T0_B)])
        return torch.cat([trhs(yp, inp), st.h_a * (Tf - T_amb) * st.dz, ports])

    yp = torch.tensor(np.asarray(y[:2 * N], dtype=float), dtype=F64)
    row_cell = np.concatenate([np.tile(np.arange(N), 2), np.arange(N), np.full(4, out_cell)])
    return colored_rows(f, yp, N, 2, row_cell, 3 * N + 4, 3 * N + 4)
