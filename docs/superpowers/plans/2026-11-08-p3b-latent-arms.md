# Paper 3B Latent Model and Arms (Plan 2 of 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement every arm of PREREG Section 3 as a tested, differentiable module on top of
the shared autoencoder: the physical functionals of decoded states, the autoencoder and its
decoder gate, the RK4 integrator with the Gauss-Newton balance projection, and the dynamics of
U, U+F, P, S, S-L, S−M, S−Π, U+proj, U+F+proj, FNO, C, C+L and MUSCL-C. Training loops,
metrics and the pilot rules are Plans 3 and 4.

**Architecture:**
- A subpackage `p3b/latent/`, one job per module: physics functionals (torch, float64), data
  windows, autoencoder, integrator, unstructured arms, structured arms, FNO.
- A subpackage `p3b/solvers/` for the solver arms C, C+L and MUSCL-C, which reuse
  `p3b.column.Column`, `p3b.thermal_store.ThermalStore` and the Plan 1 simulators.
- The physics in torch is checked against the numpy reference (Plan 1) to round-off; every
  structural guarantee of formulation Section 5.3 is a test.

**Tech Stack:** Python 3.12, torch 2.12.0+cpu (already in the Paper 1 venv; nothing is
installed into it), numpy 2.4.6, scipy 1.17.1, pytest 9.0.3.

**Spec:** `PREREG_P3B_draft.md` Sections 3, 4 and 5 (branch plan1-reference-data, after the
Task 10 merge); `docs/formulation_3B.md` Sections 3, 4.4 and 5; Plan 1's stored-file format
(`p3b/store.py`, `p3b/gates.py`).

## Global Constraints

- All learned arms share one frozen autoencoder per seed, system, material and training size.
- Autoencoder: MLP encoder 386 → 256 → 256 → d, decoder d → 256 → 256 → 386 (system A: 128 × 3
  channels (log c, log x, log T) + 2 outlet channels (log c_out, log T_out); system B: 128 × 2
  channels (T_f, T_s) + 1 outlet channel T_out, so 257); SiLU; standardised channels; MSE; Adam
  1e-3 with cosine decay to 1e-5; batch 256 snapshots; 60 epochs.
- d ∈ {16, 24, 32, 48, 64}; conv fallback decoder: 4 layers, 32 channels.
- U: f_θ MLP (d + inputs) → 128 → 128 → 128 → d, SiLU. S: G_θ the same; A(z) MLP d → 128 → d·d.
  S-L: E_θ and S_θ MLPs d → 64 → 64 → 1. U, U+F, P and FNO widths matched to S's trainable
  dynamics parameter count within ±5 %, counts printed.
- Cycle inputs: log c_in, T_in, T_w, v, and flow direction (B), standardised.
- Training windows: batch 8 windows × 20 RK4 steps; loss = MSE of the decoded standardised
  channels along each window, outlet included, equal weights.
- Projection: pseudo-inverse rank tolerance 1e-8 (relative singular value); κ correction
  λ = max(0, δ)/(|P∇Ŝ|² + κ); "κ binds" when |P∇Ŝ|² < 10 κ. κ, τ_S and h are pilot values,
  so every module takes them as arguments, never as constants.
- P: penalty weight such that the penalty equals 1 % of the data loss at initialisation.
- FNO: 4 Fourier layers, 16 modes, width 64 (adjusted for the parameter match).
- C: hybrid scheme at 16, 32, 64, 128 cells, rtol 1e-6, atol 1e-10, injection of the conserved
  variables onto 128 cells. C+L: C at 32 cells, multipliers softplus(MLP 3 → 32 → 32 → 1) on
  D_L,eff, k_z,eff and k_LDF (B: h_v), implicit-Euler steps. MUSCL-C: van Leer, 32 and 64
  cells, labelled "not proven consistent".
- Physics in float64 everywhere (the 1e-8 relative rank tolerance is below float32 epsilon).

## Review Focus

1. A decoded state with a near-uniform field (P∇Ŝ → 0): the S step must stay finite, count
   "κ binds", and never divide by zero.
2. B = [∇Ê, ∇N̂] rank-deficient (e.g. a decoder whose N̂ and Ê gradients are parallel): the
   pseudo-inverse must not raise, and the defect must be counted.
3. System B flow reversal inside a window: the outlet side and the port signs must follow the
   direction input, per RK4 stage.
4. A window that crosses the adsorption/desorption switch: inputs are piecewise constant per
   snapshot interval and the RK4 stages of a step use the interval's inputs.
5. The parameter matcher when no width gives ±5 %: it must raise, never silently accept.

---

### Task 1: Physical functionals of decoded states (torch)

**Files:**
- Create: `p3b/latent/__init__.py`, `p3b/latent/physics.py`
- Test: `tests/test_latent_physics.py`

**Interfaces:**
- Consumes: `p3b.column.Column` (numpy reference), `p3b.thermal_store.ThermalStore`,
  `p3b.isotherms.MultiTermIsotherm`.
- Produces:
  - `ColumnPhysics(col: Column)`, torch float64, with
    `fields(ch) -> (c, x, T)` from unstandardised channels `(…, 128, 3)`;
    `conserved(c, x, T) -> (n, a, u)`; `E(ch)`, `N(ch)`, `S(ch) -> (…)` totals per m² (sum × dz);
    `ports(ch, out, inp) -> (P_N, P_E, Phi_S)` with `inp = (c_in, T_in, T_w, v)` and
    `out = (log c_out, log T_out)`.
  - `StorePhysics(st: ThermalStore, T_amb)` with `E`, `S`, `ports(ch, out, inp) -> (P_E, Phi_S)`,
    `inp = (v, T_in)` (signed v selects the outlet side).
  - `Q_torch(iso, x)`, `Pi_torch(iso, x)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_functionals_match_numpy_column():
    # random smooth state on 128 cells for MOF-303; channels from Column quantities
    assert abs(phys.E(ch) - np.sum(u) * col.dz) <= 1e-12 * abs(np.sum(u) * col.dz)
    assert abs(phys.N(ch) - np.sum(n + a) * col.dz) <= 1e-12 * ...
    assert abs(phys.S(ch) - np.sum(col.cell_entropy(n, a, u)) * col.dz) <= 1e-12 * ...

def test_ports_match_column_budget():
    # with outlet = last cell, ports equal Column.budget's portN, portE, portS to 1e-12 relative
def test_store_functionals_match_numpy(): ...   # both flow directions
def test_gradients_are_finite_through_the_decoder_fields(): ...  # torch.autograd on S
```

- [ ] **Step 2: Run** `pytest tests/test_latent_physics.py -q`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement** with `q = Q(x)` (no inverse root: the decoder outputs log x),
  `s_adsorbed` from x directly, every formula transcribed from `Column` Section 3 and 4.4.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat(latent): physical functionals and ports of decoded states`.

### Task 2: Data windows and standardisation

**Files:** Create `p3b/latent/data.py`; Test `tests/test_latent_data.py`.

**Interfaces:**
- Consumes: Plan 1 npz files (`channels`, `outlet`, `t`, `conserved_blocks`, `wall_heat_blocks`,
  `ports_cum`, meta `schedule`).
- Produces:
  - `load_set(paths) -> Dataset` with float64 tensors `X (traj, cycle, snap, 386|257)`,
    `inputs (traj, cycle, snap, k)` (the interval's cycle inputs, piecewise constant),
    `t`, and the raw arrays the metrics need.
  - `Standardiser.fit(X_train)`, `.encode(X)`, `.decode(X)`; per-channel-type statistics
    (one mean and SD per channel type, not per cell, so translation is not baked in).
  - `windows(ds, n_windows=8, n_steps=20, rng) -> (x0, inputs, targets, dt)`.

- [ ] Steps 1-5 as in Task 1. Tests: shapes; the input of snapshot interval i is the schedule
  phase containing [t_i, t_i+1); a window crossing the half-cycle switch carries both inputs;
  `decode(encode(X)) == X` to 1e-12; statistics from the training set only.
- Commit `feat(latent): data windows and standardisation`.

### Task 3: Autoencoder and decoder gate

**Files:** Create `p3b/latent/autoencoder.py`; Test `tests/test_autoencoder.py`.

**Interfaces:**
- Produces: `MLPAutoencoder(n_in, d)`, `ConvDecoderAutoencoder(n_cells, n_ch, n_out, d)` (4 conv
  layers, 32 channels), `train_autoencoder(model, X_train, epochs=60, batch=256, lr=1e-3,
  lr_min=1e-5, seed) -> history`, `decoder_gate(model, ds_val, phys) -> dict` with the four
  PREREG 5.1 quantities and `passed: bool`, `choose_d(results) -> int | None` (smallest passing d
  in the grid).
- [ ] Tests: layer sizes 386-256-256-d (A) and 257-256-256-d (B); cosine schedule ends at 1e-5;
  training on a tiny synthetic set lowers the loss; `decoder_gate` on an identity-like decoder
  (exact reconstruction) passes and on a constant decoder fails; `choose_d` returns the smallest
  passing value and None when none passes.
- Commit `feat(latent): autoencoder, decoder gate and latent-dimension rule`.

### Task 4: RK4 integrator, stage quadrature and the Gauss-Newton balance projection

**Files:** Create `p3b/latent/integrate.py`; Test `tests/test_integrate.py`.

**Interfaces:**
- Produces: `rk4_step(f, z, inp, h) -> (z_new, stages)`; `stage_quadrature(g, stages, h)` (the
  RK4 weights applied to a port function at the four stages); `balance_project(z, targets,
  funcs, iters=3) -> (z, residual)`, Gauss-Newton on the constraints N̂(z) = N̂_prev + Q[P_N],
  Ê(z) = Ê_prev + Q[P_E] with the minimum-norm update (pinv, rank tol argument);
  `violation(z_prev, z_new, stages, funcs) -> dict` (water and energy residuals with each arm's
  own stages, PREREG 5).
- [ ] Tests: RK4 is fourth-order on dz/dt = Az (error ratio 16 ± 1 on halving h); the projection
  restores two linear constraints exactly and two nonlinear ones to 1e-12 in ≤ 3 iterations; a
  rank-deficient constraint Jacobian is handled by the pseudo-inverse and flagged.
- Commit `feat(latent): RK4 with stage quadrature and balance projection`.

### Task 5: Unstructured arms U, U+F, P and the parameter matcher

**Files:** Create `p3b/latent/arms_unstructured.py`; Test `tests/test_arms_unstructured.py`.

**Interfaces:**
- Produces: `UDynamics(d, k_inputs, width)`, `UFDynamics(d, k_inputs, width, phys_terms)` (input
  adds Ê, N̂, Ŝ, P_E, P_N, Φ_S and ∇Ê, ∇N̂, ∇Ŝ; B: no N terms), `penalty_loss(...)` for P
  (water and energy residuals and negative total production of the decoded state),
  `penalty_weight(data_loss0, penalty0) = 0.01 * data_loss0 / penalty0`,
  `count_params(module)`, `match_width(build, target, tol=0.05) -> width` (raises ValueError if
  no width matches).
- [ ] Tests: U's layer sizes; U+F's input size is d + k + 6 + 3d (A) and d + k + 4 + 2d (B);
  the matcher hits ±5 % of a target and raises when impossible; the penalty weight equals 1 % of
  the initial data loss.
- Commit `feat(latent): unstructured arms and parameter matching`.

### Task 6: Structured arm S and its ablations S−M, S−Π

**Files:** Create `p3b/latent/arms_structured.py`; Test `tests/test_arm_s.py`.

**Interfaces:**
- Produces: `SDynamics(d, k_inputs, phys, ae_decoder, kappa, rank_tol=1e-8, metriplectic=True,
  port_projection=True)`; `forward(z, inp) -> (dzdt, info)` with `info` holding
  `rank_deficient`, `kappa_binds`, `residual_deficit` per sample; `projector(B, rank_tol)`.
- [ ] Tests (each a guarantee of formulation 5.3, on a random decoder, float64):
  - `∇Ê · dzdt == P_E` and `∇N̂ · dzdt == P_N` to 1e-10 relative;
  - `∇Ŝ · dzdt − Φ_S ≥ −residual_deficit` and ≥ 0 when κ does not bind;
  - the metriplectic term alone leaves Ê and N̂ unchanged and produces entropy ≥ 0;
  - a near-uniform decoded state gives finite output and `kappa_binds = True`;
  - a decoder with parallel ∇Ê, ∇N̂ gives finite output and `rank_deficient = True`;
  - S−M has no metriplectic term; S−Π has no port projection (then ∇Ê·dzdt ≠ P_E in general).
- Commit `feat(latent): structured arm S with ablations`.

### Task 7: S-L and the projected unstructured arms U+proj, U+F+proj

**Files:** Modify `p3b/latent/arms_structured.py`; Test `tests/test_arm_sl_proj.py`.

**Interfaces:**
- Produces: `SLDynamics(...)` with learned `E_θ`, `S_θ` (d → 64 → 64 → 1), the degeneracy
  projector built from [∇E_θ, ∇N̂], port projection only on N̂; `ProjectedRollout(dyn)` that
  rolls a trained U or U+F out with `balance_project` after every step (formulation 5.4), no
  retraining.
- [ ] Tests: S-L conserves N̂ to 1e-10 and the learned E_θ along its metriplectic term;
  `ProjectedRollout` leaves the trained weights unchanged and satisfies both balances per step.
- Commit `feat(latent): S-L and projected rollouts`.

### Task 8: FNO arm

**Files:** Create `p3b/latent/fno.py`; Test `tests/test_fno.py`.

**Interfaces:**
- Produces: `FNO1d(n_ch, n_out, k_inputs, modes=16, width=64, layers=4)` mapping the stored coarse
  fields and outlet channels at one snapshot to the next (autoregressive, conditioned on the
  inputs); the outlet channels enter as constant extra channels and leave through a pooled head.
- [ ] Tests: output shape; 16 retained modes; translation equivariance of the spectral layer on a
  periodic test input; the width the matcher picks is within ±5 % of S.
- Commit `feat(latent): FNO arm`.

### Task 9: Solver arms C and MUSCL-C

**Files:** Create `p3b/solvers/__init__.py`, `p3b/solvers/coarse.py`; Test `tests/test_solver_arms.py`.

**Interfaces:**
- Consumes: `simulate_trajectory_a/b(point, material, N_z, ...)`, `Column`, `ThermalStore`.
- Produces: `run_c(system, point, material, N_c) -> (channels_128, outlet, cpu_s)`; `inject(F, 128)`
  (piecewise-constant injection of conserved variables); `MusclColumn(Column)` with van Leer
  limited reconstruction of c and T at faces; `run_muscl(...)`.
- [ ] Tests: at N_c = 128 C reproduces the stored reference coarse blocks of a 128-cell solve
  exactly; injection conserves N and E; MUSCL-C reduces to upwind on a monotone profile with
  limiter 0 and is labelled `consistent = False`.
- Commit `feat(solvers): arms C and MUSCL-C`.

### Task 10: C+L (learned non-negative multipliers through implicit Euler)

**Files:** Create `p3b/solvers/c_plus_l.py`; Test `tests/test_c_plus_l.py`.

**Interfaces:**
- Produces: `TorchColumnRHS(col32)` (the 32-cell hybrid-scheme RHS in torch, float64, checked
  against `Column.rhs`), `Multipliers()` (three softplus MLPs 3 → 32 → 32 → 1 on local cell
  inputs), `implicit_euler_step(rhs, y, inp, h, newton_tol)` with gradients by the implicit
  function theorem, `CPlusL(...)` (B: the exchange coefficient h_v).
- [ ] Tests: the torch RHS equals `Column.rhs` to 1e-12; with all multipliers equal to 1, C+L is C;
  multipliers are ≥ 0 for any weights; the entropy production of a step is ≥ 0 (PSD Onsager
  blocks); the implicit-function gradient matches finite differences to 1e-6.
- Commit `feat(solvers): C+L with implicit-Euler gradients`.

## Self-review notes

- Spec coverage: every arm of PREREG Section 3 maps to a task (U, U+F, P: 5; S, S−M, S−Π: 6;
  S-L, U+proj, U+F+proj: 7; FNO: 8; C, MUSCL-C: 9; C+L: 10). U-long is U with a longer iteration
  count (Plan 3). Training, metrics, the competence gate and timing are Plan 3; pilot rules,
  the power simulation and the Kaggle orchestration are Plan 4.
- The pilot values (h, κ, τ_S, learning rates, iterations) are arguments everywhere.
