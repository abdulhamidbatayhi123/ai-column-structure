# Paper 3B Training, Evaluation and Statistics (Plan 3 of 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train every arm of Plan 2 under one trainer, roll each out over long horizons, compute
every pre-registered metric and violation count, and turn per-trajectory values into the
pre-registered verdict words, attribution labels, coarse-solver statement and headline row,
all from committed code. The pilot rules, the power simulation and the Kaggle orchestration
are Plan 4.

**Architecture:**
- `p3b/eval/` with one job per module:
  - `metrics.py`: e_k, W_k, Q_k, failure, capping;
  - `violations.py`: balance residuals with each arm's own stages, normalisations, the reference
    floor, entropy and projection defects;
  - `rollout.py`: long rollouts of every arm from the encoded initial state;
  - `cost.py`: latency and throughput;
  - `evaluate.py`: one trained model on one set → a per-trajectory table.
- `p3b/train/trainer.py`: one trainer for all learned arms (windows, Adam, cosine decay, fixed
  iteration count, final checkpoint, seed control, parameter counts, wall-clock).
- `p3b/stats/`:
  - `tests.py`: HL, Wilcoxon with Pratt zeros, pigeonhole bootstrap, TOST, Holm;
  - `words.py`: directional words, H4 words, attribution, coarse-solver statement, consistency
    words, the decision list.
- Results are JSON files with the commit, the Paper 1 hashes, the constants, the parameter counts
  and the per-trajectory values. No error value of the pilot is ever printed by arm (Plan 4
  enforces the blinding).

**Tech Stack:** as Plan 2; scipy.stats for the signed-rank and t distributions.

**Spec:** `PREREG_P3B_draft.md` Sections 2 (sets), 4 (training), 5 (metrics and gates), 6
(tests and words), 7 (consistency), 8 (decision list), at e466875 or later; formulation 5.4
(the discrete balance definition).

## Global Constraints

- Training: Adam; learning rate from the pilot per arm; cosine decay to 1e-5; batch 8 windows ×
  20 RK4 steps; one fixed iteration count for every arm (pilot); final checkpoint, no early
  stopping; loss = MSE of decoded standardised channels along each window, outlet included, equal
  weights. U-long: U for the iterations that fit S's measured wall-clock.
- Integrator: RK4 with step h (pilot; an integer number of steps per snapshot interval); the
  projected arms use `structured_step` / `projected_rk4_step` (Plan 2).
- e_k: Paper 1's nRMSE per physical field, averaged over the fields (A: c, q, T; B: T_f, T_s).
- W_k (A) = ∫_des v (c_out − c_in,des) dt; Q_k (B) = ∫_dis v ρ_f c_f (T_out − T_cold) dt, from each
  arm's own outlet; ε = |pred − ref| / |ref|.
- Failure at cycle k: non-finite state, ε > 0.5 or e_k > 0.5; values capped at 0.5 from then on.
- Violations per step: |N̂_{k+1} − N̂_k − Q_RK4[P_N]| over the cycle's feed water; energy over
  |ΔH| W_ref + ∫|wall heat_ref| dt (B: Q_ref + ∫|wall loss| dt); threshold 10 × the reference
  floor (99.9th percentile of the same residual on stored pilot data, ports computed as the
  models compute them); entropy below −τ_S; projection defects.
- Statistics: unit = test trajectory, value = seed-averaged log error ratio arm/U; HL estimate;
  (a) one-sided Wilcoxon signed-rank at α_h with Pratt zeros; (b) (1 − 2α_h) pigeonhole
  bootstrap (trajectories and seeds resampled independently, 10,000 draws); Holm over H1, H2, H4;
  words exactly as PREREG Section 6; H4 TOST with margin m.
- Cost: CPU time on an idle Kaggle session, 4 threads; latency at batch 1, throughput at batch 48
  (solver arms: 48 parallel processes).

## Review Focus

1. A rollout that diverges at cycle k: e and ε capped at 0.5 from k on, never NaN in the tables,
   and the failure cycle recorded.
2. Ties and zeros in the signed-rank test (identical seed-averaged values, e.g. C, which is
   deterministic): Pratt's method, the count reported, no crash at n with all zeros.
3. The pigeonhole bootstrap when a draw repeats one seed five times: the seed-average is still
   defined and the interval is computed.
4. Holm when step-down stops: α_h for unreached hypotheses as PREREG 6 defines it.
5. W_k quadrature: the reference W used in ε comes from the same quadrature on the stored outlet
   channel, so a quadrature error does not count as model error; the exact-port W is reported.

---

### Task 1: Cycle metrics (`p3b/eval/metrics.py`)

**Interfaces:** `cycle_errors(pred_X, true_X, phys, n_cells) -> e (cycles,)` (physical-field nRMSE
per cycle); `release(outlet, raw_inputs, t, system, phase) -> (cycles,)` (W or Q by the trapezoid
on the 100 snapshots of the half plus the closing snapshot); `relative_error(pred, ref)`;
`apply_failure(e, eps, finite) -> (e_capped, eps_capped, failed_cycle)`.
- [ ] Tests: e = 0 for identical inputs and equal to Plan 2's `field_nrmse` on physical fields;
  W of the stored reference outlet by trapezoid agrees with `gates.cycle_release` to the
  quadrature order (stated bound) on a generated trajectory; failure capping after a NaN, after
  ε > 0.5, after e > 0.5; the failure cycle is 1-based.

### Task 2: Violations (`p3b/eval/violations.py`)

**Interfaces:** `step_residuals(lf, z_prev, z_new, stages, h, raw) -> (r_N, r_E)` (each arm's own
RK4 stages; for non-RK4 arms the same quadrature on the arm's own snapshot ports, documented);
`normalisers(ds) -> (feed_water, energy_scale)` per trajectory-cycle; `reference_floor(ds, phys)
-> dict` (99.9th percentile of the model-style residual on stored data); `count_violations(...)`.
- [ ] Tests: zero residual for a step that meets both balances; the reference floor on a generated
  pilot-like file is finite and positive; counts exceed 0 exactly when the residual exceeds
  10 × floor; entropy and projection-defect counts come from `structured_step` flags.

### Task 3: Trainer (`p3b/train/trainer.py`)

**Interfaces:** `TrainConfig(arm, lr, iterations, h_steps, seed, ...)`; `train(arm_model, ds,
cfg) -> history` with `history = {loss, wall_s, params, lr_schedule}`; the arm adapters: latent
arms (RK4 or projected step), FNO (autoregressive at snapshot spacing), C+L (implicit Euler with K
substeps); P adds `penalty_loss` with the weight fixed at iteration 0.
- [ ] Tests: the loss decreases on a two-trajectory toy set for U, S and FNO; the final checkpoint
  is returned (no early stopping); the cosine schedule ends at 1e-5; the same seed gives identical
  weights; the penalty weight equals 1 % of the initial data loss; the frozen autoencoder's
  weights are unchanged after training any arm.

### Task 4: Long rollouts (`p3b/eval/rollout.py`)

**Interfaces:** `rollout(arm, ae, std, ds_traj, n_cycles, h_steps) -> RolloutResult(X, finite,
flags, cpu_s)`: encode snapshot 0, integrate with the snapshot-interval inputs, decode every
snapshot; projected arms collect the stage and projection flags; C, MUSCL-C read their stored
files (Plan 2 `run_c`).
- [ ] Tests: a rollout of an identity-like model over two cycles reproduces the stored snapshots;
  inputs switch at the half-cycle; flags are counted per step; a diverging toy model stops at the
  first non-finite state and reports it.

### Task 5: Cost (`p3b/eval/cost.py`)

**Interfaces:** `latency(arm, ...) -> CPU-s per simulated cycle at batch 1`; `throughput(arm, ...,
batch=48) -> trajectory-cycles per CPU-s`; process-parallel variant for the solver arms.
- [ ] Tests: CPU time (process_time) not wall time; threads fixed to 4; the batch-48 path runs one
  tensor for learned arms and 48 processes for solver arms (a mock of known cost checks the
  arithmetic).

### Task 6: Statistics (`p3b/stats/tests.py`)

**Interfaces:** `hodges_lehmann(x)`; `signed_rank(x, alternative="less") -> (p, n_zeros)` with
Pratt's zero handling (exact for n ≤ 50 without ties, normal approximation with tie correction
otherwise, stated); `pigeonhole_bootstrap(values (traj, seed), stat, draws=10_000, seed) ->
array`; `tost(x, margin, alpha)`; `holm(pvals, alpha) -> dict of (alpha_h, rejected)` with α_h
for unreached hypotheses.
- [ ] Tests: HL against a brute-force Walsh-average median; the signed-rank p against
  scipy.stats.wilcoxon(zero_method="pratt") on fixed vectors; Pratt's zeros counted; the
  bootstrap with a degenerate seed draw stays finite; resampling seeds independently widens the
  interval against a seeds-within-trajectory bootstrap on a constructed crossed example; TOST
  rejects inside the margin and not outside; Holm against a hand-computed example including an
  unreached hypothesis.

### Task 7: Words and the decision list (`p3b/stats/words.py`)

**Interfaces:** `directional_word(a_holds, b_holds, ratio, interval) ->
SUPPORTED|SMALL|REFUTED[+REVERSED]|INCONCLUSIVE`; `h4_word(...)`; `attribution(beats, equiv) ->
list`; `coarse_solver_statement(superior_all, noninferior_any) -> str`; `consistency(p, o)`;
`headline(gate_failed, h1, h2, h4, h3a) -> (row, text)`. `verify/check_decision_list.py` imports
`headline`'s condition table so the CI check and the code cannot diverge.
- [ ] Tests: every row of PREREG Section 8 reached by its defining case; the 256-case partition of
  the checker holds against `headline`; consistency table cells as PREREG Section 7; attribution
  needs TOST equivalence for "the physical information, not the constraint"; the coarse-solver
  statement's three outcomes.

### Task 8: Evaluation and results files (`p3b/eval/evaluate.py`)

**Interfaces:** `evaluate(model_spec, test_set) -> table` (per trajectory and seed: e_k, ε_k,
failure cycle, violation counts, flags, cost); `write_results(path, table, meta)` with the commit,
Paper 1 hashes, constants, parameter counts, the pilot values used.
- [ ] Tests: the table has one row per (trajectory, seed); metadata complete; a results file
  round-trips; the parameter counts of U, U+F, P and FNO lie within ±5 % of S's.

## Self-review notes

- Spec coverage: metrics (5), gates (Plan 2 decoder gate; competence gate from Task 8 tables),
  tests and words (6, 7), decision list (8), cost (5), violations (5, H3a). H3b's Spearman ρ and
  H6's regret use Task 6 primitives and Task 8 tables; their thin wrappers sit in Task 7.
- Pilot values (lr, iterations, h, thresholds, K) are inputs everywhere; Plan 4 fixes them.
