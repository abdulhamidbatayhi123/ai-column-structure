# Paper 3B Pilot Runner (Plan 5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce, from the pilot data sets, every input file that `scripts/pilot_rules.py` reads,
by committed jobs that never print an error value by arm. That closes PREREG Section 10 end to end.

**Architecture:**
- `p3b/pilot/runs.py` holds one function per pilot input, each writing its JSON file into a
  pilot-results directory. The functions are:
  - `lr_sweep`;
  - `iteration_curves`;
  - `h_check`;
  - `threshold_inputs`;
  - `cplusl_K`;
  - `sds`;
  - `css`;
  - `decoder_gate_grid`;
  - `positive_control`;
  - `wall_per_iter`.
- Each function trains and evaluates through Plans 3 and 4 (trainer, rollout, evaluate,
  `train_cell` building blocks) on the pilot sets only: A_primary pilot_train and pilot_val,
  A_replication pilot, B pilot.
- `kaggle/pilot_jobs.py` turns them into Kaggle jobs. The heavy ones, the learning-rate sweeps and
  the 2-seed runs, become one kernel per arm and system. `scripts/pilot_collect.py` assembles the
  downloaded pieces into the directory `pilot_rules.py` reads.

**Spec:** PREREG Section 10 (plan4 branch, merged with plan1 f810d0b); `p3b/pilot/rules.py` (the
rules consuming these files); `docs/compute_3B.md` (pilot block, 50 h).

## Global Constraints

- **Pilot data only.** It is never reused, and no confirmatory set is opened.
- **Blinding.** The per-arm losses live only inside the input files, which `pilot_rules.py` reads
  and never prints. Job logs print progress counts, never losses.
- **lr sweep.** 3 rates × each learned arm (U, U+F, P, S, S-L, S−M, S−Π, FNO, C+L), 2,000
  iterations, one seed, scored by the validation rollout loss over one cycle.
- **Iteration curves.** The pooled validation loss every 1,000 iterations, up to 16,000, at the
  chosen learning rate.
- **h check.** S's entropy-violation rate (production below −τ_S per step) and the near-dry
  stiffness check at h_steps ∈ {1, 2, 4}, on pilot_val. The stiffness criterion is the share of
  snapshots violating S's inequality and the correction size (p3b.pilot.diagnostics).
- **Thresholds.** The reference floors and the reference production rates come from the stored
  pilot data. The pilot median of |P∇Ŝ|² and the defect steps per step come from S's pilot runs.
- **K.** C+L with unit multipliers against C (BDF) on the pilot trajectories, ε_W, for K ∈ {1, 2,
  4, 8}.
- **SDs.** Two seeds × every primary arm at full pilot data and at size 32, giving the log error
  ratios per trajectory and seed for H1, H2, H4 and H5.
- **CSS.** The per-trajectory CSS cycle from the stored pilot_val references (gates.css_cycle).
- **Decoder gate.** The autoencoder over d ∈ {16, 24, 32, 48, 64} on pilot data, scored by
  `decoder_gate`.
- **Positive control.** U at full iterations against U at a quarter of the iterations, through the
  full decision rule.
- **Wall per iteration.** Measured training seconds per iteration for U and S, which set U-long's
  iteration count.

## Review Focus

1. A blinding leak through a job log, an exception message or a progress line.
2. A pilot run that diverges: it is recorded as such, and the rule sees "not finished" (NaN),
   never a silent zero.
3. Keys: system and material keys via `pilot_key` everywhere.
4. The h check uses the τ_S that comes from the reference data, never from S itself.
5. Reproducibility: fixed seeds, and the hashes of every input in each output.

### Task 1: `threshold_inputs`, `css`, `decoder_gate_grid` (data-only inputs; local)
### Task 2: `lr_sweep` and `iteration_curves` (training; Kaggle jobs per arm and system)
### Task 3: `h_check`, S's |P∇Ŝ|² and defects (S rollouts on pilot_val)
### Task 4: `cplusl_K` and `wall_per_iter`
### Task 5: `sds` and `positive_control` (two-seed runs; Kaggle)
### Task 6: `pilot_jobs.py` and `pilot_collect.py`, and an end-to-end dry run on tiny data feeding
`pilot_rules.main`

**Task 6 design (2026-11-09).** The pilot's choices depend on each other: d (decoder gate) comes
before any training, the learning rates come before the curves, the iterations before S's h check,
and h before the two-seed runs. So the Kaggle pilot runs in five phases. Between phases,
`scripts/pilot_collect.py stage N` applies only the rule functions of `p3b/pilot/rules.py` that the
next phase needs, and writes `pilot_stage.json`. That file holds chosen values only (d, K, tau_S,
lr, iterations, h_steps), the values `pilot_rules.py` prints anyway, so no error value leaves a
results file.

| phase | jobs (one kernel each) | piece files | stage written after it |
|---|---|---|---|
| 0 | `data-<key>` (3 keys), `cplk-A_MOF-303` (C+L runs on system A primary only) | `data_<key>.json` (CSS, decoder gate, reference noise), `cplusl_K_<key>.json` | d, K, tau_S per key |
| 1 | `ae-<key>` (3), `lr-<arm>` (9 arms, system A primary, after its autoencoder) | `lr_<arm>.json` | lr per arm |
| 2 | `curve-<arm>` (9) | `curve_<arm>.json` | iterations |
| 3 | `s-<key>` (3: S at h_steps 1, h check, raw g2, defects), `wall` (U and S) | `s_<key>.json`, `wall_per_iter.json` | h_steps per key |
| 4 | `two-<arm>-s<seed>` (U and S, seeds 0 and 1), `pcq-s<seed>` (U at a quarter of the iterations) | `rows_<arm>_s<seed>.json` | - |

`scripts/pilot_collect.py final` then assembles the ten input files of `pilot_rules.py`.

The data layout in the kit is `data/pilot/<key>/{train,validation}`:
- system A primary: pilot_train and pilot_val;
- A_replication and B (one set of 8 × 60): design indices 0-5 for training and 6-7 for validation.

CSS and the reference noise use every trajectory of a key that has the key's full cycle count.

C+L's learning-rate sweep and curve use a deterministic validation loss: one cycle from each
validation trajectory's first stored state, standardised MSE in its own variables. It is the
counterpart of the latent arms' one-cycle rollout loss.

Each task: the test first (tiny generated data), RED, implement, GREEN, then commit. Each output
JSON has a fixed schema, the one `pilot_rules.py` already reads.
