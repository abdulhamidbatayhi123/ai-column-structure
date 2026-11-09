# Paper 3B Pilot Rules, Power Simulation and Kaggle Training (Plan 4 of 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every value PREREG Section 10 leaves to the pilot is set by one committed, hashed script
that prints only the chosen values (blinding). The power of the full decision rules is simulated
from the pilot's upper-80 % SDs, and every training and evaluation run of the study is a resumable
Kaggle job.

**Architecture:**
- `scripts/pilot_rules.py`: reads pilot results files (Plan 3 `write_results`) and prints a JSON of
  the chosen values only. It never prints an error value by arm, and it writes `results/pilot_values.json`
  with its own SHA-256 and the hashes of its inputs.
- `p3b/pilot/` holds the pure rule functions, one per PREREG Section 10 bullet, unit-tested on
  synthetic inputs.
- `scripts/power_sim.py`: simulates the full decision rules (signed-rank + pigeonhole bootstrap +
  Holm + words) on synthetic crossed data with the upper-80 % trajectory and seed SDs. It prints the
  power per hypothesis and the H4 margin rule's outcome.
- `kaggle/train_jobs.py` and `kaggle/train_chunk.py`:
  - each study cell (arm × seed × system × material × size) is one job, trained and evaluated in
    one kernel;
  - the queue of Plan 1 (`kaggle/run_queue.py`) runs them, resumable and never re-pushing.

**Tech Stack:** as Plans 2 and 3; the Kaggle CLI as in Plan 1.

**Spec:** `PREREG_P3B_draft.md` Sections 9, 10, 11 (e466875 or later); `docs/compute_3B.md`.

## Global Constraints

- Learning rate per arm: the best of {3e-4, 1e-3, 3e-3} on 2,000-iteration runs (one seed), by
  validation rollout loss over one cycle.
- Iterations: one value for every arm, the smallest at which every arm's pooled validation loss
  improves by less than 2 % per 1,000 iterations.
- h: the largest in {T_cycle/200, /400, /800} at which S's entropy-violation rate is below 1e-4 of
  steps and the near-dry stiffness check passes.
- Thresholds, set separately per system and material:
  - the violation thresholds (10 × the reference floor);
  - τ_S, τ_ref and κ (10⁻³ × the pilot median of |P∇Ŝ|²);
  - the projection-defect bound: 10 × the pilot's 99.9th percentile of defect steps per 10⁴ steps,
    capped at 10⁻³ of steps.
- C+L substeps K: the smallest in {1, 2, 4, 8} at which C+L with unit multipliers reproduces C's
  ε_W within 1 %.
- SDs: the upper-80 % trajectory and seed SDs for the power simulation and the H4 margin; m
  widened from 1.10 up to 1.20 if power < 0.8 at the Holm worst case, else H4 demoted.
- CSS cycle per system; the decoder gate on pilot data over the d grid; the positive control (U at
  full iterations against U at a quarter of the iterations, through the full rule).
- Blinding: no pilot error value by arm is printed or stored outside the rule functions.
- Kaggle: never touch other projects' kernels; at most the free CPU slots; outputs verified by
  manifest; the compute ceiling of 700 CPU-equivalent hours counted as a running total.

## Review Focus

1. A rule with no admissible value (no h passes, no lr run finishes): the script prints "none
   passes" with the PREREG fallback, never a silent default.
2. Blinding leaks: an exception message or a log line that carries a per-arm loss.
3. The power simulation's crossed structure: the seeds are shared across trajectories, as in the
   pigeonhole bootstrap.
4. Kaggle job identity: a re-run of the queue never re-pushes a finished cell; an UNKNOWN status is
   never treated as absent.
5. The running compute total crosses 700 h: the reduction order of `docs/compute_3B.md` §3 is
   applied in sequence and logged.

---

### Task 1: Rule functions (`p3b/pilot/rules.py`)

**Interfaces:** `choose_lr(losses: dict[arm, dict[lr, float]]) -> dict[arm, lr]`;
`choose_iterations(curves: dict[arm, list[(iter, loss)]]) -> int | None`;
`choose_h(viol_rates: dict[h, float], stiff_ok: dict[h, bool]) -> h | None`;
`thresholds(floors, prod_neg, g2, defects) -> dict`; `choose_K(eps_by_K) -> int | None`;
`upper80_sd(values (traj, seed)) -> (sd_traj, sd_seed)`; `h4_margin(power_by_m) -> m | "demoted"`.
- [ ] Tests, one per rule:
  - ties broken towards the smaller learning rate;
  - iterations: the first point where every arm's improvement is < 2 % per 1,000;
  - h: the largest passing; None when none passes;
  - κ = 1e-3 × median |P∇Ŝ|²;
  - the defect bound is capped at 1e-3 of steps;
  - K: the smallest within 1 %;
  - the upper-80 % SD on a χ² example;
  - the margin rule at 0.8.

### Task 2: The blinded script (`scripts/pilot_rules.py`)

**Interfaces:** `main(results_dir, out_path)`. It reads the pilot results files, applies Task 1's
rules, and writes `pilot_values.json` with the chosen values, the script's own SHA-256 and the
SHA-256 of every input. Its printed output is the chosen values only.
- [ ] Tests:
  - on synthetic results files, the output has exactly the PREREG Section 10 keys;
  - stdout contains no number that equals any per-arm loss in the inputs (blinding check);
  - re-running gives identical output and hashes.

### Task 3: Power simulation (`scripts/power_sim.py`)

**Interfaces:** `simulate(effect, sd_traj, sd_seed, n_traj=48, n_seed=5, reps, seed) -> dict[word,
fraction]`, using Plan 3's `signed_rank`, `pigeonhole_bootstrap`, `holm` and `directional_word`;
`power_table(...)`.
- [ ] Tests:
  - at effect 0 the SUPPORTED rate is ≤ α (within Monte Carlo error);
  - at a large effect the power is near 1;
  - the seed effect is shared across trajectories in the generated data;
  - fixed seeds give identical tables.

### Task 4: Kaggle training jobs (`kaggle/train_jobs.py`, `kaggle/train_chunk.py`)

**Interfaces:**
- `cells(prereg_design) -> list[Cell]` from the design table of PREREG Section 0 (arm, seeds,
  system, material, size);
- `job_spec(cell, pilot_values) -> dict` for `run_queue.py`;
- `train_chunk.py --cell ...` trains the cell's model, evaluates it on its test sets and writes
  `results/<cell>.json`;
- a running compute total in `results/compute_ledger.json`, with the reduction order applied when
  it crosses the ceiling.
- [ ] Tests:
  - the number of cells per arm equals the PREREG design table;
  - job names are stable and unique;
  - a kit-mode run of one tiny cell writes a valid results file;
  - the ledger triggers the first reduction step at 700 h and logs it;
  - a finished cell is skipped on re-run.

## Self-review notes

- Covers every bullet of PREREG Section 10, Section 9's power simulation and Section 11's ceiling.
- The pilot data generation itself uses Plan 1's kit (pilot design sets in `data/design/`).
