# Paper 3B: compute plan

Revised 2026-11-08 for the Stage-1 design: two systems, a second isotherm, arms S-L, U+F+proj,
C+L, MUSCL-C and FNO, the regret endpoint, the sensitivity analyses and the redesigned pilot.
Earlier versions are in git history (ledger P3B-13).

## 0. Where it runs

- **Kaggle, primary** (free; the author's account, already used by Paper 2 through private
  datasets, kernels and a resumable queue):
  - CPU sessions (4 cores, much more RAM than the laptop) for reference data, arm C and, as
    implemented (plan 4, 2026-11-09), every training run: each kernel runs one cell on 4 threads
    with resumable checkpoints against the 12-hour session limit;
  - GPU sessions are not used. The earlier plan put the decoder-gradient arms (S, S-L, S−M, S−Π,
    U+F) on GPU; the code runs in float64 throughout (the projection's 10⁻⁸ rank tolerance is
    below float32 resolution), where consumer GPUs gain little, so the pilot measures their CPU
    cost and the ceiling of Section 3 governs.

  At most 5 concurrent CPU sessions.
- **The laptop, secondary:** i7-1255U, 15.7 GB RAM, of which other sessions often leave under
  1 GB free (0.3 GB on 2026-11-08). It runs only small jobs, checks and analysis, and never a
  heavy job while other sessions hold the RAM.

Training never memory-maps the full data. Each run loads its own training windows into a
compact in-RAM array: a full train pool of one system at float32 is about 0.6 GB.

## 1. Measured, and what is still unknown

Measured on Kaggle CPU, one thread per trajectory, two cycles at the centre and the fastest corner
of each training box (Task 10, `results/timing_reference.json`):

| quantity | value | source |
|---|---|---|
| reference, system B, 512 to 2048 cells | 0.3 to 1.5 CPU-s per cycle | `results/timing_reference.json` |
| reference, system A (MOF-303, Al-fumarate), finite-difference Jacobian, 2048 cells | 24 to 30 CPU-s per cycle in 6 of 7 records (one first cycle 54 s) | same |
| the same, MOF-303 at the box centre | stalled: 1024 cells cycle 1 took 3,753 CPU-s (642,420 evaluations), 2048 cells gave no result in 11 h | same, and `verify/stall_diagnostic.py` |
| **reference, system A, exact Jacobian (the fix), 2048 cells** | **18 to 26 CPU-s per cycle, mean 22.7, in all 8 records (both materials, centre and corner, 2 cycles); the formerly stalled point 25 and 18 s; 1,600 to 2,450 evaluations per cycle** | `results/timing_reference.json` (kernel p3b-timing-a2, kit d222a54) |
| grid convergence, outlet change relative to its swing | about 1e-5 from 512 to 1024 cells, 3e-6 to 1e-5 from 1024 to 2048 (A); 3e-5 (B); Paper 1's gate is 2 % | `results/timing_reference.json` |
| the Paper 1 limit at 2048 cells | c 1.8e-9, q 8.9e-9, T 1.8e-7 K, all within their bounds | `verify/results_v1_2048.json` |
| structured-arm block of 2560 dynamics evaluations | 4.6 s (unstructured 0.42 s), laptop CPU, 4 threads | 2026-10-12 benchmark |

**The arms' cost, measured on Kaggle (2026-11-10).** Kernel p3b-time-arms3, kit fcc5f73, after the
speed-up below (`results/timing_arms.json`; before it, kernel p3b-time-arms at kit 9940214,
`results/timing_arms_pre_speedup.json`). At 128 stored cells, 200 snapshots per cycle and d = 16,
seconds per training iteration (8 windows × 20 snapshot intervals), 10 timed iterations, none
skipped, one thread:

| arm | h = 1 | h = 2 | h = 2 before the speed-up | × U (h = 1) |
|---|---|---|---|---|
| U | 0.080 | 0.149 | 0.151 | 1 |
| U+F | 1.13 | 2.39 | 2.99 | 14 |
| P | 0.38 | 0.77 | 1.11 | 5 |
| S | 1.55 | 3.19 | 4.69 | 19 |
| S-L | 2.08 | 3.84 | 5.65 | 26 |
| S−M | 1.63 | 3.00 | 4.53 | 20 |
| S−Π | 1.67 | 3.34 | 5.07 | 21 |
| FNO | 0.34 | 0.33 | 0.36 | 4 |

- **Threads.** Four threads are no faster than one (the tensors are small), so each run uses one
  thread.
- **Jobs per session** (`results/timing_concurrency_batch.json`). A session's 4 virtual cores behave like
  about two physical ones: n one-thread runs at once give 1.55 × (n = 2) and 2.08 × (n = 4) the
  throughput of one run for S, 1.91 × and 1.99 × for U, but each run takes 1.3 × and 1.9 × longer.
  Four runs per session are the most throughput; an S run at h = 2 and 8,000 iterations then takes
  about 14 h, so it resumes across sessions from its checkpoints.
- **h.** At h = 1, S's rollout fails at the start-up interval, as on the laptop; at h = 2 it does not.
- **Batched evaluation.** S's rollout step for 48 trajectories at once takes 0.19 s per snapshot
  interval at h = 2, against 0.075 s for one: 19 × the throughput. The evaluation and the pilot's
  validation rollouts therefore batch 48 trajectories per tensor (`rollouts` and `rollout_batch` in
  `p3b/eval/rollout.py`, 2026-11-10, as PREREG Section 5 says), each trajectory's result as if rolled out
  alone (`tests/test_eval_rollout_batch.py`). The evaluation line of Section 2 is 74 hours batched,
  against about 1,440 one trajectory at a time, at h = 2 (42 and 1,021 at h = 1).

`scripts/budget.py` turns these into hours. Training at 8,000 iterations comes to **381 CPU-hours at
h = 1 and 756 at h = 2** (one run alone on a session), against 298 budgeted in Section 2. The
budgeted total of about 620 hours becomes about 650 at h = 1 and about 1,060 at h = 2, with the
batched evaluation, the pilot scaled alike and C+L at its budgeted 1 hour per run (not timed). At
h = 2 the whole reduction order of Section 3 saves about 204 hours of training, which leaves about
860, still above the 700-hour ceiling. In wall time, 5 sessions with 4 runs each do about 10 runs'
worth of work at once, so 1,060 hours take about 4 to 5 days of Kaggle time plus queueing.

**C+L, timed (2026-11-11).** C+L had not been timed, and it was far from its 1-hour budget. In the
pilot's K piece it took about 146 s per trajectory-cycle at K = 1 (local, one thread), and a training run
would have taken about 230 hours. The solver was made faster without changing its numerics (commit
212ba17):
- the Jacobian by vectorised reverse mode;
- modified Newton;
- trajectories and training windows integrated together in one Newton solve.

On Kaggle (one thread, `results/timing_cplusl_eval.json`), 16 MOF-303 trajectories over 2 cycles take
39.7 s at K = 1 and 140 s at K = 8, that is 1.2 and 4.4 s per trajectory-cycle. Evaluation is now a
small item, and the pilot's K piece fits one session. The training cost per iteration is being timed
(`p3b-cl-train-timing`) and replaces the 1-hour placeholder at the freeze.

**The speed-up (commit 638d3bc).** A profile of S's iteration showed three repeated computations:
- the isotherm evaluated separately for E^, N^ and S^;
- the projected step decoding again the four RK4 stages that the dynamics had just evaluated;
- a second Jacobian at the point where the projection had converged.

Removing them leaves the arithmetic unchanged (forward results bit for bit, parameter gradients to
2 × 10⁻¹⁶, `tests/test_s_speed_refactor.py`). On Kaggle it cut S's iteration by 32 to 33 % and the
other structured arms' by 22 to 34 %.

**Pilot layout (2026-11-10).** One piece per kernel on one thread (autoencoders on four). The longest
piece is S-L's learning curve: 16,000 iterations at h = 1, about 9.2 hours at 2.08 s per iteration,
plus 16 batched validations. Kaggle sessions end at 12 hours, so the margin is about 2.5 hours (one
earlier measurement on another session ran 50 % slower).

**Resuming across sessions (2026-11-10).** Confirmatory runs at 4 per session (S about 14 hours at h = 2)
need it. A cell still training after `--hours` (default 11) checkpoints and writes `RESUME.json`
(`TrainConfig.deadline`, `train_cell`, `kaggle/train_chunk.py`). A kernel given that output as input
continues it with `--resume-from`, using the same autoencoder, to the same weights; CPU time accumulates
across sessions (`tests/test_resume_sessions.py`). Still open: the queue does not yet push the
continuation kernel and re-point the evaluation jobs to it. That is part of the confirmatory run plan.
Kernels already run each of their commands as a separate process, so 4 runs per session is 4 commands
in one job.

**Still open: the ceiling.** It is in CPU-equivalent hours and was set to bound cost. On Kaggle CPU,
which is free, the binding constraint is wall time (5 sessions at once). Keeping it at 700,
redefining it (for example in session-hours), or reducing the design is the author's decision at the
freeze. It is taken from these costs and the pilot's iteration count and h, which the pilot sets
blind to errors.

**The stall and its fix.** scipy's BDF estimates the Jacobian by finite differences. On the steep
MOF-303 isotherm that estimate failed as the front crossed the isotherm step, so the solver
crawled. Integral states and undershoots were ruled out. The reference simulators now pass the exact Jacobian:
it is computed from the torch right-hand sides (checked against numpy to 1e-12) by one vmapped
forward-mode pass over nine colour seeds, and stored sparse. Both Jacobians give the same stored
trajectory within the solver tolerance (`tests/test_exact_jacobian.py`). MUSCL-C keeps finite
differences on its five-point pattern.

**Data-cost decision** (pre-registration, Section 2): the contingency applies only if the
reference costs more than twice the budgeted 30 CPU-s per system A cycle. Measured with the exact
Jacobian on Kaggle, the largest system A cycle at 2048 cells took 26.1 CPU-s and the mean was
22.7 s, under the budget itself. **The contingency is not applied: N_z stays 2048 for system A and
1024 for system B, as pre-registered**, and the budget of Section 2 stands at 30 CPU-s per system A
cycle (`results/timing_reference.json`, key `data_cost_decision`). The reference is converged by a
wide margin at that grid (outlet changes of 2 to 10 × 10⁻⁶ from 1024 to 2048 cells against Paper
1's 2 % gate).

## 2. Budget (hours; training per run at 8,000 iterations of 640 dynamics evaluations, CPU-equivalent)

*The per-run values and the total below are the pre-pilot budget; the measured costs in Section 1
(2026-11-10) supersede them, and this table is replaced by measured values at the freeze.*

Per-run training: U 0.23, P 0.5, FNO 0.5, C+L 1.0, S-L 1.5, U+F 2.3, S−Π 2.3, S−M 2.9, S 3.25,
U-long 3.25. On a GPU the decoder-gradient arms are expected to run several times faster,
which is measured in the pilot.

| block | CPU-equivalent hours |
|---|---|
| reference data, System A (primary 6,720 cycles + replication 5,328 + H6 designs about 4,680), at 30 s per cycle | 139 |
| reference data, System B (5,328 cycles, at about 5 s per cycle) | 7 |
| pilot: learning-rate sweeps (3 rates × all arm types, 2,000 iterations), 2 seeds × all arms, size-32 SD models, positive control, decoder gate on pilot data | 50 |
| autoencoders (all systems, sizes and seeds, plus the pilot d grid) | 15 |
| A-primary: full data (79), size 32 (43), learning curve 16/64/128 (36) | 158 |
| A-replication: U, U+F, P, S × 3 seeds at full data and at size 32 | 38 |
| System B: U, U+F, S × 5 seeds at full data and at size 32; S-L × 3 seeds | 67 |
| sensitivities: next latent dimension, degraded data | 35 |
| evaluation: about 120 projected-arm rollouts (48 trajectories × 60 cycles, h = T_cycle/200), plain arms, H6 searches | 105 |
| arm C and MUSCL-C on every test set | 5 |
| **total** | **about 620 CPU-equivalent hours** |

Running in parallel on Kaggle (up to 5 CPU sessions, plus GPU for the decoder-gradient
arms) brings the wall time to an estimated 4 to 6 weeks, including queueing. The evaluation
line assumes h = T_cycle/200. At T_cycle/800 it grows about fourfold, so the evaluation
block counts toward the ceiling.

## 3. Ceiling and reduction order

Hard ceiling: **700 CPU-equivalent hours**, measured as the running total. Past it, these
are removed in order (pre-registration, Section 11), all secondary or descriptive:
1. the degraded-data sensitivity;
2. U-long;
3. S−Π;
4. the latent-dimension sensitivity;
5. MUSCL-C;
6. the learning-curve sizes 16 and 128 drop to 2 seeds (size 64 is kept);
7. the System B S-L arm.

Never reduced: the primary arms, the 5 seeds of every primary contrast, the 48-trajectory
test sets, the second isotherm, and System B's U, U+F, U+F+proj and S.

## 4. Organisation

- **Run layout.** One queue per block on Kaggle, after the Paper 2 pattern: a private dataset
  with the code at the frozen commit, kernels per block, and a resumable downloader that
  writes each result file with the frozen commit hash.
- **Local jobs.** They check for at least 1.5 GB of free RAM before starting, and wait
  otherwise.
- **Pilot first.** The pilot runs first and measures every per-run cost. The table above is
  then replaced by measured values before the freeze.
