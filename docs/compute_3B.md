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
| reference, system A, exact Jacobian (the fix), 2048 cells | to be filled from kernel p3b-timing-a2; laptop: the stalled cycle 61 CPU-s with 2,333 evaluations | `verify/stall_exact_jacobian.py` |
| grid convergence, outlet change relative to its swing | about 1e-5 from 512 to 1024 cells, 3e-6 to 1e-5 from 1024 to 2048 (A); 3e-5 (B); Paper 1's gate is 2 % | `results/timing_reference.json` |
| the Paper 1 limit at 2048 cells | c 1.8e-9, q 8.9e-9, T 1.8e-7 K, all within their bounds | `verify/results_v1_2048.json` |
| structured-arm block of 2560 dynamics evaluations | 4.6 s (unstructured 0.42 s), laptop CPU, 4 threads | 2026-10-12 benchmark |

**The stall and its fix.** scipy's BDF estimates the Jacobian by finite differences. On the steep
MOF-303 isotherm that estimate failed as the front crossed the isotherm step, so the solver
crawled. Integral states and undershoots were ruled out. The reference simulators now pass the exact Jacobian:
it is computed from the torch right-hand sides (checked against numpy to 1e-12) by one vmapped
forward-mode pass over nine colour seeds, and stored sparse. Both Jacobians give the same stored
trajectory within the solver tolerance (`tests/test_exact_jacobian.py`). MUSCL-C keeps finite
differences on its five-point pattern.

**Data-cost decision** (pre-registration, Section 2): the contingency applies only if the
reference costs more than twice the budgeted 30 CPU-s per system A cycle. The decision is taken
on the exact-Jacobian Kaggle timing (p3b-timing-a2) and written here.

## 2. Budget (hours; training per run at 8,000 iterations of 640 dynamics evaluations, CPU-equivalent)

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
