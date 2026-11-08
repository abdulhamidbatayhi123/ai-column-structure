# Paper 3B: compute plan

Revised 2026-11-08 for the Stage-1 design: two systems, a second isotherm, arms S-L, U+F+proj,
C+L, MUSCL-C and FNO, the regret endpoint, the sensitivity analyses and the redesigned pilot.
Earlier versions are in git history (ledger P3B-13).

## 0. Where it runs

- **Kaggle, primary** (free; the author's account, already used by Paper 2 through private
  datasets, kernels and a resumable queue):
  - CPU sessions (4 cores, much more RAM than the laptop) for reference data and arm C;
  - GPU sessions (weekly quota) for the arms that differentiate through the decoder (S, S-L,
    S−M, S−Π, U+F).

  At most 5 concurrent CPU sessions.
- **The laptop, secondary:** i7-1255U, 15.7 GB RAM, of which other sessions often leave under
  1 GB free (0.3 GB on 2026-11-08). It runs only small jobs, checks and analysis, and never a
  heavy job while other sessions hold the RAM.

Training never memory-maps the full data. Each run loads its own training windows into a
compact in-RAM array: a full train pool of one system at float32 is about 0.6 GB.

## 1. Measured, and what is still unknown

| quantity | value | source |
|---|---|---|
| reference cycle, Paper 1's isotherm (n = 3 to 6), 2048 cells | 12 to 30 CPU-s (extrapolated from 128 to 512 cells) | `verify/timing_scaling.json` |
| **reference, MOF-303 isotherm (n ≈ 9), adsorption step** | **3.6 CPU-s at 128 cells, 17.5 s at 256; 512 did not finish within the time limit** | `verify/timing_materials.json` (PARTIAL) |
| structured-arm block of 2560 dynamics evaluations | 4.6 s (unstructured 0.42 s), laptop CPU, 4 threads | 2026-10-12 benchmark |
| verification code against Paper 1's solver | 1.7 against 1.3 CPU-s for the same step | `verify/timing_compare_paper1.py` |

**The main risk is the steep-isotherm reference cost.** It grew about fivefold per doubling
of N between 128 and 256 cells, and no 512-cell value exists yet. The pilot measures 512,
1024 and 2048 cells on Kaggle CPU and decides by the data-cost contingency
(pre-registration, Section 2). The table below assumes 30 CPU-s per System A cycle at the
grid the pilot chooses. If the pilot measures more than 2× that, the contingency applies
before the freeze.

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
