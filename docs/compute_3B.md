# Paper 3B: compute estimate for this machine

Written 2026-10-12. Machine: Intel i7-1255U (10 cores, 12 threads), 15.7 GB RAM of which about
1.5 GB is typically free, no GPU, about 16 GB free disk. Rule: one heavy job at a time.

Two numbers below are measured. Everything else is extrapolated from them and labelled.

| measured | value | source |
|---|---|---|
| reference solve, one breakthrough (N_z = 2000, BDF, horizon 2.5 t_st) | 16.6 worker-seconds | Paper 1, `cost_accounting.py`, CONTINUE_HERE.md |
| structured latent dynamics, one block of 2560 evaluations (decoder 16 → 256 → 256 → 384, three decoder gradients with `create_graph`, projector, backward) | 4.61 s | micro-benchmark run 2026-10-12, torch 2.12 CPU, 4 threads |
| the same block, unstructured | 0.42 s | same benchmark |

The structured step costs **11 times** the unstructured one. The cause is three
vector-Jacobian products through the decoder per evaluation, each kept differentiable for
training. This ratio drives the whole budget. It is the main reason the design trains on
short windows (below).

## 1. Reference data

A cycle runs for `(1 + β) α t_st ≤ 2.4 t_st` of simulated time, about one breakthrough's worth
of simulated time. Temperature switches restart BDF's step-size control, so a factor of 1.5
is allowed: **about 25 worker-seconds per cycle** (extrapolated; the pilot measures it). The
u-form solver gives T explicitly from (n, a, u), so it costs no more than Paper 1's. Entropy
is computed afterwards, outside the solver loop.

| set | cycles | core-hours |
|---|---|---|
| pilot | 16 × 10 = 160 | 1.1 |
| train pool | 256 × 6 = 1536 | 10.7 |
| validation | 32 × 6 = 192 | 1.3 |
| long test | 32 × 60 = 1920 | 13.3 |
| OOD test | 16 × 30 = 480 | 3.3 |
| **total** | **4288** | **29.8** |

With 4 worker processes (about 150 MB each, 0.6 GB in total), that is **about 7.5 hours of
wall time**, run as a background chain while no training job runs.

Storage at 128 cells × 3 fields × 200 snapshots per cycle in float32 is 307 KB per cycle,
**1.3 GB on disk** in total. Training reads it through a memory map, so the resident set
stays small.

## 2. Training and evaluation

Training design chosen for this budget: 16 windows of 10 RK4 steps per iteration (640
dynamics evaluations), 8,000 iterations for every arm. The pilot may change the iteration
count, but only to one value shared by all arms.

| run | per iteration (extrapolated from the benchmark) | per run |
|---|---|---|
| U | 0.105 s | 0.23 h |
| P (penalties on 160 loss states) | about 0.2 s | 0.5 h |
| S | 1.15 s | 2.6 h |
| S−M, S−Π | about 1.0 s | 2.3 h each |
| autoencoder, full data | n/a | about 0.5 h (proportionally less on subsets) |
| evaluation rollout, 60 cycles × 32 trajectories, arm S (forward only) | n/a | about 0.5 h |

Peak memory of one training process: about 0.4 GB for torch and the model, plus the
memory-mapped data pages. That fits in about 1.5 GB if nothing else heavy is running.

## 3. Total

| block | hours |
|---|---|
| pilot (data, 2 seeds × all arms, timing) | 10 |
| reference data | 7.5 |
| autoencoders (5 seeds at full data, plus the subsets) | 7 |
| primary, full data: U, P, S, S−M, S−Π × 5 seeds | 40 |
| H2 learning curve: U, P, S at sizes 16, 64 and 128 × 3 seeds, and at size 32 × 5 seeds | 47 |
| evaluation rollouts and arm C (coarse solvers: minutes) | 10 |
| **total** | **about 122 hours, about 5 days of continuous background running** |

Uncertainty: a factor of 0.5 to 2. The benchmark used random inputs and a stand-in entropy
of the right shape. The real entropy evaluates logarithms and the potential per cell. It
costs a little more per evaluation but does not change the number of decoder passes, which
dominate.

A GPU would cut the structured runs by about 10×. It speeds the work up but is not
required.

## 4. What fits and what does not

- **Fits:** the whole primary study (H1 to H5, with all ablations and arm C) on this laptop
  in about a week of background running, one job at a time, inside the RAM limit.
- **Does not fit comfortably:** the conditional second-material replication. It adds about
  25 hours at 3 seeds, primary arms only. It is pre-declared as conditional for this reason.
- **Does not fit at all:** 10 seeds everywhere (about 2.2× the training block), or a
  full-order learned model on 2000 cells. Neither is in the design.

## 5. How the run is organised

- One `chain_p3b.sh` with done-tests per step, in the background, appending to logs,
  resumable (Paper 1's chain pattern).
- Data generation (4 workers) and training never overlap.
- torch is pinned to 4 threads, so the laptop stays usable.
- Every results file carries the frozen commit hash.

## 6. Pre-declared reduction order (if the pilot measures more than 2× this budget)

1. Drop the S−Π ablation (keep S−M, the cheaper and more informative one).
2. H2 sizes 16 and 128 drop to 2 seeds (size 32 keeps 5, because H2 is tested there).
3. Training iterations drop to the smallest value at which the pilot's pooled validation loss
   of every arm has stopped improving by more than 2 % per 1,000 iterations. The same value
   is used for every arm.
4. Never reduced: the 5 seeds of the primary comparison, the 32 long-test trajectories, or
   the 60-cycle horizon.
