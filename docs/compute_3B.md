# Paper 3B: compute estimate for this machine

First written 2026-10-12. Revised 2026-11-03 with measured reference timings and with the arms
added after the adversarial review (review F36, ledger P3B-13).

Machine: Intel i7-1255U (10 cores, 12 threads), 15.7 GB RAM, no GPU, about 16 GB free disk.
About 1.5 GB of RAM is typically free. On 2026-11-03 only 0.4 GB was free, with four other
sessions' jobs running. Rule: one heavy job at a time.

## 0. What is measured and what is extrapolated

| measured | value | source |
|---|---|---|
| verification code against Paper 1's solver, same adsorption step (256 cells, 0.8 t_st) | 1.7 CPU-s against 1.3 CPU-s; 1,408 against 1,311 RHS evaluations | `verify/timing_compare_paper1.py`, 2026-11-03, single thread |
| one full cycle (adsorption 0.8 t_st plus desorption 0.6 t_st at 353 K), c_w = 33.6, MOF-303-like isotherm, Glueckauf kinetics (k = 5.4×10⁻⁴ s⁻¹) | n = 3: 2.5 / 3.8 / 6.1 CPU-s at 128 / 256 / 512 cells. n = 6: 3.0 / 5.1 / 8.1 CPU-s | `verify/timing_scaling.json`, single thread, machine under load |
| RHS evaluations per step | nearly independent of N: about 1,350 to 1,560 (adsorption) and 660 to 980 (desorption) | same |
| structured latent dynamics, one block of 2560 evaluations (decoder 16 → 256 → 256 → 384, three decoder gradients with `create_graph`, projector, backward) | 4.61 s; unstructured 0.42 s | micro-benchmark, 2026-10-12, torch 2.12 CPU, 4 threads |
| Paper 1's own figure | about 17 worker-seconds per breakthrough at 2000 cells (horizon 2.5 t_st), 8 workers | Paper 1's `cost_accounting.py`, v2 manifest |

**Reference cost per cycle at 2048 cells: about 12 to 30 CPU-seconds**, extrapolated from
512 cells over two doublings. The whole-cycle cost grew by 1.6× for both n = 3 and n = 6
from 256 to 512 cells. The per-phase factors range from 1.4× (adsorption) to 1.9×
(desorption), and the upper end assumes the faster factor for the whole cycle. It agrees with Paper 1's
own figure. The budget uses 30 s per cycle, the upper end.

**Two cautions.**
1. **Memory, not CPU, decides the wall time.** A 1024-cell timing run started while 0.4 GB
   was free was paged out to a 13 MB working set and crawled. Its result was discarded. Data
   generation needs at least 1.5 GB free and no competing heavy job. That is a precondition
   in the run chain, checked before each batch.
2. **Faster kinetics are much stiffer.** The independent review measured 199 s at 512 cells
   for one adsorption step with k_LDF = 0.05 s⁻¹, its own illustrative choice and 100 times
   the pellet-derived value. 3B derives k from the pellet, but the pilot measures the cost
   over the sampled range of v, and the data-cost contingency (pre-registration, Section 2)
   applies if it exceeds 2× this budget.

## 1. Reference data

| set | cycles | core-hours at 30 s per cycle |
|---|---|---|
| pilot | 16 × 10 = 160 | 1.3 |
| train pool | 256 × 6 = 1536 | 12.8 |
| validation | 32 × 6 = 192 | 1.6 |
| long validation | 4 × 60 = 240 | 2.0 |
| long test | 48 × 60 = 2880 | 24.0 |
| OOD test | 24 × 30 = 720 | 6.0 |
| replication material: train pool, validation, long test | 1536 + 192 + 2880 = 4608 | 38.4 |
| **total** | **10,336** | **86.1** |

With 4 worker processes (about 150 MB each), that is **about 21.5 hours of wall time** on a
free machine. Storage at 128 cells × 3 fields × 200 snapshots per cycle in float32, plus the
outlet channels and the per-block wall-heat field (128 values per snapshot), is about
0.41 MB per cycle, **about 4.2 GB on disk**, which fits in the 16 GB free. Training reads it
through a memory map.

## 2. Training and evaluation

Training design: 16 windows of 10 RK4 steps per iteration (640 dynamics evaluations), and
8,000 iterations for every arm. The pilot may change the iteration count, but only to one
value shared by all arms (pre-registration, Section 9).

| run | per iteration (extrapolated from the benchmark) | per run |
|---|---|---|
| U | 0.105 s | 0.23 h |
| U+F (all of S's physical features as inputs, including the three latent gradients) | about 1.0 s (as S−Π: the same three differentiable decoder gradients) | 2.3 h |
| U-long (secondary, compute-matched to S) | 0.105 s, for S's wall-clock | 3.25 h |
| P (penalties on 160 loss states) | about 0.2 s | 0.5 h |
| S (with the Gauss-Newton projection, +25 %) | 1.45 s | 3.25 h |
| S−M (projection, no metriplectic term) | about 1.3 s | 2.9 h |
| S−Π (metriplectic term, no projection) | about 1.0 s | 2.3 h |
| U+proj | no training (U's weights, projection at rollout) | evaluation only |
| autoencoder, full data | n/a | about 0.5 h (proportionally less on subsets) |
| FNO (descriptive arm) | small 1-D operator on 128 cells | about 0.5 h (estimate; measured in the pilot) |
| evaluation rollout, 60 cycles × 48 trajectories, projected arms | n/a | about 0.75 h |

Peak memory of one training process: about 0.4 GB for torch and the model, plus the
memory-mapped data pages.

## 3. Total

| block | hours |
|---|---|
| reference data, all sets, both materials | 21.5 |
| pilot training (2 seeds × U, U+F, P, S, S−M, S−Π at full iterations: 2 × 11.48 h) and timings | 23 |
| autoencoders (both materials, all sizes and seeds) | 9 |
| primary, full data: U, U+F, P, S, S−M, S−Π × 5 seeds (5 × 11.48 h) | 57 |
| H2 at size 32: U, U+F, P, S × 5 seeds (5 × 6.28 h) | 31 |
| H2 learning curve at sizes 16, 64 and 128: U, P, S × 3 seeds (9 × 3.98 h) | 36 |
| U-long, compute-matched (3 seeds × 3.25 h) | 10 |
| replication material: U, U+F, P, S × 3 seeds at full data and at size 32 (6 × 6.28 h) | 38 |
| FNO, descriptive (3 seeds at full data and at size 32) | 3 |
| evaluation rollouts (projected arms, U+proj, arm C, FNO; 48 long-test trajectories; both materials) | 26 |
| **total** | **about 254 hours: about 10.6 days of continuous background running on a free machine** |

Uncertainty: a factor of 0.5 to 2 on the training blocks (the benchmark used a stand-in
entropy of the right shape), and the measured scaling for the data. The design grew from 187
to 254 hours on 2026-11-05. The second material, the larger test sets and the FNO arm were
added after the author's approval, at the author's instruction that time is not a constraint.

## 4. What fits and what does not

- **Fits:** the whole study (H1 to H5 and the replication, all arms and ablations, arm C,
  the FNO arm) in about 11 days of background running, one job at a time, inside the RAM limit, *when the machine is not
  shared with other heavy jobs*. With other sessions' jobs running, as on 2026-11-03,
  nothing heavy should start.
- **Does not fit at all:** 10 seeds everywhere, or a full-order learned model on 2048 cells.
  Neither is in the design.
- **A GPU would help but is not required.** The arms that differentiate through the decoder
  (S, S−M, S−Π, U+F) dominate, at about 170 of the 254 hours. The program's shared GPU account
  would need the author's and their professor's approval, so the plan assumes none.

## 5. How the run is organised

- One `chain_p3b.sh` with done-tests per step, in the background, appending to logs,
  resumable (Paper 1's chain pattern).
- Before every batch, the chain checks that at least 1.5 GB is free, and waits if not.
- Data generation (4 workers) and training never overlap.
- torch is pinned to 4 threads, so the laptop stays usable.
- Every results file carries the frozen commit hash.

## 6. Pre-declared reduction order (if the pilot measures more than 2× this budget)

1. Drop the S−Π ablation (keep S−M, the more informative one) and U-long.
2. H2 sizes 16, 64 and 128 drop to 2 seeds (size 32 keeps 5, because H2 is tested there).
3. Training iterations drop to the smallest value at which the pilot's pooled validation loss
   of every arm has stopped improving by more than 2 % per 1,000 iterations. The same value
   is used for every arm.
4. For the data: the data-cost contingency of the pre-registration, Section 2
   (N_z = 1024 if it is grid-converged, then shorter long tests, then a smaller train pool).
5. Never reduced: the 5 seeds of the primary comparison, the 48 long-test trajectories, the
   replication material, and
   the arms U, U+F, U+proj, P and S.
