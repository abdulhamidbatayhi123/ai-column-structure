# Paper 3B: decisions for the author before the freeze

Updated 2026-11-10. Each item says what is decided, the evidence, the options and when it must be
decided. Numbers are from the files named.

## 1. Axial conduction δ (system A) and the exchange coefficient h_v (system B): needed before the pilot results are used

**What.** PREREG Section 2 sets system A's axial conductivity `k_z = 0.10 + δ ρ_g C_pg v d_p`, with
δ = 0.5 "to be verified in the full text". System B's fluid conduction uses δ = 0.5 too, with a
constant h_v = 2 × 10⁴ W m⁻³ K⁻¹ and rock d_p = 5 cm "to be verified against h_v".

**Evidence** (Claude Science T7, hub message 20261108-1955; read through secondary sources only):
- System B: δ = 0.5 is right (Wakao and Kaguei, fluid phase, Re > 0.8; the design has Re 11 to 248).
- System A, a one-temperature model, needs the one-phase value 0.7 to 0.8 (Yagi et al. 1960), since it
  must also carry the fluid-particle resistance. T7 proposes δ = 0.75.
- d_p = 5 cm is consistent with h_v = 2 × 10⁴. h_v varies 2- to 3-fold with v, so T7 proposes h_v(v)
  per trajectory from the Nusselt correlation. This keeps the structure: the model is
  positive-semidefinite for any h_v ≥ 0.

T7 asks that both primary sources (Yagi, Kunii and Wakao 1960; Wakao and Kaguei 1982) be read before
δ is frozen.

Live check (2026-11-10). The record is confirmed: Yagi, Kunii and Wakao, "Studies on axial effective
thermal conductivities in packed beds", AIChE J. 6(4), 543–546 (1960), doi:10.1002/aic.690060407. Its
abstract reports that the axial conductivity rises with air flow more steeply than the radial one and
that the two coincide as Re → 0. The abstract gives no coefficient, and the full text is paywalled
(Wiley), so 0.7 to 0.8 is not yet verified. A review notes that published correlations for the axial
conductivity differ by up to an order of magnitude. The full text (YTÜ library) settles it.

**Why now.** Both values enter the reference data. The 64 pilot trajectories (2026-11-10) were
generated with δ = 0.5 and a constant h_v. The pilot's thresholds are per system and material (τ_ref, the violation
thresholds, κ, the defect bound) and come from these data. If δ or h_v changes:
- the pilot data is regenerated: 9 Kaggle kernels, half a day to a day;
- the pilot phases rerun on the new data.

If they stay, nothing is redone.

**Options.** (a) Keep δ = 0.5 and a constant h_v. State that system A's conductivity is
the fluid-phase value, a lower bound for a one-temperature model. (b) δ = 0.75 for A and h_v(v) for B,
as T7 proposes, after the primary sources; regenerate the pilot data.

## 1b. The decoder gate's balance criterion is at risk: needed at the freeze (added 2026-11-11)

**What.** PREREG 5.1 requires the full-data autoencoders to reconstruct the water inventory and the
energy to within 1 % of the cycle's release (|N̂ − N| ≤ 0.01 W_ref, taken as the maximum over every
validation snapshot). If no d passes, with the MLP or the conv decoder, the system is "not testable"
(PREREG 4), and H1 to H5 cannot be answered for it.

**Evidence (pilot data only; no arm trained; `results/decoder_diag/decoder_diag.json`).**
For MOF-303 (32 training trajectories), the field error passes (nRMSE 0.009 to 0.016 against 0.02),
but the balance errors are 5 to 20 times the threshold (N 0.07 to 0.20, E 0.05 to 0.12). Training five
times longer (300 epochs) lowers the field error and raises the balance error, and the conv decoder
is no better. A plain reconstruction loss does not target the conserved totals. For Al-fumarate and B
the pilot has only 6 training trajectories and the networks overfit. The 256-trajectory study set will
help there, but there is no sign it fixes MOF-303's balance error.

**Second diagnostic** (`results/decoder_diag/decoder_diag2.json`; MOF-303, d = 32, the 16
long validation trajectories):

| training trajectories | loss | field | outlet c | balance N (max) | balance E (max) | N, 90th pct | snapshots with N > 0.01 |
|---|---|---|---|---|---|---|---|
| 8 | MSE | 0.044 | 0.067 | 0.251 | 0.156 | 0.040 | 59 % |
| 16 | MSE | 0.023 | 0.036 | 0.175 | 0.106 | 0.031 | 37 % |
| 32 | MSE | 0.014 | 0.024 | 0.117 | 0.068 | 0.010 | 10 % |
| 32 | MSE + 0.1 × balance² | 0.015 | 0.023 | 0.086 | 0.055 | 0.008 | 7 % |
| 32 | MSE + 1 × balance² | 0.018 | 0.024 | 0.047 | 0.028 | 0.007 | 6 % |

- The maximum falls with data roughly as n^−0.55. Extrapolated to the study's 256 trajectories with
  plain MSE it is about 0.04: still four times the threshold. This is an extrapolation, not a measurement.
- The maximum is phase-locked. After cycle 7 the worst snapshot repeats to four digits in every cycle
  (one trajectory, mid-cycle), so it is a systematic reconstruction bias at one state of the cyclic
  steady state, not noise. Taking the maximum over 60 cycles therefore adds little over one cycle.
- The balance term at λ = 1 halves both balance errors at a small cost in field error (0.014 to 0.018,
  still under 0.02). For B it brings the energy balance under the threshold (0.0125 to 0.0064). B then
  fails only on the outlet temperature (0.022 against 0.02).
- The outlet concentration of MOF-303 (0.024 at 32 trajectories) also fails at pilot size; it is not
  helped by the balance term.
- Al-fumarate (6 trajectories) overfits under every loss: the pilot cannot inform it.

**Third diagnostic** (`results/decoder_diag/decoder_diag3.json`; MOF-303 at 32 trajectories unless stated):

| loss | d | field | outlet c | balance N | balance E |
|---|---|---|---|---|---|
| MSE + 1 × balance² (diag2) | 32 | 0.018 | 0.024 | 0.047 | 0.028 |
| MSE + 3 × balance² | 32 | 0.021 | 0.028 | 0.081 | 0.055 |
| MSE + 10 × balance² | 32 | 0.027 | 0.031 | 0.067 | 0.043 |
| MSE + 30 × balance² | 32 | 0.037 | 0.039 | 0.106 | 0.062 |
| MSE + 3 × balance² | 64 | 0.021 | 0.026 | 0.060 | 0.030 |

- A larger weight does not help. Above λ = 1 the field error passes 0.02 and the balance errors do not fall
  further (one seed each, so the order among λ = 3, 10 and 30 is not reliable). d = 64 does not change this.
- With λ = 1 the balance error falls faster with data than with plain MSE: balance N 0.339, 0.131 and 0.047 at
  8, 16 and 32 trajectories (about n^−1.4, against n^−0.55 for MSE). Extrapolated to 256 trajectories that is
  about 0.003, under the gate. This rests on three points and one seed: it is a reason for hope, not evidence.
- B (6 trajectories): λ = 10 gives balance E 0.0045 and outlet T 0.021, against the threshold 0.02.

**The pilot no longer stops on this.** The pilot gate is a risk report (PREREG 10). d is set at full data
by `scripts/decoder_choice.py` (PREREG 4, 5.1).

**Options.** (a) Keep the gate and the loss as registered, and accept a real chance of "not testable".
(b) Pre-freeze revision: train the autoencoder on MSE + λ × (the gate's own balance errors)² with λ
fixed before the freeze, so the decoder is trained for what the gate checks. This is blind to the arms,
and every arm shares the autoencoder. (c) Relax the criterion (for example an RMS over snapshots instead of
the maximum). This is not recommended: it changes the test after seeing the pilot. Recommended: (b) with λ = 1.
Before diag3 I wrote down a rule (the largest λ in {1, 3, 10, 30} whose MOF-303 field error stays at or
below 0.015). It selects none, because λ = 1 already gives 0.018. λ = 1 is proposed instead because it is
the only tested weight that lowers the balance errors without failing the field criterion, and its errors
fall fastest with data. This choice was made after seeing pilot-data diagnostics (no arm trained or
evaluated), and the change log must say so. Even with (b), MOF-303 may be "not testable" at the registered
thresholds, and the paper must be ready to report that honestly.

## 2. The compute ceiling: needed at the freeze

**What.** PREREG Section 11: a hard ceiling of 700 CPU-equivalent hours, with a reduction order that
removes only secondary items.

**Evidence** (Kaggle, `results/timing_arms.json`, `results/timing_concurrency_batch.json`,
`scripts/budget.py`, `docs/compute_3B.md`). The structured arms cost 19 to 26 × U per training iteration.
The study totals:

| h | training | total |
|---|---|---|
| 1 | 381 CPU-hours | about 650 |
| 2 | 756 CPU-hours | about 1,060 |

The totals assume 8,000 iterations and batched evaluation. The pilot sets h and the iteration count
blind to errors. S fails at h = 1 on the start-up interval in every test so far, so h = 2 is likely.
At h = 2 the whole reduction order saves about 204 hours, which leaves about 860. On Kaggle (free, 5
sessions, 4 runs each) 1,060 hours take about 4 to 5 days of compute.

**Update 2026-11-11: C+L.** These totals carry C+L at its budgeted 1 hour per run, which was never
timed. Measured on Kaggle after two rounds of speed-ups that keep its numerics (`results/timing_cplusl_train.json`):
8.1 s per iteration at the pilot's K = 4, that is about 18 hours per run and about 108 hours for the
six C+L cells. The totals rise by about 100 hours: about 750 at h = 1 and 1,160 at h = 2. The cells
resume across Kaggle sessions, so this is time, not feasibility.

**Options.**
- (a) Keep 700 and apply the reduction order. At h = 2 it is not enough, and further cuts would reach
  the primary design.
- (b) Raise the ceiling to the measured need with a margin (for example 1,200 CPU-hours), keeping the
  reduction order beyond it.
- (c) Redefine the ceiling in Kaggle session-hours, since the compute is free.

Any change is decided now, before the pilot sets h and before any error is seen, and it goes in the
rationale.

## 3. Already decided or set by rule (for reference)

- τ_ref, κ, τ_S, the violation thresholds and the defect bound are set by the pilot rules (PREREG
  Section 10). They are not choices.
- N_z stays 2048 (A) and 1024 (B): the data-cost contingency was not triggered (`docs/compute_3B.md`).
- Venue route: a standard pre-registered article; primary target Machine Learning: Science and
  Technology (hub T4).
