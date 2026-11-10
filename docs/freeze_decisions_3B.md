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
