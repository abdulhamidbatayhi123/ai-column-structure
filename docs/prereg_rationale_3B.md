# Rationale and change history of the Paper 3B pre-registration

The pre-registration (`PREREG_P3B_draft.md`) states rules only. This file gives the reason
for each rule that is not self-evident, and the history of changes after the author's
approval on 2026-11-05. Review codes:
- F1 to F36: adversarial review, 2026-11-03;
- R2-n: second round, 2026-11-03;
- R3-n: referee round 3, 2026-11-05 (`research/review_round3_referee_2026-11-05.md`);
- CS-M1, CS-M2, CS-P1 to CS-P6: Claude Science review, 2026-11-07 (hub
  `projects/ai-column-structure/REVIEW_cs.md`).

## Changes after the author's approval (2026-11-05 to 2026-11-08)

| change | reason | code |
|---|---|---|
| The question was reframed: constraint against information against projection, and fixed against learned potentials | a five-qualifier gap reads as "nobody did exactly this"; the general question is the one a broad reader recognises, and the design answers it unusually cleanly | R3-1 |
| A second physical system, a two-temperature thermocline store with flow reversal | a second isotherm on the same column and cycle generalises across parameters, not systems | R3-3 |
| Arm S-L (learned scalar potentials) | the claimed novelty, fixed physical E and S, was not tested against learned E and S (tLaSDI, GFINN) | R3-5 |
| Arm U+F+proj | U+F and U+proj each control one confound; "the constraint in training" needs both at once | R3-6 |
| Arm C+L (coarse consistent solver plus a learned non-negative closure); arm MUSCL-C (descriptive) | the literature's strongest 1-D approach is structure in the discretisation with a learned closure; the most accurate non-learned scheme belongs in the accuracy-cost comparison even without a consistency proof | R3-9 |
| Arm C compensates numerical conduction as well as dispersion, and gains a 128-cell grid | upwind adds conductivity `v ρ_g C_pg Δz/2`, 1.1 to 11 times k_z at 16 cells; uncompensated, it handicaps C | CS-M1, R3-10 |
| H6, design regret | every endpoint was a prediction error; designers care about decisions | R3-4 |
| Mechanism diagnostics, degraded-data sensitivity, positive control, latent-dimension sensitivity | a verdict without a mechanism invites "why?"; a null needs proof that the pipeline can detect an effect | R3-11, R3-12, R3-43 |
| The Hodges-Lehmann estimator; (a) and (b) at the same level | the signed-rank test targets the pseudomedian; the old conditions ran at different error levels | R3-14, R3-15, CS-P5 |
| REVERSED qualifier | REFUTED hid "S significantly worse" | CS-P1 |
| Headline rows name their hypotheses; EQUIVALENT prints its margin; attribution covers the S BETTER rows | "Structure pays." could print with stability refuted, and with up to 20 % worse accuracy | R3-13, R3-16, R3-17, CS-P6 |
| A full consistency table and demotion on inconsistency | "REPLICATED" could mean two nulls; a non-replicating headline kept its wording | R3-29, R3-30 |
| Stratified replication report | the replication box straddles Al-fumarate's step | R3-31 |
| Hyperparameter table, one checkpoint rule (final checkpoint, no early stopping), equal LR sweep, P's weight fixed a priori | "matched" was claimed but not stated; early stopping and a fixed count contradicted each other; P alone was tuned | R3-21 to R3-24, CS-P4 |
| Rules for the projection-defect bound and κ; H3a fixes only before the test set, applied to every projected arm | open forking paths | R3-25, R3-26 |
| Materials and n fixed before the pilot, from fits to the published isotherms | the pilot's thresholds and timings depend on the material; Paper 1's n range (1 to 6) was far below the real steps (n ≈ 9 and 19) | R3-27, `docs/materials_3B.md` |
| Heats from the measured temperature shift of the step | the isotherm's temperature dependence controls the desorption capacity; ΔH = −Q_st over-moved Al-fumarate's step (0.38 against 0.33 RH at 60 °C) | `docs/materials_3B.md` |
| Pilot redesigned: 16 validation trajectories to 60 cycles, size-32 models, replication and system B pilots, upper-80 % SD bounds, blinded rules script, decoder gate on pilot data, before any Stage-1 submission | the old pilot could not estimate the cycle-50 variance; SDs with 3 df drove the H4 margin; a Stage-1 reviewer cannot judge placeholders | R3-18 to R3-20, R3-42, CS-P2, CS-P3 |
| OOD box moved to a drier feed (partial step) plus a hotter desorption; decoder error reported on it | the old OOD box mostly tested decoder extrapolation in temperature | R3-44 |
| Cost defined as latency and batched throughput; a Pareto statement against C and C+L | the arm-C comparison turned on an undefined cost; "milliseconds" was not supported by the plan's own numbers | R3-7, R3-8, R3-36 |
| A hard ceiling of 700 CPU-equivalent hours, with heavy work on Kaggle (free) | the worst case exceeded the budget without a stopping rule; the laptop's free RAM (often under 1 GB) cannot host the heavy runs | R3-45, R3-46 |
| The Section 1 background quotes Paper 1's current physics-loss result (a 9 % gain at weight 10⁻⁵ on held-out materials, none on held-out times) | the draft misquoted it as "ties or hurts" (ledger P3B-25) | none |
| Long test 32 → 48, OOD 16 → 24, mandatory second isotherm, FNO arm | power and generality (2026-11-05) | none |

## Reasons for rules that may look arbitrary

- **Net release, not gross outflow.** The gross outflow includes the feed water passing
  through, which is comparable in size and would hide errors.
- **Violation thresholds at 10 × the reference's own floor.** A model that reproduced the
  coarse data exactly would otherwise be scored as violating, because the coarse wall heat
  differs from the fine one by a covariance term.
- **The RK4 quadrature on each arm's own stages.** Snapshot trapezoids would make S show
  O(h³) residuals that are artefacts of the measurement.
- **Penalty weight at 1 % of the data loss.** Paper 1 found that only small physics weights
  help, and that heavy weights fail to train. A fixed a-priori weight avoids giving P a
  search the other arms do not get.
- **U+F receives latent gradients.** S's vector field uses ∇Ê, ∇N̂ and ∇Ŝ, so an
  information-matched control must see them.
- **S-L keeps the physical water balance.** A learned energy has no physical port flux, so
  only the water balance can be projected. S-L therefore isolates the potentials, not the
  balances.
- **Two systems instead of more seeds.** The between-system difference is what a broad
  reader asks about. Seed variance is already reported.
- **System B's ambient bath fixed at 293.15 K** (implementation, plan 1 Task 5). The draft left
  the wall-loss ambient open. A store sits in a room whose temperature does not follow the
  operating point, so it is a constant, not T_cold. It equals the module's reference
  temperature, which the earlier closure test already used.
- **Cyclic steady state at 10⁻³ relative change of the per-cycle release** (implementation,
  plan 1 Task 7). The draft used the CSS cycle for the long-validation decoder check and for
  H6 without a number. 10⁻³ is ten times looser than the integrated closure of the stored
  data (below 10⁻⁵), so solver noise cannot fake or block it, and tighter than any error the
  arms are compared at.
- **Reference entropy gate on the coarse data** (plan 1 Task 7). Per snapshot interval, the
  change of the entropy of the block averages minus the port entropy flux (outlet by the
  trapezoid rule, at the outlet of that interval's own flow direction) must be positive. The
  first test trajectories give margins of about 5 × 10⁻³ (system A) and 1 × 10⁻³ (system B) of
  the interval's entropy turnover.
- **Flow-dependent axial heat dispersion, and the hybrid scheme in both references**
  (2026-11-08, from cs T3, verified here; plan 1 Task 7b). With Paper 1's constant k_z = 0.10
  the plain-upwind reference carried about 9 % excess conduction at v 0.30 m/s (A, 2048 cells),
  and system B's reference (1024 cells, constant k_f 1.5) carried up to 4 times more numerical
  than physical conduction, so it was not a converged reference. Packed-bed practice makes the
  axial conductivity grow with flow (Yagi, Kunii and Wakao 1960), as D_L already does
  (`0.5 d_p u`). With `k_z = 0.10 + 0.5 ρ_g C_pg v d_p` and `k_f = 1.5 + 0.5 ρ_f c_f |v| d_p`
  (rock 5 cm), the upwind truncation term is at most 1.3 % (A) and 1.9 % (B) of the physical
  conduction at the reference grids, and the hybrid scheme removes it exactly. For arm C the
  clip now bites only at 16 cells (it bit on 32 % of the A velocity box at 128 cells and
  everywhere in B before). Numbers: `verify/clip_shares.json`. δ = 0.5 and the 5 cm rock are
  provisional until verified in the full texts (queued). This changes the reference physics,
  so it is the author's to confirm at the freeze.
- **τ_ref measured on the gate's own statistic; exact entropy ports** (2026-11-08, final code
  review C2). The previous definition took the round-off of the algebraic entropy identity
  (about 10⁻¹⁵), but the gate compares differences of stored states, which carry the solver's
  error. The outlet entropy was also integrated by the trapezoid rule, which gave −0.017 on an
  exact solve at the fast corner of the box. Entropy in and out are now cumulative ODE states,
  and τ_ref is set from the same statistic evaluated with the fine entropy (whose production is
  non-negative by proof). On that counter-example, at 256 cells, the gate passes at 16 and 256
  coarse cells with a minimum rate of 0.238 W K⁻¹ m⁻² against τ_ref = 1.2 × 10⁻³. The rate
  identities hold to 3 × 10⁻¹⁵ (`verify/results_entropy_gate_counterexample.json`), below the
  pre-registered 10⁻¹², and are now computed at every
  stored snapshot.
- **Revisions from the cs-org1 review round 2 (T5, 2026-11-08), each checked here before it
  was adopted.**
  - F1, feasibility: the data-cost contingency now halves N_z repeatedly while Paper 1's
    grid-convergence criterion passes, names what each later step weakens, adds the
    replication long test as a last cut, and declares the study infeasible rather than run
    it underpowered. The pilot timing is CPU time on an idle Kaggle session. Task 10 of plan 1
    measures the hybrid-scheme reference at 512 to 2048 cells; T5's extrapolation (2,050
    CPU-s per cycle at 2048) came from the plain-upwind reference on a loaded laptop.
  - F2, burden of proof against the coarse solver: "useful against the coarse solver" now
    needs S shown superior to every C grid and C+L; "unnecessary" needs a C grid shown
    non-inferior; otherwise "unresolved". C is scored after a conservative minmod
    reconstruction onto the 128 cells, with injection and own-grid scores as sensitivity. Cost
    is CPU time on one Kaggle session type, with C run as parallel processes.
  - F3, S-L: its port term is now S's own projection, so S against S-L differs in the
    bracket's potentials alone.
  - F4, attribution to information: printed only with S against U+F EQUIVALENT by TOST.
  - F5, bootstrap: seeds are crossed with trajectories, so the bootstrap resamples both
    independently (Owen 2007, Ann. Appl. Stat. 1, 386-411, read on Crossref 2026-11-08). α_h
    is defined for hypotheses step-down did not reach; the H4 directional words are outside
    the Holm family.
  - F6, admissibility: rows 8 and 10 split on H3a, and `verify/check_decision_list.py` now
    enumerates the gate and H3a and checks the PREREG table labels (256 combinations).
  - F7, H6: a fixed benchmark set, a 1 % regret floor in the ratio, ties counted, and the
    expected INCONCLUSIVE at n = 8 stated in advance.
  - F8, C+L: one integrator for training and evaluation (implicit Euler at the snapshot
    interval), its numerical diffusion and the non-negative multipliers' limit stated, and
    flux learning named as future work. With the flow-dependent k_z and k_f of Task 7b the
    clip bites only at 16 cells, so at C+L's 32 cells the multipliers act on unclipped
    coefficients (T5's 91 % clipped share predates Task 7b). System B's arm C uses the hybrid
    scheme too (plan 1, `make_store(..., compensate=True)`).
