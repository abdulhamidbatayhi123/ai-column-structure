# Pre-registration (DRAFT for the author's review): what does thermodynamic structure buy a learned model of a temperature-swing adsorption column?

Status: **DRAFT, not frozen.** Written 2026-10-12. Nothing in this document has been run. It
becomes binding only when the author approves it and it is committed as
`PREREG_P3B.md` with its commit hash recorded in the freeze log. The freeze happens *after*
the pilot (Section 9) and *before* any confirmatory data are generated. Every number marked
⟨pilot⟩ is filled from the pilot, under the rules of Section 9, before the freeze.

Companion documents: `docs/formulation_3B.md` (the mathematics), `docs/lit_review_3B.md`
(the gap), `docs/compute_3B.md` (the budget), `docs/INTERFACE_3A.md` (the constitutive layer).

## 1. Question

A learned reduced model of a non-isothermal adsorption column can be given, by construction:
exact water and energy balances against its inlet, outlet and wall, and non-negative total
entropy production, with the energy and entropy taken from known thermodynamics rather than
learned (formulation, Section 5). **Does that structure buy what it is supposed to buy
(stable long multi-cycle rollouts and data efficiency), and at what cost in accuracy, when
compared with the same model without it?**

The standing warning is Paper 1's L4 result: physics added as a loss tied with or hurt a
data-only surrogate, and helped only at small weights. Admissibility is not accuracy. This
study is designed so that a negative answer is as publishable as a positive one.

## 2. The system and the data

**Reference.** The consistent u-form finite-volume column of formulation Section 4, with
water heat capacity `c_w = 33.6` J mol⁻¹ K⁻¹. It is verified by the ladder of formulation
Section 6 before any data are generated: it must reproduce Paper 1's `solver_fd.py` in the
`c_w = 0` limit, close water and energy to machine precision, show non-negative entropy
production per step, and pass Paper 1's grid-convergence criterion. Solved at N_z = 2000
(Paper 1's verified resolution) and stored on 128 cells, 200 snapshots per cycle, float32.

**Material.** Decided by the rule in INTERFACE_3A.md. If no frozen 3A potential for MOF-303
exists at the freeze, a MOF-303-like dual-site isotherm is used: q_max 24.5 mol kg⁻¹,
step at RH 0.12 at 303 K, ΔH = −52 kJ mol⁻¹, f_H = 0.17, and n set to ⟨fixed at freeze from
the figure reading⟩. Sources: research/water_mof_parameters.md in Paper 1's repository
(Hanikel et al. 2019). Bed properties are the midpoints of Paper 1's ranges (ε 0.39,
ρ_p 1050 kg m⁻³, d_p 3.25 mm), with Paper 1's fixed properties (L 0.10 m, D_in 0.05 m,
k_z 0.10, h_w 10).

**Cycle.** A co-current two-step temperature swing, which is the column analogue of a
MOF water harvester:

- adsorption: humid feed at (RH_ads, T_ads), wall at T_ads, for `t_A = α t_st`;
- desorption: the same absolute humidity heated to T_des, wall at T_des, for `t_D = β t_A`.

Here `t_st` is the stoichiometric time at the adsorption condition. The feed velocity v is
held through the cycle. Each trajectory starts from a clean bed at T_ads.

**Sampled per trajectory** (scrambled Sobol, seeds ⟨fixed at freeze⟩):

| quantity | training and in-distribution test | out-of-distribution test |
|---|---|---|
| RH_ads | 0.20 to 0.60 | 0.60 to 0.80 |
| T_ads | 293 to 308 K | same |
| T_des | 333 to 363 K | 363 to 383 K |
| v | 0.03 to 0.30 m s⁻¹ (Paper 1's range) | same |
| α | 0.5 to 1.2 | same |
| β | 0.5 to 1.0 | same |

**Sets**, generated once and never regenerated:

| set | trajectories | cycles each | use |
|---|---|---|---|
| pilot | 16 | 10 | timing, variance, gate calibration; never used in confirmatory analysis |
| train pool | 256 | 6 | fitting (subsets for H2 are nested: 16 ⊂ 32 ⊂ 64 ⊂ 128 ⊂ 256) |
| validation | 32 | 6 | early stopping, the penalty-weight choice for arm P, decoder width |
| long test | 32 | 60 | H1, H3, H4 |
| OOD test | 16 | 30 | H5 |

The test sets are touched once, after every model is frozen.

## 3. Arms

Every learned arm shares, per seed, **one autoencoder** (MLP, d → 384 decoded values = 128
cells × (log c, log x, log T)). It is trained on the arm's training snapshots and then frozen.
So the dynamics are the only difference between U, P and S. All learned arms are matched on:
number of trainable dynamics parameters (within ±5 %), optimiser, iterations, batch, rollout
window, learning-rate schedule, and the RK4 step.

| arm | dynamics | role |
|---|---|---|
| **U** | `dz/dt = f_θ(z, u(t))`, an unconstrained MLP | the same model without structure (control) |
| **P** | U plus soft penalties on the water-balance residual, the energy-balance residual and negative total entropy production of the decoded state; penalty weight chosen on validation from {10⁻³, 10⁻², 10⁻¹, 1} | Paper 1's L4 question, in its strongest configuration |
| **S** | formulation Section 5.3: metriplectic internal term plus projected port term, with RK4 and constraint projection | the structured model |
| S−M | S without the internal metriplectic term (projection only) | ablation: is it the balances? |
| S−Π | S without the port projection (metriplectic internal term plus a free port term) | ablation: is it the bracket? |
| **C** | the consistent finite-volume scheme on 16, 32 and 64 cells, no learning | the non-learned competitor: structure for free, at a cost |

Arm C exists because the discrete scheme is thermodynamically consistent at *any* grid size
(formulation, Section 0, point 4). If a 32-cell consistent solver is as fast and as accurate
as the latent model, the latent model has no reason to exist, and the paper says so.

**Seeds:** 5 per learned arm and training size. A seed fixes the autoencoder initialisation,
the dynamics initialisation and the mini-batch order. U, P, S, S−M and S−Π with the same
seed share the same autoencoder and the same data order. Comparisons are paired.

## 4. Metrics

- **Field error** `e_k`: Paper 1's per-sample normalised RMSE over c, q and T on the 128-cell
  grid, for cycle k.
- **Working capacity** `W_k`: water released during the desorption step of cycle k, from
  the outlet flux. Error `ε_W,k = |W_k^pred − W_k^ref| / W_k^ref`. This is the quantity a
  harvester designer reads.
- **Rollout failure at cycle k**: the first cycle at which the state is non-finite, or
  `ε_W,k > 0.5`, or `e_k > 0.5`. A failed rollout keeps the capped values (0.5) for every
  later cycle, so failures cannot improve a mean.
- **Violations**, per rollout step, measured identically on every arm from the decoded
  fields: water-balance residual and energy-balance residual relative to the cycle's
  throughput, above `10⁻⁶`; total entropy production below `−τ_S` with `τ_S` ⟨pilot: 10× the
  99.9th percentile of |numerical production error| of arm S on pilot data⟩.
- **Cost**: wall-clock seconds per simulated cycle on this machine, single process, 4 threads.

## 5. Hypotheses, tests and verdict words (fixed in advance)

Unit of analysis: a test trajectory. For each trajectory the per-seed log-error ratio
`log(err_S / err_U)` is averaged over the 5 seeds. Tests are one-sided Wilcoxon signed-rank
tests over the 32 long-test trajectories, unless stated. **Seed consistency rule:** a
directional verdict also requires the per-seed median ratio to point the same way in at
least 4 of 5 seeds. Holm correction across the three primary tests (H1, H2, H4) at family
α = 0.05. Effect sizes come with hierarchical bootstrap 95 % intervals (resampling
trajectories, then seeds; 10,000 draws).

**H1 (long-rollout stability, primary).** At cycle 50, S has lower working-capacity error
than U.
- Statistic: `ε_W,50` per trajectory, failures capped. Secondary: failure counts (Fisher
  exact test), survival curves over cycles 1 to 60, and `e_50`.
- **SUPPORTED**: the Holm-adjusted p < 0.05, the seed rule holds, and the median ratio is
  ≤ 0.8.
- **REFUTED**: the median ratio ≥ 1.0 with the bootstrap interval excluding 0.8.
- **INCONCLUSIVE**: anything else.

**H2 (data efficiency, primary).** Trained on 32 trajectories, S has lower field error than
U over cycles 1 to 6 of the in-distribution test.
- Statistic: mean of `e_1` to `e_6`. Secondary: the full learning curve (16 to 256), and the
  *data multiplier*: the smallest training size at which U matches S-at-32, by log-linear
  interpolation of the curve.
- Verdict words and thresholds as H1.

**H3 (admissibility).**
- **H3a (verification, not a finding).** Arm S has zero balance violations and a total
  entropy-violation count at or below the bound declared at freeze. If not, that is a
  *defect* in the implementation. It is logged in the correction ledger, fixed, and the
  affected runs are repeated. It is never reported as a scientific result.
- **H3b (secondary, the scientific part).** Within U, trajectories with larger balance drift
  have larger field error: one-sided Spearman ρ > 0 between cumulative relative water-balance
  drift at cycle 50 and `e_50`, over seeds × trajectories.
  - **SUPPORTED**: p < 0.05 and ρ ≥ 0.3. **REFUTED**: ρ ≤ 0. **INCONCLUSIVE** otherwise.
- U's and P's violation rates are reported descriptively in every case.

**H4 (the cost of structure, primary).** In distribution, cycles 1 to 6, full training data:
is S's field error equivalent to U's?
- Two one-sided tests on the trajectory-level log ratio, margin ±log(1.10).
- **EQUIVALENT**: both one-sided tests reject (Holm-adjusted). **S WORSE**: the ratio's lower
  95 % bound > 1.10. **S BETTER**: the upper bound < 1/1.10. **INCONCLUSIVE** otherwise.

**H5 (secondary).** On the OOD test (hotter desorption, wetter feed), S has lower `ε_W` than U
at cycle 20. Words as H1, 16 trajectories, no Holm.

**Secondary contrasts, reported with intervals and without verdict words.** P versus U
(the L4 replication), P versus S, S−M and S−Π versus S (which part buys what), and every
learned arm against C on the accuracy-cost plane.

### Headline rule (fixed in advance)

| condition | headline |
|---|---|
| (H1 or H2 SUPPORTED) and H4 is EQUIVALENT or S BETTER | **"Structure pays."** |
| (H1 or H2 SUPPORTED) and H4 is S WORSE | **"Structure pays, at a price in accuracy."** |
| neither H1 nor H2 SUPPORTED, and H4 EQUIVALENT | **"Structure buys admissibility only."** |
| neither H1 nor H2 SUPPORTED, and H4 S WORSE | **"Structure costs accuracy and buys only admissibility."** |
| H1 and H2 both REFUTED | the title states the refutation |
| anything else | **"Inconclusive at this power"**, with the minimum detectable effects |

A second line of the headline is fixed as well. If arm C (some grid size) is no slower than S
and no less accurate than S on `ε_W,50`, the paper states that **a coarse consistent solver
makes the learned model unnecessary for this column.**

## 6. Gates (checked before the dynamics comparison, blind to arms)

1. **Decoder gate.** The smallest d in {16, 24, 32} whose frozen autoencoder reconstructs
   validation fields with mean nRMSE ≤ 0.02 is used. If none passes, the study stops and
   reports "not testable with this decoder class". The gate uses autoencoder reconstruction
   only, so it cannot favour an arm.
2. **Competence gate.** U at full data must reach in-distribution `e` (cycles 1 to 6) ≤ 0.10.
   If it does not, the comparison is reported as between two incompetent models and no
   headline is drawn.
3. **Reference gate.** Formulation Section 6, all four checks.

## 7. Power and minimum detectable effects

With 32 paired trajectories, a one-sided Wilcoxon at α = 0.05 (Holm worst case α = 0.0167)
and power 0.8 detects a standardised mean log ratio of about 0.45 (0.54 at the Holm worst
case). This uses the 0.955 asymptotic relative efficiency of the Wilcoxon test versus the
t-test. The standard deviation of trajectory-level log ratios is ⟨pilot⟩. It is converted to
a minimum detectable ratio and printed beside every null result. A null below that power is
INCONCLUSIVE, never "no effect".

## 8. What would make this study wrong, and the defences

| threat | defence |
|---|---|
| a weak control (straw man) | U and P are matched on parameters, budget and decoder; P gets its best weight on validation; C is a strong non-learned competitor |
| structure helps only because S has extra inductive bias unrelated to thermodynamics | the S−M and S−Π ablations, and P (the same information as a loss) |
| the guarantee is about a surrogate entropy | E and S are the physical functionals of the decoded fields (formulation, Section 5.1); never learned |
| the result is specific to one material | a second material (Al-fumarate-like) is a pre-declared conditional replication: run if compute allows after the primary, 3 seeds, H1 and H4 only, reported whatever it shows |
| the decoder, not the dynamics, sets the error | shared frozen autoencoder; decoder gate; reconstruction error reported beside every dynamics error |
| post-hoc metric shopping | every metric, threshold and verdict word is above; anything else is labelled exploratory |

## 9. The pilot (before the freeze)

The pilot uses its own 16 trajectories and 2 seeds. It is never reused. It is allowed to set
exactly these values and nothing else: the RK4 step, the training iterations (the same for
every arm), the entropy tolerance τ_S, the standard deviation for the MDE, and the per-run
timings that fix the compute plan. If the timing exceeds the budget in `docs/compute_3B.md`
by more than 2×, the pre-declared reduction order applies (compute, Section 6). The pilot's
error values are not looked at by arm. Only pooled variance and timing are read, by a script
that prints only those.

## 10. Discipline carried over from Papers 1 and 2

Matched budgets; at least 3 seeds (5 here); the strongest competitor configuration; an MDE
beside every null; no number in the paper without a script that prints it; a citation gate
(every reference opened live before citing); a correction ledger (`RETRACTIONS_P3B.md`,
continuing Paper 2's numbering scheme with a P3B prefix); the frozen commit hash in every
results file.

## 11. Deviations

Any deviation after the freeze is logged with its date, reason and the affected hypotheses,
and the paper reports both the pre-registered and the deviated analysis.

## 12. AI declaration (for the eventual paper)

"During the preparation of this work the author used a generative AI assistant (Claude,
Anthropic) as a tool to assist with drafting and language editing, with writing and checking
code, and with checking bibliographic records. The author designed the study, made every
scientific decision, reviewed and edited all content, and takes full responsibility for the
content of the published article."
