# Pre-registration (DRAFT for the author's review): what does thermodynamic structure buy a learned model of a temperature-swing adsorption column?

Status: **DRAFT, not frozen.** First written 2026-10-12. Revised 2026-11-03 after an
independent adversarial review (`research/review_adversarial_2026-11-03.md`, findings F1 to
F36) and after design-verification runs on small grids (`verify/`). Nothing in this document
has been run as a study. It becomes binding only when the author approves it, it is committed
as `PREREG_P3B.md`, and the commit hash is recorded in the freeze log. The freeze happens
*after* the pilot (Section 9) and *before* any confirmatory data are generated. Every
⟨pilot⟩ value is filled from the pilot, under the rules of Section 9, before the freeze.

Companion documents: `docs/formulation_3B.md` (the mathematics), `docs/lit_review_3B.md`
(the gap), `docs/compute_3B.md` (the budget), `docs/INTERFACE_3A.md` (the constitutive layer).

## 1. Question

A learned reduced model of a non-isothermal adsorption column can be given three properties
by construction: exact water and energy balances against its inlet, outlet and wall; and
non-negative total entropy production, with the energy and entropy taken from known
thermodynamics rather than learned (formulation, Section 5). **Does that structure buy what
it is supposed to buy, stable long multi-cycle rollouts and data efficiency, and at what cost
in accuracy, when compared with the same model without it?** A second question rides on the
first: **if it helps, is it the structure, or only the extra physical information the
structured model sees?**

The standing warning is Paper 1's L4 result: physics added as a loss tied with or hurt a
data-only surrogate, and helped only at small weights. Admissibility is not accuracy. This
study is designed so that a negative answer is as publishable as a positive one.

## 2. The system and the data

**Reference.** The consistent u-form finite-volume column of formulation Section 4:
- water heat capacity `c_w = 33.6` J mol⁻¹ K⁻¹;
- dilute-solute gas model (formulation, Section 3.1). This is the working decision. The
  author may switch to the ideal-gas vapour model before the freeze (INTERFACE_3A.md, last
  section);
- first-order upwind convection;
- dispersion in the exact discrete-gradient Onsager form, which equals Paper 1's discrete
  Fick flux;
- **N_z = 2048 cells** (16 × 128; formulation, Section 5.2).

It passes the reference gate (Section 6.3) before any data are used.

**What is stored per snapshot** (formulation, Section 5.2), float32, 200 snapshots per cycle:
1. Block averages over 16 fine cells of the conserved variables (n, a, u), converted offline
   to (log c, log x, log T) on 128 coarse cells.
2. The fine outlet state (c_out, T_out).
3. The exact fine wall heat per block.
4. Cumulative inlet and outlet water and energy, for the closure gate.

**Initial state.** Every trajectory starts at equilibrium at T_ads with a trace humidity
`c_0 = 10⁻⁴ c_in`, so that log variables are finite. This loading is about 3×10⁻⁶ of the
working capacity.

**Material.** Decided by the rule in INTERFACE_3A.md. If no frozen 3A potential for MOF-303
exists at the freeze, a MOF-303-like dual-site isotherm is used:
- q_max 24.5 mol kg⁻¹;
- step at RH 0.12 at 303 K;
- ΔH = −52 kJ mol⁻¹;
- f_H = 0.17;
- Henry-to-cluster affinity ratio 1/20 (Paper 1's fixed value);
- n set to ⟨fixed at freeze from the figure reading⟩.

Sources: research/water_mof_parameters.md in Paper 1's repository (Hanikel et al. 2019).

Bed properties are the midpoints of Paper 1's ranges (ε 0.39, ρ_p 1050 kg m⁻³, d_p 3.25 mm),
with Paper 1's fixed properties (L 0.10 m, D_in 0.05 m, k_z 0.10 W m⁻¹ K⁻¹, h_w 10 W m⁻² K⁻¹).
The LDF coefficient follows from the pellet by Paper 1's Glueckauf relation. Parameters are
recorded at the design condition.

**Cycle.** A co-current two-step temperature swing, the column analogue of a MOF water
harvester:
- adsorption: humid feed at (RH_ads, T_ads), wall at T_ads, for `t_A = α t_st`;
- desorption: the same absolute humidity heated to T_des, wall at T_des, for `t_D = β t_A`.

Here `t_st` is the stoichiometric time at the adsorption condition. The feed velocity v is
held through the cycle.

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
| train pool | 256 | 6 | fitting (H2 subsets are nested: 16 ⊂ 32 ⊂ 64 ⊂ 128 ⊂ 256) |
| validation | 32 | 6 | early stopping, the penalty weight of arm P, decoder width |
| long validation | 4 | 60 | decoder gate at cyclic steady state (Section 6.1) |
| long test | 32 | 60 | H1, H3, H4, and H2 on its cycles 1 to 6 |
| OOD test | 16 | 30 | H5 |

"In-distribution test" means cycles 1 to 6 of the long-test trajectories. The test sets are
touched once, after every model is frozen.

**Data-cost contingency** (if the pilot's reference timing exceeds the budget by more than
2×, compute §1), applied in this order before the freeze:
1. N_z = 1024, if it passes Paper 1's grid-convergence criterion for this isotherm on the
   pilot;
2. long-test cycles cut from 60 to 40, with H1 read at cycle 30 instead of 50;
3. the train pool cut to 192 trajectories (the H2 sizes are unchanged).

## 3. Arms

Every learned arm shares, per seed, **one autoencoder**:
- an MLP mapping d → 128 × (log c, log x, log T) plus the two outlet channels
  (log c_out, log T_out);
- trained on the arm's training snapshots, then frozen.

So the dynamics are the only difference between learned arms with the same seed. All learned
arms are matched on:
- the number of trainable dynamics parameters, within ±5 %;
- optimiser, iterations, batch, rollout window (multi-step rollout training for every arm),
  learning-rate schedule and RK4 step.

| arm | dynamics | role |
|---|---|---|
| **U** | `dz/dt = f_θ(z, u(t))`, an unconstrained MLP | the same model without structure (control) |
| **U+F** | U with the physical features S computes as extra inputs: Ê, N̂, Ŝ, P_E, P_N, Φ_S; no constraint | separates *information* from *structure* (review F27) |
| **U+proj** | U trained as U, then rolled out with S's Gauss-Newton balance projection after every step; no retraining | separates *structure in training* from *projection at rollout* (F28) |
| **P** | U plus soft penalties on the water-balance residual, the energy-balance residual and negative total entropy production of the decoded state; weight chosen on validation from {10⁻³, 10⁻², 10⁻¹, 1} | Paper 1's L4 question, in its strongest configuration |
| **S** | formulation, Section 5.3: metriplectic internal term plus projected port term, RK4 with constraint projection | the structured model |
| S−M | S without the internal metriplectic term (projection only) | ablation: is it the balances? |
| S−Π | S without the port projection (metriplectic term plus a free port term) | ablation: is it the bracket? |
| **C** | the consistent finite-volume scheme on 16, 32 and 64 cells, no learning, with dispersion compensated for upwind numerical diffusion, `D_L,eff = max(0, D_L − u Δz/2)` with `u = v/ε` | the non-learned competitor in its strongest simple configuration (F15) |

Arm C exists because the discrete scheme is thermodynamically consistent at *any* grid size,
for any non-negative dispersion (formulation, Section 0, point 4). If a coarse consistent
solver is as fast and as accurate as the latent model, the latent model has no reason to
exist, and the paper says so. Second-order entropy-stable schemes are not included. The
Bregman proof covers first-order upwind only (formulation, Section 4.3), and an unproven
competitor would weaken the comparison's guarantee.

**Seeds:** 5 per learned arm and training size. A seed fixes the autoencoder
initialisation, the dynamics initialisation and the mini-batch order. All learned arms with
the same seed share the same autoencoder and data order, so comparisons are paired.
**The dimension d is fixed once, from the full-data autoencoders** (Section 6.1), and kept for
every training size.

## 4. Metrics

- **Field error** `e_k`: Paper 1's per-sample normalised RMSE over c, q and T on the 128 coarse
  cells, for cycle k.
- **Working capacity** `W_k`: the *net* water released during the desorption step of cycle k,
  `W_k = ∫_des v (c_out − c_in,des) dt`. It is computed from the decoded outlet channel for
  every learned arm, from the fine outlet for the reference, and identically for arm C. Gross
  outflow is not used: it includes the feed water, which is comparable in size and would
  roughly halve the apparent error. The error is `ε_W,k = |W_k^pred − W_k^ref| / W_k^ref`.
- **Rollout failure at cycle k**: the first cycle at which the state is non-finite, or
  `ε_W,k > 0.5`, or `e_k > 0.5`. A failed rollout keeps the capped value 0.5 for every later
  cycle, so failures cannot improve a mean.
- **Violations**, per rollout step, measured identically on every arm:
  - **water**: `|N̂_{k+1} − N̂_k − Q_RK4[P_N]|`, with the RK4 quadrature on the arm's own
    stages and P_N computed from its own decoded outlet (formulation, Section 5.4). It is
    normalised by the cycle's feed water, `∫ v c_in dt`.
  - **energy**: the same with P_E. It is normalised by a reference-invariant scale, the heat
    of the water cycled plus the wall heat, `|ΔH| W_k + ∫ |wall heat| dt`. Absolute energy
    throughput is not used: it depends on the temperature origin and on e_0 (F34).
  - **threshold** for both: `10 ×` the reference's own self-consistency floor. That floor is
    the 99.9th percentile of the same residual computed on the stored pilot data, with the
    ports computed the way models compute them (decoded coarse temperatures for the wall).
    Value ⟨pilot⟩. A model that reproduced the data exactly is then never scored as violating
    (F20).
  - **entropy**: total production below `−τ_S`. `τ_S` = ⟨pilot: 10 × the 99.9th percentile of
    \|numerical production error\| of arm S on pilot data⟩. Because τ_S is calibrated on S, S's
    zero count is by construction. The paper never presents the U-versus-S entropy count as a
    finding about U without saying so.
  - **projection defects** (S, S−M, S−Π): steps where the rank tolerance or κ binds
    (formulation, Section 5.3). They are counted under H3a.
- **Cost**: wall-clock seconds per simulated cycle on this machine, single process, 4 threads.

## 5. Hypotheses, tests and verdict words (fixed in advance)

**Unit of analysis and scope.** A test trajectory. For each trajectory, the per-seed
log-error ratio `log(err_S / err_U)` is averaged over the 5 seeds. Two primary conditions
apply:
- **(a)** a one-sided Wilcoxon signed-rank test over the long-test trajectories. Zero
  differences, for example both arms failed and capped, are handled by Pratt's method, and
  the number of zeros is reported;
- **(b)** the hierarchical bootstrap 95 % interval of the median ratio (resampling
  trajectories, then seeds within trajectories; 10,000 draws) lies entirely on the declared
  side of 1.

Condition (b) carries the training randomness that (a) averages out (F25). A two-level
variance decomposition (trajectory, seed) is reported. The paper states that inference is
over trajectories for these training runs, with seed variance shown beside it. Holm
correction is applied across the three primary tests (H1, H2, H4) at family α = 0.05.

**Verdict words for a directional hypothesis** (H1, H2, H5):
- **SUPPORTED**: (a) Holm-adjusted p < 0.05, (b) holds, and the observed median ratio ≤ 0.8;
- **SMALL**: (a) and (b) hold, but the median ratio is between 0.8 and 1. The effect is real
  but below the declared minimum. It is reported as such, never as "no effect" (F35);
- **REFUTED**: the median ratio ≥ 1.0 and the bootstrap interval excludes 0.8;
- **INCONCLUSIVE**: anything else, printed with the power of the decision rule (Section 7).

**H1 (long-rollout stability, primary).** At cycle 50, S has lower working-capacity error
than U.
- Statistic: `ε_W,50` per trajectory, failures capped.
- Secondary: failure counts (Fisher exact test), survival curves over cycles 1 to 60, `e_50`,
  and the cycle-50 decoder reconstruction error per trajectory beside `ε_W,50`.

**H2 (data efficiency, primary).** Trained on 32 trajectories, S has lower field error than U
over cycles 1 to 6 of the in-distribution test.
- Statistic: the mean of `e_1` to `e_6`.
- Secondary: the full learning curve (16 to 256), and the *data multiplier*: the smallest
  training size at which U matches S-at-32, by log-linear interpolation.
- If the size-32 autoencoder fails the decoder gate's threshold (Section 6.1), H2 is
  reported as **NOT TESTABLE AT 32** and the learning curve stays descriptive.

**Attribution to structure** (applies to H1 and H2 when SUPPORTED or SMALL). The same test is
run for S against U+F and for S against U+proj (one-sided, not Holm-corrected, reported
with intervals). The paper attributes a benefit to **the structure** only if S also beats
U+F. It attributes it to **the structure in training**, rather than a projection at rollout,
only if S also beats U+proj. Otherwise it names what the evidence supports: "physical
features", "projection at rollout", or both.

**H3 (admissibility).**
- **H3a (verification, not a finding).** S has zero balance violations above threshold, an
  entropy-violation count at or below the declared bound, and no projection-defect steps
  beyond ⟨pilot bound⟩. If not, that is a *defect* in the implementation. It is logged in the
  correction ledger, fixed, and the affected runs are repeated. It is never reported as a
  scientific result.
- **H3b (secondary, associational).** Within U, trajectories with larger cumulative
  water-balance drift at cycle 50 have larger `e_50`. Both are averaged over seeds per
  trajectory (n = 32, no pseudo-replication), and the test is a one-sided Spearman
  correlation with a permutation p-value over trajectories.
  - **SUPPORTED**: p < 0.05 and ρ ≥ 0.3;
  - **REFUTED**: ρ ≤ 0;
  - **INCONCLUSIVE**: otherwise.

  Trajectory difficulty can drive both, so the result is described as an association, not a
  cause.
- The violation rates of U, U+F, U+proj and P are reported descriptively in every case.

**H4 (the cost of structure, primary).** In distribution, cycles 1 to 6, full training data:
is S's field error equivalent to U's?
- Test: two one-sided paired t-tests on the trajectory-level log ratios (TOST), margin
  ±log(m).
- m = 1.10 by default. Its power is computed from the pilot SD (Section 7). If the power at
  the Holm worst-case α is below 0.8, m is widened before the freeze to the smallest value
  that reaches 0.8, capped at 1.20. If even 1.20 does not reach 0.8, H4 is demoted to a
  secondary contrast and removed from the Holm family (F33).
- Intervals are the (1 − 2α_Holm) confidence intervals that TOST implies.
- Verdicts:
  - **EQUIVALENT**: both one-sided tests reject at the Holm-adjusted level;
  - **S WORSE**: the lower bound of that interval > log(m);
  - **S BETTER**: the upper bound < −log(m);
  - **INCONCLUSIVE**: otherwise.

**H5 (secondary).** On the OOD test (hotter desorption, wetter feed), S has lower `ε_W` than U
at cycle 20. Verdict words as for H1, with 16 trajectories and no Holm correction. Its MDE
(about 0.64 SD at α = 0.05, before the decision-rule simulation) is printed beside it.

**Secondary contrasts, reported with intervals and without verdict words:** P versus U (the
L4 replication), P versus S, S−M and S−Π versus S (which part buys what), U+F and U+proj
versus U, and every learned arm against C on the accuracy-cost plane.

**Arm C against the latent model.** A paired one-sided non-inferiority test of C's
`ε_W,50` against S's, margin 1.10 (ratio), for each of the three grids, Holm-corrected over
the three. The cost comparison uses the measured wall-clock per cycle.

### Headline: an ordered decision list (fixed in advance; the first matching row wins)

Let A = (H1 SUPPORTED or H2 SUPPORTED), and A_s = (H1 or H2 is SMALL, and neither is
SUPPORTED).

| order | condition | headline |
|---|---|---|
| 1 | a reference, decoder or competence gate failed (Section 6) | the gate outcome, as written in Section 6; no other headline |
| 2 | A and H4 ∈ {EQUIVALENT, S BETTER} | **"Structure pays."** |
| 3 | A and H4 = S WORSE | **"Structure pays, at a price in accuracy."** |
| 4 | A and H4 = INCONCLUSIVE (or H4 demoted) | **"Structure pays in [H1 and/or H2]; its accuracy cost is unresolved at this power."** |
| 5 | A_s | **"Structure helps, but less than the declared minimum."** |
| 6 | H1 REFUTED and H2 REFUTED and H4 ∈ {EQUIVALENT, S BETTER} | **"Structure buys admissibility only."** |
| 7 | H1 REFUTED and H2 REFUTED and H4 = S WORSE | **"Structure costs accuracy and buys only admissibility."** |
| 8 | not A, and H4 = S BETTER | **"Structure improves short-horizon accuracy but not stability or data efficiency at this power."** |
| 9 | anything else | **"No measurable benefit at this power"**, with the power of each decision rule |

Two qualifiers are appended to rows 2 to 5:
- the attribution (structure, structure in training, physical features, projection at
  rollout), decided by the attribution rule above;
- the arm C line. If C is non-inferior to S on `ε_W,50` at any grid and no slower, the paper
  states that **a coarse consistent solver makes the learned model unnecessary for this
  column.**

"Admissibility only" appears only when both benefits are *refuted*, never when they are
merely untested at this power (F32). Admissibility of S is by construction (H3a), so it is
never itself presented as a finding.

## 6. Gates (checked before the dynamics comparison, blind to arms)

**6.1 Decoder gate.** The smallest d in {16, 24, 32} whose frozen full-data autoencoders (all
5 seeds) meet all of the following on validation fields:
- mean field nRMSE ≤ 0.02;
- outlet-channel relative error (c_out, T_out) ≤ 0.02;
- decoded N and E relative error ≤ 0.005.

If the pilot shows the cyclic steady state arrives after cycle 6, the same thresholds must
also hold on the long-validation trajectories at cycles 40 to 60. If no d passes, the study
stops and reports **"not testable with this decoder class"**. The gate uses reconstruction
only, so it cannot favour an arm.

**6.2 Competence gate.** U at full data and S at full data must each reach in-distribution
`e` (cycles 1 to 6) ≤ 0.10.
- **both fail**: reported as "two incompetent models"; no headline is drawn;
- **only U fails**: reported as **"only the structured model is competent"**, a pre-declared
  outcome with S's numbers. H1, H2 and H4 are not tested, because the control is not a model;
- **only S fails**: reported as **"structure prevents a competent fit"**, the strongest form
  of Paper 1's warning.

**6.3 Reference gate** (formulation, Section 6). It requires:
- the Paper 1 limit at N_z = 2048;
- water and energy closure of the stored coarse data with the stored ports to 10⁻¹⁰;
- non-negative fine production at every step;
- Paper 1's grid-convergence criterion;
- the cyclic-steady-state cycle recorded per trajectory;
- `min_t (dS̄/dt − Φ_S)` on every stored trajectory (both steps of every cycle).

If any coarse production is negative beyond τ_S, it is reported, and arm S is declared
knowingly over-constrained on those trajectories.

## 7. Power

Power is computed **for the actual decision rules**, not only for the test inside them, by
simulation before the freeze (`scripts/power_sim.py`, written during the pilot) (F26). The
simulation draws trajectory-level log ratios with the pilot's trajectory SD and seed SD at a
range of true ratios. It applies the full rule (Wilcoxon, Pratt zeros, Holm, bootstrap
condition, median threshold) and prints the power curve. For each primary hypothesis the
pre-registration records the true ratio detectable with 0.8 power ⟨pilot⟩.

For orientation only, the classical MDE of the inner test is:
- 32 trajectories, one-sided Wilcoxon at α = 0.05: about 0.45 SD of the log ratio;
- the same at the Holm worst case α = 0.0167: 0.54 SD;
- H5 (16 trajectories, α = 0.05): about 0.64 SD.

These use the 0.955 asymptotic relative efficiency of the Wilcoxon test against the t-test.
The decision rules have less power than these numbers, because they also require the median
threshold and the bootstrap condition.

A null result is INCONCLUSIVE with its printed power, never "no effect".

## 8. What would make this study wrong, and the defences

| threat | defence |
|---|---|
| a weak control (straw man) | U, U+F and P are matched on parameters, budget, decoder and rollout training; P gets its best weight on validation; C has compensated dispersion |
| structure helps only because S sees more physics | U+F gets the same physical features without the constraint; the attribution rule (Section 5) |
| structure helps only through projection at rollout | U+proj; the attribution rule |
| the working-capacity metric favours S | W is net release from each arm's own decoded outlet, identical for every arm (Section 4) |
| the guarantee is about a surrogate entropy | E and S are the physical functionals of the decoded fields (formulation, Section 5.1), never learned |
| the balance metric scores the truth as violating | fine outlet stored; thresholds 10× the reference's own floor (Section 4) |
| the result is specific to one material | a second material (Al-fumarate-like) is a pre-declared conditional replication: run if compute allows after the primary, 3 seeds, H1 and H4 only, reported whatever it shows |
| the decoder, not the dynamics, sets the error | shared frozen autoencoder; decoder gate including outlet, N and E; reconstruction error reported beside every dynamics error, including at cycle 50 |
| post-hoc metric shopping | every metric, threshold and verdict word is above; anything else is labelled exploratory |
| pilot choices as forking paths | the pilot's choices are rule-based (Section 9) |

## 9. The pilot (before the freeze)

The pilot uses its own 16 trajectories and 2 seeds and is never reused. Only pooled variance
and timing are read, by a script that prints only those. Error values are never looked at by
arm.

It sets exactly these values, each by the stated rule, and nothing else:
- **the RK4 step:** the largest h in {T_cycle/200, /400, /800} at which S's discrete
  entropy-violation count is below 10⁻⁴ of steps and the near-dry stiffness check passes
  (formulation, Section 5.4);
- **training iterations:** one value for every arm, the smallest at which the pooled
  validation loss of every arm has stopped improving by more than 2 % per 1,000 iterations
  (compute, §6.3);
- **the thresholds:** the violation thresholds and τ_S (Section 4), and the projection-defect
  bound;
- **power and margins:** the trajectory and seed SDs for the power simulation and the H4
  margin rule (Section 5);
- **compute:** the per-run timings, the reference timing per cycle and the data-cost
  contingency (Section 2);
- **the steady-state cycle:** whether CSS arrives after cycle 6, which decides the
  long-validation decoder check (Section 6.1);
- **the coarse gate:** coarse-graining at 2048 → 128 cells (`min_t (dS̄/dt − Φ_S)`, the
  self-consistency floor).

If the timing exceeds the budget in `docs/compute_3B.md` by more than 2×, the pre-declared
reduction orders apply (compute §6, and the data-cost contingency in Section 2).

## 10. Discipline carried over from Papers 1 and 2

- Matched budgets.
- At least 3 seeds (5 here).
- The strongest competitor configuration.
- The power of the decision rule beside every null.
- No number in the paper without a script that prints it.
- A citation gate: every reference is opened live before it is cited.
- A correction ledger, `RETRACTIONS_P3B.md`, numbered P3B-1 onward.
- The frozen commit hash in every results file.

## 11. Deviations

Any deviation after the freeze is logged with its date, its reason and the affected
hypotheses. The paper reports both the pre-registered and the deviated analysis.

## 12. AI declaration (for the eventual paper)

"During the preparation of this work the author used a generative AI assistant (Claude,
Anthropic) as a tool to assist with drafting and language editing, with writing and checking
code, and with checking bibliographic records. The author designed the study, made every
scientific decision, reviewed and edited all content, and takes full responsibility for the
content of the published article."
