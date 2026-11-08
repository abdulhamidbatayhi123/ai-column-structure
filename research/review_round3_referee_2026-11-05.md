# Referee 2 report, Stage 1 (design), Paper 3B, 2026-11-05

Documents read: README.md, PREREG_P3B_draft.md (all), docs/formulation_3B.md,
docs/lit_review_3B.md, docs/compute_3B.md, docs/terminology_3B.md, docs/INTERFACE_3A.md
(first half), RETRACTIONS_P3B.md, and the earlier reviews in research/ (adversarial F1 to
F36, round 2, both reference audits). Points already resolved there are not repeated unless
the fix is wrong or incomplete. Two quick calculations were run from a scratch folder
outside the repository (equilibrium swing over the sampled box, using Paper 1's
`p_sat_water` and the dual-site isotherm with n = 3 and r = 20; and the dispersion
compensation of arm C, using Paper 1's `D_L = 0.7 D_m + 0.5 d_p u`). Nothing in the
repository was edited except this file.

The reviewer's standard is a top broad-readership venue (Nature Communications, Nature
Computational Science) and its Registered Report track. Whether those venues take a
purely computational Registered Report, and what they require at Stage 1, must be checked
on their live author pages before submission. This report does not assert either.

## Summary recommendation

**Major revision at Stage 1. At a broad-readership journal, I expect a desk rejection as the
design is now framed.**

The thermodynamics is careful and well verified. The control logic (U, U+F, U+proj, P,
ablations, C) is better than most published structure-preserving studies. The decision
rules have been through two adversarial rounds and are mostly sound. The problems are of a
different kind:

- **Scope.** This is one 1-D synthetic column with one adsorbate, and the replication only
  changes the isotherm parameters. The gap is defined by stacking five qualifiers.
- **The claimed novelty is not tested.** No arm compares fixed physical E and S against
  learned ones.
- **The reason to learn at all is not shown.** The repository's own cost numbers put the
  latent model within a small factor of a coarse consistent solver.
- **A Stage 1 reviewer cannot judge power.** Every power-relevant number is a ⟨pilot⟩
  placeholder, and the pilot as designed cannot estimate the variance of the primary
  endpoint.

All four can be fixed before Stage 1 submission without changing the core design.

### Top 8 changes, ranked by how much they raise the chance of acceptance

1. **Reframe the contribution so it matters beyond this column, and widen the evidence to
   match** (points 1 to 4). Lead with the general question: does structure help by the
   *constraint*, the *information* it supplies, or the *projection* at rollout? Say that
   arXiv:2606.01366 already tested projection after the fact. Then do one of two things.
   Either add a second, physically distinct open system that uses the same three-way
   decomposition, or add a pre-registered downstream decision task (cycle optimisation)
   in which the surrogate's errors have consequences. Otherwise target a specialist
   journal (Digital Discovery, CMAME, JCP, Comput. Chem. Eng.) and say so in the cover
   letter.
2. **Add the arm that tests the novelty claim:** a latent metriplectic model with *learned*
   E and S (tLaSDI or GFINN style), with the same decoder and budget. Also add U+F+proj,
   which costs nothing to train. Without the learned-potential arm, "energy and entropy
   fixed from thermodynamics", the load-bearing qualifier of the gap, is never compared
   against the prior art it is meant to improve on (points 5 and 6).
3. **Make the cost case explicit and give it a fair test** (points 7 to 9). The compute
   document implies about 1 s per trajectory-cycle for projected arms. A 128-cell
   reference-class solve measured 2.5 CPU-s. Formulation 5.1 promises "milliseconds".
   Define the cost metric (batching, threads, amortisation). Pre-register a Pareto or
   equal-accuracy cost comparison. Add a structured non-latent hybrid: the coarse
   consistent FV scheme with a learned closure whose Onsager coefficients stay
   non-negative. The literature this review cites shows that this hybrid is the arm most
   likely to win.
4. **Run the pilot before Stage 1, fill every ⟨pilot⟩ value, and redesign the pilot so it
   can estimate the primary variance** (points 18 to 20). The pilot runs 10 cycles and
   estimates SDs from 4 validation trajectories. H1 is read at cycle 50. The H4 margin and
   its demotion rule rest on an SD estimate with 3 degrees of freedom.
5. **Close the remaining forking paths** (points 21 to 27):
   - unspecified architecture and optimiser hyperparameters;
   - early stopping and a fixed iteration count both stated;
   - the decoder-width choice;
   - the projection-defect bound, which has no rule;
   - H3a's open-ended "fix and repeat", which could be triggered after the test set is seen;
   - the choice of material and of the exponent n, both fixed "at the freeze" although the
     pilot needs them first.
6. **Specify the replication fully** (points 28 to 31). It has no pilot, gate, threshold or
   long-validation set of its own. Its replication words call two INCONCLUSIVE verdicts
   "REPLICATED". The H4 verdict pairs are not mapped. A "Structure pays" headline that did
   not replicate keeps its wording.
7. **De-risk the decoder gate and add a latent-dimension sensitivity** (points 42 to 44).
   Paper 1's measured n-width needs 27 to 36 linear modes for 99.9 % of the energy of a
   single breakthrough. The gate asks for nRMSE ≤ 0.02 at d ≤ 32 on full temperature-swing
   cycles, and the only outcome on failure is "stop". Pre-declare a fallback decoder class
   or a wider d grid. Report the d-sensitivity, and the decoder error at the OOD
   temperatures.
8. **Repair the statistical rules that can still mislead** (points 13 to 17):
   - rows 2 and 3 print "Structure pays." when H1 is REFUTED and only H2 is SUPPORTED;
   - conditions (a) and (b) run at different levels (one-sided 0.05 or Holm against an
     implied one-sided 0.025);
   - the Wilcoxon test's location parameter is not the median the threshold uses;
   - "EQUIVALENT" can mean up to 20 % worse, with the margin not printed in the headline;
   - attribution has no row for the cheapest confound-free control.

## Detailed points

### A. Significance and novelty

**1. The gap is a conjunction of qualifiers, not a question a broad reader recognises.**
Location: lit_review Section 5, the gap quote and its qualifier table. The gap sentence is
true because of the conjunction of five qualifiers: non-isothermal, latent, fixed E and S,
an adsorption column with ports, and a pre-declared test. An editor reads this as "nobody
has done exactly this", which is the classic desk-reject pattern ("incremental,
domain-specific benchmark"). The general question in Section 1 of the pre-registration,
structure against information against projection, is the interesting one, and the design
can answer it unusually cleanly through U+F and U+proj. Fix:
- rewrite the abstract-level gap as that general question;
- present the column as a hard, realistic test bed (open, advection-dominated, stiff,
  non-isothermal; the weak spot named in lit review Section 1);
- move the five-qualifier sentence to the related-work section as the precise novelty
  statement.

**2. Prior work on part of the decomposition is not credited in the gap.** Location:
lit_review Section 4, first bullet. arXiv:2606.01366 is reported as showing that "projection
after the fact and rollout training did not rescue the unconstrained models". That is the
U+proj question, already asked on a 1-D drift-diffusion benchmark. Fix: state in the gap
paragraph that projection-versus-training has been examined (2606.01366), and that 3B's
addition is the information control (U+F) and the fixed-potential latent setting. Say
whether 2606.01366 had an information-matched control. If it did, the novelty narrows
further and the framing must say so.

**3. The replication generalises across parameters, not across systems.** Location:
Prereg Section 2, "Replication material", and Section 8, row "specific to one material".
The second material has the same PDE, column, cycle, gas model, decoder class and
dual-site isotherm family. Only q_max, the step position and f_H change. A referee at a
broad journal will say the finding is "one system, two parameter sets". Fix, in order of
preference:
- (a) add a second physically distinct open dissipative system to which the same
  formulation applies with little new theory. Candidates are a counter-current heat
  exchanger or a non-isothermal packed-bed reactor (journal B has port-Hamiltonian
  formulations of both, P13 and P14). Run it with arms U, U+F, U+proj and S only, as a
  pre-registered replication of H1 and H2;
- (b) if compute forbids (a), keep the second material and rename the claim
  "replicated on a second isotherm". Never write "a second material" without "same
  column and cycle".

**4. No decision-relevant endpoint.** Location: Prereg Section 4 (metrics) and Section 5.
Every endpoint is a prediction error. The program's stated application is MOF water
harvesting. What a broad reader cares about is whether the structured surrogate gives
better decisions, for example the optimal (α, β, T_des) for water per unit heat, or
whether the cyclic-steady-state working capacity ranks designs correctly. Fix: add one
pre-registered secondary endpoint. Optimise the cycle with each surrogate (S, U, U+F, C)
over the training box. Re-simulate each optimum with the reference. Report the regret
(reference performance at the surrogate optimum against the reference optimum), with
intervals over seeds. This costs a few dozen reference cycles.

### B. Validity of the design and missing controls

**5. The claimed novelty, fixed physical E and S, is not tested against learned E and S.**
Location: Prereg Section 3 (arms table) against lit_review Section 5 (qualifier "E and S
fixed from thermodynamics; excludes tLaSDI ... all learn them"). The design compares S
with an unconstrained U, and with S's own ablations, which keep the fixed potentials. The
nearest prior art (tLaSDI, the thermodynamics-aware ROMs, GENERIC-FNO) learns latent E and
S. The design cannot say whether fixing them from the physics buys anything over learning
them. That is the first question a referee who knows tLaSDI will ask. Fix: add arm S-L.
- Same decoder and parameter budget.
- Latent metriplectic dynamics with learned scalar potentials E_θ(z) and S_θ(z), and the
  degeneracy enforced by construction (GFINN or tLaSDI parameterisation).
- Ports handled by the same projection, but against the learned E_θ.
- Pre-register "S beats S-L" (same definition as the attribution rule) as the condition
  for the label "fixed physical potentials". Report it beside H1 and H2.

Budget: one more S-like arm, about 3.25 h per run. Dropping U-long or S−Π (compute §6
already ranks them as expendable) roughly pays for it at full data.

**6. The cheapest confound-free control is missing: U+F+proj.** Location: Prereg Section 3
and the attribution rule in Section 5. U+F controls information, and U+proj controls
rollout projection, but each controls only one. If S beats U+F, and S beats U+proj, it is
still possible that "features plus projection at rollout" equals S. Then "the structure
in training" would be a wrong label. U+F+proj needs no training (U+F's weights, S's
projection at rollout), so its only cost is evaluation. Fix: add it. Define the label
"the structure in training" as S beating U+F+proj, and keep U+proj as a secondary
contrast.

**7. The reason to learn a model is not established by the repository's own numbers.**
Location: formulation 5.1 ("A latent model of dimension d ≈ 16 to 32 can roll out in
milliseconds"); compute §0 and §2. The numbers:
- compute §2 budgets 0.75 h to evaluate 60 cycles × 48 trajectories for a projected arm.
  That is 2,880 trajectory-cycles, so about 0.94 s per trajectory-cycle;
- compute §0 measures 2.5 CPU-s per full cycle for the consistent solver at 128 cells
  (single thread, machine under load). Arm C at 16 to 64 cells will be cheaper still;
- the 2048-cell reference costs 12 to 30 s per cycle.

So the latent model is perhaps 10 to 30 times faster than the reference, and possibly no
faster than arm C. That is not milliseconds. The decision-list qualifier "a coarse
consistent solver makes the learned model unnecessary for this column" is therefore a
likely outcome, not a corner case, and it would leave the paper with no use case. Fix:
- correct formulation 5.1;
- measure, in the pilot, the cost per trajectory-cycle of S, U and C on the same footing;
- state in the pre-registration which outcome is expected, and why the study is still
  worth running if C wins (the information-against-structure result stands on its own);
- if possible, pre-register the cost comparison at equal accuracy (see point 8).

**8. The cost metric is undefined in ways that decide the arm C comparison.** Location:
Prereg Section 4, "Cost: wall-clock seconds per simulated cycle on this machine, single
process, 4 threads"; Section 5, "Arm C against the latent model". Learned arms can batch 48
trajectories in one tensor call, but a BDF solver cannot do this without extra work. Either choice changes the
ratio by an order of magnitude. Also, "no slower" is not tested, and C's non-inferiority
uses only ε_W,50, an integral quantity that a coarse conservative scheme is expected to get
right even when its fronts are smeared. Fix:
- define cost as both single-trajectory latency and batched throughput (trajectories ×
  cycles per second at batch 48), with the same thread count;
- add C's field error e_50 to the comparison, mapped to the 128 cells by a stated
  operator (point 39);
- pre-register a Pareto statement: S is "useful against C" only if no C grid is both
  non-inferior in ε_W,50 and in e_50 and no slower in throughput.

**9. A structured non-latent hybrid is the competitor a referee will demand.** Location:
Prereg Section 3, arm C and the paragraph after the table; lit_review Section 4 ("A
conservative finite-volume model with a learned flux correction reached a rollout error of
7×10⁻⁹, against 25 to 67 for unconstrained regressors") and Section 3 (FINN). The design's
own literature says that structure in the discretisation with a learned closure is the
strongest known approach for a 1-D transport problem. The design has the discretisation
without learning (C) and learning in a latent space (S), but not the hybrid. The hybrid
also inherits the formulation's proof directly: a learned non-negative correction to D_L,
k_z or the LDF coefficient keeps every Onsager block positive semi-definite (formulation
4.2). Fix: add arm C+L.
- Arm C at 32 cells.
- Learned, non-negative, state-dependent corrections to the effective dispersion,
  conduction and LDF coefficients, trained on the same coarse data with the same budget.
- Report it as a secondary contrast with intervals.

Second-order schemes are excluded "because the Bregman proof covers first-order upwind
only". That is a reason not to call a second-order scheme consistent. It is not a reason
to leave the most accurate non-learned competitor out of an accuracy-cost comparison. Add
a MUSCL variant of C as a descriptive arm, labelled "not proven consistent".

**10. Arm C at 16 cells cannot be dispersion-compensated, and thermal numerical diffusion
is not compensated at all.** Location: Prereg Section 3, arm C. With Paper 1's correlation
D_L = 0.7 D_m + 0.5 d_p u and d_p = 3.25 mm, the physical part is about 1.6×10⁻³ u. The
numerical part u Δz/2 is:
- 3.1×10⁻³ u at 16 cells (Δz = 6.25 mm), so D_L,eff = 0 and the scheme carries about
  twice the physical dispersion;
- 1.6×10⁻³ u at 32 cells, so almost all of D_L is cancelled;
- 0.8×10⁻³ u at 64 cells.

The thermal analogue at 16 cells, v ρ_g C_pg Δz/2 ≈ 0.38 W m⁻¹ K⁻¹ at v = 0.1 m s⁻¹, is
about four times k_z = 0.10. Round 2 flagged the thermal half (problem 23) and it was not
carried. Fix: state that the 16-cell grid is over-diffusive by construction. Compensate
the conductivity in the same way, `k_z,eff = max(0, k_z − v ρ_g C_pg Δz/2)`, or drop
"strongest simple configuration" from Section 3 and Section 8.

**11. No mechanism analysis is pre-specified.** Location: Prereg Section 5 (secondary
contrasts) and H3b. If S beats U, the paper will be asked *why*, and right now only H3b, an
association, answers. The formulation itself names the candidate mechanisms:
- latent drift along directions that leave Ê, N̂ and Ŝ unchanged (formulation 5.5, second
  bullet);
- inventory drift against outlet-timing error in ε_W;
- error growth per cycle.

Fix: pre-register three exploratory diagnostics, reported for every arm and labelled
exploratory:
- (i) the decomposition `W^pred − W^ref = (inventory error) + (outlet-shape error)`, per
  cycle;
- (ii) the norm of the latent displacement projected onto the null space of B = [∇Ê, ∇N̂]
  over cycles 1 to 60;
- (iii) the fitted per-cycle growth rate of e_k after the cyclic steady state.

These cost nothing extra and turn a verdict into an explanation.

**12. There is no condition where structure is predicted to fail, and no positive
control.** Location: Prereg Section 1 and Section 5. H2 (small data) is where structure
should help, and H4 (full data, short horizon) is where its cost should show. That pair is
good. Two things are missing:
- a sensitivity condition the literature predicts to matter: degraded training data
  (Prakash and Klasky, lit review Section 4), for example 1 % multiplicative noise on the
  stored fields at size 32. It should be descriptive only;
- a check that the analysis pipeline can detect an effect of the declared size. For
  example, in the pilot, compare U at full iterations with U at a quarter of the
  iterations, through the full decision rule.

Fix: add both, descriptive, so that a null H1 cannot be dismissed as a measurement failure.

### C. Statistics

**13. Rows 2 and 3 can print "Structure pays." when the long-rollout hypothesis is
refuted.** Location: Prereg Section 5, the headline list. A = "H1 SUPPORTED or H2
SUPPORTED". With H1 REFUTED, H2 SUPPORTED and H4 EQUIVALENT, row 2 prints "Structure pays."
with no qualifier, although the study's first claim, stability, has been ruled out. Round 2
asked for the hypothesis to be named in rows 2 and 3 (its problem 2, last paragraph). Only
row 4 does it. The list itself is an ordered partition: I re-checked the 4 × 4 × 4 cells
by hand. Fix: rows 2 and 3 read "Structure pays in [H1 and/or H2]", and when the other
hypothesis is REFUTED they add "; it buys no [stability | data efficiency]".

**14. Conditions (a) and (b) run at different error levels.** Location: Prereg Section 5,
"Unit of analysis". Condition (a) is a one-sided test at a Holm-adjusted level (between
0.0167 and 0.05). Condition (b) asks for a two-sided 95 % bootstrap interval to lie
entirely on one side of 1, which is a one-sided test at 0.025 and is not Holm-adjusted.
For H1 at the first Holm step, (a) is the binding condition. For H5 (α = 0.05), (b) is.
The rule is therefore not "two views of one test", and its power is not what Section 7's
orientation numbers suggest. Fix: run (b) at the same one-sided level as (a), with a
two-sided 1 − 2α_Holm interval. The power simulation (Section 7) already applies the full
rule, so this is a one-line change.

**15. The statistic in the threshold is not the one the test examines.** Location: Prereg
Section 5, verdict words. SUPPORTED needs Wilcoxon p < α and an "observed median ratio ≤
0.8". The signed-rank test concerns the pseudo-median (the Hodges-Lehmann estimator), and
the bootstrap takes the median. With the skewed log-ratio distributions that capped
failures produce, all three can disagree. Fix: use the Hodges-Lehmann estimator and its
bootstrap interval throughout. Or use the sign test, whose parameter is the median. State
exactly what "the median ratio" in (b) is: the exponentiated median over trajectories of
the seed-mean log ratios.

**16. "EQUIVALENT" can mean up to 20 % worse, and the headline does not say so.**
Location: Prereg H4 (margin rule, capped at 1.20) and headline rows 2, 5 and 8. With m
widened to 1.20, "Structure pays." is compatible with S being 20 % less accurate at short
horizon. The margin will be fixed from the pilot, so the reader of the headline does not
know it. Fix: every row that uses H4 = EQUIVALENT prints "(equivalent within ±X %)". If m
is widened beyond 1.10, the abstract states the margin.

**17. Attribution is defined only for SUPPORTED or SMALL, and not for row 9 or row 12.**
Location: Prereg Section 5, "Attribution to structure" (applies to H1 and H2 when
SUPPORTED or SMALL). Round 2 (C3) noted that the H4 = S BETTER rows also claim a benefit
and have no attribution. That is still open. Fix: apply the same attribution to H4 when S
BETTER, using U+F and U+F+proj (point 6).

### D. Power and the pilot

**18. The pilot cannot estimate the variance of the primary endpoint.** Location: Prereg
Section 9 ("12 trajectories (cycles 1 to 6) train, and 4 trajectories (cycles 1 to 10)
validate and supply the variance") against H1 ("At cycle 50"). The power simulation
(Section 7) draws "trajectory-level log ratios with the pilot's trajectory SD". No pilot
trajectory reaches cycle 50, so the SD of log(ε_W,50,S / ε_W,50,U) is never observed. It is
also estimated from 4 trajectories, at training size 12, not 256. Fix:
- run 8 to 12 pilot validation trajectories to 60 cycles. At 30 s per cycle that is under
  6 core-hours;
- estimate the H1 SD at cycle 50 and the H2 SD from models trained at size 32 on the pilot
  pool, if the pool is enlarged;
- see point 19 for how to use the estimate.

**19. Decisions that hinge on an SD with 3 degrees of freedom.** Location: Prereg H4
margin rule and Section 7. With 4 trajectories, the 95 % interval for an SD runs from
about 0.57 to 3.7 times the estimate. Whether m is widened, and whether H4 is demoted from
the Holm family, will then be decided largely by noise. Fix: use the upper 80 % confidence
bound of the pilot SD in the power simulation and in the margin rule. State this in
Section 7.

**20. ⟨pilot⟩ placeholders at Stage 1.** Location: Prereg header and Sections 2, 4, 5, 6,
7 and 9. These values are placeholders:
- n of both isotherms, and the Sobol seeds;
- every threshold: the violation thresholds, τ_S, τ_ref and the projection-defect bound;
- the RK4 step and the iteration count;
- the power curves and the detectable ratios;
- the H4 margin.

A Stage 1 reviewer is asked to judge whether the study is adequately powered and whether
its thresholds are sensible, and cannot do so here. Rule-based filling after acceptance in
principle is defensible for thresholds, but not for the sample-size justification. Fix: run
the pilot before Stage 1 submission and submit with the numbers filled in. Keep the rules
in the text so the reviewer can check that the numbers follow from them.

### E. Forking paths still open

**21. Architecture and optimiser hyperparameters are "matched" but not stated.** Location:
Prereg Section 3 ("All learned arms are matched on ... optimiser, iterations, batch,
rollout window ... learning-rate schedule"). None of these values appears in the
pre-registration. Also not fixed:
- the dynamics parameter count that the ±5 % refers to;
- the depth and width of f_θ and G_θ;
- the rank of A(z);
- κ and the rank tolerance (formulation 5.3, "a declared κ").

Only compute §2 gives "16 windows of 10 RK4 steps" and "8,000 iterations". These choices
can move S against U by more than the declared 20 %. In particular, a 10-step window
covers 1/20 to 1/80 of a cycle, and long-horizon stability is known to depend on the
window. Fix: put a hyperparameter table in the pre-registration with every value or the
rule that sets it. Give U the same tuning budget as P, or none for either (see point 22).

**22. Arm P is tuned and the control is not.** Location: Prereg Section 3, arm P ("weight
chosen on validation from {10⁻³, 10⁻², 10⁻¹, 1}"). P gets a best-of-4 selection on
validation, while U, U+F and S get no tuning. This is a mild advantage to P, and P's
comparisons are secondary. But Section 8 cites "P gets its best weight" as a defence
against a straw man, and the same logic demands that S and U each get a matched search of
the same size (for example over the learning rate). Fix: either a matched four-point
search for every arm on validation, or none, with P's weight fixed a priori from Paper 1's
L4 optimum.

**23. Early stopping and a fixed iteration count are both stated.** Location: Prereg
Section 2 sets table (validation: "early stopping"), Section 9 ("training iterations: one
value for every arm"), and compute §2 ("8,000 iterations for every arm"). If early
stopping selects a checkpoint per arm and per seed, arms get different effective budgets,
which contradicts the matching claim. Fix: state one of these. Either a fixed count with
the final checkpoint, or best-validation checkpoint selection within the fixed count,
applied identically to all arms, with the validation metric named (one-step loss or
rollout loss).

**24. "Decoder width" is selected on validation by no stated rule.** Location: Prereg
Section 2 sets table (validation: "decoder width"); Section 6.1 chooses only d. Fix: fix
the decoder architecture in the pre-registration, or give the rule (for example the
smallest width in a stated set meeting the 6.1 thresholds at the chosen d).

**25. The projection-defect bound has no rule.** Location: Prereg H3a ("no
projection-defect steps beyond ⟨pilot bound⟩") and Section 9 ("the projection-defect
bound"). The RK4 step and τ_S have rules. This bound does not. Fix: for example, "10 × the
pilot's 99.9th percentile of defect steps per 10⁴ steps for S", or a fixed 10⁻⁴ of steps.

**26. H3a's "fix and repeat" can be triggered after the test set is seen.** Location:
Prereg H3a ("If not, that is a defect ... fixed, and the affected runs are repeated").
Neither the set on which H3a is checked nor the fixes allowed are stated. If the defects
are found on the long test, changing κ, h or the integrator and re-running S is a
data-dependent modification of one arm after its test result is known. Fix:
- H3a is checked on the validation and long-validation sets before the test set is
  touched;
- the permitted fixes are listed in order (halve h; switch to the discrete-gradient
  integrator of formulation 5.4; raise κ tenfold), and each is applied to every projected
  arm (S, S−M, S−Π, U+proj, U+F+proj);
- defects that first appear on the test set are reported as deviations and not fixed.

**27. The material and its exponent are fixed "at the freeze", but the pilot needs them
first.** Location: Prereg Section 2 ("Decided by the rule in INTERFACE_3A.md. If no frozen
3A potential for MOF-303 exists at the freeze ...", and "n set to ⟨fixed at freeze from the
figure reading⟩"); Section 9 (the pilot precedes the freeze). If a 3A potential appears
between pilot and freeze, the calibrated thresholds, step, timings and SDs belong to a
different material. Fix: fix the primary material and both exponents n before the pilot,
and record the 3A decision at that date.

### F. The replication

**28. The replication has no pilot, gates or thresholds of its own.** Location: Prereg
Section 2 sets table (replication: train pool, validation, long test only) and
Section 5. The following are material-dependent but defined only for the primary
material:
- the violation thresholds (from the reference floor);
- τ_S and τ_ref;
- the RK4 step and its stiffness check;
- the cyclic-steady-state cycle, and with it the need for the long-validation decoder
  check;
- the decoder dimension d;
- the competence gate.

Section 6.1 is written for "all 5 seeds", but the replication has 3. Fix: a short
replication block that states which values are re-calibrated (thresholds, τ_S, τ_ref, the
CSS cycle, by the same rules on 4 replication pilot trajectories), which are inherited (h,
iterations, d, architecture), and that Sections 6.1 to 6.3 apply with 3 seeds. Add a
replication long-validation set (4 × 60, 240 cycles, 2 core-hours).

**29. The replication words can call two null results "REPLICATED", and leave H4 pairs
unmapped.** Location: Prereg Section 5, "Replication on the second material".
- "REPLICATED: the replication verdict equals the primary verdict" includes INCONCLUSIVE
  with INCONCLUSIVE.
- "Point the same way" is not defined for H4's verdicts. EQUIVALENT against S WORSE falls
  to "INCONCLUSIVE", although it is a substantive discrepancy.
- SMALL against REFUTED is not mapped.

Fix: write a full table over (primary verdict × replication verdict) for each hypothesis
type, with "both INCONCLUSIVE" reported as "not informative" rather than "replicated".

**30. A headline that did not replicate keeps its wording.** Location: Prereg Section 5,
paragraph after the replication words ("Its row is then qualified with the replication
outcome"). "Structure pays (not replicated on a second material)" is self-contradictory
as a headline. Fix: if a SUPPORTED hypothesis is NOT REPLICATED, the headline row is
demoted by one strength level. For example row 2 becomes "Structure pays on the primary
material only."

**31. The replication box straddles the second material's step, so the two materials are
tested in different regimes.** Location: Prereg Section 2, replication material (step at
about 0.26 RH) and the sampling table (RH_ads 0.20 to 0.60, the same box). My equilibrium
calculation (n = 3, r = 20; it ignores α and β, so it is an upper bound) gives:
- **MOF-303-like in distribution**: an equilibrium swing of 0.64 to 0.88 of q_max, with
  0.64 at the 1st percentile;
- **Al-fumarate-like**: 0.31 at the 1st percentile and 0.45 at the 10th. Trajectories
  with RH_ads below about 0.3 adsorb on or below the step.

This difference is a strength if it is analysed and a confound if it is not. A NOT
REPLICATED outcome could come from the regime rather than the material. Fix: pre-register
a stratified secondary report of R-H1 and R-H2 by RH_ads below and above 1.2 × the step
RH at T_ads. State the regime difference in Section 2.

### G. Internal consistency after the 2026-11-05 changes

**32.** README status table: "confirmatory study (about 254 CPU-hours, two materials)".
compute §3 gives 254 *wall-clock* hours ("10.6 days of continuous background running"),
including 23 h of pilot. Data generation alone is 86 core-hours on 4 workers, and training
uses 4 threads. Fix: "about 254 hours of wall time, pilot included".

**33.** Prereg Section 3, "Seeds: 5 per learned arm at full data and at training size 32,
and 3 at sizes 16, 64 and 128". This contradicts:
- the arms table: U-long and FNO have 3 seeds;
- compute §3: S−M and S−Π are not run at size 32, U+F is not run on the learning curve,
  and the learning curve has only U, P and S;
- the replication: 3 seeds.

Fix: replace the sentence with a table of arm × training size × seeds that matches compute
§3.

**34.** Prereg Section 10 ("At least 3 seeds (5 here)") and lit_review Section 7 ("Every
result is reported over 5 seeds with intervals") are both untrue for the replication, the
learning curve, U-long and FNO. Fix: "5 seeds for every primary contrast; 3 for the
replication and the descriptive arms".

**35.** README point 5 says the decision list "was checked by brute force over all 64
combinations". No script that does so exists in the repository (searched for headline,
decision-list and brute-force code). Prereg Section 10 promises "No number in the paper
without a script that prints it". Fix: add `verify/check_decision_list.py`, which
enumerates H1 × H2 × H4 (with NOT TESTABLE and demoted collapsed as Section 5 states) and
asserts one row per cell, and run it in CI.

**36.** Formulation 5.1, "can roll out in milliseconds", against compute §2 (point 7).
Fix: replace it with the measured pilot figure.

**37.** The compute evaluation budget does not state the RK4 step it assumes. Location:
compute §2 ("evaluation rollout ... about 0.75 h") against Prereg Section 9 (h chosen from
T_cycle/200, /400, /800). At T_cycle/800 every evaluation rollout costs four times as
much. The 26 h evaluation block could then become about 100 h, a 30 % overrun that is
outside the training-only uncertainty band. Fix: state the h assumed (presumably /200).
Add the evaluation block to the "2× budget" trigger.

**38.** Data-cost contingency (Prereg Section 2, items 2 and 3) does not say whether it
applies to the replication sets. Cutting the primary long test to 40 cycles but not the
replication's would break "unchanged in statistic". Fix: "applies identically to both
materials".

**39.** Arm C's field error is undefined. Location: Prereg Section 4 (e_k "on the 128
coarse cells") against arm C on 16, 32 and 64 cells. Fix: state the map. Piecewise-constant
injection of the conserved variables onto 128 cells, then conversion to (c, q, T), is
consistent with Section 5.2. Also state whether arm C is run on the replication material.

**40.** README status line "literature review ... live reference audit in progress".
Two audit rounds were completed on 2026-11-05 (research/audit_references_*.md, ledger
P3B-20 and P3B-22). Fix: "done; two live audit rounds".

**41.** The terminology file has no entries for the replication words (REPLICATED, PARTLY
REPLICATED, NOT REPLICATED), SMALL, NOT TESTABLE AT 32, or the arms added on 2026-11-05.
Fix: add them with their "may be used only when" conditions, as for the other verdict
words.

No numerical inconsistency was found in the set sizes and cycle counts themselves. Prereg
Section 2 and compute §1 agree (10,336 cycles, 86.1 core-hours, about 4.2 GB). The MDEs in
Section 7 recompute correctly (0.37, 0.44 and 0.52 SD), the compute totals add to 254.5 h,
and H3b's n = 48 matches the new long test.

### H. Feasibility and risk

**42. The decoder gate is the most likely single point of failure, and its only fallback is
"stop".** Location: Prereg 6.1. Paper 1's RESULTS.md measured, on single breakthroughs:
- c needs 32 POD modes for 99.9 % of the energy (q 27, T 36);
- that is a relative L2 residual of about 3 %;
- the n-width decays only algebraically (d_n ∼ n^−1.2).

The 3B gate asks for mean field nRMSE ≤ 0.02 at d ≤ 32, with an MLP decoder, on full
temperature-swing cycles: adsorption and desorption fronts, a thermal wave, and three
fields together. A nonlinear decoder can beat POD, but nothing in the repository shows
that this one will. If it fails, the study ends with "not testable with this decoder
class", which costs the 22 h of data generation and returns little. Fix:
- train the d ∈ {16, 24, 32} autoencoders on pilot data now (about 0.5 h each) and report
  the result in the Stage 1 submission;
- pre-declare a fallback: the d grid extends to {48, 64}, with the compute consequences
  stated, and then a convolutional decoder;
- state in advance how a larger d is reported.

**43. No sensitivity to the latent dimension.** Location: Prereg Section 3 ("The dimension
d is fixed once"). The effect of a projection onto two constraints plus one inequality
should shrink as d grows, because the constraints remove a smaller fraction of the
directions. So the S−U contrast is plausibly d-dependent. A reader will ask whether the
verdict holds at the next d. Fix: a descriptive secondary contrast of H1 and H4 at the next
passing d (3 seeds; U, U+F and S only).

**44. The OOD test mostly measures decoder extrapolation in temperature.** Location: Prereg
Section 2 (OOD: RH 0.60 to 0.80, T_des 363 to 383 K) and H5. On the MOF-303-like isotherm,
the OOD equilibrium swing is 0.85 to 0.86 of q_max (my calculation). That is inside the
upper part of the in-distribution range (up to about 0.88), because the step at RH 0.12
means a wetter feed changes little. The new content of the OOD set is bed temperatures up
to 383 K, which the decoder never saw. H5 therefore tests mainly whether a shared decoder
extrapolates in log T, and that cannot favour either arm. Fix: report the decoder
reconstruction error on the OOD set beside H5. Pre-declare that H5 is "not interpretable
as a dynamics result" if that error exceeds the 6.1 thresholds. Or move the OOD box so
that it changes the loading regime (a lower RH_ads, below the training box).

**45. The memory-mapped training reads are the realistic bottleneck on this machine.**
Location: compute §0, caution 1, and §1 ("Training reads it through a memory map"). The
data are about 4.2 GB, against about 1.5 GB of free RAM. Random window sampling over a
memory-mapped file larger than free memory will page constantly. Caution 1 records a run
that "crawled" for exactly this reason. Fix: pre-compute per-run training windows into a
compact array that fits in RAM. The full-data train pool of one material at float32 is
about 0.63 GB (1,536 cycles × 0.41 MB). Load it once per run, and state that the full set
is never memory-mapped during training.

**46. The 254 h budget has no stated overall ceiling.** Location: compute §3 and §6.
Training uncertainty is a factor of 0.5 to 2, evaluation may be four times as much (point
37), and H3a repeats are unbounded (point 26). The worst plausible case is well over 450
h, about three weeks of exclusive use of a shared laptop. The reduction order is good but
is triggered only by the pilot measurement. Fix: add a hard ceiling (for example 400 h)
after which the reduction order is applied in sequence. That is acceptable, because the
items it removes are all secondary.

### I. Presentation for a Stage 1 reviewer

**47.** The pre-registration is 524 lines of dense prose, and its design is hard to see.
Fix: add three items at the top.
- (i) A design figure: sets (rows) × arms (columns) × training sizes, with seeds in each
  cell and the hypothesis that uses it.
- (ii) A hypothesis table: H, the question in one line, the set and cycles, the statistic,
  the test, α (Holm or not), the verdict words, and the power from the pilot.
- (iii) A one-paragraph summary of the expected outcomes and why each is publishable.

**48.** The text cites internal review codes (F20, F25, F27, F33, F34 and others) and
Paper 1 rung names (L4, L5) that an external reviewer cannot resolve. Fix: in the
submitted Stage 1 version, replace each with a one-clause reason. Move the change history
(Section 13 and the review codes) to a supplementary rationale file.

**49.** The pre-registration assumes the reader knows Paper 1. Fix: add a half-page
background with Paper 1's L4 result in numbers (the result that physics as a loss tied with
or hurt the data-only surrogate). That result motivates the study, and a Stage 1 reviewer
must be able to judge the prior from it.

**50.** Long justifications sit inside rules. For example, the REFUTED definition carries
a parenthetical about coin flips, and the working-capacity definition carries the argument
against gross outflow. Fix: keep each rule to its operational statement, and move the
reasons to the rationale file.
