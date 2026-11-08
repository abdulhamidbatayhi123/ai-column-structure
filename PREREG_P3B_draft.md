# Pre-registration (Stage-1 draft): does thermodynamic structure help a learned reduced model of an open dissipative system through the constraint, the information, or the projection?

**Status.** Draft, approved by the author on 2026-11-05, not frozen. The author asked that the
design keep improving until the freeze. This version (2026-11-08) answers:
- a top-journal referee round (`research/review_round3_referee_2026-11-05.md`);
- an independent Claude Science review (hub `projects/ai-column-structure/REVIEW_cs.md`).

Each change is in `docs/prereg_rationale_3B.md`, which also holds every reason, so the rules
here stay operational. The freeze happens after the pilot (Section 10) and before any
confirmatory data: this file is copied to `PREREG_P3B.md`, and its commit hash and SHA-256 are
recorded. Values marked ⟨pilot⟩ are set by the rules in Section 10. If the study goes to a
Registered Report, the pilot runs before the Stage-1 submission, so a Stage-1 reviewer sees
filled numbers.

Companion documents:
- the mathematics (`docs/formulation_3B.md`);
- the materials (`docs/materials_3B.md`);
- the literature (`docs/lit_review_3B.md`);
- compute (`docs/compute_3B.md`);
- terms (`docs/terminology_3B.md`).

## 0. At a glance

**The question.** Physics can enter a learned reduced model in three ways that are usually
confounded:
- as a *constraint* on the dynamics;
- as *information*: physically meaningful features the model can use;
- as a *projection* applied at rollout.

Separately, the model's energy and entropy can be *fixed from physics* or *learned*. This
study separates these effects in learned latent models of two open, advection-dominated,
dissipative systems:
- **A**, a non-isothermal adsorption column, with a MOF-303 isotherm and a second isotherm,
  Al-fumarate;
- **B**, a two-temperature packed-bed thermocline store with flow reversal.

It measures what each contributes to long multi-cycle stability, data efficiency,
short-horizon accuracy and design decisions. The structured model has exact balances and
non-negative total entropy production, with the energy and entropy fixed from
thermodynamics.

**Hypotheses** (Section 6; HL = Hodges-Lehmann estimate of the seed-averaged log error ratio S/U):

| | question | set, cycles | statistic | family | verdict words |
|---|---|---|---|---|---|
| H1 | Does structure keep long rollouts accurate? | A-primary long test, cycle 50 | ε_W,50 | Holm (H1, H2, H4) | SUPPORTED / SMALL / REFUTED (+REVERSED) / INCONCLUSIVE |
| H2 | Does structure learn from less data? | A-primary, models trained on 32 trajectories, cycles 1-6 | mean e_1..6 | Holm | as H1 |
| H4 | What does structure cost in short-horizon accuracy? | A-primary, full data, cycles 1-6 | mean e_1..6 | Holm | EQUIVALENT (±m) / S BETTER / S WORSE / INCONCLUSIVE |
| H3a | Is the structured model admissible as built? | validation and long validation | violation counts | verification | pass / defect |
| H3b | Does balance drift go with error in U? | A-primary long test, cycle 50 | Spearman ρ | secondary | as H1 (ρ) |
| H5 | Out of distribution (drier feed, hotter desorption) | A-primary OOD, cycle 20 | ε_W,20 | secondary | as H1 |
| H6 | Does structure give better designs? | 8 design scenarios, cycle-optimisation regret | regret | secondary | as H1 |
| R-H1, R-H2, R-H4 | the same, second isotherm (Al-fumarate) | as H1, H2, H4 | as H1, H2, H4 | replication, α 0.05 | as H1, H2, H4, then consistency words (Section 7) |
| B-H1, B-H2, B-H4 | the same, system B | as H1, H2, H4 | ε_Q (heat delivered) and e | generality, α 0.05 | as above |

**The design** (seeds per cell; "-" = not run):

| arm | what it is | A-primary full | A-primary size 32 | A-primary 16/64/128 | A-replication full / 32 | B full / 32 |
|---|---|---|---|---|---|---|
| U | unconstrained latent ODE | 5 | 5 | 3 | 3 / 3 | 5 / 5 |
| U+F | U with all of S's physical inputs (values and latent gradients) | 5 | 5 | - | 3 / 3 | 5 / 5 |
| U+F+proj | U+F's weights, S's projection at rollout (no training) | 5 | 5 | - | 3 / 3 | 5 / 5 |
| U+proj | U's weights, S's projection at rollout (no training) | 5 | 5 | - | - | - |
| U-long | U trained for S's wall-clock | 3 | - | - | - | - |
| P | U with physics penalties, weight fixed a priori | 5 | 5 | 3 | 3 / 3 | - |
| S | structured (fixed E and S, metriplectic term, port projection) | 5 | 5 | 3 | 3 / 3 | 5 / 5 |
| S-L | as S, with learned scalar potentials E_θ and S_θ | 5 | 5 | - | - | 3 / 3 |
| S−M | S without the metriplectic term | 5 | - | - | - | - |
| S−Π | S without the port projection | 5 | - | - | - | - |
| FNO | autoregressive 1-D Fourier neural operator (descriptive) | 3 | 3 | - | - | - |
| C | consistent finite-volume scheme, 16/32/64/128 cells, compensated (no learning) | 1 (deterministic) | - | - | 1 | 1 |
| C+L | C at 32 cells with learned non-negative closure multipliers | 3 | 3 | - | - | - |
| MUSCL-C | second-order C, not proven consistent (descriptive) | 1 | - | - | - | - |

**Expected outcomes, and why each is publishable.**
- **If S beats U+F, U+F+proj and S-L:** structure helps through the constraint itself, and
  fixing the potentials from physics beats learning them. That is a positive, mechanistic
  result.
- **If S beats U but not U+F:** the gain is information, not constraint. That is a useful
  negative result for the structure-preserving literature.
- **If nothing beats U:** structure buys admissibility only. That is the program's standing
  warning, now tested with the strongest controls.
- **If arm C or C+L matches S at lower cost:** learned latent models are unnecessary for these
  systems, a direct message to the field.
- **If the systems disagree:** the result does not generalise, and the paper says so.

## 1. Background and the prior

Paper 1 of this program (archive doi:10.5281/zenodo.23323685; numbers from its `results/l4b_v3_verdict.json` via `paper/numbers.tex`) measured what a
physics-residual loss does for a neural surrogate of the same column. On held-out materials,
the best physics-weighted arm (weight 10⁻⁵) had error 0.0211 against 0.0231 for its data-only
twin, a 9 % gain. On held-out times it gave nothing (0.0132 against 0.0131). Two heavily
weighted arms failed to train. Physics as a loss therefore helps only when it is weighted
small. This study asks the stronger question: what physics buys when it is built into the
model rather than added to the loss, and which part of it does the buying.

## 2. Systems, materials and data

**System A, the adsorption column.** The reference is the consistent column of formulation
Sections 3 and 4:
- dilute-solute gas model (the author's decision, 2026-11-05);
- water heat capacity c_w = 33.6 J mol⁻¹ K⁻¹;
- convection by the hybrid scheme (Spalding 1972; Patankar 1980): first-order upwind plus the
  physical dispersion and conduction lowered, per face, by the upwind truncation term and
  clipped at zero (the same rule as arm C). At N_z = 2048 the clip never bites
  (`verify/clip_shares.json`: numerical conduction at most 1.3 % of the physical one), so the
  reference is central differencing with the physical coefficients, and it keeps the
  consistency proof because it is written as upwind plus non-negative diffusion;
- exact discrete-gradient Fick dispersion;
- wall and convection as ports;
- N_z = 2048.

Bed: ε 0.39, ρ_p 1050 kg m⁻³, d_p 3.25 mm, L 0.10 m, D_in 0.05 m, h_w 10 W m⁻² K⁻¹. Axial
effective conductivity `k_z = 0.10 + δ ρ_g C_pg v d_p` W m⁻¹ K⁻¹ with δ = 0.5 ⟨δ to be
verified in the full text⟩: it grows with flow as in packed-bed practice (Yagi, Kunii and Wakao
1960), and it is the heat analogue of D_L's `0.5 d_p u` term. Paper 1's constant 0.10 is its
stagnant part.
The LDF coefficient follows from the pellet by Paper 1's Glueckauf relation with the
material's own isotherm.

**Materials** (fixed now, before the pilot; `docs/materials_3B.md`, numbers from
`verify/materials/fit_isotherms.json`):
- three-term isotherms fitted to the published 30 °C isotherms (Hanikel et al. 2019, Fig. 2);
- **MOF-303** (primary): step at RH 0.129, n 9.2;
- **Al-fumarate** (replication): step at RH 0.256, n 18.6;
- concentration-based heats from the measured temperature shift of the step: −ΔH = 51.3
  (MOF-303) and 48.5 kJ mol⁻¹ (Al-fumarate).

These are fits to figure readings, labelled so wherever they are used. Paper 3A decision, as
of 2026-11-08: no frozen 3A potential exists, so these fits are primary. A 3A potential
enters later as a secondary arm, by the rule in `docs/INTERFACE_3A.md`.

**Cycle (A).** A co-current two-step temperature swing:
- adsorption at (RH_ads, T_ads), wall at T_ads, for α t_st;
- desorption with the same absolute humidity heated to T_des, wall at T_des, for β α t_st.

The bed starts at equilibrium with a trace humidity of 10⁻⁴ c_in.

**System B, the thermocline store** (formulation Section 7). Water through a packed bed of
rock, with fluid and solid temperatures, local exchange, conduction in both phases and wall
loss to an ambient bath at 293.15 K. Charging: hot water at T_hot enters at z = 0, for α t_front. Discharging: cold water at
T_cold enters at z = L, for β α t_front. The bed starts uniform at T_cold. Bed and fluid
properties are those of `verify/thermal_store.py` (ε 0.38, quartzite-like solid, L 1.0 m), with
N_z = 1024 ⟨grid-convergence check in the pilot⟩. The fluid's axial conductivity includes the
flow-dependent dispersion, `k_f = 1.5 + δ ρ_f c_f |v| d_p` with rock d_p = 5 cm
⟨d_p to be verified against h_v⟩, and the reference uses the same hybrid scheme as system A
(numerical conduction at most 1.9 % of the physical one at 1024 cells, `verify/clip_shares.json`).

**Sampling** (scrambled Sobol, seeds fixed in `scripts/design_sets.py` before the pilot):

| system | quantity | train and in-distribution | OOD (A-primary only) |
|---|---|---|---|
| A | RH_ads | 0.20 to 0.60 | 0.10 to 0.20 (partial step) |
| A | T_ads | 293 to 308 K | same |
| A | T_des | 333 to 363 K | 363 to 383 K |
| A | v | 0.03 to 0.30 m s⁻¹ | same |
| A | α, β | 0.5 to 1.2; 0.5 to 1.0 | same |
| B | T_hot, T_cold | 343 to 363 K; 293 to 303 K | - |
| B | v (superficial) | 0.5 to 3.0 mm s⁻¹ | - |
| B | α, β | 0.4 to 0.9; 0.6 to 1.0 | - |

For the replication isotherm the RH_ads box straddles the step. R-H1 and R-H2 are
therefore also reported stratified by RH_ads below and above 1.2 × the step RH at T_ads.

**Stored per snapshot** (formulation Section 5.2), float32, 200 snapshots per cycle:
- block averages of the conserved variables over 16 fine cells (A: (n, a, u); B: (u_f, u_s));
- the fine outlet state;
- the exact wall heat per block;
- cumulative port fluxes.

**Sets** (generated once, never regenerated; trajectories × cycles):

| set | A-primary | A-replication | B | use |
|---|---|---|---|---|
| pilot | 48 (32 × 6 train, 16 × 60 validate) | 8 × 60 | 8 × 60 | Section 10 only; never confirmatory |
| train pool | 256 × 6 | 256 × 6 | 256 × 6 | fitting; nested subsets 16 ⊂ 32 ⊂ 64 ⊂ 128 ⊂ 256 |
| validation | 32 × 6 | 32 × 6 | 32 × 6 | the decoder gate, H3a |
| long validation | 4 × 60 | 4 × 60 | 4 × 60 | the decoder gate at cyclic steady state, H3a |
| long test | 48 × 60 | 48 × 60 | 48 × 60 | H1, H2 (its cycles 1-6), H3b, H4 |
| OOD test | 24 × 30 | - | - | H5 |
| design scenarios | 8 operating scenarios | - | - | H6 |

**Data-cost contingency**, applied identically to every system and material if the pilot's
reference timing exceeds the budget by more than 2×, in this order:
1. N_z halved, if Paper 1's grid-convergence criterion passes;
2. long tests cut to 40 cycles, with H1 at cycle 30 and the long-validation check at cycles
   20 to 40;
3. train pools cut to 192 ("full data" then means 192).

## 3. Arms: definitions

All learned arms share, per seed, system and training size, one frozen autoencoder (Section
4). They differ only in their dynamics. Every arm is trained on multi-step rollouts under the
hyperparameters of Section 4.

- **U.** `dz/dt = f_θ(z, u(t))`.
- **U+F.** f_θ also receives Ê, N̂, Ŝ, P_E, P_N, Φ_S and the latent gradients ∇Ê, ∇N̂, ∇Ŝ
  (system B: no N terms), with no constraint.
- **U+proj and U+F+proj.** The trained U and U+F, rolled out with S's projection
  (formulation, Section 5.4) after every step. No retraining.
- **U-long.** U trained for the iterations that fit S's measured wall-clock.
- **P.** U plus soft penalties on the water-balance and energy-balance residuals and on
  negative total entropy production of the decoded state. The penalty weight is fixed a
  priori: the penalty equals 1 % of the data loss at initialisation (Paper 1 found that only
  small weights help).
- **S.** Formulation, Section 5.3: `dz/dt = P A Aᵀ P ∇Ŝ + Π[G_θ]`, with Ê, N̂ and Ŝ the
  physical functionals of the decoded fields. The port fluxes come from the decoded outlet
  channels and the known feed. RK4 with Gauss-Newton balance projection.
- **S-L.** As S, but the metriplectic term uses learned scalar potentials E_θ(z) and S_θ(z),
  with degeneracy with respect to ∇E_θ and ∇N̂. The port term is projected only on the
  physical water balance N̂, because a learned energy has no physical port flux. This is
  the tLaSDI/GFINN-style control for "fixed physical potentials".
- **S−M and S−Π.** S without the metriplectic term, and S without the port projection.
- **FNO.** An autoregressive 1-D FNO on the stored coarse fields and outlet channels,
  conditioned on the cycle inputs: 4 Fourier layers, 16 modes, width 64. Same multi-step
  rollout training and budget rule.
- **C.** The consistent finite-volume scheme on 16, 32, 64 and 128 cells, with the hybrid
  scheme of the reference (Spalding 1972; Patankar 1980; the truncation term as in Lantz
  1971):
  - `D_L,eff = max(0, D_L − u Δz/2)`;
  - per face, `k_z,eff = max(0, k_z − (ρ_g C_pg + c̄ c_w) v Δz/2)`, with c̄ the face mean;
  - system B: `k_f,eff = max(0, k_f − ρ_f c_f |v| Δz/2)`.

  With the flow-dependent k_z and k_f, the clip bites only at 16 cells, in both systems
  (`verify/clip_shares.json`). That grid stays over-diffusive by construction, which the paper
  states. From 32 cells up the compensation is complete over the whole velocity box.
  Integration tolerances (rtol 10⁻⁶, atol 10⁻¹⁰) are the same for C's accuracy and its timing,
  because the compensation assumes the semi-discrete scheme. C's fields are mapped to the 128 cells by
  piecewise-constant injection of the conserved variables. It runs on all systems and
  materials.
- **C+L.** C at 32 cells, with learned non-negative multipliers on D_L,eff, k_z,eff and the
  LDF coefficient (system B: the exchange coefficient). Each multiplier is the softplus of an
  MLP (3 local inputs → 32 → 32 → 1) of the cell state. It is trained on the same coarse data
  and iteration budget, with gradients through implicit-Euler steps. Every Onsager block stays
  positive semi-definite, so C+L is consistent by construction (formulation, Section 4.2).
- **MUSCL-C.** C with a van Leer reconstruction, 32 and 64 cells. Descriptive, and labelled
  "not proven consistent". The Bregman proof covers first-order upwind plus non-negative
  diffusion, which includes the hybrid scheme of C and the reference, but not limited
  reconstructions.

## 4. Hyperparameters (fixed now; the pilot sets only what Section 10 lists)

| item | value |
|---|---|
| autoencoder | MLP; encoder 386 → 256 → 256 → d, decoder d → 256 → 256 → 386 (128 cells × 3 channels + 2 outlet channels; system B: 128 × 2 + 1), SiLU; standardised channels; MSE loss; Adam, learning rate 10⁻³ with cosine decay to 10⁻⁵, batch 256 snapshots, 60 epochs |
| latent dimension d | the smallest in {16, 24, 32, 48, 64} that passes the decoder gate (Section 5.1) at full data, then fixed for every size and arm of that system and material. If none passes, a fixed 1-D convolutional decoder (4 layers, 32 channels) is tried on the same grid before "not testable" is declared |
| dynamics networks | U: f_θ is an MLP (d + inputs) → 128 → 128 → 128 → d, SiLU. S: G_θ the same; A(z) an MLP d → 128 → d·d (rank d). S-L: E_θ and S_θ are MLPs d → 64 → 64 → 1. Widths of U, U+F, P and FNO are adjusted so that the trainable dynamics parameters match S's within ±5 % (counts printed in every results file) |
| cycle inputs | log c_in, T_in, T_w, v, and the flow direction (system B), standardised |
| training | Adam; learning rate from {3×10⁻⁴, 10⁻³, 3×10⁻³}, chosen per arm in the pilot (Section 10); cosine decay to 10⁻⁵; batch 8 windows × 20 RK4 steps (640 dynamics evaluations per iteration); one fixed iteration count for every arm (pilot rule); **the final checkpoint is used, with no early stopping** |
| loss | MSE of the decoded standardised channels along each window, the outlet channels included, equal weights |
| integrator | RK4 with step h (pilot rule) |
| projection | pseudo-inverse rank tolerance 10⁻⁸ (relative singular value); κ = 10⁻³ × the pilot median of \|P∇Ŝ\|²; "κ binds" when \|P∇Ŝ\|² < 10 κ |
| P | penalty weight fixed a priori (Section 3) |

## 5. Metrics and gates

**Metrics.**
- **Field error e_k:** Paper 1's normalised RMSE over all stored channels on the 128 coarse
  cells, for cycle k.
- **Working capacity (A):** `W_k = ∫_des v (c_out − c_in,des) dt`, the net release, from each
  arm's own outlet. Its error is ε_W,k.
- **Heat delivered (B):** `Q_k = ∫_dis v ρ_f c_f (T_out − T_cold) dt`. Its error is ε_Q,k.
- **Cyclic steady state (reference):** the first cycle k ≥ 2 with
  `|W_k − W_(k−1)| / |W_k| < 10⁻³` (system B: Q_k), with W_k and Q_k integrated over the whole
  half-cycle from the stored reference ports.
- **Failure at cycle k:** a non-finite state, ε > 0.5 or e_k > 0.5. Values are capped at 0.5
  from then on.
- **Violations** (every arm, per step):
  - water and energy balance residuals, `|N̂_{k+1} − N̂_k − Q_RK4[P_N]|`, using each arm's
    own RK4 stages;
  - normalisation: by the cycle's feed water, and by `|ΔH| W_ref + ∫|wall heat_ref| dt`
    (system B: `Q_ref + ∫|wall loss| dt`);
  - threshold: 10 × the reference's own floor, the 99.9th percentile of the same residual on
    stored pilot data with ports computed as the models compute them;
  - entropy: total production below −τ_S;
  - projection defects: rank or κ binding.
- **Cost:** single-trajectory latency (seconds per simulated cycle, batch 1) and batched
  throughput (trajectory-cycles per second at batch 48), both with 4 threads on this machine.
- **Mechanism diagnostics** (exploratory, every arm):
  - (i) the split of `W^pred − W^ref` into inventory error and outlet-shape error, per cycle;
  - (ii) the norm of the latent displacement in the null space of B = [∇Ê, ∇N̂], cycles 1 to
    60;
  - (iii) the per-cycle growth rate of e_k after the cyclic steady state.

**Gates** (checked before any comparison, blind to arms):
- **5.1 Decoder gate.** All seeds of the full-data autoencoders must meet, on validation:
  - mean field nRMSE ≤ 0.02;
  - outlet RMSE of c_out/c_in ≤ 0.02, and of (T_out − T_low)/(T_high − T_low) ≤ 0.02;
  - |N̂ − N| ≤ 0.01 W_ref and |Ê − E| ≤ 0.01 (|ΔH| W_ref + ∫|wall heat| dt) (B: the analogues).

  The same must hold on long validation at cycles 40 to 60, if the pilot shows the cyclic
  steady state after cycle 6. Decoder error is also reported on the OOD set and at cycle 50.
- **5.2 Competence gate.** U and S must each reach in-distribution e (cycles 1 to 6) ≤ 0.10
  at full data. The pre-declared outcomes:
  - both fail: "two incompetent models";
  - only U fails: "only the structured model is competent";
  - only S fails: "structure prevents a competent fit".
- **5.3 Reference gate** (formulation, Section 6). It requires:
  - the Paper 1 limit at N_z = 2048;
  - closure: float64 rate identities ≤ 10⁻¹²; integrated closure with cumulative-flux states
    ≤ 10 × rtol; stored data ≤ 10⁻⁵;
  - non-negative fine production on cells above the entropy floor, with the share below it
    reported;
  - grid convergence;
  - the cyclic-steady-state cycle per trajectory;
  - `min_t (dS̄/dt − Φ_S) ≥ −τ_ref` on every stored trajectory.

  Coarse violations are reported, and S is declared over-constrained there.

## 6. Tests and verdict words

**Unit and estimator.** The unit is a test trajectory. Its value is the seed-averaged log
error ratio, arm over U. The estimate is the Hodges-Lehmann (HL) estimator of these values,
which is the parameter the signed-rank test targets. "Ratio" means exp(HL).
- **(a)** One-sided Wilcoxon signed-rank test at the hypothesis's level α_h (Holm-adjusted
  in the primary family). Zeros by Pratt's method, and their number reported.
- **(b)** The (1 − 2α_h) two-sided percentile bootstrap interval of HL (trajectories, then
  seeds within trajectories; 10,000 draws) lies on the declared side of 0.

Both conditions run at the same level. A trajectory × seed variance decomposition is
reported. Inference is over trajectories for these training runs, with seed variance shown
beside it.

**Directional words** (H1, H2, H5, H6, H3b on ρ; the first that applies):
1. **SUPPORTED:** (a) and (b) hold, and the ratio ≤ 0.8.
2. **SMALL:** (a) and (b) hold, and the ratio is in (0.8, 1).
3. **REFUTED:** the (1 − 2α_h) interval of the ratio lies entirely above 0.8. The qualifier
   **REVERSED** is added when the interval lies entirely above 1, meaning S is
   significantly worse.
4. **INCONCLUSIVE:** otherwise, printed with the decision rule's power.

**H4 words.** The test is TOST (two paired one-sided t-tests on the trajectory-level log
ratios) with margin ±log m, m = 1.10. If the pilot's upper-80 % SD gives power < 0.8 at the
Holm worst case, m is widened to the smallest value reaching 0.8, capped at 1.20. Beyond
that, H4 is demoted to secondary.
- **EQUIVALENT (±m):** both one-sided tests reject.
- **S WORSE:** the lower bound of the (1 − 2α) interval exceeds log m.
- **S BETTER:** the upper bound lies below −log m.
- **INCONCLUSIVE:** otherwise.

**H3a** is checked on validation and long validation before any test set is opened. A
defect allows only these fixes, in order, each applied to every projected arm (S, S−M, S−Π,
S-L, U+proj, U+F+proj):
1. halve h;
2. the discrete-gradient integrator;
3. κ × 10.

Defects that first appear on a test set are reported as deviations and never fixed.

**H3b** uses seed-averaged values per trajectory (n = 48) and a permutation p-value. It is
described as an association.

**H5:** 24 trajectories, α 0.05. It is "not interpretable as a dynamics result" if the OOD
decoder error exceeds the Section 5.1 thresholds.

**H6** (design decisions), 8 operating scenarios:
- Each scenario fixes (RH_ads, T_ads, v) and maximises the cyclic-steady-state productivity
  W_CSS/(t_A + t_D) over (α, β, T_des) on a 6 × 6 × 5 grid inside the training box.
- Each arm's optimum is re-simulated with the reference. The reference also evaluates a
  20-point Sobol design per scenario.
- Regret is `1 − (reference productivity at the arm's optimum)/(best reference productivity
  found among all evaluated designs)`.
- The unit is a scenario (seed-averaged), α 0.05, words as H1, applied to the regret ratio
  S/U.
- Arms: U, U+F, S, C (32 cells) and C+L.

**Attribution** (applies whenever S is credited: rows 2 to 7, 9 and 12 of Section 8). "S
beats X" means (a) and (b) hold for S against X, at α 0.05.
- If S beats U+F: **"the constraint"**.
- If S beats U+F+proj: **"the constraint in training"** (U+proj is reported beside it).
- If S beats S-L: **"fixed physical potentials"**.
- If S does not beat U+F and U+F beats U: **"the physical information, not the constraint"**.
- If S does not beat U+F and U+F does not beat U: **"not separable from the information at
  this power"**.

**Arm C, C+L and cost.** A Pareto statement fixed in advance: S is **"useful against the
coarse solver"** only if no C grid (and not C+L) is non-inferior to S, margin 1.10, in both
ε_W,50 and e_50 while being no slower in batched throughput. Each non-inferiority test is
one-sided at α 0.05, Holm over the grids.

**Secondary contrasts** (intervals, no words):
- P against U and S;
- S−M and S−Π against S;
- U-long against U;
- FNO against S and U;
- MUSCL-C against C;
- every arm on the accuracy-cost plane.

**Descriptive sensitivity analyses:**
- degraded data, 1 % multiplicative noise on the stored fields at size 32 (U, U+F, S);
- the next passing latent dimension (U, U+F, S; 3 seeds; H1 and H4 statistics);
- the 3A potential, when available.

## 7. Replication and generality words

For each hypothesis, the primary verdict p (A, MOF-303) is compared with the verdict o on
the second isotherm (R-) and on system B (B-):

| p \ o | SUPPORTED | SMALL | REFUTED | INCONCLUSIVE |
|---|---|---|---|---|
| SUPPORTED | CONSISTENT | PARTLY CONSISTENT | INCONSISTENT | NOT INFORMATIVE |
| SMALL | PARTLY CONSISTENT | CONSISTENT | INCONSISTENT | NOT INFORMATIVE |
| REFUTED | INCONSISTENT | INCONSISTENT | CONSISTENT | NOT INFORMATIVE |
| INCONCLUSIVE | NOT INFORMATIVE | NOT INFORMATIVE | NOT INFORMATIVE | NOT INFORMATIVE |

For H4:
- equal verdicts other than INCONCLUSIVE are CONSISTENT;
- {EQUIVALENT, S BETTER} is PARTLY CONSISTENT;
- any pair with S WORSE against EQUIVALENT or S BETTER is INCONSISTENT;
- any INCONCLUSIVE is NOT INFORMATIVE.

## 8. Headline: ordered decision list (first matching row wins; checked by `verify/check_decision_list.py` in CI)

Definitions:
- A = H1 or H2 SUPPORTED; A_s = not A, and H1 or H2 SMALL; R = H1 and H2 both REFUTED;
- [H] names the hypotheses that carry the row;
- "+ no [stability | data efficiency]" is appended when the other one is REFUTED, and
  "(REVERSED)" when it is REVERSED;
- H4 = "unresolved" means INCONCLUSIVE or demoted;
- an EQUIVALENT H4 prints "(accuracy equivalent within ±(m−1)·100 %)".

| row | condition | headline |
|---|---|---|
| 1 | a gate failed (Section 5) | the gate outcome only |
| 2 | A, H4 ∈ {EQUIVALENT, S BETTER} | "Structure pays in [H]." |
| 3 | A, H4 = S WORSE | "Structure pays in [H], at a price in short-horizon accuracy." |
| 4 | A, H4 unresolved | "Structure pays in [H]; its accuracy cost is unresolved at this power." |
| 5 | A_s, H4 ∈ {EQUIVALENT, S BETTER} | "Structure helps in [H], but less than the declared minimum." |
| 6 | A_s, H4 = S WORSE | "Structure helps in [H] less than the declared minimum, at a price in accuracy." |
| 7 | A_s, H4 unresolved | "Structure helps in [H] less than the declared minimum; its accuracy cost is unresolved." |
| 8 | R, H4 = EQUIVALENT | "Structure buys admissibility only." |
| 9 | R, H4 = S BETTER | "Structure improves short-horizon accuracy but buys neither stability nor data efficiency." |
| 10 | R, H4 = S WORSE | "Structure costs accuracy and buys only admissibility." |
| 11 | R, H4 unresolved | "Structure buys neither stability nor data efficiency; its accuracy cost is unresolved." |
| 12 | none of A, A_s, R; H4 = S BETTER | "Structure improves short-horizon accuracy; stability and data efficiency are unresolved at this power." |
| 13 | none of A, A_s, R; H4 = S WORSE | "Structure costs accuracy; its benefits are unresolved at this power." |
| 14 | otherwise | "No measurable benefit at this power", with each rule's power |

**Qualifiers appended to rows 2 to 13, in order:**
1. the attribution labels;
2. the consistency words for each [H] on the second isotherm and on system B. A SUPPORTED
   hypothesis that is INCONSISTENT on either is demoted one strength level: "pays" becomes
   "helps in one setting", and the headline names where it holds;
3. the Pareto statement, or "a coarse consistent solver makes the learned model unnecessary
   for this system".

## 9. Power

Power is simulated for the full decision rules, not just the inner test, by
`scripts/power_sim.py` (committed and hashed before it is run). It uses the **upper 80 %
confidence bounds** of the pilot's trajectory and seed SDs for H1 (at cycle 50) and H2 (at
size 32). For orientation only, the classical minimum detectable effects of the inner test
are:
- 48 trajectories: 0.37 SD (α 0.05) and 0.44 SD (Holm worst case);
- 24 trajectories (H5): 0.52 SD.

The decision rules have less power than these numbers. A null result is INCONCLUSIVE with
its power, never "no effect". A positive control is run in the pilot: U at full iterations
against U at a quarter of the iterations, through the full rule.

## 10. Pilot (before the freeze; before a Stage-1 submission)

The pilot uses its own trajectories (Section 2) and is never reused. Every choice is made by
a committed, hashed script (`scripts/pilot_rules.py`) that prints only the chosen values, so
no error value is seen by arm (blinding). It sets exactly the following, and nothing else:
- **learning rate per arm:** the best of {3×10⁻⁴, 10⁻³, 3×10⁻³} on 2,000-iteration runs (one
  seed), by validation rollout loss over one cycle;
- **iterations:** one value for every arm, the smallest at which every arm's pooled
  validation loss improves by less than 2 % per 1,000 iterations;
- **h:** the largest in {T_cycle/200, /400, /800} at which S's entropy-violation rate is below
  10⁻⁴ of steps and the near-dry stiffness check passes;
- **thresholds:** the violation thresholds, τ_S, τ_ref, κ, and the projection-defect bound
  (10 × the pilot's 99.9th percentile of defect steps per 10⁴ steps, capped at 10⁻³ of steps);
  each is set separately per system and material;
- **SDs:** the upper-80 % trajectory and seed SDs for the power simulation and the H4 margin;
- **timing:** the reference timing at 512, 1024 and 2048 cells, grid convergence at
  N_z/2, the data-cost contingency, and the cost of every arm;
- **CSS:** the cyclic-steady-state cycle per system, which decides the long-validation
  decoder check;
- **the decoder gate on pilot data** (d grid), so the latent dimension's risk is known
  before the freeze;
- **the positive control.**

Pilot errors by arm are never reported or used.

## 11. Budget and ceiling

Compute is in `docs/compute_3B.md`: about 620 CPU-equivalent hours, run mainly on Kaggle
(free). A hard ceiling of 700 CPU-equivalent hours applies, counted as a running total. Past
it, the reduction order of `docs/compute_3B.md` §3 is applied in sequence. It removes only
secondary or descriptive items, never the primary arms, seeds or test sets.

## 12. Discipline, deviations and the AI declaration

The program's discipline applies:
- matched budgets;
- 5 seeds for every primary contrast, and 3 for the replication, the learning curve and the
  descriptive arms;
- the strongest competitor configurations;
- the power of the decision rule beside every null;
- no number without a script;
- a live citation gate;
- the correction ledger `RETRACTIONS_P3B.md`;
- the frozen commit hash in every results file.

Deviations after the freeze are logged with their date, reason and affected hypotheses, and
both analyses are reported.

AI declaration for the paper: "During the preparation of this work the author used a
generative AI assistant (Claude, Anthropic) as a tool to assist with drafting and language
editing, with writing and checking code, and with checking bibliographic records. The author
designed the study, made every scientific decision, reviewed and edited all content, and
takes full responsibility for the content of the published article."
