# Second-round referee report on the Paper 3B design (2026-11-03)

Documents read: `docs/formulation_3B.md`, `PREREG_P3B_draft.md`, `docs/compute_3B.md`,
`docs/INTERFACE_3A.md`, `docs/lit_review_3B.md`, `README.md`, `RETRACTIONS_P3B.md`,
`verify/consistent_column.py`, `verify/run_checks.py`, `verify/results_run_checks.json`,
`verify/run_checks.log`, `verify/timing_scaling.json`, `tests/test_formulation.py`, and the
first-round report. Nothing in the repository was edited except this file. Checks run from a
scratch folder outside the repository (no byte-code or pytest cache written into the repo;
`git status` clean afterwards):

- `pytest tests`: 5 passed in 15.6 s.
- A re-run of V4 (400 to 40 cells, one cycle) with the wall port computed from coarse
  temperatures and the fine outlet: energy-rate residual 2.49×10⁻⁵, largest
  |T(x̄) − mean T| = 0.0159 K, smallest coarse production / scale = +0.0118 (positive).
- Numerical checks of RT/|ΔH|, the pre-wet loading, the TOST power and the null behaviour of
  the REFUTED rule (numbers quoted where used).

## Summary

**F1 to F36: 28 ADDRESSED, 8 PARTLY, 0 NOT.**
PARTLY: F5, F11, F14, F19, F21, F27, F31, F36.

The thermodynamics, the code and the verification numbers hold. The code implements Sections
4.2 to 4.4 as written. Every number in the formulation's Section 6 table is in the JSON. The
remaining problems are in the decision logic of the pre-registration, one gate that cannot be
passed as written, and fixes that were made in one place but not carried everywhere. No em
dashes (U+2014) occur in any document, script or test.

### Remaining problems, ranked by severity

**High**

1. **The REFUTED verdict is a coin flip under an exact null.** Prereg Section 5, "Verdict
   words": REFUTED needs "the median ratio ≥ 1.0 and the bootstrap interval excludes 0.8".
   With a true ratio of exactly 1, the observed median is ≥ 1 in 50.5 % of simulated
   32-trajectory samples, however precise the estimate. So a well-measured null is
   INCONCLUSIVE half the time ("no measurable benefit at this power", although the power is
   high), and "both REFUTED" (rows 6 and 7) fires only about a quarter of the time when the
   truth is exactly no effect. This is the null-side twin of F35's SMALL point.
   *Fix:* define the words in order of precedence: SUPPORTED, then SMALL, then
   **REFUTED = the bootstrap interval lies entirely above 0.8** (the declared minimum
   benefit is excluded), then INCONCLUSIVE. Also write SMALL as "median in (0.8, 1)".

2. **Four cells of the headline decision list give a wrong or misleading headline** (Prereg
   Section 5, "Headline: an ordered decision list"). The list is now an ordered partition
   (first match wins, row 9 catches the rest, so no gaps and no overlaps), but:
   - **Both REFUTED and H4 = S BETTER fires row 6, "Structure buys admissibility only."**
     S is then measurably *more accurate* in distribution, so "only" is false, and row 8,
     which states exactly this case, can never be reached when both are refuted.
     *Fix:* row 6 for H4 = EQUIVALENT only; add "Structure improves short-horizon accuracy
     but neither stability nor data efficiency" for both REFUTED with S BETTER.
   - **Both REFUTED and H4 INCONCLUSIVE or demoted falls to row 9, "No measurable benefit at
     this power".** The benefits were refuted, not underpowered. *Fix:* add "Structure buys
     no stability or data efficiency; its accuracy cost is unresolved at this power."
   - **Not A, not both REFUTED, and H4 = S WORSE falls to row 9.** A measured accuracy cost
     disappears from the headline. *Fix:* add "Structure costs accuracy; no measurable
     benefit at this power" before row 9.
   - **Row 5 (A_s, SMALL) ignores H4.** SMALL with S WORSE reads "Structure helps, but less
     than the declared minimum" with no mention of the cost. *Fix:* append the H4 verdict as
     a qualifier to row 5, as rows 2 to 4 do.
   Also state how H2 = NOT TESTABLE AT 32 enters the list (presumably as "not SUPPORTED, not
   SMALL, not REFUTED"), whether H1 and H4 are still tested when the competence gate's "only
   S fails" outcome occurs, and name the hypothesis in rows 2 and 3 as row 4 does.
   Full walk-through in Part C.

3. **The reference gate's 10⁻¹⁰ closure cannot be met on the stored data.** Prereg 6.3
   requires "water and energy closure of the stored coarse data with the stored ports to
   10⁻¹⁰". But Prereg Section 2 stores the snapshots in float32 (relative precision about
   6×10⁻⁸), as (log c, log x, log T), which must be mapped back nonlinearly to (n, a, u). The
   formulation's own Section 6 item 2 shows trapezoidal closure on 401 points reaches only
   1×10⁻⁵ to 8×10⁻⁵. A failed reference gate fires decision row 1 and stops the study.
   *Fix:* run the 10⁻¹⁰ closure inside the solver in float64 (cumulative port integrals
   carried as extra ODE states, as item 4 of the storage list implies), and gate the *stored*
   float32 data at a stated float32 level (for example 10⁻⁶). Say which one row 1 refers to.

**Medium**

4. **The retracted L = 0 argument survives in the summary** (F14, PARTLY). Formulation
   Section 0, point 3, still says: "The energy is linear in the state, so its gradient is
   constant, and the energy does not generate the advection." Ledger P3B-2 retracts exactly
   this sentence, and Section 4.1 now gives the degeneracy argument. *Fix:* replace it with
   the Section 4.1 wording (any field is L∇E for some antisymmetric L; the obstruction is
   L∇S = 0; 3B chooses not to split upwind).

5. **2000 cells survives in the verification ladder** (F19, PARTLY). Formulation Section 6,
   "Still to do", item 1 ("Paper 1 limit at N_z = 2000") and item 5 ("Coarse-graining at
   2000 → 128 cells") contradict Section 5.2, Prereg 2, Prereg 6.3 ("the Paper 1 limit at
   N_z = 2048") and Prereg 9 ("2048 → 128"). The docstring of `verify/run_checks.py` (line 7)
   says "2000->128-style". *Fix:* 2048 in all three places.

6. **Seed counts disagree.** Prereg Section 3: "Seeds: 5 per learned arm and training size."
   Compute Section 3: H2 sizes 16, 64 and 128 at **3** seeds, size 32 at 5. Compute Section 6
   item 2 then reduces "sizes 16 and 128" to 2 seeds and does not mention size 64. *Fix:*
   Prereg: "5 seeds at full data and at size 32; 3 seeds at sizes 16, 64 and 128"; compute
   reduction item 2: "sizes 16, 64 and 128".

7. **The attribution rule does not say what "beats" means, and it turns a non-significant
   contrast into a positive attribution** (Prereg Section 5, "Attribution to structure";
   F27 and F28). The S-versus-U+F and S-versus-U+proj tests are "one-sided, not
   Holm-corrected", and the threshold for "beats" (p < 0.05? condition (b)? the median
   threshold?) is not given. If S does not beat U+F, the paper "names what the evidence
   supports: physical features". A failed, possibly underpowered contrast is not evidence
   for the features. This repeats the F32 pattern one level down. *Fix:* define "beats" as
   one-sided Wilcoxon p < 0.05 *and* the bootstrap interval of the S/U+F ratio below 1.
   When S does not beat U+F, write "not separable from the physical features at this power",
   not "physical features".

8. **U+F still lacks the gradient features S uses** (F27, PARTLY). S's vector field uses
   ∇Ê, ∇N̂, ∇Ŝ (decoder VJPs). U+F gets only Ê, N̂, Ŝ, P_E, P_N, Φ_S (Prereg Section 3;
   compute: "no gradients"). The compute-matched U that F27 asked for as a secondary
   contrast is absent. *Fix:* either add the three latent gradients (or B^T G-type
   projections of them) to U+F's inputs, or state in Section 8 that U+F controls for scalar
   physical information only. Add the compute-matched U, or say why not.

9. **Two decoder-gate thresholds are not scale-invariant** (Prereg 6.1, F30). "Outlet-channel
   relative error (c_out, T_out) ≤ 0.02": relative to absolute temperature this allows about
   6 K at 300 K, more than a third of the 16.7 K front temperature rise in V1. "Decoded N and
   E relative error ≤ 0.005": E = Σu Δz with u = C_s(T − T_0) + (n+a)e_w + aΔH depends on
   T_0 and e_0 and can be near zero, the same defect F34 removed from the violation metric.
   *Fix:* T_out error relative to (T_des − T_ads) of the trajectory; E error relative to the
   reference-invariant energy scale of Prereg Section 4 (|ΔH| W_k + ∫|wall heat| dt).

10. **Ungated undershoot cells in the fine entropy check** (F11, PARTLY). η is fixed and the
    flux is linear (formulation 4.2, code lines 147 to 155, test `test_dispersion_is_exact_fick`),
    and the bed is pre-wetted. But F11 also asked that entropy be evaluated "only as a
    diagnostic, with a stated floor", counting cells below the floor. Prereg 6.3 requires
    "non-negative fine production at every step", which needs ln c in every cell and is
    undefined where BDF undershoots to c ≤ 0 (formulation 4.4 says the reference does not
    guarantee positivity). *Fix:* state the floor (for example c ≥ 10⁻⁶ c_0), evaluate
    production only above it, and report the count and location of cells below it.

11. **Numbers without a script that prints them** (Prereg Section 10: "No number in the paper
    without a script that prints it").
    - Formulation 5.2: "3×10⁻⁵ of the energy rate" and "within 0.016 K". Neither is in
      `results_run_checks.json` or printed by `run_checks.py`. My re-run gives 2.49×10⁻⁵ and
      0.0159 K, so the numbers are right in substance, but "3×10⁻⁵" should read 2.5×10⁻⁵.
      Ledger P3B-5 repeats "3×10⁻⁵".
    - Formulation 5.2: "Empirically, it held **with margin**." The JSON cannot show a margin,
      because `sigma_c_min_over_scale_*` is initialised at 0.0 and only lowered
      (`run_checks.py` lines 183 to 226), so it prints 0.0. My re-run finds a minimum of
      +0.0118 of the scale, so the claim is true, but nothing in the repository shows it.
    - Compute Section 0, first row: "1.7 CPU-s against 1.3 CPU-s; 1,408 against 1,311 RHS
      evaluations" is in no file.
    *Fix:* add the coarse-wall residual, the block-temperature gap, the true minimum of the
    coarse production (initialise at +inf) and the Paper 1 timing comparison to
    `run_checks.py`, re-run, and quote the printed values.

12. **Status lines contradict the revision.**
    - Formulation line 3: "Status: design draft, 2026-10-12. Nothing here has been run."
      Section 6 says V1 to V4 were run on 2026-11-03.
    - README: "36 findings, all addressed." Eight are PARTLY (this report).
    - Ledger P3B-13: fix "compute document revised with the **measured** reference timing",
      found by "the **2048-cell timing run**". Compute Section 0 says the 2048-cell cost is
      **extrapolated** from 512 cells, and that the 1024-cell run was discarded. No 2048-cell
      timing is reported anywhere. (F36, PARTLY for this reason.)
    - Formulation 5.1: the autoencoder is "trained once per seed and frozen". Prereg H2
      ("the size-32 autoencoder") and compute ("plus the subsets") train one per seed *and
      training size*.
    *Fix:* update each line to match.

**Low**

13. **Storage arithmetic omits the wall field.** Compute Section 1: "128 cells × 3 fields ×
    200 snapshots ... plus the outlet and wall channels, is about 0.31 MB per cycle". The
    per-block wall heat is a fourth 128-value field: 128 × 4 × 200 × 4 B = 0.41 MB per cycle,
    and 4528 cycles give about 1.85 GB, not 1.4 GB. Still fits the disk.

14. **The regularised inequality correction is not exact, and "κ binds" is undefined**
    (formulation 5.3). With λ = max(0, δ)/(|P∇Ŝ|² + κ), the correction raises ∇Ŝ·Π by
    δ|P∇Ŝ|²/(|P∇Ŝ|² + κ) < δ, so a deficit δκ/(|P∇Ŝ|² + κ) remains whenever κ > 0. The
    sentence "Consequences, exact in continuous time whenever the projection is well posed"
    should say so. *Fix:* define δ = Φ_S − ∇Ŝ·Π, define a "κ-binding step" as one where the
    remaining deficit exceeds τ_S, and count those under H3a. (The correction is along P∇Ŝ,
    so the two equalities are preserved; that part is right.)

15. **Energy-violation denominator is ambiguous** (Prereg Section 4): "|ΔH| W_k" does not say
    whether W_k is the arm's prediction or the reference. A model predicting W ≈ 0 would get
    a small denominator. *Fix:* "the reference W_k".

16. **Contingency side effects not carried** (Prereg Section 2, data-cost contingency). If the
    long test is cut to 40 cycles: the long-validation decoder check "at cycles 40 to 60"
    (6.1) and the survival curves "over cycles 1 to 60" (H1) need new windows. If the train
    pool is cut to 192, "full data" for H4 and the competence gate becomes 192 and the top of
    the H2 learning curve changes. State both.

17. **Gate tolerance borrowed from arm S** (Prereg 6.3; formulation 5.2 uses "τ", Prereg
    "τ_S"). τ_S is calibrated on arm S's numerical production error, which requires the
    pilot data that this gate checks. A data property should have its own tolerance (for
    example a multiple of the fine model's production at that snapshot). Also say whether a
    negative coarse production is a gate *failure* (row 1) or a reported caveat; 6.3 implies
    the latter.

18. **RT/|ΔH| lower bound uses a temperature 3B never visits** (F5, PARTLY). Formulation
    Section 0 point 1 and 3.4 row 2 say "288 to 383 K, the temperatures 3B's cycles visit".
    Prereg T_ads is 293 to 308 K. At 293 K and −60 kJ mol⁻¹ RT/|ΔH| = 4.06 %; with 3B's fixed
    ΔH = −52 kJ mol⁻¹ the range is 4.7 to 6.1 %. *Fix:* "293 to 383 K" and either "4.1 to
    8.4 % over Paper 1's ΔH range" or "4.7 to 6.1 % for the 3B material".

19. **Wrong section pointer.** Formulation 3.2, first bullet: "the LDF entropy production
    non-negative (Section 4.3)". The LDF block is in Section 4.2; 4.3 is convection.

20. **Reference-timing extrapolation mixes growth rates** (compute Section 0). "1.4 to 1.9×
    per doubling" are per-phase growth factors (adsorption 1.38 to 1.62, desorption 1.58 to
    1.88), applied to whole-cycle totals. Whole-cycle growth is 1.53 to 1.67, which gives 14
    to 23 s at 2048 cells. The 30 s budget is therefore conservative, which is fine, but the
    derivation should say so. The timed cycle is 1.4 t_st; the sampled cycles reach
    α(1 + β) = 2.4 t_st (mean about 1.5). The pilot rule covers this.

21. **"Verified to 10⁻⁹"** (README; lit review Section 5 item 1; ledger P3B-1). V1 gives
    7×10⁻¹⁰ in c, 6×10⁻⁹ in q and 1.5×10⁻⁷ K in T. "Better than 10⁻⁸ relative" is accurate.

22. **Pilot training set unspecified** (Prereg 9). The iteration rule uses "the pooled
    validation loss of every arm" in the pilot, but the pilot's 16 trajectories are its only
    data; it does not say what the arms are trained on, and an iteration count tuned on 16
    trajectories may not transfer to 256.

23. Minor: formulation 4.3 writes g = (c, ρ_gC_pg T + c e_w(T)) while 3.3 and the code use
    C(T − T_0). The constant cancels in the port difference, but use one form.
    Arm C compensates numerical dispersion of c but not the thermal numerical diffusion
    (v ρ_g C_pg Δz/2 against k_z); optional, but say so if "strongest configuration" is kept.

---

## Part A. F1 to F36

| F | status | where the fix is | remaining issue |
|---|---|---|---|
| F1 | ADDRESSED | (correct as written; no change needed) | |
| F2 | ADDRESSED | (correct) | |
| F3 | ADDRESSED | formulation 5.2, last bullet (e_0 = s_0 = 0); code `e_w`, `s_w` have no constants | |
| F4 | ADDRESSED | formulation 3.1 ("follows once μ_a is defined ...") | |
| F5 | PARTLY | formulation 0 pt 1 and pt 2 (upper bound at saturation), 3.4 row 2; INTERFACE last section; ledger P3B-10 | 288 K is not visited (problem 18) |
| F6 | ADDRESSED | (correct) | |
| F7 | ADDRESSED | superseded by the exact η (formulation 4.2) | |
| F8 | ADDRESSED | (correct) | |
| F9 | ADDRESSED | formulation 4.2 LDF paragraph; lit review Section 5 item 2; ledger P3B-11 | |
| F10 | ADDRESSED | formulation (G) and 4.2 "Wall: a port, not a block of M"; code `rhs` and `budget` | |
| F11 | PARTLY | formulation 4.2 η; code lines 147 to 155; test; pre-wet in 5.2 | no floor for entropy diagnostics; Prereg 6.3 fine-production check undefined at undershoots (problem 10) |
| F12 | ADDRESSED | formulation 0 pt 4, 4.3; Prereg Section 3 arm C | |
| F13 | ADDRESSED | (correct) | |
| F14 | PARTLY | formulation 4.1 rewritten; ledger P3B-2 | formulation 0 pt 3 keeps the retracted argument (problem 4) |
| F15 | ADDRESSED | Prereg Section 3 arm C (D_L,eff) and the paragraph after the table | thermal analogue optional (problem 23) |
| F16 | ADDRESSED | (correct) | |
| F17 | ADDRESSED | formulation 5.3 "When the projection is ill posed" | κ deficit and "κ binds" undefined (problem 14) |
| F18 | ADDRESSED | formulation 5.4 "One discrete balance definition"; Prereg Section 4 water/energy; compute S +25 % | |
| F19 | PARTLY | formulation 5.2; Prereg 2, 6.3, 9; ledger P3B-3 | formulation 6 to-do items 1 and 5, `run_checks.py` docstring (problem 5) |
| F20 | ADDRESSED | formulation 5.2 (fine outlet, floor); Prereg Section 4 threshold; ledger P3B-4, P3B-5 | floor value not printed by any script (problem 11) |
| F21 | PARTLY | items 1 to 3: formulation 5.2, Prereg 2 and 3; item 4: Prereg 6.3 | 10⁻¹⁰ gate unattainable on float32 stored data (problem 3) |
| F22 | ADDRESSED | formulation 5.2 (pre-wet), 5.4 (stiffness); Prereg 2, 9 | pre-wet loading checked: 6.9×10⁻⁵ mol kg⁻¹, swing 20.24, ratio 3.4×10⁻⁶ |
| F23 | ADDRESSED | formulation 5.2 (stored wall heat, covariance quantified) | number not in JSON (problem 11) |
| F24 | ADDRESSED | formulation 5.2 (counterexample, empirical, gate); Prereg 6.3; ledger P3B-12 | "with margin" unsupported by JSON; τ versus τ_S (problems 11, 17) |
| F25 | ADDRESSED | Prereg Section 5 "Unit of analysis" (bootstrap co-primary, variance decomposition; the 4-of-5 rule is gone) | |
| F26 | ADDRESSED | Prereg Section 7 (rule-level simulation, `scripts/power_sim.py` at the pilot) | |
| F27 | PARTLY | Prereg Section 3 arm U+F; attribution rule; Section 8 | no gradient features, no compute-matched U, "beats" undefined (problems 7, 8) |
| F28 | ADDRESSED | Prereg Section 3 U+proj; Section 4 net W; attribution rule | see problem 7 for the rule's wording |
| F29 | ADDRESSED | Prereg 2 (long validation 4 × 60; in-distribution test defined); 6.1; H1 secondary | contingency windows (problem 16) |
| F30 | ADDRESSED | Prereg 6.2 (both arms, asymmetric outcomes); 6.1 (outlet, N, E); H2 "NOT TESTABLE AT 32"; Section 3 (d fixed from full data) | gate scales (problem 9) |
| F31 | PARTLY | Prereg Section 5 ordered decision list | now a partition, but four cells are wrong or misleading (problem 2) |
| F32 | ADDRESSED | row 6 requires both REFUTED; paragraph after the list | new: row 6 also fires with S BETTER (problem 2) |
| F33 | ADDRESSED | Prereg H4 (paired t TOST, 1 − 2α_Holm intervals, margin rule, demotion) | checked: m = 1.20 reaches power 0.8 at α = 0.0167 up to an SD of about 0.29 |
| F34 | ADDRESSED | Prereg Section 4 (feed water; |ΔH| W_k + wall heat; τ_S caveat) | W_k ambiguous (problem 15) |
| F35 | ADDRESSED | Prereg H3b (n = 32, permutation, associational), Pratt zeros, SMALL, H5 MDE, arm C non-inferiority, Section 9 rules | REFUTED rule (problem 1) |
| F36 | PARTLY | compute Sections 0 to 3 (pilot 2 × 9.74 h, S +25 %, measured scaling, contingency) | ledger P3B-13 claims a measured 2048-cell timing that does not exist (problem 12) |

## Part B. Cross-document consistency

| # | document 1 | document 2 | disagreement |
|---|---|---|---|
| B1 | formulation 6, to-do 1 and 5: N_z = 2000, "2000 → 128" | formulation 5.2; Prereg 2, 6.3, 9: 2048 | grid |
| B2 | formulation 0 pt 3: "energy is linear ... does not generate the advection" | formulation 4.1; ledger P3B-2 | retracted argument still stated |
| B3 | Prereg 3: "5 per learned arm and training size" | compute 3: sizes 16, 64, 128 at 3 seeds | seed counts |
| B4 | compute 6 item 2: "sizes 16 and 128 drop to 2 seeds" | compute 3: sizes 16, 64, 128 at 3 seeds | size 64 missing |
| B5 | formulation 5.1: autoencoder "trained once per seed" | Prereg H2 ("size-32 autoencoder"); compute 3 ("plus the subsets") | per size |
| B6 | formulation line 3: "Nothing here has been run" | formulation 6: V1 to V4 run 2026-11-03 | status |
| B7 | ledger P3B-13: "measured reference timing", "2048-cell timing run" | compute 0: 2048 extrapolated from 512; 1024 run discarded | timing provenance |
| B8 | README: "36 findings, all addressed" | this report: 8 PARTLY | overclaim |
| B9 | formulation 0 pt 1, 3.4 row 2: "288 to 383 K, the temperatures 3B's cycles visit" | Prereg 2: T_ads 293 to 308 K; INTERFACE req. 6 "about 293 to 383 K" | lower temperature |
| B10 | formulation 5.2: "3×10⁻⁵"; ledger P3B-5 "3×10⁻⁵" | no JSON value; re-run gives 2.5×10⁻⁵ | number and provenance |
| B11 | formulation 5.2: reference gate violation "beyond τ" | Prereg 6.3: "beyond τ_S" | which tolerance |
| B12 | Prereg 6.3: closure "to 10⁻¹⁰" of stored data | Prereg 2: float32 storage; formulation 6 item 2: trapezoid closure 10⁻⁵ | unattainable |
| B13 | formulation 3.2: LDF production "(Section 4.3)" | formulation 4.2 holds the LDF block | wrong pointer |
| B14 | README, lit review 5.1, ledger P3B-1: "to 10⁻⁹" | formulation 6 V1: 6×10⁻⁹ in q, 1.5×10⁻⁷ K in T | rounding claim |
| B15 | compute 1: 0.31 MB per cycle "plus the outlet and wall channels", 1.4 GB | Prereg 2 item 3: wall heat per block stored (a 128-value field) | 0.41 MB, about 1.85 GB |
| B16 | Prereg 6.1: long-validation check "at cycles 40 to 60"; H1 survival "cycles 1 to 60" | Prereg 2 contingency 2: long test cut to 40 cycles | windows not adjusted |
| B17 | formulation 4.3: g_u = ρ_gC_pg T + c e_w | formulation 3.3 and code: C(T − T_0) | form only |
| B18 | INTERFACE ("4 to 8 % lower"), formulation 7 ("4 to 8 %") | formulation 0, 3.4, INTERFACE last section ("4.0 to 8.4 %") | rounding only; harmless |

Cross-references checked and correct: formulation 0 (3.4, 4.1, 4.3, 5.4, 6), 3.1 (3.2, 3.4
last row), 3.4 (4.2), 4.1 (4.3, 7), 4.2 (6, 4.4), 4.3 (6 V3), 4.4 (5.2), 5.2 (Prereg 6),
5.3 (5.2), 5.4 (Prereg 9); Prereg 2 (formulation 5.2, 6.3, compute §1), 3 (formulation 0
pt 4, 4.3), 4 (formulation 5.3, 5.4), 5 (Sections 6.1, 7), 6.3 (formulation 6), 9
(formulation 5.4, compute §6 item 3, Sections 2, 5, 6.1); compute (Prereg 2, 9); INTERFACE
(formulation 0 pt 6, 3.2, 4.2, 7); lit review (formulation 4.1, 4.2, 5.5, 6). Arm lists agree
between Prereg 3 and compute 2 and 3 (U+proj has no training cost, by design). Set sizes and
cycle counts agree between Prereg 2 and compute 1 (4528 cycles).

## Part C. New errors and re-checks

### C1. Compute arithmetic (docs/compute_3B.md)

- Cycles: 160 + 1536 + 192 + 240 + 1920 + 480 = 4528 ✓. At 30 s: 37.7 core-hours ✓; on 4
  workers 9.4 h, quoted 9.5 ✓.
- Storage: 128 × 3 × 200 × 4 B = 0.307 MB ✓ for three fields, but the wall field makes it 0.41
  MB per cycle and about 1.85 GB in total (problem 13).
- Per run at 8000 iterations: U 0.233 h ✓; U+F 0.556 h ✓; P 0.44 h (quoted 0.5, rounded up);
  S 1.45 s gives 3.22 h (quoted 3.25); S−M 2.89 h ✓; S−Π 2.22 h (quoted 2.3). The quoted sum
  9.74 h ✓. H2 arms 0.23 + 0.56 + 0.5 + 3.25 = 4.54 h ✓; 14 runs (9 + 5) × 4.54 = 63.6 h ✓.
- Totals: 9.5 + 20 + 7 + 49 + 64 + 13 = 162.5 h ✓. "About 110 of the 162 hours" for the
  structured arms: 42.3 (primary) + 16.9 (pilot) + 45.5 (H2, S only) = 104.7 h plus their
  share of evaluation ✓.
- Timing table against `timing_scaling.json`: 2.47 / 3.83 / 6.08 and 3.03 / 5.06 / 8.09
  CPU-s ✓; RHS ranges 1347 to 1564 and 659 to 983 ✓. Growth factor caveat in problem 20.

### C2. Decision-list walk-through (Prereg Section 5)

A = H1 or H2 SUPPORTED. A_s = no SUPPORTED, at least one SMALL. "dem" = H4 demoted.

| H1, H2 | H4 = EQ | S BETTER | S WORSE | INCONCL. | dem | sensible? |
|---|---|---|---|---|---|---|
| at least one SUPPORTED (7 combinations) | 2 | 2 | 3 | 4 | 4 | yes; rows 2 and 3 should name which hypothesis |
| no SUPPORTED, at least one SMALL (5) | 5 | 5 | 5 | 5 | 5 | no for S WORSE: cost hidden |
| both REFUTED (1) | 6 | **6** | 7 | **9** | **9** | no for S BETTER ("only" false) and INC/dem ("at this power" false) |
| one REFUTED and one INCONCLUSIVE, or both INCONCLUSIVE (3) | 9 | 8 | **9** | 9 | 9 | no for S WORSE: measured cost hidden |
| any gate failure | 1 | 1 | 1 | 1 | 1 | yes |

Every combination reaches exactly one row (first match plus the catch-all row 9), so the list
is an ordered partition. The bold cells are the four defects of problem 2. Under the current
REFUTED rule the "both REFUTED" row is itself reached by chance about a quarter of the time
under an exact null (problem 1).

### C3. Attribution rule

See problem 7: "beats" undefined; failure to beat is turned into a positive attribution;
only scalar features in U+F (problem 8). One more gap: the rule is applied "when SUPPORTED or
SMALL", but row 8 (H4 S BETTER) also claims a benefit and has no attribution.

### C4. Verification numbers (formulation Section 6 table) against the JSON

| claim | JSON | verdict |
|---|---|---|
| V1: 7×10⁻¹⁰ c/c_in, 6×10⁻⁹ q/q_max, 1.5×10⁻⁷ K, rise 16.7 K | 7.25e-10, 5.82e-9, 1.49e-7, 16.67 | supported |
| V2: water 6×10⁻¹⁵, energy 3×10⁻¹⁶, entropy identity 3×10⁻¹⁶, worst −7×10⁻²² | 5.53e-15, 2.74e-16, 3.47e-16, −6.80e-22 | supported |
| V2 trapezoid closure 1×10⁻⁵ to 8×10⁻⁵ (Section 6 item 2) | 1.29e-5 to 8.17e-5 | supported |
| V3: 200 states, −2×10⁻⁵ and −3×10⁻¹⁰ | 200, −2.34e-5, −3.41e-10 | supported (the note correctly claims semidefinite only) |
| V4: 602 snapshots, inequality held, balances exact only with fine outlet; 4.0 % and 0.5 % with the coarse outlet (5.2) | 602, fraction negative 0.0, 4.85e-16 / 4.06e-16 fine, 0.0403 / 0.0052 coarse | supported |
| V4 "with margin"; 3×10⁻⁵ wall floor; 0.016 K (5.2) | not in JSON | unsupported by the repository (true on re-run: +0.0118, 2.49e-5, 0.0159 K) |
| Compute: k = 5.4×10⁻⁴ s⁻¹ | 5.41e-4 | supported |

### C5. Code against formulation 4.2 to 4.4 (`verify/consistent_column.py`)

- Signs: face fluxes positive in +z; `div` removes J/Δz from cell i and adds it to i+1
  (lines 171 to 175) ✓. LDF r = A k (q* − q), dn −= r, da += r ✓; σ_LDF = r(w_a − w_n) =
  r(μ_v − μ_a)/T ✓.
- η = Δ(φ°/T)/Δ(1/T) (lines 150 to 154), so X_n + ηX_u = −RΔln c exactly ✓
  (Gibbs-Helmholtz d(φ°/T)/d(1/T) = e_w re-derived); fallback e_w(T_mid) when Δ(1/T) ≈ 0 ✓.
  J_u = ηJ_n − k_zΔT/Δz ✓. Face production divided by Δz ✓.
- Wall as a port: W = h_a(T − T_w) leaves u in `rhs`; −ΣW/T_w in the entropy port; σ_wall =
  W(1/T_w − 1/T) ✓. No wall block in any operator ✓.
- First-order upwind: g_{i−1} − g_i with g_0 the feed, outflow at the cell's own state ✓;
  σ_adv is the Bregman form of 4.3 ✓.
- e_0 = s_0 = 0 ✓ (T_0 = 298.15 K).
- One difference from the design: `initial_state` uses a 10⁻¹² clean bed, not the pre-wetted
  equilibrium of 5.2. Harmless for design verification; the study solver must use the
  pre-wet.

### C6. Other re-checks that hold

- Pre-wet loading 6.9×10⁻⁵ mol kg⁻¹, swing 20.24 mol kg⁻¹ (303 to 353 K at RH 0.40 feed),
  ratio 3.4×10⁻⁶ ✓ ("about 3×10⁻⁶").
- RT/|ΔH|: 3.99 % (288 K, 60 kJ), 8.38 % (383 K, 38 kJ) ✓ as quoted.
- MDEs 0.45, 0.54, 0.64 SD ✓.
- TOST power (paired t, n = 32): at SD 0.20, 0.68 (α 0.05) and 0.36 (α 0.0167) at m = 1.10;
  at m = 1.20, 0.94 at α 0.0167 up to SD 0.25 and 0.77 at SD 0.30. The cap at 1.20 is
  reasonable.

## Part D. Style

No em dash (U+2014) occurs in any of the documents, scripts or tests. En dashes appear only in
ranges and finding ranges ("F31–F33", "(M1)–(M3)"), which is standard. The prose is plain.
One stylistic nit: lit review Section 1 has two consecutive sentences starting "The nearest"
(lines 76 to 78), which name two different "nearest" models.
