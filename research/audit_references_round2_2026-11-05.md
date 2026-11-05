# Reference audit, round 2 (re-check of the fixes), 2026-11-05

Scope: every sentence that cites a source and changed in `git diff 1c02d1e..HEAD` (commit
1613f4e) in `docs/lit_review_3B.md` (LR), `docs/formulation_3B.md` (F) and
`PREREG_P3B_draft.md` (PR), checked against the round-1 report
`research/audit_references_2026-11-05.md`. Read-only: no design document was edited.

Method. Live fetches on 2026-11-05 with curl, no e-mail address or mailto parameter in any
request: Crossref `/works/DOI` (26 DOIs), the arXiv API (2610.05570v1, 2604.04266,
2506.08475, 2610.37333, 2609.04530, 2302.11002, 2210.07182, 2403.05848), arXiv HTML full
texts (2403.05848v2, 2604.04266v1, 2609.04530v2, 2302.11002v4), OpenAlex abstracts (7
works, plus a second check of the CMAME 2021 volume), the HAL API (tel-05363400), the
Europe PMC supplementary files of PMC6813556 (Hanikel et al. SI PDF, Figure S25 viewed as
an image), and the NOLCOS 2013 PDF (P28). Journal B entries P6, P8, P10, P15, P28, P29 were
read from `research/lit_B_thermo_column_models.md`.

## 1. Table

| # | sentence location | citation | details OK? | supports? | fix |
|---|---|---|---|---|---|
| 1 | LR §1 heading "every latent version found learns the entropy" | (section) | n/a | PARTLY. "found" is there, but one of the four latent versions listed, Schaffer and Exl, learns a single scalar and no entropy (v2 abstract: "A learned scalar generates the latent dynamics") | e.g. "every latent GENERIC model found learns its entropy" |
| 2 | LR §1 GFINNs | Phil. Trans. R. Soc. A 380, 20210207, 2022 | OK (Crossref: 380(2229), art. 20210207, online 2022-06-20) | yes | none |
| 3 | LR §1 port-metriplectic | Comput. Mech. 2023, doi:10.1007/s00466-023-02296-w | OK (Crossref: 72(3) 553-561, 2023) | yes | optional: add 72, 553 |
| 4 | LR §1 Hernández et al. thermodynamics-aware ROM | CMAME 380, 113763, 2021 | **WRONG volume.** Crossref and OpenAlex: CMAME **379** (June 2021), 113763, doi:10.1016/j.cma.2021.113763. (Round 1 gave 380; the error came from there.) | yes | 380 -> 379 |
| 5 | LR §1 tLaSDI | Park et al., CMAME 429, 117144, 2024 | OK (Crossref: 429, 117144, Sept 2024) | yes | none |
| 6 | LR §1 parametric tLaSDI | He et al., TMLR 2026 | OK (arXiv journal_ref "TMLR, 2026") | yes | none |
| 7 | LR §1 magnetisation operator | Schaffer and Exl, arXiv:2609.04530, v2 Section 4.5 and rollout figures | OK (v2, 24 Oct 2026) | yes. v2 full text: the latent scalar is "a structural constraint rather than an approximation of the Gibbs energy, as discussed in Section 4.5"; longer rollouts in the results | none |
| 8 | LR §1 "about 3 to 7 times lower ... 2.47×10⁻³ against 1.70×10⁻² ... smallest training loss of 10 ... in one of its two ablations the modelling loss added nothing once the Jacobian loss was present" | 2403.05848v2 (C17) | OK | yes. Ratios 5.7, 6.9 (Couette), 3.3, 5.1 (gas containers); "smallest training loss among 10 simulations"; gas containers: 5.52e-3 with Jacobian loss, 5.52e-3 with all terms; Couette: full loss best (3.57e-3) | none |
| 9 | LR §1 Zakwan et al. | IFAC-PapersOnLine 56(2), 5855, 2023 | OK (56(2) 5855-5860) | yes (unchanged content) | none |
| 10 | LR §1 "In the first version ... (arXiv:2610.05570v1), the friction class provably excludes transport" | 2610.05570v1 | OK (v1, 4 Oct 2026) | yes ("provably excluding transport") | none |
| 11 | LR §1 Beckers and Colombo: GP-learned Hamiltonian, shallow-water channel with linear friction, dissipation, no thermodynamic entropy or second-law structure | arXiv:2604.04266 | OK | yes. Abstract: GP-dPHS infers the Hamiltonian, shallow water; full text: "subject to linear friction", D ≥ 0; the words entropy, thermodynamic, second law, irreversible do not occur | none |
| 12 | LR §1 Bermejo et al.: nodal port-metriplectic, incompressible flow past obstacles, E and S gradients learned | Adv. Model. Simul. Eng. Sci. 12:27, 2025 | OK (12(1), art. 27, 2025-09-30) | yes (abstract, word for word in substance) | none |
| 13 | F §4.1 "Their first version (v1) also notes that their friction class cannot represent transport" | 2610.05570v1 | OK | yes | none |
| 14 | F §4.1 BC-IPHS | Ramirez, Le Gorrec, Maschke, Chem. Eng. Sci. 248, 117107, 2022, doi:10.1016/j.ces.2021.117107 | OK (248, 117107, Feb 2022); the wrong arXiv DOI field was not copied | yes | none |
| 15 | F §4.1 port-metriplectic models | Hernández et al., arXiv:2211.01873 | VoR not added here (round-1 fix 20 said "wherever the arXiv ID is cited") | yes | add Comput. Mech. 72, 553 (2023) |
| 16 | LR §2 heading and F §7 "every one found / every prior structured column model found is isothermal, judged from abstracts only" | Baaiu 2008, Peng 2007, Baaiu J. Process Control 2009 | OK | PARTLY. The J. Process Control paper has no abstract (as the same paragraph says), so it cannot be judged isothermal "from abstracts"; the LR §5 table also lists it under "isothermal" | say "judged from abstracts (none exists for the J. Process Control paper)", or drop it from the isothermal list |
| 17 | LR §2 and F §7 "The J. Process Control paper has no abstract in any database" | doi:10.1016/j.jprocont.2008.07.002 | OK (19(3) 394-404, March 2009) | PARTLY. New absolute: round 1 checked only Crossref, OpenAlex and HAL | "no abstract in Crossref, OpenAlex or HAL" |
| 18 | LR §2 "In the 2008 column model there is no heat of adsorption, no thermal domain ..." | Baaiu 2008, MCMDS 14, 195 | OK | yes (abstract "under isothermal conditions"; unchanged from round 1) | none |
| 19 | LR §2 and F §7 "The 2009 general theory covers heat as well as mass transfer, but no fetched abstract applies it to a non-isothermal (adsorption) column" | doi:10.1080/13873950902808578 | OK (MCMDS 15(3) 233-254) | yes (abstract: "systems subject to heat and mass transfer"; no column) | none |
| 20 | LR §2 "Irreversible port-Hamiltonian models of reaction-diffusion and conduction-diffusion leave convection out (P6, P8, P10)" | journal B P6, P8, P10 | OK | yes (P8 "Excluding systems with convection"; P10 "without convection"; P6 diffusion-reaction example) | none |
| 21 | LR §2 "boundary supply in passivity analyses (P15, P29), where in one study convection is what can break the otherwise automatic dissipation condition (P28)" | P15, P28 (NOLCOS 2013 full text), P29 | OK | PARTLY. P28 (read live): "the dissipation condition holds" without convection; the case "where nonlinear phenomenological relations ... and convection are present ... may destroy the dissipation condition". Convection is one of two factors. Also "where in one study" reads as if P28 were one of P15/P29 | "... (P15, P29). A third study finds that convection, together with nonlinear kinetics, may destroy the otherwise automatic dissipation condition (P28)" |
| 22 | LR §2 Santamaria-Holek | J. Non-Equilib. Thermodyn. 37, 2012, doi:10.1515/jnetdy-2011-0029 | OK (37(3), 2012) | yes | none (F §4.2 gives the journal without DOI: fine) |
| 23 | LR §2 and F §3 Schwamberger and Schmidt: reversible-path heat capacities make residuals much smaller (water on zeolite 13X) | Ind. Eng. Chem. Res. 52, 16958, 2013 | OK (52(47) 16958-16965) | yes ("significantly smaller", "water–zeolite 13X") | none |
| 24 | LR §2 "No adsorption-cycle model found carries entropy production as structure; in these studies it is computed after the fact" | Li 2019, Guo 2019 | OK | yes (scoped, "found") | none |
| 25 | LR §3 Lastrucci et al. | ESCAPE 35, Syst. Control Trans. 4, 1718, 2025, doi:10.69997/sct.108423 | OK (4, 1718-1723, 2025-07-01) | yes | none |
| 26 | LR §3 FINN | Praditia et al., Water Resour. Res. 58, e2022WR033149, 2022 | OK (58(12), e2022WR033149) | yes (abstract: learns closure relations; diffusion-sorption in clay; distinguishes sorption isotherms) | optional: add the DOI 10.1029/2022WR033149 |
| 27 | LR §3 hybrid chromatography | Frandsen et al., AIChE J. 71, e70045, 2025, doi:10.1002/aic.70045 | OK (71(12), e70045) | yes (content unchanged) | none |
| 28 | LR §3 PINN surrogates of adsorption | Galanti et al., Processes 13, 2824, 2025, doi:10.3390/pr13092824 | OK | yes | none |
| 29 | LR §3 PSA PINN | Wu et al., Green Chem. Eng. 6, 233, 2025, doi:10.1016/j.gce.2024.08.004 | OK (6(2) 233-248, June 2025) | yes | none |
| 30 | LR §3 "Bilal, Haq and Asif 2026: scalar-output surrogates of a non-isothermal chromatography general rate model, with physics used only to build input features" | IECR 65, 13942, doi:10.1021/acs.iecr.6c02282; ACS Omega 11, 48428, doi:10.1021/acsomega.6c04740 | details OK (IECR 65(26) 13942-13958; ACS Omega 11(32) 48428-48449). Author list: the ACS Omega paper has a fourth author (Alhartomi) | PARTLY. IECR: "physics-derived dimensionless ratio features" appended to inputs (supported). ACS Omega: XGBoost on 31 dimensionless GRM parameters plus a SHAP consistency check against the GRM sensitivities; no physics-derived features | "Bilal et al. 2026 (two papers): scalar-output surrogates of a non-isothermal GRM, unconstrained; one adds physics-derived input features (IECR), the other checks SHAP rankings against the GRM (ACS Omega)". Round-1 fix 12 also asked for a journal D entry: none exists (grep finds Bilal only in LR) |
| 31 | LR §3 and §7 Glover: "20 times faster than a representative MATLAB implementation" | Comput. Chem. Eng. 216, 109904, online 2026 | OK (216, 109904; Crossref issue date Jan 2027) | yes ("by a factor of 20 relative to a representative MATLAB implementation") | none |
| 32 | LR §3 diff-soup "trains a neural field for the catalyst activity inside a stiff, differentiable 3-D packed-bed reactor solver on GPU" | arXiv:2610.37333 | OK | yes (abstract) | none |
| 33 | LR §3 "No learned sorption model found reports zero unphysical outputs over many cycles" | (search) | n/a | yes (scoped) | none |
| 34 | LR §3 Mousa et al. "One candidate exception was checked and ruled out: ... It cites GFINNs, but the first author's open thesis shows neural networks corrected by data reconciliation on balances and equilibria, with no adsorption" | Comput. Chem. Eng. 199, 109156 (2025); HAL tel-05363400 | OK (199, 109156, Aug 2025; HAL produced 2025-10-10) | PARTLY. The thesis abstract supports the description (NN + data reconciliation; mass balances, reaction equilibria; flash drum, distillation, Gibbs reactor). But the article itself is still unread (LR §6 item 1 says so), so "checked and ruled out" rests on the thesis, not the article | "was checked through the first author's thesis and is very likely not a threat" or "ruled out on the evidence of the thesis" |
| 35 | LR §4 Urdeitx et al. | J. Non-Newton. Fluid Mech. 349, 105601, 2026 | OK (349, 105601, June 2026) | yes | none (LR §7 still cites only arXiv:2512.16305: optional) |
| 36 | LR §4 ProbConserv: "a baseline that enforces conservation by shifting every point equally produced a negative bias in the zero-density region" | Hansen et al., Physica D 457, 133952, 2024 | OK (457, 133952, Jan 2024; arXiv journal_ref agrees) | yes (v4: "Since HardC-ANP shifts every point equally, it induces a negative bias in the zero (degeneracy) region") | none |
| 37 | LR §4 Nooraiepour et al. | Energy and AI 24, 100757, 2026 | OK (24, 100757, May 2026) | yes (content unchanged) | none |
| 38 | LR §4 "Headline numbers can come from the best of several runs ... (C17). The standard benchmark suite does not test thermodynamic structure at all (PDEBench, C31)" | 2403.05848v2; arXiv:2210.07182 | OK | yes for C17. C31: supported only by absence in the abstract; "The standard ... at all" is stronger than an abstract can show | "PDEBench (C31), a standard benchmark suite, does not test thermodynamic structure (judged from its abstract)" |
| 39 | LR §5 table "non-isothermal" row | Baaiu 2008; Peng 2007; Baaiu 2009 J. Process Control; "read from abstracts only" | OK | PARTLY (same as row 16: no abstract exists for the J. Process Control paper) | as row 16 |
| 40 | PR, Material: Hanikel source line | ACS Cent. Sci. 5, 1699 (2019), doi:10.1021/acscentsci.9b00745 | OK (5(10) 1699-1706) | yes | none |
| 41 | PR: "the step at about 0.12 RH is stated in the text" | Hanikel 2019 | OK | yes (round 1, PMC text "∼12% RH") | none |
| 42 | PR: "ΔH = −52 kJ mol⁻¹ is a reading of SI Figure S25 (Section S3.2 ...). The isosteric heat there plateaus at about 51 to 54 kJ mol⁻¹ between about 80 and 440 cm³ g⁻¹ of uptake, and falls at both ends. The main text gives 50 to 54 kJ mol⁻¹ for three sorbents together" | SI PDF from Europe PMC PMC6813556 | OK (Figure S25, page S20, Section S3.2 "Estimation of the Isosteric Heats of Adsorption") | yes, with a small omission. Figure S25 (viewed): Q_st about 51 at 80 cm³ g⁻¹, rising slowly to about 53.7 at 400 to 440; then a peak of about 55.5 near 470 before falling to about 45 at 560; low end falls to about 41 at 40. 52 is a fair reading of the plateau (its mean is nearer 53) | optional: "falls at both ends, after a peak of about 55 near 470 cm³ g⁻¹" |
| 43 | PR: convention ΔH = −Q_st; Q_st = −ΔH + RT; RT/\|ΔH\| ≈ 5 % | (derivation, no citation) | n/a | arithmetic OK: RT/52 kJ = 4.8 % at 303 K; F row 202: 4.1 to 8.4 % over 293 to 383 K and ΔH 38 to 60; 4.7 to 6.1 % at 52 kJ mol⁻¹ | none |
| 44 | PR: "q_max and f_H are read from the figure (Fig. 1a)" | Hanikel 2019 | OK | consistent with round 1 (figure readings) | none |

## 2. Round-1 fixes: present?

| round-1 fix | present? |
|---|---|
| 1 tLaSDI ratio, ablation, VoR | yes |
| 2 Beckers and Colombo reworded, "The nearest" merged | yes |
| 3 Bermejo port-metriplectic | yes |
| 4 pin to 2610.05570v1 (LR and F) | yes |
| 5 LR §1 heading "found" | yes (but see row 1) |
| 6 entropy production scoped to adsorption cycles | yes |
| 7 Baaiu 2009 MCMDS heat transfer; "no thermal domain" limited to 2008 | yes |
| 8 "only isothermal" -> "found ... from abstracts" (LR heading, F §7) | yes (but see rows 16, 39) |
| 9 P6/P8/P10 and P15/P28/P29 sentences | yes (P28 nuance, row 21) |
| 10 Schwamberger and Schmidt restated | yes (LR §2, F §3; LR §5 and §7 only say the principle is known, which is fine) |
| 11 FINN | yes (DOI optional) |
| 12 Bilal DOIs and a journal D entry | DOIs yes; **journal D entry missing**; description wrong for ACS Omega (row 30) |
| 13 Glover baseline (§3 and §7) | yes |
| 14 diff-soup | yes |
| 15 "found" in the unphysical-outputs sentence | yes |
| 16 ProbConserv HardC baseline, Physica D | yes |
| 17 best-of-N replaced | yes (row 38 wording) |
| 18 Ramirez et al. Chem. Eng. Sci. | yes |
| 19 MOF-303 values labelled; SI S3.2 read | yes |
| 20 versions of record | yes in LR §1-§4, but CMAME volume wrong (row 4) and F §4.1 still cites 2211.01873 alone (row 15); Gruber, Zhong, Lanthaler conference versions still to check before submission, as allowed |
| 21 minor metadata | Green Chem. Eng. 6 (2025) yes; Chung vol. 463 not added (optional) |

## 3. New absolute claims introduced by the fixes

- "no abstract in any database" (LR §2, F §7): unscoped; round 1 searched three databases (row 17).
- "The standard benchmark suite does not test thermodynamic structure at all" (LR §4): "at all" from an abstract (row 38).
- "every latent version found learns the entropy" (LR §1 heading): has "found", but is false for one listed model (row 1).
- "checked and ruled out" (LR §3, Mousa): a definitive verdict from indirect evidence (row 34).
- "Every prior structured column model found is isothermal" (F §7), "every one found is isothermal (from abstracts only)" (LR §2): scoped, but the scope does not cover the J. Process Control paper (rows 16, 39).

## 4. Still wrong (list)

1. LR §1: Hernández et al. thermodynamics-aware ROM is CMAME **379** (2021) 113763, not 380.
2. LR §3: Bilal et al. sentence: "physics used only to build input features" fits the IECR paper only; the ACS Omega paper has four authors (adds Alhartomi). The journal D entry asked for in round 1 is missing.
3. LR §2 and F §7: "no abstract in any database" should be scoped to Crossref, OpenAlex and HAL.
4. LR §2 heading, LR §5 table, F §7: the J. Process Control 2009 paper is classed as isothermal "from abstracts", but it has no abstract.
5. LR §2: P28 attributes the possible loss of the dissipation condition to convection together with nonlinear phenomenological relations; "where in one study" wrongly suggests P28 is one of P15/P29.
6. LR §1 heading: "every latent version found learns the entropy" is not true of the Schaffer and Exl operator listed under it (a single learned scalar, no entropy).
7. LR §3: Mousa et al. "checked and ruled out" overstates; the article is unread and the verdict rests on the thesis abstract.
8. LR §4: "does not test thermodynamic structure at all" goes beyond what the PDEBench abstract shows.
9. Minor: F §4.1 cites port-metriplectic networks by arXiv ID only (add Comput. Mech. 72, 553, 2023); PR's Figure S25 description omits the about 55 kJ mol⁻¹ peak near 470 cm³ g⁻¹ before the high-end fall.
