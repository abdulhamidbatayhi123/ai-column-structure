# Reference audit of the Paper 3B design documents, 2026-11-05

Scope: every citation in `docs/lit_review_3B.md` (LR), `docs/formulation_3B.md` (F),
`PREREG_P3B_draft.md` (PR) and `docs/INTERFACE_3A.md` (IF). Read-only audit: no design
document was edited.

Method. Each record was fetched live on 2026-11-05 with curl: the arXiv API
(export.arxiv.org, one batch call for 33 IDs plus v1 calls for 2610.05570 and 2609.04530),
Crossref `/works/DOI` and `query.bibliographic`, OpenAlex `/works/doi:` (abstract rebuilt from
the inverted index), the Zenodo records API, the HAL API, Springer abstract pages, arXiv HTML
full texts (2403.05848v2, 2601.09491v1, 2609.04530v2, 2012.02334v6, 2606.15053v2,
2604.04266v1, 2302.11002v4), open PDFs (Santamaria-Holek arXiv PDF, the NOLCOS 2013 and
DYCOPS 2022 papers) and the PMC full text of Hanikel et al. 2019 (PMC6813556, via NCBI
efetch). No e-mail address or mailto parameter was sent in any request. OpenAlex search was
rate-limited (HTTP 429) and DBLP returned HTTP 202 with no body, so three conference
versions of record could not be confirmed live (marked "VoR not verified").

INTERFACE_3A.md cites no external literature (only 3A's and 3B's own files), so it has no rows.

## Summary

| verdict | count |
|---|---|
| SUPPORTS | 41 |
| PARTLY | 11 |
| DOES NOT SUPPORT | 3 |
| CANNOT VERIFY | 4 |
| total rows | 59 |

Separately, 13 citations lack a version of record that exists (metadata fix, no support issue).

## 1. Summary table

"VoR" = version of record. Row numbers are used in Section 2.

| # | citation | where in our docs | metadata OK? | support verdict | fix |
|---|---|---|---|---|---|
| 1 | GFINNs, Zhang, Shin, Karniadakis, arXiv:2109.00092 | LR §1 | VoR missing: Phil. Trans. R. Soc. A 380 (2022) 20210207, doi:10.1098/rsta.2021.0207 | SUPPORTS | add VoR |
| 2 | Gruber et al., arXiv:2405.16305 | LR §1 | OK (v3). VoR not verified (Crossref shows only an OSTI report) | SUPPORTS | check the NeurIPS 2024 record before submission |
| 3 | Hernández et al., port-metriplectic NNs, arXiv:2211.01873 | LR §1, §5 table; F §4.1 | VoR missing: Comput. Mech. (2023), doi:10.1007/s00466-023-02296-w | SUPPORTS | add VoR |
| 4 | Sulskis and Ravi, GENERIC-FNO, arXiv:2606.08343 (v9, 1 Nov 2026) | LR §1, §5 table | OK, preprint "Under review" | SUPPORTS | none |
| 5a | Sulskis and Ravi, Poisson-GENERIC, arXiv:2610.05570: 61 of 72 wins, fixed energy terms, learned-entropy ablation (v1), Lie-Poisson L | LR §1, §4; F §0.3, §4.1 | OK (v1 4 Oct, v2 12 Oct 2026) | SUPPORTS | none |
| 5b | same, "friction class provably cannot represent transport" | LR §1; F §4.1 | wording only in v1 | PARTLY | pin to v1 |
| 6 | Hernández et al., arXiv:2007.03758 | LR §1, §5 table | VoR missing: CMAME 380 (2021) 113763 | SUPPORTS | add VoR |
| 7 | Park et al., tLaSDI, arXiv:2403.05848 | LR §1, §4 | VoR missing: CMAME 429 (2024) 117144 | PARTLY ("5 to 8 times" is wrong; ablation overstated) | correct numbers, add VoR |
| 8 | He et al., parametric tLaSDI, arXiv:2506.08475 | LR §1 | VoR missing: TMLR 2026 (arXiv journal_ref) | SUPPORTS | add VoR |
| 9 | Schaffer and Exl, arXiv:2609.04530 | LR §1, §4, §7; F §5.5 | now v2 (24 Oct 2026), same title; the v2 abstract drops the Gibbs and rollout sentences, which survive in the v2 full text (Sec. 4.5 and the rollout figures) | SUPPORTS | optional: cite v1 abstract or v2 Sec. 4.5; add "on their test problem" |
| 10 | Alim et al., arXiv:2610.20614 | LR §1, §5 table | OK (API published date 2026-07-12; SSRN copy doi:10.2139/ssrn.7113502) | SUPPORTS | none |
| 11 | Chung, Bu, Verma, Physics-conforming Latent Twins, arXiv:2606.15053 | LR §1, §5 table | CMAME 463, 119351; issue dated Jan 2027 (online 2026) | SUPPORTS | optional: give volume 463 |
| 12 | Baheri and Lindemann, arXiv:2509.19526 | LR §1 | OK | SUPPORTS | none |
| 13 | Zakwan et al., arXiv:2211.06130 | LR §1 | VoR missing: IFAC-PapersOnLine 56(2) (2023) 5855-5860, doi:10.1016/j.ifacol.2023.10.079 | SUPPORTS | add VoR |
| 14 | Beckers and Colombo, arXiv:2604.04266 | LR §1 | OK | DOES NOT SUPPORT ("conservative") | reword |
| 15 | Bermejo et al. 2025, AMSES 12:27, doi:10.1186/s40323-025-00311-8 | LR §1 | OK | PARTLY (omits that it is port-metriplectic) | reword |
| 16 | Ouardi, Sarkar, Ramirez, Le Gorrec, arXiv:2603.11260 | LR §2; F §4.1 | OK, preprint | SUPPORTS | none |
| 17 | Ramirez, Le Gorrec, Maschke, BC-IPHS, arXiv:2104.13459 | F §4.1 (and LR §2 as P6) | VoR missing: Chem. Eng. Sci. 248 (2022) 117107, doi:10.1016/j.ces.2021.117107. The arXiv DOI field points to a different paper (Entropy 24:1478, "An Overview on IPHS") | SUPPORTS | cite Chem. Eng. Sci.; do not copy the arXiv DOI field |
| 18 | P6, P8, P10 (Ramirez 2021; Mora et al. arXiv:2603.11265; Zhou et al. 2012/2015/2016) | LR §2 | OK | PARTLY ("reactor models", "on purpose") | reword |
| 19 | P14, Kadima Kazaku et al., IFAC-PapersOnLine 53 (2020), doi:10.1016/j.ifacol.2020.12.1066 | LR §2 | OK | SUPPORTS | none |
| 20 | P15 (Zarate-Navarro et al., IEEE Access 2022), P28 (Hoang and Dochain, NOLCOS 2013), P29 (Cai and Ydstie, DYCOPS 2022) | LR §2 | OK | PARTLY (only P28 says convection breaks automatic passivity) | reword |
| 21 | P13, Zitte et al. 2019, hal-02385418 | LR §2 | OK | SUPPORTS | none |
| 22 | Baaiu et al. 2008, MCMDS 14:195-211, doi:10.1080/13873950701844840 | LR §2, §5, §6; F §7 | OK | SUPPORTS | none |
| 23 | Baaiu et al. 2009, MCMDS 15:233-254, doi:10.1080/13873950902808578 | LR §2, §5 table, §6 | OK | PARTLY (abstract covers heat transfer; it is not an isothermal column model) | reword |
| 24 | Peng, Baaiu, Couenne, Le Gorrec, IEEE CDC 2007, pp. 3513-3518, doi:10.1109/cdc.2007.4434663 | LR §2; F §7 | OK (title: "Energy based discretization of a class of distributed parameters processes...") | SUPPORTS | none |
| 25 | Baaiu et al., J. Process Control 19 (2009) 394-404, doi:10.1016/j.jprocont.2008.07.002 | LR §2; F §7 | OK | CANNOT VERIFY (no abstract in Crossref, OpenAlex or HAL; title only) | keep the "not read" caveat; do not attach "no thermal domain" to it |
| 26 | Santamaria-Holek, Grzywna, Rubi, arXiv:1204.3841 | LR §2, §5; F §4.2 | VoR missing: J. Non-Equilib. Thermodyn. 37 (2012), doi:10.1515/jnetdy-2011-0029 | SUPPORTS (full text) | add VoR |
| 27 | Dytrych and Dlaskova, HARMONIA, ChemRxiv 2025, doi:10.26434/chemrxiv-2025-c860f | LR §2, §5, §6; F §4.2 | OK (posted 2025-11-30; no VoR found) | SUPPORTS | none |
| 28 | Walton and LeVan, Adsorption 11 (2005) 555-559, doi:10.1007/s10450-005-5984-6 | LR §2, §5, §7; F §3 | OK | SUPPORTS | optional: pages |
| 29 | Schwamberger and Schmidt, Ind. Eng. Chem. Res. 52 (2013) 16958-16965, doi:10.1021/ie4011832 | LR §2, §5, §7; F §3 | OK | PARTLY (direction of the finding restated) | reword |
| 30 | Myers and Monson, Adsorption 20 (2014) 591-622, doi:10.1007/s10450-014-9604-1 | LR §2 | OK | SUPPORTS | none |
| 31 | Li et al., Sci. Rep. 9:16599 (2019), doi:10.1038/s41598-019-53398-6 | LR §2 | OK | SUPPORTS (63.27 % and 53.72 % exact) | none |
| 32a | Guo et al., Entropy 21:285 (2019), doi:10.3390/e21030285 | LR §2 | OK | SUPPORTS | none |
| 32b | absolute: "No model has entropy production as structure." | LR §2 | n/a | DOES NOT SUPPORT (our own IPHS sources have it) | restrict to adsorption cycles |
| 33 | Mukherjee and Bhattacharyya, Ind. Eng. Chem. Res. 63 (2024) 14211, doi:10.1021/acs.iecr.4c01429 | LR §3 | OK | SUPPORTS | none |
| 34 | Lastrucci et al., Picard-KKT-hPINN, arXiv:2501.17782 | LR §3 | VoR missing: ESCAPE 35, Syst. Control Trans. 4 (2025) 1718-1723, doi:10.69997/sct.108423 | SUPPORTS | add VoR |
| 35 | Praditia et al., FINN, Water Resour. Res. 58 (2022) e2022WR033149 | LR §3 | OK (no authors or DOI given in LR) | PARTLY ("learning only the isotherm") | reword, add DOI |
| 36 | Frandsen et al., AIChE J. 71 (2025) e70045, doi:10.1002/aic.70045 | LR §3 | OK (no authors or DOI in LR) | SUPPORTS | add DOI |
| 37 | Galanti et al., Processes 13 (2025) 2824, doi:10.3390/pr13092824 | LR §3 | OK (no authors or DOI in LR) | SUPPORTS | add DOI |
| 38 | Wu et al., Green Chem. Eng. 6 (2025) 233-248, doi:10.1016/j.gce.2024.08.004 | LR §3 | LR says 2024; the issue is 2025 (online 2024) | SUPPORTS | give 6 (2025) |
| 39 | Staudt et al., Zenodo 10.5281/zenodo.22556794 | LR §3 | OK (2026-09-06; the article is not yet in Crossref) | SUPPORTS (1.1 million exact) | none |
| 40 | Ceccanti et al., arXiv:2601.09491 | LR §3 | OK, preprint | SUPPORTS (full text) | optional: add "one species, linear isotherm" |
| 41 | "Bilal et al. 2026" | LR §3 | no identifier; not in any source journal | CANNOT VERIFY as cited | add the DOI (two candidate papers found) |
| 42 | An, Wald, Chen, King, Front. Environ. Eng. 5 (2026) 1881554 | LR §3 | OK | SUPPORTS (about 120,000 exact) | none |
| 43 | Glover, Papathanasiou, Pini, Comput. Chem. Eng. 216, 109904 | LR §3, §7 | issue dated Jan 2027 (online Sep 2026) | PARTLY ("20 times faster" lacks its baseline) | reword |
| 44 | Degnan-Morgenstern and Bazant, diff-soup, arXiv:2610.37333 | LR §3 | OK | PARTLY ("neural closure") | reword |
| 45 | Mousa et al., Comput. Chem. Eng. 199 (2025) 109156 | LR §3, §6 | OK | SUPPORTS (claims are hedged) | none |
| 46 | Mousa PhD thesis, HAL tel-05363400 | LR §6 | OK (produced 2025-10-10, deposited 2025-11-13) | SUPPORTS | none |
| 47 | Wang, Wei, Ling, arXiv:2606.01366 | LR §4, §7 | OK | SUPPORTS (all numbers exact) | none |
| 48 | Zhong, Dey, Chakraborty, arXiv:2012.02334 | LR §4 | OK. VoR (L4DC 2021) not verified live | SUPPORTS (full text) | check VoR |
| 49 | Prakash and Klasky, arXiv:2607.10965 | LR §4 | OK | SUPPORTS | none |
| 50 | Gruver et al., arXiv:2202.04836 | LR §4 | OK (ICLR 2022 per arXiv comment) | SUPPORTS | optional: name ICLR 2022 |
| 51 | Urdeitx et al., arXiv:2512.16305 | LR §4, §7 | VoR missing: J. Non-Newton. Fluid Mech. 349 (2026) 105601, doi:10.1016/j.jnnfm.2026.105601 | SUPPORTS | add VoR |
| 52 | Hansen et al., ProbConserv, arXiv:2302.11002 | LR §4 | VoR missing: Physica D 457 (2024) 133952 | PARTLY (it was the HardC baseline, not ProbConserv) | reword, add VoR |
| 53 | Krishnapriyan et al., arXiv:2109.01050 (C34) | LR §4 | OK (NeurIPS 2021) | SUPPORTS | none |
| 54 | Nooraiepour et al., arXiv:2604.13992 (C37) | LR §4 | VoR missing: Energy and AI 24 (2026) 100757, doi:10.1016/j.egyai.2026.100757 | SUPPORTS | add VoR |
| 55 | PDEBench, Takamoto et al., arXiv:2210.07182 (C31), for "Reporting is often best-of-N runs" | LR §4 | OK (NeurIPS 2022 D&B) | DOES NOT SUPPORT | change source and wording |
| 56 | Lanthaler et al., arXiv:2210.01074 | F §0.6 | OK. VoR (ICLR 2023) not verified live | SUPPORTS (narrow point) | check VoR |
| 57 | Hanikel et al., ACS Cent. Sci. 5 (2019) 1699-1706, doi:10.1021/acscentsci.9b00745 | PR, Material | OK | CANNOT VERIFY in part (step supported; 52 kJ/mol only in the SI; q_max and f_H are figure readings) | see detail |
| 58 | Döppel, Bracconi, Maestri 2026; Myers 2002 (AIChE J. 48:145) | LR §6 | both labelled SEEN-ONLY in LR | CANNOT VERIFY (already disclosed) | none beyond the existing caveat |

## 2. Details for every item that is not SUPPORTS

### 5b. Poisson-GENERIC: friction class and transport (PARTLY)

Ours (LR §1): "Poisson-GENERIC shows that its friction class provably cannot represent
transport, so transport must enter through a Lie-Poisson reversible operator (T5)."
Ours (F §4.1): "They also note that their friction class cannot represent transport."

Source: the v1 abstract says "permits diffusion and damping while provably excluding
transport". The current v2 abstract (12 Oct 2026) softens this to "supports diffusion and
damping without an independent reversible transport generator". The claim holds for v1 only.

Corrected (LR): "In the first version of the Poisson-GENERIC paper (arXiv:2610.05570v1), the
friction class provably excludes transport, so transport enters through a Lie-Poisson
reversible operator (T5)." Same pin to v1 in F §4.1.

### 7. tLaSDI (PARTLY)

Ours (LR §1): "It reports extrapolation errors 5 to 8 times lower than its baselines, for
example 2.47×10⁻³ against 1.70×10⁻² for a plain network on Couette flow (journal C, C17).
Two caveats travel with that number: it is the best of 10 runs, and an ablation found that one
of its losses added nothing."

Source (v2 HTML, Sec. 5.1): Couette 2.47e-3 against 1.41e-2 (TA-ROM) and 1.70e-2
(Vanilla-FNN); gas containers 3.96e-3 against 1.31e-2 and 2.01e-2. The ratios are 5.7, 6.9,
3.3 and 5.1, so the range is about 3 to 7, not 5 to 8. The figures come from "the model with
the smallest training loss among 10 simulations" (supported). In the loss ablation, the
modelling loss added nothing (5.52e-3 with and without it) only in the gas-container example,
not in the Couette one. The VoR is CMAME 429 (2024) 117144.

Corrected: "It reports extrapolation errors about 3 to 7 times lower than its baselines, for
example 2.47×10⁻³ against 1.70×10⁻² for a plain network on Couette flow (Park et al., CMAME
429, 117144, 2024; journal C, C17). Two caveats travel with that number: it comes from the run
with the smallest training loss of 10, and in one of the two ablations the modelling loss added
nothing once the Jacobian loss was present."

### 14. Beckers and Colombo 2026 (DOES NOT SUPPORT)

Ours (LR §1): "The nearest learned distributed-port model is conservative and learns its
Hamiltonian (Beckers and Colombo, arXiv:2604.04266, boundary control of port-Hamiltonian
PDEs)."

Source: the GP-dPHS model "infer[s] the unknown Hamiltonian structure from data" (supported).
But the test system is a shallow-water channel "subject to linear friction", and the paper is
built around the "dissipation obstacle" of internal friction. The model is not conservative.

Corrected: "The nearest learned distributed-port model learns its Hamiltonian with a Gaussian
process, on a shallow-water channel with linear friction. It has dissipation but no
thermodynamic entropy and no second-law structure (Beckers and Colombo, arXiv:2604.04266)."

Also, two consecutive sentences in LR §1 both begin "The nearest". Merge them.

### 15. Bermejo et al. 2025 (PARTLY)

Ours (LR §1): "The nearest is a thermodynamics-informed MeshGraphNet for open flow around
obstacles (Bermejo et al. 2025, T16). Its E and S are learned, it is full-field, and its
abstract does not say how boundary fluxes are treated."

Source (Crossref/OpenAlex abstract): it incorporates structure "via a nodal port-metriplectic
formulation"; decoders "predict local energy and entropy gradients". Our three statements are
true, but leaving out "port-metriplectic" understates how close it is to qualifier (c).

Corrected: "The nearest is a thermodynamics-informed MeshGraphNet with a nodal
port-metriplectic formulation for incompressible flow past obstacles (Bermejo et al., Adv.
Model. Simul. Eng. Sci. 12:27, 2025, T16). Its E and S gradients are learned, it is full-field
rather than latent, and its abstract does not say how inflow and outflow fluxes are treated."

### 18. IPHS models without convection (P6, P8, P10) (PARTLY)

Ours (LR §2): "Irreversible port-Hamiltonian reactor models leave convection out on purpose
(P6, P8, P10)."

Source: P10 (Zhou et al., HAL 2012) covers reactors "without convection". P6 (Ramirez et al.)
is a general BC-IPHS class whose reaction-diffusion example has no convection. P8 (Mora et
al., arXiv:2603.11265) treats conduction-diffusion and excludes convection. Only P10 is a
reactor model, and no source states a motive ("on purpose").

Corrected: "Irreversible port-Hamiltonian models of reaction-diffusion and conduction-diffusion
leave convection out (P6, P8, P10)."

### 20. Convection as boundary supply and passivity (P15, P28, P29) (PARTLY)

Ours (LR §2): "... as a boundary supply in passivity analyses where convection is exactly what
breaks automatic passivity (P15, P28, P29)".

Source: P15 (IEEE Access 2022) finds the entropy difference at the boundaries is
"directly proportional to the supply rate" (boundary supply: supported). P28 (NOLCOS 2013 full
text) says that with convection present "such a situation may destroy the dissipation
condition" (supported). P29 (DYCOPS 2022) is a lumped equilibrium reactor. It shows the feed-flow
to pressure map is passive and says nothing about convection breaking passivity.

Corrected: "... as a boundary supply in passivity analyses (P15, P29). In one of them convection
is what can break the otherwise automatic dissipation condition (P28)."

### 23. Baaiu et al. 2009, MCMDS (PARTLY)

Ours (LR §2): "The general port-based theory of mass transport followed (Baaiu et al. 2009 ...)
... There is no heat of adsorption, no thermal domain, no entropy as a state, and no learning."
Ours (LR §5 table): "the isothermal port-based column models of Baaiu et al. 2008 and 2009".

Source (OpenAlex abstract): the 2009 paper "applies to systems subject to heat and mass
transfer". So "no thermal domain" contradicts it, and it is a general theory, not an isothermal
column model. The 2008 paper is isothermal (supported). The 2009 J. Process Control paper has
no abstract anywhere (row 25).

Corrected (LR §2): "In the 2008 column model there is no heat of adsorption, no thermal
domain, no entropy as a state, and no learning. The 2009 general theory covers heat as well as
mass transfer, but no fetched abstract applies it to a non-isothermal adsorption column."
Corrected (table): "the isothermal port-based column model of Baaiu et al. 2008, and the
structure-preserving discretisation and reduction applied to adsorption (Peng et al. 2007;
Baaiu et al. 2009, J. Process Control)."

### 25. Baaiu et al., J. Process Control 2009 (CANNOT VERIFY)

Crossref, OpenAlex and HAL (hal-01625132) serve the title, authors and venue (19, 394-404,
March 2009) but no abstract. The title, "Structure-preserving infinite dimensional model
reduction: Application to adsorption processes", supports only the way LR §2 and F §7 cite it.
Nothing about isothermality can be checked. The existing "full texts were not reachable"
caveat already covers this, but the sentence in row 23 must not extend "no thermal domain" to it.

### 29. Schwamberger and Schmidt 2013 (PARTLY)

Ours (LR §2): "Schwamberger and Schmidt (Ind. Eng. Chem. Res. 2013) show that inconsistent
choices leave energy and entropy balance residuals."
Ours (F §3): "show that inconsistent heat-capacity choices leave energy and entropy balance
residuals."

Source (abstract): the adsorbed-phase heat capacities are computed "employing reversible
thermodynamic paths", and as a result "the residuals of the energy and entropy balances are
significantly smaller" (water and zeolite 13X). The paper reports smaller residuals from a
better estimate. It does not study "inconsistent choices" as such.

Corrected: "Schwamberger and Schmidt (Ind. Eng. Chem. Res. 52, 16958, 2013) show that
computing the adsorbed-phase heat capacity along reversible thermodynamic paths makes the
energy and entropy balance residuals significantly smaller (water and zeolite 13X)."

### 32b. "No model has entropy production as structure." (DOES NOT SUPPORT, absolute)

Ours (LR §2, end of the entropy-generation bullet). As written, the sentence covers every
model. Our own journal B sources contradict it. Ramirez et al. (P6) state "the first and second
principle are satisfied as a structural property", and P10 builds entropy production into
reactor models. The evidence (Li 2019, Guo 2019) supports the claim only for adsorption cycles.

Corrected: "No adsorption-cycle model found carries entropy production as structure. In
these studies it is computed after the fact."

### 35. FINN, Water Resour. Res. 2022 (PARTLY)

Ours (LR §3): "FINN for groundwater diffusion and sorption (Water Resour. Res. 2022),
isothermal, learning only the isotherm."

Source (abstract): FINN "learns components of the investigated system equations, such as
numerical stencils, model parameters, and arbitrary closure/constitutive relations". In the
diffusion-sorption application it learns the sorption isotherm. "Isothermal" is not stated, but
the clay diffusion-sorption problem has no energy equation, so that is acceptable. "Only the
isotherm" is too narrow for the method.

Corrected: "FINN for groundwater diffusion and sorption (Praditia et al., Water Resour. Res.
58, e2022WR033149, 2022): a finite-volume network with no energy equation that, in this
application, learns the sorption isotherm."

### 41. "Bilal et al. 2026" (CANNOT VERIFY as cited)

Ours (LR §3): "Bilal et al. 2026: a non-isothermal chromatography surrogate." The citation has
no identifier and no entry in journals A to D or the structure journal, which breaks the
review's own citation gate. A live Crossref search found two 2026 papers by Bilal, Haq and
Asif. Both fit the description, as KPI surrogates (scalar outputs) of a non-isothermal general
rate model:
- ACS Omega 11 (2026) 48428-48449, doi:10.1021/acsomega.6c04740 (XGBoost and other ensembles,
  1375 simulations; a correction followed, doi:10.1021/acsomega.6c11117);
- Ind. Eng. Chem. Res. 65 (2026) 13942-13958, doi:10.1021/acs.iecr.6c02282 (a deep-learning
  ensemble with "physics-derived dimensionless ratio features" as inputs; no structural
  constraint).

Corrected: "Bilal, Haq and Asif 2026: scalar-output surrogates of a non-isothermal
chromatography general rate model, with physics used only for feature construction (Ind. Eng.
Chem. Res. 65, 13942, doi:10.1021/acs.iecr.6c02282; ACS Omega 11, 48428,
doi:10.1021/acsomega.6c04740)." Log both in journal D first.

### 43. Glover, Papathanasiou, Pini (PARTLY)

Ours (LR §3): "a differentiable mechanistic VPSA model that reaches the cyclic steady state
about 20 times faster". Ours (LR §7): "already reaches the cyclic steady state 20 times faster."

Source (abstract): Newton iteration cuts the iterations and the time to cyclic steady state
"by a factor of 20 relative to a representative MATLAB implementation". The number is right,
but the baseline is missing. Crossref dates the issue (volume 216) January 2027, while the
article was online in September 2026.

Corrected: "... reaches the cyclic steady state about 20 times faster than a representative
MATLAB implementation (Glover, Papathanasiou, Pini, Comput. Chem. Eng. 216, 109904,
doi:10.1016/j.compchemeng.2026.109904)." The same clause goes in §7.

### 44. diff-soup (PARTLY)

Ours (LR §3): "diff-soup (Degnan-Morgenstern and Bazant, arXiv:2610.37333) puts a neural
closure inside a stiff packed-bed solver on GPU."

Source (abstract): it trains "a neural field for the catalyst activity of a three-dimensional
packed bed reactor". That is a learned spatial coefficient field in a reactor, not a closure
law in an adsorption column.

Corrected: "diff-soup (Degnan-Morgenstern and Bazant, arXiv:2610.37333) trains a neural field
for the catalyst activity inside a stiff, differentiable 3-D packed-bed reactor solver on GPU."

### 52. ProbConserv (PARTLY)

Ours (LR §4): "Hard conservation can break positivity. A global conservation projection created
negative densities in ProbConserv (arXiv:2302.11002)."

Source (v4 HTML, GPME results): "Since HardC-ANP shifts every point equally, it induces a
negative bias in the zero (degeneracy) region". HardC is a baseline that the ProbConserv paper
compares against. ProbConserv itself does not do this. VoR: Physica D 457 (2024) 133952.

Corrected: "Hard conservation can break positivity. In the ProbConserv paper, a baseline that
enforces conservation by shifting every point equally produced a negative bias in the
zero-density region (Hansen et al., Physica D 457, 133952, 2024)."

### 55. "Reporting is often best-of-N runs (journal C, C31)" (DOES NOT SUPPORT)

C31 is PDEBench. Its abstract presents a benchmark suite with baselines and new metrics and
says nothing about best-of-N reporting. Journal C's own synthesis (line 361) attributes
best-of-N to C17 (tLaSDI) and says C31 shows that standard benchmarks do not test
thermodynamic structure. One paper does not license "often".

Corrected: "Headline numbers can come from the best of several runs: tLaSDI's come from the run
with the smallest training loss of 10 (C17). The standard benchmark suite does not test
thermodynamic structure at all (PDEBench, C31)."

### 57. Hanikel et al. 2019 (CANNOT VERIFY in part)

Ours (PR, Material): MOF-303-like isotherm with "q_max 24.5 mol kg⁻¹; step at RH 0.12 at 303
K; ΔH = −52 kJ mol⁻¹; f_H = 0.17", sourced to Paper 1's `research/water_mof_parameters.md`
(Hanikel et al. 2019).

Source (PMC full text): "an inflection point at ∼12% RH" (supported). The main text gives
heats of adsorption of "50–54 kJ mol–1" for MOF-303, Al-fumarate and SAPO-34 together, and
points to SI Section S3.2. The single value 52 for MOF-303 is in the SI, which was not read.
q_max 24.5 mol kg⁻¹ (uptake at RH 0.9) and f_H 0.17 are figure readings, as Paper 1's file
says. They cannot be checked against text. The "at 303 K" temperature also comes from the
figure caption chain in Paper 1's file.

Fix: keep the source line, and say which values are text and which are figure readings: "step
at about 0.12 RH (text); ΔH −52 kJ mol⁻¹ (SI S3.2; main text 50 to 54 for three sorbents);
q_max and f_H read from Fig. 1a". Before the freeze, open SI S3.2 to confirm 52.

### 58. Döppel et al. 2026 and Myers 2002 (CANNOT VERIFY, already disclosed)

LR §6 already labels both SEEN-ONLY and lists them as open. No support claim rests on them. A
Crossref query for Döppel failed during this audit, so its metadata is still unchecked.

## 3. Numbers attributed to sources

| number in our docs | source | live value | verdict |
|---|---|---|---|
| "61 of 72 seed-level comparisons" | 2610.05570 v1 and v2 abstracts | "61 of 72" (v1), "61/72" (v2) | correct |
| "7×10⁻⁹, against 25 to 67" | 2606.01366 abstract | 7.35e-9 against 42.3, 25.3, 67.2, 27.1 | correct |
| "60 of 64 ... only 19" | 2606.01366 abstract | 60/64 and 19/64 | correct |
| "5 to 8 times lower" | 2403.05848 v2 Sec. 5.1 | 3.3 to 6.9 times | wrong, fix (row 7) |
| "2.47×10⁻³ against 1.70×10⁻²" | 2403.05848 v2 Sec. 5.1 | 2.47e-3 and 1.70e-2 (Vanilla-FNN) | correct |
| "best of 10 runs" | 2403.05848 v2 | smallest training loss among 10 | correct |
| "63.27 % ... 53.72 % with heat recovery" | Li et al. 2019 abstract | 63.27 % to 53.72 % with internal heat recovery | correct |
| "about 20 times faster" | Glover et al. abstract | factor of 20 relative to MATLAB | correct number, baseline missing (row 43) |
| "about 120,000 cycle-resolved simulations" | An et al. abstract | "approximately 120,000 cycle-resolved simulations" | correct |
| "1.1 million simulations" | Zenodo 22556794 description | "The 1.1 million training simulations" | correct |
| "10 to 10,000 trajectories" | 2012.02334 v6 Sec. 4 | "as low as 10 ... as high as 10,000" | correct |
| "v9, 1 November 2026" | arXiv API, 2606.08343 | updated 2026-11-01, v9 | correct |
| "4 October 2026" (Poisson-GENERIC v1) | arXiv API | published 2026-10-04 | correct |
| "tel-05363400, 2025-10-10" | HAL API | producedDate 2025-10-10 | correct |
| "Comput. Chem. Eng. 199, 109156 (2025)" | Crossref | 199, 109156, Aug 2025 | correct |
| MOF-303 0.12 / 52 / 24.5 / 0.17 | Hanikel 2019 | 0.12 in text; 52 in SI; others figure readings | partly checkable (row 57) |

## 4. Absolute words in sentences about the literature

| sentence (abbreviated) | where | licensed? |
|---|---|---|
| heading "its latent versions all learn the entropy" | LR §1 | No as written. The body says "found" and names the Latent Twins exception. Change to "the latent versions found learn the entropy". |
| "In every latent GENERIC or metriplectic model found ..." | LR §1 | Yes ("found"; the exception is named). |
| "None of these is a latent model with E and S fixed ..." | LR §1 | Yes (T5, T13, T17 checked: none is a latent autoencoder model). |
| "No fetched paper shows ..." | LR §1 | Yes (qualified). |
| heading "Structured models of adsorption columns exist, but only isothermal ones" and F §7 "exist only for the isothermal case" | LR §2, F §7 | Only from abstracts, and one of the three abstracts is missing (row 25). Write "the only ones found are isothermal (from abstracts; full texts unread)". |
| "There is no heat of adsorption, no thermal domain ..." | LR §2 | No for Baaiu 2009 MCMDS (row 23). |
| "No model has entropy production as structure." | LR §2 | No (row 32b). |
| "No learned sorption model reports zero unphysical outputs over many cycles." | LR §3 | Needs "found": "No learned sorption model found reports ...". |
| "No learned adsorption ... model ... was found" | LR §3 | Yes (search terms listed). |
| "every sorption surrogate found ... none tested this way" | LR §5 table | Yes ("found"). |
| "No published work, as of 2026-11-03 within the sources searched ..." | LR §5 | Yes (qualified by date and sources). |
| "Reporting is often best-of-N runs" | LR §4 | No (row 55). |
| "is standard non-equilibrium thermodynamics" | LR §2 | Acceptable. It rests on one source, and the full text derives the flux from the chemical-potential difference. |
| "injects energy on every reversible-irreversible problem" | LR §1 | A quote from the source (v1), correctly attributed. |

## 5. Required fixes (list)

1. LR §1: tLaSDI ratio "5 to 8" becomes "about 3 to 7". The ablation caveat applies to one of
   two examples. Cite CMAME 429 (2024) 117144.
2. LR §1: Beckers and Colombo is not conservative (shallow water with linear friction). Reword
   and merge the duplicated "The nearest" sentences.
3. LR §1: Bermejo et al. is described as port-metriplectic.
4. LR §1 and F §4.1: pin "friction class provably excludes transport" to 2610.05570v1.
5. LR §1 heading: add "found".
6. LR §2: "No model has entropy production as structure" becomes "No adsorption-cycle model
   found ...".
7. LR §2 and §5 table: Baaiu 2009 MCMDS covers heat transfer and is not an isothermal column
   model. Limit "no thermal domain" to the 2008 paper.
8. LR §2 heading and F §7: "only isothermal" becomes "the only ones found are isothermal (from
   abstracts)".
9. LR §2: P6/P8/P10 sentence (drop "reactor" and "on purpose"). P15/P28/P29 sentence: only P28
   supports "breaks passivity".
10. LR §2, §5, §7 and F §3: Schwamberger and Schmidt sentence restated as "reversible-path
    heat capacities give significantly smaller energy and entropy residuals".
11. LR §3: FINN is not "learning only the isotherm". Add authors and DOI.
12. LR §3: Bilal et al. 2026 gets a DOI (IECR 65, 13942 and/or ACS Omega 11, 48428) and a
    journal D entry.
13. LR §3 and §7: Glover "20 times faster than a representative MATLAB implementation".
14. LR §3: diff-soup trains "a neural field for the catalyst activity", not "a neural closure".
15. LR §3: "No learned sorption model found reports zero unphysical outputs ...".
16. LR §4: ProbConserv negativity belongs to the HardC baseline. Cite Physica D 457 (2024)
    133952.
17. LR §4: "Reporting is often best-of-N runs (C31)" becomes the tLaSDI (C17) statement plus
    the PDEBench point.
18. F §4.1: cite Ramirez, Le Gorrec, Maschke as Chem. Eng. Sci. 248 (2022) 117107, not the
    DOI that the arXiv record carries (that DOI is a different Entropy 2022 overview).
19. PR, Material: mark which MOF-303 values are text, SI or figure readings. Read SI S3.2 for
    the value 52 before the freeze.
20. Versions of record to add wherever the arXiv ID is cited: GFINNs (Phil. Trans. A 2022),
    port-metriplectic NNs (Comput. Mech. 2023), Hernández et al. (CMAME 2021), parametric
    tLaSDI (TMLR 2026), Zakwan et al. (IFAC-PapersOnLine 2023), Santamaria-Holek et al. (J.
    Non-Equilib. Thermodyn. 2012), Lastrucci et al. (ESCAPE 35, 2025), Urdeitx et al. (J.
    Non-Newton. Fluid Mech. 2026), Nooraiepour et al. (Energy and AI 2026). Before submission,
    check the conference versions of Gruber et al. (2405.16305), Zhong et al. (2012.02334) and
    Lanthaler et al. (2210.01074); they could not be confirmed live today.
21. Minor metadata: Green Chem. Eng. volume 6 is the 2025 issue. Glover et al. (vol. 216) and
    Chung et al. (vol. 463) are in January 2027 issues, online in 2026. Either convention is
    acceptable if it is used consistently.
