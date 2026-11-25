Paper 3B's central question is still open. I found no prior or 2025–2026 work that tests, pre-registered, whether thermodynamic structure pays off against unstructured, information-matched, projected, learned-potential and coarse-solver baselines, on any open system. But three 2026 preprints use the same kind of matched-control design on other questions. Those, plus the risk that the plain solver wins outright, are the real threats.

The repo already has a full review (`docs/lit_review_3B.md`, last updated 2026-11-03). This one adds to it: it re-checks the DOIs that review leans on most, fills gaps a referee would notice, and sweeps everything up to 2026-11-16.

**How things were checked.** All checks were run on 2026-11-16. "DOI ok" means the DOI resolved on Crossref with matching title and first author. A one-line claim rests on an abstract read live from arXiv, OpenAlex or the publisher's page. Where I only saw a title or record, the entry says so.

- **No DOI on Crossref:** the NeurIPS/ICLR-type papers (Greydanus, Cranmer, Lee–Trask–Stinis, Botev, Gruver). They are cited by arXiv ID.
- **Repo DOI that only looked wrong:** `10.1002/aic.70045` (Frandsen 2025) failed once and resolved on retry, so it is correct.
- **No DOI on any page I opened:** the Zakwan 2023 entry in the repo has none. Crossref gives 10.1016/j.ifacol.2023.10.079 ("Physically Consistent Neural ODEs for Learning Multi-Physics Systems", IFAC-PapersOnLine 2023). Check that this is the paper the repo means (the repo cites 56(2), 5855).
- **Acronyms used below:** E and S are energy and entropy; U, U+F and C are 3B's unstructured, unstructured-plus-inputs and coarse-solver arms; PINN is a network with physics as a loss term; PSA is pressure-swing adsorption.

## 1. Key works

### A. Structure-preserving and metriplectic learning (GENERIC-type)

| work | DOI / ID | what it shows | how 3B differs |
|---|---|---|---|
| Hernández, Badías, González, Chinesta, Cueto, "Structure-preserving neural networks", JCP 2021 | 10.1016/j.jcp.2020.109950 (ok; abstract not opened) | the first GENERIC-informed network, built on a learned metriplectic bracket | full-state, closed systems, E and S learned |
| Hernández et al., thermodynamics-aware reduced models, CMAME 2021 | 10.1016/j.cma.2021.113763 (ok) | sparse autoencoder plus GENERIC latent dynamics | latent E and S have no physical meaning; no ports; no matched controls |
| Hernández et al., Thermodynamics-Informed GNNs, IEEE TAI 2024 | 10.1109/TAI.2022.3179681 (ok) | GENERIC built into a graph network | full-field, E and S learned |
| Hernández et al., Port-metriplectic NNs, Comput. Mech. 2023 | 10.1007/s00466-023-02296-w (ok) | open systems learned part by part through ports | no latent physical potentials; no sorption |
| Cueto and Chinesta, "Thermodynamics of Learning Physical Phenomena", ACME 2023 | 10.1007/s11831-023-09954-5 (ok) | review of the Zaragoza line of work | background citation |
| Bermejo et al., MeshGraphNets informed locally by thermodynamics, AMSES 2025 | 10.1186/s40323-025-00311-8 (ok; the repo had no DOI) | nodal port-metriplectic graph network for flow past objects | full-field, E and S learned |
| Zhang, Shin, Karniadakis, GFINNs, Phil. Trans. A 2022 | 10.1098/rsta.2021.0207 (ok) | degeneracy conditions satisfied by construction | the template for 3B's dynamics; closed systems |
| Lee, Trask, Stinis, NeurIPS 2021 | arXiv:2106.12619 (abstract read) | learned metriplectic brackets; a penalty does not guarantee the structure | supports 3B's P arm being weaker than S |
| Gruber, Lee, Trask, bracket-based GNNs, NeurIPS 2023 | 10.52202/075280-1670 (ok) | reversible and irreversible brackets in deep graph networks | the same "which structure" question, on graphs, without information controls |
| Gruber et al., efficiently parameterised metriplectic systems | arXiv:2405.16305 (abstract read) | parameterisation that scales quadratically; includes a case with unknown entropic variables | E and S learned |
| Park et al., tLaSDI, CMAME 2024 | 10.1016/j.cma.2024.117144 (ok) | autoencoder plus GFINN latent model; better extrapolation than baselines | **closest architectural template**; latent E and S are learned |
| He et al., parametric tLaSDI, TMLR 2026 | arXiv:2506.08475 (journal ref on arXiv) | parametric GFINNs with active learning | same; constraints hold only on latent variables |
| tLaSDI applied to fracture | arXiv:2610.20614 (abstract read) | a 2026 application | E and S still learned |
| Chung, Bu, Verma, Physics-conforming Latent Twins, CMAME 2027 | 10.1016/j.cma.2026.119351 (ok) | latent flow preserves or dissipates *prescribed* quantities | not a thermodynamic E–S pair; no degeneracy; no ports |
| Poisson-GENERIC operators; GENERIC-FNO | arXiv:2610.05570 (v3); 2606.08343 | metriplectic structure in function space | closed systems; the repo already handles the v1/v2 wording |

### B. Hamiltonian, Lagrangian, port-Hamiltonian and energy-dissipative networks

| work | DOI / ID | what it shows | how 3B differs |
|---|---|---|---|
| Greydanus et al., HNN 2019 | arXiv:1906.01563 | learned Hamiltonian conserves energy exactly | conservative systems only |
| Cranmer et al., LNN 2020 | arXiv:2003.04630 | Lagrangian form as the inductive bias | conservative systems only |
| Desai et al., port-Hamiltonian NNs, PRE 2021 | 10.1103/PhysRevE.104.034312 (ok) | time-dependent forcing and damping | no entropy |
| Yu, Tian, E, OnsagerNet, PR Fluids 2021 | 10.1103/PhysRevFluids.6.114402 (ok) | generalised Onsager structure for stable learned dynamics | energy-dissipative; no second law with physical S |
| Huang et al., VONNs, JMPS 2022 | 10.1016/j.jmps.2022.104856 (ok) | variational Onsager networks | constitutive setting |
| Zakwan et al., IFAC-PapersOnLine 2023 | 10.1016/j.ifacol.2023.10.079 (see the check note above) | irreversible port-Hamiltonian neural ODE; structure from physics, parameters learned | lumped; not latent |
| Model reduction of port-Hamiltonian systems via NNs | arXiv:2608.30788 (abstract read, 2026-08) | Galerkin projection, then a network learns the reduced J and R matrices | intrusive; no entropy |
| Comparing representations for port-Hamiltonian identification | arXiv:2611.05031 (abstract read, 2026-11-03) | networks against analytic forms against sparse regression | mechanical and electrical toys; no structured-vs-unstructured question |

### C. Projection-based structure-preserving reduction (non-learned)

All four DOIs are ok: Öttinger 2015, 10.1103/PhysRevE.91.032147; Gruber et al. 2023, 10.1016/j.cma.2022.115709; Chan 2020, 10.1016/j.jcp.2020.109789; Klein et al. 2025, 10.1016/j.jcp.2025.113817. They keep the physical E and S but do not learn the dynamics. The repo already handles them correctly.

### D. Learned closures inside consistent coarse solvers

None of these tests entropy structure or open systems with ports.

| work | DOI / ID | what it shows |
|---|---|---|
| Bar-Sinai et al., PNAS 2019 | 10.1073/pnas.1814058116 (ok) | learned discretisations on coarse grids |
| Kochkov et al., PNAS 2021 | 10.1073/pnas.2101784118 (ok) | learned corrections inside a CFD solver |
| List, Chen, Thuerey, JFM 2022 | 10.1017/jfm.2022.738 (ok) | training a closure through a differentiable solver |
| van Gastelen, Edeling, Sanderse, JCP 2024 | 10.1016/j.jcp.2024.113003 (ok; OpenAlex abstract) | energy-conserving learned closure, "discretize first, filter next"; stability from subgrid-energy conservation. **The structure-preserving closure analogue in turbulence** |
| Melchers et al., CAMWA 2023 | 10.1016/j.camwa.2023.04.030 (ok) | comparison of neural closure models |
| Sanderse et al., review, FoDS 2025 | 10.3934/fods.2024043 (ok) | closure-model review |
| Praditia et al. (FINN), WRR 2022 | 10.1029/2022WR033149 (ok) | finite-volume network for diffusion–sorption; learns the isotherm |
| Neural entropy-stable flux networks | arXiv:2507.01795 (abstract read) | learn a flux and an entropy jointly |
| SymCLaw | arXiv:2601.21080 (abstract read) | conservation, entropy stability and hyperbolicity by design |

### E. Neural surrogates of adsorption columns and packed-bed thermal stores, 2019–2026

The 2019–2022 PSA network papers below are absent from the repo review. A referee in the adsorption community will expect them.

| work | DOI / ID | what it shows |
|---|---|---|
| Leperi et al., I&ECR 2019 | 10.1021/acs.iecr.9b02383 (ok; OpenAlex abstract) | networks trained on full PSA model runs as reduced models |
| Subraveti et al., I&ECR 2019 | 10.1021/acs.iecr.9b04173 (ok; title only) | multi-objective PSA optimisation with machine learning |
| Pai, Subraveti, Rajendran, MAPLE, I&ECR 2020 | 10.1021/acs.iecr.0c02339 (ok; abstract) | adsorbent-agnostic network that predicts performance at the cyclic steady state |
| Subraveti et al., PANACHE, I&ECR 2022 | 10.1021/acs.iecr.1c04731 (ok; abstract) | physics-constrained-loss network that predicts full column dynamics per cycle step. **The closest "physics-in-the-model" adsorption-column precedent; physics as a loss only** |
| Wu et al., PSA PINN, Green Chem. Eng. 2025 | 10.1016/j.gce.2024.08.004 (ok) | physics as a loss |
| Galanti et al., Processes 2025 | 10.3390/pr13092824 (ok) | PINNs for direct air capture |
| Frandsen et al., hybrid chromatography models, AIChE J. 2025 | 10.1002/aic.70045 (ok) | screening of hybrid models |
| Bilal et al., I&ECR 2026 | 10.1021/acs.iecr.6c02282 (ok) | scalar-output surrogate of a non-isothermal chromatography rate model |
| An et al., Front. Environ. Eng. 2026 | 10.3389/fenve.2026.1881554 (ok) | cycle-level direct-air-capture surrogate |
| Glover, Papathanasiou, Pini, Comput. Chem. Eng. 2027 | 10.1016/j.compchemeng.2026.109904 (ok) | differentiable mechanistic VPSA model; the strongest non-learned competitor, so it supports 3B's arm C |
| Ceccanti et al., DeepONet for cyclic TVSA | arXiv:2601.09491 | isothermal |
| Mousa et al., Comput. Chem. Eng. 2025 | 10.1016/j.compchemeng.2025.109156 (ok) | already shown in the repo to involve no adsorption |
| Sene, Lin, Kim, Radaideh, engrXiv 2025 | 10.31224/4630 (ok; page read) | plain feed-forward surrogate of INL's thermocline tank |
| Ananth et al., Front. Energy Effic. 2026 | 10.3389/fenef.2026.1845713 (ok; full text read) | soft-loss PINN for a sand packed-bed store; no data-only baseline |

No structured-by-construction surrogate of a thermocline or packed-bed store was found, so 3B's system B is also unclaimed.

### F. Does structure help? Comparisons and negative results

| work | DOI / ID | what it shows |
|---|---|---|
| Botev et al. 2021 | arXiv:2111.05458 | across 17 datasets, physics-prior latent models "fail to significantly improve upon standard techniques"; continuous, time-reversible dynamics help every class |
| Gruver et al. 2022 | arXiv:2202.04836 | HNN gains come from modelling acceleration directly, not from conservation |
| Zhong et al., energy-conserving NN benchmark | arXiv:2012.02334 | benchmark of energy-conserving networks |
| McGreivy and Hakim, Nat. Mach. Intell. 2024 | 10.1038/s42256-024-00897-5 (ok; arXiv abstract) | 79 % of ML-for-PDE papers claiming to beat a numerical method compare to a weak baseline; reporting bias is widespread. **This is the strongest external justification for 3B's pre-registration and arm C** |
| Conservative structure in a 1-D drift-diffusion-Poisson benchmark | arXiv:2606.01366 | structure wins rollout error in 60 of 64 configurations, but the classical core alone reaches round-off |
| Urdeitx et al., JNNFM 2026 | 10.1016/j.jnnfm.2026.105601 (ok) | with hidden variables, a Transformer beats a structure-preserving network |
| Hansen et al., ProbConserv, Physica D 2024 | 10.1016/j.physd.2023.133952 (ok) | hard conservation by uniform shifting breaks positivity |
| Janssen et al., A&A 2026 | 10.1051/0004-6361/202658937 (ok) | in astrochemistry, fully connected models beat latent-evolution ones on accuracy |
| Decision-aware evaluation of physics-informed surrogates | arXiv:2606.07146 | low curve error often fails to pick the best design; supports 3B's design-decision metric |

## 2. Synthesis

### (a) Closest prior work, and is the question answered?

The closest work depends on the angle:
- **Architecture:** tLaSDI (Park 2024; He 2026) and Hernández et al. 2021.
- **Physics-in-the-model adsorption column:** PANACHE (Subraveti 2022), with physics only as a loss.
- **Structure for adsorption columns:** the Lyon port-based models, which are isothermal and not learned.
- **Experimental logic:** three 2026 preprints. arXiv:2606.01366 compares structure with projection and rollout training, with a solver core as a baseline. arXiv:2609.19674 removes one structural component at a time with matched controls and finds a double dissociation. arXiv:2609.34012 asks whether physics gains come from the information or from regularisation, with matched baselines and one pre-registered hypothesis (falsified); it is about PINN residuals, not architecture.

**Not answered.** None of them uses thermodynamic (entropy-producing, metriplectic) structure. None has an information-matched control (U+F), a fixed-versus-learned potential control, or an open system with ports. None tests on two physical systems under a pre-registration.

### (b) Defensible novelty claims (suggested wording)

1. "To our knowledge, the first pre-registered test of whether thermodynamic (metriplectic) structure improves learned latent models of open, advection-dominated dissipative systems. It uses controls that separate constraint, information and projection, and compares fixed with learned potentials."
2. "The first learned latent model of a non-isothermal adsorption column whose energy and entropy are fixed from the sorbent's and the gas's thermodynamics and evaluated on the decoded fields, with inlet, outlet and wall as ports." Qualify it: projection-based reduced models (Chan 2020; Klein 2025) already keep the physical entropy, so the claim rests on "learned" and "open column".
3. "A port-metriplectic formulation of the non-isothermal column, extending the isothermal, free-energy-based port models of Baaiu et al. Paper 1's model is its exact limit when the water's heat capacity is set to zero." Present this as supporting material, not a headline.
4. "Evaluation against a coarse, consistent numerical solver at equal cost, plus a decision-level metric." Frame it as answering McGreivy and Hakim 2024, not as new methodology.
5. Optionally: "the first structure-preserving surrogate study of a packed-bed thermocline store". Only "not found" supports it, so word it as "to our knowledge".

### (c) Papers a referee would expect cited

- Greydanus, Cranmer, Hernández 2021 (JCP), Lee–Trask–Stinis 2021, GFINN, tLaSDI, port-metriplectic NNs, OnsagerNet, Desai port-Hamiltonian NNs.
- The Öttinger, Chan and Gruber 2023 reduction papers.
- Kochkov 2021 and van Gastelen 2024.
- Botev 2021, Gruver 2022, McGreivy and Hakim 2024, arXiv:2606.01366.
- Leperi 2019, Pai 2020, Subraveti 2022, Glover 2027, Baaiu 2008 and 2009.
- Walton and LeVan 2005.
- For an MLST audience, also Botev and McGreivy prominently.

Missing from `docs/lit_review_3B.md` today: Hernández JCP 2021, TIGNN, Lee–Trask–Stinis, Gruber NeurIPS 2023, HNN/LNN, Desai, OnsagerNet, VONNs, all of the learned-closure block (D), Leperi/Subraveti/Pai/PANACHE, Botev, McGreivy, the thermocline surrogates, and the 2026 items in (d).

### (d) Recent work that could scoop or undercut the study

No direct scoop was found up to 2026-11-16. That rests on arXiv sweeps sorted by date (metriplectic; port-Hamiltonian; adsorption + surrogate; thermodynamics-informed; structure-preserving + latent; thermal energy storage + neural; entropy stable + learned) plus several web searches. Undercutting risks:
- **arXiv:2609.34012 (2026-09-27)** already frames "physics-specific information vs regularisation" and calls a hypothesis pre-registered. 3B cannot claim the information-vs-constraint question, or pre-registration in physics ML, as new in general. Its claims must be specific to thermodynamic structure, latent models and open systems. Cite it.
- **arXiv:2609.19674 (2026-09-17)** uses matched single-component ablations ("conservation buys stability"). Cite it as methodological kin.
- **arXiv:2606.01366**: the solver core alone reached round-off. If 3B's arm C dominates the same way, the headline becomes "use the solver". The pre-registration already allows that outcome; frame it in advance as a valid result.
- **arXiv:2603.02645 (2026-03)** compares thermodynamic frameworks, including metriplectic, under one shared architecture for constitutive models. It is nearby on "which structure", but not dynamics surrogates.
- **arXiv:2611.05031 (2026-11-03)** and **2608.30788** are new port-Hamiltonian learning comparisons. They are not threats, but should be cited if the related work discusses port-Hamiltonian models.

Nothing was written to the repos.
## Addendum 2026-11-25: Claude Science T11 refresh (window 2026-11-10 to 2026-11-24)

Source: hub `projects/ai-column-structure/T11_novelty_refresh.md` (arXiv API 28 queries / 463 records, Crossref 14 queries /
380 DOIs, 9 full texts). Verdict: no paper in the window separates constraint, information and projection in learned latent
models of open dissipative systems, and none pre-registers such a comparison. Claim (b)1 survives but is narrowed:
- arXiv:2610.05570 (Sulskis and Ravi, "Poisson-GENERIC Neural Operators", v5): closed periodic PDEs, a compute-matched
  unconstrained arm, fixed vs learned energy, a projected-GENERIC comparator; structure loses at equal wall-clock but secures the
  energy balance. Cite for "fixed vs learned potentials" and as a prior instance of H1/U-long's question.
- arXiv:2611.27149 (Christou, Kar, Xhelollari et al., "Where Structure Enters a World Model", 2026-11-20): information- and
  capacity-matched controls separating structure from supervision (closed mechanics). Methodological kin, with 2609.19674 and
  2609.34012.
- arXiv:2611.14574 (Cao and Song, "The norm, not the labels", 2026-11-09): T11 reports hash-gated pre-registered FE-surrogate
  comparisons. **To verify in the full text before citing it for pre-registration** (the title does not say so).
- Lower: 2509.19526v2 (rollout energy projection; cite where U+proj is motivated); 2611.22549 (terminal projection can hide a
  collapsed distribution: a possible mechanism if U+proj differs from S); 2611.27393 (non-learned oblique-projection MOR, block C).

IDs, titles and dates of the three main papers checked on the live arXiv API on 2026-11-25. Suggested wording of (b)1 (T11):
"To our knowledge, the first pre-registered test of whether thermodynamic (metriplectic) structure improves learned latent
models of open, advection-dominated dissipative systems, with arms that separate the constraint from the physical information
(U+F) and from projection at rollout (U+proj, U+F+proj), alongside fixed-versus-learned potentials and compute- and
solver-matched baselines. Matched-control designs exist for closed systems [2610.05570; 2611.27149; 2609.19674; 2609.34012],
and pre-registration of surrogate comparisons exists in computational mechanics [2611.14574]."

Not covered by T11 (to do right before submission): OpenReview (ICLR 2027 submissions), ChemRxiv/engrXiv, citation graphs
(OpenAlex/Semantic Scholar), publisher tables of contents.
