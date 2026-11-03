# Literature journal D: update sweep for Paper 3B (2026-10-01 to 2026-11-03)

Session date: 2026-11-03. Updates `docs/lit_review_3B.md` (completed 2026-10-13); target is the gap in its section 5 and the risks in its section 6.
Rule (as in journals A to C): a source counts (N#) only if its live record WITH an abstract (or full text) was fetched this session. Items seen only in a listing title are SEEN-ONLY leads.
Tools: curl against export.arxiv.org API (abstract is in the Atom record), api.openalex.org (abstract reconstructed from abstract_inverted_index), api.crossref.org. No mailto or email parameter was sent anywhere. No archived copies.
Window: anything first posted/indexed 2026-10-01 or later, plus anything older that journals A to C missed. "Already known" = its arXiv ID or DOI appears in journals A to C or the review.

## Search log (query -> hits; what was scanned)

arXiv API (export.arxiv.org/api/query, sortBy=submittedDate desc; "TOTAL" = API totalResults; first 15 to 50 scanned):
- A1 `all:metriplectic` -> 86; in window only 2610.05570 (known). Nothing new on learning.
- A2 `abs:"GENERIC formalism"` / `ti:GENERIC` + neural/learning -> 4,089 / 141,826 (arXiv matches the common word "generic"; unusable, replaced by OpenAlex OA3 and the citer sweeps).
- A3 `all:"port-Hamiltonian" AND (learning OR neural)` -> 89; in window: 2610.33679, 2610.22575, 2610.05755, 2610.01356 (none on sorption or thermal transport).
- A4 `abs:"thermodynamics-informed" OR "thermodynamically consistent neural" OR "thermodynamics-aware"` -> 113; in window only 2610.36348 (message passing, irrelevant) and 2610.05570.
- A5 `abs:"port-metriplectic" OR "irreversible port-Hamiltonian"` -> 10; nothing after 2603.11260/2603.11265 (both known).
- A6 `abs:adsorption AND (neural OR surrogate OR "machine learning") AND (column OR bed OR breakthrough OR cycle)` -> 7; none relevant in window.
- A7 `abs:adsorption AND (thermodynamic OR entropy) AND ("port-Hamiltonian" OR GENERIC OR metriplectic OR Onsager)` -> 102; no learned structured column model (2609.26332 is a non-ML phase-field model of surfactant adsorption).
- A8 `latent AND (metriplectic OR "thermodynamics-informed" OR "thermodynamically consistent") AND (autoencoder OR reduced-order)` -> 4; one not in journals A-C: 2610.20614 (N2).
- A9 `LaSDI` -> 13; no new thermodynamic LaSDI beyond N2.
- A10-A15 author sweeps: Cueto AND Chinesta -> 18 (nothing after 2512.16305); (Gruber OR Trask) + structure-preserving terms -> 202 (name noise; nothing new beyond known 2606.24106, 2606.11650, 2605.08436); Le_Gorrec -> 23 (nothing on adsorption; PH-KAN 2606.14708 seen-only); Sulskis/Ravi -> 5 (2606.24851 off-topic; GENERIC-FNO revised to v9 on 2026-11-01, N6); Shin/Choi -> 20 (nothing new thermodynamic); Maschke/Couenne/Ramirez -> 34 (only revisions of control papers).
- A16 `Onsager AND (neural OR "machine learning") AND (dynamics OR "reduced order")` -> 18; in window 2610.12745, 2610.06099 (off-topic).
- A17 `("structure-preserving" OR "energy-stable" OR "entropy-stable") AND (inflow OR outflow OR "open system(s)") AND (learning OR neural OR "reduced-order")` -> **0**.
- A18 `("temperature/pressure/vacuum swing" OR TSA OR PSA OR VSA) AND adsorption` -> 22; nothing in window (2606.21523 is the known Glover et al. paper).
- A19 `("water harvesting" OR desiccant OR "adsorption heat pump/chiller" OR "thermal energy storage") AND (neural OR surrogate OR ML)` -> 12; nothing relevant.
- A20 `(chromatography OR "packed bed" OR "fixed bed") AND (neural OR surrogate OR "reduced-order")` -> 32; in window 2610.33269 (N12), 2610.37333 (N13).
- A21 `("heat exchanger" OR "porous media") AND ("port-Hamiltonian" OR metriplectic OR GENERIC OR "thermodynamics-informed")` -> 772 (GENERIC noise); first 30 scanned, nothing structured.
- A22 `"structure-preserving" AND (advection OR convection OR transport) AND (latent OR autoencoder OR "reduced-order")` -> 18; one missed earlier: 2607.10965 (N4).
- A23 `(metriplectic OR GENERIC OR "port-Hamiltonian") AND ("known/prescribed/given energy" OR "known/physical entropy" OR "free energy") AND (latent OR autoencoder OR "reduced-order")` -> 69; nothing relevant in first 30.
- A24 `("port-Hamiltonian" OR metriplectic) AND ("boundary ports" OR "boundary control" OR inflow OR outflow) AND (learning OR neural OR "model reduction")` -> 1: 2604.04266 (N11).
- A25 `(adsorption OR sorption OR adsorbent) AND ("neural ODE" OR "neural operator" OR "reduced-order" OR autoencoder OR latent)` -> 11; nothing on process models.
- A26 `ti:"Latent Twins"` -> 6; 2606.15053 (N3) relevant.

OpenAlex citer sweeps (filter=cites:W..., no key, no mailto):
- tLaSDI CMAME (W4400182292) -> 15 citers; new relative to journal A: Physics-conforming Latent Twins (N3), CEJ thermophysical-property paper W7214049840 (SEEN-ONLY); the rest known (N-GENNs, WGFINNs, ICOPS state-variables paper) or off-topic. tLaSDI report records W4392736944 -> 1, W4403241567 -> 1 (off-topic).
- 2506.08475 (W4417256849) -> 0. Poisson-GENERIC (W7220648815) -> 0. GENERIC-FNO (W7164052780) -> 0. Ouardi et al. 2603.11260 (W7135224826) -> 0.
- Port-metriplectic NN, Comput. Mech. (W4360976744) -> 19; only new one in window is an Optics Express symplectic neural operator (W7220918966, off-topic).
- Baaiu et al. 2008 (W2143798946) -> 9 citers, all 2008 to 2019: Baaiu 2009 JPC, Duindam et al. book, Jallut/Couenne/Maschke CRC chapter, Calchand et al. 2013, theses Chera 2009 (N19), Hoang 2009 (N18), Vincent 2019 (N17). **No citer extends the column to non-isothermal operation or to learning.** Baaiu 2009 JPC (W2031548184) citers since 2026-07 -> 0.
- GFINNs (W3198730848) citers since 2026-07 -> 6 (N8 new in window; others known or off-topic). TA-ROM 2007.03758 (W3040708241) citers since 2026-07 -> 7 (N23 in window; none on sorption).

OpenAlex title/abstract searches (from 2026-09-15 unless noted):
- OA1 `metriplectic` -> 35, nearly all Zenodo self-deposits by one author on Navier-Stokes; none relevant.
- OA2 `(GENERIC OR metriplectic OR port-Hamiltonian OR Onsager OR thermodynamics-informed) AND (adsorption OR sorption OR chromatography OR packed bed OR fixed bed OR heat exchanger)` from 2025-01-01 -> 362; top 20 are noise (boolean not honoured); nothing structured.
- OA3 `"thermodynamics-informed" OR "thermodynamically consistent neural" OR "GENERIC formalism"` -> 13; new: 2610.23352 (N7), PoF viscoelastic (N8); the rest constitutive (rocks, soils, interfaces), none sorption.
- OA4 `adsorption neural network surrogate column` -> **0**. OA5 `adsorption breakthrough neural network` -> 4 (one PSA ML paper, SEEN-ONLY). OA6 `swing adsorption machine learning surrogate` -> 2 (both SEEN-ONLY, no abstract in OpenAlex or Crossref).
- OA7 `physics-informed adsorption` -> 18; relevant: ChemRxiv differentiable LC (N14), Water Research biochar PINN (SEEN-ONLY). OA8 `(water harvesting OR desiccant OR adsorption chiller OR adsorption heat pump) neural` -> 3, none sorption.
- OA9 `port-Hamiltonian` -> 50; learning ones are 2610.01356, 2610.22575 (N9, N10); 2610.38465, 2610.21790 (N21, N22) are non-ML.
- OA10 fulltext `"port-metriplectic"` from 2026-06-01 -> 3 (2609.04530 and 2607.20535 known; N20). OA11 `thermodynamics neural network packed bed` from 2026-06-01 -> **0**. OA12 `adsorption cycle neural surrogate` from 2026-08-01 -> 2 (N15, one SEEN-ONLY). OA13 `latent dynamics thermodynamic structure preserving` from 2026-08-01 -> 8, all Zenodo noise.

Crossref (query.bibliographic, filter from-pub-date:2026-09-01, 25 rows, top 12 by score scanned):
- CR1 `metriplectic neural network` -> 29,477 (relevance-ranked); only non-ML metriplectic papers (Morrison and Oh; Bressan et al.).
- CR2 `GENERIC formalism neural network thermodynamics` -> 30,849; WGFINNs (known), a tribology TINN, a rock TINN; nothing on sorption.
- CR3 `port-Hamiltonian adsorption column` -> 6,592; no adsorption item; ISO-pHNN (Physica D, SEEN-ONLY).
- CR4 `thermodynamically consistent non-isothermal adsorption column model` -> 69,463; no column model in the top 12.
- CR5 `neural network surrogate temperature swing adsorption` -> 46,802; MISA surrogate optimisation (CHERD), kNN TSA cycle-time forecasting (DCHE), PCM entropy surrogate (N16); the first two SEEN-ONLY.

Other: HAL API (api.archives-ouvertes.fr) `Baaiu` -> 3 records (2008 MCMDS, 2009 JPC, 2009 MCMDS), **none with a deposited file**; OpenAlex marks all closed. HAL `authFullName_t:(Jana Mousa)` -> her PhD thesis with full text (N1). Semantic Scholar: one request, HTTP 429, skipped.

## Fetched sources

### N1. Physically Consistent Hybrid Modeling for Chemical Process Systems via Neural Networks and Data Reconciliation (PhD thesis)
- Jana Mousa. HAL tel-05363400, dated 2025-10-10 (Toulouse, LGC line of Negny and Montastruc).
- Fetched: https://api.archives-ouvertes.fr/search/?q=halId_s:(tel-05363400) (abstract) and the full text https://theses.hal.science/tel-05363400/document (179 pp., read with pdftotext and grep).
- Supports: the dissertation behind Mousa et al., Comput. Chem. Eng. 199, 109156 (2025), which it lists as published article [1]. Method: NN soft sensors followed by linear or nonlinear data reconciliation (hard equality constraints for mass balances and reaction equilibria), benchmarked against PINN, NN+KKT and KKT+PINN. Case studies: flash drum, distillation column, Gibbs reactor, integrated reactor-distillation. Full-text grep: "adsorp" 0 hits, "metriplectic" 0, "port-Hamiltonian" 0; Hernández et al. 2021 (GENERIC) appears only in the literature review.
- Quote: "NNs function as soft sensors generating preliminary predictions"
- THREAT: **none**. Steady-state balance and equilibrium reconciliation; no adsorption, no entropy structure, no dynamics with ports. Closes risk item 1 of review section 6 (caveat: the CCE article itself is still closed; the thesis is its full-length version).

### N2. Thermodynamically Consistent Machine Learning for Fracture Prediction in Heterogeneous Materials
- Muhamad Alim, Xiaolong He, Mei-Na Tsao, Chi-Hua Yu, J. S. Chen. arXiv:2610.20614v1 (API "published" field 2026-07-12; filed in the 2610 series).
- Fetched: https://export.arxiv.org/api/query?id_list=2610.20614 (abstract).
- Supports: tLaSDI-pGFINNs applied to fracture; GENERIC structure by hard constraint in latent space; energy and entropy histories respected on unseen samples; beats AE-LSTM.
- Quote: "learns the thermodynamic state variable in a lower-dimensional latent space"
- THREAT: **low**. Another latent GENERIC model, but E and S are learned latent quantities; solid mechanics, closed, no ports. It adds one instance to "every latent GENERIC model learns E and S".

### N3. Physics-conforming Latent Twins
- Matthias Chung, Yutong Bu, Deepanshu Verma. arXiv:2606.15053 (v1 2026-06-13, v2 2026-06-24); CMAME 2026, doi:10.1016/j.cma.2026.119351 (OpenAlex date 2026-09-06; cites tLaSDI).
- Fetched: https://export.arxiv.org/api/query?id_list=2606.15053 (abstract); OpenAlex W7164943669 (metadata only).
- Supports: a latent surrogate (encoder, decoder, latent flow map) whose latent dynamics are constrained to preserve or dissipate prescribed structural quantities; a constraint-transfer view from state space to latent space, with bounds on decoded physical defects; algebraic conditions for linear and quadratic invariants and dissipative inequalities; ODE and PDE benchmarks.
- Quote: "constraining the latent dynamics to preserve or dissipate prescribed structural quantities"
- THREAT: **medium for the review's section-1 sentence, low for the gap.** Here the conserved or dissipated quantities are given and carried over from physical space, not learned, which is the move 3B makes. Per the abstract, though, it is not GENERIC or metriplectic, fixes no thermodynamic energy-entropy pair, has no ports or open boundaries, and no sorption. The sentence "In every latent model found, the energy and entropy are learned" must be narrowed, and N3 cited as the nearest prior for prescribed quantities in latent space.

### N4. Structure-preserving variational neural fields: Uncertainty-quantified reduced-order modeling of nonlinear conservation laws
- Aviral Prakash, Marc L. Klasky. arXiv:2607.10965v1, 2026-07-13 (missed by journals A-C).
- Fetched: https://export.arxiv.org/api/query?id_list=2607.10965 (abstract).
- Supports: latent neural-field ROM with exact conservation through a divergence-free space-time solution-flux representation; 1-D advection, 2-D Euler, shallow water; unconstrained, physics-informed and conservation-preserving variants compared; the structured one is more robust to sparse and noisy data.
- Quote: "conservation-structure preserving latent representations improve robustness to degraded training data"
- THREAT: **low**. Conservation only (no energy, entropy or second law), no ports, not sorption. Extra "for" evidence for review section 4 (structure helps with degraded data on an advection case).

### N5. Poisson-GENERIC Neural Operators (revised abstract, v2)
- Jason Sulskis, Sathya Ravi. arXiv:2610.05570v2 (v1 2026-10-04, v2 2026-10-12). Anchor; 0 citers in OpenAlex.
- Fetched: https://export.arxiv.org/api/query?id_list=2610.05570 and ?id_list=2610.05570v1 (both abstracts); https://arxiv.org/html/2610.05570v2 (full text, grep).
- Supports: the v2 abstract says the energy combines a fixed mechanical quadratic, a learned gauge-free potential and a convex internal energy, and that entropy is a latent density added to the state and made a Casimir. The v2 abstract replaces v1's learned-entropy sentence with a learned-energy ablation. The review's quote "a learned-entropy ablation injects energy on every reversible-irreversible problem" is in the **v1 abstract** and still in the **v2 full text**, so it stands but should be cited to the v1 abstract or the v2 body.
- Quote (v2 abstract): "a learned-energy ablation shows that conserving learned energy need not constrain physical energy"
- THREAT: **low**; it strengthens 3B's case for fixing E from thermodynamics. Not an autoencoder model, closed PDEs, no ports.

### N6. GENERIC-FNO (revised abstract, v9)
- Jason Sulskis, Sathya Ravi. arXiv:2606.08343v9, updated 2026-11-01 (v1 2026-06-06).
- Fetched: https://export.arxiv.org/api/query?id_list=2606.08343 (v9 abstract).
- Supports: E and S learned as neural operators; degeneracy exact by construction; the guarantees are stated to concern learned functionals within closed conservative-dissipative dynamics; over 200-step rollouts every unconstrained model tested diverges or collapses while GENERIC-FNO stays bounded, at 4-10x the compute, losing accuracy on pure transport.
- Quote: "within GENERIC's scope of closed conservative-dissipative dynamics"
- THREAT: **none**; it supports the gap (closed systems, learned E and S, weak on transport). The review can cite the v9 wording; the "pure transport" quote is unchanged.

### N7. A GENERIC-Informed Neural Network for Generative Rheology (GINNGER)
- Imanol García de Beristain, David Nieto Simavilla, Marco Ellero. arXiv:2610.23352v1, 2026-10-16.
- Fetched: https://export.arxiv.org/api/query?id_list=2610.23352 (abstract).
- Supports: learns GENERIC generators for polymer constitutive models in the conformation-tensor eigenspace; first and second laws by construction; generalises across flows and geometries.
- Quote: "satisfy the first and second laws of thermodynamics by construction"
- THREAT: **none**. Constitutive (closure) learning for complex fluids with learned generators; not a latent process model.

### N8. Thermodynamically consistent physics-informed neural network-finite element method for viscoelastic fluids using square-root conformation
- Qiyang Lei, Xiaodong Wang. Physics of Fluids, 2026-10-01, doi:10.1063/5.0348584 (cites GFINNs).
- Fetched: https://api.crossref.org/works/10.1063/5.0348584 (abstract).
- Supports: a PINN-FEM hybrid guided by GENERIC; polymer entropy learned as a scalar function of eigenvalues; Poiseuille, four-roll mill and cylinder-array flows.
- Quote: "strictly guided by the general equation for non-equilibrium reversible–irreversible coupling"
- THREAT: **none**. Constitutive learning, entropy learned, not sorption, not latent.

### N9. Port-Hamiltonian Neural Networks for Systems with Multiple Asymptotically Stable Equilibria
- Simon Heilig, Jens Püttschneider, Mohammad Itani, Asja Fischer, Timm Faulwasser. arXiv:2610.01356v2 (v1 2026-10-01).
- Fetched: https://export.arxiv.org/api/query?id_list=2610.01356 (abstract).
- Supports: Hamiltonian built from Bregman divergences of one input-convex network so several stable equilibria can coexist; 1.8x to 8.5x faster convergence on three systems.
- Quote: "their Hamiltonian is a global Lyapunov function with a single global minimum"
- THREAT: **none**. Finite-dimensional pH learning with a learned Hamiltonian; no thermodynamics of open transport.

### N10. Structure-Aware Spatial Kernels for Gaussian Process Distributed Port-Hamiltonian Systems
- Tianyi Yang, Thomas Beckers. arXiv:2610.22575v1, 2026-10-15.
- Fetched: https://export.arxiv.org/api/query?id_list=2610.22575 (abstract).
- Supports: GP-dPHS with tensor-product state and space kernels; a nonlinear string with heterogeneous stiffness.
- Quote: "enable data-driven modeling of distributed dynamical systems while preserving their physical structure"
- THREAT: **low**. Learned distributed pH (energy learned), mechanical, no entropy, no advective ports.

### N11. Data-Driven Boundary Control of Distributed Port-Hamiltonian Systems
- Thomas Beckers, Leonardo Colombo. arXiv:2604.04266v1, 2026-04-05 (older; missed by journals A-C).
- Fetched: https://export.arxiv.org/api/query?id_list=2604.04266 (abstract).
- Supports: learned GP-dPHS combined with boundary control by interconnection; probabilistic boundedness under model mismatch; a simulated shallow-water system.
- Quote: "The GP-dPHS model is used to infer the unknown Hamiltonian structure from data"
- THREAT: **low**. The nearest item found to "learned distributed structure with boundary ports on a fluid", but the Hamiltonian is learned, the structure is conservative pH (no entropy, no irreversible part), it is not a latent autoencoder model, and not sorption. Worth one citation under qualifier (c).

### N12. A unified and composable framework for modeling packed-bed liquid chromatography
- Jan Michael Breuer, Samuel Leweke, Johannes Schmölder, Antonia Berger, Gregor J. Gassner, Eric von Lieres. arXiv:2610.33269v1, 2026-10-23.
- Fetched: https://export.arxiv.org/api/query?id_list=2610.33269 (abstract).
- Supports: a mechanistic model hierarchy derived from a 3-D model by formal model order reduction; modular transport, mass-transfer and binding modules; CADET-Equations.
- Quote: "integrates diverse mechanistic packed-bed chromatography models into a coherent structure"
- THREAT: **none**. Non-ML, isothermal liquid chromatography, and no energy or entropy structure in the abstract. A possible citation for column-model taxonomy.

### N13. diff-soup: GPU-native Differentiable Simulation and Inference for Large-scale Stiff Dynamical Systems
- Samuel Degnan-Morgenstern, Martin Z. Bazant. arXiv:2610.37333v1, 2026-10-27.
- Fetched: https://export.arxiv.org/api/query?id_list=2610.37333 (abstract).
- Supports: a differentiable JAX PDE suite; one demonstration trains a neural field for catalyst activity in a 3-D packed-bed reactor.
- Quote: "training a neural field for the catalyst activity of a three-dimensional packed bed reactor"
- THREAT: **none**. Tooling, with a neural closure inside a mechanistic solver and no thermodynamic structure. It shows differentiable mechanistic competitors (arm C) getting stronger.

### N14. Differentiable mass transfer modeling enables wet lab validated inverse analysis of liquid chromatography
- Baochen Li, Zihan Zhou, Yanqing Yu, Sen Lin, Peng Wang et al. ChemRxiv 2026-09-23, doi:10.26434/chemrxiv.15009352/v1.
- Fetched: https://api.openalex.org/works?filter=openalex:W7214085388 (abstract).
- Supports: a PINN surrogate of a mechanistic LC column (RMSE 10^-3 in dimensionless concentration) used for gradient-based estimation of transport and adsorption parameters; observability-guided experiment design; 2,553 wet-lab chromatograms.
- Quote: "The primary PINN surrogate reproduces the mechanistic reference"
- THREAT: **none**. Physics as a loss, isothermal LC, parameter inversion, no structure by construction, no multi-cycle rollout.

### N15. Cycle-resolved dynamic modeling of atmospheric CO2 separation systems under time-varying ambient and electricity conditions
- Keju An, Dylan Wald, Kejun Chen, Ryan King. Frontiers in Environmental Engineering, 2026-09-01, doi:10.3389/fenve.2026.1881554.
- Fetched: https://api.openalex.org/works?filter=openalex:W7204955176 (abstract).
- Supports: a physics-informed state-transition ROM of a TVSA DAC unit with CO2-H2O co-adsorption and cycle-to-cycle sorbent memory; about 120,000 cycle-resolved simulations; an NN surrogate for cyclic prediction and optimisation.
- Quote: "supports the development of a neural-network surrogate for rapid cyclic prediction"
- THREAT: **low**. A 2026 neural surrogate of a TSA-type cycle (target d), but unstructured and at cycle level (a lumped state-transition model, not a column field), with no energy or entropy structure and no admissibility test. Add to review section 3, "Unstructured".

### N16. Artificial Neural Network Surrogate for Entropy Generation Analysis in PCM-Based Thermal Energy Storage
- Łukasz Łach. Energies, 2026-10-30, doi:10.3390/en19215164.
- Fetched: https://api.crossref.org/works/10.3390/en19215164 (abstract).
- Supports: an MLP maps five dimensionless groups to an entropy generation number for 1-D PCM melting (R^2 0.991, MAPE 5.73 %).
- Quote: "requires fast and accurate prediction of entropy generation across wide operating envelopes"
- THREAT: **none**. Entropy generation is a regression output computed after the fact, not structure. It supports review section 2's "computed only after the fact".

### N17. Modélisation et analyse des systèmes de commande multi-physiques (PhD thesis)
- Benjamin Vincent. 2019-07-02 (Lyon LAGEP line), OpenAlex W7148860358 (a citer of Baaiu et al. 2008).
- Fetched: https://api.openalex.org/works?filter=openalex:W7148860358 (abstract).
- Supports: port-Hamiltonian and open-GENERIC formulations of distributed multi-physics systems, with passivity based on entropy production; applications are tokamak plasma profiles and a Rijke tube.
- Quote: "Un modèle thermomagnétique du plasma est formulé avec le formalisme GENERIC ouvert"
- THREAT: **none** for 3B's column. Bears on risk item 2: per the abstract, the Lyon line's later open-GENERIC work went to plasma and thermo-acoustics, not to a non-isothermal adsorption column. Optional citation for open GENERIC in distributed systems.

### N18. Approche thermodynamique pour la stabilisation des réacteurs chimiques (PhD thesis)
- Ngoc Ha Hoang. 2009-12-01, OpenAlex W7143438863 (a Baaiu citer; the Hoang line is already in journal B).
- Fetched: https://api.openalex.org/works?filter=openalex:W7143438863 (abstract).
- Supports: a pseudo-Hamiltonian port formulation of open CSTRs; the Hamiltonian is tied to the Gibbs free enthalpy in the isothermal case and to ectropy in the non-isothermal case; passivity-based control.
- Quote: "l’Hamiltonien est lié à l’enthalpie libre de Gibbs dans le cas isotherme"
- THREAT: **none**. A lumped reactor, non-ML, consistent with journal B.

### N19. Contribution à l'extension de l'approche énergétique à la représentation des systèmes à paramètres distribués (PhD thesis)
- Cătălin-Marian Chera. 2009-07-01, HAL; OpenAlex W83606874 (a Baaiu citer).
- Fetched: https://api.openalex.org/works?filter=openalex:W83606874 (abstract).
- Supports: bond-graph and port-Hamiltonian representations of distributed-parameter systems, applied to road-traffic flow.
- Quote: "son extension aux systèmes à paramètres distribués est proposée"
- THREAT: **none**. Not adsorption, not thermal.

### N20. Learning thermodynamic master equations for open quantum systems
- Peter Sentz, Stanley Nicholson, Yujin Cho, Sohail R. Reddy, Brendan Keith et al. Quantum, 2026-07-01, doi:10.22331/q-2026-07-01-2151 (its full text mentions port-metriplectic NNs).
- Fetched: https://api.openalex.org/works?filter=openalex:W4414898802 (abstract).
- Supports: a data-driven open-quantum-system model with learnable, thermodynamically consistent nonlinear terms and an interpretable Hamiltonian estimate.
- Quote: "includes learnable, thermodynamically consistent terms"
- THREAT: **none**. Here "open" means coupling to an environment, not inflow and outflow transport.

### N21. Stable Dynamical Mass Transportation via port-Hamiltonian Structure
- Dongjun Wu, Yankai Lin, Gang Feng. arXiv:2610.38465v1, 2026-10-28.
- Fetched: https://export.arxiv.org/api/query?id_list=2610.38465 (abstract).
- Supports: density-feedback control that steers distributions using a stochastic pH reference model.
- Quote: "We use a stochastic port-Hamiltonian system as a reference model"
- THREAT: **none**. "Mass transport" in the optimal-transport sense; control, non-ML.

### N22. Optimal energy conversion -- A port-Hamiltonian approach
- Guanru Pan, Karl Worthmann, Manuel Schaller, Timm Faulwasser. arXiv:2610.21790v1, 2026-10-15.
- Fetched: https://export.arxiv.org/api/query?id_list=2610.21790 (abstract).
- Supports: optimal cyclic efficiency of pH storage-conversion systems and a generalized Carnot-type bound.
- Quote: "establish a generalized Carnot-type efficiency bound"
- THREAT: **none**. Non-ML theory; a possible discussion citation if 3B reports cycle efficiency.

### N23. Autoencoder-based non-intrusive model order reduction for anisotropic and multi-physics problems
- Jannick Kehls, Ellen Kuhl, Tim Brepols, Kevin Linka, Hagen Holthusen. Computational Mechanics, 2026-10-13, doi:10.1007/s00466-026-02870-y (cites 2007.03758).
- Fetched: https://api.openalex.org/works?filter=openalex:W7221636614 (abstract).
- Supports: an unstructured AE-plus-regression ROM, including thermo-mechanical multi-field outputs; maps parameters to fields, not dynamics.
- Quote: "a multi-field architecture that enables coupled field predictions, such as in thermo-mechanical systems"
- THREAT: **none**.

## SEEN-ONLY leads (title and metadata only; no abstract in OpenAlex or Crossref; not counted)
- Li, Shi, Yang, Yuan, Chen et al., "Machine learning methodology and SHAP interpretability for predicting multi-component breakthrough times of pressure swing adsorption hydrogen purification", Int. J. Hydrogen Energy 2026-10-19, doi:10.1016/j.ijhydene.2026.158114 (by title, unstructured regression of breakthrough times).
- Son, Lee, Kim, "Resource-constrained Bayesian optimization assisted by machine learning for vacuum pressure swing adsorption", Sep. Purif. Technol. 2026-10-22, doi:10.1016/j.seppur.2026.140580.
- Bahrun, Zaini, Satriyo, Anugraha, Juwari et al., "Tree-based versus neural surrogate-assisted evolutionary algorithms for multi-objective optimization of DR-PSA biogas upgrading ...", Sep. Purif. Technol. 2026-10-13, doi:10.1016/j.seppur.2026.140549.
- Sharma, Adams, "Cycle design and surrogate-based multi-objective optimisation of magnetic induction swing adsorption ...", Chem. Eng. Res. Des. 2026-10, doi:10.1016/j.cherd.2026.09.029.
- Algoufily, Borghesan, Vanhoutte, "A k-nearest neighbours framework with Trajectory-Embedded Decay for cycle-time forecasting in temperature swing adsorption", Digital Chem. Eng. 2026-09, doi:10.1016/j.dche.2026.100323.
- Fang, Wen, Zheng, Shu, You et al., "The necessity of physics-informed neural networks for adsorption-based water treatment machine learning models: A case study of biochar adsorption", Water Res. 2026-10-13, doi:10.1016/j.watres.2026.127101.
- Lockyear, Garimella, "First-principles-based identification of the significance of heat and mass transfer in chemisorption systems", Int. J. Refrig. 2026-09-21, doi:10.1016/j.ijrefrig.2026.107133 (non-ML).
- Wang, Zhang, Yuan et al., "Reliable extrapolation of multicomponent thermophysical properties to extreme operating conditions via physics-constrained generative learning", Chem. Eng. J. 2026-09-22, doi:10.1016/j.cej.2026.182098 (cites tLaSDI; by title about properties, not dynamics).
- Cherifi, El Messaoudi, Gernandt, "Nonlinear port-Hamiltonian system identification from input-state-output data (ISO-pHNN)", Physica D 2026-11, doi:10.1016/j.physd.2026.135368; and El Messaoudi, Cherifi, Le Gorrec, Wu, "PH-KAN", arXiv:2606.14708 (the Le Gorrec group's learned-pH line; listing titles only).
- Li, Fan, Zhang, "Prediction of MOF Adsorption Isotherms ... Physics-Informed Membranes Graph Neural Network", ChemRxiv 2026-10-12, doi:10.26434/chemrxiv.15010295/v1 (isotherms, not dynamics).
- Shutaywi, Shah, Vrînceanu, Deeban, "Machine learning-driven adsorption science: A review ...", Chem. Phys. Impact 2026-10-15, doi:10.1016/j.chphi.2026.101183 (a review; worth checking for any structured column model).
- Mousa et al., Comput. Chem. Eng. 199, 109156 (2025): the article page is still closed with no abstract (OpenAlex is_oa false, cited 12 times), but N1 is its full-length dissertation.

None of the SEEN-ONLY titles points to a learned column model that is thermodynamically structured (GENERIC, metriplectic, pH or Onsager).

## Synthesis (2026-11-03)

**Does the section-5 gap still hold as worded? Yes.** The sweep ran 26 arXiv queries, 13 OpenAlex topical searches, 10 OpenAlex citer sweeps, 5 Crossref queries and HAL. Nothing published or indexed from 2026-10-01 to 2026-11-03, and no older item found here, is a learned latent model with thermodynamic structure by construction of any of these:
- a non-isothermal adsorption column;
- any sorption, chromatography, packed-bed or heat-exchanger system.

Still less does any fix E and S from thermodynamics or treat the inlet, outlet and wall as ports. The two queries closest to qualifier (c) both returned 0 hits: structure-preserving plus inflow/outflow/open system plus learning (arXiv A17), and adsorption plus neural surrogate plus column (OpenAlex OA4). The anchors 2610.05570, 2606.08343, 2506.08475 and 2603.11260 have no citers yet in OpenAlex, and no paper has cited Baaiu et al. 2008 since 2019.

The qualifiers against the new items:
- **Non-isothermal column:** no new structured column model. The Baaiu citers (N17 to N19) went elsewhere: plasma, the Rijke tube, traffic, CSTRs.
- **Learned, latent:** the new latent GENERIC model N2 learns E and S. N4 is conservation-only.
- **E and S fixed from thermodynamics:** N2 and N6 learn both. N5 fixes only a mechanical quadratic part of E and learns the rest, plus a latent entropy. N3 prescribes quantities, but not a thermodynamic E and S pair.
- **Adsorption column with ports:** the nearest is N11, a learned distributed pH system with boundary control, but it is conservative and its energy is learned. GENERIC-FNO v9 limits itself to closed dynamics (N6).
- **Tested what structure buys, pre-declared:** the 2026 sorption surrogates (N15 and the SEEN-ONLY PSA, VPSA and TSA items) are unstructured.

**Rewording needed.** Only the gap sentence's date and scope change:

> No published work, as of 2026-11-03 within the sources searched (review journals A to C and update journal D), learns a reduced (latent) model of a **non-isothermal adsorption column** whose **energy and entropy are fixed from the adsorbent's and the gas's thermodynamics** (not learned), with the **inlet, outlet and wall as ports**, and **tests, against the same model without structure and under a pre-declared protocol, what that structure buys** in long multi-cycle rollouts, data efficiency and admissibility.

Supporting text in the review that should change (the gap itself does not):
1. **Section 1, "In every latent model found, the energy and entropy are learned in latent coordinates without physical meaning".** Replace with: "In every latent GENERIC or metriplectic model found (now including a fracture application of tLaSDI, arXiv:2610.20614), the energy and entropy are learned in latent coordinates without physical meaning. The nearest exception in spirit is Physics-conforming Latent Twins (Chung, Bu, Verma, CMAME 2026, arXiv:2606.15053). It constrains the latent flow to preserve or dissipate prescribed quantities carried over from the physical state space. It does not fix a thermodynamic energy-entropy pair, has no GENERIC degeneracy and no ports." The section-5 table row "E and S fixed from thermodynamics" should add "Physics-conforming Latent Twins (prescribed invariants, not a thermodynamic E and S)".
2. **Section 1, Poisson-GENERIC.** Cite the "injects energy" quote to the v1 abstract (2026-10-04) or the v2 full text. The v2 abstract (2026-10-12) instead reports a learned-energy ablation, and says the energy combines a fixed mechanical quadratic with learned parts.
3. **Section 1, transport and open systems.** Add two citations:
   - GENERIC-FNO v9's own scope statement, "closed conservative-dissipative dynamics";
   - Beckers and Colombo 2026 (arXiv:2604.04266), the nearest learned distributed-port model, which is conservative and learns its Hamiltonian.
4. **Section 3.** Add An et al. 2026 (Front. Environ. Eng., doi:10.3389/fenve.2026.1881554) under "Unstructured": an NN surrogate of a TVSA DAC cycle ROM.
5. **Section 4, "For".** Optionally add Prakash and Klasky (arXiv:2607.10965): on 1-D advection, latent fields that preserve conservation are more robust to degraded training data.
6. **Section 6:**
   - Item 1 (Mousa et al.) is **resolved: no threat**. The HAL full-text thesis (N1) shows NN plus data reconciliation on a flash drum, a distillation column, a Gibbs reactor and a reactor-distillation system, with no adsorption.
   - Item 2 (Baaiu full texts) is **still open**. HAL holds no file and OpenAlex marks all three closed, so library access is still required.
   - Item 4: Semantic Scholar again returned 429. OpenAlex did not hit its limit during this sweep.
   - The citation-gate sweep before submission is still required.

