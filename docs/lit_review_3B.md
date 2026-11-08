# Paper 3B: literature review and the exact gap

Compiled 2026-10-12 to 13 and updated 2026-11-03. The review rests on five source journals. A source counts only if
its live page (an abstract page, a full text, or a bibliographic record with an abstract)
was fetched in the session that cites it. Snippets from search results are leads, labelled
SEEN-ONLY, and nothing below rests on one without saying so. No archived copies were used.

| journal | where | fetched | lens |
|---|---|---|---|
| structure-preserving learning, 2026-09-25 | Paper 1 repo, `research/lit_structure_preserving.md` | 55 (S1 to S55) | the method families and their fit to sorption |
| A: metriplectic ML | `research/lit_A_metriplectic_ml.md` | 43 (T1 to T43) | 2024 to 2026 GENERIC, metriplectic, port and Onsager learning |
| B: thermodynamic column models | `research/lit_B_thermo_column_models.md` | 33 (30 with abstract or text) | non-ML GENERIC, IPHS and port-based models of columns, reactors, exchangers |
| C: sorption surrogates and rollouts | `research/lit_C_sorption_surrogates_rollouts.md` | 39 (C1 to C39) | learned adsorption models; what structure buys over long rollouts |
| D: update, 2026-11-03 | `research/lit_D_update_2026-11-03.md` | 23 (N1 to N23) | anything new since 2026-10-01; citers of the anchor papers; search hit counts logged |

The sources this review leans on most were re-checked live on 2026-10-13 against arXiv,
Crossref, OpenAlex or Zenodo (title, authors, date). That covers 2610.05570, 2606.08343,
2606.01366, 2403.05848, 2506.08475, 2603.11260, 2512.16305, 2601.09491, 2211.06130,
2609.04530, 1204.3841 and 2302.11002, Baaiu et al. 2008 and 2009 and Peng et al. 2007, Walton
and LeVan 2005, Schwamberger and Schmidt 2013, Myers and Monson 2014, Dytrych and Dlaskova
2025, Glover et al. 2026, Mukherjee and Bhattacharyya 2024, Bermejo et al. 2025, Li et al.
2019, and Staudt et al. 2026. On 2026-11-03: 2606.15053, 2604.04266, 2607.10965, 2610.20614,
2610.05570v1, the v9 record of 2606.08343, An et al. 2026 (Crossref), and the HAL record of
the Mousa thesis (tel-05363400).

## 1. Structure-preserving learning is mature, and every latent version found learns its potentials

Building the GENERIC or metriplectic structure into the architecture, rather than adding it
as a loss, is now routine for ODEs, graphs and neural operators.
- **Structure as architecture.** GFINNs satisfy the degeneracy conditions by architecture
  (Zhang, Shin, Karniadakis, Phil. Trans. R. Soc. A 380, 20210207, 2022; arXiv:2109.00092).
  Gruber et al. make the parameterisation scale quadratically in the state
  (arXiv:2405.16305). Port-metriplectic networks extend it to open systems learned by parts
  (Hernández et al., Comput. Mech. 2023, doi:10.1007/s00466-023-02296-w; arXiv:2211.01873).
- **Function space.** GENERIC-FNO (arXiv:2606.08343) and Poisson-GENERIC neural operators
  (arXiv:2610.05570, 4 October 2026) carry the structure into function space. The second wins
  61 of 72 seed-level comparisons.

Latent (autoencoder) versions exist too:
- the thermodynamics-aware reduced model of Hernández et al. (CMAME 379, 113763, 2021;
  arXiv:2007.03758);
- tLaSDI (Park et al., CMAME 429, 117144, 2024; arXiv:2403.05848) and its parametric
  successor (He et al., TMLR 2026; arXiv:2506.08475);
- a magnetisation-dynamics operator (Schaffer and Exl, arXiv:2609.04530; the statements used
  here are in its v2 full text, Section 4.5 and the rollout figures). It learns a single
  scalar potential and has no entropy.

tLaSDI is the closest architectural template to 3B. It reports extrapolation errors about 3
to 7 times lower than its baselines, for example 2.47×10⁻³ against 1.70×10⁻² for a plain
network on Couette flow (journal C, C17). Two caveats travel with that number: it comes from
the run with the smallest training loss of 10, and in one of its two ablations the modelling
loss added nothing once the Jacobian loss was present.

**In every *learned* latent GENERIC or metriplectic model found, the energy and entropy are
learned in latent coordinates without physical meaning.** That now includes a fracture application
of tLaSDI (arXiv:2610.20614). The magnetisation paper says explicitly that its learned
potential is not the Gibbs free energy (journal A, T11). The nearest exception in spirit is
Physics-conforming Latent Twins (Chung, Bu, Verma, CMAME 463, 119351, 2027 issue,
doi:10.1016/j.cma.2026.119351; arXiv:2606.15053): its latent
flow preserves or dissipates *prescribed* quantities carried over from the physical state.
It does not fix a thermodynamic energy-entropy pair, has no GENERIC degeneracy and has no
ports. That matters for two reasons:
- "respects the second law" is then a statement about a surrogate quantity;
- learning the entropy has a documented failure: in Poisson-GENERIC operators, a
  learned-entropy ablation "injects energy on every reversible-irreversible problem" (T5).
  That sentence is in the v1 abstract of arXiv:2610.05570 (4 October 2026) and the v2 full
  text. The v2 abstract reports a learned-energy ablation instead, so cite v1.

Fixing a trusted energy is partly anticipated outside latent models:
- the fixed structural energy terms of Poisson-GENERIC (T5);
- a trusted-energy projection in metriplectic flow matching (arXiv:2509.19526, T13);
- irreversible port-Hamiltonian neural ODEs, whose structure comes from physics and whose
  parameters are learned (Zakwan et al., IFAC-PapersOnLine 56(2), 5855, 2023;
  arXiv:2211.06130, T17), a lumped thermal model.

None of these is a learned latent model with E and S fixed from the physics of the decoded
fields.

**Projection-based reduced models already keep the full model's energy and entropy.** This
line of work is intrusive: the reduced dynamics are derived by projection, not learned.
- Öttinger preserves GENERIC structure in model reduction (Phys. Rev. E 91, 032147, 2015,
  doi:10.1103/PhysRevE.91.032147).
- Gruber, Gunzburger, Ju and Wang build energetically consistent reduced models of
  metriplectic systems (CMAME 404, 115709, 2023, doi:10.1016/j.cma.2022.115709).
- Chan builds entropy-stable reduced models from entropy-stable finite-volume full models,
  the same setting as 3B's reference (J. Comput. Phys. 423, 109789, 2020,
  doi:10.1016/j.jcp.2020.109789).
- Klein, Sanderse, Costa, Pecnik and Henkes keep the physical entropy of the decoded state on
  a nonlinear (rational quadratic) manifold (J. Comput. Phys. 528, 113817, 2025,
  doi:10.1016/j.jcp.2025.113817). That is the closest analogue to 3B's "E and S evaluated on
  decoded fields".

All four records were checked on Crossref on 2026-11-08 (found by the Claude Science review).
"E and S fixed from physics in reduced coordinates" is therefore **not new by itself**. What
3B adds is learned latent dynamics with those fixed potentials, for an open,
advection-dominated, non-isothermal system with ports, and a pre-declared test of which part
of the structure helps. An entropy-stable Galerkin reduced model of the reference is a
natural further non-learned competitor. It is named as future work, because its cost on a
stiff, steep-isotherm column would need its own study.

**Transport is the documented weak spot.** GENERIC-FNO loses accuracy "on pure transport"
(T4). In the first version of the Poisson-GENERIC paper (arXiv:2610.05570v1), the friction
class provably excludes transport, so transport enters through a Lie-Poisson reversible
operator (T5). GENERIC-FNO's latest version (v9, 1 November 2026) also limits its guarantees
to "closed conservative-dissipative dynamics". No fetched paper shows a learned metriplectic
latent model of an advection-dominated *open* system with inflow and outflow ports and a
quantified rollout error. Two models come closest:
- a learned distributed-port model that learns its Hamiltonian with a Gaussian process, on a
  shallow-water channel with linear friction. It has dissipation but no thermodynamic entropy
  and no second-law structure (Beckers and Colombo, arXiv:2604.04266);
- a thermodynamics-informed MeshGraphNet with a nodal port-metriplectic formulation, for
  incompressible flow past obstacles (Bermejo et al., Adv. Model. Simul. Eng. Sci. 12:27,
  2025, T16). Its E and S gradients are learned, it is full-field rather than latent, and its
  abstract does not say how inflow and outflow fluxes are treated.

## 2. Structured models of adsorption columns exist; every one is isothermal (full texts read)

The Lyon LAGEP group built a **port-based distributed model of a multiscale adsorption
column "under isothermal conditions"**, using bond graphs, Dirac structures and the axioms
of irreversible thermodynamics (Baaiu, Couenne, Le Gorrec, Lefèvre, Tayakout, Math. Comput.
Model. Dyn. Syst. 14, 195, 2008, doi:10.1080/13873950701844840). A general port-based theory
of transport followed (Baaiu et al., Math. Comput. Model. Dyn. Syst. 15, 233, 2009,
doi:10.1080/13873950902808578), and so did structure-preserving discretisation and reduction
applied to adsorption (Peng, Baaiu, Couenne, Le Gorrec, IEEE CDC 2007, 3513,
doi:10.1109/cdc.2007.4434663; Baaiu, Couenne, Lefevre, Le Gorrec, Tayakout, J. Process Control
19, 394, 2009, doi:10.1016/j.jprocont.2008.07.002).

**The full texts were read on 2026-11-05** (the author's library access: the CDC and J.
Process Control papers in full, the 2008 paper from the publisher's reader, and the 2009
theory paper in full, the last from the publisher's free-access page). What they contain:
- **All three adsorption papers are isothermal and isobaric by assumption.** The 2008 paper
  takes the column "at constant temperature and pressure" and says it will "omit the thermal
  and spatial domain". The CDC paper assumes temperature "uniform and constant". The J.
  Process Control paper supposes "isothermal and isobaric operating conditions". The storage
  function in each is the Gibbs free energy, which is a valid potential only at fixed T and P.
  There is no energy balance, no heat of adsorption, no entropy as a state and no learning.
- **The 2009 theory paper** names heat and mass transfer in its abstract, but states in
  Section 4: "In this article, the thermal domain is not considered." It adds that otherwise
  the entropy production would enter a thermal balance, and "the energy balance should then
  be computed on the internal energy rather than on the Gibbs free energy". That is the step
  3B takes (formulation, Sections 3 and 4). Its Section 5 treats convection by a given
  velocity whose energy source is "in the system environment": the velocity is "a control
  variable", entered through a CONV element modulated by it.
- **Convection is outside the dissipative structure in every one.** In the J. Process
  Control paper, convection and dispersion enter "as a distributed source term added to the
  diffusion term". That is the same choice 3B makes for an imposed velocity (formulation,
  Section 4.1), and it gives that choice a precedent in this line of work.

So the non-isothermal, entropy-explicit column formulation of 3B is not anticipated by this
group's papers. 3B extends their isothermal, free-energy-based structure to a fixed energy and
entropy with a thermal domain and, unlike their models, it is learned in latent coordinates.

The rest of the non-ML literature (journal B):

- **Convection by an imposed velocity has no settled treatment.**
  - Irreversible port-Hamiltonian models of reaction-diffusion and conduction-diffusion
    leave convection out (P6, P8, P10).
  - A 2026 extension brings it inside the structure, but only where the velocity is a state
    of the fluid (Ouardi et al., arXiv:2603.11260).
  - Others treat it as a "dissipation port" (a counter-current heat exchanger, P14), as a
    boundary supply in passivity analyses (P15, P29), or as compartments in series (a heat
    exchanger, P13). In a related study, convection together with nonlinear
    phenomenological relations "may destroy the dissipation condition" (P28).

  3B's choice (formulation, Section 4.1) is the last of these, with a proof of non-negative
  mixing entropy.
- **Uptake as a flux driven by chemical-potential difference** is standard
  non-equilibrium thermodynamics (Santamaria-Holek, Grzywna, Rubi, J. Non-Equilib.
  Thermodyn. 37, 2012, doi:10.1515/jnetdy-2011-0029; arXiv:1204.3841). A 2025
  preprint claims standard isotherm-plus-LDF couplings can violate the second law
  (HARMONIA, doi:10.26434/chemrxiv-2025-c860f; read as an abstract record only). Formulation
  Section 4.2 shows that for one adsorbate with an increasing isotherm they cannot.
- **Consistency of heats and heat capacities is known.** Walton and LeVan (Adsorption 2005)
  derive energy balances along consistent thermodynamic paths and tie the adsorbed-phase
  heat capacity to the temperature dependence of the isosteric heat. Schwamberger and Schmidt
  (Ind. Eng. Chem. Res. 52, 16958, 2013) show that computing the adsorbed-phase heat capacity
  along reversible thermodynamic paths makes the energy and entropy balance residuals much
  smaller (water on zeolite 13X). Myers and Monson (Adsorption 2014) argue for absolute adsorption as the
  thermodynamic basis. 3B cites them and does not claim the consistency point as new.
- **Entropy generation in adsorption cycles has been computed only after the fact.** A
  lumped TSA cycle attributes 63.27 % of entropy generation to heat transfer, 53.72 % with
  heat recovery (Li et al., Sci. Rep. 2019). A CFD TSA study does the same spatially (Guo et
  al., Entropy 2019). No adsorption-cycle model found carries entropy production as
  structure; in these studies it is computed after the fact.

## 3. Learned sorption models: unstructured, penalised, or balance-constrained

From most to least structure (journal C, Part 1):

- **Hard balances without a second law:** mass- and energy-constrained networks for
  distributed chemical systems (Mukherjee and Bhattacharyya, Ind. Eng. Chem. Res. 2024) and
  hard enthalpy balances in a packed-bed reactor surrogate (Lastrucci et al., ESCAPE 35,
  Syst. Control Trans. 4, 1718, 2025, doi:10.69997/sct.108423; arXiv:2501.17782). Neither is an adsorption column, and neither carries entropy.
- **A conservative solver with a learned part:** FINN for groundwater diffusion and sorption
  (Praditia et al., Water Resour. Res. 58, e2022WR033149, 2022), a finite-volume network with
  no energy equation that, in this application, learns the sorption isotherm. Hybrid
  chromatography models screened by Frandsen et al. (AIChE J. 71, e70045, 2025,
  doi:10.1002/aic.70045).
- **Physics as a loss:** PINN surrogates of adsorption processes (Galanti et al., Processes 13,
  2824, 2025, doi:10.3390/pr13092824) and of a PSA cycle (Wu et al., Green Chem. Eng. 6, 233,
  2025, doi:10.1016/j.gce.2024.08.004).
- **Unstructured:**
  - Staudt et al. 2026: a neural surrogate of binary non-isothermal breakthrough trained on
    1.1 million simulations (Zenodo 22556794).
  - Ceccanti et al. 2026: a DeepONet for cyclic TVSA, which assumes isothermal operation
    (arXiv:2601.09491).
  - Bilal, Haq, Asif and co-authors 2026: scalar-output surrogates of a non-isothermal
    chromatography general rate model. One uses a deep ensemble with physics-derived
    dimensionless ratios as input features (Ind. Eng. Chem. Res. 65, 13942,
    doi:10.1021/acs.iecr.6c02282). The other uses XGBoost on 31 dimensionless model
    parameters and checks its feature rankings against the model's sensitivities (Bilal, Haq,
    Asif, Alhartomi, ACS Omega 11, 48428, doi:10.1021/acsomega.6c04740). Neither constrains
    its outputs.
  - An, Wald, Chen, King 2026: a neural surrogate of a temperature-vacuum swing (TVSA)
    direct-air-capture cycle, trained on about 120,000 cycle-resolved simulations of a
    lumped state-transition model (Front. Environ. Eng., doi:10.3389/fenve.2026.1881554).
    It is cycle-level, not a column field, and has no energy or entropy structure.
- **A strong non-learned competitor:** a differentiable mechanistic VPSA model that reaches
  the cyclic steady state about 20 times faster than a representative MATLAB implementation
  (Glover, Papathanasiou, Pini, Comput. Chem. Eng. 216, 109904, online 2026,
  doi:10.1016/j.compchemeng.2026.109904).

Differentiable mechanistic simulators keep getting stronger: diff-soup (Degnan-Morgenstern
and Bazant, arXiv:2610.37333) trains a neural field for the catalyst activity inside a stiff,
differentiable 3-D packed-bed reactor solver on GPU. That strengthens the case for arm C. No
learned sorption model found reports zero unphysical outputs over many cycles. **No learned
adsorption, sorption or chromatography model that is thermodynamically structured by
construction was found** in journals A and C. The searches covered GENERIC, metriplectic,
port-Hamiltonian, Onsager, entropy-stable and "thermodynamically consistent", crossed with
adsorption, sorption, chromatography, desiccant, PSA, TSA, breakthrough and water harvesting.

One candidate exception was checked: Mousa, Negny, Ouaret, Di Pretoro, Montastruc,
"Incorporating Physical Constraints inside Neural Networks … Chemical Engineering Unit
Operations Modeling", Comput. Chem. Eng. 199, 109156 (2025). It cites GFINNs. The article itself
is unread (closed), but the first author's open thesis, which lists it as its own, describes
neural networks corrected by data reconciliation on balances and equilibria, with no adsorption
(Section 6, item 1).

## 4. What structure buys: the evidence is real but mostly from toy systems

**For:**
- **Conservative structure, 1-D benchmark.** The cleanest numbers come from a 1-D
  drift-diffusion-Poisson benchmark close in form to a column (arXiv:2606.01366). A
  conservative finite-volume model with a learned flux correction reached a rollout error of
  7×10⁻⁹, against 25 to 67 for unconstrained regressors. Projection after the fact and
  rollout training did not rescue the unconstrained models. This study has already asked
  part of 3B's question, projection against training. Its abstract lists the controls added
  to the regressors: Poisson recomputation of the field, charge projection and 4-step
  rollout training. None of them gives an unconstrained model the structured model's
  physical features as inputs, which is what 3B's U+F does, as far as the abstract states.
- **Latent extrapolation:** tLaSDI's extrapolation gains (above).
- **Function-space operators:** Poisson-GENERIC's 61 of 72 seed-level wins.
- **Data efficiency:** constrained Hamiltonian and Lagrangian forms improved it on a double
  pendulum over 10 to 10,000 trajectories (arXiv:2012.02334).
- **Robustness to poor data:** on 1-D advection, latent fields that preserve conservation
  degraded less when the training data were degraded (Prakash and Klasky,
  arXiv:2607.10965).

**Against, or qualifying:**
- **One-step accuracy is traded away.** The same benchmark was best on rollout error in 60 of
  64 configurations, but best on one-step error in only 19. Its authors note that the
  classical finite-volume core alone reaches near round-off, so the gain comes from the
  discrete structure, not the learned part.
- **The wrong structure hurts.** Latent gradient flows lose rollout accuracy against models
  with an antisymmetric part (arXiv:2609.04530). Hamiltonian networks gain from second-order
  form, not from conservation, and strict conservation hurts on dissipative systems
  (arXiv:2202.04836).
- **Hidden state reverses the ranking.** With variables missing, a Transformer beats a
  structure-preserving network (Urdeitx et al., J. Non-Newton. Fluid Mech. 349, 105601,
  2026; arXiv:2512.16305).
- **Hard conservation can break positivity.** In the ProbConserv paper, a baseline that
  enforces conservation by shifting every point equally produced a negative bias in the
  zero-density region (Hansen et al., Physica D 457, 133952, 2024; arXiv:2302.11002).
- **Penalties have their own failures:** ill-conditioning, and ensemble collapse under
  shared constraints (journal C, C34; Nooraiepour et al., Energy and AI 24, 100757, 2026).
- **Headline numbers can come from the best of several runs.** tLaSDI's come from the run
  with the smallest training loss of 10 (C17). The standard benchmark suite's abstract lists
  no thermodynamic-structure metric (PDEBench, C31).

## 5. The exact gap

**The question, as a broad reader would put it.** When physics is built into a learned
reduced model of an open dissipative system, does it help through the constraint itself,
through the physical information the constraint supplies, or through a projection at
rollout? And does fixing the model's energy and entropy from physics beat learning them? No
study found separates these. Projection against training has been examined on one 1-D
benchmark (arXiv:2606.01366), whose abstract lists no information-matched control. Fixed against learned
potentials has not been tested in learned latent models, and projection-based reduced models
fix them but do not learn.

**The precise novelty statement** (for the related-work section):

> No published work, as of 2026-11-08 within the sources searched (journals A to D and the
> Claude Science review), learns a reduced (latent) model of a **non-isothermal adsorption
> column** whose **energy and entropy are fixed from the adsorbent's and the gas's
> thermodynamics** (not learned), with the **inlet, outlet and wall as ports**, and **tests,
> against information-matched, projection-matched and learned-potential controls and under a
> pre-declared protocol, what that structure buys**, on this column and on a second open
> dissipative system (a thermocline store).

Each qualifier is load-bearing. Here is what each one excludes:

| qualifier | excludes |
|---|---|
| non-isothermal | the isothermal port-based adsorption-column work of the Lyon group: Baaiu et al. 2008, Peng et al. 2007 and Baaiu et al. 2009 (J. Process Control), all isothermal and isobaric by assumption (full texts read 2026-11-05) |
| learned, latent | the non-ML IPHS, GENERIC and port-based formulations |
| E and S fixed from thermodynamics | among *learned* latent models: tLaSDI and its fracture application, the thermodynamics-aware reduced models, the magnetisation operator, GENERIC-FNO (all learn them); Physics-conforming Latent Twins (prescribed invariants, not a thermodynamic E and S). Projection-based reduced models (Öttinger 2015; Gruber et al. 2023; Chan 2020; Klein et al. 2025) keep the physical potentials but do not learn the dynamics |
| controls that separate constraint, information and projection | arXiv:2606.01366 examined projection against training, without an information control |
| adsorption column with ports | port-metriplectic networks and thermodynamics-informed graph networks (no sorption, E and S learned) |
| tested what structure buys, pre-declared | every sorption surrogate found (unstructured, penalised, or balance-only, none tested this way) |

**What the formulation adds is supporting material, not a second headline.**
1. **Paper 1's non-isothermal column is the exact `c_w = 0` member of a thermodynamically
   consistent family.** This is verified numerically to better than 10⁻⁸ (formulation, Section 6).
   Consistency of heats and heat capacities as a principle is known (Walton and LeVan 2005;
   Schwamberger and Schmidt 2013). The exact reduction for this model is the new part.
2. **Concentration-difference LDF produces entropy whenever the isotherm is increasing.**
   This is a one-line sign argument, presented as an observation that answers the HARMONIA
   claim for single adsorbates, not as a contribution. Non-equilibrium treatments of sorption
   kinetics already use chemical-potential driving forces (arXiv:1204.3841).

## 6. What could still sink the novelty claim

1. **Resolved: Mousa et al. 2025 is no threat.** The Comput. Chem. Eng. article is still
   closed, but the first author's thesis is open on HAL (tel-05363400, 2025-10-10, abstract
   read live 2026-11-03). It lists the article as its own and describes neural networks
   corrected by data reconciliation on mass and energy balances and reaction equilibria,
   for a flash drum, a distillation column and reactors. There is no adsorption and no
   entropy structure.
2. **Resolved: the Lyon papers.** All four were read in full on 2026-11-05 (Section 2). The
   three adsorption papers are isothermal and isobaric by assumption, and the 2009 theory
   paper states that "the thermal domain is not considered", naming the internal-energy
   formulation as the extension. The formulation's novelty stands.
3. **Open: pages refused during the searches.**
   - Döppel, Bracconi and Maestri 2026, ACS Eng. Au (hard atom balances in a catalytic
     reactor surrogate, SEEN-ONLY);
   - Myers 2002 (adsorbed-phase thermodynamics, SEEN-ONLY);
   - the HARMONIA full text.
4. **The sweep is good but not exhaustive.** Semantic Scholar refused requests in every
   sweep (HTTP 429). Journal D logs every search with its hit count, and the two searches
   closest to the gap returned none. A final citation-gate sweep before submission is still
   required.

## 7. How the review changed the design

- **A non-learned competitor is mandatory** (arm C in the pre-registration). The 1-D
  benchmark's core alone reached round-off, and a differentiable mechanistic model already
  reaches the cyclic steady state 20 times faster than a representative MATLAB
  implementation.
- **Convection is a port, the Poisson part is empty, and both are argued** (formulation,
  Section 4.1). Section 4.1 also cites the Lie-Poisson alternative.
- **The latent model is not a pure gradient flow.** The projected port term carries the
  transport (formulation, Section 5.5), because latent gradient flows lose rollouts.
- **Positivity is designed together with conservation:** a log-variable decoder, so the
  projection cannot create negative values.
- **The full state is observed in training.** The design makes no claim about learning from
  outlet-only data (arXiv:2512.16305).
- **The unstructured control is trained on multi-step rollouts,** like every arm, not on
  one-step fits.
- **Every result is reported over 5 seeds with intervals, never best-of-N.**
- **The consistency of heats and heat capacities is cited** to Walton and LeVan and to
  Schwamberger and Schmidt, not claimed.

## 8. AI declaration (for the eventual paper)

"During the preparation of this work the author used a generative AI assistant (Claude,
Anthropic) as a tool to assist with drafting and language editing, with writing and checking
code, and with checking bibliographic records. The author designed the study, made every
scientific decision, reviewed and edited all content, and takes full responsibility for the
content of the published article."
