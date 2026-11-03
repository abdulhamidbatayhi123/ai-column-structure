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

## 1. Structure-preserving learning is mature, and its latent versions all learn the entropy

Building the GENERIC or metriplectic structure into the architecture, rather than adding it
as a loss, is now routine for ODEs, graphs and neural operators.
- **Structure as architecture.** GFINNs satisfy the degeneracy conditions by architecture
  (arXiv:2109.00092). Gruber et al. make the parameterisation scale quadratically in the state
  (arXiv:2405.16305). Port-metriplectic networks extend it to open systems learned by parts
  (arXiv:2211.01873).
- **Function space.** GENERIC-FNO (arXiv:2606.08343) and Poisson-GENERIC neural operators
  (arXiv:2610.05570, 4 October 2026) carry the structure into function space. The second wins
  61 of 72 seed-level comparisons.

Latent (autoencoder) versions exist too:
- the thermodynamics-aware reduced model of Hernández et al. (arXiv:2007.03758);
- tLaSDI (arXiv:2403.05848) and its parametric successor (arXiv:2506.08475);
- a magnetisation-dynamics operator (arXiv:2609.04530).

tLaSDI is the closest architectural template to 3B. It reports extrapolation errors 5 to 8
times lower than its baselines, for example 2.47×10⁻³ against 1.70×10⁻² for a plain network
on Couette flow (journal C, C17). Two caveats travel with that number: it is the best of 10
runs, and an ablation found that one of its losses added nothing.

**In every latent GENERIC or metriplectic model found, the energy and entropy are learned
in latent coordinates without physical meaning.** That now includes a fracture application
of tLaSDI (arXiv:2610.20614). The magnetisation paper says explicitly that its learned
potential is not the Gibbs free energy (journal A, T11). The nearest exception in spirit is
Physics-conforming Latent Twins (Chung, Bu, Verma, arXiv:2606.15053, CMAME 2026): its latent
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
  parameters are learned (arXiv:2211.06130, T17), a lumped thermal model.

None of these is a latent model with E and S fixed from the physics of the decoded fields.

**Transport is the documented weak spot.** GENERIC-FNO loses accuracy "on pure transport"
(T4). Poisson-GENERIC shows that its friction class provably cannot represent transport, so
transport must enter through a Lie-Poisson reversible operator (T5). GENERIC-FNO's latest version (v9, 1 November 2026) also limits its
guarantees to "closed conservative-dissipative dynamics". No fetched paper shows a learned
metriplectic latent model of an advection-dominated *open* system with inflow and outflow
ports and a quantified rollout error. The nearest learned distributed-port model is
conservative and learns its Hamiltonian (Beckers and Colombo, arXiv:2604.04266, boundary
control of port-Hamiltonian PDEs). The nearest is a thermodynamics-informed
MeshGraphNet for open flow around obstacles (Bermejo et al. 2025, T16). Its E and S are
learned, it is full-field, and its abstract does not say how boundary fluxes are treated.

## 2. Structured models of adsorption columns exist, but only isothermal ones

The Lyon LAGEP group built a **port-based distributed model of a multiscale adsorption
column "under isothermal conditions"**, using bond graphs, Dirac structures and the axioms
of irreversible thermodynamics (Baaiu, Couenne, Le Gorrec, Lefèvre, Tayakout, Math. Comput.
Model. Dyn. Syst. 2008, doi:10.1080/13873950701844840; abstract read live). The general
port-based theory of mass transport followed (Baaiu et al. 2009,
doi:10.1080/13873950902808578), and so did structure-preserving discretisation and reduction
applied to adsorption (Peng et al., IEEE CDC 2007, doi:10.1109/cdc.2007.4434663; Baaiu et al.,
J. Process Control 2009, doi:10.1016/j.jprocont.2008.07.002). There is no heat of adsorption,
no thermal domain, no entropy as a state, and no learning. **The full texts were not
reachable**, so their state variables and exact boundary treatment are not verified. That is
the one reading the author must do through library access before the paper is written.

The rest of the non-ML literature (journal B):

- **Convection by an imposed velocity has no settled treatment.**
  - Irreversible port-Hamiltonian reactor models leave convection out on purpose (P6, P8,
    P10).
  - A 2026 extension brings it inside the structure, but only where the velocity is a state
    of the fluid (Ouardi et al., arXiv:2603.11260).
  - Others treat it as a "dissipation port" (a counter-current heat exchanger, P14), as a
    boundary supply in passivity analyses where convection is exactly what breaks automatic
    passivity (P15, P28, P29), or as compartments in series (a heat exchanger, P13).

  3B's choice (formulation, Section 4.1) is the last of these, with a proof of non-negative
  mixing entropy.
- **Uptake as a flux driven by chemical-potential difference** is standard
  non-equilibrium thermodynamics (Santamaria-Holek, Grzywna, Rubi, arXiv:1204.3841). A 2025
  preprint claims standard isotherm-plus-LDF couplings can violate the second law
  (HARMONIA, doi:10.26434/chemrxiv-2025-c860f; read as an abstract record only). Formulation
  Section 4.2 shows that for one adsorbate with an increasing isotherm they cannot.
- **Consistency of heats and heat capacities is known.** Walton and LeVan (Adsorption 2005)
  derive energy balances along consistent thermodynamic paths and tie the adsorbed-phase
  heat capacity to the temperature dependence of the isosteric heat. Schwamberger and Schmidt
  (Ind. Eng. Chem. Res. 2013) show that inconsistent choices leave energy and entropy balance
  residuals. Myers and Monson (Adsorption 2014) argue for absolute adsorption as the
  thermodynamic basis. 3B cites them and does not claim the consistency point as new.
- **Entropy generation in adsorption cycles has been computed only after the fact.** A
  lumped TSA cycle attributes 63.27 % of entropy generation to heat transfer, 53.72 % with
  heat recovery (Li et al., Sci. Rep. 2019). A CFD TSA study does the same spatially (Guo et
  al., Entropy 2019). No model has entropy production as structure.

## 3. Learned sorption models: unstructured, penalised, or balance-constrained

From most to least structure (journal C, Part 1):

- **Hard balances without a second law:** mass- and energy-constrained networks for
  distributed chemical systems (Mukherjee and Bhattacharyya, Ind. Eng. Chem. Res. 2024) and
  hard enthalpy balances in a packed-bed reactor surrogate (Lastrucci et al.,
  arXiv:2501.17782). Neither is an adsorption column, and neither carries entropy.
- **A conservative solver with a learned part:** FINN for groundwater diffusion and sorption
  (Water Resour. Res. 2022), isothermal, learning only the isotherm. Hybrid chromatography
  models screened in AIChE J. 2025.
- **Physics as a loss:** PINN surrogates of adsorption processes (Processes 2025) and of a
  PSA cycle (Green Chem. Eng. 2024).
- **Unstructured:**
  - Staudt et al. 2026: a neural surrogate of binary non-isothermal breakthrough trained on
    1.1 million simulations (Zenodo 22556794).
  - Ceccanti et al. 2026: a DeepONet for cyclic TVSA, which assumes isothermal operation
    (arXiv:2601.09491).
  - Bilal et al. 2026: a non-isothermal chromatography surrogate.
  - An, Wald, Chen, King 2026: a neural surrogate of a temperature-vacuum swing (TVSA)
    direct-air-capture cycle, trained on about 120,000 cycle-resolved simulations of a
    lumped state-transition model (Front. Environ. Eng., doi:10.3389/fenve.2026.1881554).
    It is cycle-level, not a column field, and has no energy or entropy structure.
- **A strong non-learned competitor:** a differentiable mechanistic VPSA model that reaches
  the cyclic steady state about 20 times faster (Glover, Papathanasiou, Pini, Comput. Chem.
  Eng. 2026, doi:10.1016/j.compchemeng.2026.109904).

Differentiable mechanistic simulators keep getting stronger: diff-soup (Degnan-Morgenstern
and Bazant, arXiv:2610.37333) puts a neural closure inside a stiff packed-bed solver on GPU.
That strengthens the case for arm C. No learned sorption model reports zero unphysical
outputs over many cycles. **No learned
adsorption, sorption or chromatography model that is thermodynamically structured by
construction was found** in journals A and C. The searches covered GENERIC, metriplectic,
port-Hamiltonian, Onsager, entropy-stable and "thermodynamically consistent", crossed with
adsorption, sorption, chromatography, desiccant, PSA, TSA, breakthrough and water harvesting.

One possible exception is unverified: Mousa, Negny, Ouaret, Di Pretoro, Montastruc,
"Incorporating Physical Constraints inside Neural Networks … Chemical Engineering Unit
Operations Modeling", Comput. Chem. Eng. 199, 109156 (2025). It cites GFINNs. ScienceDirect
refused both browsers and no repository holds a copy. Its reference list (read live on
Crossref) is mostly data reconciliation for heat exchangers, power plants and refrigeration,
plus PINNs with hard linear constraints, and has no adsorption entry. It is most likely a
balance-constraint paper, but it must be read before the novelty sentence is final.

## 4. What structure buys: the evidence is real but mostly from toy systems

**For:**
- **Conservative structure, 1-D benchmark.** The cleanest numbers come from a 1-D
  drift-diffusion-Poisson benchmark close in form to a column (arXiv:2606.01366). A
  conservative finite-volume model with a learned flux correction reached a rollout error of
  7×10⁻⁹, against 25 to 67 for unconstrained regressors. Projection after the fact and
  rollout training did not rescue the unconstrained models.
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
  structure-preserving network (arXiv:2512.16305).
- **Hard conservation can break positivity.** A global conservation projection created
  negative densities in ProbConserv (arXiv:2302.11002).
- **Penalties have their own failures:** ill-conditioning, and ensemble collapse under
  shared constraints (journal C, C34 and C37).
- **Reporting is often best-of-N runs** (journal C, C31).

## 5. The exact gap

> No published work, as of 2026-11-03 within the sources searched (journals A to D), learns a reduced (latent)
> model of a **non-isothermal adsorption column** whose **energy and entropy are fixed from
> the adsorbent's and the gas's thermodynamics** (not learned), with the **inlet, outlet and
> wall as ports**, and **tests, against the same model without structure and under a
> pre-declared protocol, what that structure buys** in long multi-cycle rollouts, data
> efficiency and admissibility.

Each qualifier is load-bearing. Here is what each one excludes:

| qualifier | excludes |
|---|---|
| non-isothermal | the isothermal port-based column models of Baaiu et al. 2008 and 2009 |
| learned, latent | the non-ML IPHS, GENERIC and port-based formulations |
| E and S fixed from thermodynamics | tLaSDI and its fracture application, the thermodynamics-aware reduced models, the magnetisation operator, GENERIC-FNO (all learn them); Physics-conforming Latent Twins (prescribed invariants, not a thermodynamic E and S) |
| adsorption column with ports | port-metriplectic networks and thermodynamics-informed graph networks (no sorption, E and S learned) |
| tested what structure buys, pre-declared | every sorption surrogate found (unstructured, penalised, or balance-only, none tested this way) |

**What the formulation adds is supporting material, not a second headline.**
1. **Paper 1's non-isothermal column is the exact `c_w = 0` member of a thermodynamically
   consistent family.** This is verified numerically to 10⁻⁹ (formulation, Section 6).
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
2. **Open: the full texts of Baaiu et al. 2008 and 2009.** HAL holds no file and all three
   records are closed. If they contain a non-isothermal extension that the abstracts do not
   mention, the formulation is no longer new. The learned-model claim would still stand.
   This needs library access (Taylor & Francis and Elsevier).
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
  reaches the cyclic steady state 20 times faster.
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
