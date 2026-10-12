# Literature journal A: ML of metriplectic / GENERIC / port-metriplectic / port-Hamiltonian / Onsager models

Researcher lens for Paper 3B (structure-preserving learned latent model of a non-isothermal 1-D adsorption column, E and S given).
Session date: 2026-10-12. Rule: an entry counts only if its live page was fetched this session. "SEEN-ONLY" = lead from an API listing/search snippet, not fetched.

## Step 0: citation-API sweeps (fetched live, Semantic Scholar Graph API)

- api.semanticscholar.org/graph/v1/paper/arXiv:2211.01873/citations : 27 citers
- .../arXiv:2405.16305/citations : 23 citers
- .../arXiv:2109.00092/citations : 61 citers 2024-2026
- .../arXiv:2007.03758/citations : 45 citers 2024-2026

Leads (SEEN-ONLY until fetched below): tLaSDI 2403.05848; Thermo-consistent latent dynamics ID parametric 2506.08475; GENERIC-FNO 2606.08343; Poisson-GENERIC neural operators 2610.05570; WGFINNs 2604.02601; Comparative investigation of thermodynamic structure-informed NNs 2603.26803; N-GENNs 2605.09058; "Incorporating Physical Constraints inside NNs ... Chemical Engineering Unit Operations" (Comput. Chem. Eng. 2025, 10.1016/j.compchemeng.2025.109156); Stable port-Hamiltonian NNs 2502.02480; Structure-preserving UQ for GENERIC 2608.12624; GNNs informed locally by thermodynamics (EAAI 2025); MeshGraphNets informed locally by thermodynamics (AMSES 2025); Can transformers overcome lack of data ... history-dependent flows 2512.16305; Metriplector 2603.29496; single vs double generator 2404.01060; NN-OpInf 2603.08488; Identifiable learning of dissipative dynamics 2510.24160; GIMLET 2512.19936; THINNs 2509.19467; Energy-based conservative-dissipative latent operator (magnetization) 2609.04530; Generalized metriplectic via free energy 2410.06233; GNN zero-shot digital twins 2607.20535; physics-informed LaSDI NLTE kinetics 2604.16664; "Choosing the State Variables Matters: Testing GENERIC structure on a simplified linear hydrodynamic model" (ICOPS 2026).

## Fetched sources

### T1. Incorporating Physical Constraints inside Neural Networks to Improve their Accuracy and Physical Reliability for Chemical Engineering Unit Operations Modeling
- Authors: Jana Mousa, Stéphane Negny, Rachid Ouaret, Alessandro Di Pretoro, Ludovic Montastruc. 2025. Computers & Chemical Engineering 199, 109156. DOI 10.1016/j.compchemeng.2025.109156
- Fetched: api.semanticscholar.org (DOI record) and api.openalex.org/works/doi:10.1016/j.compchemeng.2025.109156. METADATA ONLY: no abstract in either record; ScienceDirect returned 403; HAL search found nothing.
- Supports: exists; cites GFINNs (2109.00092) and 2007.03758; OpenAlex keywords "physics-informed neural networks, physical constraints, data reconciliation, model reliability". Unit operations NOT identified from fetched text. Not shown to be GENERIC/port-Hamiltonian. Quote: (title only) "Chemical Engineering Unit Operations Modeling".
- Relevance: the only chemical-engineering unit-operation citer of the GENERIC-NN line found. Could not verify whether it covers adsorption/columns. ACTION for author: read full text via library access before submission.

### T2. tLaSDI: Thermodynamics-informed latent space dynamics identification
- Jun Sur Richard Park, Siu Wun Cheung, Youngsoo Choi, Yeonjong Shin. arXiv 2403.05848 (v1 9 Mar 2024, v2 22 Mar 2024). No journal ref on page.
- Fetched: https://arxiv.org/abs/2403.05848 and https://arxiv.org/html/2403.05848
- Supports: autoencoder + GFINN latent dynamics (GENERIC), trained jointly; error estimate gives an AE-Jacobian loss. Systems: Couette flow of Oldroyd-B (400-dim, latent 8), two gas containers (400-dim, latent 30, 100 trajectories), 1D parametric inviscid Burgers (N=201, latent 10, 25 greedy training params), 1D parametric heat.
- E,S: LEARNED (E_NN, S_NN, L_NN, M_NN). Quote: "yield energy and entropy functions in the latent space".
- Numbers: Couette extrapolation error 2.47e-3 vs TA-ROM 1.41e-2 vs vanilla FNN 1.70e-2; gas containers 3.96e-3 vs 1.31e-2 vs 2.01e-2; Burgers worst-case max rel. error 1.5% vs gLaSDI 6.0%.
- Weak negative: in gas-container ablation, adding the model loss on top of Jacobian loss gave no accuracy gain (5.52e-3 both); gains attributed to the Jacobian loss.
- No advection-dominated open system; no ports; latent E/S have no physical identity.

### T3. Thermodynamically Consistent Latent Dynamics Identification for Parametric Systems
- Xiaolong He, Yeonjong Shin, Anthony Gruber, Sohyeon Jung, Kookjin Lee, Youngsoo Choi. arXiv 2506.08475 (10 Jun 2025). Journal ref on page: TMLR 2026.
- Fetched: https://arxiv.org/abs/2506.08475
- Supports: parametric tLaSDI = autoencoder + parametric GFINNs (pGFINNs) + greedy residual-based active learning. Systems: Burgers, 1D/1V Vlasov-Poisson. Numbers in abstract: up to 3,528x speed-up, 1-3% relative errors, 50-90% training-cost and 57-61% inference-cost reduction.
- E,S: not stated on abstract page (pGFINN lineage implies learned). Quote: "reveal the underlying thermodynamic behavior of the system".
- No open/inflow-outflow system.

### T4. GENERIC-FNO: Embedding Energy Conservation and Entropy Production into Fourier Neural Operators
- Jason Sulskis, Sathya Ravi. arXiv 2606.08343 (v1 6 Jun 2026, v7 1 Oct 2026). "Under review at TMLR".
- Fetched: https://arxiv.org/abs/2606.08343
- Supports: metriplectic structure in function space for FNO/DeepONet; degeneracy enforced by construction (Fourier multipliers, rank-one projections); identities to ~1e-13; four canonical scalar PDEs; Jacobi not enforced; decomposition non-unique, gauge-invariant dissipation diagnostic proposed. ~half parameters, 4-10x compute.
- E,S: LEARNED ("learns the energy and entropy functionals as neural operators").
- Rollout: "over 200-step rollouts, where every unconstrained model we test diverges or collapses, GENERIC-FNO stays bounded".
- NEGATIVE (key for 3B): "while losing accuracy on pure transport and on the smallest 1D backbone."

### T5. Poisson-GENERIC Neural Operators: Exact Metriplectic Structure in Function Space via Casimir Entropies
- Jason Sulskis, Sathya Ravi. arXiv 2610.05570 (v1 4 Oct 2026). Preprint, under review.
- Fetched: https://arxiv.org/abs/2610.05570
- Supports: reversible operator as Lie-Poisson on augmented state (u,s), total entropy a Casimir, so degeneracy without projection; energy = fixed mechanical quadratic + learned gauge-free potential + convex internal energy; friction M = A A^T; discrete-gradient integrator. PDEs: heat, Burgers, advection, damped waves, 1D/2D; FNO, Transolver, CNO backbones. Wins 61/72 seed-level comparisons; dissipation rate within 13% on heat and Burgers.
- E,S: partly fixed (structural energy terms fixed, part learned); a learned-entropy variant is an ablation.
- NEGATIVE/structural (key for 3B): friction structure "permits diffusion and damping while provably excluding transport"; "a learned-entropy ablation injects energy on every reversible-irreversible problem"; constant-L (exact Jacobi) ablation costs accuracy on Burgers.
- Direct support for 3B's choice to FIX entropy rather than learn it; also flags that transport must live in L, not M.

### T6. A Comparative Investigation of Thermodynamic Structure-Informed Neural Networks
- Guojie Li, Liu Hong. arXiv 2603.26803 (26 Mar 2026). No journal ref.
- Fetched: https://arxiv.org/abs/2603.26803
- Supports: systematic comparison of Newtonian/Lagrangian/Hamiltonian (conservative) and Onsager variational principle / extended irreversible thermodynamics (dissipative) NNs on ODE/PDE benchmarks for accuracy, consistency, noise robustness, interpretability. Quote: "structure-preserving formulation significantly enhances parameter identification". Newtonian-residual PINNs reconstruct states but do not reliably recover thermodynamic quantities.
- No numbers on abstract page; GENERIC/port-Hamiltonian not in the comparison.

### T7. WGFINNs: Weak formulation-based GENERIC formalism informed neural networks
- Jun Sur Richard Park, Auroni Huque Hashim, Siu Wun Cheung, Youngsoo Choi, Yeonjong Shin. arXiv 2604.02601 (v1 3 Apr 2026, v2 7 Apr 2026).
- Fetched: https://arxiv.org/abs/2604.02601
- Supports: weak-form loss for GFINNs for noise robustness; state-wise weighted loss + residual attention for scale imbalance. Quote: "the strong-form estimator diverges as the time step decreases in the presence of noise". Systems and E/S status not on abstract page.
- Relevance: scale imbalance across state variables (concentration vs temperature vs loading in a column) is a 3B issue too.

### T8. Nonlinear GENERIC-Embedded Neural Networks (N-GENNs): Learning GENERIC dynamics with non-quadratic dissipation potentials
- Vojtěch Votruba, Zequn He, Weilun Qiu, Celia Reina, Michal Pavelka. arXiv 2605.09058 (v1 9 May 2026, v2 17 Aug 2026). S2 lists DOI 10.1016/j.cma.2026.119277 (CMAME); not shown on arXiv page.
- Fetched: https://arxiv.org/abs/2605.09058
- Supports: nonlinear GENERIC with convex (non-quadratic) dissipation potentials, enforced by reparametrisation of reversible operator and dissipation potential. Systems: harmonic oscillator + heat bath, idealized chemical motor, 1D Perzyna viscoplastic model. Quote: "Thermodynamic structure is strongly enforced by construction". E/S status and numbers not on abstract page.
- Relevance: non-quadratic dissipation is the natural home for LDF/Arrhenius-type kinetics far from equilibrium; 3B's LDF uptake is linear in driving force so quadratic metric suffices, but N-GENN is the prior art if nonlinear kinetics are added.

### T9. Stable Port-Hamiltonian Neural Networks
- Fabian J. Roth, Dominik K. Klein, Maximilian Kannapinn, Jan Peters, Oliver Weeger. arXiv 2502.02480 (v1 4 Feb 2025, v2 5 Nov 2025).
- Fetched: https://arxiv.org/abs/2502.02480
- Supports: pHNN architecture guaranteeing global Lyapunov stability; aimed at surrogates for multi-physics simulation data. Quote: "strong inductive biases facilitate robust learning of stable dynamics from sparse data". No numbers or system names on the abstract page; Hamiltonian learned (implied by "learning ... from data", not explicitly stated).

### T10. Structure-preserving uncertainty quantification for GENERIC dynamics
- Zequn He, Celia Reina. arXiv 2608.12624 (12 Aug 2026).
- Fetched: https://arxiv.org/abs/2608.12624
- Supports: S-PENNs: epinets attached to constrained components of a pretrained GENERIC model so sampled rollouts respect 1st/2nd law; split conformal calibration; ~1-3 orders of magnitude cheaper than deep ensembles. Systems: oscillator + heat bath, chemical motor, 1D viscoplastic PDE. Quote: "every sampled realization remains physically admissible by construction".
- Relevance: if 3B reports UQ, this is the structure-compatible prior art.

### T11. An Energy-Based Conservative-Dissipative Latent Neural Evolution Operator for Magnetization Dynamics
- Sebastian Schaffer, Lukas Exl. arXiv 2609.04530 (3 Sep 2026).
- Fetched: https://arxiv.org/abs/2609.04530
- Supports: conv. autoencoder + latent neural ODE = antisymmetric operator + symmetric PSD dissipative operator acting on gradient of a LEARNED scalar potential; trained on short windows; muMAG Standard Problem 4. Antisymmetric-dissipative models much more accurate than dissipative-only on uninterrupted rollouts (dissipative-only matches on short windows). Quote: "decreases monotonically along autonomous continuous-time solutions".
- E: LEARNED, in non-unique latent coordinates, explicitly NOT identified with the physical Gibbs free energy; no energy labels. No numbers on abstract page.
- Relevance to 3B: closest recent latent-AE + conservative/dissipative split; contrast = 3B fixes E,S from physics. Also evidence that the reversible (antisymmetric) part matters for long rollouts.

### T12. Thermodynamics-informed super-resolution of scarce temporal dynamics data
- Carlos Bermejo-Barbanoj, Beatriz Moya, Alberto Badías, Francisco Chinesta, Elías Cueto. arXiv 2402.17506 (v1 27 Feb 2024, v2 3 Jul 2024); related DOI 10.1016/j.cma.2024.117210.
- Fetched: https://arxiv.org/abs/2402.17506
- Supports: adversarial autoencoder to latent space + structure-preserving (metriplectic) latent dynamics; flow over a cylinder (two fluids). Quote: "It learns the metriplectic-structure of the system and applies a physical bias". E,S learned. No numbers on abstract page.
- Relevance: a latent GENERIC model for an open-boundary (inflow/outflow) flow, but treated as closed in latent space; no port terms stated.

### T13. Metriplectic Conditional Flow Matching for Dissipative Dynamics
- Ali Baheri, Lars Lindemann. arXiv 2509.19526 (v1 23 Sep 2025, v2 11 Jun 2026).
- Fetched: https://arxiv.org/abs/2509.19526
- Supports: conservative-dissipative split in vector field and sampler; trained on short transitions (no rollout adjoints); Strang-prox integrator gives discrete energy decay; optional projection "when a trusted energy is available". Quote: "Neural surrogates often inject energy and destabilize long-horizon rollouts". Controlled mechanical benchmark only; qualitative ("markedly fewer" energy-increase events).
- Relevance: supports 3B's argument that a GIVEN (trusted) energy enables strict guarantees.

### T14. Data-driven particle dynamics: Structure-preserving coarse-graining for emergent behavior in non-equilibrium systems
- Quercus Hernandez, Max Win, Thomas C. O'Connor, Paulo E. Arratia, Nathaniel Trask. arXiv 2508.12569 (v1 18 Aug 2025, v3 29 Dec 2025).
- Fetched: https://arxiv.org/abs/2508.12569
- Supports: metriplectic coarse-graining of particle trajectories with discrete 1st/2nd laws, momentum conservation, fluctuation-dissipation; self-supervised identification of unlabeled entropic variables; star polymers, colloidal suspensions. Quote: "the framework guarantees discrete notions of the first and second laws of thermodynamics". No numbers on abstract page.

### T15. Graph neural networks informed locally by thermodynamics
- Alicia Tierz, Iciar Alfaro, David González, Francisco Chinesta, Elías Cueto. arXiv 2405.13093 (v1 21 May 2024, v2 20 Jan 2025); published Eng. Appl. Artif. Intell. 2025 (DOI 10.1016/j.engappai.2025.110108 per S2 record).
- Fetched: https://arxiv.org/html/2405.13093v1, export.arxiv.org API record, S2 DOI record
- Supports: local (node-wise) metriplectic biases avoiding global L/M assembly. Examples: 3D and 2D viscoelastic beams, SPH sloshing in a glass (closed). E,S gradients LEARNED per node. Sloshing relative RMSE: position 0.0147, velocity 0.118, energy 0.00056; ~42 / ~30 training sims for beams.
- Mild negative: 2D beam "the comparative improvement with respect to vanilla GNNs is not as substantial". Quote: "still enhances the predictions substantially". No inflow/outflow.

### T16. Meshgraphnets informed locally by thermodynamics for the simulation of flows around arbitrarily shaped objects
- C. Bermejo, Alberto Badías, David González, Elías Cueto. 2025. Advanced Modeling and Simulation in Engineering Sciences. DOI 10.1186/s40323-025-00311-8.
- Fetched: api.semanticscholar.org DOI record (abstract). Springer and Zaguan pages blocked (redirect to login / 403).
- Supports: MeshGraphNet with GENERIC decoders (local E,S gradients + L,M assembled per node); incompressible flow around obstacles incl. unseen shapes; accurate long-term predictions and large speedups (no numbers in fetched abstract). Quote: "thermodynamics-informed graph neural network framework for learning the time evolution of complex physical systems".
- Relevance: an advection-dominated OPEN flow (inflow/outflow) learned with GENERIC structure, E,S learned; fetched text does not say how the boundary energy/entropy fluxes are handled. Could not verify port terms.

### T17. Physically Consistent Neural ODEs for Learning Multi-Physics Systems
- Muhammad Zakwan, Loris Di Natale, Bratislav Svetozarevic, Philipp Heer, Colin N. Jones, Giancarlo Ferrari Trecate. arXiv 2211.06130 (11 Nov 2022; submitted to IFAC 2023).
- Fetched: https://arxiv.org/abs/2211.06130
- Supports: Irreversible port-Hamiltonian Systems (IPHS) + neural ODEs learn system PARAMETERS from data; thermodynamic consistency by design; prior knowledge built into training. Systems: building thermodynamics from real measurements; simulated gas-piston. Quote: "IPHS models are consistent with the first and second principles of thermodynamics by design".
- E,S: structure supplied by physics (IPHS), parameters learned; which functions are fixed is not stated on abstract page. No numbers on abstract page.
- Relevance to 3B: the closest "structure and thermodynamic potentials from physics, only closures/parameters learned" precedent in a thermal (heat-transfer) engineering setting; lumped ODE, not a transport PDE; no adsorption.

### T18. Structure- and Stability-Preserving Learning of Port-Hamiltonian Systems
- Binh Nguyen, Nam T. Nguyen, Truong X. Nghiem. arXiv 2604.13297 (14 Apr 2026).
- Fetched: https://arxiv.org/abs/2604.13297
- Supports: drops convexity constraint on neural Hamiltonian; uses known stable equilibria to preserve multiple isolated equilibria. Hamiltonian LEARNED. Quote: "relaxes the convexity constraint commonly imposed by neural network-based Hamiltonian approximations". No numbers on page.

### T19. Data-driven identification of latent port-Hamiltonian systems
- Johannes Rettberg, Jonas Kneifl, Julius Herb, Patrick Buchfink, Jörg Fehr, Bernard Haasdonk. arXiv 2408.08185 (v1 15 Aug 2024, v2 16 Aug 2024).
- Fetched: https://arxiv.org/abs/2408.08185
- Supports: autoencoder to latent space + LINEAR port-Hamiltonian latent system (Cholesky-parameterized), learned jointly. Systems: parametric mass-spring-damper, nonlinear pendulum, high-dimensional thermoelastic disc brake. Quote: "guaranteeing the useful system theoretical properties of passivity and stability". Hamiltonian LEARNED. No numbers on page.
- Relevance: the latent-port-Hamiltonian-with-inputs prior art (ports = inputs/outputs), but with learned quadratic energy and no entropy.

### T20. Identify then Realize: Contrastive Learning of Latent Port-Hamiltonian Dynamics from Partial Observations
- Peilun Li, Kaiyuan Tan, Daniel Moyer, Thomas Beckers. arXiv 2605.16682 (v1 15 May 2026, v2 1 Oct 2026).
- Fetched: https://arxiv.org/abs/2605.16682
- Supports: CIPHER, contrastive encoder + invertible transformation + latent pH dynamics; identifiability up to diffeomorphism; ten clean/noisy datasets; long-horizon forecasting competitive or better; gains from pH parameterization. Hamiltonian LEARNED. Quote: "identify-then-realize framework for learning latent port-Hamiltonian models of conservative and dissipative systems" (ODE systems).

### T21. Irreversible Port-Hamiltonian Formulations for 1-Dimensional fluid systems  (NOT ML; structural prior for 3B ports)
- Ahlam Ouardi, Arijit Sarkar, Hector Ramirez, Yann Le Gorrec. arXiv 2603.11260 (11 Mar 2026).
- Fetched: https://arxiv.org/abs/2603.11260
- Supports: IPHS extended to non-isentropic viscous fluids in Eulerian form; convective transport handled by modified differential operators; extended boundary-port parametrisation satisfying 1st and 2nd laws -> boundary-controlled IPHS. Quote: "convective transport can be consistently encompassed by the framework". No learning.
- Relevance: the most recent first-principles template for treating ADVECTION + inlet/outlet ports thermodynamically in 1-D; 3B should cite it (and the 2021 boundary-controlled IPHS, arXiv 2104.13459, SEEN-ONLY) for its port construction.
- Companion SEEN-ONLY (API listing): 2603.11265 "Conduction-Diffusion in N-Dimensional settings as irreversible port-Hamiltonian systems" (Mora, Le Gorrec, Ramirez, Matignon, 2026).


### T22. GIMLET: Generalizable and Interpretable Model Learning through Embedded Thermodynamics
- Suguru Shiratori, Elham Kiyani, Khemraj Shukla, George Em Karniadakis. arXiv 2512.19936 (v1 22 Dec 2025, v2 30 Dec 2025).
- Fetched: https://arxiv.org/abs/2512.19936
- Supports: known terms (time derivative, CONVECTION, pressure gradient) are PRESCRIBED; unknown constitutive closures = functional derivatives of NN free-energy and NN dissipation functionals (monotone free-energy decay, non-negative entropy production). Systems: viscous Burgers, Kuramoto-Sivashinsky, incompressible NS (Newtonian, non-Newtonian), scalar transport. Quote: "avoids reliance on a predefined library of candidate functions".
- E/S: free energy and dissipation LEARNED; advection given. No numbers on abstract page.
- Relevance to 3B: closest "keep the known transport operator, learn only a thermodynamically admissible closure" design for transport PDEs. Difference: 3B fixes E and S and learns in a latent space; GIMLET learns potentials in full field space.

### T23. Flexible and Stable Dynamics Discovery with Onsager's Variational Principle
- Anthony Gruber, Ritoban Roy-Chowdhury, Irina Tezaur, Nathan M. Urban. arXiv 2606.24106 (23 Jun 2026).
- Fetched: https://arxiv.org/abs/2606.24106
- Supports: variational OVP time discretisation (Rayleighian minimisation), unconditionally energy-stable; learns uncertain terms of free energy and dissipation potential (polynomials, shallow NNs, spectral kernels), incl. nonstandard boundary conditions; Allen-Cahn, Fokker-Planck, Cahn-Hilliard. Quote: "provably energy-stable under arbitrarily long rollouts". Partially-known functional, learn only the uncertain parts (closest in spirit to "E given"). No numbers on page. Gradient flows only (no transport/advection).

### T24. Can Transformers overcome the lack of data in the simulation of history-dependent flows?
- P. Urdeitx, I. Alfaro, D. Gonzalez, F. Chinesta, E. Cueto. arXiv 2512.16305 (18 Dec 2025); S2 lists J. Non-Newtonian Fluid Mech. 2026 (10.1016/j.jnnfm.2026.105601).
- Fetched: https://arxiv.org/abs/2512.16305
- Supports: benchmarks cylinder flow, Oldroyd-B Couette, FENE. NEGATIVE RESULT for structure: with missing (hidden) state variables "the Transformer outperforms a thermodynamically consistent, structure-preserving neural network with metriplectic bias"; the metriplectic model wins only when the full state is known.
- Relevance to 3B: a column observed only at the outlet has hidden states (loading, internal T). Structure helps when the state is complete; 3B's autoencoder trained on full simulated fields is in the favourable regime, but any claim about learning from outlet data alone must face this result.

### T25. Learning Thermodynamically Stable and Galilean Invariant Partial Differential Equations for Non-equilibrium Flows
- Juntao Huang, Zhiting Ma, Yizhou Zhou, Wen-An Yong. arXiv 2009.13415 (v1 28 Sep 2020, v2 25 Nov 2020); related DOI 10.1515/jnet-2021-0008.
- Fetched: https://arxiv.org/abs/2009.13415
- Supports: conservation-dissipation formalism (CDF) gives learned hyperbolic balance laws, NN-parameterised, thermodynamically stable; trained on smooth kinetic data, generalise to discontinuous data and Sod shock tube over a wide Knudsen range. Quote: "satisfy the conservation-dissipation principle automatically". Older, but it is the transport/hyperbolic-relaxation branch of structure-preserving learning; LDF uptake is a relaxation source term of exactly this kind.

### T26. Model reduction of port-Hamiltonian systems via neural networks
- Silke Glas, Alexander Heinlein, Harald Monsuur, Hongliang Mu. arXiv 2608.30788 (31 Aug 2026).
- Fetched: https://arxiv.org/abs/2608.30788
- Supports: manifold Galerkin projection, then NN maps reduced state to reduced J (skew) and R (PSD) so reduced model stays pH; nonlinear mass-spring-damper; "significant" speed-up at comparable accuracy. Quote: "achieves a significant computational speed-up compared to the original". Hamiltonian handling not stated on page.
- Relevance: projection-plus-learned hybrid where the energy comes from the full model (closest pH analogue to "E given, operators learned"), but no transport, no entropy.

### T27. Petrov-Galerkin operator inference with application to stability-encouraging identification
- Johannes Rettberg, Jonas Nicodemus, Harsh Sharma, Boris Kramer, Jörg Fehr, Benjamin Unger. arXiv 2610.01295 (1 Oct 2026).
- Fetched: https://arxiv.org/abs/2610.01295
- Supports: Petrov-Galerkin OpInf; error bounds intrusive vs non-intrusive; convex program enforcing pH structure on inferred operators; CD player, atmospheric model, mass-spring-damper, poroelasticity. Quote: "explicitly enforces the port-Hamiltonian structure on the inferred operators".

### T28. SMORE: Stability-Promoting Mesh-Agnostic Model Reduction for Time-Dependent PDEs
- Yangyuan Li, Weichao Li, Shaowu Pan. arXiv 2609.33205 (27 Sep 2026).
- Fetched: https://arxiv.org/abs/2609.33205
- Supports: mesh-agnostic ROM with linear / linear-quadratic latent dynamics and Lyapunov-guided stability regularisation; wave, Navier-Stokes, shallow water; "competitive accuracy at comparable parameter budgets" vs DINo, FNO, CNO, Transolver. Quote: "Its latent dynamics are trained with Lyapunov-guided stability regularization". Non-thermodynamic stability baseline class.

### T29. NN-OpInf: an operator inference approach using structure-preserving composable neural networks
- Eric Parish, Anthony Gruber, Patrick Blonigan, Irina Tezaur. arXiv 2603.08488 (9 Mar 2026).
- Fetched: https://arxiv.org/abs/2603.08488
- Supports: composable NN operators with skew-symmetry, (semi-)positive definiteness, gradient structure; better accuracy/stability/robustness than polynomial OpInf, at higher, non-convex training cost. Quote: "a structure-preserving, composable, and minimally restrictive operator inference framework". No numbers on page.

### T30. Physics-Informed Latent Space Dynamics Identification for Time-Dependent NLTE Atomic Kinetics
- Jeongwoo Nam, William Anderson, Youngsoo Choi, Hai P. Le, Mark E. Foord, Byoung Ick Cho, Haewon Jeong, Min Sang Cho. arXiv 2604.16664 (17 Apr 2026). LLNL-JRNL-2018060.
- Fetched: https://arxiv.org/abs/2604.16664
- Supports: pLaSDI with physics-informed losses for macroscopic consistency, stability and correct steady state over long integrations; tin NLTE kinetics; errors < 2%, speedups ~5e4-1e5. Quote: "physics-informed loss terms that enforce macroscopic consistency". Soft constraints, not GENERIC; a kinetics latent-model contrast (relaxation to the correct equilibrium is a 3B check too).

### T31. Hypothesis-driven construction of mesoscopic dynamics
- Zhuoyuan Li, Aiqing Zhu, Qianxiao Li. arXiv 2605.16211 (15 May 2026).
- Fetched: https://arxiv.org/abs/2605.16211
- Supports: hypothesis class from a generalised Onsager principle with a priori well-posedness, asymptotic stability, unique-factorisation identifiability, discrete energy dissipation; continuum PDE check + microscopic chain data. Quote: "learning mesoscopic dynamics within a mathematically constrained hypothesis class". No numbers on page.

### T32. Identifiable learning of dissipative dynamics
- Aiqing Zhu, Beatrice W. Soh, Grigorios A. Pavliotis, Qianxiao Li. arXiv 2510.24160 (v1 28 Oct 2025, v2 25 Nov 2025).
- Fetched: https://arxiv.org/abs/2510.24160
- Supports: learns energy landscape + reversible/irreversible split + entropy production from stochastic trajectories, with uniqueness/identifiability; polymer stretching in elongational flow, SGLD. Energy LEARNED. Phrase from abstract: "unique energy landscape".
- Relevance: identifiability is the flip side of the non-uniqueness flagged by GENERIC-FNO (T4); 3B sidesteps it by fixing E and S.

### Also fetched, low relevance (logged for completeness)
- 2603.29496 "Metriplector: From Field Theory to Neural Architecture" (Dan Oprisa, Peter Toth, 2026; fetched arxiv.org/abs/2603.29496): metriplectic dynamics as a generic NN layer for vision/language/control; not a physics surrogate. Quote: "the dynamics of that system is the computation".
- 2509.19467 "THINNs: Thermodynamically Informed Neural Networks" (Javier Castro, Benjamin Gess, 2025; fetched): PINN penalty from large-deviation structure; not GENERIC.
- 2404.01060 "A comparison of Single- and Double-generator formalisms for Thermodynamics-Informed Neural Networks" (Urdeitx, Alfaro, González, Chinesta, Cueto, 2024; fetched): compares one- vs two-generator formalisms on rheology benchmarks; abstract gives no winner. Quote: "universally valid, regardless of the level of abstraction".

## Adsorption / sorption / chromatography / packed-bed / reactor sweep (most important check)

Searches run: OpenAlex "port-Hamiltonian adsorption", "GENERIC neural network adsorption", "thermodynamics-informed ... chromatography / packed bed / fixed bed"; arXiv API (metriplectic OR GENERIC OR port-Hamiltonian OR Onsager) AND (adsorption OR chromatography OR packed bed OR fixed bed OR sorption) (892 hits, first ~41 read); arXiv API "irreversible port-Hamiltonian"; three extended web searches. OpenAlex later rate-limited (HTTP 429).
RESULT: NO paper found that applies a learned GENERIC / metriplectic / port-metriplectic / port-Hamiltonian / Onsager model to adsorption, sorption, chromatography, packed beds, porous-media heat and mass transfer, or chemical reactors/columns. Unverified exception: T1 (Mousa et al. 2025, unit operations, full text not reachable). Fetched neighbours below.

### T33. Deep Operator Networks for Surrogate Modeling of Cyclic Adsorption Processes with Varying Initial Conditions
- Beatrice Ceccanti, Mattia Galanti, Ivo Roghair, Martin van Sint Annaland. arXiv 2601.09491 (14 Jan 2026).
- Fetched: https://arxiv.org/abs/2601.09491
- Supports: DeepONet surrogate for TVSA cycle steps trained on heterogeneous initial conditions; tested out of range and on unseen functional forms. Quote: "The governing equations exhibit steep traveling fronts". NO structure/thermodynamic constraint stated. No numbers on page.

### T34. A systematic screening of neural network-based hybrid models of adsorption in chromatography processes
- Jesper Frandsen, Vinícius V. Santana, Peter Jul-Rasmussen, Idelfonso B. R. Nogueira, Jakob K. Huusom, Krist V. Gernaey, Jens Abildskov. AIChE J. 71(12), 2025. DOI 10.1002/aic.70045.
- Fetched: api.openalex.org/works/doi:10.1002/aic.70045 (abstract reconstructed from inverted index).
- Supports: 10 hybrid structures (NN adsorption term inside a mechanistic chromatography transport model); binary non-reactive case mostly fine, four-component reactive case fewer hybrids match the mechanistic model. Phrase: "optimal balance between mechanistic and data-driven components". No thermodynamic structure stated.
- Relevance: the "keep transport mechanistic, learn the sorption term" design is established in chromatography, without thermodynamic guarantees.

### T35. Reproducibility package: A generalized neural surrogate for binary nonisothermal adsorption breakthrough curves
- Eduardo Andrei Staudt, Antonio Augusto Ignacio, Carlos Eduardo Borba, Evandro Alves Nakajima. Zenodo 10.5281/zenodo.22556794, v1 6 Sep 2026; article submitted to Computers & Chemical Engineering.
- Fetched: https://zenodo.org/records/22556794
- Supports: neural surrogate for binary NON-ISOTHERMAL breakthrough curves; FD reference solver (NumPy/JAX), 1.1 M training simulations (not included), 10,000-simulation independent test set, gPROMS cross-check, digitised Huang et al. (2008) data. Quote: "binary nonisothermal adsorption breakthrough curves". No structure / energy / entropy constraint mentioned.
- THREAT LEVEL: medium for the application claim (same physical system class, very recent, huge data budget); none for the structure claim. 3B should cite it and contrast data budget and guarantees.

### T36. Thermodynamically Consistent Incorporation of the Langmuir Adsorption Model into Compressible Fluctuating Hydrodynamics (NOT ML)
- Hyun Tae Jung, Hyungjun Kim, Alejandro L. Garcia, Andrew J. Nonaka, John B. Bell, Ishan Srivastava, Changho Kim. arXiv 2510.15329 (17 Oct 2025); J. Chem. Phys. 164, 094103 (2026).
- Fetched: https://arxiv.org/abs/2510.15329
- Supports: mass-energy update scheme coupling Langmuir adsorption to fluctuating hydrodynamics; reproduces equilibrium statistical mechanics; internal-energy correction needed. Quote: "a gas-solid interfacial system where chemical species undergo reversible adsorption". No GENERIC wording, no learning. Useful physics citation for energy bookkeeping of adsorption heat.

### T37. A Systematic Modeling Framework for Dynamic Simulation of Fixed-Bed Reactors (NOT ML)
- Marcus Johan Schytt, John Bagterp Jørgensen. arXiv 2604.22468 (24 Apr 2026); submitted to Comput. Chem. Eng.
- Fetched: https://arxiv.org/abs/2604.22468
- Supports: modular fixed-bed reactor models with cubic EOS, advective + dispersive transport, internal energy as state variable. Quote: "modular and thermodynamically consistent modeling framework". No learning, no adsorption, no GENERIC stated.
- Relevance: supports choosing internal energy as the energy state in a column model; recent chem-eng precedent.

### T38. Composing Partial Differential Equations with Physics-Aware Neural Networks (FINN)
- Matthias Karlbauer, Timothy Praditia, Sebastian Otte, Sergey Oladyshkin, Wolfgang Nowak, Martin V. Butz. arXiv 2111.11798 (v1 23 Nov 2021, v2 26 May 2022); ICML 2022.
- Fetched: https://arxiv.org/abs/2111.11798
- Supports: finite-volume physics-aware NN for advection-diffusion; Burgers, DIFFUSION-SORPTION, diffusion-reaction, Allen-Cahn; learns the unknown retardation factor (sorption isotherm effect); ~1/10 parameters; beats a calibrated physical model on sparse real diffusion-sorption data. Quote: "outperforms pure machine learning and other state-of-the-art physics-aware models in all cases". No thermodynamic structure.
- Relevance: the strongest "structure helps for sorption transport" precedent, but the structure is numerical (FV fluxes), not thermodynamic.

## Further structure-preserving ML entries

### T39. Neural Operators Meet Energy-based Theory: Operator Learning for Hamiltonian and Dissipative PDEs
- Yusuke Tanaka, Takaharu Yaguchi, Tomoharu Iwata, Naonori Ueda. arXiv 2402.09018 (14 Feb 2024).
- Fetched: https://arxiv.org/abs/2402.09018
- Supports: ENO = neural operator + penalty biasing toward energy conservation/dissipation, energy LEARNED ("the energy functional is modeled by another DNN"); better than DNN baselines, esp. super-resolution. Soft constraint. No numbers on page.

### T40. A Hybridizable Neural Time Integrator for Stable Autoregressive Forecasting
- Brooks Kinch, Xiaozhe Hu, Yilong Huang, Martine Dyring Hansen, Sunniva Meltzer, Nathaniel Donald Hamlin, David Sirajuddin, Eric C. Cyr, Nathaniel Trask. arXiv 2604.21101 (v1 22 Apr 2026, v2 28 Apr 2026).
- Fetched: https://arxiv.org/abs/2604.21101
- Supports: transformer inside shooting-based mixed FE scheme; discrete energy preservation and uniform training-gradient bounds; 65x fewer parameters than foundation models; fusion component mini-foundation model from 12 simulations, 9,000x speedup over PIC. Quote: "provably avoiding the exploding gradient problem". Data-efficiency number (12 sims) is a useful comparator.

### T41. Structure-Preserving Learning Improves Geometry Generalization in Neural PDEs
- Benjamin D. Shaffer, Shawn Koohy, Brooks Kinch, M. Ani Hsieh, Nathaniel Trask. arXiv 2602.02788 (v1 2 Feb 2026, v2 8 Jun 2026).
- Fetched: https://arxiv.org/abs/2602.02788
- Supports: Geo-NeW (neural Whitney forms + FEEC, transformer geometry encoding), steady-state PDEs, exact conservation; OOD geometry gains. Quote: "provides a significant improvement over conventional baselines on out-of-distribution geometries". Steady only.

### T42. Neural Constitutive Learning for Generalized Reaction-Diffusion Systems
- Shang-Ke Chen, Yu-Peng Wang, Shih-Hsuan Hung, Wei-Fang Sun, Chao-Shun Zhan, Simon See, Min-Jhe Lu. arXiv 2609.37113 (29 Sep 2026). Cites Chang et al. Onsager operator learning.
- Fetched: https://arxiv.org/abs/2609.37113
- Supports: learn constitutive responses (mobility, thermodynamic driving force, reaction rates) as functions of current density with a shared Mass-Compression-Transport integrator; 7 systems; rollout relative L2 errors 1e-4 to 1e-2; trajectory-free training when laws are known. Quote: "constitutive responses as an effective learning target for a shared neural PDE framework". Thermodynamic framework not named on page.
- Relevance: "learn the closure, keep the integrator/balance laws" for reaction-transport; nearest in spirit to learning LDF/isotherm closures inside fixed balances.

### T43. Meta-learning Structure-Preserving Dynamics
- Cheng Jing, Uvini Balasuriya Mudiyanselage, Woojin Cho, Minju Jo, Anthony Gruber, Kookjin Lee. arXiv 2508.11205 (v1 15 Aug 2025, v2 2 May 2026); accepted ICML 2026 per comments.
- Fetched: https://arxiv.org/abs/2508.11205
- Supports: modulation-based meta-learning for Hamiltonian (conservative) structure-preserving models across parameter space; few-shot adaptation. Quote: "modulation-based meta-learning enables accurate few-shot adaptation". Conservative only.

## SEEN-ONLY leads (not fetched; listing/snippet only)
- 2607.10439 metriplectic port-Hamiltonian NN for EEG cortical digital twin (2026); 2607.20535 GNN zero-shot digital twins (2026); 2412.12034 TIGNN digital human twins (2024); 2603.02645 hierarchy of thermodynamics learning frameworks for inelastic constitutive modeling (2026); 2410.06233 generalized metriplectic via free energy, bilevel convex ID (2024); 2610.01356 pHNN with multiple asymptotically stable equilibria (2026); 2603.10078 stochastic pHNN (2026); 2502.14432 pHNN with output-error noise models (Automatica 2026); 2603.11265 conduction-diffusion as IPHS (2026); 2104.13459 boundary-controlled IPHS (2021); 2512.05168 comparison of geometric frameworks for dissipative evolution (2025); "Choosing the State Variables Matters: Testing GENERIC structure on a simplified linear hydrodynamic model" (ICOPS 2026, IEEE DOI 10.1109/ICOPS53334.2026.11660322; web search could not locate it); "A Port Hamiltonian approach to dynamical chemical process systems network modeling and analysis" (Chem. Eng. Sci. 2022, S0009250922004912); "Port-Hamiltonian modeling of district heating networks" (arXiv 1908.11226); 2606.11650 structure-preserving neural surrogates with tractable UQ; 2606.00937 cellular sheaf neural operators; 2605.08436 meshfree exterior calculus; 1907.00195 generic breakthrough curve model.

## Synthesis

### What is now established (2024-2026)
1. Hard-wired GENERIC/metriplectic structure in learned models is mature for ODEs, graphs and neural operators: exact degeneracy to ~1e-13 (T4), bounded 200-step rollouts where unconstrained operators diverge (T4), UQ that stays admissible (T10), non-quadratic dissipation (T8), weak-form noise robustness (T7), local/graph versions (T15, T16), parametric latent versions with active learning (T2, T3).
2. Latent (autoencoder) + GENERIC is established (tLaSDI T2/T3; 2007.03758; super-resolution T12; magnetisation T11; latent pH T19/T20). In ALL of these E and S (or the potential) are LEARNED in non-physical latent coordinates; T11 says explicitly the learned potential is not the Gibbs free energy.
3. Structure-preserving reduction with energy from the full model exists for port-Hamiltonian ODEs (T26, T27), and IPHS-based learning with physics-given structure exists for lumped thermal systems (T17).
4. First-principles IPHS now covers convective transport with thermodynamically consistent boundary ports in 1-D (T21), so the port construction 3B needs has a citable non-ML template.
5. "Keep transport/balance laws known, learn a thermodynamically admissible closure" is established in full-field form (GIMLET T22, OVP T23, NCL-MCT T42) and, without thermodynamics, in chromatography hybrids (T34) and FINN sorption (T38).

### What is open / documented weaknesses
- TRANSPORT is the weak spot: GENERIC-FNO loses accuracy on pure transport (T4); Poisson-GENERIC proves its friction class excludes transport and needs a Lie-Poisson L for it (T5). No fetched paper demonstrates a learned metriplectic latent model for an advection-dominated OPEN system with inflow/outflow ports and quantified rollout error.
- LEARNED ENTROPY is harmful: Poisson-GENERIC's learned-entropy ablation "injects energy on every reversible-irreversible problem" (T5); GENERIC-FNO notes (E,S,L,M) non-uniqueness (T4). Fixing E and S from thermodynamics (3B) is a direct answer, and is not done in any latent GENERIC paper found.
- Hidden state hurts structure: with missing variables a Transformer beats a metriplectic network (T24). Structure helps when the state is complete.
- Weak/negative ablations: tLaSDI gas containers, model loss adds nothing beyond the Jacobian loss (T2); 2-D beam gains small (T15); exact Jacobi costs accuracy on Burgers (T5).
- Adsorption/sorption/chromatography/packed beds/reactors: NO learned thermodynamic-structure model found. Neural surrogates for the same physical systems exist without structure (T33, T34, T35, T38).

### Closest prior work to the Paper 3B claim (ranked)
1. tLaSDI and parametric tLaSDI (T2 2403.05848; T3 2506.08475, TMLR 2026): autoencoder + GENERIC latent dynamics, parametric, active learning. Differs: E,S learned; closed benchmark systems (Couette, gas containers, Burgers, heat, Vlasov-Poisson); no ports, no advection-dominated open system.
2. Port-metriplectic NNs (2211.01873, known) + MeshGraphNets informed locally by thermodynamics (T16, AMSES 2025): ports/open advective flow with GENERIC structure. Differs: E,S learned, full-field graph rather than latent, no adsorption, no heat of adsorption.
3. Hernández et al. thermodynamics-aware ROM (2007.03758, known) and its super-resolution successor (T12): latent metriplectic of cylinder flow (an inflow/outflow flow). Differs: E,S learned; boundary fluxes not treated as ports in fetched text.
4. Physically Consistent Neural ODEs / IPHS (T17): thermodynamic structure supplied by physics, parameters learned, thermal application. Differs: lumped ODE, no latent AE, no transport PDE, no adsorption.
5. Poisson-GENERIC neural operators (T5) and GENERIC-FNO (T4): show the transport limitation and the learned-entropy failure that 3B's fixed-E/S design addresses; function-space, not latent; no ports.
6. GIMLET (T22) and NCL-MCT (T42): known advection, learned admissible closures for transport PDEs. Differ: full-field, potentials learned, no adsorption.
7. Latent pH with inputs (T19 Rettberg et al.; T26 Glas et al.): ports as inputs in reduced models. Differ: no entropy, learned or projected quadratic energy, no transport.
8. Application side (no structure): Staudt et al. non-isothermal binary breakthrough neural surrogate (T35, Sep 2026), DeepONet TVSA (T33), chromatography hybrids (T34), FINN diffusion-sorption (T38).

### Threats to novelty
- No fetched work does "learned latent GENERIC/port-metriplectic model of a non-isothermal adsorption column with E and S fixed from thermodynamics". The combination appears novel as of 2026-10-12.
- Residual risks: (a) T1 Mousa et al. (Comput. Chem. Eng. 2025) cites GFINNs and concerns chemical unit operations; content unverified, so read it via library access. (b) T35 (Staudt et al.) occupies the "neural surrogate for non-isothermal breakthrough" application space, so 3B must claim structure, guarantees and data efficiency, not "first surrogate". (c) Reviewers will cite T4/T5 on transport: 3B must place advection in the reversible/port part (as in T21 and T5's Lie-Poisson L), not in the metric, and report transport-dominated accuracy honestly. (d) T24: any claim of learning from outlet-only data is exposed. (e) "Given energy" is partly anticipated in non-latent settings (T5's fixed energy terms, T13's trusted-energy projection, T17's IPHS, T23's partially known functional), so the claim must be specific: fixed physical E and S in a learned latent coordinate system for an open, advective column.
