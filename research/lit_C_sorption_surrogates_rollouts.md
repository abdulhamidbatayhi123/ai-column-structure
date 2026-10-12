# Lit journal C: learned sorption-process surrogates + what hard structure buys in rollouts

Date: 2026-10-12. Researcher lens C for Paper 3B (GENERIC/port-metriplectic latent model of a non-isothermal 1-D adsorption column).
Rule: a source is counted (C#) only if its live page/abstract was fetched this session. SEEN-ONLY = lead from a search listing, not fetched.
Semantic Scholar API returned HTTP 429 at start; used arXiv API, OpenAlex API (abstract reconstructed from abstract_inverted_index), and publisher/arXiv abs pages.

## Search log
- arXiv API: abs:adsorption AND (surrogate OR "neural operator" OR DeepONet OR "neural ODE"), sorted by date -> only Ceccanti 2601.09491 relevant at process level.
- OpenAlex: "surrogate pressure swing adsorption neural network" (from 2022) -> 263 hits, first 50 scanned.
- OpenAlex: "port-Hamiltonian adsorption" (from 2018) -> 381 hits, none a learned adsorption-column model (only N-GENNs 2026, generic).
- OpenAlex: "thermodynamics-informed neural network adsorption column" (from 2020) -> 639 hits, first 50 scanned; no structured-by-construction column model.

## PART 1 sources

### C1
- Title: Towards Surrogate Modeling for Adsorption Processes Using Physics-Informed Neural Networks
- Authors: Mattia Galanti, Mik Janssen, Ivo Roghair, Jean-Yves Dieulot, Pejman Shoeibi Omrani, Jurriaan Boon, Martin van Sint Annaland
- Year/venue: 2025, Processes 13(9) 2824, doi:10.3390/pr13092824
- Fetched: https://api.openalex.org/works/doi:10.3390/pr13092824 (abstract; MDPI page returned 403)
- Supports: hybrid PINN vs plain NN surrogate for the adsorption step of a DAC process; equal accuracy with abundant data, PINN better as data gets scarce; curriculum learning needed in extreme low-data; PINN ~10x training cost. No multi-cycle rollout and no hard constraint reported in abstract (physics is a soft loss).
- Quote: no verbatim quote captured (the tool paraphrased the abstract index); do not quote it without re-checking.

### C2
- Title: A systematic screening of neural network-based hybrid models of adsorption in chromatography processes
- Authors: Jesper Frandsen, Vinicius V. Santana, Peter Jul-Rasmussen, Idelfonso B. R. Nogueira, Jakob K. Huusom, Krist V. Gernaey, Jens Abildskov
- Year/venue: 2025, AIChE Journal 71(12), doi:10.1002/aic.70045
- Fetched: https://api.openalex.org/works/doi:10.1002/aic.70045
- Supports: 10 hybrid structures where an NN replaces the isotherm/adsorption term inside a mechanistic transport model (so the column mass balance is kept by construction, but no thermodynamic/entropy structure on the NN). Most hybrids matched the mechanistic model for a binary nonreactive case; fewer did for a four-component reactive case. No numbers in abstract.
- Quote: "fewer hybrid models demonstrated performance comparable to that of the mechanistic model"

### C3
- Title: Deep Operator Networks for Surrogate Modeling of Cyclic Adsorption Processes with Varying Initial Conditions
- Authors: Beatrice Ceccanti, Mattia Galanti, Ivo Roghair, Martin van Sint Annaland
- Year/venue: 2026 (14 Jan), arXiv:2601.09491 (cs.LG)
- Fetched: https://arxiv.org/abs/2601.09491
- Supports: DeepONet surrogates of TVSA steps, trained on mixed initial conditions, tested out-of-range and on unseen functional forms of ICs; motivation is that each step starts from the previous step's end state (i.e. cycle chaining). Abstract gives no error numbers, does not state a multi-cycle CSS rollout or any enforced physical constraint.
- Quote: "starts from the final state of the preceding step"

### C4
- Title: Accelerating simulation and optimisation of cyclic adsorption processes with differentiable programming
- Authors: Alex Glover, Maria M. Papathanasiou, Ronny Pini
- Year/venue: 2026 (online 22 Sep), Computers & Chemical Engineering 216, 109904, doi:10.1016/j.compchemeng.2026.109904
- Fetched: https://api.openalex.org/works/doi:10.1016/j.compchemeng.2026.109904
- Supports: NOT learned - an end-to-end differentiable mechanistic VPSA model in JAX; AD Jacobian of the single-cycle map used in Newton iteration to CSS, ~20x fewer iterations/time to CSS than a MATLAB implementation, optimisation >100x faster; gradient-based IPOPT beat NSGA-II Pareto front. Relevant as the strongest new non-learned competitor for fast CSS.
- Quote: (abstract summarised by tool; no verbatim quote captured)

### C5
- Title: Pressure swing adsorption process modeling using physics-informed machine learning with transfer learning and labeled data
- Authors: Zhiqiang Wu, Yunquan Chen, Bingjian Zhang, Jingzheng Ren, Qinglin Chen, Huan Wang, Chang He
- Year/venue: 2024, Green Chemical Engineering 6(2) 233-248, doi:10.1016/j.gce.2024.08.004
- Fetched: https://api.openalex.org/works/doi:10.1016/j.gce.2024.08.004
- Supports: PINN for a 5-step PSA cycle, one sub-network per step; parameter transfer learning + domain decomposition for "long-term integration of periodic PDEs"; labelled data added to handle sharp fronts (pure soft PINN struggled). Claims match to numerical solution over cyclic behaviour; no error numbers in abstract. Physics is soft (residual loss).
- Quote: "PSA modeling remains a challenging task since it exhibits strong dynamic and cyclic behavior"

### C6
- Title: Learning Groundwater Contaminant Diffusion-Sorption Processes With a Finite Volume Neural Network
- Authors: Timothy Praditia, Matthias Karlbauer, Sebastian Otte, Sergey Oladyshkin, Martin V. Butz, Wolfgang Nowak
- Year/venue: 2022, Water Resources Research 58(12), doi:10.1029/2022WR033149
- Fetched: https://api.openalex.org/works/doi:10.1029/2022wr033149
- Supports: FINN keeps the finite-volume (conservative flux) structure of the PDE and learns only closures (e.g. the sorption isotherm); outperforms other ML models under modified boundary conditions and a calibrated PDE model on a lab experiment. Closest prior "conservative-by-construction learned sorption transport" model - but isothermal, no entropy/GENERIC structure, not a cycle.
- Quote: "outperforms other ML models when tested under modified boundary conditions"

### C7
- Title: Picard-KKT-hPINN: Enforcing Nonlinear Enthalpy Balances for Physically Consistent Neural Networks
- Authors: Giacomo Lastrucci, Tanuj Karia, Zoe Gromotka, Artur M. Schweidtmann
- Year/venue: 2025, arXiv:2501.17782
- Fetched: https://arxiv.org/abs/2501.17782
- Supports: hard projection layer enforcing nonlinear enthalpy + linear atomic balances to machine precision in a steady-state surrogate of a methanol packed-bed reactor; better than vanilla MLP when data are scarce. Hard balance constraints on an input-output map, not a dynamical structure (no entropy production / dissipation bracket), not adsorption.
- Quote: "enforces multiplicatively separable constraints by sequentially freezing and projecting"

### C8
- Title: Physics-Informed Neural Networks for Predicting Hydrogen Sorption in Geological Formations: Thermodynamically Constrained Deep Learning Integrating Classical Adsorption Theory
- Authors: Mohammad Nooraiepour, Mohammad Masoudi, Zezhang Song, Helge Hellevang
- Year/venue: 2026 (30 Mar), arXiv:2603.28328
- Fetched: https://export.arxiv.org/api/query?id_list=2603.28328
- Supports: PINN for equilibrium H2 sorption (isotherm-level, not column dynamics) with adsorption-theory/thermodynamic constraints in training plus curriculum and ensemble; better cross-lithology generalisation than tuned random forest. Shows "thermodynamically constrained" in the sorption-ML literature currently means soft loss terms on isotherms.
- Quote: "Accurate prediction of hydrogen sorption in fine-grained geological materials is essential"

### C14
- Title: Development of Steady-State and Dynamic Mass and Energy Constrained Neural Networks for Distributed Chemical Systems Using Noisy Transient Data
- Authors: Angan Mukherjee, Debangsu Bhattacharyya
- Year/venue: 2024, Ind. Eng. Chem. Res. 63(32) 14211-14239, doi:10.1021/acs.iecr.4c01429
- Fetched: https://api.openalex.org/works/doi:10.1021/acs.iecr.4c01429
- Supports: equality-constrained training/forward solves make NN models (incl. hybrid series-parallel dynamic-static nets) EXACTLY conserve overall mass and energy, even with noisy balance-violating data; dynamic enforcement needs holdup information; superheater/reheater, nonisothermal CSTR, electrically heated PFR; RMSE <1%. Hard balance constraints, but no adsorption column and no entropy/second-law structure. Companion CACE 2024 paper (doi:10.1016/j.compchemeng.2024.108722, mass-only; abstract absent in OpenAlex) keyworded post-combustion CO2 capture - SEEN-ONLY for content.
- Quote: "exactly conserve the overall mass and energy of distributed chemical process systems"

### C15
- Title: Process-informed optimal design principles for cooperative step-isotherm adsorbents in direct air capture using ANN-accelerated TVSA modelling
- Authors: Maryam Nasiri Ghiri, Hamid Reza Nasriani, Mohammad Bagheri, Leila Khajenoori, Samira Mohammadkhani, Andrei Chamchine
- Year/venue: 2026, Chem. Eng. J. Advances 27, 101283, doi:10.1016/j.ceja.2026.101283
- Fetched: https://api.openalex.org/works/doi:10.1016/j.ceja.2026.101283
- Supports: typical 2026 practice - an ANN input->KPI surrogate trained on 2,866 packed-bed TVSA simulations, used for sensitivity/optimisation of step-isotherm sorbents (mmen-Mg2(dobpdc)). No dynamics, no structure; illustrates that most process-ML is static KPI regression.
- Quote: "Direct air capture (DAC) using solid sorbents is limited by energy demand"

SEEN-ONLY leads (Part 1, not fetched): Chung, Kim, Jung, Lee 2024 CEJ 153664 "Hybrid modeling of vacuum swing adsorption carbon capture process..." (abstract null in OpenAlex); Hamidi & Fakhroleslam 2025 SSRN 5084045 NARX/LSTM/GRU dynamic PSA surrogates (abstract null); "Vector Acceleration Methods for Faster Convergence of Cyclic Steady State in Adsorption Process Simulations" IECR 2026 doi:10.1021/acs.iecr.6c00407; "Surrogate-assisted cyclic performance optimisation of DAC using amine-functionalised MOFs" Sep. Purif. Technol. 2025 136177; "Lightweight neural ODEs for learning adsorption kinetics" Discover Appl. Sci. 2026; "Machine learning for thermal energy storage in buildings: ... sorption storage" Energy Reports 2026; "From data to physics: hybrid ANN-GA ... adsorption cooling and desalination" Appl. Therm. Eng. 2026; Santana et al. CES 2023 119223 (= arXiv 2303.13555, known).

## PART 2 sources

### C9
- Title: Stable and Counterfactually Robust Physical World Models from Imposed Structure and Learned Physics
- Authors: Yufeng Wang, Parivesh Priye, Lu Wei, Haibin Ling
- Year/venue: 2026 (24 Sep), arXiv:2610.00280
- Fetched: https://arxiv.org/abs/2610.00280
- Supports: learned energy pushed through a fixed reversible operator plus one-way dissipation port; ~9k parameters; stable out to 100x the training horizon; equal-capacity unstructured models near chance or worse; separates conserving vs dissipating systems by 4 orders of magnitude; 2-17x better held-out intervention prediction than a converged linear model. Abstract reports no failure case of structure.
- Quote: "out to one hundred times the training horizon"

### C10
- Title: Poisson-GENERIC Neural Operators: Exact Metriplectic Structure in Function Space via Casimir Entropies
- Authors: Jason Sulskis, Sathya Ravi
- Year/venue: 2026 (4 Oct), arXiv:2610.05570
- Fetched: https://arxiv.org/abs/2610.05570
- Supports: argues projection-based GENERIC neural operators are state-dependent and break Jacobi; builds degeneracy in via Casimir entropy on augmented state. Heat, Burgers, advection (+1) in 1D/2D with FNO, Transolver, CNO backbones: wins 61/72 seed-level comparisons vs same-backbone unconstrained baselines (so loses 11/72). Ablations: constant-L variant lost on Burgers (cost of exact Jacobi); learned-entropy ablation injected energy on every reversible-irreversible problem. No rollout numbers in abstract.
- Quote: "the model wins 61 of 72 seed-level comparisons against same-backbone unconstrained baselines"

### C11
- Title: Port-Hamiltonian Koopman Operator Synthesis for Mechanical Systems
- Authors: Rajpal Singh, Aditya Singh, Jishnu Keshavan
- Year/venue: 2026 (15 Sep), arXiv:2609.17249
- Fetched: https://export.arxiv.org/api/query?id_list=2609.17249
- Supports: passive-by-construction pH generator in a Koopman lifted space with energy-balance-preserving discretisation; better prediction accuracy, data efficiency and closed-loop tracking than Koopman baselines (sim + experiment).
- Quote: "Finite-dimensional Koopman models enable efficient linear prediction and control of nonlinear robotic systems."

### C12
- Title: Benchmarking Energy-Conserving Neural Networks for Learning Dynamics from Data
- Authors: Yaofeng Desmond Zhong, Biswadip Dey, Amit Chakraborty
- Year/venue: 2020 (v6 later; L4DC 2021), arXiv:2012.02334
- Fetched: https://arxiv.org/abs/2012.02334
- Supports: 10 energy-conserving models (HNN, LNN, DeLaN, SymODEN, CHNN, CLNN, variants) on 4 systems; abstract gives no numbers/ranking (results in body, not fetched).
- Quote: none captured verbatim from abstract

### C13
- Title: Deconstructing the Inductive Biases of Hamiltonian Neural Networks
- Authors: Nate Gruver, Marc Finzi, Samuel Stanton, Andrew Gordon Wilson
- Year/venue: 2022, arXiv:2202.04836 (ICLR 2022)
- Fetched: https://arxiv.org/abs/2202.04836
- Supports: NEGATIVE/qualifying result - HNN gains come mainly from modelling acceleration directly (second-order structure) and simple coordinates, not from symplecticity/energy conservation; relaxing conservation matches or beats HNNs on conservative systems and substantially improves on non-conservative (contact) systems. Important caveat: strict conservative structure can hurt when the true system dissipates/is forced.
- Quote: "modeling acceleration directly and avoiding artificial complexity from the coordinate system"


Update to C12 (full text fetched: https://arxiv.org/html/2012.02334v6): CHNN and CLNN (explicit Cartesian constraints) consistently best on test-trajectory error, long-term prediction and energy drift; on 2-pendulum with 10 to 10,000 training trajectories, structured forms improved data efficiency (trend; no table numbers captured). Structure did NOT help on the 1-pendulum (simple enough without it) nor on the gyroscope (unexplained by the authors). Quote: "CLNN and CHNN, which enforce constraints explicitly, have a significant advantage in prediction."

### C16
- Title: Conservative Discrete Structure Stabilizes Autoregressive Rollouts in a 1D Drift Diffusion Poisson Benchmark
- Authors: Yufeng Wang, Lu Wei, Haibin Ling
- Year/venue: 2026 (31 May), arXiv:2606.01366 (cs.CE)
- Fetched: https://arxiv.org/abs/2606.01366
- Supports: cleanest structured-vs-unstructured rollout numbers found. Learned flux correction inside a conservative FV update with positivity-aware limiting vs direct next-state regressors (+Poisson recomputation, +charge projection, +4-step rollout training). Headline rollout MSE 7.35e-9 vs 4.23e1 (unconstrained), 2.53e1, 6.72e1, 2.71e1. Over 64 configs: best rollout MSE in 60/64 but best one-step MSE in only 19/64. Post-hoc projection and rollout training did not rescue the unconstrained model. Caveat (authors): the classical FV core alone reaches near-roundoff error, so the gain is from discrete structure rather than the learned closure. Directly analogous to a 1-D column.
- Quote: "local conservative finite volume structure is more important than one step neural regression accuracy"

### C17
- Title: tLaSDI: Thermodynamics-informed latent space dynamics identification
- Authors: Jun Sur Richard Park, Siu Wun Cheung, Youngsoo Choi, Yeonjong Shin
- Year/venue: 2024, arXiv:2403.05848 (v2)
- Fetched: https://arxiv.org/abs/2403.05848 and https://arxiv.org/html/2403.05848v2
- Supports: autoencoder + GENERIC-structured latent dynamics, the closest architectural template to Paper 3B. Extrapolation error, Couette (Oldroyd-B): tLaSDI 2.47e-3 vs TA-ROM 1.41e-2 vs vanilla FNN 1.70e-2; two gas containers: 3.96e-3 vs 1.31e-2 vs 2.01e-2; parametric 1D Burgers worst-case max rel. error 1.5% vs gLaSDI 6.0%. Caveats: smallest-loss of 10 runs reported; no unstructured baseline on Burgers/heat; in gas-container ablation the extra model loss added nothing over Jacobian loss.
- Quote: "tLaSDI gives the smallest extrapolation errors for all 10 simulations"

### C18
- Title: GFINNs: GENERIC Formalism Informed Neural Networks for Deterministic and Stochastic Dynamical Systems
- Authors: Zhen Zhang, Yeonjong Shin, George Em Karniadakis
- Year/venue: 2021, arXiv:2109.00092
- Fetched: https://arxiv.org/abs/2109.00092
- Supports: degeneracy conditions satisfied by architecture (not penalty), universal approximation proof; outperforms "existing methods" on two gas containers exchanging heat/volume, thermoelastic double pendulum, Langevin. No data-efficiency numbers in abstract.
- Quote: "obey the symmetric degeneracy conditions of the GENERIC formalism"

### C19
- Title: Deep learning of thermodynamics-aware reduced-order models from data
- Authors: Quercus Hernandez, Alberto Badias, David Gonzalez, Francisco Chinesta, Elias Cueto
- Year/venue: 2020, arXiv:2007.03758
- Fetched: https://arxiv.org/abs/2007.03758
- Supports: sparse autoencoder + metriplectic latent integrator (first GENERIC latent ROM); energy conservation and entropy inequality guaranteed; fluid + solid mechanics examples; no long-horizon or baseline numbers in abstract.
- Quote: "guaranteed to conserve the total energy of the system"

### C20
- Title: Port-metriplectic neural networks: thermodynamics-informed machine learning of complex physical systems
- Authors: Quercus Hernandez, Alberto Badias, Francisco Chinesta, Elias Cueto
- Year/venue: 2022, arXiv:2211.01873
- Fetched: https://arxiv.org/abs/2211.01873
- Supports: port (open-system) extension of GENERIC learning; learn subsystems by parts and couple them. This is the formalism a column with inlet/outlet and heating ports needs. Abstract has no baseline comparison or numbers.
- Quote: "satisfy by construction the principles of thermodynamics in the learned physics"

### C21
- Title: Efficiently Parameterized Neural Metriplectic Systems
- Authors: Anthony Gruber, Kookjin Lee, Haksoo Lim, Noseong Park, Nathaniel Trask
- Year/venue: 2024, arXiv:2405.16305 (v3)
- Fetched: https://arxiv.org/abs/2405.16305
- Supports: metriplectic parameterisation with cost quadratic in state size and metriplectic rank; claims superior accuracy and scalability and gives an error estimate; no numbers in abstract.
- Quote: "scales quadratically in both the size of the state"

### C22
- Title: Learning Stable Deep Dynamics Models
- Authors: Gaurav Manek, J. Zico Kolter
- Year/venue: NeurIPS 2019; arXiv:2001.06116
- Fetched: https://arxiv.org/abs/2001.06116
- Supports: jointly learned dynamics + Lyapunov function, stability guaranteed globally by construction; simple systems and video textures. No structured-vs-unstructured numbers in abstract.
- Quote: "guaranteed to be stable over the entire state space"

### C23
- Title: Stabilized Neural Ordinary Differential Equations for Long-Time Forecasting of Dynamical Systems
- Authors: Alec J. Linot, Joshua W. Burby, Qi Tang, Prasanna Balaprakash, Michael D. Graham, Romit Maulik
- Year/venue: 2022, arXiv:2203.15706
- Fetched: https://arxiv.org/abs/2203.15706
- Supports: linear (sparse conv) + nonlinear split RHS; on Kuramoto-Sivashinsky stabilized NODEs stay on the attractor where standard NODEs fail; on Burgers better short-time tracking/spectra and robustness to noisy ICs. Qualitative in abstract.
- Quote: "stabilized neural ODEs keep long-time trajectories on the attractor"

### C24
- Title: Learning Dissipative Neural Dynamical Systems
- Authors: Yuezhu Xu, S. Sivaranjani
- Year/venue: 2023, arXiv:2309.16032
- Fetched: https://arxiv.org/abs/2309.16032
- Supports: dissipativity imposed AFTER unconstrained training via weight perturbation, then bias perturbation to restore the fit; authors note imposing dissipativity during training is hard. Accuracy cost not quantified in abstract.
- Quote: none captured verbatim.

### C25
- Title: Learning Dissipative Chaotic Dynamics with Boundedness Guarantees
- Authors: Sunbochen Tang, Themistoklis Sapsis, Navid Azizan
- Year/venue: 2024 (v4 Dec 2025), arXiv:2410.00976
- Fetched: https://arxiv.org/abs/2410.00976
- Supports: dissipative projection layer + jointly learned energy-like function, giving formally bounded trajectories; Lorenz 96 and a KS ROM; states prior data-driven models are often unbounded; no numbers in abstract.
- Quote: "are still often prone to generating unbounded trajectories"

### C26
- Title: An Energy-Based Conservative-Dissipative Latent Neural Evolution Operator for Magnetization Dynamics
- Authors: Sebastian Schaffer, Lukas Exl
- Year/venue: 2026 (3 Sep), arXiv:2609.04530
- Fetched: https://arxiv.org/abs/2609.04530
- Supports: latent dynamics from a learned potential; dissipative-only vs antisymmetric+dissipative structures equally accurate on training-length windows, but antisymmetric-dissipative markedly better on uninterrupted rollouts; deep-quadratic energy has slower error growth at 2x training horizon. Lesson: a structure that does not match the physics (pure gradient flow) degrades rollouts. No unstructured baseline.
- Quote: "antisymmetric-dissipative models provide markedly more accurate trajectory predictions"

### C27
- Title: Stochastic Port-Hamiltonian Neural Networks: Universal Approximation with Passivity Guarantees
- Authors: Luca Di Persio, Matthias Ehrhardt, Youness Outaleb
- Year/venue: 2026 (10 Mar), arXiv:2603.10078
- Fetched: https://arxiv.org/abs/2603.10078
- Supports: skew J, PSD R, NN Hamiltonian; weak passivity in expectation; noisy mass-spring, Duffing, Van der Pol: better long-horizon rollouts and lower energy error than an MLP (no numbers in abstract).
- Quote: "enforce skew symmetry of the interconnection matrix"

### C28
- Title: A Structure-Preserving Graph Neural Solver for Parametric Hyperbolic Conservation Laws
- Authors: Jiamin Jiang, Shanglin Lv, Jingrun Chen
- Year/venue: 2026 (17 Apr), arXiv:2604.15617
- Fetched: https://arxiv.org/abs/2604.15617
- Supports: GNN as learned reconstruction-and-flux operator (local conservation + upwinding built in); claims superior long-horizon rollout stability vs strong surrogate baselines; no numbers in abstract.
- Quote: "designed as a learned reconstruction-and-flux operator rather than a black-box state updater"

### C29
- Title: PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers
- Authors: Phillip Lippe, Bastiaan S. Veeling, Paris Perdikaris, Richard E. Turner, Johannes Brandstetter
- Year/venue: 2023, arXiv:2308.05732 (NeurIPS 2023)
- Fetched: https://arxiv.org/abs/2308.05732
- Supports: common rollout strategies become unstable over long horizons; primary cause is neglected non-dominant (often high) spatial frequencies; fix is multistep diffusion-style refinement (not physical structure), which also improves data efficiency. An alternative explanation of rollout failure that thermodynamic structure does not address.
- Quote: "as the primary pitfall limiting stable, accurate rollout performance"

### C30
- Title: Message Passing Neural PDE Solvers
- Authors: Johannes Brandstetter, Daniel Worrall, Max Welling
- Year/venue: ICLR 2022; arXiv:2202.03376
- Fetched: https://arxiv.org/abs/2202.03376
- Supports: autoregressive stability treated as a training/domain-adaptation problem (zero-stability motivated); stability via training rather than physical structure.
- Quote: "demonstrating fast, stable, and accurate performance across different domain topologies"

### C31
- Title: PDEBench: An Extensive Benchmark for Scientific Machine Learning
- Authors: Makoto Takamoto, Timothy Praditia, Raphael Leiteritz, Dan MacKinlay, Francesco Alesiani, Dirk Pflueger, Mathias Niepert
- Year/venue: NeurIPS 2022 Datasets and Benchmarks; arXiv:2210.07182
- Fetched: https://arxiv.org/abs/2210.07182
- Supports: benchmark (FNO, U-Net, PINN baselines) with new metrics; abstract reports no structured-vs-unstructured rollout comparison, i.e. the standard benchmark does not test thermodynamic structure.
- Quote: "a benchmark suite of time-dependent simulation tasks based on Partial Differential Equations (PDEs)"

### C32
- Title: Learning Physical Models that Can Respect Conservation Laws
- Authors: Derek Hansen, Danielle C. Maddix, Shima Alizadeh, Gaurav Gupta, Michael W. Mahoney
- Year/venue: ICML 2023; Physica D 457 (2024); arXiv:2302.11002
- Fetched: https://arxiv.org/abs/2302.11002 and https://arxiv.org/html/2302.11002
- Supports: ProbConserv (integral conservation + Bayesian update). Conservation error, diffusion: unconstrained ANP 4.68e-3, soft constraint 3.47e-3, hard/ProbConserv 0. NEGATIVE evidence on naive hard constraints: HardC shifts every point equally, creating negative bias in the zero-density region (non-physical) on PME and Stefan; on Stefan HardC MSE 5.18e-3 vs ProbConserv 1.89e-3 and only ~3% better than unconstrained; HardC had the lowest MSE on diffusion (1.37 vs 1.55). Soft differential-form constraint did not give a conservative solution on Stefan. Lesson: conservation by global projection can break positivity; conservation and positivity must be co-designed.
- Quote: "leading to a non-physical solution"

### C33
- Title: Learning differentiable solvers for systems with hard constraints
- Authors: Geoffrey Negiar, Michael W. Mahoney, Aditi S. Krishnapriyan
- Year/venue: ICLR 2023; arXiv:2207.08675
- Fetched: https://arxiv.org/abs/2207.08675
- Supports: differentiable PDE-constrained layer; hard constraints in the architecture give much lower test error than an unconstrained objective (no numbers in abstract).
- Quote: "Incorporating hard constraints directly into the NN architecture achieves much lower test error"

### C34
- Title: Characterizing possible failure modes in physics-informed neural networks
- Authors: Aditi S. Krishnapriyan, Amir Gholami, Shandian Zhe, Robert M. Kirby, Michael W. Mahoney
- Year/venue: NeurIPS 2021; arXiv:2109.01050
- Fetched: https://arxiv.org/abs/2109.01050
- Supports: soft physics penalties fail on moderately hard convection/reaction/diffusion problems because the regulariser makes optimisation ill-conditioned, not because of expressivity; curriculum and seq-to-seq give ~10-100x lower error. The failure mode of soft structure.
- Quote: "the PINN's setup makes the loss landscape very hard to optimize."

### C35
- Title: MC-LSTM: Mass-Conserving LSTM
- Authors: Pieter-Jan Hoedt, Frederik Kratzert, Daniel Klotz, Christina Halmich, Markus Holzleitner, Grey Nearing, Sepp Hochreiter, Guenter Klambauer
- Year/venue: 2021, arXiv:2101.05186
- Fetched: https://arxiv.org/abs/2101.05186
- Supports: recurrent architecture that conserves mass by construction; state of the art on arithmetic tasks and peak-flow prediction on a large hydrology benchmark; no LSTM-vs-MC-LSTM numbers in abstract.
- Quote: "adheres to these conservation laws by extending the inductive bias of LSTM"

### C36
- Title: A meshfree exterior calculus for generalizable and data-efficient learning of physics from point clouds
- Authors: Benjamin D. Shaffer, Brooks Kinch, M. Ani Hsieh, Nathaniel Trask
- Year/venue: 2026 (8 May), arXiv:2605.08436
- Fetched: https://arxiv.org/abs/2605.08436
- Supports: exact discrete conservation by construction (learned edge fluxes); 1-2 orders lower OOD error than neural-operator baselines on five PDE benchmarks; single-solution training transfers; fewer training geometries needed on SimJEB. Data-efficiency evidence for conservative structure.
- Quote: "1-2 orders of magnitude lower out-of-distribution error than baseline neural-operator approaches"

### C37
- Title: Physics-Informed Neural Networks for Methane Sorption: Cross-Gas Transfer Learning, Ensemble Collapse Under Physics Constraints, and Monte Carlo Dropout Uncertainty Quantification
- Authors: Mohammad Nooraiepour, Zezhang Song, Wei Li, Sarah Perez
- Year/venue: 2026 (15 Apr), arXiv:2604.13992
- Fetched: https://arxiv.org/abs/2604.13992
- Supports: (Part 1 and a Part 2 negative) sorption-isotherm PINN; R2 0.932 on held-out coal; H2 pre-training gives 18.9% lower RMSE. Deep ensembles sharing physics constraints degrade ("ensemble collapse") regardless of architecture/initialisation: a sorption-domain case of constraints hurting ensemble UQ. Equilibrium only, soft constraints.
- Quote: "regardless of architectural diversity or initialization strategy"

### C38
- Title: A Graph Neural Network approach to zero-shot Digital Twins
- Authors: Alicia Tierz, Iciar Alfaro, David Gonzalez, Elias Cueto
- Year/venue: 2026 (10 Jul), arXiv:2607.20535
- Fetched: https://arxiv.org/abs/2607.20535
- Supports: metriplectic GNN (local energy conservation, non-negative entropy production) + data assimilation; viscoelastic beam and sloshing; ~25 ms/frame; no unstructured baseline numbers in abstract.
- Quote: "enables the direct projection of latent mechanical variables via Augmented Reality"

### C39
- Title: Neural Petri flows for chemical reactions
- Authors: Jose Eduardo Escrig Molina, Daniel Probst
- Year/venue: 2026 (6 Oct), arXiv:2610.08750
- Fetched: https://arxiv.org/abs/2610.08750
- Supports: proves conservation forces m' = m + C sigma and non-negativity forces an enabling rule; only rate laws learned. Positivity + conservation by construction for chemistry, but evaluated on reaction prediction (87.7% full data, 67.4% at 1% data), not dynamics rollouts.
- Quote: "every top-1 prediction is a valid molecule without a filter"

Fetched but NOT counted: Li et al. 2021, arXiv:2106.06898, "Learning Dissipative Dynamics in Chaotic Systems". The v2 abstract only says the learned solution operator reproduces invariant statistics of Kolmogorov flow up to Re 5000; it does not mention a dissipativity regulariser or blow-up of unconstrained models. Do not cite it for "dissipativity prevents blow-up" without reading the body.

SEEN-ONLY (Part 2, arXiv listings, not fetched): 2608.23546 inertial-manifold neural operator (claims better autoregressive stability than FNO); 2606.00937 cellular sheaf neural operators (MHD rollouts); 2503.01775 Hurwitz-constrained NODEs for stiff systems; 2608.00571 CoSynFlow conformal symplectic flows; 2407.17214 ML subgrid flux + convex limiting (positivity); 2511.04991 structure-preserving radiative-transfer surrogates (positivity); 2603.17849 Koopman generator decomposition for pH systems; 2610.09255 numerical provenance in neural PDE solvers.

## SYNTHESIS

### Part 1: learned surrogates of adsorption / cyclic adsorption processes (2022-2026)
Answer to the key question: NO. Searches turned up no learned model of an adsorption or sorption column or cycle that is thermodynamically structured by construction (GENERIC, metriplectic, port-Hamiltonian, Onsager or entropy-stable). Searches run: arXiv API (adsorption/swing adsorption/chromatography/breakthrough/fixed-bed/packed bed x neural/ML/surrogate/operator/neural ODE; adsorption|sorption|chromatography|desiccant x metriplectic|port-Hamiltonian|Onsager|entropy stable|thermodynamically consistent|thermodynamics-informed, 24 hits, none learned+structured); OpenAlex (PSA surrogate NN; port-Hamiltonian adsorption; thermodynamics-informed NN adsorption column; neural ODE adsorption breakthrough; FNO adsorption; AWH sorbent bed ML; adsorption heat pump/desiccant wheel/chiller surrogate; dynamic surrogate CSS adsorption; TSA ML surrogate). Semantic Scholar was rate-limited (429) all session, and a final OpenAlex query also hit 429, so coverage is good but not exhaustive.
What exists instead, from most to least structure:
1. Hard balance constraints with no second-law structure: Picard-KKT-hPINN, hard enthalpy + atom balances, steady packed-bed reactor (C7); Mukherjee & Bhattacharyya, exact mass + energy conservation in dynamic NNs for CSTR/PFR/heat exchanger, not adsorption (C14).
2. Conservative numerical skeleton with a learned closure: FINN, finite-volume sorption transport, isothermal (C6); NN-isotherm hybrids inside a mechanistic chromatography model (C2; also Santana 2023, known). Mass balance holds because the PDE skeleton is kept, but there is no entropy structure and no cycle.
3. Soft physics (PINN residuals): DAC adsorption step (C1), 5-step PSA with transfer learning and domain decomposition for "long-term integration of periodic PDEs" (C5), sorption isotherms (C8, C37).
4. Unstructured operator / KPI surrogates: DeepONet for TVSA steps chained by initial conditions (C3; no CSS-rollout or error numbers in the abstract), ANN KPI surrogate for TVSA design (C15), plus many static PSA KPI regressors (OpenAlex listings).
5. Non-learned competitor: differentiable JAX VPSA with a Newton-on-cycle-map, reaching cyclic steady state ~20x faster (C4). Paper 3B should cite this as the fast physics baseline.
Gap for Paper 3B: no paper reports zero unphysical outputs (negative loadings, entropy decrease, energy non-closure) over many cycles for a learned sorption model, and none applies GENERIC or port-metriplectic structure to a non-isothermal column. Multi-cycle rollout of learned models is barely tested: C5 claims cyclic behaviour without numbers, and C3 chains steps.

### Part 2: what hard structure buys (honest state of evidence)
For structure:
- Long rollouts: the strongest number is C16. In a 1-D conservative FV drift-diffusion benchmark, rollout MSE was 7e-9 vs 25-67 for unconstrained regressors, and post-hoc projection or rollout training did not rescue the unconstrained models. Also C9 (stable to 100x the training horizon; unstructured models near chance), C23 (stabilized NODE stays on the KS attractor), C27, C28, C25, C22 (guarantees, mostly qualitative).
- Latent GENERIC ROMs: tLaSDI (C17) has ~5-8x lower extrapolation error than TA-ROM and a vanilla FNN, and 1.5% vs 6.0% worst-case on Burgers. C10 wins 61/72 seed-level comparisons against same-backbone unconstrained neural operators.
- Data efficiency: C12 (structured Hamiltonian/Lagrangian models more data-efficient from 10 to 10k trajectories), C36 (1-2 orders lower OOD error, single-solution training), C11, C1 and C7 (gains in scarce data).
Against or qualifying:
- Gains are often in rollout or extrapolation, not one-step fit: C16 lost one-step MSE in 45/64 configurations.
- Wrong or over-strict structure hurts. C13: HNN gains come from second-order form and coordinates, not conservation, and relaxing conservation helps on non-conservative systems. C26: pure gradient-flow latents degrade on rollouts. C10: the exact-Jacobi constant-L ablation lost on Burgers and the learned-entropy ablation injected energy. C12: no gain on simple or gyroscope systems.
- Naive hard conservation by global projection can create non-physical negative values (C32). For a sorption column, conservation and positivity therefore have to be built in together, e.g. flux-form plus limiter as in C16.
- Soft constraints are not structure. They give ill-conditioned optimisation (C34), non-conservative solutions (C32 Stefan), ensemble collapse (C37) and ~10x training cost (C1).
- Rollout instability has non-structural causes and fixes: spectral neglect (C29) and training tricks (C30). A fair Paper 3B baseline should include an unstructured model trained with rollout or pushforward losses, not only a naive one.
- Most abstracts report best-of-N runs (C17) or no numbers. The standard benchmarks (C31) do not test thermodynamic structure, so no field-wide controlled comparison exists.
Net: the evidence that hard, correctly matched structure improves long-horizon stability and out-of-distribution or low-data accuracy is consistent and sometimes large, but it comes mostly from small mechanical or toy-PDE systems with selective reporting. One-step accuracy is often traded away, and a mismatched structure can hurt. A controlled, many-seed, matched-capacity test on an open (ported), non-isothermal, positivity-constrained 1-D column, with rollout-trained unstructured baselines, would be new.
