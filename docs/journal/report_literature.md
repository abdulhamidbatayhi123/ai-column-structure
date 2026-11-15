# Paper 3B literature check against live Crossref and arXiv records (2026-11-16)

I checked every DOI and arXiv ID in the playbook's must-cite list (§5.3), every item in playbook_critic §(c), and the Baaiu, Peng and Walton–LeVan papers on the Crossref REST API or the arXiv API. Where an item had no identifier, I found it with a Crossref title search. A few conference venues come from web searches, and I say which. Nothing was edited, and no e-mail went into any request.

## A. Items that are missing, wrong or need care

1. **TIGNN (10.1109/TAI.2022.3179681):** the paper exists, but its year is **2024**, not 2022. Crossref gives IEEE Trans. Artif. Intell. 5(3):967–976, March 2024. Cite it as 2024.
2. **Schaffer & Exl (10.1088/2632-2153/aebff4):** published 2026-11-09. Crossref has no volume or article number for it yet, so re-check before the reference list is finalised.
3. **Owen 2007, pigeonhole bootstrap:** the journal is **Annals of Applied Statistics**, 1:386–411, DOI 10.1214/07-aoas122. It is not Annals of Statistics. Check that PREREG and the manuscript do not name the wrong journal.
4. **Holm 1979 (Scand. J. Stat.):** **not found** on Crossref. It cannot be confirmed from Crossref; use the JSTOR record.
5. **Ruthven 1984 textbook:** Crossref only has a book review of it (AIChE J. 31:523, 1985, DOI 10.1002/aic.690310335), which confirms Wiley, 1984, xxiv+433 pp. The book has no DOI.
6. **Bilal et al. 2026:** Crossref lists exactly three authors (Bilal, Haq, Asif). lit_review_3B.md writes "Bilal, Haq, Asif and co-authors", which is wrong.
7. **Parametric tLaSDI is missing from the must-cite list.** It is He, Shin, Gruber, Jung, Lee and Choi, arXiv:2506.08475, journal reference *Transactions on Machine Learning Research* 2026. The critic's point (d5) relies on it, and it is the closest learned-latent GENERIC precedent after tLaSDI.
8. **Gruber et al., arXiv:2405.16305:** it was published at **ICLR 2025**, in the proceedings at proceedings.iclr.cc (web search). Cite the published version.
9. **Lee–Trask–Stinis, arXiv:2106.12619:** it was published in **NeurIPS 34 (2021)**, in the proceedings at proceedings.neurips.cc (web search). Cite the published version.
10. **arXiv:2610.20614 (tLaSDI fracture):** the arXiv page says v1 was submitted 12 Jul 2026, although the ID says October 2026. Give the submission date as 12 Jul 2026, not October.
11. **GENERIC-FNO and Poisson-GENERIC have moved on.** arXiv:2606.08343 is now at v9 (2026-11-01) and arXiv:2610.05570 at v3 (2026-11-07). The lit review quotes text from v1 and v2, so re-read the current versions before stating what they claim.
12. **Ceccanti, arXiv:2601.09491:** the paper exists. The isothermal claim was not visible in the abstract I fetched, so I could not confirm it. Check the full text.
13. **Zhong et al., arXiv:2012.02334:** this is a benchmark of ten energy-conserving models on 4 physical systems. The lit review's "pendulum over 10 to 10,000 trajectories" is **not** in the abstract; check the full text.
14. **Walton & LeVan 2005:** the exact record is Adsorption 11:555–559, DOI 10.1007/s10450-005-5984-6. The title begins "Development of Energy Balances for Fixed-Bed Adsorption Processes: Thermodynamic Paths, Heat Capacities and Isosteric He…". I only have it cut off at that point, so read the full title on the record page.
15. **Unverified conference venues.** These items have no Crossref DOI, and I did not check their venues live. Cite each by its arXiv record, or check the venue first:
    - Greydanus et al., arXiv:1906.01563
    - Cranmer et al., arXiv:2003.04630
    - Neural ODE, Chen et al., arXiv:1806.07366
    - FNO, Li et al., arXiv:2010.08895
    - Agarwal et al., arXiv:2108.13264
    - Bouthillier et al., arXiv:2103.03098
    - Dissipative SymODEN, Zhong, Dey, Chakraborty, arXiv:2002.08860
    - Sosanya & Greydanus, arXiv:2201.10085

Every other item exists as described.

## B. Must-cite list (§5.3): what Crossref and arXiv returned

All entries below match the playbook unless they appear in section A.

**MLST precedents** (journal: Mach. Learn.: Sci. Technol.)

| Paper | Record | Relevance to 3B |
|---|---|---|
| Schaffer & Exl | "An energy-based conservative-dissipative latent neural evolution operator for magnetization dynamics", 2026 | Closest MLST latent energy–dissipation operator |
| Nguyen & Sandfeld | "Inductive bias and out-of-distribution generalization in deep learning surrogates for 2D PDE dynamics", 7(5) 055055, 2026 | Inductive-bias and extrapolation framing |
| Lenau, Dimiduk, Niezgoda | "Training variation of physically-informed deep learning models", 7(3) 035005, 2026 | Seed variance |
| Feng, Deng, Zhang, Fan, Wu | "Mass-conserved hierarchical multi-resolution deep learning for long-term global atmospheric pollutant transport prediction", 7(5) 055032, 2026 | Conservation in transport |
| Shen, Song, Zhang | "Two-stage projection physics-informed neural networks for preserving conservation laws", 7(5) 055035, 2026 | Projection to conservation. Still unread. |
| Goyal, Yıldız, Benner | "Deep learning for structure-preserving universal stable Koopman-inspired embeddings for nonlinear canonical Hamiltonian dynamics", 6(1) 015063, 2025 | Structured latent embeddings |
| Nam, Anderson, Choi, Le, Foord, Cho, Jeong, Cho | "Physics-informed latent space dynamics identification for time-dependent NLTE atomic kinetics", 7(5) 055046, 2026 | LaSDI family in MLST |

**Structure-preserving and metriplectic learning**

| Paper | Record | Relevance to 3B |
|---|---|---|
| Hernández, Badías, González, Chinesta, Cueto | "Structure-preserving neural networks", J. Comput. Phys. 426:109950, 2021 | Origin of the metriplectic network |
| Same group | "Deep learning of thermodynamics-aware reduced-order models from data", Comput. Methods Appl. Mech. Eng. (CMAME) 379:113763, 2021 | Learned latent with GENERIC |
| Hernández, Badías, Chinesta, Cueto | "Port-metriplectic neural networks…", Comput. Mech. 72(3):553–561, 2023 | Open systems through ports, the closest formalism |
| Zhang, Shin, Karniadakis | "GFINNs…", Phil. Trans. R. Soc. A 380:20210207, 2022 | Architecture that enforces the degeneracy conditions |
| Gruber, Lee, Trask | "Reversible and irreversible bracket-based dynamics for deep graph neural networks", NeurIPS 36:38454–38484, 2023 | Bracket-based networks |
| Park, Cheung, Choi, Shin | "tLaSDI: Thermodynamics-informed latent space dynamics identification", CMAME 429:117144, 2024 | Direct latent GENERIC precedent |
| Cueto & Chinesta | "Thermodynamics of Learning Physical Phenomena", Arch. Comput. Methods Eng. 30:4653–4666, 2023 | Review |

**Hamiltonian, port-Hamiltonian and Onsager models**

| Paper | Record | Relevance to 3B |
|---|---|---|
| Desai, Mattheakis, Sondak, Protopapas, Roberts | "Port-Hamiltonian neural networks for learning explicit time-dependent dynamical systems", Phys. Rev. E 104:034312, 2021 | Forced and dissipative systems |
| Yu, Tian, E, Li | "OnsagerNet…", Phys. Rev. Fluids 6:114402, 2021 | Dissipative structure |
| Zakwan, Di Natale, Svetozarevic, Heer, Jones, Ferrari Trecate | "Physically Consistent Neural ODEs for Learning Multi-Physics Systems", IFAC-PapersOnLine 56(2):5855–5860, 2023 | Matches the repo's citation. Structure from physics, parameters learned. |

**Structure-preserving reduction**

| Paper | Record | Relevance to 3B |
|---|---|---|
| Öttinger | "Preservation of thermodynamic structure in model reduction", Phys. Rev. E 91:032147, 2015 | Projection-based reduction that keeps the thermodynamic structure |
| Gruber, Gunzburger, Ju, Wang | "Energetically consistent model reduction for metriplectic systems", CMAME 404:115709, 2023 | Same line, metriplectic |
| Chan | "Entropy stable reduced order modeling of nonlinear conservation laws", J. Comput. Phys. 423:109789, 2020 | Entropy-stable reduction |
| Klein, Sanderse, Costa, Pecnik, Henkes | "Entropy-stable model reduction of one-dimensional hyperbolic systems using rational quadratic manifolds", J. Comput. Phys. 528:113817, 2025 | One-dimensional hyperbolic, close to the column setting |

**Learned closures**

| Paper | Record | Relevance to 3B |
|---|---|---|
| Kochkov et al. | "Machine learning–accelerated computational fluid dynamics", PNAS 118:e2101784118, 2021 | Hybrid solver with a learned correction, the comparison for the coarse-solver arm |
| van Gastelen, Edeling, Sanderse | "Energy-conserving neural network for turbulence closure modeling", J. Comput. Phys. 508:113003, 2024 | Structure-preserving closure |
| Bar-Sinai, Hoyer, Hickey, Brenner | "Learning data-driven discretizations for partial differential equations", PNAS 116:15344–15349, 2019 | Learned discretisation |

**Isothermal port-based adsorption models (Lyon group)**

| Paper | Record | Relevance to 3B |
|---|---|---|
| Baaiu, Couenne, Le Gorrec, Lefevre, Tayakout | "Port based modelling of a multiscale adsorption column", Math. Comput. Model. Dyn. Syst. (MCMDS) 14(3):195–211, 2008, 10.1080/13873950701844840 | Isothermal port model, the direct ancestor of 3B's formulation |
| Baaiu, Couenne, Eberard, Jallut, Lefevre, Le Gorrec, Maschke | "Port-based modelling of mass transport phenomena", MCMDS 15(3):233–254, 2009, 10.1080/13873950902808578 | Port-based theory of transport. The author list differs from the 2008 paper. |
| Baaiu, Couenne, Lefevre, Le Gorrec, Tayakout | "Structure-preserving infinite dimensional model reduction: Application to adsorption processes", J. Process Control 19(3):394–404, 2009, 10.1016/j.jprocont.2008.07.002 | Structure-preserving reduction applied to adsorption |
| Peng, Baaiu, Couenne, Le Gorrec | "Energy based discretization of a class of distributed parameters processes comparison with classical approach", IEEE CDC 2007, 3513–3518, 10.1109/cdc.2007.4434663 | Exists. The critic is right that it is missing from must-cite; add it. |

## C. Critic §(c), items the novelty report found

| Item | Record | Relevance to 3B |
|---|---|---|
| Praditia, Karlbauer, Otte, Oladyshkin, Butz, Nowak | "Learning Groundwater Contaminant Diffusion-Sorption Processes With a Finite Volume Neural Network", Water Resour. Res. 58(12):e2022WR033149, 2022 | Closest analogue to system A: a finite-volume network for sorption transport |
| Subraveti, Li, Prasad, Rajendran 2019 | "Machine Learning-Based Multiobjective Optimization of Pressure Swing Adsorption", I&ECR 58(44):20412–20422, 10.1021/acs.iecr.9b04173 | Machine learning in pressure swing adsorption (PSA) design |
| Wu et al. | "Pressure swing adsorption process modeling using physics-informed machine learning…", Green Chem. Eng. 6(2):233–248, 2025 | Physics enters only as a loss |
| Galanti, Janssen, Roghair, Dieulot, Shoeibi Omrani, Boon, van Sint Annaland | "Towards Surrogate Modeling for Adsorption Processes Using Physics-Informed Neural Networks", Processes 13(9):2824, 2025 | PINN adsorption surrogate |
| Bilal, Haq, Asif | "Interpretable Physics-Guided Deep Learning Ensemble for Nonisothermal General Rate Model of Chromatography", I&ECR 65(26):13942–13958, 2026 | Non-isothermal column with scalar outputs |
| An, Wald, Chen, King | "Cycle-resolved dynamic modeling of atmospheric CO2 separation systems under time-varying ambient and electricity conditions", Front. Environ. Eng. 5:1881554, 2026 | Cycle-level direct-air-capture surrogate |
| Ceccanti, Galanti, Roghair, van Sint Annaland | "Deep Operator Networks for Surrogate Modeling of Cyclic Adsorption Processes with Varying Initial Conditions", arXiv:2601.09491, 14 Jan 2026 | Operator surrogate of cyclic temperature–vacuum swing adsorption (TVSA) |
| Bermejo, Badías, González, Cueto | "Meshgraphnets informed locally by thermodynamics…", Adv. Model. Simul. Eng. Sci. 12:27, 2025 | Nodal port-metriplectic graph network |
| Gruber, Lee, Lim, Park, Trask | "Efficiently Parameterized Neural Metriplectic Systems", arXiv:2405.16305, ICLR 2025 | Scalable metriplectic parameterisation |
| Chung, Bu, Verma | "Physics-conforming Latent Twins", CMAME 463:119351, January 2027 issue; arXiv:2606.15053 | Latent flow with prescribed invariants |
| Sulskis & Ravi | GENERIC-FNO, arXiv:2606.08343 (v9) | Metriplectic structure in a neural operator, relevant to the FNO arm |
| Sulskis & Ravi | Poisson-GENERIC Neural Operators, arXiv:2610.05570 (v3) | Same line, exact structure through Casimir entropies |
| Alim, He, Tsao, Yu, Chen | "Thermodynamically Consistent Machine Learning for Fracture Prediction in Heterogeneous Materials", arXiv:2610.20614 | tLaSDI applied to fracture |
| List, Chen, Thuerey | "Learned turbulence modelling with differentiable fluid solvers…", J. Fluid Mech. 949:A25, 2022 | Training through a solver |
| Melchers, Crommelin, Koren, Menkovski, Sanderse | "Comparison of neural closure models for discretised PDEs", Comput. Math. Appl. 143:94–107, 2023 | Controlled closure comparison |
| Sanderse, Stinis, Maulik, Ahmed | "Scientific machine learning for closure models in multiscale problems: A review", Found. Data Sci. 7(1):298–337, 2025 | Closure review |
| Liu, Zhang, Gelb | SymCLaw, "Parametric Hyperbolic Conservation Laws: A Unified Framework for Conservation, Entropy Stability, and Hyperbolicity", arXiv:2601.21080 | Entropy-stable learned flux |
| Liu, Zhang, Gelb | NESCFN, entropy-stable conservative flux networks, arXiv:2507.01795 | Same line |
| Zhong, Dey, Chakraborty | "Benchmarking Energy-Conserving Neural Networks for Learning Dynamics from Data", arXiv:2012.02334 | Benchmark of structured models |
| Janssen, Branca, Buck | "Systematic selection of surrogate models for nonequilibrium chemistry", Astron. Astrophys. 708:A227, 2026 | Latent models lose to fully connected ones there, a negative-result precedent |
| Jones & Fuhg | "A hierarchy of thermodynamics learning frameworks for inelastic constitutive modeling", arXiv:2603.02645 | Which-structure comparison, but for constitutive models |
| Eidnes & Myhr | "Comparing model representations and fitting methods for port-Hamiltonian system identification", arXiv:2611.05031 | Port-Hamiltonian identification comparison |
| Glas, Heinlein, Monsuur, Mu | "Model reduction of port-Hamiltonian systems via neural networks", arXiv:2608.30788 | Intrusive port-Hamiltonian reduction |
| Shen et al. (P10) | See section B | |
| McGreivy & Hakim | "Weak baselines and reporting biases lead to overoptimism in machine learning for fluid-related partial differential equations", Nat. Mach. Intell. 6:1256–1269, 2024 | Justifies the coarse-solver arm and the pre-registration |
| Subraveti et al., PANACHE | "Physics-Based Neural Networks for Simulation and Synthesis of Cyclic Adsorption Processes", I&ECR 61(11):4095–4113, 2022 | Closest adsorption-column precedent; physics only as a loss |
| Wang, Wei, Ling | "Conservative Discrete Structure Stabilizes Autoregressive Rollouts in a 1D Drift Diffusion Poisson Benchmark", arXiv:2606.01366 | A solver core alone reaches round-off; the "use the solver" risk |

## D. Critic §(c), items not in any report

| Topic | Record |
|---|---|
| GENERIC | Grmela & Öttinger, Phys. Rev. E 56:6620–6632, 1997, 10.1103/physreve.56.6620 |
| GENERIC | Öttinger & Grmela, Phys. Rev. E 56:6633–6655, 1997, 10.1103/physreve.56.6633 |
| Metriplectic | Morrison, Physica D 18:410–419, 1986, 10.1016/0167-2789(86)90209-5 |
| Metriplectic | Kaufman, Phys. Lett. A 100:419–422, 1984, 10.1016/0375-9601(84)90634-0 |
| Port-Hamiltonian | van der Schaft & Jeltsema, Found. Trends Syst. Control 1:173–378, 2014, 10.1561/2600000002 |
| Neural ODE | Chen, Rubanova, Bettencourt, Duvenaud, arXiv:1806.07366 |
| Autoencoder reduced models | Lee & Carlberg, J. Comput. Phys. 404:108973, 2020, 10.1016/j.jcp.2019.108973 |
| Autoencoder reduced models | Fresca, Dede', Manzoni, J. Sci. Comput. 87, 2021, 10.1007/s10915-021-01462-7 |
| LaSDI | Fries, He, Choi, CMAME 399:115436, 2022, 10.1016/j.cma.2022.115436 |
| gLaSDI | He, Choi, Fries, Belof, Chen, J. Comput. Phys. 489:112267, 2023, 10.1016/j.jcp.2023.112267 |
| Operator inference | Peherstorfer & Willcox, CMAME 306:196–215, 2016, 10.1016/j.cma.2016.03.025 |
| SINDy | Brunton, Proctor, Kutz, PNAS 113:3932–3937, 2016, 10.1073/pnas.1517384113 |
| FNO | Li et al., arXiv:2010.08895 |
| DeepONet | Lu, Jin, Pang, Zhang, Karniadakis, Nat. Mach. Intell. 3:218–229, 2021, 10.1038/s42256-021-00302-5 |
| Kolmogorov n-width | Ohlberger & Rave, arXiv:1511.02021, Proc. ALGORITMY 2016, 1–12 (no DOI) |
| Dissipative Hamiltonian networks | Sosanya & Greydanus, arXiv:2201.10085 |
| Dissipative SymODEN | Zhong, Dey, Chakraborty, arXiv:2002.08860 |
| ML reproducibility | Henderson et al., "Deep Reinforcement Learning That Matters", AAAI 32, 2018, 10.1609/aaai.v32i1.11694 |
| ML reproducibility | Bouthillier et al., arXiv:2103.03098 |
| Bootstrap intervals (rliable) | Agarwal et al., arXiv:2108.13264 |
| Pigeonhole bootstrap | Owen, Ann. Appl. Stat. 2007 (see A3) |
| Multiple testing | Holm 1979 (see A4) |
| Hodges–Lehmann | Hodges & Lehmann, Ann. Math. Stat. 34:598–611, 1963, 10.1214/aoms/1177704172 |
| Pre-registration in ML | NeurIPS pre-registration workshops: PMLR vol. 148 (2020) and vol. 181 (2021), editors Bertinetto, Henriques, Albanie, Paganini, Varol (from a web search of PMLR, NeurIPS and dblp) |
| MOF-303 water harvesting | Hanikel et al., ACS Cent. Sci. 5:1699–1706, 2019, 10.1021/acscentsci.9b00745 (already in docs/materials_3B.md) |
| Thermocline model | Schumann, J. Franklin Inst. 208:405–416, 1929, 10.1016/s0016-0032(29)91186-8 |
| Thermocline review | Esence, Bruch, Molina, Stutz, Fourmigué, Sol. Energy 153:628–654, 2017, 10.1016/j.solener.2017.03.032 |
| Adsorption textbook | Ruthven 1984 (see A5) |

All of these exist as named. Relevance: GENERIC, metriplectic and port-Hamiltonian papers are the foundations of the formalism. Latent, reduced-order and operator papers cover the U, S and FNO arms. The n-width paper supports the advection-dominated argument. The statistics papers support the PREREG estimators. The domain papers give materials provenance and the system-B model.

## E. 2025–2026 work the list misses

From live arXiv and Crossref searches:

1. **He et al., parametric tLaSDI**, arXiv:2506.08475, TMLR 2026. Must cite (see A7).
2. **Muixí, González, Chinesta, Cueto**, "Structure-preserving formulations for data-driven analysis of coupled multi-physics systems", Comput. Mech. 75(1):357–368, 10.1007/s00466-024-02508-x. Published online 2024, in the 2025 volume. Port-metriplectic coupling of open subsystems, which is close to 3B's port formulation.
3. **Li, Tan, Moyer, Beckers**, "Identify then Realize: Contrastive Learning of Latent Port-Hamiltonian Dynamics from Partial Observations" (CIPHER), arXiv:2605.16682, v2 2026-10-01. Latent port-Hamiltonian models of dissipative systems; a direct neighbour of a structured latent model.
4. **Hernández, Win, O'Connor, Arratia, Trask**, "Data-driven particle dynamics: Structure-preserving coarse-graining…", arXiv:2508.12569, 2025. Metriplectic coarse-grained dynamics learned from data.
5. **Baheri & Lindemann**, "Metriplectic Conditional Flow Matching for Dissipative Dynamics", arXiv:2509.19526, v2 2026-06. A metriplectic surrogate with a structure-preserving integrator, aimed at rollout stability.
6. **Tierz, Alfaro, González, Cueto**, "A Graph Neural Network approach to zero-shot Digital Twins", arXiv:2607.20535, 2026. Metriplectic TIGNN applied to digital twins.
7. **Putkaradze**, "Latent Lie-Poisson Neural Networks (LLPNNs)…", Neural Networks 206:109691, 2027. Structured latent dynamics, but conservative only.
8. **Moradi, Beintema, Jaensson, Tóth, Schoukens**, "Port-Hamiltonian neural networks with output error noise models", Automatica 187:112892, 2026. Port-Hamiltonian identification under noise.
9. **Adsorption and storage surrogates.** All are scalar or physics-as-a-loss surrogates, so none threatens the "structured by construction" claim:
   - **Li, Li, Zhuang, Li, Li**, "Surrogate modeling for design and multi-objective optimization of vacuum pressure swing adsorption-based CO2 capture from flue gas", Chem. Eng. Sci. 320:122467, 2026.
   - **Ananth, Selvakumar, Sher Gill**, "Physics informed neural network prediction of heat transfer mechanisms in manufactured sand for thermal energy storage systems", Front. Energy Effic. 4:1845713, 2026. A PINN for a packed-bed store with physics as a loss.
10. **Of lower relevance**, cite only if space allows:
    - García de Beristain, Nieto Simavilla, Ellero, GINNGER, arXiv:2610.23352, 2026. GENERIC-informed constitutive discovery.
    - Sigdel, arXiv:2607.10439. A metriplectic port-Hamiltonian network for EEG.

No 2025–2026 structure-preserving latent surrogate of an adsorption column or a packed-bed store was found. arXiv searches pairing adsorption or thermal-storage terms with "surrogate" or "neural operator" returned nothing relevant. Only the Ceccanti DeepONet came up, and it is an operator surrogate, not a structure-preserving one. So the novelty claim still stands, worded as "to our knowledge".

Scratch outputs are in C:\Users\ABDULH~1\AppData\Local\Temp\claude\c6eff6e074d30b33\4add04dd-e721-4eee-b705-23ef12c2b27a\scratchpad\lit\ (scripts doi.py, arx.py, q.py, qd.py, as.py; no saved logs).