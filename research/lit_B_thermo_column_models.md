# Lit B: thermodynamically structured (non-ML) column / bed models

Lens: GENERIC / metriplectic / (irreversible) port-Hamiltonian / Onsager formulations of adsorption, chromatography, packed-bed, reactor and heat-exchanger columns; adsorbed-phase thermodynamics; entropy generation in adsorption cycles.
Session date: 2026-10-12. Rule: a source counts only if its live page was fetched this session; otherwise SEEN-ONLY.

## Search log
- OpenAlex: "port-Hamiltonian adsorption column" (281 hits, no direct hit on p1 except Baaiu-related via next query), "port-Hamiltonian chromatography" (272 hits, nothing relevant on p1), "irreversible port-Hamiltonian" (2055 hits, first 50 scanned), "GENERIC metriplectic adsorption" (5 hits, ML/SEA only), "structure-preserving infinite dimensional model reduction adsorption processes" (Baaiu 2008 hit).
- HAL API: "adsorption AND (hamiltonian OR bond graph)" (18 hits, none relevant); "(Couenne OR Tayakout) AND (Hamiltonian OR port)" (23 hits; LAGEPP Lyon group: adsorption column, heat exchanger, chemical reactors, reaction-diffusion).

## Sources

### P1. Port based modelling of a multiscale adsorption column
- Authors: Ahmed Baaiu, Francoise Couenne, Yann Le Gorrec, Laurent Lefevre, Melaz Tayakout-Fayolle. 2008. Mathematical and Computer Modelling of Dynamical Systems. doi:10.1080/13873950701844840
- Fetched: OpenAlex record with abstract (api.openalex.org/works/doi:10.1080/13873950701844840) and HAL API record (hal-01637003, notice only, no open file). HAL landing page blocked (Anubis bot wall); T&F full text paywalled. FULL TEXT NOT READ.
- Supports: a port-based (bond-graph / Dirac-structure) distributed model of an adsorption column already exists, but ISOTHERMAL; each phenomenon split into accumulation and dissipation elements; multi-scale (column / pellet / crystal) coupling is also a power-conserving Dirac structure.
- Quote: "A port-based distributed parameter model of an adsorption column under isothermal conditions"
- Bears on: THE KEY PRIOR ART for "does it exist" (yes: isothermal, bond graph / Dirac structure, not IPHS/GENERIC, no heat of adsorption, no learned model). Items (1)/(2) not verifiable without full text; (3) inter-scale mass transfer is a dissipative element.

### P2. Structure-preserving infinite dimensional model reduction: Application to adsorption processes
- Authors: Ahmed Baaiu, Francoise Couenne, Laurent Lefevre, Yann Le Gorrec, Melaz Tayakout-Fayolle. 2009 (online 2008). Journal of Process Control. doi:10.1016/j.jprocont.2008.07.002
- Fetched: OpenAlex record (no abstract served) + HAL API (hal-01625132, notice only). Title/venue/authors verified; CONTENT NOT READ.
- Supports: only that a structure-preserving (port-Hamiltonian) reduction was applied to adsorption processes by the Lyon LAGEP group.
- Quote: title only.
- Bears on: prior art for structure-preserving discretisation of adsorption columns (non-ML).

### P3. Energy based discretization of a class of distributed parameters processes comparison with classical approach
- Authors: Hai-Feng Peng, Ahmed Baaiu, Francoise Couenne, Yann Le Gorrec. 2007. IEEE CDC. doi:10.1109/cdc.2007.4434663
- Fetched: OpenAlex record with abstract.
- Supports: port variables defined so that the dissipation-accumulation interconnection is power preserving within the domain and through the boundary; applied to tubular reactors and adsorption processes; structure-preserving discretisation avoids oscillations of centred finite differences on an adsorption process.
- Quote: "the proposed method avoids oscillations obtained from finite difference method"
- Bears on: (1)/(2) boundary power flow explicit; structure-preserving discretisation of an adsorption column already done (non-ML, isothermal mass transport).

### P4. Port-based modelling of mass transport phenomena
- Authors: Ahmed Baaiu, Francoise Couenne, Damien Eberard, Christian Jallut, Laurent Lefevre, Yann Le Gorrec, Bernhard M. Maschke. 2009. Math. Comput. Model. Dyn. Syst. doi:10.1080/13873950902808578
- Fetched: OpenAlex abstract + HAL API abstract (hal-00579370).
- Supports: bond-graph / Dirac-structure formulation of heat and mass transfer from the first principle, conservation laws and flux-force closure laws; two-phase (multi-level) coupling is itself a Dirac structure.
- Quote: "instantaneous power conservation appears naturally as a geometric interconnection structure called Dirac structure"
- Bears on: (3) inter-phase (fluid-particle) transfer as a flux-force closure, the natural slot for LDF; (4) chemical potentials as efforts.

### P5. Irreversible port-Hamiltonian systems: A general formulation of irreversible processes with application to the CSTR
- Authors: Hector Ramirez, Bernhard Maschke, Daniel Sbarbaro. 2013 (online 2012). Chem. Eng. Sci. doi:10.1016/j.ces.2012.12.002
- Fetched: OpenAlex record + HAL API abstract (hal-00872274; the HAL PDF exists but the file download is behind the Anubis wall).
- Supports: IPHS = quasi-port-Hamiltonian system whose structure matrix is modulated by a positive function times the Poisson bracket of entropy and energy; energy conserved; entropy balance has an irreversible creation term; CSTR balance equations written as IPHS.
- Quote: "expresses simultaneously the first and second principle of thermodynamics as a structural property"
- Bears on: lumped template for the latent model; (1) feed/outflow of the CSTR enter as input terms, not inside the bracket.

### P6. Boundary controlled irreversible port-Hamiltonian systems
- Authors: Hector Ramirez, Yann Le Gorrec, Bernhard Maschke. arXiv:2104.13459 (27 Apr 2021); Chem. Eng. Sci. 248 (2022) 117107.
- Fetched: arXiv abstract page + full PDF text.
- Supports: BC-IPHS on 1-D domains; boundary port variables extended for the dissipative terms so the system is passive w.r.t. conjugated boundary inputs/outputs (including an entropy-flux port pair); examples: 1-D fluid (Lagrangian), heat equation, and a diffusion-reaction process explicitly "without convection".
- Quote: "the first and second principle are satisfied as a structural property"
- Bears on: (1) imposed-velocity convection NOT covered; (2) boundary entropy flux appears as an extra boundary port pair; (3) reaction/diffusion as entropy-producing operators (direct analogue for LDF).

### P7. Irreversible Port-Hamiltonian Formulations for 1-Dimensional fluid systems
- Authors: Ahlam Ouardi, Arijit Sarkar, Hector Ramirez, Yann Le Gorrec. arXiv:2603.11260 (11 Mar 2026), physics.flu-dyn.
- Fetched: arXiv PDF full text.
- Supports: states that earlier IPHS parametrisations suit diffusion-type and non-convective hyperbolic systems but not convective ones; extends IPHS to Eulerian non-isentropic viscous fluid by modifying the differential operators, with an extra assumption for material-thermal coupling due to convection and a new boundary-port parametrisation (v, T, entropy flux at both ends). Velocity is a STATE (momentum balance), not imposed.
- Quote: "but it is not suitable for convective systems"
- Bears on: (1) strongest evidence that convection in IPHS was open until 2026; when included it enters the operator via the state velocity; imposed velocity not treated. (2) boundary ports carry convective entropy flux.

### P8. Conduction-Diffusion in N-Dimensional settings as irreversible port-Hamiltonian systems
- Authors: Luis Mora, Yann Le Gorrec, Hector Ramirez, Denis Matignon. arXiv:2603.11265 (11 Mar 2026).
- Fetched: arXiv PDF full text.
- Supports: N-D boundary-controlled IPHS for coupled conduction-diffusion; a footnote states the prior unified class is "Excluding systems with convection."
- Quote: "Excluding systems with convection."
- Bears on: (1) convection outside the standard IPHS class; (3) coupled diffusion/conduction with correct entropy production.

### P9. Open physical systems: from GENERIC to port-Hamiltonian systems
- Authors: A. Moses Badlyan, B. Maschke, C. Beattie, V. Mehrmann. arXiv:1804.04064v4 (2018).
- Fetched: arXiv PDF full text.
- Supports: GENERIC for open (boundary-driven) systems reformulated as boundary port-Hamiltonian systems: boundary brackets paired with conjugated port variables; exergy-like generating functional; energy balance splits convective and non-convective boundary transport.
- Quote: "the GENERIC framework for open physical systems may be reformulated into a port-Hamiltonian framework"
- Bears on: (2) inflow/outflow appear as boundary terms/ports, not bulk bracket terms; (1) velocity dynamic (Navier-Stokes-Fourier), not imposed.

### P10. Zhou, Hamroun, Le Gorrec, Couenne: Infinite Dimensional Port Hamiltonian Representation of Chemical Reactors (LHMNLC 2012, hal-00803504); ... of reaction diffusion processes (IFAC-PapersOnLine 2015, doi:10.1016/j.ifacol.2015.05.119); Distributed port-Hamiltonian modelling for irreversible processes (MCMDS 2017, doi:10.1080/13873954.2016.1237970)
- Fetched: HAL API abstract (2012) and OpenAlex abstracts (2015, 2017).
- Supports: non-isothermal reaction-diffusion reactor as infinite-dimensional PHS, internal energy and minus entropy (or enthalpy) as Hamiltonians, thermodynamic co-states; explicitly without convection.
- Quote: "mass transport diffusion and chemical reaction without convection"
- Bears on: (1) convection omitted again; (4) species chemical potentials as co-states; closest non-isothermal distributed template.

### P11. Dissipative boundary control systems with application to an isothermal tubular reactor
- Authors: Weijun Zhou, Boussad Hamroun, Yann Le Gorrec, Francoise Couenne. 2015. IFAC-PapersOnLine. doi:10.1016/j.ifacol.2015.10.230
- Fetched: OpenAlex abstract.
- Supports: isothermal tubular reactor as dissipative boundary control system; two parametrisations of the differential operator; boundary port variables derived and existence of solutions checked for Danckwerts boundary conditions.
- Quote: "check the existence of solutions in the case of Dankwert boundary conditions"
- Bears on: (2) Danckwerts inlet treated as a boundary-port closure (isothermal); (1) convection inside a dissipative (non-skew) operator.

### P12. Boundary energy-shaping control of an isothermal tubular reactor
- Authors: Alessandro Macchelli, Yann Le Gorrec, Hector Ramirez. 2016. MCMDS. doi:10.1080/13873954.2016.1232282
- Fetched: OpenAlex abstract.
- Supports: isothermal tubular reactor with diffusion and convection cast as a distributed port-Hamiltonian system for boundary energy-shaping control.
- Quote: "Due to the presence of diffusion and convection, such example ... is not trivial."
- Bears on: (1) imposed-velocity convection handled in a linear, isothermal PHS with a quadratic (non-thermodynamic) Hamiltonian.

### P13. Port Hamiltonian representation of heat exchanger
- Authors: Bertrand Zitte, Boussad Hamroun, Francoise Couenne, Isabelle Pitault. 2019. 3rd IFAC TFMST workshop (hal-02385418).
- Fetched: HAL API abstract.
- Supports: irreversible port-Hamiltonian representation of a heat exchanger built from compartments.
- Quote: "an irreversible port Hamiltonian representation of a heat exchanger"
- Bears on: (1) flowing streams handled compartment-wise (tanks in series), i.e. convection as port flows between compartments.

### P14. Port-Hamiltonian Sliding Mode Observer Design for a Counter-current Heat Exchanger
- Authors: J. Kadima Kazaku, D. Dochain, J. J. Winkin, M. Mukepe Kahilu, J. Kalenga Kaunde Kasongo. 2020. IFAC-PapersOnLine. doi:10.1016/j.ifacol.2020.12.1066
- Fetched: OpenAlex abstract.
- Supports: distributed counter-current heat exchanger in port-Hamiltonian form; convection enters through "dissipation ports"; the Stokes-Dirac structure obtained by opening them is conservative.
- Quote: "opening only the dissipation ports due to the convection phenomenon, is conservative"
- Bears on: (1) one existing choice: imposed-velocity convection as a dissipation-port term outside the skew structure.

### P15. Dissipative Boundary Control for an Adiabatic Plug Flow Reactor With Mass Recycle
- Authors: M. A. Zarate-Navarro, S. S. Dubljevic, A. Campos-Rodriguez, E. Aguilar-Garnica, J. P. Garcia-Sandoval. 2022. IEEE Access. doi:10.1109/access.2022.3157335
- Fetched: OpenAlex abstract.
- Supports: convective PFR consistent with both laws; entropy functional as storage function, internal entropy production as dissipation; entropy difference at the boundaries proportional to the supply rate.
- Quote: "usually not well defined for systems with mass flows"
- Bears on: (1)/(2) supports treating imposed-velocity advection as a boundary supply (port) in an entropy-based passivity framework.

### P16. Energy dissipation evaluation of temperature swing adsorption (TSA) cycle based on thermodynamic entropy insights
- Authors: Shuangjun Li, Shuai Deng, Li Zhao, Weicong Xu, Xiangzhou Yuan, Zhihao Guo, Zhenyu Du. 2019. Scientific Reports 9:16599. doi:10.1038/s41598-019-53398-6
- Fetched: OpenAlex abstract + full PDF from nature.com.
- Supports: lumped 4-step TSA cycle; heat and mass transfer decoupled; mass-transfer entropy generation = -dG/T with the integral Gibbs energy from adsorption potential theory (an isotherm integral of q d ln P, i.e. a spreading-pressure-type quantity); heat transfer dominates entropy generation.
- Quote: "the entropy generation distribution contributed by heat transfer decreases from 63.27% to 53.72%"
- Bears on: (5) cycle-level, not column-resolved, no LDF-resolved entropy production; (4) adsorbed-phase free energy from the isotherm integral.

### P17. Entropy Analysis of Temperature Swing Adsorption for CO2 Capture Using the Computational Fluid Dynamics (CFD) Method
- Authors: Zhihao Guo, Shuai Deng, Shuangjun Li, Yahui Lian, Li Zhao, Xiangzhou Yuan. 2019. Entropy 21(3):285. doi:10.3390/e21030285
- Fetched: OpenAlex abstract.
- Supports: spatially resolved (CFD) entropy-generation analysis of a TSA cycle based on non-equilibrium thermodynamics.
- Quote: "Based on the theory of non-equilibrium thermodynamics, the entropy analysis of TSA cycle is conducted"
- Bears on: (5) column-resolved entropy generation exists but as post-processing, not as model structure.

### P18. HARMONIA: A Unified Entropy-Driven Framework for Adsorption and Transport - From Molecular Stochastics to Macroscopic Kinetics
- Authors: Pavel Dytrych, Martina Dlaskova. 2025. ChemRxiv. doi:10.26434/chemrxiv-2025-c860f
- Fetched: OpenAlex record with abstract.
- Supports: claims ad hoc isotherm + LDF couplings lack thermodynamic consistency and can violate the 2nd law in reactor simulations; proposes a convex free-energy functional (Orlicz entropy) whose limits are classical isotherms and whose transport is a Wasserstein gradient flow with monotone entropy production; validated on CO2/Co-MOF-74 and N2/Mn-MOF.
- Quote: "coupled with disconnected kinetic laws (LDF, Fickian diffusion) in a manner that lacks thermodynamic consistency"
- Bears on: (3)/(4)/(7) closest recent claim that standard isotherm+LDF models are thermodynamically inconsistent; purely dissipative gradient flow, not GENERIC/IPHS, no column advection structure.
- Note: chemrxiv.org (article page and public API) is behind a Cloudflare challenge for both WebFetch and curl; the live chemrxiv page itself could NOT be read. Claude in Chrome was not connected. Only the OpenAlex record was fetched.

### P19. Development of Energy Balances for Fixed-Bed Adsorption Processes: Thermodynamic Paths, Heat Capacities and Isosteric Heats
- Authors: Krista S. Walton, M. Douglas LeVan. 2005. Adsorption. doi:10.1007/s10450-005-5984-6
- Fetched: link.springer.com article page (abstract) via curl.
- Supports: energy balances for fixed beds derived via thermodynamic paths; with adsorbed-phase enthalpy properly defined, reference-state terms cancel and only gas-phase heat capacities appear; adsorbed-phase heat capacity may deviate strongly from gas-phase heat capacity when the isosteric heat is weakly T-dependent; path choice must avoid inconsistent heat-capacity definitions.
- Quote: "special attention should be given to the path used to derive the energy balance"
- Bears on: (6) and (7) directly: isosteric heat, its T-dependence and the adsorbed-phase heat capacity are linked (Kirchhoff-type relation, dq_st/dT ~ c_p,gas - c_p,ads); assuming a constant q_st AND an independently chosen constant c_p,ads is a known source of inconsistency.

### P20. Consistency of Energy and Material Balances for Bidisperse Particles in Fixed-Bed Adsorption and Related Applications
- Authors: Krista S. Walton, M. Douglas LeVan. 2003. Ind. Eng. Chem. Res. doi:10.1021/ie0303855
- Fetched: Crossref metadata only (title/authors/venue). ACS page 403 (Cloudflare). CONTENT SEEN-ONLY (via P19's own description: the 2003 method develops energy balances for bidisperse particles using thermodynamic paths).
- Quote: title only.
- Bears on: (6)/(7) origin of the "consistent energy balance" argument in fixed-bed adsorption.

### P21. Estimating the Heat Capacity of the Adsorbate-Adsorbent System from Adsorption Equilibria Regarding Thermodynamic Consistency
- Authors: Valentin Schwamberger, Ferdinand P. Schmidt. 2013. Ind. Eng. Chem. Res. 52(47):16958-16965. doi:10.1021/ie4011832
- Fetched: KIT repository record (publikationen.bibliothek.kit.edu/1000037321) with abstract.
- Supports: adsorbate-adsorbent heat capacity can be estimated from gas-phase properties, adsorption equilibria and isosteric heats alone; computing adsorbed-phase heat capacities along reversible equilibrium paths makes the residuals of the energy AND entropy balances significantly smaller (water/zeolite 13X).
- Quote: "the residuals of the energy and entropy balances are significantly smaller"
- Bears on: (6)/(7) strongest fetched evidence that heat capacity, isosteric heat and isotherm must be mutually consistent, and that inconsistency shows up as energy/entropy balance residuals.

### P22. Physical adsorption of gases: the case for absolute adsorption as the basis for thermodynamic analysis
- Authors: Alan L. Myers, Peter A. Monson. 2014. Adsorption. doi:10.1007/s10450-014-9604-1
- Fetched: link.springer.com article page (abstract) via curl.
- Supports: heats of adsorption undefined for excess isotherms at high pressure; converting to absolute adsorption allows a rigorous solution-thermodynamics framework (mixed-gas equilibria, isotherm equations, thermodynamic properties).
- Quote: "the special features of adsorption can be incorporated into a rigorous framework of solution thermodynamics"
- Bears on: (4) the adsorbed-phase chemical potential/entropy for the energy/entropy functionals should be built on absolute adsorption and solution thermodynamics (Myers' framework); spreading-pressure/grand-potential route.

### P23. Isosteric heats of adsorption from gas mixtures
- Authors: Narasimhan Sundaram, Ralph T. Yang. 1998. J. Colloid Interface Sci. 198(2):378-388.
- Fetched: SUNY research portal record (abstract).
- Supports: isosteric heats used in energy balances of adsorption process design are computed from a thermodynamically consistent multicomponent isotherm; temperature dependence of saturation capacity couples each component's heat to the others.
- Quote: "required in the energy balances of adsorption process design"
- Bears on: (6) heats in column energy balances should come from the isotherm's T-dependence (multicomponent case).

### P24. Isosteric heat of adsorption from thermodynamic Langmuir isotherm
- Authors: Htet Tun, Chau-Chyun Chen. 2021. Adsorption. doi:10.1007/s10450-020-00296-3
- Fetched: link.springer.com article page (abstract) via curl.
- Supports: isotherms at several temperatures fitted with the thermodynamic Langmuir model; isosteric heat obtained by Clausius-Clapeyron as a function of loading.
- Quote: "correctly estimates the isosteric heat of adsorption as a function of surface loading"
- Bears on: (6) standard route: q_st(n,T) = isotherm-derived via Clausius-Clapeyron, i.e. not an independent constant.

### P25. A non-equilibrium thermodynamics model for combined adsorption and diffusion processes in micro- and nanopores
- Authors: I. Santamaria-Holek, Z. J. Grzywna, J. M. Rubi. arXiv:1204.3841 (17 Apr 2012).
- Fetched: arXiv PDF full text.
- Supports: entropy production split into bulk and surface parts; surface part contains a term -(1/T) j (mu_1 - mu_2) for the bulk-to-surface flux; linear flux-force laws give j proportional to the chemical-potential difference; Langmuir kinetics recovered from this NET picture (isothermal).
- Quote: "assuming linear relationships between fluxes and forces"
- Bears on: (3) THE cleanest fetched statement that the adsorption (LDF-like) flux is an Onsager flux driven by mu_gas - mu_ads with entropy production j(mu_1-mu_2)/T >= 0; (4) adsorbed-phase chemical potential needed.

### P26. Nonequilibrium thermodynamics for open systems
- Authors: Hans Christian Ottinger. 2006. Phys. Rev. E 73, 036126.
- Fetched: journals.aps.org abstract page.
- Supports: GENERIC extended to open systems by separating bulk and boundary contributions to Poisson and dissipative brackets; applied to open-system hydrodynamics; appendix on surface excess variables at walls of a diffusion cell.
- Quote: "a clear distinction between bulk and boundary contributions to the Poisson and dissipative brackets"
- Bears on: (2) open inlet/outlet in GENERIC = boundary terms in the brackets (equivalently ports, cf. P9); (4) surface-excess variables are admissible GENERIC state variables.

### P27. GENERIC: Review of successful applications and a challenge for the future
- Authors: Hans Christian Ottinger. arXiv:1810.08470 (2018; 14th JETC 2017).
- Fetched: arXiv PDF full text.
- Supports: list of GENERIC applications dominated by complex fluids (polymers, blends, colloids, relativistic hydro, discretisations); boundary conditions and interfaces flagged as an open area; no adsorption/chromatography/packed-bed application listed.
- Quote: "Most of the applications of nonequilibrium thermodynamics deal with the modeling of bulk systems."
- Bears on: negative evidence: GENERIC has not been applied to adsorption columns in the GENERIC community's own review; (2) open boundaries flagged as a challenge.

### P28. A thermodynamic approach to the passive boundary control of tubular reactors
- Authors: N. Ha Hoang, Denis Dochain. 2013. 9th IFAC NOLCOS (Toulouse). PDF at skoge.folk.ntnu.no/prost/proceedings/nolcos-2013/papers/0018.pdf
- Fetched: full PDF text.
- Supports: availability (from entropy concavity) as storage function for boundary control of non-isothermal tubular reactors with convection, diffusion, conduction, reaction; passivity requires a dissipation condition tied to transport and reaction; without convection it holds automatically via entropy production.
- Quote: "For a simple study without convection, the dissipation condition holds"
- Bears on: (1) convection by imposed velocity is NOT automatically dissipative/structured; it breaks the automatic passivity and must be handled as boundary supply; (2) boundary inputs chosen from availability.

### P29. Passivity of Single-phase Thermodynamic Systems with Equilibrium Reactions
- Authors: Zicheng Cai, B. Erik Ydstie. 2022. IFAC DYCOPS. PDF at skoge.folk.ntnu.no/prost/proceedings/dycops-2022/files/0198.pdf
- Fetched: full PDF text (first pages).
- Supports: thermodynamic availability (convex, from concave entropy) as storage function; feed flow -> pressure and heating -> temperature maps are passive; open-system inflow/outflow enter as supply terms.
- Quote: "the mapping from the feed flow rate to pressure ... is passive"
- Bears on: (1)/(2) Ydstie-Alonso school: convective in/out flows are ports/supply rates of an availability storage function, not bracket terms.

### P30. Exergetic Port-Hamiltonian Systems Modeling Language
- Authors: Markus Lohmayer, Owen Lynch, Sigrid Leyendecker. arXiv:2402.17640v2 (28 Feb 2025).
- Fetched: arXiv PDF full text (scanned).
- Supports: compositional port-Hamiltonian modelling language (EPHS) with exergy as storage function guaranteeing 1st and 2nd law structurally; advection of mass/momentum handled as reversible (skew) energy exchange components in fluid examples; no adsorption/column example found.
- Quote: "represents reversible energy exchange (advection of momentum and mass)"
- Bears on: (1) the alternative view: advection as a reversible (skew) interconnection; but there the velocity is a state, not imposed.

### P31. Physically Consistent Neural Network Surrogates of Catalytic Reactors with Detailed Surface Kinetics
- Authors: Felix A. Doppel, Mauro Bracconi, Matteo Maestri. 2026. ACS Engineering Au. doi:10.1021/acsengineeringau.5c00106
- Fetched: Crossref and Semantic Scholar metadata only (title, authors, venue, Gold OA). ACS page behind Cloudflare in WebFetch, curl and the built-in browser. CONTENT SEEN-ONLY.
- Quote: title only.
- Bears on: ML-side neighbour (physically consistent reactor surrogates); content unverified this session.

### P32. Modeling and Simulation of Nitrogen Generation by Pressure Swing Adsorption for Power-to-Ammonia
- Authors: Marcus J. Schytt, Lorenz T. Biegler, John B. Jorgensen. arXiv:2604.09053 (10 Apr 2026).
- Fetched: arXiv abstract page.
- Supports: first-principles 1-D PSA column model coupling mass, energy and momentum balances with "thermodynamically consistent equations of state"; no entropy/port-Hamiltonian/GENERIC structure mentioned in the abstract.
- Quote: "thermodynamically consistent equations of state"
- Bears on: shows that "thermodynamically consistent" in current column modelling means consistent EOS, not structure-preserving dynamics.

### P33. AIM: A User-friendly GUI Workflow program for Isotherm Fitting, Mixture Prediction, Isosteric Heat of Adsorption Estimation, and Breakthrough Simulation
- Authors: Muhammad Hassan, ..., Yongchul G. Chung. arXiv:2504.20713 (v1 29 Apr 2025, v2 28 Oct 2025).
- Fetched: arXiv abstract page.
- Supports: practical tool chain: 13 isotherm models, isosteric enthalpy via Clausius-Clapeyron/virial, IAST, non-isothermal multicomponent breakthrough validated on CO2/H2/N2.
- Quote: none taken (page gives no balance equations).
- Bears on: (6)/(7) typical practice computes q_st from isotherms but whether the column energy balance uses it consistently is not stated.

## SEEN-ONLY leads (not fetched or content not readable this session)
- A. L. Myers, "Thermodynamics of adsorption in porous materials", AIChE J. 48 (2002) 145, doi:10.1002/aic.690480111 (Wiley 403; author PDF on seas.upenn.edu now 404). Search snippet: classical q_st formula applies only to a pure perfect gas; its link to adsorbed-phase enthalpy "has led to considerable confusion".
- S. A. Al-Muhtaseb, J. A. Ritter, J. Phys. Chem. B 1999, doi:10.1021/jp984110s and doi:10.1021/jp990556c (T-dependence of isosteric heat and adsorbed-phase heat capacity; ACS 403).
- S. Sircar et al., "Isosteric Heat of Adsorption: Theory and Experiment", J. Phys. Chem. B 1999, doi:10.1021/jp9903817; S. Sircar, "Temperature Dependence of the Isosteric Heat of Adsorption", Adsorption Sci. Technol. 2001, doi:10.1260/0263617011494646 (SAGE 403); S. Sircar, "Gibbsian thermodynamics and column dynamics for adsorption of liquid mixtures", IECR 1993, doi:10.1021/ie00022a028.
- Z. Guo et al., "Non-equilibrium thermodynamic analysis of adsorption carbon capture: contributors, mechanisms and verification of entropy generation", Energy 2020, doi:10.1016/j.energy.2020.118348 (no abstract retrievable).
- D. T. Kearns, P. A. Webley, "Application of an adsorption non-flow exergy function to an exergy analysis of a PSA cycle", Chem. Eng. Sci. 2004, doi:10.1016/j.ces.2004.05.018.
- J.-M. Simon, "Non-equilibrium Thermodynamics Applied to Adsorption", chapter in Experimental Thermodynamics Vol. X (RSC 2015/16; Kjelstrup co-editor).
- Kjelstrup/Bedeaux, Non-Equilibrium Thermodynamics of Heterogeneous Systems (book) and NET for Engineers (doi:10.1142/10286); Kingston & Kjelstrup minimum entropy production in air-separation distillation (NTNU handle 11250/2644409, redirect not followed). No Kjelstrup/Bedeaux entropy-production-minimisation study of an adsorption column found.
- Alonso & Ydstie, Comput. Chem. Eng. 2001 (availability-based passivity of process systems); Lohmayer, Kotyczka, Leyendecker, "Exergetic port-Hamiltonian systems: modelling basics", MCMDS 27 (2021) 489, arXiv:2008.04091.
- Liang et al. 1997 J. Chromatogr. (NET/entropy-balance model of capillary electrophoresis separation); 1973 Pakistan J. Sci. Ind. Res. paper on entropy in chromatography.
- Ozdural, Alkan, Kerkhof 2004 J. Chromatogr. A non-equilibrium packed-bed model (not GENERIC).
- Baaiu PhD thesis (Lyon 1, c. 2008) on port-based adsorption-column modelling: not located.

## Additional search log
- Crossref: "consistency of energy balances for bulk separations by adsorption" (found Walton & LeVan 2003/2005), "heat capacity of adsorbed phase", "isosteric heat of adsorption thermodynamic consistency", "non-equilibrium thermodynamics adsorption surface Kjelstrup Bedeaux", "physically consistent machine learning surrogates reactor Maestri", "GENERIC reaction diffusion Grmela Ottinger", "port-Hamiltonian chromatography" (0 relevant), "Sircar isosteric heat", "Sircar Gibbsian thermodynamics".
- arXiv API: "port-Hamiltonian" AND adsorption (0); AND chromatography (0); metriplectic AND adsorption (0); "port-Hamiltonian" AND "packed bed" (0); "port-Hamiltonian" AND reactor (1, CSTR); "port-Hamiltonian" AND convection AND entropy (0); "irreversible port-Hamiltonian" (9, scanned); GENERIC AND adsorption AND thermodynamics (92, first 25 scanned, only 1204.3841 relevant); "linear driving force" AND entropy (0).
- Web search: GENERIC + chromatography/adsorption column/packed bed (none); NET + adsorption column + LDF + Onsager (none combining all); Kjelstrup entropy production minimisation + adsorption (none for adsorbers; membranes and distillation only); thermodynamically consistent fixed-bed model + isosteric heat (pieces only); entropy production + chromatography (only CE and 1973 qualitative paper).
- OpenAlex budget for this IP was exhausted mid-session (rate limit), after which Crossref, Semantic Scholar, arXiv API and publisher pages were used.

## Synthesis

**Most important outcome: does a structure-preserving formulation of an adsorption/chromatography column already exist (non-ML)?**
Partly yes. The Lyon LAGEP group (Baaiu, Couenne, Le Gorrec, Lefevre, Tayakout-Fayolle; with Maschke, Jallut, Eberard) published a port-based (bond graph / Dirac structure) distributed model of a multiscale ADSORPTION COLUMN (P1, MCMDS 2008), the general port-based mass-transport theory behind it (P4, 2009), and structure-preserving discretisation/reduction applied to adsorption processes (P2, P3). All are ISOTHERMAL by their own abstracts: no heat of adsorption, no thermal domain, no entropy-as-state, no IPHS modulation, no GENERIC brackets, and no learned model. Their full texts are paywalled and the HAL notices carry no file, so state variables and the exact convection/boundary treatment could NOT be verified this session (needs institutional access or the author). No GENERIC, metriplectic, or irreversible-port-Hamiltonian (non-isothermal) model of an adsorption, chromatography or packed-bed adsorber column was found in OpenAlex, HAL, Crossref, arXiv or web searches. The closest non-isothermal distributed IPHS/PHS models are reaction-diffusion reactors that explicitly EXCLUDE convection (P6, P8, P10) and a 2026 IPHS fluid paper where velocity is a dynamic state (P7). Paper 3B's novelty claim should therefore be: first NON-ISOTHERMAL, entropy-explicit (GENERIC/IPHS) formulation of an adsorption column with LDF and heat of adsorption, and first learned latent model with that structure; it must cite P1-P4 as the isothermal port-based precedent.

**(1) Convection by an imposed velocity.** No consensus treatment. Options found: (a) excluded entirely from the IPHS class (P6, P8, P10); (b) inside a modified skew operator, but only when velocity is a state of the fluid (P7, P30 "reversible advection"); (c) as a dissipative/non-skew "dissipation port" term (counter-current heat exchanger, P14) or within dissipative operator parametrisations of an isothermal tubular reactor (P11, P12); (d) as boundary supply in entropy/availability-based passivity frameworks, where convection is exactly what breaks automatic passivity (P15, P28, P29); (e) compartment/tanks-in-series ports (P13). For an imposed velocity, the defensible choice consistent with the literature is: advection is an external, non-dynamic transport operator whose net effect on the total energy and entropy is a boundary flux (port), not part of the Poisson/L operator; inside the domain it should be energy- and entropy-neutral up to boundary terms (skew w.r.t. the L2 pairing for incompressible imposed flow), with no claim of Jacobi identity.

**(2) Open boundaries / Danckwerts inlet.** In GENERIC, open boundaries are bulk-vs-boundary splits of the brackets (P26) equivalent to boundary port pairs (P9); in BC-IPHS, boundary ports are extended with an entropy-flux pair (P6, P7). Danckwerts conditions have been analysed as boundary-port closures for an isothermal tubular reactor (P11) and appear in P3-type adsorption-column port models. Treating inlet/outlet/wall as ports in Paper 3B matches this literature.

**(3) LDF as entropy-producing flux.** NET gives the adsorption flux as an Onsager flux driven by the chemical-potential difference between fluid and adsorbed phase, with surface entropy production -(1/T) j (mu_1 - mu_2) >= 0 (P25); port-based models put inter-phase transfer in a dissipative flux-force closure (P1, P4). The classical concentration-difference LDF is NOT automatically of this form; HARMONIA (P18) explicitly claims standard isotherm+LDF couplings can be thermodynamically inconsistent. Paper 3B should write LDF as k (mu_g - mu_a)-type or show its concentration form is a monotone function of the affinity.

**(4) Adsorbed-phase chemical potential and entropy.** Rigorous route: solution thermodynamics on absolute adsorption (Myers & Monson, P22; Myers 2002 SEEN-ONLY), i.e. spreading pressure / grand potential from the isotherm integral; Li et al. (P16) already use the isotherm integral (adsorption potential) as the adsorbed-phase Gibbs energy in entropy generation. Surface-excess variables are legitimate GENERIC states (P26 appendix).

**(5) Entropy-generation analyses of adsorption cycles.** Exist but as post-processing, not model structure: lumped TSA cycle (Li et al. 2019, P16; heat transfer dominates, 63.27% -> 53.72% with heat recovery); CFD column-resolved TSA (Guo et al. 2019, P17); Guo 2020, Kearns & Webley 2004 exergy of PSA (SEEN-ONLY). None embeds entropy production in the model's structure or proves its sign for the discretised/learned dynamics.

**(6) Isosteric heat consistent with the isotherm's T-dependence.** Supported: q_st is the Clausius-Clapeyron derivative of the isotherm (P24, P33), multicomponent heats must come from a thermodynamically consistent isotherm for energy balances (P23), and Walton & LeVan show energy balances must be derived along consistent thermodynamic paths with adsorbed-phase enthalpy properly defined (P19, P20).

**(7) Constant heat of adsorption + constant heat capacity is inconsistent?** Supported in substance, not as a one-line theorem: Walton & LeVan (P19) show adsorbed-phase heat capacity is tied to the T-dependence of the isosteric heat (deviating strongly from gas-phase c_p when q_st is weakly T-dependent) and warn of inconsistent heat-capacity definitions; Schwamberger & Schmidt (P21) show inconsistent choices leave non-zero energy AND entropy balance residuals. Hence a model with constant q_st, an independently chosen constant c_p,ads, and an isotherm whose Clausius-Clapeyron heat is not that constant is thermodynamically inconsistent; the paper can cite P19/P21 (and P23) for this rather than claim it as new. Deriving energy and entropy from one free-energy function of (c, q, T) removes the issue by construction.
