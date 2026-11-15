The IOP article pages for MLST could not be opened. Every request to iopscience.iop.org, including the /pdf links, is redirected to a Radware bot check (validate.perfdrive.com), and I did not try to get around it. So I read each paper as follows:
- **Published version:** only one paper (P8), from the University of Glasgow repository, CC BY.
- **arXiv versions:** all the other papers. These may differ from the published text, and their data-availability statements may be missing or different.
- **Bibliographic data:** volume, article number, publication date, author list and reference count come from Crossref and OpenAlex. All sources were opened on 2026-11-16.

Search: Crossref queries on journal ISSN 2632-2153, from 2023 on.
- **Pre-registration and negative results:** queries for "pre-registered", "preregistration", "registered report", "negative result" and "does not help" returned no relevant MLST paper. I found no pre-registered or null-result surrogate paper in MLST.
- **Metriplectic and GENERIC:** "metriplectic" returned no MLST paper. "GENERIC thermodynamics" also returned nothing. P1 below is the closest MLST paper of that kind.

## The papers

**P1. Schaffer and Exl, "An energy-based conservative-dissipative latent neural evolution operator for magnetization dynamics", MLST 2026 (accepted manuscript, published online 2026-11-09), doi:10.1088/2632-2153/aebff4. Read from arXiv 2609.04530v2 (24 Oct 2026). This is the closest match to 3B.**
- **Length and structure:** 37 pages, 19 figures, 6 tables, 34 references. Sections: Introduction; Problem formulation (including the energy decay of the reference dynamics); Model architecture; Latent energy-based dynamics (antisymmetric J plus positive-definite R, coercivity, spectra); Learning from snapshot windows; Differentiation, integration and scaling; Relation to structured dynamical models; Numerical results (implementation, comparisons, time-resolved errors, operator diagnostics, retention of equilibria beyond the training horizon); Limitations and prospective applications; Conclusion.
- **Abstract:** 10 sentences. Contribution, then mechanism, then a theory claim hedged "under the stated assumptions", then a scope disclaimer (the scalar is a structural constraint, not the physical energy), then the training set-up, inference, benchmark (NIST muMAG Standard Problem 4), comparators, a hedged result ("competitive" accuracy; better retention of stationary states "in selected longer rollouts"), and an inductive-bias conclusion.
- **Baselines:** a dissipative-only model and a structured J-R model, each with three kinds of scalar, plus an unstructured latent neural ODE.
- **Statistics:** 3 seeds per configuration (2 in one case, because a job hit its time limit). Results are mean ± sample SD. The paper says plainly that these are not confidence intervals and not evidence of a general ranking. No statistical tests.
- **Guarantees:** analytic propositions only (the scalar decreases, trajectories stay bounded, LaSalle-type convergence to the critical set). The authors note that a finite-tolerance solver need not keep the decrease at every step. They report that the decoded physical state breaks the unit-norm constraint (maximum deviation about 0.10 and 0.92).
- **Wording:** heavily hedged, for example "does not establish that an antisymmetric operator is universally required". The limitations are listed: fixed hyperparameters, models not matched in parameter count, no break-even cost analysis.
- **Code:** on Zenodo, doi:10.5281/zenodo.22281254.
- **What a 3B reviewer would compare:** this is the direct MLST precedent for "structured versus unstructured latent ODE". 3B goes further: more seeds, pre-registered tests, an open system with inflow and outflow, decisions at design level, and checks of conservation and entropy on the decoded state, which P1 does not have.

**P2. Goyal, Yildiz and Benner, "Deep learning for structure-preserving universal stable Koopman-inspired embeddings for nonlinear canonical Hamiltonian dynamics", MLST 6 015063 (2025), doi:10.1088/2632-2153/adb9b5, 67 references. Read from arXiv 2308.13835v1 (Aug 2023), which predates the published version.**
- **Structure:** Introduction; Symplectic transformation; Data-driven canonical Hamiltonian systems (with a stability guarantee); results on low-dimensional systems (pendulum, oscillator, Lotka-Volterra); extension to high-dimensional data (nonlinear Schrödinger, wave); Conclusions; three appendices.
- **Figures and tables:** about 15 figures, including violin plots of per-trajectory error over 25 test initial conditions and best and worst trajectories shown side by side. 3 tables of L2 errors.
- **Abstract:** 7 sentences, from motivation to background, aim, method, extension, the constraint, and the result.
- **Baselines:** quadratic embeddings and OpInf-Ham, both structure-preserving. No unstructured baseline in v1.
- **Statistics:** no seeds or confidence intervals. The spread shown is over test initial conditions.
- **Guarantees:** symplecticity is a soft penalty; stability is proved by theorem; checking is by long rollouts past the training window (about 2.5 to 12 times longer).
- **Data and code:** not visible in v1.

**P3. Nguyen and Sandfeld, "Inductive bias and out-of-distribution generalization in deep learning surrogates for 2D PDE dynamics", MLST 7 055055 (2026), doi:10.1088/2632-2153/aeb34b, 52 references. Read from arXiv 2601.08404. This is the benchmark-type precedent.**
- **Structure:** Introduction; Model and architecture (datasets, architecture, training set-up, metrics including physics-aware measures, software); Results (in-distribution, out-of-distribution, data efficiency, network behaviour); Discussion; Conclusion; then Authors' contributions, Competing interests, Data availability and Code availability; appendices A to H.
- **Figures and tables:** about 31 figures (11 in the main text) and Table 1 listing the PDE families and datasets: 6 PDE families, 10 datasets.
- **Abstract:** 8 sentences. Why PDEs matter, then the task, scope (autoregressive, out-of-distribution, small data), the model, the evaluation (5 PDE families against 4 baselines in one training set-up), the main result, secondary results, and a hedged interpretation ("strong contenders").
- **Baselines:** ViT, AFNO, PDE-Transformer and KAN-UNet, all trained under one protocol.
- **Metrics:** RMSE, PSD cosine similarity, and physics diagnostics (mass, energy, dislocation density).
- **Statistics:** mean ± values, but the number of runs is not stated. No tests.
- **Out-of-distribution:** defined explicitly as a shift in the family of initial conditions only. Other kinds of shift are named as future work.
- **Limitations:** explicit, including "we do not impose any hard conservation constraints".
- **Data and code:** anonymised supplementary material for review, made public on acceptance.

**P4. Lenau, Dimiduk and Niezgoda, "Training variation of physically-informed deep learning models", MLST 7 035005 (2026), doi:10.1088/2632-2153/ae5e18, 41 references. Read from arXiv 2510.03416. This is the statistics and reproducibility precedent.**
- **Structure:** IMRaD (Introduction, Methods, Results, Discussion, Conclusions) plus supplementary figures.
- **Figures and tables:** 13 figures and 4 tables, including a table comparing 10 runs with 30 runs.
- **Abstract:** 5 sentences. It opens on the problem that training reliability is rarely reported, then motivates reporting variation for fair comparison, the case study, the finding, and recommended practice.
- **Statistics:** 30 training sessions per method, seeds not fixed. Mean and SD, plus a bootstrap (10,000 resamples) of the variation estimate against the number of runs. Conclusion: about 15 sessions are needed. No confidence intervals or tests, although "significantly" is used.
- **Results:** reported openly, including trade-offs; for example, the best baseline run beats the best physics-informed run.
- **Code and data:** github.com/mesoOSU/PB-GAN.
- **Why it matters for 3B:** MLST published a paper whose main contribution is that physics-informed models must be compared across many seeds. A reviewer may hold 3B to that standard, and 3B's seed counts and pre-registered statistics fit it directly.

**P5. Feng, Deng, Zhang, Fan and Wu, "Mass-conserved hierarchical multi-resolution deep learning for long-term global atmospheric pollutant transport prediction", MLST 7 055032 (2026), doi:10.1088/2632-2153/aeabf8. This is the published PDF, from eprints.gla.ac.uk/396312.**
- **Dates:** received 1 May 2026, revised 16 Sep, accepted 23 Sep, published 6 Oct 2026. That is about 5 months from receipt to publication.
- **Front matter:** CC BY 4.0, keywords, an online supplement.
- **Length and structure:** 29 PDF pages (including the IOP cover page), 15 figures, no tables, about 69 references. Sections: Introduction; Related works; Preliminaries (including problem formulation and data collection); Method; Experiments (which open with "Limitations of existing autoregressive models": diverging and identity-collapse baselines are shown first, then the daily, weekly and monthly results, ablations, robustness to noise, cost); Conclusion; Acknowledgments; Data availability statement; Conflict of interest.
- **Abstract:** one paragraph. Long-horizon bottleneck, then the application need, the method, the hard constraint (a Lagrange multiplier), and quantified results (relative error 0.0198 per month; mass change "at the hardware precision", about 1e-16).
- **Claims:** strong, for example "for the first time". One limitation sentence in the Conclusion.
- **Data statement:** the IOP template wording "Restrictions apply. Data cannot be shared publicly via a URL/DOI", with data available on request. This passed review, so MLST accepts restricted data. 3B's fully open archive goes beyond that.
- **Why it matters for 3B:** a hard conservation projection on an advective transport surrogate, checked by plotting mass drift over time. This is a direct precedent for 3B's projection arm and its conservation metrics.

**P6. Nam, Anderson, Choi, Le, Foord, Cho, Jeong and Cho, "Physics-informed latent space dynamics identification for time-dependent NLTE atomic kinetics", MLST 7 055046 (2026), doi:10.1088/2632-2153/aeb227, 43 references. Read from arXiv 2604.16664.**
- **Structure:** physics background (rates, conservation constraints, steady state), then the LaSDI framework (latent space dynamics identification) with DMDc, then the physics-informed version (Hurwitz stability, steady-state consistency, regularisation), set-up and data, results (identified coefficients, reconstruction, the role of the constraints, efficiency), and conclusion.
- **Figures and tables:** 7 figures, 1 table.
- **Abstract:** 6 sentences, ending on a strong claim that physics-informed design "rather than data fitting alone, is essential".
- **Baselines:** the comparison with the unconstrained model is an ablation shown in figures only.
- **Statistics:** none; single representative runs.
- **Constraints:** soft penalties throughout.
- **Code statement:** none in the arXiv text.

**P7. Xie, Tang and Tang, "Latent space dynamics learning for stiff collisional-radiative models", MLST 5 045070 (2024), doi:10.1088/2632-2153/ad9ce7, 34 references. Read from arXiv 2409.05893v2.**
- **Structure:** autoencoder plus flow-map latent dynamics; 30 figures, no tables.
- **Baselines:** none. Single runs; mass conservation is a soft penalty.
- **Wording:** strong ("accurately predict"), with a "proof-of-concept" caveat. No data or code statement in the arXiv text.
- **Why it matters:** it shows how low the acceptance bar has sometimes been. 3B should not be pitched at this level.

**P8. Bishnoi, Bhattoo, Jayadeva, Ranu and Krishnan, "Discovering symbolic laws directly from trajectories with Hamiltonian graph neural networks", MLST 5 035049 (2024), doi:10.1088/2632-2153/ad6be6, 41 references. Read from arXiv 2307.05299v1.**
- **Structure:** Nature-style layout (results first, then Outlook, then Methods, then a Supplement). Main text has 4 figures and 1 table, plus a large supplement.
- **Metrics and baselines:** energy-violation, momentum and rollout errors averaged over 100 initial conditions. Structured baselines only (HNN and HGN), placed in the supplement. No seeds or error bars.
- **Wording:** strong ("strictly follows", "superior"). The limitation sits in the Outlook ("only particle-based systems with at most two-body interactions").

**P9. Kumar and Mukherjee, "Investigating the ability of PINNs to solve Burgers' PDE near finite-time blowup", MLST 5 025063 (2024), doi:10.1088/2632-2153/ad51cd, 73 references. Read from arXiv 2310.05169v2.**
- **Structure:** an "Informal summary of our results" section near the start, then theorems, then experiments run to test the theorems (bound against true risk, correlation as the blow-up time is approached), then appendices with proofs. 3 to 5 figures, 6 seeds in one study.
- **Wording:** cautious. Bounds are often "vacuous"; stability in higher dimensions is "an open question".
- **Why it matters:** it is the cautionary, theory-then-check kind of paper.

**P10. Shen, Song and Zhang, "Two-stage projection physics-informed neural networks for preserving conservation laws", MLST 7 055035 (2026), doi:10.1088/2632-2153/aeab20, 39 references.** No arXiv copy, and the DOAJ page returned 403, so the content was not read. Bibliographic data only. 3B should cite it as projection-to-conservation prior art in MLST.

## The common pattern of a successful MLST paper of this type
1. **Length and figures.** 20 to 37 pages in IOP single-column format, 7 to 31 figures, often few or no tables. Results are carried by rollout-error-against-time curves, field snapshots set against the reference, and diagnostic plots of conserved quantities or energy over time (P1, P5, P8). An appendix or supplement of 10 to 20 extra figures is normal (P3, P8). References run from 34 to 73.
2. **Structure.** Problem and physics, then a section on structure or method, often with propositions or theorems (P1, P2, P6, P9), then data and set-up, then results, then a Discussion or a stand-alone Limitations section, then Conclusion. A separate "Relation to structured models" or Related-work section is common. P5 opens its Experiments with a section showing how the existing baselines fail, before presenting the new model.
3. **Abstract.** One paragraph of 5 to 10 sentences: bottleneck, method, mechanism or guarantee, benchmark and comparators, a quantified or hedged result, a one-line conclusion. The careful papers (P1, P3, P4) put a scope disclaimer inside the abstract.
4. **Baselines and statistics are the weak spot of the published work.** Most papers use single runs or no seeds (P2, P6, P7, P8). The best use 3 seeds with mean ± SD and say these are not confidence intervals (P1). Only P4 uses 30 runs and a bootstrap. None of the papers I read reports a pre-specified statistical test, a confidence interval on a comparison between models, or matched parameter counts; P1 lists that last point as a limitation. On these points, 3B's design (pre-registered tests, verdict words, power analysis, multiple seeds, matched arms) goes beyond every precedent. A reviewer is likely to see that as the contribution, and the paper should say so explicitly.
5. **Structure claims are made analytically and checked by long rollouts past the training horizon.** Few papers check them on the decoded physical state; P1 admits this gap. P5's conservation-drift plot (about 1e-16) is the model for showing a hard constraint numerically. 3B should report conservation and entropy-production residuals on the decoded fields, plotted over cycles, for every arm.
6. **Wording.** The rigorous papers (P1, P3, P4, P9) hedge in a systematic way: "under the stated assumptions", "does not establish", "competitive", an explicit list of what is not claimed, and limitations in their own section. The weaker papers (P5, P6, P7, P8) make strong claims ("for the first time", "essential"). Those claims still got published, but they are what a careful reviewer attacks. 3B's pre-set verdict words fit the hedged style.
7. **Data and code.** MLST prints a "Data availability statement" in the IOP template (P5, published version). Restricted or on-request data has been accepted. The better papers give a GitHub or Zenodo DOI (P1, P4) or release anonymised material at review (P3). Back matter seen: Acknowledgments with grant numbers, Data availability, Conflict of interest, sometimes Authors' contributions and Code availability (P3, P4).
8. **Turnaround.** One data point: P5 took about 5 months from receipt to publication (received 1 May, published 6 Oct 2026).

## What a reviewer of 3B would compare against
- **P1:** structured against unstructured latent ODE, the same J/R decomposition idea, 3 seeds.
- **P3 and P4:** benchmark protocol and the standard for variation across seeds.
- **P5 and P10:** a hard conservation projection.
- **P6 and P7:** constrained latent dynamics, LaSDI-type.

3B should cite P1, P3, P4, P5 and P10 at least. It should state its differences: open dissipative systems with inflow and outflow, constraint versus information versus projection separated, pre-registration and power analysis, nulls reportable, and metrics at the level of design decisions.

Sources: Crossref API (journal 2632-2153 works and per-DOI records); OpenAlex works API; arxiv.org/abs/2609.04530, arxiv.org/html/2609.04530, arxiv.org/html/2601.08404, arxiv.org/html/2510.03416, arxiv.org/html/2409.05893, arxiv.org/abs/2308.13835, arxiv.org/html/2308.13835, arxiv.org/html/2310.05169, arxiv.org/html/2307.05299, arxiv.org/html/2604.16664; eprints.gla.ac.uk/396312 and its published PDF. My downloaded copy of P5 and its extracted text are in `C:\Users\ABDULH~1\AppData\Local\Temp\claude\c6eff6e074d30b33\4add04dd-e721-4eee-b705-23ef12c2b27a\scratchpad\mlst_dl\` (hmr.pdf, hmr.txt). Nothing was written to any repository.