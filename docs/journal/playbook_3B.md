# Paper 3B journal playbook: Machine Learning: Science and Technology (MLST, IOP)

Compiled 2026-11-16 from four research reports run that day: the lessons report, the journal report, the recent-papers report and the novelty report. I also read the 3B repo's `README.md`, `PREREG_P3B_draft.md` (§0, 1, 6, 8, 9) and `docs/freeze_decisions_3B.md`, without editing anything.

Every journal fact below carries its source and fetch date. Facts marked **(2026-11-01, not re-checked)** come from the project file `MLST_FORMAT_PLAN.md` and were not opened live today. Facts marked **(hub, 2026-11-08)** come from `T4_venue_3B.md`. iopscience.iop.org redirected every automated request to a Radware bot check on 2026-11-16, and the reports did not try to get past it. So MLST metrics and article pages were not read live; the papers were read from arXiv, apart from one published PDF.

---

## 1. Decision

**Target: keep MLST. Article type: Research paper.**

The reasons, each tied to its evidence:

1. **Scope fit.** The MLST About page (publishingsupport.iopscience.iop.org/journals/machine-learning-science-and-technology/about-machine-learning-science-technology/, read 2026-11-16) lists these scope items:
   - "Simulation of materials, chemical processes and biological systems"
   - "New (physics inspired) learning algorithms"
   - "Benchmark studies"
   - "Codes and datasets"

   3B is all four: a MOF column, a thermocline store, metriplectic latent models and a controlled multi-arm benchmark.
2. **Direct precedent with a gap 3B fills.** Schaffer and Exl, MLST 2026, doi:10.1088/2632-2153/aebff4, compares a structured J–R latent model against an unstructured latent neural ODE. It uses 3 seeds, no tests, no parameter matching, no checks on the decoded state, and a closed system. Lenau et al., MLST 7 035005 (2026), doi:10.1088/2632-2153/ae5e18, was published with seed variation of physics-informed models as its main contribution. Together they show the editors take the "does physics actually help, measured properly" question, and that 3B's statistics exceed every precedent read (recent-papers report).
3. **No fee.** The APC is £2,680 / €3,215 / $3,350 (journal report). The TÜBİTAK–IOP Read & Publish agreement (1 Jan 2026 to 31 Dec 2028) covers it:
   - MLST is on the official funded list ("List D … MLST … Gold OA", ULAKBİM xlsx downloaded 2026-11-16).
   - YTÜ is on the EKUAL 2026 member list.
   - Condition: the **corresponding author** must carry the EKUAL (YTÜ) affiliation. One library (İYTE, 10 Mar 2026) adds that the corresponding author must be active academic staff. A YTÜ faculty member as corresponding author is therefore the safe choice; confirm with the YTÜ library (§7).
4. **The policies reward 3B's design.** IOP's open-data policy is mandatory, and review can be single-anonymous (journal report, 2026-11-16). 3B's public repo, CI and frozen protocol fit this.

**Why a Research paper and not a Benchmark.** The Benchmark type exists to "characterize the performance of differing methods / models / codes /algorithms / software" (About page, 2026-11-16). 3B instead tests falsifiable hypotheses with a decision rule. Research papers are "normally not more than 8500 words". Letters (≤5,000 words, urgency justified) do not fit.

**There is no registered-report article type** (About page, 2026-11-16). The frozen protocol should therefore get its independent timestamp elsewhere: a Zenodo DOI before any confirmatory run, and optionally PCI RR (lessons item 8). The paper then cites that DOI.

**Alternatives, if MLST fails or the policy conflicts below cannot be settled:**
- **Journal of Computational Physics.** The safe option if the formulation and solver comparison carry the paper. The hybrid subscription route has no fee. Hub (2026-11-08): IF 3.9, 7 days to first decision, 211 days to acceptance; the site was blocked on 2026-11-16.
- **Nature Computational Science.** Only if the result is strongly mechanistic on both systems; its pages were blocked.
- **Excluded:**
  - Digital Discovery: overlaps with Paper 1, which is going there.
  - npj Computational Materials: the TÜBİTAK–Springer Nature quota is used up (Altınbaş library, 9 Jul 2026).
  - Machine Learning: Engineering: not on the TÜBİTAK list.
  - Data-Centric Engineering: prestige too low (2024 IF 2.8).

**Two policy conflicts the author must settle before choosing MLST for good.** Both are from IOP's generative-AI page, publishingsupport.iopscience.iop.org/questions/generative-ai-tools/, read 2026-11-16.

- **A. Revision workflow.** IOP prohibits uploading reviewer reports to AI tools and AI-generated responses to reviewers. The Paper 1 and Paper A revisions used agents for exactly this. At MLST the author reads the reports and writes the replies; AI may only polish language. The author must accept this up front.
- **B. References and the disclosure wording.**
  - AI may not generate or modify the reference list. The author compiles the final list and checks each source against the original.
  - The disclosure sits in the Acknowledgements and states how the outputs were checked. The template reads "The authors reviewed, validated and critically assessed all AI-assisted outputs". That conflicts with the author's rule (`PAPER_WORKFLOW_LESSONS.md` §1b.10; memory note `ai-declaration-wording`).
  - Proposed resolution, for the author to approve: describe the checks as procedures, not as the author supervising AI. For example: "every number is generated from the archived results files by the build script; every reference was checked by the author against its source."
  - If no honest wording fits, the 2026-11-15 rule applies: change the journal, never the disclosure.

---

## 2. Dated compliance table

| # | Requirement | Live rule | Source and date | What 3B must do |
|---|---|---|---|---|
| 1 | Length | Research paper "normally not more than 8500 words"; what counts toward it is not stated | MLST About page, 2026-11-16; counting rule: MLST_FORMAT_PLAN (2026-11-01, not re-checked) | Write to about 8,300 words from the start, counting the main text including captions to be safe. Run a word-count gate in the build. Keep every pre-registered verdict, its interval and every quantified caveat in the main text. |
| 2 | Abstract | "Not normally be more than 300 words"; self-contained; no references, equations, figure or table numbers, or undefined acronyms | publishingsupport.iopscience.iop.org/authoring-for-journals/writing-journal-article/ (undated page; Word template Jan 2026), 2026-11-16 | Aim for about 250 words. No "MOF", "ML", "ODE", "FNO" or arm codes: write "metal-organic framework", "latent differential-equation model", "coarse consistent solver". Add a gate that rejects acronyms and digits not bound to results. |
| 3 | Structure | Introduction, Methods, Results, Discussion, Conclusion, then Acknowledgements (conflicts of interest, funding, CRediT), then References | same page, 2026-11-16 | Use the outline in §3 (method sections may carry descriptive names, as P1 and P3 do). Acknowledgements hold the AI statement, "no competing interests", "no specific funding" (memory: no grant), and CRediT if a co-author is added. |
| 4 | Figures | Vector EPS or PDF preferred; text 8–12 pt at final size; widths 8.5 or 15 cm; colour never the only cue and distinguishable in greyscale; captions self-contained; no colour in tables | same page, 2026-11-16 | One style module (lessons item 14). Font floor 8 pt, asserted in `save()` and measured in the compiled PDF. Okabe–Ito plus marker or line style per arm. Widths fixed at 85 or 150 mm. Captions without acronyms. |
| 5 | References | Vancouver or Harvard, used consistently; DOIs or arXiv IDs; cite the Version of Record | same page, 2026-11-16 | One style. Cite P1 by its MLST DOI, not its arXiv number. Conference papers without a DOI (Greydanus, Cranmer, Lee–Trask–Stinis, Botev, Gruver) by arXiv ID. **The author compiles the final list** (IOP AI policy). |
| 6 | Data | IOP open-data policy: the data needed to verify the results must be public at publication; a data availability statement is a condition of publication, with DOIs or links; "upon request" is not acceptable | publishingsupport.iopscience.iop.org/iop-publishing-open-data-policy (undated), 2026-11-16 | Versioned Zenodo DOI for the reference data, trained-model outputs and results JSON. Reserve the DOI early. The statement **cannot be amended after publication** (MLST_FORMAT_PLAN, 2026-11-01, not re-checked), so get it right first time. |
| 7 | Code | Code in a trusted repository with a DOI; a GitHub copy must also go to a DOI repository; the MLST page asks for OSI principles, code in the supplement or linked | data-availability policy page and About page, 2026-11-16 | GitHub repo plus a Zenodo release DOI. The repo already has a LICENSE; confirm it is OSI-approved. Scripts rebuild every figure and number; CI badge. Scan for paths and tokens before the deposit. |
| 8 | Supplementary material | Up to 50 MB per file, 150 MB in total; each file has a title (≤30 characters) and a description (≤30 words) | writing-journal-article page, 2026-11-16 | Short titles. Large data stays on Zenodo. |
| 9 | Supplement and review | Supplement "not included in peer review by default" | MLST_FORMAT_PLAN (2026-11-01, not re-checked) | Proofs a referee must check (non-negative production, the Bregman upwind term, the Paper 1 limit) go in an **appendix**, not the supplement. Re-check this rule live (§7). |
| 10 | Submission format | A single PDF; the iopjournal LaTeX class is optional (its `[anonymous]` option removes author details); ORCID recommended | writing-journal-article page, 2026-11-16 | Use the iopjournal class and give ORCIDs (author 0009-0007-0374-6755, plus a verified co-author ORCID). |
| 11 | Review model | Single- or double-anonymous, author's choice; publishing the reports is opt-in; usually two reviewers; criteria are "quality, scientific rigour, novelty, significance to the field, and presentation"; IOP rejection "typically above 50%" and sound-but-limited-interest papers are usually rejected | About page and .../authoring-for-journals/review-process-journals/, 2026-11-16 | **Single-anonymous**: the Zenodo protocol, the repo and Paper 1 all carry the author's name. Argue significance in the abstract, the introduction and the cover letter, not only rigour. |
| 12 | Generative AI | Material use disclosed in the Acknowledgements (tool, version, use, how outputs were checked); research use also in the Methods; prompts and logs encouraged as supplementary material; routine language editing exempt. Prohibited: fabricating data, AI-generated or modified reference lists, AI-generated responses to reviewers, uploading reports to AI, hidden prompts | publishingsupport.iopscience.iop.org/questions/generative-ai-tools/ (undated), 2026-11-16 | Resolve conflicts A and B (§1) before submission. Put the statement in the Acknowledgements; the Methods describe any AI use in the research pipeline. Any AI-made schematic must be clearly marked as distinct from data. |
| 13 | Preprints | Encouraged | About page, 2026-11-16 | Post to arXiv (or ChemRxiv) at submission. |
| 14 | Fees | Gold OA APC £2,680 / €3,215 / $3,350; covered by TÜBİTAK–IOP R&P 2026–2028 for an EKUAL-member corresponding author | journal report: ULAKBİM announcement docx, funded-list xlsx, EKUAL list xlsx, all downloaded 2026-11-16 | YTÜ-affiliated corresponding author, preferably faculty. Confirm with the YTÜ library before submission. |
| 15 | Ethics | not found in the four reports (computational study, no human or animal data) | not found | Check the live IOP ethics page; probably a one-line "not applicable" (§7). |
| 16 | Competing interests | In the Acknowledgements block | writing-journal-article page, 2026-11-16 | "The authors declare no competing interests." |
| 17 | Novelty and negative results policy | not found: no MLST editorial on negative results, benchmarking or reproducibility in two searches | journal report, 2026-11-16 | Argue the value of a null result in the paper itself, citing McGreivy and Hakim 2024 (§5). |
| 18 | Metrics and timing | IF 4.2, 5-year IF 5.5, CiteScore 6.7 (hub, 2026-11-08); not re-verified 2026-11-16. One turnaround data point: P5 received 1 May 2026, accepted 23 Sep, published 6 Oct (about 5 months) | hub file; published P5 PDF (eprints.gla.ac.uk/396312), 2026-11-16 | Re-check in a normal browser before submission. |

---

## 3. What successful MLST papers of this type look like, and the 3B blueprint

### 3.1 The pattern in the recent papers

Ten papers were found and nine read; P10 (Shen et al.) gave bibliographic data only (recent-papers report, 2026-11-16).

- **Length.** 20–37 pages in IOP single-column format, 7–31 figures, few tables, 34–73 references. P1 has 37 pages, 19 figures, 6 tables and 34 references. P5 has 29 pages, 15 figures and about 69 references.
- **Order.** Problem and physics, then structure or method (often with propositions: P1, P2, P6, P9), then set-up, then results, then a stand-alone Limitations section or a Discussion, then Conclusion.
  - P1 has a separate "Relation to structured dynamical models" section.
  - P5 opens its Experiments by showing how the baselines fail before showing its own model.
  - P9 has an "Informal summary of our results" section near the start.
- **Abstract.** One paragraph of 5–10 sentences: bottleneck, method, mechanism or guarantee, benchmark and comparators, a quantified or hedged result, a conclusion. The careful papers (P1, P3, P4) put a scope disclaimer *inside* the abstract. P1's example: the scalar is "a structural constraint, not the physical energy".
- **Evidence figures.** Rollout error against time; field snapshots against the reference; conserved quantity or energy against time (P5 plots mass drift at about 1e-16, its model for showing a hard constraint numerically); violin plots per trajectory (P2); best and worst trajectories (P2).
- **Statistics are the weak spot of the field.**
  - Most papers use a single run (P2, P6, P7, P8). P1 uses 3 seeds with mean ± SD, explicitly "not confidence intervals". P4 uses 30 runs and a bootstrap.
  - None reports a pre-specified test, a confidence interval on a model comparison, or matched parameter counts; P1 lists the last as a limitation.
  - No MLST paper was found with "pre-registered", "registered report", "negative result" or "metriplectic" (Crossref, ISSN 2632-2153, 2023 on).
- **Wording.** The rigorous papers hedge systematically (P1, P3, P4, P9). The weaker papers claim "for the first time" (P5) or "essential" (P6). Those claims passed review, but they are what a careful referee attacks.
- **Back matter.** Data availability statement, Conflict of interest, sometimes Authors' contributions and Code availability (P3, P4, P5).

### 3.2 Concrete structure for 3B (about 8,300 words)

Title: keep the README question, "Constraint, information or projection? What thermodynamic structure buys learned reduced models of open dissipative systems". It asks the question, which is the Paper A pattern, and has no acronyms.

| § | Section | ~words | Content |
|---|---|---|---|
| — | Abstract | 250 | shape below |
| 1 | Introduction | 1,000 | Six paragraphs, no subheadings (lessons item 11): (1) cyclic sorption columns and thermal stores need fast multi-cycle surrogates; (2) consistent solvers and their cost; (3) learned surrogates drift over cycles; (4) structure-preserving and metriplectic models, with known failure modes (Botev 2021; Gruver 2022; Paper 1's small-weight finding: 0.0211 against 0.0231 on held-out materials, nothing on held-out times); (5) "Three gaps follow" with a scoped "we found no study that…" sentence; (6) "The present study therefore asks…", up to five numbered contributions (each with its number, label and section), what the paper does **not** claim, and a roadmap. |
| 2 | Two open dissipative systems and their thermodynamic structure | 900 | Column A (MOF-303, Al-fumarate) and thermocline B. Port-metriplectic form with an empty Poisson part. Propositions stated (non-negative production; upwind mixing as a Bregman divergence; Paper 1 as the exact limit, below 2×10⁻⁹); proofs in Appendix A. |
| 3 | Learned models and the controls that separate constraint, information and projection | 1,000 | The latent model; S, U, U+F, U+proj, U+F+proj, U-long, P, S-L, S−M, S−Π, FNO, C, C+L, MUSCL-C. **Table 1** defines each arm once, as a sentence, with seeds per cell. Equal tuning, budget and wall-clock documented. After this table the prose uses words, not codes. |
| 4 | Pre-registered protocol | 800 | Zenodo DOI of the frozen text; hypotheses H1–H6 in plain words; unit and estimator (Hodges-Lehmann, Wilcoxon plus the pigeonhole bootstrap, Holm); verdict words; the 14-row decision list (full list in Appendix B); simulated power; the pilot and positive control; **Table 3: deviations from the pre-registration** (δ and h_v revision R3, decoder loss R1, ceiling R2, and any later ones). |
| 5 | Results | 2,800 | Opens with one ordering sentence along the thesis axis. 5.1 Gates and admissibility, labelled *checks* (decoder gate, H3a). 5.2 Primary family: long-horizon stability, data efficiency, short-horizon cost (H1, H2, H4); key forest figure. 5.3 Attribution: S against U+F, U+F+proj, S-L and the ablations. 5.4 Against the coarse consistent solver at equal cost. 5.5 Replication (Al-fumarate) and generality (system B), with consistency words. 5.6 Secondary results: out of distribution (H5), drift–error link (H3b), design regret (H6). FNO and MUSCL-C labelled descriptive. |
| 6 | Discussion | 1,000 | Mechanism of any win or loss, measured (lessons item 9); relation to tLaSDI, port-metriplectic networks, PANACHE and Schaffer–Exl; what known synthetic truth buys and cannot show. |
| 7 | Limitations | 350 | Stand-alone section, as in P1 and P3: synthetic references only; two systems; 3 seeds in some cells; decoder balance tolerance; 1-D columns. |
| 8 | Conclusions | 300 | "The main conclusions, demonstrated in the investigated numerical setting, are:", a numbered list, then a scope sentence. |
| — | Acknowledgements, AI statement, competing interests, CRediT, data and code availability | — | per §2 |
| App. A–C | Proofs; full decision list; hyperparameters and cost per arm | outside the main count only if allowed (§7) | |

### 3.3 Figure plan (main text: 7 figures and 3 tables; supplement 10–20 figures)

1. **Fig. 1, schematic: three ways physics enters.** Constraint (metriplectic latent operator), information (U+F inputs) and projection (at rollout), plus fixed or learned potentials. Each arm is placed in this 2×2×2 space, with its legend serving as the arm key.
2. **Fig. 2, the two systems and their reference behaviour.**
   - (a, b) Column and store geometry with ports.
   - (c) Outlet water front and outlet temperature against time over cycles. These are the concrete trajectory panels co-author comment C32 asked for.
   - (d) Thermocline profile under flow reversal.
3. **Fig. 3, map of the study.** Workflow with every Results subsection as a box tagged confirmatory, replication, generality, secondary or descriptive (Paper A `fig_workflow.py` pattern). It can be a panel of Fig. 1 if space is short.
4. **Fig. 4, key figure: forest plot.**
   - Every hypothesis × setting (A-primary, A-replication, B) shown as exp(HL) with its (1−2α_h) interval against the frozen bands (0.8, 1; the ±m equivalence band for H4).
   - The verdict word printed beside each row, and a visible "pending" slot.
   - **Script written on synthetic fixtures before the freeze** (lessons item 13).
5. **Fig. 5, attribution.** Paired log-ratio distributions per trajectory: S against U+F, U+F+proj, S-L, S−M and S−Π. This is the figure that answers the title.
6. **Fig. 6, long rollouts on the decoded fields** (the P5 model, extended to entropy).
   - (a) Error against cycle, 1 to 50, per arm.
   - (b) Water and energy balance residual against cycle, decoded state, log scale.
   - (c) Cumulative entropy production, showing the sign.
   - (d) Drift against error for U (H3b).
7. **Fig. 7, cost against accuracy.** Wall-clock per cycle (training and inference separate) against cycle-50 error, for every learned arm and C at 16/32/64/128 cells plus C+L. This is the figure that answers McGreivy and Hakim.
8. If space allows, design regret per scenario (H6), or this goes to the supplement with the verdict in the text.

- **Table 1:** the arms.
- **Table 2:** decision summary (family | question | outcome and estimate | label).
- **Table 3:** deviations from the pre-registration.
- **Table style:** no vertical rules, no colour, bold only the best value per column.

### 3.4 Abstract shape (about 250 words, 8 sentences, no acronyms or equations)

1. Context: cyclic adsorption columns and thermal stores need surrogates that stay accurate and physically admissible over many cycles.
2. Gap: structure-preserving learned models promise this, but published comparisons mix the constraint, the extra physical information and rollout projection, and rarely test against a consistent numerical solver at equal cost.
3. Design, part 1: we separated these effects in learned latent models of a non-isothermal metal-organic-framework water-adsorption column (two sorbents) and a packed-bed thermocline store, using controls matched in information, projection, training time and potentials.
4. Design, part 2: the hypotheses, thresholds and verdict rules were fixed and deposited before any confirmatory data (cite "pre-registered").
5. Result 1: long-horizon stability and data efficiency, each with its estimate and verdict word.
6. Result 2: attribution, that is, constraint against information against projection.
7. Result 3: against the coarse consistent solver at equal cost, with the generality words for the second sorbent and the second system.
8. Implication with limit: "…in these simulated one-dimensional systems; the result does not by itself extend to …".

Keep it to three results (round-3 lesson F9). The scope disclaimer goes inside the abstract, as in P1, P3 and P4.

---

## 4. Lessons from PINN-machining Paper 1/A and the lessons file, ranked, with actions for 3B

| Rank | Lesson (source) | Action for 3B |
|---|---|---|
| 1 | Live journal rules first, in a compliance table (LESSONS §1a; `paperA/submission/EAAI_COMPLIANCE_2026-11-12.md`) | §2 above. Re-verify rows 9, 15 and 18 live before submission. Settle conflicts A and B now. |
| 2 | Referees attacked claims beyond the evidence, not the method; Paper 1 removed "first", "glass-box", "real-time", "digital-twin core" (`PAPER1_LESSONS_FOR_EAAI.md` §2; response letter item 7) | Extend `docs/terminology_3B.md` with banned words: "first"; "thermodynamically consistent" except on the decoded state where proven; "stable" without its metric; "generalizes" except under H5; "validated"; "physics-informed" used for S. Add a forbidden-word build gate. |
| 3 | Fair comparison at matched information was the most-demanded control (decision letter R2.3, R4.M2, R6.6–6.7; v6 M3) | Lead the abstract and Fig. 1 with the U+F, U+F+proj, U-long and S-L controls. Report parameter counts, tuning budget and wall-clock per arm (Appendix C). Check that no arm a referee could derive is missing; see question 3 in §6. |
| 4 | Do not headline a result true by construction (v6 M1/M2) | Label S's zero balance drift, H3a admissibility and non-negative production as **checks** in §5.1. The informative contrasts are S against U+F, U+F+proj and S-L, and S against C. |
| 5 | Evidence labels instead of adverbs; report adverse results in the abstract (`PAPER1_STYLE_GUIDE.md` §2; Paper A contribution 4, praised by v6 and v7) | Tag every number as (pre-registered), (replication), (exploratory), (descriptive) or (post hoc). If the decision list prints row 8a, 11 or 14, or C wins, that headline goes in the abstract unchanged. |
| 6 | Registered verdicts true to the frozen text; scorer deviations printed beside it (round-3 F1, round-4 F2) | Headline = the decision-list row computed by the frozen script. Any difference is shown in Table 3. A build gate forbids mixing 3-seed and 5-seed cells in one contrast; say "(3 seeds)" wherever it applies (replication, FNO, C+L, U-long). |
| 7 | Tick every "reported beside it" item in the protocol (Paper 3A R1; Paper 2) | A checklist from PREREG §5–8 (H3b, H5, H6, FNO, MUSCL-C, Pratt zero counts, trajectory × seed variance decomposition, rule power for every INCONCLUSIVE, consistency words), each mapped to a location in the manuscript. |
| 8 | Independent timestamp for the protocol (v6 Mo6; Paper 4) | Deposit the frozen PREREG on Zenodo before the confirmatory run (optionally PCI RR). Cite its DOI in §4 and the cover letter. MLST has no registered-report type, so this is the only external guarantee. |
| 9 | Find the mechanism before generalizing a failure (R4.M6; Paper 1's 6.4×10⁴ gradient imbalance) | If S, P or S-L loses, report the measured cause: conditioning of the friction operator, step size, decoder balance error, or whether the projection was active. Otherwise write "untested". Never "structure does not help" in general. |
| 10 | Seeds and statistics (R2.4, R4.M5; round-3 F6; LESSONS §3) | Already designed in. In the text, state rule power for each INCONCLUSIVE, the positive control's result (U at full against quarter iterations), and that REFUTED needs the interval above 0.8. Cite Lenau 2026 as the standard 3B meets. |
| 11 | Six-paragraph introduction with numbered contributions and a "does not claim" list (style guide §1.2–1.3) | §3.2 row 1. Contributions come early (professor rule C3). |
| 12 | Question title; ordered abstract free of acronyms (CHECKLIST §1b.1) | §3.4. |
| 13 | Diagram first and one key figure (CHECKLIST §1b.2; LESSONS kemal 2026-11-09) | Fig. 1 schematic, Fig. 4 forest plot written before the freeze, trajectory panels in Fig. 2. |
| 14 | One figure style module (style guide §3.1; LESSONS §5) | Raise Paper A's 6–6.5 pt floor to 8 pt. Provenance file with SHA-256 hashes and a figure-data CSV. Measure the font size in the PDF. Look at every page. |
| 15 | Caption, table and prose rules (CHECKLIST §1b.3; C15, C23, C42) | Captions of 1–2 sentences, definitions only, no interpretation. "Fig. N shows…" paragraphs ending in a takeaway. No bold run-in labels. |
| 16 | Readability over audit-log style (v7 M4) | No H5d, R1–R3, T7 or "cs-org1 review" codes in the main text. Arm codes appear only in Table 1 and figure legends; prose says "the information-matched model". |
| 17 | Hold the length with a conservation gate (MLST_FORMAT_PLAN §3–4; Paper A went from 27k to 8.3k words) | Write to 8,300 words. A REQUIRED-wording gate protects qualifiers. Proofs go in an appendix, not the supplement. |
| 18 | Build gates on numbers and words, each proven by a planted defect (LESSONS §1b.6) | `\V{key}{value}` binding to the results JSON; unbound digits fail the build; plant one defect per gate. A hostile read of the rendered PDF is still needed. |
| 19 | Review rounds until nothing material remains (directives §7) | Separate briefs: number trace; live references (does each cited work support its sentence?); Methods against code against PREREG; simulated MLST editor plus two referees (one ML, one adsorption or thermal engineering); readability. Queue them for Claude Science per the hub rule, giving the exact commit. |
| 20 | Reproducibility package (R4.m3/m4) | Zenodo DOI reserved early; separate training and inference cost per arm; Kaggle hardware stated; OSI licence. |
| 21 | Synthetic-only evidence is the standing reject risk (v6 F1; round-4 F8.1) | Page 1 states what known truth buys (exact balances, a consistent reference, a design-regret ground truth) and what it cannot show (measured column data). Fixed vocabulary, never "validated". |
| 22 | Cover letter of about one page (Paper A round-4 letter) | Question; three contributions with numbers and labels; "Why simulated references with exact balances"; the registration in two sentences; companion papers (Paper 1 DOI); AI statement; limits conceded. Use MLST's scope words. Re-read it against the final paper. |
| 23 | Response-letter template (`09_response_letter.md`) | At MLST the author writes it; AI only for language (conflict A). |
| 24 | Novelty discipline: 12 "first" claims fell (directives §5) | §5 below: property and measurement claims only, with a positioning table. Present the port-metriplectic formulation as organizing the test, not as the headline. |
| 25 | The plain baseline may win (Paper 4 lesson; arXiv:2606.01366) | Draft the "C makes the learned model unnecessary" version of the abstract now, so a failed primary is a ready paper. |

---

## 5. Positioning and novelty

### 5.1 Closest work

All DOIs below were resolved on Crossref on 2026-11-16 (novelty and recent-papers reports).

- **Architecture: tLaSDI.** Park et al., CMAME 2024, 10.1016/j.cma.2024.117144. Autoencoder plus GFINN latent model, but with *learned* latent energy and entropy, no ports and no matched controls. Its parametric version (He et al., TMLR 2026, arXiv:2506.08475) has the same limits.
- **Within MLST: Schaffer and Exl 2026,** 10.1088/2632-2153/aebff4. Structured against unstructured latent model, 3 seeds, closed system. Its decoded state breaks the constraint (deviations up to about 0.92).
- **Learned GENERIC models.** Hernández et al., JCP 2021, 10.1016/j.jcp.2020.109950; CMAME 2021, 10.1016/j.cma.2021.113763. Port-metriplectic networks, Comput. Mech. 2023, 10.1007/s00466-023-02296-w (open systems, but no physical latent potentials and no sorption).
- **Adsorption columns.** PANACHE, Subraveti et al., I&ECR 2022, 10.1021/acs.iecr.1c04731: physics only as a loss.
- **Same experimental logic, other questions** (all 2026 preprints):
  - arXiv:2609.34012: information against regularization, one pre-registered hypothesis, PINN residuals.
  - arXiv:2609.19674: matched ablation of single components.
  - arXiv:2606.01366: structure against projection; the solver core alone reaches round-off.

### 5.2 Defensible novelty claims

Use the wording of `PAPER1_STYLE_GUIDE`, never "first".

1. "We found no study that tests, under a pre-registered protocol, whether metriplectic structure improves learned latent models of open, advection-dominated dissipative systems with controls that separate the constraint, the physical information and the rollout projection, and that compares physically fixed with learned potentials." Do **not** claim pre-registration in physics ML in general, or the information-against-constraint question in general (arXiv:2609.34012 has both).
2. "We found no learned latent model of a non-isothermal adsorption column whose energy and entropy are fixed from the sorbent's and the gas's thermodynamics, with inlet, outlet and wall treated as ports." Qualify it: projection-based reduced models that keep the physical entropy already exist (Chan 2020; Klein 2025). The claim rests on "learned" and "open column".
3. A port-metriplectic formulation of the non-isothermal column that reduces exactly to Paper 1's model when the adsorbed water's heat capacity is zero (below 2×10⁻⁹), extending the isothermal port models of Baaiu et al. Present it as supporting material, not as the headline.
4. Comparison with a compensated coarse consistent solver at equal cost, plus a decision-level (design-regret) endpoint. Frame it as answering McGreivy and Hakim 2024, not as new methodology.
5. Optional: "to our knowledge, no structured-by-construction surrogate of a packed-bed thermocline store has been reported" (supported only by "not found"; Sene 2025 and Ananth 2026 use plain or soft-loss networks).
6. A statistical contribution, stated explicitly for MLST readers: trajectory-level paired tests with crossed seed–trajectory bootstrap intervals, simulated power and pre-fixed verdict words. No MLST precedent read (P1–P9) reports a pre-specified test or a confidence interval on a model comparison.

### 5.3 Must-cite list

All DOIs ok on Crossref on 2026-11-16, unless arXiv.

**MLST precedents**
- Schaffer and Exl 2026, 10.1088/2632-2153/aebff4
- Nguyen and Sandfeld 2026, 10.1088/2632-2153/aeb34b
- Lenau et al. 2026, 10.1088/2632-2153/ae5e18
- Feng et al. 2026, 10.1088/2632-2153/aeabf8
- Shen et al. 2026, 10.1088/2632-2153/aeab20 (content not read; read it before citing)
- Goyal et al. 2025, 10.1088/2632-2153/adb9b5
- Nam et al. 2026, 10.1088/2632-2153/aeb227

**Structure-preserving and metriplectic learning**
- Hernández 2021 JCP, 10.1016/j.jcp.2020.109950
- Hernández 2021 CMAME, 10.1016/j.cma.2021.113763
- TIGNN, 10.1109/TAI.2022.3179681
- Port-metriplectic NNs, 10.1007/s00466-023-02296-w
- GFINN, 10.1098/rsta.2021.0207
- Lee–Trask–Stinis, arXiv:2106.12619
- Gruber et al. NeurIPS 2023, 10.52202/075280-1670
- tLaSDI, 10.1016/j.cma.2024.117144
- Cueto and Chinesta 2023, 10.1007/s11831-023-09954-5

**Hamiltonian, port-Hamiltonian and Onsager models**
- Greydanus, arXiv:1906.01563
- Cranmer, arXiv:2003.04630
- Desai, 10.1103/PhysRevE.104.034312
- OnsagerNet, 10.1103/PhysRevFluids.6.114402
- Zakwan 2023, 10.1016/j.ifacol.2023.10.079 (confirm it is the intended paper)

**Structure-preserving reduction**
- Öttinger 2015, 10.1103/PhysRevE.91.032147
- Gruber 2023, 10.1016/j.cma.2022.115709
- Chan 2020, 10.1016/j.jcp.2020.109789
- Klein 2025, 10.1016/j.jcp.2025.113817

**Learned closures**
- Kochkov 2021, 10.1073/pnas.2101784118
- van Gastelen 2024, 10.1016/j.jcp.2024.113003
- Bar-Sinai 2019, 10.1073/pnas.1814058116

**Does structure help?**
- Botev 2021, arXiv:2111.05458
- Gruver 2022, arXiv:2202.04836
- McGreivy and Hakim 2024, 10.1038/s42256-024-00897-5
- arXiv:2606.01366
- arXiv:2609.34012
- arXiv:2609.19674
- ProbConserv, 10.1016/j.physd.2023.133952
- Urdeitx 2026, 10.1016/j.jnnfm.2026.105601
- Decision-aware evaluation, arXiv:2606.07146

**Adsorption and thermal-storage surrogates**
- Leperi 2019, 10.1021/acs.iecr.9b02383
- Pai 2020, 10.1021/acs.iecr.0c02339
- PANACHE 2022, 10.1021/acs.iecr.1c04731
- Glover 2027, 10.1016/j.compchemeng.2026.109904
- Frandsen 2025, 10.1002/aic.70045
- Sene 2025, 10.31224/4630
- Ananth 2026, 10.3389/fenef.2026.1845713

**Program and physics**
- Paper 1, archive doi:10.5281/zenodo.23323685 (and its Digital Discovery record if published)
- Yagi, Kunii and Wakao 1960, 10.1002/aic.690060407 (if δ is set from it)
- Baaiu et al. 2008 and 2009, and Walton and LeVan 2005: DOIs not checked in these reports; take them from `docs/lit_review_3B.md` and verify live.

These items are missing from `docs/lit_review_3B.md` today: Hernández JCP 2021, TIGNN, Lee–Trask–Stinis, Gruber 2023, HNN/LNN, Desai, OnsagerNet, VONNs, the learned-closure block, Leperi, Subraveti, Pai, PANACHE, Botev, McGreivy, the thermocline surrogates, the 2026 preprints and all the MLST papers above. **The author compiles the final reference list from originals** (IOP AI policy).

### 5.4 Scoop and undercut risks (sweep to 2026-11-16: no direct scoop found)

- **arXiv:2609.34012** removes "pre-registered" and "information against constraint" as general novelties. Scope 3B's claims to thermodynamic structure, latent models and open systems, and cite it.
- **arXiv:2606.01366** (solver core reaches round-off): if C or C+L dominates, the headline becomes "use the solver". The decision list already prints this ("a coarse consistent solver makes the learned model unnecessary for this system"). Write that framing now.
- **The tLaSDI line** (arXiv:2610.20614, fracture, 2026) is moving fast. A tLaSDI paper on an open system with physical potentials would take claim 2. Re-sweep arXiv monthly until submission.
- **arXiv:2603.02645** (comparison of thermodynamic frameworks under one architecture, constitutive models) is near on "which structure". Cite it and distinguish it as constitutive models, not dynamics surrogates.

---

## 6. The ten hardest referee questions, and how the paper answers each

1. **"Your structured model has exact balances by construction. Isn't the conservation result trivial?"**
   It is, and the paper says so. It is labelled a check (§5.1, H3a). The informative results are H1, H2 and H4 against matched controls, and the attribution in Fig. 5. Say where the balances are exact (latent dynamics with port projection) and report the decoded-field residual separately (Fig. 6b), including the 1 % decoder gate (PREREG 5.1). P1 was weak exactly here.
2. **"Does S win only because it receives physical inputs the baseline lacks?"**
   The U+F arm receives all of S's physical inputs, values and latent gradients. The S-against-U+F contrast and the "information" outcome row answer this directly. Lead with it.
3. **"Is the comparison fair in capacity, tuning and compute?"**
   Give parameter counts, the shared hyperparameter protocol (PREREG §4, fixed before the pilot), U-long (U trained for S's wall-clock) and separate training and inference cost (Fig. 7, Appendix C). If parameter counts are not matched, say so and show U-long as the compute control. P1 listed this as a limitation, and a referee will hold 3B to more.
4. **"Why not simply use a coarse consistent solver?"**
   Arm C at 16–128 cells, compensated, plus C+L, at equal throughput, with the three-way solver statement pre-fixed. Cite McGreivy and Hakim 2024 on weak baselines. If C wins, that is the headline (rows 2–13, qualifier 3).
5. **"Five seeds, some cells three: is that enough?"**
   The inference unit is the trajectory (n = 48 for A-primary). The pigeonhole bootstrap over crossed trajectories × seeds carries the seed variance. Power was simulated for the full decision rule from upper 80 % pilot SD bounds, and every INCONCLUSIVE prints its power. Show the trajectory × seed variance decomposition. Cite Lenau 2026 and state where 3 seeds apply.
6. **"Synthetic references only: why should anyone trust this for real columns?"**
   State it on page 1 and in Limitations. Known truth is what makes the attribution test possible: exact balances, exact regret and a consistent reference. Name the materials' provenance: the isotherms are fitted to published MOF-303 and Al-fumarate data (`docs/materials_3B.md`). Never say "validated".
7. **"Are the reference simulator's physical parameters right?"**
   This is a live risk: δ and h_v (freeze decision 1, revision R3) are pending the primary-source full texts. The paper must give the final values with their sources, and Table 3 must record the change and the regenerated pilot. If the full texts are not obtained, state δ = 0.5 as the fluid-phase lower bound (option a).
8. **"Why would metriplectic structure help an advection-dominated open system, when Botev and Gruver found priors often don't?"**
   Section 1, paragraph 4 states the prior honestly: Paper 1's null on held-out times, Botev and Gruver. The paper is built so that every outcome is informative. Section 6 gives the measured mechanism of whatever happened, not speculation.
9. **"Is the pre-registration real and external?"**
   Zenodo DOI with a timestamp before the confirmatory data, the decision list verified by CI over all 64 combinations, and Table 3 for every deviation. The headline is computed by the frozen script. Be ready to show the commit hash of the freeze.
10. **"What does a practitioner gain: design decisions, not curve error?"**
    H6 design regret over 8 scenarios, with the reference re-evaluating each surrogate's optimized cycle. Cite arXiv:2606.07146 for why curve error can mislead. It is a secondary hypothesis, so report it with its own label and keep it out of the headline.

Add as a reserve question: **"Does the result generalize beyond one material and one system?"** Answer it with the replication and generality words (Al-fumarate; system B) and the demotion rule ("helps in one setting") already in the decision list.

---

## 7. Open items to verify

1. **Author decisions on IOP AI conflicts A and B** (revision workflow; disclosure wording that fits both IOP's "how outputs were checked" and §1b.10; the author compiling references). If no honest wording fits, switch to JCP.
2. **TÜBİTAK–IOP eligibility.** Confirm with the YTÜ library whether the corresponding author must be faculty. Recruit and name the YTÜ faculty co-author (genuine contribution, verified ORCID, CRediT).
3. **Live re-checks in a normal browser** (iopscience blocked automated access 2026-11-16):
   - MLST IF, CiteScore and time to first decision;
   - whether the supplement is peer-reviewed;
   - what counts toward the 8,500 words (appendices? captions?);
   - the ethics statement rule.
4. **Read Shen et al. 2026** (10.1088/2632-2153/aeab20) before citing; its content is unread.
5. **Compare the published MLST versions of P1, P3 and P4 with their arXiv copies** (data statements, final abstracts). Only P5 was read as published.
6. **Zakwan 2023 DOI** (10.1016/j.ifacol.2023.10.079) against the repo's citation (56(2), 5855). Get live DOIs for Baaiu 2008/2009 and Walton–LeVan 2005.
7. **δ and h_v primary sources** (Yagi et al. 1960; Wakao and Kaguei 1982) through the YTÜ library. Then regenerate the pilot and record revision R3 in Table 3.
8. **Decoder gate feasibility** (freeze decision 1b). If no latent dimension passes the 1 % balance criterion, system A is "not testable" and the paper's shape changes. Settle this before writing.
9. **Deposit the frozen PREREG on Zenodo** (DOI) before the confirmatory run; decide on PCI RR.
10. **Write the Fig. 4 forest-plot and Fig. 6 drift scripts on synthetic fixtures before the freeze.** Build the style module to the 8 pt floor.
11. **Confirm the repo LICENSE is OSI-approved,** and reserve the Zenodo DOI for the code and data release.
12. **Re-sweep arXiv monthly until submission** (metriplectic; tLaSDI; port-Hamiltonian; adsorption surrogate; thermocline neural). Re-read arXiv:2609.34012 and 2606.01366 result tables, not only their abstracts.
13. **Update `docs/lit_review_3B.md`** with the missing entries in §5.3 (read-only in this task; for the author or a later session).