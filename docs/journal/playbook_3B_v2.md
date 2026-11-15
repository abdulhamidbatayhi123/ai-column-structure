# Paper 3B journal playbook, version 2

**Paper 3B:** a pre-registered falsification study of structure-preserving latent machine-learning surrogates for adsorption-column and thermal-storage cycles.
**Compiled:** 2026-11-16. This version replaces v1 (`docs/journal/playbook_3B.md`) and answers every point in `docs/journal/playbook_critic.md`. The table of resolved points is at the end.

### Source keys

Every fact below carries one of these keys. Anything not covered by a key is marked **not found** or **blocked**.

| Key | Source | Date |
|---|---|---|
| [V16] | Venue comparison report. Live journal pages plus the official ULAKBİM EKUAL lists (cabim.ulakbim.gov.tr/ekual/ekual-veritabanlari/), downloaded that day | 2026-11-16 |
| [AI16] | AI-policy report. Live IOP, Springer Nature, Elsevier and TMLR pages, read in full | 2026-11-16 |
| [L16] | Literature check on the live Crossref REST API and arXiv API. Conference venues marked "web search" were not opened on the publisher's site | 2026-11-16 |
| [J16], [R16], [N16], [LS16] | The v1 journal, recent-papers, novelty and lessons reports (`docs/journal/report_*.md`) | 2026-11-16 |
| [FP] | `MLST_FORMAT_PLAN.md`. **Not re-checked live** | 2026-11-01 |
| [Hub] | `T4_venue_3B.md`. **Not re-checked live** | 2026-11-08 |
| [Repo] | Files read in this task: `README.md` (title; line 24 "No confirmatory data exist yet"), `PREREG_P3B_draft.md` (design table, lines 56–74; estimator, lines 383–422; AI declaration, line 654), `LICENSE`, `docs/lit_review_3B.md`, `~/.claude/AUTHOR_PROFILE.md` | 2026-11-16 |
| [Crit] | Repo facts the critic checked: README and PREREG numbers not re-read here | 2026-11-16 |

### Blocked on 2026-11-16

- iopscience.iop.org: MLST metrics and article pages (Radware bot check) [J16, V16]
- pnas.org and academic.oup.com (Cloudflare) [V16]
- sciencedirect.com: Elsevier guide-for-authors and review-time pages [V16]
- doaj.org [V16]
- PCI RR friendly-journal list (Anubis bot check, not bypassed) [V16]

### Names used in this version (these replace the ambiguous "P1")

| Name | Meaning |
|---|---|
| **M1–M10** | The ten recent MLST papers. Mapping in §3.1 |
| **MOF Paper 1** | The program's first paper: archive doi:10.5281/zenodo.23323685, retargeted to Digital Discovery. It is **not** an MLST paper |
| **Machining Paper 1 / Paper A** | The PINN-machining papers whose review rounds supply the lessons in §5 |
| **3B** | This study |

---

## 0. Decisions for the author

Only the items below need the author. Everything else in this playbook can be carried out without a decision.

### D1. Venue

**Recommendation:** Machine Learning: Science and Technology (MLST) as the base target, and Journal of Computational Physics (JCP) as the fallback.

- **Top-tier first attempt, only under a condition.** Make one first attempt at *Nature Machine Intelligence* (Analysis) or *Nature Computational Science* (Analysis or Article) **only if** the frozen decision list gives the same decisive verdict in A-primary, the Al-fumarate replication and system B, with intervals clear of the bands (criteria in §1.3).
- **Why the attempt is cheap.** Both Nature journals have a subscription route with no fee. A desk rejection costs about 10–13 days (median first decision: NMI 10 days, NCS 13 days) [V16].
- **Mixed, INCONCLUSIVE or single-setting results** go straight to MLST.
- **Choose JCP instead of MLST if any of these holds:**
  - the author cannot accept the IOP AI rules in D4;
  - no YTÜ faculty corresponding author joins (D3).

### D2. Registered Report route (new in v2)

**Recommendation: do not pursue it. Deposit the frozen PREREG on Zenodo before any confirmatory run, and use neither PCI RR nor a journal Registered Report.**

- **The option is still open.** README line 24 says "No confirmatory data exist yet" [Repo].
- **Only one journal in the comparison offers the format:** Nature Communications, with in-principle acceptance after Stage 1 [V16].
- **Against it:**
  - The Nature Communications APC (£5,490 / $7,350 / €6,150) is not covered by TÜBİTAK. Waivers are "case-by-case… in cases of financial need" and must be requested at submission [V16].
  - The format is for "hypothesis-driven quantitative research with primary research data". Whether simulation counts is unknown [V16].
  - Stage 1 review takes months.
  - The pilot is already running [Repo, README].
  - MLST's PCI RR-friendly status could not be checked (blocked) [V16].
  - Royal Society Open Science commits to publishing "regardless of the outcome", but it is not on the TÜBİTAK lists and needs a waiver [V16].
- **If the author still wants this route,** ask the Nature Communications editors whether simulation-only designs qualify, before the freeze.

### D3. Corresponding author and co-author for the fee agreement

**Facts:**
- MLST is gold open access only. Its APC is £2,680 / €3,215 / $3,350 [J16].
- MLST is on the TÜBİTAK–IOP funded list ("List D … Gold OA") [V16, J16].
- YTÜ is row 228 of the EKUAL 2026 member list [V16].
- The IOP notice covers "authors at EKUAL member institutions" [V16].
- The v1 sources add two conditions: the corresponding author must carry the EKUAL affiliation [J16], and one library (İYTE, 10 Mar 2026) requires active academic staff [J16].

**Recommendation:**
1. Recruit a YTÜ faculty corresponding author **only if that person makes a genuine contribution** (CRediT, verified ORCID).
2. The one YTÜ faculty member in `AUTHOR_PROFILE.md` with an existing collaboration is Assoc. Prof. Hasan Cankurt (YTÜ Food Engineering; ORCID 0009-0006-3269-5398; corresponding co-author on MOF Paper 1) [Repo]. A YTÜ faculty member in chemical, mechanical or energy engineering would fit the content better.
3. Confirm the eligibility rule with the YTÜ library before submission. In particular, ask whether a master's-level first author with a faculty corresponding author qualifies.
4. If nobody joins, submit to JCP alone through the free subscription route [V16]. The Nature attempt also needs no fee agreement.

### D4. AI-policy items

The AI policies were read live on 2026-11-16 [AI16]. Each item needs a yes or no from the author.

**a. Revision workflow (binding at IOP; largely also at Springer Nature).**
- IOP forbids two things: uploading reviewer reports to AI tools, and generating responses to reviewers with AI. AI may be used "only to improve the language of their responses" [AI16].
- Springer Nature forbids sharing "manuscripts, peer review reports, and sensitive data… with unsecured or public AI systems" [AI16]. This also affects the NMI/NCS attempt.
- No author-side rule on this was found for Elsevier or TMLR [AI16].
- **Recommendation:** accept it. The author reads the reports and writes the replies; AI is used only for language. If the author will not accept this, choose JCP.

**b. Reference list (IOP).**
- IOP prohibits AI that would "generate or modify reference lists". AI may support discovery, management, formatting and checking, "but every cited source must be verified by the authors against the original source" [AI16].
- The author's set wording deliberately avoids claiming "verified all references" unless that happened (memory note `ai-declaration-wording`).
- **Recommendation:** the author compiles the final list and checks each source against the original. The disclosure can then truthfully say so. If the author will not do this, choose JCP: Elsevier only requires AI use for references to be declared [AI16].

**c. Additions to the disclosure.** The set text stays whole; only facts are added. The current text is at PREREG line 654 [Repo]. As written it is not compliant at IOP [AI16] for three reasons: it does not say how the outputs were checked, it gives no tool version, and it does not describe the AI-assisted code in Methods.

**Recommendation:** approve these three additions.
1. A version after "Claude", taken from the project's records. For MOF Paper 1 the memory note records "Claude Opus 5 / 5.5 via Claude Code"; confirm which models were used for 3B.
2. One sentence on the checks: "All code was checked by automated tests and by regenerating every reported number from the archived results files, and every cited source was checked by the author against the original." Use the second clause only if D4b is accepted.
3. One Methods sentence saying the analysis code was written with AI assistance and how it was tested. This Methods sentence is **mandatory** at IOP and at Elsevier. Springer Nature asks for "validation steps" [AI16].

Placement and template by venue:
- **IOP:** a separate statement in the Acknowledgements. Template: "…The authors reviewed, validated and critically assessed all AI-assisted outputs…" [AI16]. This sentence is real and current; it is one half of IOP's two-sentence template, which settles critic point (a)2. The proposed wording meets the same requirement by naming the checks as procedures.
- **Elsevier:** the heading "Declaration of generative AI and AI-assisted technologies in the manuscript preparation process" [AI16].
- **Springer Nature:** no fixed section and no template were found [AI16].
- **TMLR:** no disclosure is required [AI16].

If no honest wording is accepted, the 2026-11-15 rule applies: change the journal, never the disclosure.

**d. If a co-author joins.**
- Change "the author" to "the authors".
- Give "designed the study, made every scientific decision" either to the whole group, if true, or to the first author by name [AI16].
- **Recommendation:** name the first author.

---

## 1. Venue analysis

### 1.1 Funding facts that shape the ranking

All from the official EKUAL lists downloaded 2026-11-16 [V16]:

- **Funded:** MLST (IOP List D, Gold OA) and AIChE Journal (both Wiley lists).
- **Not funded:**
  - NCS, NMI and Nature Communications (absent from the 923-title Springer Nature list);
  - all Elsevier titles (the agreement covers access only);
  - PNAS Nexus (absent from the 396-title OUP list);
  - Science Advances (no AAAS agreement).
- Hybrid journals (JCP, CMAME, C&CE, CES, AIChE J., NMI, NCS) can be published in through the subscription route at no cost [V16].

### 1.2 Venue table

All facts from [V16] unless another key is given.

| Venue | Fit, in the journal's own words | Fee-free route | Time to decision | Metrics | Verdict |
|---|---|---|---|---|---|
| **MLST (IOP)** | "Simulation of materials, chemical processes and biological systems", "New (physics inspired) learning algorithms", "Benchmark studies" | TÜBİTAK, needs D3 | One data point: M5 received 1 May 2026, accepted 23 Sep 2026 [R16]. Other figures blocked | IF 4.2, 5-year 5.5, CiteScore 6.7 [Hub, not re-verified] | **Base target** |
| **JCP (Elsevier)** | "the frontier between predictive simulation and machine learning… a comparison should be provided… efficacy, robustness, computational complexity, as well as reproducibility should be addressed" | Subscription. OA APC would be $4,240 | 7 days to first decision, 211 to acceptance [Hub, blocked on 2026-11-16] | IF 3.9, 5-year 4.7 | **Fallback.** Best if D3 or D4 fails |
| **NMI (Springer Nature)** | ML/AI and "the significant impact… on other scientific disciplines". Analysis = "new data obtained in a comparative analysis that leads to novel and arresting conclusions" | Subscription | Median 10 days to first decision, 231 to acceptance | JIF 29.8, 5-year 36.0 | **Conditional first attempt** (§1.3). McGreivy and Hakim, the weak-baselines paper, appeared here |
| **NCS (Springer Nature)** | "methodologies that use computing capabilities in novel ways to find new insights". Article or Analysis: 3,500 words excluding Methods, 6 display items, 150-word abstract | Subscription | Median 13 days to first decision, 193 to acceptance | JIF 20.3, 5-year 21.9 | Conditional first attempt. Papers of "insufficient general interest" are "rejected promptly without external review" |
| **Computers & Chemical Engineering** | Publishes "comparisons of alternative methodologies" | Subscription. OA APC would be $3,710 | not found | IF 4.9, 5-year 4.5 | Strong fit with a high chance of acceptance; mid-level prestige |
| **CMAME** | "physically-based machine-learning methods"; "significant developments of computational methods". tLaSDI appeared here [L16] | Subscription. OA APC would be $4,780 | not found | IF 7.6, 5-year 8.3 | Scope is method development. MOF Paper 1 was desk-rejected here on 2026-10-17 (project memory) |
| **TMLR** | Studies that "shed light on the strengths and weaknesses of the methods"; novelty "is not a necessary criterion". Parametric tLaSDI appeared here [L16] | "no fees" | Desk rejection within 1 week; reviews due in 2 weeks | none shown | Best fit for a null result. Double-blind review clashes with the named Zenodo protocol and public repository |
| **AIChE Journal** | "highest standards in rigor, novelty, and innovation" | TÜBİTAK-funded, or subscription | Median 10 days to first decision; acceptance rate 26% | JIF 4.4, CiteScore 7.9 | Overlaps with Paper 4, which is going there |
| **Chemical Engineering Science** | "foundations the Science of Chemical Engineering" | Subscription | not found | IF 5.1 | Weak fit |
| **Nature Communications** | "important advances of significance to specialists"; offers Registered Reports | None. APC £5,490; waiver at the editors' discretion | Median 9 days to first decision, 242 to acceptance | JIF 18.1 | Only via the Registered Report plus a waiver (D2: not recommended) |
| **Science Advances** | any area of science | None. APC $5,450 | not found | not found | Excluded |
| **PNAS / PNAS Nexus** | blocked | Apparently none: $6,075 / ~$3,025 from university pages, unverified | blocked | blocked | Excluded |

Still excluded, on the v1 evidence [J16]:
- Digital Discovery: overlaps with MOF Paper 1.
- npj Computational Materials: the TÜBİTAK–Springer Nature quota is exhausted (Altınbaş library, 9 Jul 2026).
- Machine Learning: Engineering: not on the TÜBİTAK list.
- Data-Centric Engineering: 2024 IF 2.8.

**Expected-value order.** This is a judgement, not a measured probability [V16]:

> MLST > JCP > NMI/NCS (conditional) > C&CE > CMAME > TMLR > AIChE J. > CES > Nature Communications

### 1.3 When to aim at NMI or NCS first

Criteria from [V16]. Aim at NMI or NCS first only if the verdict is decisive and the same in all three settings, with intervals clear of the bands, and it is one of these:

- **A clear positive.** The constraint gives a large long-horizon gain over the information-matched and projection-matched controls, which isolates "constraint" as the mechanism.
- **A clear negative.** The compensated coarse solver at equal cost beats every learned arm in both systems, and the cause is measured.

Practical cost of the attempt:
- NCS needs a separate manuscript version: 3,500 words excluding Methods, 6 display items, 150-word abstract [V16].
- Budget about two weeks for that version.
- The Springer Nature reviewer-report confidentiality rule applies (D4a).
- Whether these journals accept preprints was **not checked** in v2; check before posting.

### 1.4 Why a Research paper at MLST

- **Not a Benchmark.** The Benchmark type characterises "the performance of differing methods / models / codes /algorithms / software". 3B instead tests falsifiable hypotheses with a decision rule [J16].
- **Length.** Research papers are "normally not more than 8500 words" [J16].
- **Not a Letter.** Letters (≤5,000 words, urgency justified) do not fit [J16].
- **No registered-report article type at MLST** [J16]. The protocol's independent timestamp is therefore the Zenodo DOI (D2).

### 1.5 Evidence for the scope fit, worded as the reports support it

- **MLST has published on this question.** Two recent MLST papers address whether physics-based structure helps:
  - M1, Schaffer and Exl, compares a structured J–R latent model with an unstructured latent neural ODE.
  - M4, Lenau et al., is about seed variation of physically informed models [R16].
  - That is the only evidence. No editorial statement on negative results, benchmarking or reproducibility was found [J16, V16]. The paper must therefore argue the value of an informative null itself.
- **The policies fit the design.** IOP's open-data policy is mandatory, and review can be single-anonymous [J16]. 3B's public repository, CI and frozen protocol fit these policies. The reports do not say that review *rewards* this.

---

## 2. MLST compliance table

"not found" means none of the 2026-11-16 reports covers the item. Every such row is in the §8 live-check list.

| # | Requirement | Live rule | Source | What 3B does |
|---|---|---|---|---|
| 1 | Length | Research paper "normally not more than 8500 words". **What counts (appendices? captions?) is not stated** | About page [J16]. Counting: [FP], not re-checked | Resolve this first (§8 item 1). Plan to about 7,500 words of main text so that captions fit if they count. Word-count gate in the build |
| 2 | Abstract | "Not normally be more than 300 words"; self-contained; no references, equations, figure or table numbers, or undefined acronyms | writing-journal-article page, read live [J16] | About 230 words. Gate in §4.6 |
| 3 | Structure | Introduction, Methods, Results, Discussion, Conclusion; Acknowledgements (conflicts of interest, funding, CRediT); References | same page [J16] | §4 outline. Method sections may carry descriptive names, as M1 and M3 do [R16] |
| 4 | Keywords | M5's published front matter has keywords [Crit, R16]. The rule itself was not found | — | Prepare 5–6 keywords without acronyms; check the live rule |
| 5 | Figures | Vector EPS/PDF preferred; text 8–12 pt at final size; widths 8.5 or 15 cm; never colour as the only cue; captions self-contained; no colour in tables. Limits on figure count: not found | same page [J16] | One style module; 8 pt floor asserted in `save()` and measured in the PDF; Okabe–Ito palette plus marker per arm |
| 6 | Graphical abstract | not found | — | Check the live page |
| 7 | References | Vancouver or Harvard, consistent; DOIs or arXiv IDs; cite the Version of Record | same page [J16] | Cite published versions where they exist (§6.3). **The author compiles the final list** (D4b) |
| 8 | Data | Data needed to verify the results public at publication; a data availability statement is a condition of publication; "upon request" not accepted. Template wordings: not found | open-data policy page [J16]. "Cannot be amended after publication": [FP], not re-checked | Versioned Zenodo DOI for the reference data, model outputs and results JSON, reserved early |
| 9 | Code | Trusted repository with a DOI; a GitHub copy must also go to a DOI repository; OSI principles | data and About pages [J16] | GitHub plus a Zenodo release. `LICENSE` is MIT [Repo]. MIT is generally listed as OSI-approved; confirm on opensource.org at deposit (not opened) |
| 10 | Access during review | not found | — | Recommendation: public repository at submission (single-anonymous review, so nothing needs anonymising) |
| 11 | Supplementary files | ≤50 MB per file, 150 MB in total; title ≤30 characters, description ≤30 words | same page [J16] | Large data stays on Zenodo |
| 12 | Supplement in review | "not included in peer review by default" | [FP], not re-checked | Proofs a referee must check go in an **appendix**. Re-check live |
| 13 | Submission format | Single PDF; iopjournal class optional; ORCID recommended | same page [J16] | iopjournal class; ORCID 0009-0007-0374-6755, plus the co-author's if D3 |
| 14 | Review model | Single- or double-anonymous by choice; usually two reviewers; criteria "quality, scientific rigour, novelty, significance to the field, and presentation"; rejection "typically above 50%"; sound papers of limited interest are usually rejected; IOP pre-review checks for scope and quality | About and review-process pages [J16, V16] | Single-anonymous. Argue significance in the abstract, the introduction and the cover letter |
| 15 | Generative AI | Separate statement in the Acknowledgements (tool, version, use, how outputs were verified); research and code use also described "in detail in the Methods"; prompts and logs may be requested; routine language editing exempt. Prohibited: fabricated data, AI-generated or AI-modified reference lists, AI-generated reviewer responses, uploading reports to AI, hidden prompts (immediate rejection). A non-existent citation is "strong evidence of irresponsible AI usage" | generative-ai-tools page (dateModified 2026-10-15); Ethical Policy for Journals (modified 2026-09-04) [AI16] | D4 a–d. The Methods sentence is mandatory. Any AI-made schematic must be "clearly distinguishable from research data" |
| 16 | Preprints | Encouraged | About page [J16] | arXiv at submission. Compatibility of the arXiv licence with CC BY was not found; choosing CC BY 4.0 on arXiv avoids the question |
| 17 | Licence and copyright | CC BY under the agreement; licence form: **not found** | — | Check live |
| 18 | Fee eligibility at submission | How the corresponding author claims TÜBİTAK eligibility (institutional address, institution selection): **not found** | — | Ask the YTÜ library (D3) |
| 19 | Similarity check and text recycling | not found | — | Write Methods in new words, not copied from the Zenodo PREREG, `docs/lit_review_3B.md` or the preprint. Cite the PREREG DOI for the full protocol. State the deposit in the cover letter |
| 20 | Related manuscripts | IOP concurrent and redundant publication policy: not found | — | Name MOF Paper 1 (Digital Discovery) as a related, non-overlapping manuscript in the cover letter |
| 21 | Ethics statement | not found | — | Probably "not applicable" (computational study); check live |
| 22 | Competing interests and funding | In the Acknowledgements | same page [J16]. Funding wording template: not found | "The authors declare no competing interests." "This work received no specific funding." (memory: no grant) |
| 23 | Suggested and excluded reviewers, cover-letter fields, submission system | not found | — | Check in the submission system |
| 24 | Negative-results policy | none found | [J16, V16] | Argue the value of an informative null in the paper itself, citing McGreivy and Hakim 2024 |
| 25 | Metrics and timing | IF 4.2 / 5-year 5.5 / CiteScore 6.7; one timing point (M5) | [Hub] not re-verified; [R16] | Re-check in a normal browser |

---

## 3. Recent MLST papers M1–M10 and what they show

### 3.1 Mapping (renamed from v1's P1–P10)

All from [R16]; bibliographic records confirmed in [L16].

| M | Paper | Read from |
|---|---|---|
| M1 | Schaffer and Exl, latent energy-based conservative–dissipative operator for magnetisation, 2026, 10.1088/2632-2153/aebff4 | arXiv 2609.04530v2 |
| M2 | Goyal, Yıldız, Benner, MLST 6 015063 (2025), 10.1088/2632-2153/adb9b5 | arXiv 2308.13835v1 |
| M3 | Nguyen and Sandfeld, MLST 7 055055 (2026), 10.1088/2632-2153/aeb34b | arXiv 2601.08404 |
| M4 | Lenau, Dimiduk, Niezgoda, MLST 7 035005 (2026), 10.1088/2632-2153/ae5e18 | arXiv 2510.03416 |
| M5 | Feng et al., MLST 7 055032 (2026), 10.1088/2632-2153/aeabf8 | **Published PDF**, eprints.gla.ac.uk/396312 |
| M6 | Nam et al., MLST 7 055046 (2026), 10.1088/2632-2153/aeb227 | arXiv 2604.16664 |
| M7 | Xie, Tang, Tang, MLST 5 045070 (2024), 10.1088/2632-2153/ad9ce7 | arXiv 2409.05893v2 |
| M8 | Bishnoi et al., MLST 5 035049 (2024), 10.1088/2632-2153/ad6be6 | arXiv 2307.05299v1 |
| M9 | Kumar and Mukherjee, MLST 5 025063 (2024), 10.1088/2632-2153/ad51cd | arXiv 2310.05169v2 |
| M10 | Shen, Song, Zhang, MLST 7 055035 (2026), 10.1088/2632-2153/aeab20 | **Not read** (bibliographic data only) |

**Copy warning.** The preface of `report_papers.md` calls the Glasgow published PDF "P8". Its own list, and this playbook, identify that paper as M5 (Feng et al.). Do not copy "P8".

### 3.2 The pattern

These are observations about the papers, for planning only. None of this wording goes into the manuscript or the cover letter.

- **Length and figures.** 20–37 IOP single-column pages, 7–31 figures, few tables, 34–73 references [R16].
- **Order.** Problem and physics, then method (often with propositions: M1, M2, M6, M9), then set-up, then results, then a stand-alone Limitations section or Discussion, then Conclusion.
  - M1 has a "Relation to structured dynamical models" section.
  - M9 has an "Informal summary of our results" near the start [R16].
- **Abstract.** One paragraph of 5–10 sentences. M1, M3 and M4 place a scope statement inside the abstract [R16].
- **Evidence figures.** Rollout error against time, field snapshots against the reference, conserved quantities over time (M5 plots mass drift at about 1e-16), per-trajectory violin plots (M2) [R16].
- **Statistical reporting varies.**
  - Several papers report single runs (M2, M6, M7, M8).
  - M1 reports 3 seeds with mean ± SD.
  - M4 reports 30 runs with a bootstrap.
  - None of the nine papers read reports a pre-specified test or an interval on a model comparison [R16].
  - 3B describes its own statistical design positively and does not compare it with these papers (critic point (d)3).
- **Wording.** M1, M3, M4 and M9 hedge systematically ("under the stated assumptions", an explicit not-claimed list) [R16]. 3B adopts that style and uses no priority words.
- **Back matter.** Data availability, conflict of interest, and sometimes author contributions and code availability (M3, M4, M5) [R16].

---

## 4. Lean blueprint for 3B

### 4.1 Title, now scoped

Do not use the README title: it ends in "…of open dissipative systems" [Repo], which claims more than two simulated one-dimensional systems support.

**Recommended title:**
> "Constraint, information or projection? A pre-registered test of thermodynamic structure in learned latent models of cyclic adsorption and thermal-storage columns"

**Shorter alternative:**
> "What does thermodynamic structure buy learned latent models of cyclic adsorption columns? A pre-registered test"

Both keep the question form and contain no acronyms. "Columns" covers both systems (an adsorption column and a packed-bed thermocline store). Update the README title when the paper is drafted.

### 4.2 What stays in the main text

The main text carries the **primary family** (H1 long-horizon stability, H2 data efficiency, H4 short-horizon cost) in A-primary. It also carries the pre-registered decision-list qualifiers that change the headline:
- **Attribution:** S against U+F, U+F+proj and S-L.
- **The coarse-solver statement:** C and C+L at equal cost. It is kept in the main text because the frozen decision list prints it as a headline qualifier [Crit: rows 2–13, qualifier 3].

Replication (Al-fumarate) and generality (system B) appear in the main text only as rows of the forest plot, each with its consistency word. Their detail goes to an appendix.

**Appendices, each reported in the main text by a one-line verdict in Table 2:**

| Appendix | Content |
|---|---|
| C | H3a admissibility and the decoder gate (checks) |
| D | Replication and system-B detail |
| E | H5 out of distribution |
| F | H3b drift–error link |
| G | H6 design regret |
| H | Ablations S−M and S−Π, U+proj, U-long |
| I | Size sweep 16/64/128 |
| J | FNO and MUSCL-C (descriptive) |

### 4.3 Outline (target about 7,500 main-text words; see compliance row 1)

| § | Section | ~words | Content |
|---|---|---|---|
| — | Abstract | 230 | §4.5 |
| 1 | Introduction | 900 | Six paragraphs, no subheadings: (1) cyclic sorption columns and thermal stores need fast multi-cycle surrogates; (2) consistent solvers and their cost; (3) learned surrogates drift over cycles; (4) structure-preserving and metriplectic models, with what earlier studies found when the prior did not help (Botev 2021; Gruver 2022; MOF Paper 1's small-weight finding of 0.0211 against 0.0231 on held-out materials and nothing on held-out times [Crit: README]); (5) "Three gaps follow", with a scoped "to our knowledge, no study…" sentence; (6) "The present study therefore asks…", up to four numbered contributions, a does-not-claim list, and a roadmap |
| 2 | Two simulated open dissipative columns and their thermodynamic structure | 800 | Column A (MOF-303, Al-fumarate) and thermocline B. Port-metriplectic form with an empty Poisson part. Propositions only (non-negative production; upwind mixing as a Bregman divergence; MOF Paper 1 as the exact limit, below 2×10⁻⁹ [Crit]). Proofs in Appendix A |
| 3 | Learned models and the controls that separate constraint, information and projection | 800 | Latent model. **Table 1** defines each main-text arm in one sentence, with seeds per cell (from the PREREG design table [Repo]). Appendix-only arms are listed in one line. Equal tuning, budget and wall-clock. Prose uses words, not codes, after Table 1 |
| 4 | Pre-registered protocol | 700 | Zenodo DOI; H1, H2, H4 in plain words, other hypotheses named; unit and estimator (Hodges–Lehmann, one-sided Wilcoxon, crossed pigeonhole bootstrap after Owen 2007, Holm [Repo: PREREG lines 383–422]); verdict words; the decision list (in full in Appendix B); simulated power; the pilot and positive control; **Table 3: deviations** |
| 5 | Results | 2,300 | **5.1** Checks: one paragraph with pointers (decoder gate, H3a). **5.2** Primary family (Fig. 3, forest plot). **5.3** Attribution (Fig. 4). **5.4** Against the coarse consistent solver at equal cost (Fig. 5b). **5.5** Long rollouts on the decoded fields (Fig. 5a). Replication and system B as forest-plot rows plus a paragraph |
| 6 | Discussion | 900 | Measured mechanism of any win or loss; relation to tLaSDI, port-metriplectic networks, PANACHE and M1; what known synthetic truth buys and cannot show |
| 7 | Limitations | 300 | Simulated references only; two systems; 3-seed cells (list in §5 lesson 6); decoder balance tolerance; one-dimensional columns |
| 8 | Conclusions | 250 | Numbered list prefaced "in the investigated numerical setting", then a scope sentence |
| — | Back matter | — | Acknowledgements with the AI statement (D4), competing interests, funding, CRediT, data and code availability |
| App. A–J | Proofs, full decision list, appendices C–J above, hyperparameters and cost per arm | — | Count status per §8 item 1 |

### 4.4 Display items (main text: 5 figures, 3 tables)

1. **Fig. 1: three ways physics enters.** Constraint, information and projection, plus fixed against learned potentials. Each main-text arm is placed in this space, so the legend serves as the arm key. Panel (b) maps the study: each Results subsection and appendix tagged confirmatory, replication, generality, secondary or descriptive.
2. **Fig. 2: the two systems and their reference behaviour.**
   - (a, b) Geometry with ports.
   - (c) Outlet water front and outlet temperature over cycles: the trajectory panels co-author comment C32 asked for.
   - (d) Thermocline profile under flow reversal.
3. **Fig. 3: key figure, forest plot.**
   - Rows: H1, H2, H4 and the attribution contrasts, each in A-primary, A-replication and B.
   - Each row shows exp(HL) with its (1−2α_h) interval against the frozen bands, with the verdict word beside it.
   - Seed counts are printed per row.
   - Script written on synthetic fixtures before the freeze.
4. **Fig. 4: attribution.** Paired per-trajectory log-ratio distributions: S against U+F, U+F+proj and S-L.
5. **Fig. 5: long horizon and cost.**
   - (a) Error and decoded balance residual against cycle (1–50), log scale.
   - (b) Wall-clock per cycle (training and inference separate) against cycle-50 error, for the learned arms and C at 16/32/64/128 cells plus C+L.

**Tables:**
- **Table 1:** the arms.
- **Table 2:** decision summary for every hypothesis, including the appendix verdicts in one line each (question | outcome and estimate | label | location).
- **Table 3:** deviations from the pre-registration.
- Style: no vertical rules, no colour.

Entropy-production sign, drift against error, design regret and the FNO and MUSCL-C panels go to the appendices.

### 4.5 Abstract shape (about 230 words, 8 sentences)

1. **Need:** surrogates of cyclic adsorption columns and thermal stores that stay accurate and physically admissible over many cycles.
2. **Gap:** comparisons of structure-preserving learned models seldom separate the constraint from the extra physical information and the rollout projection, or test against a consistent solver at equal cost.
3. **Design:** learned latent models of a simulated non-isothermal metal-organic-framework water-adsorption column (two sorbents) and a packed-bed thermocline store, with controls matched in information, projection and training time.
4. **Protocol:** hypotheses, thresholds and verdict rules were fixed and deposited before any confirmatory data.
5. **Result 1:** long-horizon stability and data efficiency, each with its estimate and verdict word.
6. **Result 2:** attribution.
7. **Result 3:** against the coarse solver at equal cost, with the generality words.
8. **Scope:** "…in these simulated one-dimensional systems; it does not by itself extend to…".

Three results at most (round-3 lesson F9) [LS16].

### 4.6 Abstract gate (fixes critic point (d)8)

The gate rejects:
- undefined acronyms;
- equations;
- figure and table numbers;
- any digit that is neither bound to a results key nor on an **allow-list**.

Allow-list: count words and their digits fixed by the design, for example "two sorbents", "two systems", "50 cycles", "one-dimensional". The allow-list is versioned with the gate. Each part of the gate is proven by one planted defect.

### 4.7 Outcome-branch abstracts (fixes critic point (d)7)

- Do not draft any outcome-specific abstract before the Zenodo freeze.
- The PREREG's "Expected outcomes, and why each is publishable" already fixes each branch's meaning [Repo].
- After the freeze, but before unblinding the confirmatory results, draft all branches symmetrically outside the repository:
  - structure helps by constraint;
  - the gain is information, not constraint;
  - INCONCLUSIVE;
  - the coarse solver makes the learned model unnecessary.
- Record their SHA-256 hash in the run log. Then use the branch the frozen script selects.

---

## 5. Lessons from the machining papers and the lessons file, with actions for 3B

All rows from [LS16]. v1 ranks 11–25 are merged where they duplicated §2 or §4.

| # | Lesson (source) | Action for 3B |
|---|---|---|
| 1 | Live journal rules first, in a compliance table (LESSONS §1a; Paper A EAAI compliance file) | §2. Close the §8 live-check items before writing |
| 2 | Referees attacked claims beyond the evidence (Machining Paper 1 removed "first", "glass-box", "real-time", "digital-twin core") | Banned-word gate in `docs/terminology_3B.md`: "first"; "thermodynamically consistent" except on the decoded state where proven; "stable" without its metric; "generalizes" except under H5; "validated"; "physics-informed" used for S. **The scoped title (§4.1) applies the same lesson** |
| 3 | Fair comparison at matched information was the most-demanded control | U+F, U+F+proj and S-L lead Fig. 1 and the abstract. Parameter counts, tuning budget and wall-clock per arm in an appendix |
| 4 | Do not headline a result that is true by construction | S's zero latent balance drift, H3a and non-negative production are **checks** (§5.1) |
| 5 | Evidence labels instead of adverbs; adverse results in the abstract | Tag every number (pre-registered, replication, exploratory, descriptive, post hoc). If the coarse solver wins, that goes in the abstract |
| 6 | Registered verdicts true to the frozen text; seed counts stated | Headline = the decision-list row computed by the frozen script. **Corrected 3-seed list, from the PREREG design table [Repo]:** U-long (A-primary full); U, P and S at A-primary sizes 16/64/128; **every** A-replication cell (U, U+F, U+F+proj, P, S, full and 32); S-L on system B (full and 32); FNO; C+L. The gate therefore **labels** mixed contrasts (for example system-B attribution, S 5 seeds against S-L 3 seeds) as "(5 vs 3 seeds)" instead of forbidding them. The estimator follows the frozen script |
| 7 | Tick every "reported beside it" item in the protocol | Checklist from PREREG §5–8 mapped to manuscript locations, including the appendices |
| 8 | Independent timestamp for the protocol | Zenodo DOI before the confirmatory run (D2) |
| 9 | Find the mechanism before generalising a failure | If S, P or S-L loses, report the measured cause (friction-operator conditioning, step size, decoder balance error, projection activity), or write "untested" |
| 10 | Seeds and statistics | Rule power beside each INCONCLUSIVE; the positive control's result; the trajectory × seed variance decomposition (PREREG line 400 [Repo]). Cite M4 as related work on seed variation, without comparison |
| 11 | Six-paragraph introduction, early numbered contributions, a does-not-claim list | §4.3 |
| 12 | Question title, ordered abstract free of acronyms | §4.1, §4.5 |
| 13 | Diagram first, one key figure | Fig. 1, Fig. 3 |
| 14 | One figure style module | 8 pt floor (Paper A used 6–6.5 pt); provenance with SHA-256 hashes; figure-data CSV; look at every page |
| 15 | Captions, tables and prose | Captions of 1–2 sentences with definitions only; no bold run-in labels |
| 16 | Readability over audit-log style (v7 M4) | No H5d, R1–R3 or reviewer-org codes in the main text; arm codes only in Table 1 and legends |
| 17 | Density failure (round-3 F9) | §4.2: about 8 main-text arms, three hypotheses in the main text, the rest one line each |
| 18 | Build gates on numbers and words, each proven by a planted defect | `\V{key}{value}` binding, the §4.6 gate, banned words, and a REQUIRED-wording gate for qualifiers |
| 19 | Review rounds until nothing material remains | Briefs: number trace; live references; Methods against code against PREREG; a simulated MLST editor plus two referees (one ML, one adsorption/thermal engineering); readability. Queue for Claude Science with the exact commit |
| 20 | Synthetic-only evidence is the standing reject risk | Page 1 states what known truth buys and what it cannot show; never "validated" |
| 21 | One-page cover letter | Question; three contributions with labels; why simulated references; the registration in two sentences; related manuscripts (row 20); AI statement; limits conceded. No comparison with prior papers' statistics |
| 22 | Response letter | At IOP and Springer Nature the author writes it, and AI is used only for language (D4a) |
| 23 | Novelty discipline (12 "first" claims fell) | §6.2 |
| 24 | The plain baseline may win (Paper 4; arXiv:2606.01366) | Handled by §4.7, without prejudging the result |

---

## 6. Positioning and novelty

### 6.1 Closest work

Records from [N16, L16]. Each item is described by what it does, not ranked against 3B.

- **tLaSDI.** Park, Cheung, Choi, Shin, CMAME 429:117144 (2024), 10.1016/j.cma.2024.117144. Autoencoder plus GFINN latent model, with learned latent energy and entropy.
  - **Parametric tLaSDI:** He, Shin, Gruber, Jung, Lee, Choi, arXiv:2506.08475, TMLR 2026 [L16].
  - 3B differs in fixing the potentials from physics, in treating ports, and in matched controls.
- **M1** (Schaffer and Exl 2026). Structured against unstructured latent model, 3 seeds, closed system. It reports that the decoded state departs from the unit-norm constraint (about 0.10 and 0.92) [R16]. It does not report decoded conservation or entropy checks, and 3B adds these.
- **Learned GENERIC and port-metriplectic models:**
  - Hernández et al., JCP 426:109950 (2021) and CMAME 379:113763 (2021);
  - port-metriplectic networks, Comput. Mech. 72(3):553–561 (2023);
  - Muixí, González, Chinesta, Cueto, Comput. Mech. 75(1):357–368, 10.1007/s00466-024-02508-x: port-metriplectic coupling of open subsystems, close to 3B's port formulation [L16].
- **Isothermal port-based adsorption models**, the direct ancestors of 3B's formulation:
  - Baaiu et al., MCMDS 2008 and 2009;
  - Baaiu et al., J. Process Control 2009;
  - Peng et al., IEEE CDC 2007 [L16].
- **Adsorption-column ML:**
  - PANACHE (I&ECR 61(11):4095–4113, 2022), physics as a loss;
  - Praditia et al. FINN (WRR 2022), finite-volume network for sorption transport, the closest analogue to system A;
  - Ceccanti et al. DeepONet for cyclic TVSA (arXiv:2601.09491) [L16].
- **Same experimental logic, other questions:**
  - arXiv:2609.34012: information against regularisation, one pre-registered hypothesis;
  - arXiv:2609.19674: matched single-component ablation;
  - arXiv:2606.01366: structure against projection; the solver core alone reaches round-off [N16].

**Novelty sweep.** No structure-preserving latent surrogate of an adsorption column or packed-bed store from 2025–2026 was found [L16]. The claims below are therefore worded "to our knowledge".

### 6.2 Defensible novelty claims

No priority words are used. v1 claim 6 is dropped.

1. "To our knowledge, no study has tested, under a pre-registered protocol, whether metriplectic structure improves learned latent models of open, advection-dominated dissipative systems, with controls that separate the constraint, the physical information and the rollout projection, and with physically fixed against learned potentials." Do not claim pre-registration in physics ML in general, or the information-against-constraint question in general (arXiv:2609.34012).
2. "To our knowledge, no learned latent model of a non-isothermal adsorption column has had its energy and entropy fixed from the sorbent's and gas's thermodynamics, with inlet, outlet and wall treated as ports." Qualify it: projection-based reduced models that keep the physical entropy exist (Chan 2020; Klein 2025), and so do structure-preserving reductions of adsorption models (Baaiu et al., J. Process Control 2009). The claim rests on "learned" and "non-isothermal open column".
3. A port-metriplectic formulation of the non-isothermal column that reduces exactly to MOF Paper 1's model when the adsorbed water's heat capacity is zero (below 2×10⁻⁹ [Crit]). It extends the isothermal port models of Baaiu et al. and supports the test; it is not the headline.
4. Comparison with a compensated coarse consistent solver at equal cost, framed as following McGreivy and Hakim 2024.
5. Optional: no structured-by-construction surrogate of a packed-bed thermocline store was found. The evidence is "not found" only [N16, L16].

**Dropped from the manuscript:** v1 claim 6 compared 3B's statistics with the MLST papers. The statistical design is described in §4 of the paper on its own terms. The cover letter may say, in neutral words, that the study reports trajectory-level paired tests with intervals and simulated power.

### 6.3 Must-cite list, version 2

Records from [L16] unless another key is given. DOIs are listed only where a report gave them. Copy every remaining DOI from the Crossref record when the reference list is compiled (D4b).

**MLST precedents**
- M1, Schaffer and Exl 2026, 10.1088/2632-2153/aebff4. Published 2026-11-09; no volume or article number on Crossref yet, so re-check at compilation.
- M3, Nguyen and Sandfeld, 7 055055.
- M4, Lenau et al., 7 035005.
- M5, Feng et al., 7 055032.
- **M10, Shen et al., 7 055035: still unread. Read it before citing.**
- M2, Goyal et al., 6 015063.
- M6, Nam et al., 7 055046.

**Foundations** (new in v2)
- Grmela and Öttinger, PRE 56:6620 (1997), 10.1103/physreve.56.6620.
- Öttinger and Grmela, PRE 56:6633 (1997), 10.1103/physreve.56.6633.
- Morrison, Physica D 18:410 (1986), 10.1016/0167-2789(86)90209-5.
- Kaufman, Phys. Lett. A 100:419 (1984), 10.1016/0375-9601(84)90634-0.
- van der Schaft and Jeltsema, Found. Trends Syst. Control 1:173 (2014), 10.1561/2600000002.

**Structure-preserving and metriplectic learning**
- Hernández et al., JCP 2021, 10.1016/j.jcp.2020.109950.
- Hernández et al., CMAME 2021, 10.1016/j.cma.2021.113763.
- **TIGNN, 10.1109/TAI.2022.3179681: year 2024** (IEEE Trans. Artif. Intell. 5(3):967–976), not 2022.
- Port-metriplectic networks, 10.1007/s00466-023-02296-w.
- GFINN, 10.1098/rsta.2021.0207.
- **Lee–Trask–Stinis: cite NeurIPS 34 (2021)** (web search); arXiv:2106.12619.
- Gruber, Lee, Trask, NeurIPS 36 (2023), 10.52202/075280-1670.
- **Gruber et al., "Efficiently Parameterized Neural Metriplectic Systems": cite ICLR 2025** (web search); arXiv:2405.16305.
- tLaSDI, 10.1016/j.cma.2024.117144.
- **Parametric tLaSDI**, arXiv:2506.08475, TMLR 2026.
- Cueto and Chinesta 2023, 10.1007/s11831-023-09954-5.
- Muixí et al., 10.1007/s00466-024-02508-x.
- Bermejo et al., Adv. Model. Simul. Eng. Sci. 12:27 (2025).
- Chung, Bu, Verma, "Physics-conforming Latent Twins", CMAME 463:119351 (January 2027 issue); arXiv:2606.15053.
- **Neural-operator variants:**
  - GENERIC-FNO, arXiv:2606.08343: **now v9 (2026-11-01). Re-read before describing it.**
  - Poisson-GENERIC, arXiv:2610.05570: **now v3 (2026-11-07). Re-read before describing it.**
- **tLaSDI fracture**, arXiv:2610.20614: **v1 submitted 12 Jul 2026**, not October.

**Hamiltonian, port-Hamiltonian, Onsager and dissipative**
- Greydanus, arXiv:1906.01563.
- Cranmer, arXiv:2003.04630.
- Desai et al., 10.1103/PhysRevE.104.034312.
- OnsagerNet, 10.1103/PhysRevFluids.6.114402.
- Zakwan et al. 2023, 10.1016/j.ifacol.2023.10.079. **Confirmed: matches the repo citation, 56(2):5855–5860.**
- Sosanya and Greydanus, arXiv:2201.10085.
- Dissipative SymODEN, arXiv:2002.08860.
- Cite all arXiv-only items by arXiv record unless the venue is checked.

**Latent, reduced-order and operator models** (new in v2)
- Neural ODE, arXiv:1806.07366.
- Lee and Carlberg, JCP 404:108973 (2020), 10.1016/j.jcp.2019.108973.
- Fresca, Dede', Manzoni, J. Sci. Comput. 87 (2021), 10.1007/s10915-021-01462-7.
- LaSDI, CMAME 399:115436 (2022), 10.1016/j.cma.2022.115436.
- gLaSDI, JCP 489:112267 (2023), 10.1016/j.jcp.2023.112267.
- Operator inference, CMAME 306:196 (2016), 10.1016/j.cma.2016.03.025.
- SINDy, PNAS 113:3932 (2016), 10.1073/pnas.1517384113.
- **FNO, arXiv:2010.08895 (needed: it is an arm).**
- DeepONet, Nat. Mach. Intell. 3:218 (2021), 10.1038/s42256-021-00302-5.
- Kolmogorov n-width: Ohlberger and Rave, arXiv:1511.02021, Proc. ALGORITMY 2016, 1–12 (no DOI).

**Structure-preserving reduction**
- Öttinger 2015, 10.1103/PhysRevE.91.032147.
- Gruber et al. 2023, 10.1016/j.cma.2022.115709.
- Chan 2020, 10.1016/j.jcp.2020.109789.
- Klein et al. 2025, 10.1016/j.jcp.2025.113817.

**Learned closures**
- Kochkov 2021, 10.1073/pnas.2101784118.
- van Gastelen 2024, 10.1016/j.jcp.2024.113003.
- Bar-Sinai 2019, 10.1073/pnas.1814058116.
- List, Chen, Thuerey, J. Fluid Mech. 949:A25 (2022).
- Melchers et al., Comput. Math. Appl. 143:94 (2023).
- Sanderse et al., Found. Data Sci. 7(1):298 (2025).
- SymCLaw, arXiv:2601.21080.
- NESCFN, arXiv:2507.01795.

**Does structure help?**
- Botev 2021, arXiv:2111.05458.
- Gruver 2022, arXiv:2202.04836.
- McGreivy and Hakim, Nat. Mach. Intell. 6:1256 (2024), 10.1038/s42256-024-00897-5.
- arXiv:2606.01366, arXiv:2609.34012, arXiv:2609.19674.
- ProbConserv, 10.1016/j.physd.2023.133952.
- Urdeitx 2026, 10.1016/j.jnnfm.2026.105601.
- Decision-aware evaluation, arXiv:2606.07146.
- **Zhong, Dey, Chakraborty, arXiv:2012.02334:** benchmark of ten energy-conserving models on 4 systems. The lit review's "10 to 10,000 trajectories" is not in the abstract; check the full text.
- Janssen, Branca, Buck, A&A 708:A227 (2026).
- Jones and Fuhg, arXiv:2603.02645: distinguish it as constitutive models.
- Eidnes and Myhr, arXiv:2611.05031.
- Glas et al., arXiv:2608.30788.

**Adsorption and thermal-storage surrogates**
- Leperi 2019, 10.1021/acs.iecr.9b02383.
- Subraveti et al. 2019, I&ECR 58(44):20412, 10.1021/acs.iecr.9b04173.
- Pai 2020, 10.1021/acs.iecr.0c02339.
- PANACHE 2022, 10.1021/acs.iecr.1c04731.
- Praditia et al., WRR 58(12):e2022WR033149 (2022).
- Wu et al., Green Chem. Eng. 6(2):233 (2025).
- Galanti et al., Processes 13(9):2824 (2025).
- **Bilal, Haq, Asif** (exactly three authors), I&ECR 65(26):13942 (2026).
- An et al., Front. Environ. Eng. 5:1881554 (2026).
- **Ceccanti et al., arXiv:2601.09491:** the "isothermal" description is unconfirmed; check the full text.
- Li et al., Chem. Eng. Sci. 320:122467 (2026).
- Glover 2027, 10.1016/j.compchemeng.2026.109904 [N16].
- Frandsen 2025, 10.1002/aic.70045 [N16].
- Sene 2025, 10.31224/4630 [N16].
- Ananth 2026, 10.3389/fenef.2026.1845713.

**Statistics, reproducibility and pre-registration** (new in v2)
- **Owen 2007, Ann. Appl. Stat. 1:386–411, 10.1214/07-aoas122.** PREREG line 392 already names the correct journal [Repo].
- **Holm 1979: not found on Crossref.** Cite it from the JSTOR record after checking it there.
- Hodges and Lehmann, Ann. Math. Stat. 34:598 (1963), 10.1214/aoms/1177704172.
- Henderson et al., AAAI 32 (2018), 10.1609/aaai.v32i1.11694.
- Bouthillier et al., arXiv:2103.03098.
- Agarwal et al. (rliable), arXiv:2108.13264.
- NeurIPS pre-registration workshops, PMLR vol. 148 (2020) and vol. 181 (2021) (web search).

**Domain and program**
- MOF Paper 1, doi:10.5281/zenodo.23323685 (and its Digital Discovery record if published).
- Hanikel et al., ACS Cent. Sci. 5:1699 (2019), 10.1021/acscentsci.9b00745.
- Yagi, Kunii, Wakao 1960, 10.1002/aic.690060407 (if δ is set from it).
- **Baaiu et al. 2008,** MCMDS 14(3):195, 10.1080/13873950701844840.
- **Baaiu et al. 2009,** MCMDS 15(3):233, 10.1080/13873950902808578 (different author list).
- **Baaiu et al. 2009,** J. Process Control 19(3):394, 10.1016/j.jprocont.2008.07.002.
- **Peng, Baaiu, Couenne, Le Gorrec, IEEE CDC 2007, 3513–3518, 10.1109/cdc.2007.4434663 (added).**
- **Walton and LeVan 2005,** Adsorption 11:555, 10.1007/s10450-005-5984-6. Copy the full title from the record page; only a truncated title was seen.
- Schumann 1929, J. Franklin Inst. 208:405, 10.1016/s0016-0032(29)91186-8.
- Esence et al., Sol. Energy 153:628 (2017), 10.1016/j.solener.2017.03.032.
- Ruthven 1984, Wiley, xxiv+433 pp., no DOI. Confirmed via the AIChE J. book review, 10.1002/aic.690310335.

**Optional, if space allows:**
- CIPHER, arXiv:2605.16682;
- Hernández et al., arXiv:2508.12569;
- Baheri and Lindemann, arXiv:2509.19526;
- Tierz et al., arXiv:2607.20535;
- Putkaradze, Neural Networks 206:109691 (2027);
- Moradi et al., Automatica 187:112892 (2026);
- GINNGER, arXiv:2610.23352;
- Sigdel, arXiv:2607.10439.

**`docs/lit_review_3B.md` needs, in a later session** (read-only here):
- Correct "Bilal, Haq, Asif and co-authors" (line 207) to three authors [Repo, L16].
- Add the entries above that are missing.
- Re-read GENERIC-FNO and Poisson-GENERIC at their current versions.
- Note on the critic's point (a)10: the lit review already cites Baaiu 2009 (J. Process Control) and Peng 2007 with DOIs [Crit]. v2 now has every Baaiu, Peng and Walton–LeVan DOI checked live [L16].

### 6.4 Scoop and undercut risks

No direct scoop was found in sweeps up to 2026-11-16 [N16, L16].

- **arXiv:2609.34012.** Scope 3B's claims to thermodynamic structure, latent models and open systems, and cite it.
- **arXiv:2606.01366.** If C or C+L dominates, the headline becomes "use the solver". The decision list already prints this; §4.7 handles the drafting.
- **The tLaSDI line** (parametric, TMLR 2026; fracture, arXiv:2610.20614) is moving fast. Re-sweep arXiv monthly.
- **arXiv:2603.02645.** Distinguish it as constitutive models.

---

## 7. The ten hardest referee questions

1. **"Exact balances by construction: isn't conservation trivial?"**
   Yes, and the paper says so: it is a check (§5.1, Appendix C). The informative results are H1, H2 and H4 against matched controls, and the attribution (Fig. 4). The decoded-field residual is reported separately (Fig. 5a), with the 1 % decoder gate (PREREG 5.1). The decoded-state check is presented as 3B's own addition, without comment on earlier papers.
2. **"Does S win only through extra inputs?"**
   U+F receives all of S's physical inputs. The S-against-U+F row answers this, and it leads the paper.
3. **"Fair in capacity, tuning and compute?"**
   Parameter counts, the shared hyperparameter protocol, U-long (Appendix H) and separate training and inference cost (Fig. 5b, appendix). If parameter counts are not matched, say so.
4. **"Why not a coarse solver?"**
   C at 16–128 cells, compensated, plus C+L, at equal throughput, under the pre-fixed three-way statement (§5.4). Cite McGreivy and Hakim 2024. If C wins, it is the headline.
5. **"Five seeds, some cells three: enough?"**
   The unit is the trajectory (n = 48, A-primary [Crit]). The crossed pigeonhole bootstrap carries the seed variance. Power is simulated from upper-80 % pilot SD bounds. The variance decomposition is shown. Every 3-seed cell is labelled (corrected list, §5 lesson 6), including S-L on system B.
6. **"Simulated references only?"**
   Stated on page 1 and in Limitations. Known truth enables the attribution test. Isotherms are fitted to published MOF-303 and Al-fumarate data (`docs/materials_3B.md`; Hanikel 2019). Never "validated".
7. **"Are δ and h_v right?"**
   They are pending the primary-source full texts (freeze decision 1, revision R3). Give the final values with sources and record them in Table 3. Otherwise use δ = 0.5 as the fluid-phase lower bound, option (a) [Crit].
8. **"Why would metriplectic structure help an advection-dominated system, given Botev and Gruver?"**
   §1, paragraph 4 states the prior evidence, including MOF Paper 1's null on held-out times. Cite the n-width argument (Ohlberger and Rave). Every outcome is pre-declared publishable, and §6 gives the measured mechanism.
9. **"Is the pre-registration real?"**
   Zenodo DOI dated before the confirmatory data. The decision list is verified by CI over all 64 combinations [Crit]. Table 3 lists the deviations. Have the freeze commit hash ready.
10. **"What does a practitioner gain?"**
    H6 design regret over 8 scenarios [Crit]. Its one-line verdict is in Table 2, with detail in Appendix G and arXiv:2606.07146 as the citation. It stays out of the headline.

**Reserve question: "Beyond one material and one system?"** Answer with the forest-plot rows for replication and system B, their consistency words, and the demotion rule ("helps in one setting").

---

## 8. Open items, in order

1. **What counts toward MLST's 8,500 words** (appendices, captions). The §4 plan depends on it. Check in a normal browser or ask the editorial office.
2. **Author decisions D1–D4** (§0).
3. **TÜBİTAK–IOP eligibility** with the YTÜ library (D3).
4. **Live re-checks** in a normal browser (iopscience is blocked to automated access):
   - MLST IF, CiteScore and time to first decision;
   - whether the supplement is peer-reviewed;
   - ethics statement;
   - keywords;
   - graphical abstract;
   - figure limits;
   - licence form;
   - data-statement templates;
   - redundant-publication policy;
   - suggested reviewers.
   For a top-tier attempt, also check the NMI/NCS preprint policy and the JCP review times.
5. **Read M10 (Shen et al.)** before citing.
6. **Compare the published MLST versions of M1, M3 and M4** with their arXiv copies.
7. **Full texts to check:**
   - Ceccanti (the isothermal claim);
   - Zhong (the trajectory counts);
   - GENERIC-FNO v9 and Poisson-GENERIC v3;
   - the full Walton–LeVan title;
   - Holm 1979 on JSTOR;
   - M1's volume and article number.
8. **δ and h_v primary sources** (Yagi et al. 1960; Wakao and Kaguei 1982). Then regenerate the pilot and record R3.
9. **Decoder gate feasibility** (freeze decision 1b). If no latent dimension passes, system A is not testable; settle this before writing.
10. **Deposit the frozen PREREG on Zenodo** before the confirmatory run (D2).
11. **Write the Fig. 3 forest-plot and Fig. 5 scripts on synthetic fixtures before the freeze;** build the 8 pt style module and the §4.6 gate with its allow-list.
12. **Reserve the Zenodo code and data DOI.** LICENSE is MIT [Repo]; confirm its OSI listing on opensource.org at deposit.
13. **Re-sweep arXiv monthly** (metriplectic; tLaSDI; port-Hamiltonian; adsorption surrogate; thermocline neural). Re-read the result tables of arXiv:2609.34012 and 2606.01366.
14. **Update `docs/lit_review_3B.md`** (§6.3 note).

---

## 9. Critic points resolved

| Critic point | How v2 resolves it |
|---|---|
| (a)1 M1 "no checks on the decoded state" | Reworded (§6.1, §7 Q1): M1 reports decoded unit-norm deviations (about 0.10 and 0.92) but no decoded conservation or entropy checks |
| (a)2 Conflict B; reports disagree on the IOP template | IOP page re-read live [AI16]. Both quotes are halves of one two-sentence template. The current wording is non-compliant at IOP in three respects; D4c gives the minimal additions |
| (a)3 "P1" ambiguous | Renamed M1–M10 (§3.1), plus MOF Paper 1 and Machining Paper 1/A. v1's "Cite P1 by its MLST DOI" removed; MOF Paper 1 is cited by its Zenodo/DD record |
| (a)4 "Policies reward 3B's design" | Now "fit the design" (§1.5) |
| (a)5 "Editors take the question" | Now "MLST has published two papers on the question"; no editorial signal found (§1.5) |
| (a)6 Seed-count list incomplete | Full list from the PREREG design table (§5 lesson 6, §7 Q5); the mixing gate becomes a labelling gate |
| (a)7 Abstract row with two sources | Live source only (§2 row 2) |
| (a)8 Repo facts cited to reports | Cited to [Repo] or [Crit] (README/PREREG) throughout |
| (a)9 Licence item can close | LICENSE read: MIT [Repo]; reduced to confirming the OSI listing at deposit (§2 row 9, §8 item 12) |
| (a)10 Baaiu/Peng DOIs already in the lit review; Peng missing | All Baaiu, Peng and Walton–LeVan DOIs checked live [L16]; Peng 2007 added to must-cite (§6.3) |
| (a)11 P8/P5 mix-up | Copy warning in §3.1 |
| (b) Keywords | §2 row 4 (rule not found; check live) |
| (b) Licence and copyright | §2 row 17 |
| (b) Claiming TÜBİTAK eligibility; master's student | §2 row 18; D3 asks the YTÜ library about a master's first author with a faculty corresponding author |
| (b) Text recycling and similarity check | §2 row 19 (write Methods in new words; cite the PREREG DOI) |
| (b) Related manuscripts | §2 row 20; cover letter (§5 lesson 21) |
| (b) Code and data access during review | §2 row 10 (public repository; single-anonymous review) |
| (b) Data-statement templates; funding wording | §2 rows 8 and 22 (templates not found; check live) |
| (b) AI use in research must go in Methods | Confirmed mandatory at IOP and Elsevier [AI16]; D4c and §2 row 15 |
| (b) Do appendices count toward 8,500 words | §8 item 1, first in order; plan sized at about 7,500 words |
| (b) Suggested reviewers, cover-letter fields, submission system | §2 row 23 |
| (b) Figure limits, graphical abstract, arXiv licence | §2 rows 5, 6 and 16 |
| (c) Report-found items missing from must-cite | All added (§6.3) after the live check [L16]; M10 still flagged unread |
| (c) Items in no report | All checked live and added (§6.3: foundations, latent/ROM/operator, n-width, dissipative networks, statistics, pre-registration workshops, domain). Holm 1979 not found on Crossref (use JSTOR); Ruthven has no DOI |
| (c) Corrections found in the check | TIGNN year 2024; Owen in Ann. Appl. Stat.; Bilal three authors; Gruber ICLR 2025; Lee–Trask–Stinis NeurIPS 2021; arXiv:2610.20614 dated 12 Jul 2026; GENERIC-FNO v9 and Poisson-GENERIC v3; parametric tLaSDI added; Zakwan confirmed |
| (d)1 Density | §4.2–4.4: primary family plus attribution and the solver qualifier in the main text; H3a, H3b, H5, H6, ablations, size sweep, FNO and MUSCL-C in appendices with one-line verdicts in Table 2; 5 figures, 3 tables |
| (d)2 Title overgeneralises | Scoped title (§4.1) |
| (d)3 Disparaging precedents | Removed: "weak spot", "M1 was weak here", "exceed every precedent", "weaker papers". Claim 6 dropped (§6.2); §3.2 marked planning-only |
| (d)4 PCI RR now | D2 recommends the Zenodo DOI only; the Nature Communications Registered Report is assessed and not recommended |
| (d)5 Missing alternatives | TMLR, CMAME, C&CE, CES, AIChE J., NMI, NCS, Nature Communications, Science Advances and PNAS/PNAS Nexus checked live [V16] (§1.2) |
| (d)6 "Procedures, not supervision" may read as non-compliant | Decided after the live read: the wording names the checks, which is what IOP asks for, and keeps the author's set text. The reference clause is used only if the author does the check (D4b, D4c) |
| (d)7 Fallback abstract prejudges the result | §4.7: no branch drafted before the freeze; afterwards all branches drafted symmetrically outside the repository and hashed |
| (d)8 Digit gate blocks count words | §4.6 allow-list |