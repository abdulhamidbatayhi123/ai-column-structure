# MLST venue check for Paper 3B (live sources, read 2026-11-16)

**Bottom line:** keep Machine Learning: Science and Technology (MLST) as the primary target. The fee is covered: on the official TÜBİTAK list MLST is a funded "Gold OA" title, and YTÜ is an EKUAL member. Its scope, its open-data policy and its choice of review model all suit a pre-registered falsification study. There is no registered-report route.

**What I could not check today:**
- iopscience.iop.org redirects to a bot-check page (Radware), and I did not try to get past it. So MLST's impact factor, CiteScore and time to first decision were **not re-verified today**. The hub file read IF 4.2, 5-year IF 5.5 and CiteScore 6.7 on 2026-11-08.
- sciencedirect.com, rsc.org and nature.com refused automated fetching (403 or login redirect).
- I read nothing in any repo; I worked from the task brief and the hub venue file `T4_venue_3B.md`. Nothing was edited.

## 1. MLST
Sources: publishingsupport.iopscience.iop.org/?p=3555 and .../journals/machine-learning-science-and-technology/about-machine-learning-science-technology/

**Scope.** It explicitly lists:
- "Simulation of materials, chemical processes and biological systems"
- "Simulation methods and high-performance computing"
- "New (physics inspired) learning algorithms"
- "Benchmark studies"
- "Codes and datasets"

**Article types:**
- Research papers: "normally not more than 8500 words".
- Letters: at most 5,000 words, and the authors must justify why the work is urgent.
- Also: Dataset articles, Perspectives, Topical reviews, Benchmarks, Challenges, Technical notes, Roadmaps, Comments and replies.
- **No registered report or pre-registration type.**
- Benchmarks "characterize the performance of differing methods / models / codes /algorithms / software" and must give every technical detail needed to reproduce them. I still recommend a Research paper, because 3B tests hypotheses.

**Abstract (IOP general guidelines).** "Not normally be more than 300 words". It must be self-contained, with no references, equations, figure or table numbers, or undefined acronyms. Source: publishingsupport.iopscience.iop.org/authoring-for-journals/writing-journal-article/ (no date shown; the Word template is from Jan 2026).

**Other IOP formatting rules (same page):**
- Structure: Introduction, Methods, Results, Discussion, Conclusion, then Acknowledgements (conflicts of interest, funding, CRediT contributions), then References.
- References: Vancouver or Harvard, used consistently, with DOIs or arXiv IDs, citing the Version of Record.
- Figures: vector EPS or PDF preferred. Text 8–12 pt at final size. Widths 8.5 cm or 15 cm. Colour must not be the only cue, and colours must stay distinguishable in greyscale. Captions must be self-contained. No colour in tables.
- Supplementary files: up to 50 MB each and 150 MB in total. Each needs a title (at most 30 characters) and a description (at most 30 words).
- Submission: a single PDF. The iopjournal LaTeX class is optional, and its `[anonymous]` option removes author details.
- ORCID is recommended for all authors.

**Data and code.** MLST is under the IOP **open data policy**: publishingsupport.iopscience.iop.org/iop-publishing-open-data-policy (no date shown).
- The data needed to verify the results must be public at publication.
- A data availability statement is a condition of publication and must give DOIs or links. "Upon request" does not meet the requirement.
- Code should go to a trusted repository with a DOI.
- If code is on GitHub, also deposit a copy in a repository that issues a DOI (data-availability policy page). Zenodo is fine.
- The journal page also asks for code to follow Open Source Initiative principles and to be submitted as supplementary material or linked from a repository.

**Open access and fees.**
- Gold OA. APC £2,680 / €3,215 / $3,350.
- The TÜBİTAK–IOP Read & Publish agreement covers it: 1 Jan 2026 to 31 Dec 2028, unlimited articles, hybrid and gold.
  - ULAKBİM announcement: cabim.ulakbim.gov.tr/wp-content/uploads/sites/4/2026/03/IOP-DUYURU_EKUAL-3.docx
  - Funded-journals list (79 titles): .../2026/03/Institute-of-Physics-IOP-Fonlanan.xlsx. I downloaded it today and MLST is there ("List D … MLST … Gold OA").
  - EKUAL member list: .../2026/02/EKUAL_Uye_Listesi_2026-1.xlsx. "Yıldız Teknik Üniversitesi" is on it.
- **Condition:** eligibility is checked through the **corresponding author's** institution, which must be an EKUAL member. The official ULAKBİM text names no staff requirement. One library (İYTE, 10 Mar 2026) adds that the corresponding author must be "an active member of our Institute's academic staff".
- So the corresponding author should carry the YTÜ affiliation. A YTÜ faculty co-author as corresponding author is the safe choice; confirm with the YTÜ library.

**Review model.** Single-anonymous or double-anonymous, the author chooses. Publishing the reports is opt-in.
- A public Zenodo pre-registration and code with the author's name make double-anonymous hard to keep, so choose single-anonymous.
- Usually two reviewers. Reviewers judge "quality, scientific rigour, novelty, significance to the field, and presentation".
- IOP rejection rates are "typically above 50%", and papers that are technically sound but of limited interest are usually rejected. Significance has to be argued, not only rigour.
- Source: .../authoring-for-journals/review-process-journals/

**Metrics.** Not re-verified today, and no time to first decision was found. A third-party aggregator (slogix.in) gives 2024 IF 4.6, CiteScore 7.7, Q1 in Artificial Intelligence; that is not authoritative.

**Editors** (.../editorial-board/, read today):
- Editor-in-Chief: Kyle Cranmer (UW–Madison).
- Executive board: Frank Noé, Maurizio Pierini, Lei Wang.
- Closest to 3B: Naoya Takeishi (combining scientific and data-driven models) and Johannes Brandstetter (AI4Simulation).
- No editor lists thermodynamics or chemical engineering.

## 2. IOP policy on generative AI
Source: publishingsupport.iopscience.iop.org/questions/generative-ai-tools/ (no version or date shown on the page).
- **Disclosure:** material AI use goes in a separate disclosure statement **inside the Acknowledgements**. It names the tool and version, says how it was used, and says how the outputs were checked. The page gives a template: "During the preparation of this manuscript, the authors used [Tool Name, Version] to support [describe use]."
- AI used in the research itself must also be described in the Methods.
- Prompts and logs are encouraged as supplementary material.
- Routine spelling and grammar editing needs no statement.
- **Prohibited:**
  - fabricating or altering data;
  - generating or modifying reference lists;
  - generating responses to reviewers;
  - uploading reviewer reports to AI tools;
  - hidden prompts (immediate rejection).
- AI may help find literature and manage or check references, but every source must be verified against the original. A reference to a non-existent source is treated as "strong evidence of irresponsible AI usage".
- AI-made explanatory images must be clearly distinguishable from research data.
- **For our workflow:** the author must compile the final reference list himself. Our current AI declaration wording is compatible, but it has to sit in the Acknowledgements.

## 3. What the editors want
- **No MLST editorial or editor statement on negative results, benchmarking or reproducibility was found** in two searches.
- The only signals in place are the policies above: open data is mandatory, Benchmarks is an article type, and "Benchmark studies" and "Codes and datasets" are in the scope.
- A search for 2024–2025 MLST papers on GENERIC, metriplectic or port-Hamiltonian models found none (the key work is in Computational Mechanics, 2023). That suggests a gap MLST could fill, but the search was shallow, so it is only a hint.

## 4. Alternatives
Facts are as found today unless dated otherwise.

| Venue | Fit | Prestige | Fee for us | Review time | Note |
|---|---|---|---|---|---|
| **Journal of Computational Physics** | Strong if the formulation and solver carry the paper | IF 3.9, CiteScore 7.9 (hub, 2026-11-08; site blocked today) | Hybrid, so the subscription route is free | 7 days to first decision, 211 to acceptance (hub, 2026-11-08) | Best safe option |
| **Digital Discovery** (RSC) | Moderate: chemistry-data focus | 2025 IF 7.1, 46 days to first peer-review decision (RSC newsletter, Oct 2026, via search; rsc.org blocked) | Gold OA, APC £2,200 (Apr 2026 newsletter). TÜBİTAK–RSC quota 255 for 2026, gold included (Boğaziçi page, undated); DD not confirmed on the list | ~46 days to first decision | Paper 1 is going there; a second paper may look like overlap |
| **Data-Centric Engineering** (CUP) | Good on engineering and physics-enhanced ML | 2024 IF 2.8 (CUP announcement, via search) | APC £1,250, "waived without question" when the author has no funding or institutional agreement; TÜBİTAK–CUP agreement 2026–28 exists | not found | Low prestige |
| **npj Comput. Mater. / Nature Comput. Sci.** | NCS is high prestige (2024 IF 18.3 per JCR as cited on Wikipedia) | high | npj is gold OA; the TÜBİTAK–Springer Nature 2026 quota "has been fully utilized" (Altınbaş library, 9 Jul 2026), so the author pays the APC. NCS: the subscription route is presumably free, but its pages were blocked | not found | npj excluded by fee; NCS is a long shot, needing a striking, general mechanistic result |
| **Physical Review Research** | Moderate | not checked | No TÜBİTAK–APS agreement found | not found | Not recommended |

**Machine Learning: Engineering** (IOP sister journal) is new: no metrics, APC £2,500, and **not** on the TÜBİTAK funded list. Excluded.

**Recommendation:** keep MLST as primary (format: Research paper, ≤8,500 words; abstract ≤300 words; single-anonymous; YTÜ-affiliated corresponding author). JCP stays the safe alternative. Nature Computational Science is worth trying only if the results are strongly mechanistic on both systems. Before submission, re-check MLST's metrics and decision time in a normal browser, since iopscience blocks automated fetching.

Downloaded official ULAKBİM files are in `C:\Users\ABDULH~1\AppData\Local\Temp\claude\c6eff6e074d30b33\4add04dd-e721-4eee-b705-23ef12c2b27a\scratchpad\ekual\` (fon.xlsx, yta.xlsx, duyuru.docx, akis.pdf).