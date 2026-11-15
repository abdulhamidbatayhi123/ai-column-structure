# Lessons for Paper 3B from the PINN-machining papers and the lessons file

Everything below comes from the files named in each lesson. The two live sources I opened on 2026-11-16 are the MLST "About" page and the IOP generative-AI policy page. Facts about MLST taken from older project files carry their fetch date and were not re-checked here. Lessons marked **[PREREG]** apply mainly to pre-registered or falsification papers.

## Before writing: two MLST policy conflicts the author must settle

Both come from the IOP generative-AI page (publishingsupport.iopscience.iop.org/questions/generative-ai-tools/), read 2026-11-16.

- **A. Reviewer reports and responses.** IOP lists "upload reports from reviewers to generative AI tools" as unacceptable. Responses to reviewers may not be AI-generated; AI may only "improve the language".
  - The Paper 1 and Paper A revisions were run by agents that read the reports and drafted the letters (`revision_eaai/manuscript/09_response_letter.md`, the `REVISION_R*_RESPONSE` files).
  - At MLST that workflow is not allowed. Plan before submission how a revision would be handled: the author reads the reports and writes the answers, with AI for language only.
- **B. The AI statement template.** The template reads "The authors reviewed, validated and critically assessed all AI-assisted outputs". It goes in the Acknowledgements, with Methods detail if AI was used in the research itself.
  - This conflicts with the author's rule (`PAPER_WORKFLOW_LESSONS.md` §1b.10): never write that the authors "checked" the AI's work.
  - IOP also forbids AI generating or modifying reference lists. Each source must be checked by the authors against the original. `MLST_FORMAT_PLAN.md` §1 already flagged this.
  - Lesson 2026-11-15 says: "if the honest disclosure conflicts, change journal, never the disclosure." The author decides.

## Ranked lessons (highest impact on acceptance first)

1. **The target journal's live rules come first, recorded in a compliance table.**
   - Source: LESSONS §1a; PAPER_CHECKLIST §1a; worked example `paperA/submission/EAAI_COMPLIANCE_2026-11-12.md`.
   - Live on 2026-11-16 (MLST About page):
     - Research paper "normally not more than 8500 words".
     - Single- or double-anonymous review, author's choice.
     - All data needed to verify or reproduce the findings must be shared.
     - Code goes in the supplement or a linked repository; OSI licences are encouraged.
     - Preprints are encouraged.
     - The page says nothing about novelty bars or negative results ("not found").
   - From `MLST_FORMAT_PLAN.md` (fetched 2026-11-01, not re-checked):
     - Abstract normally at most 300 words, with no references, equations or undefined acronyms.
     - Figure text 8–12 pt at about 8.5 or 15 cm width; colour must not be the only carrier of information.
     - Captions self-contained and free of acronyms; no colour in tables.
     - The supplement is "not included in peer review by default".
     - What counts toward 8,500 words is not stated.
   - Still to do: read 5–10 recent MLST papers on structure-preserving or physics-informed surrogates for structure and tone.

2. **Claims must match the evidence; this is where reviewers attacked.**
   - Source: `paperA/submission/PAPER1_LESSONS_FOR_EAAI.md` §2. Paper 1's reviewers did not dispute the method. They disputed what the evidence licenses: information parity, validation tier, claim scope, seed robustness and vocabulary.
   - Overclaims Paper 1 had to remove (`COVER_LETTER_EAAI.md` before revision against `09_response_letter.md` item 7): "first", "glass-box / fully interpretable", "real-time", "digital-twin core", "unseen cutting speed".
   - For 3B: keep a terminology file with banned words (pinn-machining `manuscript/terminology.md`). Candidates:
     - "thermodynamically consistent" only for what is proven.
     - "stable" only with its metric.
     - "generalizes" only out of distribution, as H5 tests.
     - "physics-informed" kept apart from "structure-preserving".
     - Never "the first".

3. **Fair comparison at matched information.**
   - Source: decision letter R2.3, R4.M2.a–d, R6.6 and R6.7 (`revision_eaai/00_DECISION_LETTER.md`); v6 referee M3 (`paperA/REFEREE_PASS_v6_MSSP_INDEPENDENT.md`); round-4 review F1.
   - The single most-demanded control: give the unstructured model the same features and the same architecture, with equal tuning, budget and cost.
   - 3B's U+F, U+proj, U-long and S-L arms are exactly this. Lead with them in the abstract and Fig. 1, and document equal tuning and wall-clock for every arm.
   - Paper A was attacked for a missing arm its own theory recommended. Check that no arm a referee could derive from 3B's own framing is absent.

4. **Do not headline a result that follows from the design.**
   - Source: v6 M1 and M2; `PAPER1_LESSONS_FOR_EAAI.md` §5.5.
   - Paper A's +93.9 pp headline correlated r = 0.9985 with the sampled conductivity range, and its H2 was true by construction. Both were demoted to sanity checks.
   - For 3B: any verdict that holds by construction should be labelled a check (for example S has zero balance drift; H3a admissibility). Lead with the informative contrasts: S against U+F, U+F+proj and S-L.

5. **Evidence labels instead of hedging adverbs; report null and adverse results in full.** **[PREREG]**
   - Source: `paperA/submission/PAPER1_STYLE_GUIDE.md` §2.
     - Every number carries "(pre-registered)", "(exploratory)", "(descriptive)" or "(post hoc)".
     - Inconclusive results are given with their interval and "not ranked".
   - Paper A reported "results against our starting position" in the abstract and contributions (round4 `PAPER_A_article.md` contribution 4). Both independent referees praised it (v6 "What is strong"; v7 §A.1).
   - `AUTHOR_DIRECTIVES_AND_OPEN_DECISIONS.md` §3: hiding baselines that tie would destroy the argument. Turn the tie into the finding ("accuracy has saturated; the question is which is trustworthy").

6. **Keep registered verdicts true to the frozen text.** **[PREREG]**
   - Sources: round-3 review F1; round-4 F2 (`AgentHub/projects/pinn-machining/messages/`); LESSONS addenda R3 and final review.
   - When the scorer and the frozen text disagree, the text's label is the headline. The scorer's label is a declared deviation, and both are printed.
   - Pin seed sets to the text. Never mix a three-seed arm with a five-seed reference, and forbid the combination in a build gate.
   - Every deviation appears in a printed "Deviations from the pre-registration" table, not in comments (LESSONS 2026-11-11).

7. **Before submission, tick every "reported beside it" item in the pre-registration against the manuscript.** **[PREREG]**
   - Source: LESSONS, Paper 3A R1 and Paper 2 2026-11-11. Omitting a secondary or descriptive count is read as selective reporting.
   - For 3B: H3b, H5, H6, FNO, MUSCL-C and every consistency word.

8. **Deposit the frozen protocol with an independent timestamp and cite its DOI.** **[PREREG]**
   - Source: v6 Mo6 (a self-held registration with no external timestamp); LESSONS 2026-11-11 (Paper 4); `T4_venue_3B.md` §3.
   - Put it on Zenodo, optionally reviewed at PCI RR, before any confirmatory run. Paper A had to call its registration "self-held".

9. **Find the mechanism of a failure before generalizing from it.**
   - Source: R4.M6 (decision letter). Paper 1 traced its physics-only collapse to a 6.4 × 10⁴ gradient-scale imbalance (response letter item 4). v6 M3 and round-3 F8 asked the same of Paper A.
   - For 3B: if S or P fails, give the measured cause (units, conditioning, step size), or say it is untested. Never write "structure does not help" in general.

10. **Seeds and statistics.**
    - Sources: R2.4, R4.M5, R4.M8; round-3 F6 (three seeds give only 10 bootstrap resamples).
    - LESSONS §3: a REFUTED verdict needs an interval that excludes the declared effect.
    - LESSONS (Paper 3A): simulate power on the verdict; use exact Wilcoxon with Walsh intervals at small n; check the paired median against the marginal median.
    - Say "(3 seeds)" wherever 3 seeds apply. 3B uses 3 seeds in several cells.

11. **Introduction structure (the proven template).**
    - Source: `PAPER1_STYLE_GUIDE.md` §1.2–1.3; round4 `PAPER_A_article.md` §1. Six paragraphs, no subheadings:
      1. Problem and stakes.
      2. Classical models and their limit.
      3. Machine learning.
      4. The candidate method and its known failure modes.
      5. "N gaps follow", with a scoped novelty sentence ("we found no study that…", never "first").
      6. "The present study therefore asks…", then numbered contributions, then a one-line roadmap.
    - Each contribution is a noun phrase with its headline number, evidence label and section pointer. At most five.
    - Paper A also stated what the paper does NOT claim and that the theory "explains but is not tested by" the experiment.
    - The professor's rule (C3): contributions and novelty come early.

12. **The title asks the question; the abstract is ordered and acronym-free.**
    - Sources: PAPER_CHECKLIST §1b.1; style guide §5; Paper A title "When does a hybrid physics–neural model transfer…?". 3B's README title is already a question.
    - Abstract order:
      1. Context.
      2. Gap clause.
      3. Design: controls, frozen plan.
      4. Three or four results, each with its number and label.
      5. One implication sentence that states the limit.
    - Leave margin under caps (LESSONS §1b.4).
    - Round-3 F9: Paper A's abstract was too dense, with six findings and three labels. Keep 3B's to about three results.

13. **Diagrams first, and one key figure.**
    - Source: PAPER_CHECKLIST §1b.2; LESSONS kemal 2026-11-09 ("the key result as ONE figure a reader understands at a glance").
    - Fig. 1: method schematic (where physics enters: constraint, information, projection).
    - A workflow figure as a map of the paper, each Results section a box tagged confirmatory, exploratory or descriptive (`PAPER1_STYLE_GUIDE.md` §3.2, `fig_workflow.py`).
    - A key figure: a forest plot of every test against its own frozen band, with a visible "pending" slot (LESSONS Bio-PV 2026-11-11).
    - Write the key-figure script before the freeze, on synthetic fixtures (LESSONS 2026-11-11).
    - Co-author C32: include concrete trajectory panels (an outlet front or temperature against time).

14. **One figure style module, built to MLST's rules.**
    - Source: style guide §3.1; LESSONS §5.
    - Okabe–Ito colours with fixed meanings across figures, plus marker or line style so figures survive greyscale.
    - Panel letters outside the axes; frameless legends; the key metric in the legend.
    - save() asserts the font floor and fit, refuses missing glyphs, writes a provenance file with SHA-256 hashes and a figure-data CSV.
    - IOP wants 8–12 pt at about 8.5 or 15 cm (MLST_FORMAT_PLAN, 2026-11-01). Paper A's 6–6.5 pt floor must rise to 8.
    - Measure the printed font size in the compiled PDF (LESSONS Bio-PV 2026-11-09).
    - Look at every page and every figure.

15. **Captions, tables and prose.**
    - Source: PAPER_CHECKLIST §1b.3; professor comments C15, C23, C42 (`PAPER1_LESSONS_FOR_EAAI.md` §3).
    - Captions: 1–2 sentences starting with a title phrase, defining the encodings and n, with no interpretation.
    - No bold run-in labels. No subheadings in Introduction, Discussion or Conclusions. Plain engineering headings, nothing clever.
    - Figure paragraphs open "Fig. N shows…" and end with a one-line takeaway.
    - Tables: no vertical rules; bold marks only the best per column; a decision-summary table (family | question | outcome and estimate).
    - Conclusions: "The main conclusions, demonstrated in the investigated numerical setting, are:", a numbered list, then a scope sentence.

16. **Readability over audit-log style.**
    - Source: v7 M4; round-3 F9; `PAPER1_LESSONS_FOR_EAAI.md` §5.3.
    - Keep internal codes (T1A, H5d, R1–R6) and amendment numbers out of the main text. Spell out model names.
    - Open the Results with one ordering sentence along the thesis axis.
    - 3B has many arm codes (U+F+proj, S−Π…). Define them once, in a Fig. 1 legend or a table, and use words in the prose.

17. **Keep the main text under the limit with a conservation gate.**
    - Source: MLST_FORMAT_PLAN §3–4. Paper A had to go from 27k to 8.3k words.
    - Write 3B to about 8,300 words from the start. Move material verbatim to the supplement.
    - Keep every pre-registered verdict, its interval and every quantified caveat in the main text, because reviewers may not read the supplement.
    - Keep proofs or derivations a referee must check in an appendix, not the supplement.
    - Use a REQUIRED-wording gate so cuts cannot delete referee-required qualifiers (LESSONS R2).

18. **Build gates on numbers and words.**
    - Source: LESSONS §1b.6, §3; Bio-PV `\V{key}{value}`.
    - Every number is bound to its results file. Any unbound digit fails the build. Relative tolerance for small values. Required and forbidden wording. Rendered-text gates.
    - Prove each gate bites with a planted defect (`AUTHOR_DIRECTIVES_AND_OPEN_DECISIONS.md` directive 3: "a guard that is real, is passing, and is checking the wrong thing").
    - Gates bind numbers, not claims (LESSONS Bio-PV 2026-11-09). A hostile read of the rendered PDF is still needed.

19. **Independent review rounds until a full pass finds nothing material.**
    - Source: `AUTHOR_DIRECTIVES_AND_OPEN_DECISIONS.md` §7.
    - Paper A needed seven referee passes and four review rounds. Each round found majors the earlier ones missed.
    - Run separate briefs:
      - number trace;
      - live references, including whether each cited work supports its sentence;
      - Methods against code against pre-registration;
      - simulated editor plus referees holding the journal's editorials (LESSONS 2026-11-15);
      - readability.
    - Give reviewers the exact snapshot or commit, not your reasoning.
    - Check every suggested rewording against its number (LESSONS R2, R4: reviewer rewrites carried wrong rounding and wrong seed bases).

20. **Reproducibility package.**
    - Source: R4.m3 and R4.m4; Paper A HANDOFF.
    - Versioned Zenodo DOI (reserve early with prereserve_doi); scripts that rebuild every figure and number; hardware and separate training and inference cost per arm; OSI licence (MLST_FORMAT_PLAN gate 17).
    - Scan the deposit for paths, tokens and third-party data before upload.
    - The data statement cannot be amended after publication (MLST_FORMAT_PLAN, 2026-11-01).

21. **Anonymity, if double-anonymous review is chosen.**
    - Source: round-3 F3; round-4 F4; LESSONS 2026-11-10.
    - Contextual tells identify authors without names: "an earlier study's revision…", "in revision at…", orphan year letters.
    - Use an anonymized deposit copy.
    - MLST_FORMAT_PLAN §1 recommends single-anonymous because the companion papers are public. 3B builds on the public Paper 1 (doi:10.5281/zenodo.23323685), so single-anonymous is simpler.

22. **The cover letter.**
    - Source: round4 `COVER_LETTER_paperA.md` against the original `COVER_LETTER_EAAI.md`.
    - About one page (400–450 words of body; LESSONS pinn 2026-11-09).
    - Structure: the question; three contributions with numbers and labels; "why this design" (Paper A: "Why exact synthetic truths"); the registration in two sentences; related or companion work; the AI statement.
    - Concede limits up front. Paper A said it "claims no experimental validation".
    - Re-read it against the final paper; letters drift stronger (LESSONS Paper 4, 2026-11-15).
    - Answer the journal's scope words. MLST: ML "motivated by physical insights" (T4_venue_3B, 2026-11-08).

23. **Response letter, for later.**
    - Source: `revision_eaai/manuscript/09_response_letter.md`.
      - Opening line: "every comment as a request for evidence, not for rewording".
      - A numbered summary of changes.
      - Each comment quoted verbatim, with what was done, the numbers and the location, marked [FULL].
      - Kept to about 30 pages.
    - At MLST the author writes this letter (conflict A above).

24. **Prior-art and novelty discipline.**
    - Source: `AUTHOR_DIRECTIVES_AND_OPEN_DECISIONS.md` §5. Twelve candidate "first" claims all fell to literature counterexamples. What survives is the property or measurement claim, in hedged form.
    - R6.1–6.3: a critical, recent related-work section with a positioning table.
    - LESSONS 2026-11-11: read the closest prior work's result tables, not its abstract.
    - Present any theory or formulation as organizing the test, not as the main novelty when it assembles classical facts (v6 M5; Paper A §4).

25. **Synthetic-only evidence is the standing reject risk.**
    - Source: v6 F1 (MSSP fit); round-3 verdict; round-4 F8.1.
    - 3B uses simulated references only. State on page 1 what known truth buys (exact balances, a consistent reference) and what it cannot show.
    - Use a fixed vocabulary, never "validated". Paper 1's three accuracy tiers are the model.
    - Choose the venue whose readers accept simulation studies. MLST's scope (live 2026-11-16) includes conceptual and methodological ML advances tied to scientific problems.

26. **The pilot and the stopping rule.** **[PREREG]**
    - Source: LESSONS Paper 4 2026-11-10.
    - A positive control must show effect size. Fix the stopping rule before the deciding result, and obey it when it bites.
    - Run every violation threshold on reference data first. Calibrate on pilot data that is never scored. Use null worlds with no shared component.
    - Plan the paper that survives a failed primary.

## What made the final Paper A strong (summary)

Paper A was ready to submit on 2026-11-10 (hub HANDOFF). What made it strong:

- **Every number traceable.** All numbers bound to hashed artifacts; v7 checked 20 against source files and found 0 mismatches.
- **Candid deviations.** Errata and deviations declared, and the results against its own starting position stated in the abstract and contributions.
- **The informative contrast up front.** The headline became "validation on the base material does not certify transfer".
- **Clean front matter.** Acronym-free title, keywords and abstract under every cap with margin.
- **Figures that carry the argument.** Schematic, workflow map and one "seen vs unseen" key figure, all from one style module.
- **A checked package.** An anonymized, hash-verified deposit and a one-page cover letter.
- **The remaining weaknesses** were the synthetic-only evidence and the density for a broad readership.