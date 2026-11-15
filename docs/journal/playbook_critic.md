**Completeness critique: Paper 3B MLST playbook (2026-11-16)**

I checked the playbook against the four reports. I also checked the repo-derived facts against `README.md`, `PREREG_P3B_draft.md`, `docs/freeze_decisions_3B.md`, `docs/lit_review_3B.md` and `LICENSE` in C:/Users/abdulhamid batayhi/Desktop/ai-column-structure. I edited nothing.

**(a) Claims a report does not support, or contradicts**

1. **§1 item 2 says Schaffer–Exl has "no checks on the decoded state".** The papers report contradicts this. They do report that the decoded state breaks the unit-norm constraint (about 0.10 and 0.92), and §5.1 of the playbook itself cites the 0.92. What they lack is decoded conservation and entropy checks. Reword.
2. **Conflict B: the reports disagree, and the playbook hides that.**
   - The journal report says the current AI declaration wording "is compatible" and only has to move to the Acknowledgements.
   - It also quotes a different template: "During the preparation of this manuscript, the authors used [Tool Name, Version] to support [describe use]."
   - Only the lessons report quotes "reviewed, validated and critically assessed".
   - Re-read the IOP page live before the author is asked to decide.
3. **"P1" is ambiguous throughout.** It stands for Schaffer–Exl (MLST), for the program's Paper 1 (archive doi:10.5281/zenodo.23323685, going to Digital Discovery, not MLST) and for PINN-machining Paper 1 in §4.
   - Row 5, "Cite P1 by its MLST DOI", is wrong if it means the program's Paper 1.
   - Rename the recent papers M1–M10.
4. **§1 item 4, "The policies reward 3B's design".** The reports only say the policies fit the design. Nothing says review "rewards" it.
5. **§1 item 2, "the editors take the 'does physics help' question".** This is inferred from two papers, not supported by a report. The journal report found no editorial signal on the question.
6. **Seed-count list (rank 6 and Q5) is incomplete against the PREREG design table.** Other 3-seed cells:
   - A-primary sizes 16/64/128 for U, P and S;
   - every replication cell;
   - S-L on system B (3/3). This matters because the system-B attribution then mixes 5-seed S with 3-seed S-L.
7. **Row 2 (abstract) has two sources.** The journal report read it live on 2026-11-16; the lessons report took it from MLST_FORMAT_PLAN (2026-11-01). Fine, but cite the live one only.
8. **Repo facts not in any report, but true in the repo:** 0.0211/0.0231 and 0.0132/0.0131; below 2×10⁻⁹; 14-row list with 64 combinations; 8 scenarios; n = 48; δ = 0.5 under option (a). The citations should point to README/PREREG.
9. **Open item 11 (licence) can be closed now.** `LICENSE` is MIT, which is an OSI-approved licence (confirm on opensource.org).
10. **`docs/lit_review_3B.md` already cites Baaiu 2009 (J. Process Control) and Peng 2007 with DOIs** (lines 120–125). "DOIs not checked" is only partly true. Peng 2007 is also missing from the must-cite list.
11. **P8/P5 mix-up.** The papers report's preface names P8 as the Glasgow published PDF. The playbook correctly uses P5. Note the inconsistency so nobody copies P8.

**(b) Missing journal requirements and policies for the checklist** (none checked live; all need a live check)

- Keywords: the P5 front matter has them, but they are not in the table.
- Licence and copyright: CC BY under the TÜBİTAK agreement, and the licence form.
- How the corresponding author claims TÜBİTAK eligibility at submission: institutional e-mail, institution selection. Also whether a master's student counts.
- Text-recycling and similarity check. The Zenodo PREREG, the arXiv preprint and the lit review will overlap verbatim with Methods. This risk is not addressed.
- Disclosure of related manuscripts under review (Paper 1 at Digital Discovery). IOP's concurrent and redundant-publication policy is not checked.
- Code and data access during review. P3 gave reviewers anonymised material. Decide whether the repo is public at submission or a reviewer link is given.
- Data availability statement template options (IOP offers set wordings), and the funding statement wording.
- AI use in the research itself must go in Methods. Agents wrote code and analysis, but the playbook treats Methods disclosure as optional ("any AI use").
- Whether appendices count toward the 8,500 words. Listed as open, but the whole §3.2 plan depends on it, so resolve it first.
- Suggested and excluded reviewers, cover-letter fields, and the submission system: not found.
- Figure and table count limits, graphical abstract, and whether the arXiv licence is compatible with CC BY: not found.

**(c) Literature a referee would expect**

These are from my own knowledge and were not opened live, so verify each before use.

*Missing from the must-cite list although the novelty report found them:*
- Praditia FINN, WRR 2022 (diffusion–sorption finite-volume network): very close to system A.
- Subraveti 2019; Wu 2025 PSA PINN; Galanti 2025; Bilal 2026; An 2026; Ceccanti DeepONet for TVSA (arXiv:2601.09491).
- Bermejo 2025; Gruber arXiv:2405.16305; Chung, Physics-conforming Latent Twins, CMAME 2027.
- GENERIC-FNO and Poisson-GENERIC operators (arXiv:2606.08343, 2610.05570); tLaSDI fracture (arXiv:2610.20614).
- List–Chen–Thuerey 2022; Melchers 2023; Sanderse review 2025; SymCLaw; entropy-stable flux networks (arXiv:2507.01795).
- Zhong (arXiv:2012.02334); Janssen 2026; arXiv:2603.02645, 2611.05031 and 2608.30788.
- Shen P10 (must be read first).

*Not in any report:*
- **Foundations:** Grmela and Öttinger 1997 (GENERIC); Morrison 1986 and Kaufman 1984 (metriplectic); van der Schaft (port-Hamiltonian).
- **Latent and reduced-order models:** neural ODE (Chen 2018); autoencoder reduced models (Lee and Carlberg 2020; Fresca DL-ROM); LaSDI and gLaSDI (Fries 2022); operator inference (Peherstorfer and Willcox 2016); SINDy.
- **Operator learning:** FNO (Li et al. 2021), needed because it is an arm; DeepONet.
- **Advection-dominated reduced models:** Kolmogorov n-width, e.g. Ohlberger and Rave. The S2 file already uses this.
- **Dissipative Hamiltonian networks:** Sosanya and Greydanus 2022; Dissipative SymODEN.
- **ML statistics and reproducibility:** Henderson 2018; Bouthillier 2021; Agarwal 2021 (rliable, bootstrap intervals); Owen 2007 (pigeonhole bootstrap, in PREREG); Holm 1979; Hodges–Lehmann.
- **Pre-registration in ML:** for example the NeurIPS pre-registration workshop papers.
- **Domain:** MOF-303 water-harvesting sources (materials provenance); Schumann 1929 thermocline model and a thermocline review; Ruthven's textbook.

**(d) Advice that could hurt acceptance**

1. **Density.** About 14 arms, H1–H6, three settings, six Results subsections, 7 figures and 3 tables in 8,300 words repeats the round-3 F9 density failure.
   - Keep the primary family and attribution in the main text.
   - Move H5, H6, FNO, MUSCL-C and the size sweep to an appendix with one-line verdicts.
2. **The title overgeneralizes.** "Open dissipative systems" claims more than two simulated one-dimensional columns support, which breaks rank 2. Scope it in the title.
3. **Disparaging precedents.** "Statistics are the weak spot of the field", "P1 was weak exactly here" and "3B's statistics exceed every precedent" must stay out of the manuscript and cover letter. Schaffer, Exl or Lenau are likely referees. Claim 6 must be scoped to "among nine recent MLST papers we read", or dropped.
4. **PCI RR now.** Stage 1 must come before data, but the pilot has already run, and review takes months. MLST's PCI RR-friendly status was not checked. Recommend the Zenodo DOI only.
5. **Missing alternatives.**
   - TMLR: no fee, rigour-focused, and parametric tLaSDI was published there.
   - CMAME: the tLaSDI venue.
   - Computers & Chemical Engineering.
   - None checked live. The exclusions rest on a sparse comparison.
6. **The "procedures, not supervision" AI wording.** IOP asks how outputs were checked. A wording that omits human checking may be read as non-compliant. Decide only after re-reading the live page (item a2).
7. **The fallback abstract ("C makes the learned model unnecessary") is drafted now but not checked.** Make sure it does not prejudge the result in the frozen-text timeline. Keep it out of the repo before the confirmatory run, or hash it.
8. **Abstract gate rejecting "digits not bound to results".** This also blocks the legitimate count words (two sorbents, 50 cycles). Define an allow-list.