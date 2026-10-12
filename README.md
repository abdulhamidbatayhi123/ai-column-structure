# ai-column-structure: Paper 3B of the AI-for-MOF water sorption program

**What thermodynamic structure buys a learned model of a temperature-swing adsorption column.**

This repository holds the design of Paper 3B: a learned reduced model of a non-isothermal
adsorption column whose energy and entropy are fixed from known thermodynamics, with the
inlet, outlet and wall as ports. It is compared against the same model without structure, a
penalty-trained version and a coarse consistent solver, on long multi-cycle rollouts, data
efficiency and admissibility. The test was declared in advance.

Status (2026-10-13): **design only. Nothing has been run.** The pre-registration is a draft
awaiting the author's review.

| file | content |
|---|---|
| `docs/lit_review_3B.md` | verified literature review and the exact gap |
| `docs/formulation_3B.md` | state, energy, entropy, friction operator, ports; checked against Paper 1, Section 2 |
| `PREREG_P3B_draft.md` | hypotheses, arms, metrics, verdict words, gates, power |
| `docs/compute_3B.md` | compute and memory budget for a CPU-only laptop |
| `docs/INTERFACE_3A.md` | what this paper needs from Paper 3A (the learned isotherm potential) |
| `research/lit_*.md` | the source journals behind the review (every source fetched live) |

Related repositories by the same author: `ai-mof-cof-dynamics` (Paper 1, the falsification
ladder) and `ai-kinetics-discoverability` (Paper 2). The reference solver of this paper
extends Paper 1's verified solver and reduces to it exactly in a stated limit.

## Author

Abdulhamid Batayhi, Department of Mechatronics Engineering, Yıldız Technical University,
Istanbul, Türkiye. ORCID 0009-0007-0374-6755.

## Declaration of generative AI use

During the preparation of this work the author used a generative AI assistant (Claude,
Anthropic) as a tool to assist with drafting and language editing, with writing and checking
code, and with checking bibliographic records. The author designed the study, made every
scientific decision, reviewed and edited all content, and takes full responsibility for the
content of the published article.
