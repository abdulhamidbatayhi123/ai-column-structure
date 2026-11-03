# ai-column-structure: Paper 3B of the AI-for-MOF water sorption program

**What thermodynamic structure buys a learned model of a temperature-swing adsorption column.**

This repository holds the design of Paper 3B: a learned reduced model of a non-isothermal
adsorption column whose energy and entropy are fixed from known thermodynamics, with the
inlet, outlet and wall as ports. It is compared against the same model without structure, a
penalty-trained version and a coarse consistent solver, on long multi-cycle rollouts, data
efficiency and admissibility. The test was declared in advance.

Status (2026-11-03): **design, verified on small grids. No study data exist.** The
pre-registration is a revised draft awaiting the author's review. It was revised after an
independent adversarial review (`research/review_adversarial_2026-11-03.md`, 36 findings,
all addressed).

| file | content |
|---|---|
| `docs/lit_review_3B.md` | verified literature review and the exact gap (updated 2026-11-03) |
| `docs/formulation_3B.md` | state, energy, entropy, friction operator, ports; checked against Paper 1, Section 2 |
| `PREREG_P3B_draft.md` | hypotheses, arms, metrics, verdict words, gates, power, pilot rules |
| `docs/compute_3B.md` | compute and memory budget for a CPU-only laptop |
| `docs/INTERFACE_3A.md` | what this paper needs from Paper 3A (the learned isotherm potential) |
| `verify/` | design verification of the formulation on small grids, with results JSON |
| `tests/` | fast property tests (run with Paper 1's venv: `python -m pytest -q tests`) |
| `research/` | source journals (every source fetched live) and the adversarial review |

Related repositories by the same author: `ai-mof-cof-dynamics` (Paper 1, the falsification
ladder), `ai-kinetics-discoverability` (Paper 2) and `ai-isotherm-thermo` (Paper 3A). The
reference model of this paper reduces exactly to Paper 1's solver in a stated limit (water
heat capacity zero). Verified to 10⁻⁹ in `verify/`.

## Author

Abdulhamid Batayhi, Department of Mechatronics Engineering, Yıldız Technical University,
Istanbul, Türkiye. ORCID 0009-0007-0374-6755.

## Declaration of generative AI use

During the preparation of this work the author used a generative AI assistant (Claude,
Anthropic) as a tool to assist with drafting and language editing, with writing and checking
code, and with checking bibliographic records. The author designed the study, made every
scientific decision, reviewed and edited all content, and takes full responsibility for the
content of the published article.
