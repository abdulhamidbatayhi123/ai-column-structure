# Constraint, information or projection? What thermodynamic structure buys learned reduced models of open dissipative systems

[![verify](https://github.com/abdulhamidbatayhi123/ai-column-structure/actions/workflows/verify.yml/badge.svg)](https://github.com/abdulhamidbatayhi123/ai-column-structure/actions/workflows/verify.yml)

Paper 3B of a research program on AI for water sorption in metal-organic frameworks (MOFs).
This repository holds the complete, open design of a pre-registered computational study.

**The question.** Physics can enter a learned reduced model in three ways that are usually
confounded:
- as a constraint on its dynamics;
- as physically meaningful information it can use;
- as a projection applied at rollout.

Its energy and entropy can also be fixed from physics or learned. This study separates
these effects. It does so in learned latent models of two open, advection-dominated,
dissipative systems: a non-isothermal MOF water-adsorption column (two isotherms, MOF-303 and
Al-fumarate) and a two-temperature packed-bed thermocline store with flow reversal. It
measures what each part buys in long multi-cycle stability, data efficiency, short-horizon
accuracy and design decisions. Every hypothesis, metric, threshold and verdict word is fixed
in advance, so a negative answer is as publishable as a positive one.

## Status (2026-11-09)

**Design complete; the full study pipeline, including the pilot runner, is implemented, reviewed and tested; the pilot data are being generated. No confirmatory data exist yet.**

| step | state |
|---|---|
| literature review and gap | done; checked live, two reference-audit rounds plus an independent Claude Science review; Lyon prior art read in full |
| thermodynamic formulation, both systems | done; three independent review rounds; verified numerically (`verify/`, CI on every push) |
| materials | fitted to the published isotherms (`docs/materials_3B.md`) |
| pre-registration | approved by the author 2026-11-05; Stage-1 revision of 2026-11-08 |
| study code (`p3b/`) | reference simulators with an exact sparse Jacobian, latent model and every arm, training, evaluation, statistics, pilot rules, power simulation and Kaggle jobs; each plan reviewed independently; 401 tests pass (CI) |
| reference timing on Kaggle | done: system A 18 to 26 CPU-s per cycle at 2048 cells, so the data-cost contingency is not applied (`docs/compute_3B.md`) |
| pilot (data, learning rates, step size, variance, thresholds, decoder gate) | runner complete and independently reviewed: five Kaggle phases with a blinded collector (`kaggle/pilot_jobs.py`, `scripts/pilot_collect.py`); pilot data generation running on Kaggle |
| freeze | after the pilot, before any confirmatory data |
| confirmatory study | about 620 CPU-equivalent hours, on Kaggle CPU |

## The design in brief

1. **Structure that matches the physics.** Both systems are port-metriplectic with an empty
   Poisson part. Their friction operators hold the dissipative mechanisms, and convection and
   wall heat are ports. Every production term is provably non-negative, including the
   upwind mixing, which is a Bregman divergence ([formulation](docs/formulation_3B.md)).
2. **Paper 1 is an exact limit.** With the water's heat capacity set to zero, the column
   reproduces Paper 1's verified solver to below 2×10⁻⁹. The reference adds the adsorbed
   water's heat capacity, which Paper 1 omits.
3. **Controls that separate the effects** ([pre-registration](PREREG_P3B_draft.md),
   Section 3):
   - **U:** the same model without structure;
   - **U+F:** U given all of the structured model's physical inputs (information);
   - **U+proj and U+F+proj:** the same, projected at rollout without retraining (projection);
   - **S-L:** the structured model with learned instead of physical potentials;
   - **P:** a physics-penalty version;
   - **ablations;**
   - **FNO:** Paper 1's strongest field model;
   - **C:** a compensated coarse consistent solver with no learning;
   - **C+L:** C with a learned non-negative closure.
4. **Verdicts fixed in advance.** A 14-row ordered decision list maps every outcome to one
   headline (checked over all 64 combinations by `verify/check_decision_list.py` in CI).
   Consistency words qualify each headline on the second isotherm and on the second system.
5. **A decision endpoint.** Each surrogate optimises the cycle, and the reference measures
   the regret.

## Repository map

| path | content |
|---|---|
| [`PREREG_P3B_draft.md`](PREREG_P3B_draft.md) | the Stage-1 pre-registration: at-a-glance tables, arms, hyperparameters, metrics, gates, tests, verdict words, decision list, power, pilot rules |
| [`docs/prereg_rationale_3B.md`](docs/prereg_rationale_3B.md) | the reason for each rule and the change history, mapped to the review codes |
| [`docs/formulation_3B.md`](docs/formulation_3B.md) | state, energy, entropy, friction operator, ports and latent model, for both systems |
| [`docs/materials_3B.md`](docs/materials_3B.md) | the MOF-303 and Al-fumarate isotherms fitted to the published figures, and the heats from the measured step shift |
| [`docs/lit_review_3B.md`](docs/lit_review_3B.md) | the literature and the exact gap |
| [`docs/compute_3B.md`](docs/compute_3B.md) | compute plan, measured reference cost, ceiling and reduction order |
| [`docs/INTERFACE_3A.md`](docs/INTERFACE_3A.md) | what this study needs from Paper 3A's learned isotherm potential |
| [`docs/terminology_3B.md`](docs/terminology_3B.md) | every term defined once, and the phrases the paper must not use |
| [`RETRACTIONS_P3B.md`](RETRACTIONS_P3B.md) | the correction ledger: every error found, its cause, its fix and how it was found |
| [`p3b/`](p3b/) | the study code: reference simulators (`simulate_a.py`, `simulate_b.py`, `exact_jacobian.py`), data generation and gates, latent model and arms (`latent/`), coarse solvers C, C+L and MUSCL-C (`solvers/`), training (`train/`), evaluation (`eval/`), statistics and verdict words (`stats/`), pilot rules, runner and power simulation (`pilot/`) |
| [`kaggle/`](kaggle/) | the read-only code kit, job lists and the resumable queue that runs every heavy job on Kaggle |
| [`scripts/`](scripts/) | pilot rules, power simulation and timing merge entry points |
| [`results/`](results/) | measured results that the design depends on (reference timing and the data-cost decision) |
| [`verify/`](verify/) | design verification: the consistent column and thermocline store, multi-term isotherms, checks, results, timing, the isotherm fits, the number gate and the decision-list check |
| [`tests/`](tests/) | property tests of the thermodynamics of both systems and unit tests of the whole pipeline |
| [`research/`](research/) | the source journals and every review report |

## Reproduce the verification

The column code imports three files from Paper 1's public release, unchanged (tag
[v2.0.1](https://github.com/abdulhamidbatayhi123/ai-mof-cof-dynamics/tree/v2.0.1)):

```bash
git clone https://github.com/abdulhamidbatayhi123/ai-column-structure.git
git clone --branch v2.0.1 --depth 1 https://github.com/abdulhamidbatayhi123/ai-mof-cof-dynamics.git
cd ai-column-structure
pip install -r requirements.txt
python verify/run_checks.py                 # writes verify/results_run_checks.json
python verify/check_results.py --bindings   # pass criteria and number bindings
python verify/check_decision_list.py        # the headline list is an ordered partition
python -m pytest -q tests                   # thermodynamic property tests and pipeline tests
python verify/materials/fit_isotherms.py    # refits the isotherms (fetches the open figure; needs Pillow)
```

| check | what it tests | result |
|---|---|---|
| V1 | the water-heat-capacity-zero limit against Paper 1's solver | below 2×10⁻⁹ in c, below 10⁻⁸ in q, below 2×10⁻⁷ K in T |
| V2 | balance and entropy identities, and the sign of every production term, over temperature-swing cycles | identities below 10⁻¹⁴; every term ≥ 0 |
| V3 | concavity of the gas-stream and cell entropies | negative semidefinite |
| V4 | whether coarse-grained data keep the balances and the entropy inequality | yes, with the fine outlet stored; the coarse outlet cell is off by 4.0 % |
| tests | the same identities for both fitted isotherms and for system B in both flow directions, plus the pipeline | 401 pass |

## Related work by the same author

- Paper 1 (`ai-mof-cof-dynamics`, doi:[10.5281/zenodo.23323685](https://doi.org/10.5281/zenodo.23323685)):
  what binds a surrogate's error on unseen materials. Its solver is this design's exact
  limit.
- Paper 2 ([`ai-kinetics-discoverability`](https://github.com/abdulhamidbatayhi123/ai-kinetics-discoverability)):
  when equation discovery can recover adsorption kinetics.
- Paper 3A (`ai-isotherm-thermo`, in preparation): a thermodynamically consistent learned
  water isotherm. It is the planned constitutive layer of this study.

## Author

Abdulhamid Batayhi, Department of Mechatronics Engineering, Faculty of Mechanical
Engineering, Yıldız Technical University, Istanbul, Türkiye.
ORCID [0009-0007-0374-6755](https://orcid.org/0009-0007-0374-6755).

## License and citation

Code and documents are released under the MIT License ([`LICENSE`](LICENSE)). Please cite
with [`CITATION.cff`](CITATION.cff). Third-party papers are cited, never redistributed.

## Declaration of generative AI use

During the preparation of this work the author used a generative AI assistant (Claude,
Anthropic) as a tool to assist with drafting and language editing, with writing and checking
code, and with checking bibliographic records. The author designed the study, made every
scientific decision, reviewed and edited all content, and takes full responsibility for the
content of the published article.
