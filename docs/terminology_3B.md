# Paper 3B: terminology and banned phrases

Written 2026-11-05, following the program's writing rules (`~/.claude/PAPER_WORKFLOW_LESSONS.md`,
Section 4). Every technical term is defined once here, together with the evidence needed
before the paper may use it. Any manuscript, README or cover letter for Paper 3B uses these
words in these senses only.

## Terms

| term | meaning in this paper | may be used only when |
|---|---|---|
| port-metriplectic (with an empty Poisson part) | the column's structure: L = 0, a symmetric positive semi-definite friction operator M with M∇E = M∇N = 0, and inlet, outlet and wall as ports (formulation, Section 4) | always with the qualifier "empty Poisson part" or "L = 0" on first use |
| GENERIC | the Grmela-Öttinger framework in general | describing the literature, or saying that 3B's structure is a special case of it; never as the name of 3B's model without "with an empty Poisson part" |
| structure | the constraints S has by construction: exact balances of N and E against the port fluxes, and non-negative total entropy production of the decoded state | always; the word never includes positivity (a decoder property shared by every arm) |
| admissible / admissibility | a rollout step with no balance violation above threshold and no entropy violation below −τ_S (pre-registration, Section 4) | with the thresholds stated nearby |
| unphysical output | a violation as defined under "admissible", or a non-finite state | only with the definition; never as a general adjective |
| second law (for the model) | non-negative *total* entropy production of the *decoded* fields, in continuous time | with both qualifiers (total; continuous time, or "to the counted tolerance" in discrete time); never "satisfies the second law" without them |
| fixed energy and entropy | E and S are the physical functionals of the decoded fields (formulation, Section 5.1), not learned | always true for arms S, S−M, S−Π |
| working capacity W | net water released in a desorption step, from the outlet (pre-registration, Section 4) | always with "net" on first use |
| long rollout | 60 cycles from the pre-wetted bed (40 under the data-cost contingency) | with the cycle count |
| stable (rollout) | not failed by cycle k under the pre-registered failure definition | with k and the definition |
| data efficiency | H2 at 32 training trajectories, and the data multiplier | only with the H2 verdict word |
| structure pays / helps / costs | the pre-registered headline rows (pre-registration, Section 5) | only as the frozen decision list assigns them |
| consistent (thermodynamically) | derived from one fixed energy and entropy, with every production term non-negative | for the reference and for arm C; for the latent model only as "consistent in total, in continuous time" |
| Paper 1 limit | c_w = 0, where the 3B reference reproduces Paper 1's solver (V1: better than 10⁻⁸) | with the tolerance |
| SMALL | a real effect below the declared minimum (ratio in (0.8, 1), conditions (a) and (b) hold) | only as a verdict word |
| REVERSED | the qualifier on REFUTED when the interval lies entirely above 1 (S significantly worse) | only beside REFUTED |
| CONSISTENT / PARTLY CONSISTENT / INCONSISTENT / NOT INFORMATIVE | the comparison of a primary verdict with the second isotherm or with system B (pre-registration, Section 7) | only as defined there; never "replicated" when both verdicts are INCONCLUSIVE |
| the constraint / the constraint in training / fixed physical potentials | attribution labels (pre-registration, Section 6) | only when S beats U+F, U+F+proj or S-L respectively |
| the physical information, not the constraint | attribution label (pre-registration, Section 6) | only when U+F beats U and S against U+F is EQUIVALENT by TOST; otherwise "the information suffices; the constraint's added value is unresolved" |
| useful against the coarse solver / a coarse consistent solver makes the learned model unnecessary / unresolved against the coarse solver | the three-way coarse-solver statement of the pre-registration, Section 6 | the first only when S is shown superior to every C grid and C+L; the second only when one is shown non-inferior to S |
| admissibility | what S's guarantees buy when H1 and H2 are REFUTED (rows 8a, 10a) | only when H3a = pass |
| system A, system B | the adsorption column, and the two-temperature thermocline store | always; "two systems", never "two materials", for A and B |
| second isotherm | the Al-fumarate-like replication on system A | never "a second system" |
| dilute-solute gas model | vapour as an ideal solute in an incompressible carrier (formulation, Section 3.1) | always named when a heat of adsorption is compared with 3A's ideal-gas values |

## Banned phrases and replacements

| do not write | write instead | why |
|---|---|---|
| "the first ..." | the exact gap sentence of the literature review, Section 5, with its date and search scope | absolute claim; the search is not exhaustive (Semantic Scholar refused requests) |
| "provably respects the second law" | "has non-negative total entropy production of its decoded fields in continuous time, by construction" | the guarantee is total, continuous-time, and about the decoded state |
| "a GENERIC model of the column" | "a port-metriplectic model with an empty Poisson part" | L = 0 by choice (formulation, Section 4.1) |
| "physically consistent" (for the latent model) | "exact balances and non-negative total production" | local (cell-wise) production is not guaranteed |
| "digital twin" | "reduced model" or "surrogate" | the term is used for device-level systems elsewhere (program, Paper 4) |
| "zero unphysical outputs" (as a finding) | "no violations above the declared thresholds, by construction (H3a)" | admissibility of S is a construction, not a result |
| "learns the physics" | "learns reduced coordinates and dynamics; energy and entropy are fixed" | nothing thermodynamic is learned |
| "dramatically", "significantly" (without a test) | the number, with its interval and test | numbers instead of adjectives |
| "isosteric heat" for the column's heat release | "heat released per mole adsorbed (dilute-solute model)" | it differs from 3A's isosteric heat by RT |
| "Paper 1 is thermodynamically inconsistent" | "Paper 1 is the c_w = 0 member of a consistent family; it omits the water's heat capacity" | ledger P3B-1 |
