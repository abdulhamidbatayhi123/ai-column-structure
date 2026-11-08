# Paper 3B: the two material isotherms

Written 2026-11-05. Every number below is printed by `verify/materials/fit_isotherms.py` into
`verify/materials/fit_isotherms.json`. These are fits to figure readings of published
isotherms, not to raw measurements, and they are labelled that way wherever they are used.

## Why the materials had to be refitted

The first design took Paper 1's dual-site isotherm (one Langmuir term plus one cooperative
Sips term, with the Henry-site affinity fixed at 1/20 of the cluster affinity) and its
parameter ranges (step steepness n = 1 to 6). Fitting that form to the published water
isotherms of MOF-303 and Al-fumarate showed two problems:

1. **The real steps are much steeper.** In the cooperative-term form, MOF-303 needs n ≈ 9 and
   Al-fumarate n ≈ 19, far above Paper 1's sampled range. Steeper steps mean sharper fronts,
   stiffer reference solves and a harder decoder. The design must face that, not avoid it.
2. **Two terms cannot carry three regimes.** A real water isotherm has uptake before the step
   (MOF-303 holds about 6.6 mol kg⁻¹ at RH 0.10, a quarter of its capacity), a sharp step,
   and a steady rise after it. With the Henry affinity fixed at 1/20, the dual-site form puts
   almost no uptake before the step (about 0.2 mol kg⁻¹). With the affinity free, the
   "Henry" term moves to the post-step rise instead. Either way, one regime is lost.

## The three-term isotherm

```
q*(RH, 303 K) = q1 (RH/a1)/(1+RH/a1) + q2 (RH/a2)^n/(1+(RH/a2)^n) + q3 (RH/a3)/(1+RH/a3)
```

The three terms are the primary sites, the cooperative pore filling and the post-step
uptake. All three share one heat of adsorption, so q* depends on concentration and
temperature only through x = b(T) c. Everything in the formulation that depends on this
carries over unchanged (formulation, Section 3.2):
- the spreading pressure stays closed form, a sum of logarithms:
  `Π = Σ_k (q_k/n_k) ln(1 + (x/x_k)^{n_k})`;
- the inverse isotherm stays a one-dimensional monotone root;
- the temperature formula T(n, a, u) stays explicit.

Fits (Hanikel et al. 2019, Fig. 2, 30 °C; 90 % bootstrap intervals over bins):

| | MOF-303 | Al-fumarate |
|---|---|---|
| digitised bins | 193 | 184 |
| step position a2 (RH) | 0.129 (0.127 to 0.130) | 0.256 (0.255 to 0.257) |
| steepness n | 9.2 (8.6 to 10.0) | 18.6 (16.3 to 21.7) |
| uptake at RH 0.10 | 6.6 mol kg⁻¹ | 0.7 mol kg⁻¹ |
| uptake at RH 0.90 | 24.6 mol kg⁻¹ | 23.2 mol kg⁻¹ |
| fit RMSE | 1.06 mol kg⁻¹ (4.2 % of the maximum) | 1.55 mol kg⁻¹ (6.7 %) |

The RMSE includes digitising error. The adsorption and desorption branches are not
separated, and Al-fumarate shows a small hysteresis that no single-valued isotherm can
represent. The full parameter set, with all intervals, is in the JSON.

## The heat of adsorption, from the measured temperature shift of the step

In the dilute-solute gas model (formulation, Section 3.1) the isotherm's temperature
dependence is set by a concentration-based heat. The heat released per mole adsorbed equals
that same quantity. A pressure-based Clausius-Clapeyron fit of the same data gives a heat
larger by about RT. The design therefore fixes ΔH from the measured temperature shift of the
step, in concentration units. The model then moves with temperature as the real isotherm
does, which is what sets the desorption working capacity at 333 to 363 K, and it releases
about RT less heat than the measured isosteric heat. That RT offset is the one the
formulation states (Section 3.4).

| | concentration-based heat (used, = −ΔH) | pressure-based Q_st (check) | data |
|---|---|---|---|
| MOF-303 | 51.3 kJ mol⁻¹ (51.1 to 51.7 over three uptake levels) | 53.8 kJ mol⁻¹ | step positions at 10, 25, 35, 50 °C, SI Fig. S17, digitised by Paper 3A |
| Al-fumarate | 48.5 kJ mol⁻¹ | 51.1 kJ mol⁻¹ | step positions 0.25, 0.28, 0.33 at 30, 40, 60 °C (Laurenz et al. 2020, a coating) |

Independent checks:
- **MOF-303.** The pressure-based value, 53.8 kJ mol⁻¹, lies inside the plateau of the
  Clausius-Clapeyron Q_st curve in SI Fig. S25 (about 51 to 54 kJ mol⁻¹).
- **Al-fumarate.** Laurenz et al. report 52.2 ± 1.0 kJ mol⁻¹, close to the pressure-based
  51.1 kJ mol⁻¹.
- **The earlier convention would have been wrong.** With ΔH = −Q_st (the first design), the
  model would move Al-fumarate's step to 0.38 RH at 60 °C against the measured 0.33.

The Al-fumarate value rests on three step positions from a coated sample, so it has
moderate confidence. MOF-303's rests on four temperatures from the same powder as the fit.

## What this does not fix

- **Hysteresis** (small for Al-fumarate, very small for MOF-303) is outside any single-valued
  isotherm. It is stated as a limitation.
- **One heat for all three terms** is an approximation. A loading-dependent heat needs a
  potential with temperature-dependent terms, which is what Paper 3A's learned potential
  provides (INTERFACE_3A.md).
- **Figure readings are not raw data.** If the authors' numerical isotherms become
  available, the fits are repeated, and both values are reported.

## Sources

- Hanikel, Prévot, Fathieh, Kapustin, Lyu, Wang, Diercks, Glover, Yaghi, ACS Cent. Sci. 5,
  1699 (2019), doi:10.1021/acscentsci.9b00745, open access. Fig. 2 is fetched at run time
  from Europe PMC (PMC6813556). SI Fig. S17 was digitised by Paper 3A (provenance in
  `verify/materials/data/PROVENANCE.md`).
- Laurenz, Füldner, Schnabel, Schmitz, Energies 13, 3003 (2020), doi:10.3390/en13113003
  (step positions and heat as reported in Paper 1's `research/water_mof_parameters.md`; its
  full text was not reachable, so these values rest on the abstract and that record).
