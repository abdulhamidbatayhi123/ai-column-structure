# What Paper 3B needs from Paper 3A

Written 2026-10-12 by the 3B design session. Paper 3A (a thermodynamically consistent learned
isotherm) is designed in another session. This file states 3B's requirements so the 3A
session can read them. It does not edit or presume anything in 3A's files. If 3A's design
cannot meet a requirement, 3B adapts. The fallback at the end means 3B never waits on 3A.

## Why 3B needs 3A at all

3B fixes the column's entropy from known thermodynamics instead of learning it. The
adsorbed phase is the one piece whose thermodynamics is not textbook: its chemical
potential, entropy and energy come from the isotherm. 3B therefore needs the isotherm in
*potential* form, not as a curve q*(p, T). A potential gives the chemical potential, the
entropy, the energy and the heat of adsorption from one object, so they cannot contradict
each other (constraint C3 of the structure journal). That is exactly what 3A builds.

## Requirements, in order of importance

1. **One scalar potential for pure water on one framework**, in either form:
   - the Helmholtz form `f_a(q, T)` per kg of solid (preferred, because the column state
     carries the loading), with `μ_a = ∂f_a/∂q`; or
   - the grand / spreading-pressure form `Ω(μ, T)` or `Π(x, T)`, with `q = −∂Ω/∂μ`. 3B then
     obtains `f_a` by a Legendre transform. That costs one monotone scalar root per cell
     unless 3B carries the activity variable as its state, which it does.
2. **Strict monotonicity of the isotherm, exactly** (by architecture, not by penalty). It is
   equivalent to strict convexity of `f_a` in q. 3B's proof that the uptake law produces
   entropy rests on it (formulation_3B.md, Section 4.2, LDF block). One non-monotone point
   would make the "second law by construction" claim false at that point.
3. **A finite, positive Henry slope at zero loading.** The column's leading edge sits at
   near-zero loading. Without a Henry limit the chemical potential has the wrong
   singularity there.
4. **Smoothness: at least C² in (q, T) or (μ, T)**, with derivatives available by automatic
   differentiation. 3B differentiates the entropy once for the dynamics and once more for
   training.
5. **Temperature dependence consistent with an energy function.** 3B needs `e_a(q, T)` and
   the adsorbed-phase heat capacity `c_a(q, T) = ∂e_a/∂T`, and needs `c_a ≥ 0`. Otherwise the
   cell entropy is not concave in energy and temperature is not a stable function of state.
   If 3A's potential implies a temperature-dependent heat of adsorption, 3B uses it as is.
   That is the point of a consistent potential.
6. **A stated domain of validity in (q, T) or (RH, T).** 3B's cycles run from about 293 K to
   about 363 K (383 K in the out-of-distribution test) and from very dry to about 0.8 RH.
   Outside 3A's validated domain 3B will either restrict its cycles or declare the
   extrapolation.
7. **A frozen, versioned artefact.** A file plus a commit hash or DOI, with the material it
   was fitted to (MOF-303 first if 3A's data allow it).

## What 3B does not need

- Mixtures. The carrier is inert.
- Hysteresis. 3B cannot use it. A hysteretic isotherm has no single equilibrium potential
  without extra internal variables, and that is outside 3B's scope (formulation, Section 0,
  point 6).
- Uncertainty estimates. They are welcome but not used in the primary analysis.

## Decision rule (frozen in 3B's pre-registration)

At the moment 3B freezes its pre-registration:

- if a frozen 3A potential exists for MOF-303 that meets requirements 1 to 7, the 3B primary
  reference uses it;
- otherwise the primary reference uses Paper 1's dual-site isotherm with MOF-303-like
  parameters. Its spreading pressure is closed form (formulation_3B.md, Section 3.2), so it
  meets requirements 1 to 5 exactly. 3A's potential then enters 3B as a secondary arm when
  it becomes available.

Either way, 3B's code accepts any potential through one function signature:
`potential(ln_c_eq, T) -> (q, mu_a, s_a, e_a)`, batched and differentiable. Here `ln_c_eq` is
the log of the vapour concentration in equilibrium with the adsorbed phase, so it is the
isotherm's own input. For the dual-site isotherm it is `log x − ln b_H(T)`. The latent
decoder outputs this variable, so 3B never inverts an isotherm.
