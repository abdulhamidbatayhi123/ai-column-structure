# Paper 3B: the thermodynamic formulation of the column

Status: design draft, 2026-10-12. Nothing here has been run. Magnitude estimates are hand
calculations from Paper 1's parameter ranges and are labelled as such.

Paper 1 means the repository `ai-mof-cof-dynamics` (manuscript Section 2, `solver_fd.py`,
`isotherm.py`). It is read here and never edited.

## 0. Summary of what this document establishes

1. **Paper 1's column is a thermodynamically consistent open system.** Read with the gas model
   it implicitly assumes (water vapour as a dilute ideal solute in a carrier of constant
   density and heat capacity), Paper 1's equations follow exactly from a fixed energy and a
   fixed entropy. They are the special case in which water carries no heat capacity
   (`c_w = 0`). The heat term `-ΔH dq/dt` is then exactly right, not approximately right.
   The `q_st = -ΔH + RT` statement in Paper 1 (line 404) belongs to a different gas model
   (ideal-gas vapour at constant total pressure). The two models differ by RT, which is 4 to
   8 % of the heat released over Paper 1's ranges. That is a modelling choice, not an internal
   error. Section 3.4 has the details.
2. **The physically important omission is the heat capacity of the water.** At saturation the
   adsorbed water holds 27 to 101 % as much heat capacity as the solid if it is given the
   vapour's molar heat capacity (33.6 J mol⁻¹ K⁻¹). With a liquid-like value (75) it holds 60
   to 225 %. Paper 1 sets this to zero. This changes the thermal wave and the temperature
   rise, so the 3B reference uses `c_w > 0`, and Paper 1's model becomes its verified
   `c_w = 0` limit (Section 6).
3. **The Poisson (reversible) operator is set to zero, by choice, and the reason is stated.**
   The only candidate for reversible dynamics in this column is advection. In the continuum,
   advection by a fixed velocity preserves entropy and is undone by reversing the velocity.
   But here the velocity is imposed, not a state variable with its own balance. The energy is
   linear in the state, so its gradient is constant, and the energy does not generate the
   advection. A Poisson operator could reproduce advection only as a state-dependent
   construction (the Lie-Poisson route of arXiv:2610.05570) with no physical Hamiltonian
   behind it. The design therefore treats convection as a *port-driven exchange* between
   neighbouring open cells (Section 4.1), and the internal structure is `L = 0` plus a
   friction operator. The honest name is a *port-metriplectic system with an empty
   Poisson part*, not "GENERIC" without qualification. Section 4.1 has the argument and
   the alternative.
4. **Every dissipative mechanism in the column has a proof of non-negative entropy
   production**, including first-order upwind advection, which is exactly a tanks-in-series
   mixing process. So the full-order discrete reference is consistent at *any* grid size.
   This has a consequence for the paper: a coarse, consistent finite-volume solver is a
   non-learned competitor that the latent model must beat on cost and accuracy
   (pre-registration, arm C).
5. **In a latent model the guarantee can be kept on the physical E and S,** by evaluating them
   on the decoded fields. The guarantee is then (i) exact balances of total water and total
   energy against the port fluxes, and (ii) non-negative *total* entropy production. It is
   global, not cell-by-cell, and it holds in continuous time. The integrator must be built to
   carry it into discrete time (Section 5.4).
6. **What does not fit, stated plainly:** hysteretic isotherms (no single adsorbed-phase
   potential exists without extra internal variables), local (cell-wise) second-law
   guarantees in a latent model, and any claim that the structure helps front representation
   (it does not touch it; Paper 1 L5 and arXiv:2210.01074).

## 1. The reference equations (Paper 1, Section 2)

With `c` the vapour concentration in the gas (mol m⁻³ of void), `q` the loading (mol kg⁻¹),
`T` the temperature, `v` the superficial velocity, `ε` the bed void fraction and `ρ_p` the
pellet density, Paper 1 integrates

```
∂c/∂t = D_L ∂²c/∂z² − (v/ε) ∂c/∂z − ((1−ε)/ε) ρ_p ∂q/∂t                       (M1)
∂q/∂t = k (q*(c,T) − q)                                                       (M2)
C_b ∂T/∂t = k_z ∂²T/∂z² − v ρ_g C_pg ∂T/∂z + (1−ε) ρ_p (−ΔH) ∂q/∂t
            − (4 h_w / D_in)(T − T_w)                                         (M3)
C_b = ε ρ_g C_pg + (1−ε) ρ_p C_ps
```

with Danckwerts flux conditions at the inlet (no dispersive or conductive flux on the feed
side), zero gradient at the outlet, a clean bed at the wall temperature, and the dual-site
isotherm

```
q*(c,T) = q_max [ f_H x/(1+x) + (1−f_H) (r x)^n / (1 + (r x)^n) ],   x = b_H(T) c,
b_H(T) = b_H0 exp(−ΔH/RT),   r = b_C/b_H  (fixed per material; both affinities share ΔH).
```

The fact that q* depends on c and T only through the single variable `x = b_H(T) c` is what
makes everything below explicit. Write `q* = Q(x)`. Q is strictly increasing because every
term is increasing in x.

The finite-volume scheme (`solver_fd.py`): N cells, first-order upwind advection, central
dispersion and conduction, zero dispersive and conductive flux on the inlet face, zero
gradient at the outlet, BDF in time. Mass closure telescopes to machine precision (docstring
of `laplacian`).

## 2. State variables

Per finite-volume cell i of width Δz, per unit bed volume:

| symbol | meaning | relation to Paper 1 |
|---|---|---|
| `n_i` | vapour moles in the gas | `n = ε c` |
| `a_i` | adsorbed moles | `a = (1−ε) ρ_p q` |
| `u_i` | total internal energy (carrier, vapour, solid, adsorbed water) | new; T follows from it |

The energy-type variable is `u`, not T, because that is what makes the energy functional
linear and the degeneracy conditions simple (Section 4). In the latent model the decoder
outputs (log c, log x, log T) instead, for positivity. The map to (n, a, u) is explicit
(Section 3.3).

Assumptions carried by the state, all inherited from Paper 1 and stated rather than hidden:

- local thermal equilibrium between gas and solid in each cell (one T);
- no pressure drop and no momentum balance, so the velocity is an input;
- the carrier is not tracked as a species. It is a medium with constant density and heat
  capacity (Paper 1 holds ρ_g constant even as T changes);
- the gas inside pellet pores is not a separate phase;
- no hysteresis: one equilibrium surface q*(c,T).

## 3. Energy and entropy, fixed from known thermodynamics

### 3.1 Gas model

The vapour is treated as a dilute ideal solute in an incompressible carrier, the reading
under which Paper 1's constant ρ_g is exact. Per mole of water, the internal energy equals
the enthalpy (no flow-work term), and the molar heat capacity is `c_w` (taken equal in gas
and adsorbed phase, see 3.4):

```
e_w(T) = e_0 + c_w (T − T_0)
μ_v(c,T) = φ°(T) + R T ln(c/c°),     φ°(T) = e_w(T) − T s_w°(T),   s_w°(T) = s_0 + c_w ln(T/T_0)
```

Carrier and solid are constant-heat-capacity media. Their entropy is `C ln(T/T_0)` per unit
of heat capacity C.

### 3.2 The adsorbed phase from the isotherm

Equilibrium requires the adsorbed-phase chemical potential to equal the vapour's at the
equilibrium concentration: `μ_a(q,T) = μ_v(c*(q,T), T)`, with `c*` the inverse isotherm. For
the dual-site isotherm, `c* = X(q)/b_H(T)` with `X = Q⁻¹`. So

```
μ_a(q,T) = φ°(T) + ΔH + R T ln( X(q) / (b_H0 c°) ).
```

Integrating over q gives the Helmholtz energy of the adsorbed phase per kg of solid. By
parts, the only integral needed is `∫₀ˣ Q(x')/x' dx'`, which is the reduced spreading
pressure. For this isotherm it has a closed form:

```
Π(x) = q_max [ f_H ln(1+x) + ((1−f_H)/n) ln(1 + (r x)^n) ]
f_a(q,T) = q (φ°(T) + ΔH) + R T [ q ln( X(q)/(b_H0 c°) ) − Π(X(q)) ]
s_a(q,T) = −∂f_a/∂T = q s_w°(T) − R [ q ln( X(q)/(b_H0 c°) ) − Π(X(q)) ]
e_a(q,T) = f_a + T s_a = q ( e_w(T) + ΔH )
```

Three properties follow, and each is needed later:

- `∂²f_a/∂q² = R T X'(q)/X(q) > 0`. The adsorbed-phase free energy is strictly convex in
  loading *because* the isotherm is strictly increasing. This is the thermodynamic stability
  that makes the LDF entropy production non-negative (Section 4.3).
- The molar internal energy of adsorbed water is `e_w(T) + ΔH`, independent of loading. The
  heat released per mole adsorbed at fixed total energy is exactly `−ΔH`.
- `X(q)` is the only implicit object. It is a scalar monotone root per cell (a few Newton
  steps). It disappears entirely if the state is carried as `log x` instead of q, which is
  what the latent decoder does.

### 3.3 The cell's energy, entropy and temperature

With `C_s = ε ρ_g C_pg + (1−ε) ρ_p C_ps` (Paper 1's C_b):

```
u(n,a,T) = C_s (T − T_0) + (n + a) e_w(T) + a ΔH
T(n,a,u) = T_0 + [u − (n+a) e_0 − a ΔH] / [C_s + (n+a) c_w]                  (explicit)
s(n,a,u) = C_s ln(T/T_0) + n [ s_w°(T) − R ln(n/(ε c°)) + R ] + (1−ε) ρ_p s_a(a/((1−ε)ρ_p), T)
```

The entropic gradient, which is all the dynamics needs:

```
∂s/∂u = 1/T,     ∂s/∂n = −μ_v/T,     ∂s/∂a = −μ_a/T.
```

Global functionals, summed over cells (Δz times the cross-section A, dropped below):

```
E = Σ_i u_i Δz       (linear in the state)
N = Σ_i (n_i + a_i) Δz   (total water, linear)
S = Σ_i s(n_i, a_i, u_i) Δz   (strictly concave where C_s + (n+a)c_w > 0)
```

### 3.4 Where Paper 1 sits in this family, and the size of each difference

Set `c_w = 0`. Then the energy balance derived from Section 4 reduces term by term to (M3).
The `e_0` terms cancel through water conservation, and the source term is `(−ΔH) ∂a/∂t`.
Dispersion (M1) is recovered exactly, including its discrete form (Section 4.2). So **Paper 1
is the `c_w = 0` member of a consistent family**, and the 3B reference generalises it by one
parameter. Differences from physical reality, with hand-calculated sizes over Paper 1's
ranges (q_max 8 to 30 mol kg⁻¹, ΔH −60 to −38 kJ mol⁻¹, feed 288 to 313 K):

| difference | size | treatment in 3B |
|---|---|---|
| water heat capacity omitted (`c_w = 0`) | adsorbed water at saturation holds 0.27 to 1.01 × the solid's heat capacity at c_w = 33.6 J mol⁻¹ K⁻¹, and 0.60 to 2.25 × at a liquid-like 75 | **included**: primary reference uses c_w = 33.6 (keeps Paper 1's isotherm exact); c_w = 0 is the verification limit |
| flow work of the vapour (ideal gas versus dilute solute) | RT/\|ΔH\| = 4.1 to 7.7 % of the heat released (298 to 353 K) | not included; the dilute-solute model is kept for continuity with Paper 1 and stated as a limitation |
| enthalpy carried by vapour in the gas stream | vapour/carrier stream heat capacity 0.9 to 4.3 % (RH 0.2 to 0.9 at 303 K) | included automatically once c_w > 0 |
| liquid-like heat capacity of the adsorbed phase (c_a ≠ c_v) | would make ΔH temperature-dependent (Kirchhoff), about 4 % over a 50 K swing | not in the primary reference, because it changes the isotherm's T-dependence; it belongs to 3A's potential (INTERFACE_3A.md) |

The point that the adsorbed-phase heat capacity and the isosteric heat must come from one
consistent thermodynamic path is not new. Walton and LeVan (Adsorption 2005,
doi:10.1007/s10450-005-5984-6) derive fixed-bed energy balances along consistent paths.
Schwamberger and Schmidt (Ind. Eng. Chem. Res. 2013, doi:10.1021/ie4011832) show that
inconsistent heat-capacity choices leave energy and entropy balance residuals. 3B cites them
and claims only the derivation from one potential and its exact reduction to Paper 1.

None of this affects Paper 1's conclusions, which concern surrogates of a defined PDE. The
adsorbed-water heat capacity is worth one sentence in Paper 1's limitations at revision. That
is the author's call, and this session does not touch Paper 1.

## 4. Dynamics in metriplectic form, with ports

The full-order discrete system is

```
dx/dt = L ∇E + M(x) ∇S + F_flow(x; v, c_in, T_in) + F_wall(x; T_w)          (G)
with L = 0,  M = Mᵀ ⪰ 0,  M ∇E = 0,  M ∇N = 0.
```

### 4.1 Why L = 0, and where convection goes instead

A Poisson operator generates reversible motion. It conserves E and S, and it is driven by
the energy gradient: `L ∇E`. In a GENERIC fluid model the reversible part is the transport
of the fluid by its own momentum, which is a state variable, and the energy contains the
kinetic energy that drives it.

In this column the velocity is prescribed and has no balance equation. Continuum advection
by a fixed velocity is still entropy-preserving in the interior, and it is reversed by
reversing v. In that sense it is "reversible". But it is not generated by the energy:
E = Σ u_i Δz is linear in the state, so ∇E is a constant vector. Writing advection as
`L(x) ∇E` would need a state-dependent L built to reproduce the upwind fluxes, with no
physical Hamiltonian behind it. The Poisson-GENERIC operators of arXiv:2610.05570 take this
route for transport (a Lie-Poisson L), and they note that their friction class cannot
represent transport. 3B does not take it, for two reasons:

1. **Upwind convection is not reversible at the discrete level.** It is exactly a cascade of
   perfectly mixed tanks: each cell receives a stream at its upstream neighbour's state and
   releases a stream at its own state. The mixing produces entropy (Section 4.3). That is a
   *directed* exchange, so it is neither `M ∇S` with a symmetric M nor energy-conserving
   reversible motion.
2. **The flow is an input.** In a temperature-swing cycle the operator chooses v, the feed
   and the wall temperature. Treating the flow as a port keeps every input on the boundary
   of the model, where a controller would act.

So convection enters as a port interconnection between open subsystems. That is the setting
of port-metriplectic models (Hernández et al., arXiv:2211.01873), of boundary-controlled
irreversible port-Hamiltonian systems (Ramirez, Le Gorrec, Maschke, arXiv:2104.13459), and
of boundary ports in general. The closest continuum template is arXiv:2603.11260 (Ouardi,
Sarkar, Ramirez, Le Gorrec 2026). It brings convective transport *inside* an irreversible
port-Hamiltonian structure for 1-D viscous fluids, through modified differential operators
and an extended boundary-port parametrisation that satisfies both laws. 3B's discrete
tanks-in-series treatment is the simpler finite-volume counterpart for an imposed velocity.
The paper must cite both and say how they differ. The reversible half would become physically meaningful only if the model
carried the gas momentum (Ergun pressure drop, pressure waves), which Paper 1 does not. This
is not a defect of the approach. It decides what the structure can be credited with:
dissipation, balances and ports, not reversible dynamics.

### 4.2 The friction operator M, mechanism by mechanism

Each mechanism contributes a symmetric positive semi-definite block. In each block the
fluxes are driven by differences of the entropic variables `w = ∇s = (1/T, −μ_v/T, −μ_a/T)`.

**Dispersion of vapour plus the enthalpy it carries (face i+½).** Forces
`X_u = Δ(1/T)`, `X_n = −Δ(μ_v/T)`. Onsager matrix on the face:

```
[J_n]   [ L_nn       η L_nn          ] [X_n]
[J_u] = [ η L_nn     L_TT + η² L_nn  ] [X_u]      η = e_w(T_face),  L ⪰ 0 iff L_nn, L_TT ≥ 0.
```

With `L_nn = (ε D_L / (R Δz)) · logmean(c_i, c_{i+1})`, the vapour flux is
`J_n = −ε D_L (c_{i+1} − c_i)/Δz` *exactly* when T is uniform, which is Paper 1's discrete
Fick term. With non-uniform T it differs only through `e_w(T)/T` at second order in Δz, and
with `c_w = 0` it is exact everywhere. The 3B reference integrates this Onsager form, not Fick's law on c, so the entropy proof applies to the reference exactly. With `c_w = 0` the two coincide, which is why the Paper 1 limit (Section 6) still reproduces `solver_fd.py`. This works because
`Δ(μ_v/T) − h Δ(1/T) = R Δ ln c + O(c_w ΔT²)`. The logarithmic mean is the discrete
chain-rule factor that makes Fick's law an exact gradient flow of the ideal-solute entropy
(the same construction as entropy-stable Scharfetter-Gummel fluxes).

**Conduction.** `L_TT = k_z T_i T_{i+1} / Δz` gives `J_q = −k_z (T_{i+1} − T_i)/Δz` exactly,
since `Δ(1/T) = −ΔT/(T_i T_{i+1})`.

**LDF uptake (within cell i).** The flux `r = da/dt = −dn/dt` with force
`(μ_v − μ_a)/T`:

```
M_LDF,i = m_i (e_n − e_a)(e_n − e_a)ᵀ,
m_i = T_i r_i / (μ_v − μ_a) = T_i (1−ε) ρ_p k (q*(c,T) − q) / (μ_v(c,T) − μ_a(q,T)).
```

`m_i ≥ 0` because `q* > q ⇔ c > c*(q,T) ⇔ μ_v(c) > μ_a(q)`: the isotherm is increasing and
μ_v is increasing in c. At equilibrium the ratio tends to the finite positive limit
`T (1−ε) ρ_p k ∂q*/∂μ`. So **the LDF law is an Onsager flux with a positive, state-dependent
coefficient, by virtue of the isotherm's monotonicity alone.** Non-equilibrium thermodynamics
already writes uptake as a flux driven by the difference in chemical potential (Santamaria-Holek,
Grzywna, Rubi, arXiv:1204.3841). What is shown here is that the *concentration-difference*
LDF used in practice is of that form whenever the isotherm is increasing. That answers the
claim of a 2025 preprint (HARMONIA, Dytrych and Dlaskova, ChemRxiv,
doi:10.26434/chemrxiv-2025-c860f, read only as an abstract record) that standard
isotherm-plus-LDF couplings can violate the second law. For a single adsorbate with an
increasing isotherm they cannot. Violations need a non-monotone isotherm, an isotherm-heat
mismatch, or multicomponent couplings, none of which 3B has. It does not touch u, so the
heat of adsorption appears through T(n,a,u), not as a source term.

**Wall.** Treat the wall as a heat bath at T_w with energy U_w and entropy U_w/T_w. Then
`W_i = h_a (T_i − T_w)` with `h_a = 4h_w/D_in` is the Onsager flux
`L_w,i (1/T_w − 1/T_i)` with `L_w,i = h_a T_i T_w ≥ 0`. Production
`h_a (T_i − T_w)² / (T_i T_w) ≥ 0`. When T_w(t) is switched in a temperature-swing cycle,
the bath is an input port.

**Degeneracy.** Every block moves water between n and a, or between neighbouring cells, and
moves energy between neighbouring cells or into the bath. So each block's columns sum to
zero against `∇E` (all ones on u, plus the bath) and against `∇N` (all ones on n and a).
That gives `M ∇E = 0` and `M ∇N = 0`. With L = 0 the other GENERIC condition, `L ∇S = 0`,
holds trivially.

### 4.3 Convection and the inlet and outlet ports

For cell i the flow term is `F_flow,i = (v/Δz) (g_{i−1} − g_i)`, with `g = (c, ρ_g C_pg T + c e_w(T))`
the gas stream's water and energy per unit volume of flow, and `g_0` the feed. Paper 1's
convective terms are exactly this. Its entropy production is

```
σ_adv,i = (v/Δz) [ w_i · (g_{i−1} − g_i) − ( s_g(g_{i−1}) − s_g(g_i) ) ] ≥ 0
```

by concavity of the gas-stream entropy `s_g`. It is the entropy of mixing an incoming parcel
into a well-mixed tank, a Bregman divergence. Summed over cells, the interior stream
entropies telescope, leaving the port terms.

### 4.4 The balances the structure guarantees

```
dN/dt = v (c_in − c_N)                                                        (water)
dE/dt = v [ g_u(in) − g_u(N) ] − Σ_i W_i Δz                                   (energy)
dS/dt = v [ s_g(in) − s_g(N) ] − Σ_i W_i/T_w Δz + Σ_i σ_i Δz,   σ_i ≥ 0         (entropy)
σ_i = σ_disp + σ_cond + σ_LDF + σ_adv + σ_wall-film
```

The first two are exact identities of (G). The third is an inequality once the port entropy
fluxes are moved to the left. These three lines are what "zero unphysical outputs" can mean
precisely. Positivity of c, q and T is a separate property: the BDF reference does not
guarantee it (Paper 1 clips c in the isotherm), and in the latent model it is a decoder
property (Section 5.2).

## 5. The latent model

### 5.1 What is learned and what is fixed

Fixed, never learned: s(n,a,u), u, the isotherm or 3A's potential, the port flux formulas,
and the balance functionals E, N, S. Learned: an autoencoder (shared by every arm, trained
once per seed and frozen) and the latent dynamics.

The full-order system (G) is already known exactly. The reason to learn anything is cost:
the verified reference needs N_z = 2000 cells and about 17 worker-seconds per breakthrough
in Paper 1, and a 50-cycle rollout costs minutes to hours. A latent model of dimension
d ≈ 16 to 32 can roll out in milliseconds. 3B asks what the structure buys *for such a
model*, not whether physics can be learned from scratch.

### 5.2 Decoder

`D: z ∈ ℝ^d → (log c, log x, log T)` on a 128-cell grid. This gives `c > 0`,
`q = Q(x) ∈ (0, q_max)` and `T > 0` by construction, and no inverse-isotherm root is ever
needed. For a general potential (3A), `log x` is replaced by `ln c_eq`, the log of the
vapour concentration in equilibrium with the adsorbed phase (INTERFACE_3A.md). E, N and S of a latent state are defined as the physical functionals of its decoded
fields: `Ê(z) = E(D(z))`, and likewise N̂ and Ŝ. Their latent gradients are vector-Jacobian
products through the decoder.

### 5.3 Structured latent dynamics (arm S)

```
dz/dt = P(z) A(z) A(z)ᵀ P(z) ∇Ŝ(z)  +  Π_z[ G_θ(z, u(t)) ]
P(z) = I − B (BᵀB)⁻¹ Bᵀ,   B = [∇Ê, ∇N̂]                  (projector, metriplectic part)
Π_z[G]: the minimum-norm change of G such that
    ∇Ê · Π = P_E(z,u),   ∇N̂ · Π = P_N(z,u),   ∇Ŝ · Π ≥ Φ_S(z,u)
```

Here `P_E`, `P_N` and `Φ_S` are the physical port fluxes of Section 4.4. They are evaluated
on the decoded inlet and outlet cells and on the decoded temperature field for the wall,
with the known feed u(t). The projection is closed form: two equalities plus one inequality
with an active set, solved as a 3 × 3 system. It is differentiable almost everywhere.
Consequences, exact in continuous time:

```
dN̂/dt = P_N,    dÊ/dt = P_E,    dŜ/dt − Φ_S = (Pᵀ∇Ŝ)ᵀ A Aᵀ (Pᵀ∇Ŝ) + (∇Ŝ·Π − Φ_S) ≥ 0.
```

The internal metriplectic term and the port projection are separable, so they can be
ablated separately (pre-registration, arms S-M and S-Π). That decomposition is how the paper
can say *which part* of the structure buys what.

### 5.4 Time integration

The guarantees above are continuous-time. A fixed-step explicit integrator breaks the
equalities at O(h^{p+1}) per step. The primary integrator is RK4 followed by a Gauss-Newton
projection of z back onto the two affine-in-time constraints `N̂ = N̂_prev + ∫P_N dt` and
`Ê = Ê_prev + ∫P_E dt`. The integrals are taken with the same RK4 stages. This restores the
balances to solver tolerance. The entropy inequality is checked per step and *counted*, not
enforced. It can fail by O(h^5) when production is near zero, and the count is reported with
its tolerance. A discrete-gradient integrator (exact for all three) is the documented
alternative if the count is not negligible. It is implicit and roughly 3 to 5 times more
expensive per step.

### 5.5 What the latent guarantee does not cover

- **It is global.** The decoded fields can have a cell with negative local production while
  the total stays non-negative. The paper reports the decoded local production as a
  diagnostic, and claims only the global law.
- **It does not bound the latent state.** z can drift in directions that leave Ê, N̂ and Ŝ
  unchanged. Whether long rollouts stay stable is therefore an empirical question, which is
  why it is the first pre-registered test and not a theorem.
- **The latent model is not a pure gradient flow, and must not be.** The physics has L = 0,
  but in latent coordinates a travelling front is a translation, which a symmetric
  dissipative term cannot produce. A latent model built from a gradient flow alone loses
  rollout accuracy against one with an antisymmetric part (arXiv:2609.04530, magnetisation
  dynamics). In arm S the transport is carried by the projected port term `Π[G_θ]`, which is
  free in every direction the balances allow. The metriplectic term only adds relaxation.
- **It does not represent fronts.** A moving, self-sharpening front has a slowly decaying
  Kolmogorov n-width. A fixed decoder represents it only as well as d allows, and the
  structure changes nothing about that. Both arms share the same decoder, so the comparison
  stays fair, but the absolute accuracy is set by the autoencoder. The pre-registration
  gates on it.

## 6. Verification ladder for the new reference (before any data are used)

1. **Paper 1 limit.** With `c_w = 0` the new u-form solver reproduces `solver_fd.py` at
   N_z = 2000 on Paper 1's verification cases (`verify_solver.py`) to within the solver
   tolerance. This is the strongest single test, because Paper 1's solver is already
   verified.
2. **Balances.** Water and energy closure over a full cycle to machine precision
   (telescoping), and the entropy inequality of Section 4.4 per step with
   `min σ_i ≥ −tolerance`.
3. **Grid convergence.** The same `grid_convergence()` criterion as Paper 1, for c, q and T
   over cycle 1 and cycle 50.
4. **Cyclic steady state.** Convergence of per-cycle water uptake to a fixed point, with the
   convergence cycle recorded per trajectory.

## 7. Open points this document does not settle

- **The gas model (author's decision).** The dilute-solute vapour (this document, Paper 1's
  limit) or the ideal-gas vapour that Paper 3A uses. They differ by RT in the heat released,
  4 to 8 %. The recommendation is to keep the dilute solute and state the offset. See
  INTERFACE_3A.md, last section.
- **With a 3A potential in `t_mode="kirchhoff"`,** the adsorbed-phase heat capacity follows
  from the potential and differs from the vapour's. Section 3.3 then generalises: e_a(q,T)
  and s_a(q,T) come from the Legendre transform of 3A's psi at fixed q, and T(n,a,u) is no
  longer explicit (one scalar Newton step per cell). Concavity of the cell entropy needs
  `c_a >= 0` over the cycle domain.

- The exact cycle (adsorption and desorption durations, desorption by heated feed or by
  wall) is fixed in the pre-registration, not here.
- Whether `c_w` should be the vapour value (33.6) or a value 3A's potential implies (with a
  temperature-dependent ΔH) depends on 3A. See INTERFACE_3A.md.
- Prior structured column models exist only for the **isothermal** case: the port-based
  (bond-graph, Dirac-structure) adsorption column of Baaiu, Couenne, Le Gorrec, Lefèvre and
  Tayakout (Math. Comput. Model. Dyn. Syst. 2008, doi:10.1080/13873950701844840), built "under
  isothermal conditions" on irreversible thermodynamics. Its structure-preserving reduction
  and discretisation followed (J. Process Control 2009, doi:10.1016/j.jprocont.2008.07.002;
  IEEE CDC 2007, doi:10.1109/cdc.2007.4434663). Their full texts were not reachable
  (paywalled). Before the paper is written, the author should read them through library
  access, to state exactly how their convection and boundary treatment compare with
  Section 4. 3B's formulation is the non-isothermal, entropy-explicit extension, and Section 4
  must say so in those words.
