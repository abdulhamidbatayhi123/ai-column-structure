# Adversarial referee report on the Paper 3B design (2026-11-03)

Reviewer stance: chemical-engineering thermodynamics, structure-preserving ML, statistics.
Documents read: `docs/formulation_3B.md` (working copy, including the uncommitted edit of
4.2), `PREREG_P3B_draft.md`, `docs/compute_3B.md`, `docs/INTERFACE_3A.md`,
`docs/lit_review_3B.md`, and Paper 1's `manuscript.tex` (Section 2), `solver_fd.py`,
`isotherm.py`. Nothing in the 3B or Paper 1 repositories was edited. The checks were run
from a scratch folder (scripts `coarse.py`, `thermo.py`, outside the repository).

Tags: **ERROR** = demonstrably wrong; **GAP** = missing or unjustified; **WEAK** = holds but
could be stronger or is overstated. "Checked, correct" is stated where an item holds.

---

## Top 5

1. **[F20, ERROR] The 1e-6 balance-violation threshold flags the reference itself.** The
   128-cell averages of the fine reference do not close the water balance when the outlet
   flux is taken on the coarse outlet cell. Measured: up to 2e-4 of v·c_in, and 1e-4 of
   throughput cumulatively. The value is the same for 512→128 and 1024→128, because it is set
   by the coarse cell width. That is about 200× the pre-registered 1e-6 threshold. Arm S is forced to satisfy a balance the truth does not
   satisfy. H3b's "drift" for a perfect model is not zero. Fix: store conservative block
   averages of (n, a, u) on an integer partition (N_z = 2048), plus the fine outlet face state.
   Then decode the outlet face as its own output (F21).
2. **[F19, ERROR] 2000/128 = 15.625, so "solved at 2000 and stored on 128 cells" has no
   conservative cell-average map.** Averaging (log c, log x, log T) or (c, x, T) and then
   mapping does not conserve N or E either, because q = Q(x) is strongly nonlinear across
   the step. Fix: N_z = 2048 (or 1920). Average (n, a, u). Derive log x = ln X(ā/A) and T
   from (n̄, ā, ū).
3. **[F11, ERROR] The 3B reference, as now specified ("integrates this Onsager form, not
   Fick's law on c"), is undefined at the initial clean bed and at any BDF undershoot.**
   L_nn ∝ logmean(c_i, c_{i+1}) multiplies R Δln c, so c = 0 (the initial condition in every
   cell) and c < 0 (a BDF undershoot, which Paper 1 clips) give ln 0 or ln of a negative
   number. Fix: choose η as the discrete-gradient mean, η = Δ(φ°/T)/Δ(1/T). Then the
   Onsager flux equals −εD_L Δc/Δz exactly for every c_w. The entropy proof still holds
   wherever c > 0, and the flux is linear and defined at c ≤ 0.
4. **[F27/F28, GAP] The arms are confounded on information and on how the metric is
   computed.** S receives physical port fluxes and decoder-Jacobian features that U lacks.
   S's working capacity is identically −ΔN̂ plus the known feed, while U's comes from its
   outlet. Two cheap controls are missing: U with the same physical features as inputs, and
   U with the same post-hoc balance projection at rollout. Without them a positive H1 cannot
   be attributed to the structure.
5. **[F31/F32, ERROR] The headline table overlaps, and the "admissibility only" row overclaims.**
   The "both REFUTED" row overlaps the "neither SUPPORTED" rows, so the precedence is
   undefined. "H1 SUPPORTED with H4 INCONCLUSIVE" falls into "Inconclusive at this power". The
   row "Structure buys admissibility only" can be issued when H1 and H2 are only
   INCONCLUSIVE, which turns underpowered tests into a negative claim. The H4 TOST also has
   no power statement. At an SD of the log ratio of 0.2 and the Holm worst-case α, its power
   at a true ratio of 1 is 0.39.

**One-line verdict:** the thermodynamics (Sections 3 and 4) holds up under re-derivation.
The upwind/Bregman argument and the c_w = 0 reduction are correct. The design is not yet
freezable, because the data pipeline (2000→128) breaks the very balances the study measures,
the reference flux is singular at its initial condition, and the arm comparison and the
headline rules have confounds and logical holes. Each has a concrete, cheap fix.

---

## A. Section 3: gas model, adsorbed phase, cell entropy

### F1. Adsorbed-phase potential μ_a, f_a, s_a, e_a: checked, correct.
Derivation. μ_v(c,T) = φ° + RT ln(c/c°). The inverse isotherm is c* = X(q)/b_H(T) with
b_H = b_H0 e^{−ΔH/RT}. So
RT ln(c*/c°) = RT ln(X/(b_H0 c°)) + ΔH, and μ_a = φ° + ΔH + RT ln(X/(b_H0 c°)) ✓.
f_a = ∫₀^q μ_a dq'. By parts, ∫₀^q ln X(q') dq' = q ln X − ∫₀^q q' X'/X dq' = q ln X − ∫₀^X Q(x)/x dx.
Closed form: ∫ f_H dx/(1+x) = f_H ln(1+x). With y = (rx)^n and dy/y = n dx/x,
∫ (1−f_H)(rx)^n/((1+(rx)^n) x) dx = ((1−f_H)/n) ln(1+(rx)^n) ✓. The q ln X term vanishes at
q → 0 because X ~ q/(q_max f_H) (Henry). X(q) has no T dependence, so
s_a = −∂f_a/∂T = −q dφ°/dT − R[q ln(X/K) − Π] with dφ°/dT = c_w − s_w° − c_w = −s_w° ✓.
e_a = f_a + T s_a = q(φ° + ΔH) + qT s_w° = q(e_w + ΔH) ✓.

### F2. Convexity of f_a, monotonicity of μ_a: checked, correct.
∂²f_a/∂q² = ∂μ_a/∂q = RT X'(q)/X(q) > 0, because Q' > 0 for x > 0 (the Henry term carries
f_H > 0). It diverges like RT/q at q → 0, which is harmless. Numerically confirmed on a
MOF-303-like isotherm (n = 4, f_H = 0.17).

### F3. Cell entropy gradient and concavity: checked, correct.
Hand derivation with s(n,a,T) and u(n,a,T):
∂s/∂T|_{n,a} = (C_s + (n+a)c_w)/T = (∂u/∂T)/T, so ∂s/∂u = 1/T.
∂s/∂n|_{a,u} = [s_w° − R ln(c/c°)] − e_w/T = −μ_v/T.
∂s/∂a|_{n,u}: in ∂s_a/∂q the terms −R q X'/X and +R Π'(X) X' = +R (q/X) X' cancel. So
∂s/∂a = s_w° − R ln(X/(b_H0 c°)) − (e_w + ΔH)/T = −μ_a/T ✓.
Concavity. At fixed T, F(n,a,T) has a diagonal Hessian in (n,a): RT/n and RT X'/(A X), both
> 0. With ∂u/∂T > 0 this gives strict concavity of s(n,a,u) (the standard Legendre argument).
Finite-difference check at 30 random states (c_w = 33.6, e_0 ≠ 0, s_0 ≠ 0): the gradient
matches to 7e-7 relative, and the scaled Hessian has all eigenvalues < 0.
Remark (WEAK): the Hessian becomes very ill-conditioned when e_0 is large, because the
direction (dn, du = e_0 dn) is nearly flat. Choose e_0 = 0, s_0 = 0 as the reference in all
code. It matters for F20.

### F4. "c_w taken equal in gas and adsorbed phase" (3.1): WEAK wording.
This is not a choice. Once μ_a is defined from μ_v through an isotherm with a
T-independent ΔH and a constant b_H0, c_a = ∂e_a/∂T / q = c_w follows (F1). Say "follows
from", and say that any c_a ≠ c_w requires changing the isotherm's T-dependence, which
3.4 row 4 already implies.

### F5. 3.4 size estimates: arithmetic checked, one range inconsistent (WEAK).
- c_w ratio: q_max c_w / C_ps with C_ps = 1000 (fixed in Paper 1, `solver_fd.py:77`):
  8·33.6/1000 = 0.27, 30·33.6/1000 = 1.01, 8·75/1000 = 0.60, 30·75/1000 = 2.25 ✓.
  "At saturation" is an upper bound. At the cycle's loading the ratio is lower; say so.
- Vapour-stream heat capacity: c_sat(303 K) ≈ 1.69 mol m⁻³, and 0.2–0.9 RH × 33.6 / 1200 gives
  0.93–4.2 % ✓.
- Kirchhoff: (75 − 33.6) × 50 / 52000 = 4.0 % ✓.
- RT/|ΔH| = 4.1–7.7 % is computed at 298–353 K, but the text states "feed 288 to 313 K". At
  288 K / 60 kJ it is 4.0 %. The pre-registration's desorption goes to 363 K (383 K OOD), and
  at 383 K / 38 kJ it is 8.4 %. Fix: state the temperature range the number uses and quote
  4.0–8.4 % over the cycles 3B actually runs.
- q_st = −ΔH + RT for the ideal-gas reading: with b defined on c, d ln p/d(1/T)|_q =
  ΔH/R − T, so q_st = −ΔH + RT ✓.

## B. Section 3.4: Paper 1 as the c_w = 0 member

### F6. Energy, convection, LDF, wall reduction: checked, correct.
With c_w = 0, e_w ≡ e_0. Then du_i/dt = C_s dT_i/dt + e_0 d(n+a)_i/dt + ΔH da_i/dt. The
water balance makes d(n+a)_i/dt = (v/Δz)(c_{i−1} − c_i) − div J_n. The energy fluxes carry
e_0 (v c and J_n, since J_u = ηJ_n + J_q with η = e_0), so every e_0 term cancels
identically, for any e_0. What remains is
C_s dT_i/dt = (v/Δz)ρ_gC_pg(T_{i−1} − T_i) + conduction + (−ΔH) da_i/dt − h_a(T_i − T_w).
That is (M3) as `solver_fd.py` integrates it, upwind with T_in and with C_s = C_term ✓.
(M1) in n = εc: dn/dt = εD_L c_zz − v c_z − A dq/dt matches the code's D_L (not D_L/ε)
convention ✓. LDF and wall are identical ✓.

### F7. Dispersion reduction: checked, correct for c_w = 0; the c_w > 0 error estimate is sloppy (WEAK).
For c_w = 0: μ_v/T = e_0/T − s_0 + R ln c. So X_n + ηX_u = −Δ(μ_v/T) + e_0Δ(1/T) = −R Δln c.
With L_nn = εD_L logmean/(RΔz), J_n = −εD_L Δc/Δz exactly ✓. Units check out
(mol² K J⁻¹ m⁻² s⁻¹).
For c_w > 0: Gibbs–Helmholtz gives Δ(φ°/T) = ∫ e_w d(1/T). The defect is
∫(e_w(T) − e_w(T_f)) d(1/T), which is O(c_w ΔT³/T³) for an arithmetic-midpoint T_f
(the first-order term cancels) and O(c_w ΔT²/T²) otherwise. The text's "O(c_w ΔT²)" is
dimensionally incomplete. Better fix: see F11, which removes the defect.

## C. Section 4: friction operator, ports, balances

### F8. Dispersion and conduction blocks: checked, correct.
M_face = Bᵀ L B with B the face difference (X = Δw). J_u = η(L_nn X_n + η L_nn X_u) + L_TT X_u
= ηJ_n + J_q. The 2×2 matrix is PSD iff L_nn, L_TT ≥ 0, for any η ✓. The face production
is XᵀLX ≥ 0 ✓. Conduction: L_TT Δ(1/T) = k_z T_iT_{i+1}/Δz · (−ΔT/(T_iT_{i+1})) ✓.

### F9. LDF block: checked, correct. The novelty claim is overstated (WEAK).
M_LDF ∇S = m(e_n − e_a)(w_n − w_a) = −m(μ_v − μ_a)/T (e_n − e_a). With m = T r/(μ_v − μ_a)
this gives dn = −r, da = +r ✓. Sign: q* > q ⇔ c > c*(q,T) ⇔ μ_v > μ_a ✓. Near equilibrium,
(q* − q)/(μ_v − μ_a) → 1/∂_qμ_a = X/(RT X') > 0 ✓. The block leaves u and n + a unchanged ✓.
WEAK: the lit review lists "concentration-difference LDF produces entropy whenever the
isotherm is increasing" as a new result. It is a one-line sign argument (the driving forces
q* − q and μ_v − μ_a share sign by monotonicity) and appears in non-equilibrium
thermodynamics treatments of sorption kinetics. Present it as an observation that answers
HARMONIA, not as a contribution.

### F10. Wall: the bath is double-booked in (G) (GAP, minor).
4.2 treats the wall as an M-block with the bath "inside" (degeneracy "all ones on u, plus the
bath"). But (G) also has a separate F_wall port, and 4.4 books −ΣW_i/T_w as a port entropy
flux. Choose one. For the latent model it must be a port, because the bath is not in the
decoded state. So drop the wall block from M, keep F_wall, and book the film production
W_i(1/T_w − 1/T_i) ≥ 0 as part of σ (as 4.4 does) ✓.

### F11. The Onsager-form reference is singular at c = 0 (ERROR).
Paper 1's initial condition is c = 0, q = 0 in every cell. BDF gives small negative c ahead
of sharp fronts (Paper 1 clips c in the isotherm for this reason). The working copy of 4.2
now says the 3B reference "integrates this Onsager form, not Fick's law on c". That form
evaluates logmean(c_i, c_{i+1}) · Δln c and μ_v/T ∝ ln c. Both are undefined at c = 0 and
complex at c < 0, so the reference cannot start.
Fix: take η_{i+½} = Δ(φ°/T)/Δ(1/T) (a discrete-gradient mean of e_w, which equals
e_w(T) when ΔT → 0 and e_0 when c_w = 0). Then X_n + ηX_u = −R Δln c exactly for every c_w,
and J_n = −εD_L Δc/Δz exactly. The flux is linear, defined for every c, identical to Paper 1
for c_w = 0, and the F8 production proof holds wherever c > 0. Evaluate the entropy only as a
diagnostic, with a stated floor, and count the cells under the floor. The same issue hits the
stored data (log c and log x at t = 0 are −∞; see F22).

### F12. Upwind = tanks in series, Bregman production: checked, correct (the key worry is resolved).
Advected variables per unit volume of flow: g = (c, e_g), with e_g = ρ_gC_pg T + c e_w(T).
The convected entropy density is s_g(g) = ρ_gC_pg ln(T/T_0) + c[s_w° − R ln(c/c°) + R]. In the
dilute model the flow work v·p is uniform and cancels in each cell, and entropy is carried
as v·s with no flow-work analogue.
Then ∂s_g/∂e_g = 1/T and ∂s_g/∂c|_{e_g} = s_w° − R ln c − e_w/T = −μ_v/T. Because the gas in
cell i is at the cell's T_i and c_i, w_i restricted to (n, u) equals ∇s_g(g_i). Hence
σ_adv,i = (v/Δz) D_{−s_g}(g_{i−1}, g_i), the Bregman divergence of −s_g ✓.
Concavity of s_g in (c, e_g): the Hessian is negative definite iff ρ_gC_pg + c c_w > 0 and
∂μ_v/∂c = RT/c > 0. Both hold, so s_g is strictly concave ✓. The interior stream terms
telescope ✓.
Caveat: the proof needs the outflow stream state to be the cell's own state, which is exactly
first-order upwind. It fails for MUSCL/van Leer (Paper 1's optional scheme), whose face
states differ from cell states. Say "first-order upwind only".

### F13. Global balances 4.4: checked, correct.
dN/dt = v(c_in − c_N) needs zero dispersive flux on both end faces; Paper 1 has that ✓.
Energy ✓. Entropy with the wall port flux at T_w and the film production in σ ✓.

### F14. The L = 0 argument: conclusion fine, two supporting claims overstated (WEAK).
(a) "∇E is constant, so the energy does not generate the advection" is not by itself an
argument. L(x)∇E with constant ∇E is any state-dependent vector field compatible with
antisymmetry. The real obstruction is degeneracy. The naive choice
L = (a e_uᵀ − e_u aᵀ)/|e_u|² with a = −v∂_z x gives L∇S = a·Σ(1/T) ≠ 0. So a valid L
needs a genuinely constructed Lie–Poisson-type bracket, which is the doc's eventual point.
State the degeneracy obstruction, not linearity.
(b) "Upwind is … neither M∇S with symmetric M nor energy-conserving reversible motion" is
overstated. The standard entropy-stable decomposition writes the upwind flux as an
entropy-conservative flux (Tadmor-type mean) plus a dissipative flux with a PSD,
mean-value-Hessian coefficient in Δw. So upwind = (skew, entropy-conserving) +
(symmetric, dissipative), up to port terms. That decomposition is exactly what Section 5.5
says the latent model needs (an antisymmetric transport part). It would give the latent
antisymmetric part a physical counterpart instead of an unstructured G_θ. Either adopt it
or say "we choose not to split" instead of "it cannot be split".
(c) Correct and worth keeping: with open boundaries and an imposed v(t), advection exchanges
E, N and S with the outside, so a port treatment is natural.

### F15. Arm C is not the strongest coarse competitor (WEAK).
Upwind on Δz adds numerical dispersion u Δz/2 (u = v/ε). With D_L = 0.7D_m + 0.5 d_p u
(Ruthven), the two are equal when Δz ≈ d_p: about 31 cells for L = 0.1 m and d_p = 3.25 mm.
At 16 cells arm C has about twice the physical dispersion. Using D_L,eff = max(0, D_L − uΔz/2)
keeps the scheme consistent (any D ≥ 0 is admissible) and makes C much more accurate. The
pre-registration promises "the strongest competitor configuration", so do this. Also add an
entropy-stable second-order scheme or note why not.

## D. Section 5: latent model

### F16. Exactness of the continuous-time guarantees: checked, correct.
P = Pᵀ and PB = 0 give ∇Ê·(PAAᵀP∇Ŝ) = 0, ∇N̂·(…) = 0, and ∇Ŝ·(…) = |AᵀP∇Ŝ|² ≥ 0.
With Π feasible, dN̂/dt = P_N, dÊ/dt = P_E and dŜ/dt − Φ_S ≥ 0 ✓. The single-inequality
QP active-set rule (equality projection; if violated, solve with all three active) is
correct for a convex QP ✓.

### F17. The projection can be ill-posed (GAP).
(i) B = [∇Ê, ∇N̂] can be rank-deficient (∇Ê ∥ ∇N̂ or a vanishing gradient), and then
(BᵀB)⁻¹ does not exist.
(ii) The inequality normal can lie in span(B). Derivation at a decoded state that is
spatially uniform: ∇S = (1/T)∇E − (μ/T)∇N in field space, so the latent gradient (by
linearity of J_Dᵀ) has P∇Ŝ = 0. There ∇Ŝ·Π = (1/T)P_E − (μ/T)P_N is fixed by the
equalities. Using w = ∇s_g(g_N), (1/T)P_E − (μ/T)P_N − Φ_S = v·D_{−s_g}(g_in, g_N) +
Σh_a(T − T_w)²/(T T_w) ≥ 0, so a uniform state is feasible. The inequality is degenerate,
though, near the feed-and-wall equilibrium. That state is reached at the end of long
adsorption steps (α up to 1.2). There both the slack and |P∇Ŝ| go to zero, and the min-norm
correction has size δ/|P∇Ŝ|.
(iii) Away from uniform states, ∇Ŝ ∈ span(B) is a codimension-(d−2) set in latent space. It
is measure-zero but has an ill-conditioned neighbourhood, and there the feasibility sign is
not guaranteed: the QP can be infeasible.
Fix: use the pseudo-inverse with a rank tolerance. Apply the inequality correction as
Π += λP∇Ŝ, λ = max(0, δ)/(|P∇Ŝ|² + κ), with κ declared. Count every step where κ binds or
B is rank-deficient, and treat those counts as part of H3a, not as successes.

### F18. Discrete-time balance definition must be shared by all arms (GAP).
S closes N̂ against ∫P_N computed with its own RK4 stages. The pre-registered violation
metric is "measured identically on every arm from the decoded fields". If it uses snapshot
trapezoids, S violates at O(h³) relative and H3a fails spuriously. If it uses RK4 stages,
U must be evaluated on its own stages too. Fix: define the metric as
|N̂_{k+1} − N̂_k − Q_RK4[P_N]| with the RK4 quadrature on each arm's own stages, and say so.
Also note that the Gauss–Newton projection is an extra decoder evaluation with Jacobians per
step, which the compute estimate omits (F36).

## E. Item 5: coarse-graining 2000 → 128 cells

### F19. Non-integer ratio (ERROR).
2000/128 = 15.625. No block-average partition exists, so "stored on 128 cells" must be an
interpolation or a fractional-overlap remap that the documents do not specify. Interpolated
point values are not cell averages and do not conserve anything. Fix: N_z = 2048 (16 × 128,
finer than the verified 2000, so grid convergence still holds) or 1920 (15 × 128).

### F20. The coarse fields do not close the balances with coarse-outlet port fluxes (ERROR).
Derivation. Let x̄_I be the conservative block averages of (n, a, u). Then exactly
dN̄/dt = v(c_in − c_{N}^{fine}), not v(c_in − c̄_{last}). The residual is
r_N = v(c̄_last − c_N^fine) ≈ v ∂_z c · (ΔZ − Δz)/2. For energy,
r_E = v[g_u(c̄_last, T̄_last) − g_u(c_N, T_N)] (plus a wall term when c_w > 0, F23). Any
reference constant e_0 adds e_0·r_N to r_E, so the energy residual is not even
reference-invariant.
Measurement (`coarse.py`, Paper 1 equations with c_w = 0, MOF-303-like isotherm: q_max 24.5,
f_H 0.17, n = 4, ΔH −52 kJ, step at RH 0.12; feed RH 0.4 at 303 K, v = 0.1 m s⁻¹,
k_LDF = 0.05 s⁻¹, adsorption to 1.15 t_st). With N_fine = 512 → 128 (ratio 4):
- the fine model closes water to 1.5e-16 (sanity check);
- the coarse water residual reaches |r_N|/(v c_in) = 2.0e-4, sustained at about 1e-4 while
  the outlet carries vapour; the cumulative residual / throughput is 9.7e-5;
- the coarse energy residual is up to 5.8 W m⁻², which is v·ρ_gC_pg times a 0.05 K mismatch
  between the outlet cell and the block average;
- with N_fine = 1024 → 128 (ratio 8) the numbers are almost the same: |r_N|/(v c_in) up to
  2.2e-4, cumulative 1.04e-4 of throughput, energy residual up to 6.5 W m⁻². The fine model
  closes to 3e-16.
So the residual is set by the coarse cell width (≈ v ∂_z c · ΔZ/2 with ΔZ = L/128), not by
the fine grid. Expect the same ≈ 2e-4 at 2048 → 128. That is about 200× the pre-registered
1e-6 threshold (this is one adsorption step; desorption steps were not run).
Consequences.
(a) The violation metric scores the truth as violating, so it cannot separate arms
honestly. A model that is more faithful to the data violates more.
(b) Arm S is forced onto a balance the targets break. The resulting bias is small for ε_W
(≈1e-3 relative), but it is a systematic, structure-induced error, and it contaminates H4.
(c) H3b: at cyclic steady state the truth itself drifts by ∫_cycle v(c̄_last − c_N)dt per
cycle, so a perfect U shows a non-zero, growing "drift" at cycle 50. The drift–error
correlation is then partly a resolution artefact.

### F21. The cleanest fix (for F19 and F20 together).
1. Solve at N_z = 2048 and store conservative block averages of (n, a, u), not of
   (log c, log x, log T). Compute log c̄ = ln(n̄/ε), log x̄ = ln X(ā/A) (one monotone root per
   stored value, offline) and T̄ = T(n̄, ā, ū). Then N and E of the stored data are exact.
2. Store the exact fine port data at every snapshot: the outlet face state (c_N^fine,
   T_N^fine) and the wall heat per block, Σ_{i∈I} h_a(T_i − T_w)Δz.
3. Give the decoder two extra scalar outputs, the outlet face state (log c_out, log T_out),
   trained against the stored fine outlet. Evaluate P_N, P_E and Φ_S on them instead of on
   coarse cell 128. The coarse balances then hold exactly for the data, and S's constraint is
   the true one. This also improves the working-capacity metric, which is an outlet quantity.
4. Gate on it: add to reference-gate item 2 that the stored coarse data close water and
   energy with the stored port data to 1e-10, and set the violation threshold relative to a
   reference-invariant scale (F33).
A cheaper alternative, if (3) is rejected: keep coarse-cell ports, but measure the
reference's own coarse residual distribution and set the violation threshold at 10× its 99.9th
percentile. Then state that S is constrained to a balance accurate only to that level. This
is worse, because S is then systematically biased.

### F22. Clean-bed initial snapshots are −∞ in log variables (GAP).
The first snapshot of every trajectory has c = q = 0, so log c and log x are −∞. Fix:
pre-wet the initial bed to equilibrium with a trace humidity (e.g. c_0 = 1e-4 c_in, stated),
or floor the stored values and declare the floor. The entropy gradient −μ_v/T ∝ −R ln c is
large in near-dry cells, which can stiffen PAAᵀP∇Ŝ. Check the RK4 step against it in the
pilot.

### F23. Coarse T and the wall flux when c_w > 0 (GAP, small).
T(n,a,u) = T_0 + (u − (n+a)e_0 − aΔH)/(C_s + (n+a)c_w) is linear in u only when c_w = 0. For
c_w > 0, T(x̄_I) ≠ mean_I(T), so the coarse wall flux differs from the fine one by a
covariance term. It is second order and small, but non-zero, so store the per-block wall
heat (F21.2) or quantify it.

### F24. Does the coarse entropy obey dS̄/dt − Φ̄_S ≥ 0? Not guaranteed. It held with margin in the test (GAP).
Counterexample (conduction only, isolated, constant C). Four fine cells
T = (350, 300 | 340, 300) form two coarse cells with means 325 and 320. The only flux across
the coarse face goes from fine cell 3 (340 K) to fine cell 2 (300 K), that is from the
cooler coarse cell into the hotter one. So dS̄/dt = J(1/325 − 1/320) < 0 with Φ_S = 0.
Coarse-graining a non-monotone sub-cell profile breaks the second law at the coarse level.
The column has such profiles: the temperature bulge at the adsorption front, and the
switches between steps.
The advective part is also not a Bregman divergence at the coarse level. With
outflow g_out ≠ g(x̄_I), σ̄_adv,I = (v/ΔZ)[D(g_in, ḡ_I) − D(g_out, ḡ_I)], which is negative
when the front sits in the downstream part of the block.
Measured, though: in the adsorption run above, the global coarse production
dS̄/dt − Φ̄_S stayed positive at every snapshot, at both ratio 4 and ratio 8 (min 0.117 W K⁻¹ m⁻²
against the fine 0.116; coarse ≥ fine throughout). The LDF, film and mixing production dominates. Because s_0 enters
as s_0·r_N, the residual is reference-dependent at the 1e-5 W K⁻¹ m⁻² per J mol⁻¹ K⁻¹ level.
Implications. For arm S the global inequality is probably compatible with the data in
practice, but that is an empirical property. Fix: add to the reference gate the computed
coarse production min_t(dS̄/dt − Φ̄_S) over every stored trajectory (both steps of every
cycle), with the exact face ports of F21. If any is negative beyond τ, report it and declare
the arm-S constraint knowingly violated by the data there. Set s_0 = e_0 = 0 in all code.

## F. Pre-registration

### F25. Unit of analysis and inference scope (WEAK).
Averaging the paired log ratio over 5 seeds and testing over 32 trajectories is valid
inference over trajectories for these 5 training runs. It does not generalise over training
randomness, which is usually the larger variance in learned dynamics. The "4 of 5 seeds"
rule is an ad hoc patch. Under no seed effect, P(≥4 of 5 in the declared direction) =
6/32 = 0.19. Fix: keep the Wilcoxon test, and make the hierarchical bootstrap interval
(already planned) a co-primary condition. Alternatively, report a two-level variance
decomposition (trajectory, seed) and state the scope in the paper.

### F26. MDE arithmetic: correct as written, but not the MDE of the verdict rule (GAP).
(z_{0.95} + z_{0.8})/√(32 · 0.955) = 2.487/5.528 = 0.450 ✓. At α = 0.05/3:
(2.128 + 0.842)/5.528 = 0.537 ✓. But SUPPORTED also requires an observed median ratio ≤ 0.8
and the seed rule. At a true ratio of exactly 0.8, the median condition alone passes only
about 50 % of the time. So the printed "MDE" overstates the power of the actual decision rule.
Fix: compute power by simulation of the full rule (Wilcoxon, Holm, median threshold, seed
rule) with the pilot SD and seed variance, and print that.

### F27. Information confound between S and U (GAP).
S is given, inside its vector field, P_E, P_N and Φ_S. These are physical functions of
decoded outlet, wall and inlet states using h_a, C_s, ΔH and the isotherm, together with
decoder Jacobian features (∇Ê, ∇N̂, ∇Ŝ). U gets z and u(t) only. A positive H1 could come
from the information (physically meaningful features), not from the constraint.
Parameter matching (±5 %) does not address this, and neither does compute: S uses 11× the
FLOPs per evaluation.
Fix: add arm **U+F**, U with the same inputs [Ê, N̂, Ŝ, P_E, P_N, Φ_S] and optionally
BᵀG-type features, with no constraint. It is cheap (the same features S computes, no
projection). "Structure pays" then needs S > U+F, not only S > U. Also report a
compute-matched U (iterations scaled to S's wall-clock) as a secondary contrast.

### F28. The working-capacity metric is structurally favourable to S (GAP).
With exact port balance, S's desorption release satisfies W = ∫v c_out dt =
∫v c_in dt − ΔN̂ exactly, so S's ε_W measures the inventory change. U's W from its outlet can
be inconsistent with its own inventory. A post-hoc balance projection of U at rollout (the
same Gauss–Newton step, no retraining: arm **U+proj**) would get this for free.
Fix: add U+proj as a secondary arm, and pre-register that H1's attribution to the
structure "in training" requires S > U+proj. Also define W explicitly as the net release
∫_des v(c_out − c_in,des) dt. The gross outflow includes the feed water, which is comparable
in size (feed water during desorption ≈ βα × bed capacity), and it would halve the apparent
error.

### F29. Training horizon versus test horizon (GAP).
Training uses cycles 1–6. H1 is read at cycle 50, near the cyclic steady state, whose states
may lie outside the training and validation distribution. The decoder gate is checked on
validation fields (cycles 1–6) only, so H1 can measure decoder extrapolation rather than
dynamics. Fix: (a) record the CSS convergence cycle (ladder item 4) and pre-declare that if
CSS is reached after cycle 6, the decoder gate is also evaluated on validation trajectories
run to cycle 60 (a few, cheap); (b) report the cycle-50 reconstruction error per trajectory
beside ε_W,50, as already promised. Also say which set is "the in-distribution test" for H2
and H4. The sets table has none; presumably cycles 1–6 of the long test.

### F30. Gates (GAP).
(a) The competence gate checks only U. If S is competent and U is not, the stated outcome
("two incompetent models") is false and suppresses exactly the effect being tested. Gate on
both arms, and report the asymmetric case as its own pre-declared outcome.
(b) The decoder gate averages nRMSE over the field. P_N depends on the outlet cell. Add an
outlet-face error and the decoded-N and decoded-E relative errors to the gate.
(c) The decoder gate is defined at full data. At H2's sizes (16, 32) the autoencoder is
retrained on fewer snapshots. Say whether d is fixed from full data and whether a failed
reconstruction at 32 voids H2.

### F31. The headline table is not a partition (ERROR).
Let A = (H1 or H2 SUPPORTED). The rows are A∧EQ/BETTER; A∧WORSE; ¬A∧EQ; ¬A∧WORSE;
"H1 and H2 REFUTED" (⊂ ¬A, so it overlaps rows 3 and 4); and "else".
(i) The precedence between row 5 and rows 3–4 is undefined.
(ii) A ∧ H4 INCONCLUSIVE gives "Inconclusive at this power", which hides a supported primary.
(iii) ¬A ∧ H4 S BETTER gives "Inconclusive", although S is more accurate.
Fix: write the table as a decision list with explicit order, and add rows for (ii) and (iii).

### F32. "Structure buys admissibility only" overclaims (ERROR).
Row 3 is issued whenever H1 and H2 are not SUPPORTED, which includes INCONCLUSIVE, that is
underpowered. "Only" is a negative claim that needs H1 and H2 REFUTED. Moreover,
admissibility of S is by construction (H3a is verification), so it is not a finding the
data "buy". Fix: "only" requires both REFUTED. Otherwise the headline is "no measurable
benefit at this power (MDE …)".

### F33. H4 TOST: no power, interval inconsistencies, unspecified test (GAP).
Power of TOST at a true ratio of 1 with margin log 1.10 = 0.0953 and n = 32 (normal
approximation with ARE 0.955):

| SD of log ratio | power, α = 0.05 | power, α = 0.0167 |
|---|---|---|
| 0.10 | 1.00 | 1.00 |
| 0.15 | 0.94 | 0.83 |
| 0.20 | 0.68 | 0.39 |
| 0.30 | 0.09 | 0.00 |

So H4 EQUIVALENT is unreachable if the pilot SD is ≥ 0.25, and that path to "Structure pays"
quietly disappears. Also: (a) TOST at α corresponds to a 1−2α interval, but S WORSE and S
BETTER use "95 % bounds", so state which interval, and whether it is Holm-adjusted. (b) Say
whether TOST is a t-test or Wilcoxon-based. (c) Pre-declare from the pilot SD whether H4 is
powered. If it is not, widen the margin before the freeze or demote H4 to secondary.

### F34. Violation metric normalisation (GAP).
"Relative to the cycle's throughput": if the energy throughput is ∫v g_u dt with absolute
T (ρ_gC_pg T ≈ 3.6e4 W m⁻² at v = 0.1), the denominator is about 30× the energy actually
exchanged, and the metric depends on the temperature origin and e_0. Normalise by a
reference-invariant scale: ∫|P_E − v ρ_gC_pg (T_in − T_ref)| dt, or the heat of the water
cycled, |ΔH| × working capacity + ∫|wall heat| dt. Water: normalise by the cycle's feed
water. τ_S is calibrated on arm S's own error (10× its 99.9th percentile), so S's zero count
is by construction. Fine for H3a, but the paper must not present U-versus-S entropy counts
as a finding about U without that caveat.

### F35. Other statistical details (WEAK).
- H3b pools seeds × trajectories (160 points, with trajectories repeated 5×). This is
  pseudo-replication, so the Spearman p-value is invalid. Fix: average over seeds per
  trajectory (n = 32), or use a cluster permutation over trajectories. It is also
  confounded by trajectory difficulty, so call it associational.
- With failures capped at 0.5, both-fail pairs give log ratio 0. Pre-declare the zero
  handling of the Wilcoxon test (Pratt or drop) and report the number of ties.
- H1 REFUTED needs median ≥ 1.0, so a precisely estimated 0.9 (interval [0.85, 0.95]) is
  "INCONCLUSIVE". Add a word ("SMALL: real but below the declared minimum") so that a
  well-measured small effect is not reported as no information.
- H5 at n = 16 has an MDE of about 0.64 SD (α = 0.05). Print it.
- The arm C headline line ("no slower and no less accurate", best of three grids) has no
  test and selects the best grid. Use a paired non-inferiority test on ε_W,50 with a
  declared margin, Holm over the three grids.
- The pilot fixes "the RK4 step" and "training iterations" with no stated criterion. That
  is a forking path. State the rules: e.g. the largest h with S's discrete entropy-violation
  count below the bound, and iterations by the 2 %-per-1000 rule already in compute §6.3.

## G. Compute estimate

### F36. Arithmetic: mostly correct, two understatements (ERROR/GAP).
Checked: 160 + 1536 + 192 + 1920 + 480 = 4288 cycles. × 25 s = 29.8 h, 7.4 h on 4 workers ✓.
Storage 128·3·200·4 B = 307 KB/cycle, 1.32 GB ✓. Per iteration: 640/2560 × 4.61 s = 1.15 s
(S) and 0.105 s (U). × 8000 iterations: 2.56 h and 0.23 h ✓. Primary block 5 × 7.93 = 39.7 h ✓.
H2 block 9 × 3.33 + 5 × 3.33 = 46.6 h ✓. Total 121.5 h ✓.
- **Pilot understated (ERROR).** "2 seeds × all arms" at the full 8000 iterations costs
  2 × (0.23 + 0.5 + 2.6 + 2.3 + 2.3) = 15.9 h of training alone, not "10 h" in total.
  Either the pilot uses fewer iterations, which conflicts with it setting the iteration
  count, or the total is about 128 h.
- **S cost omits the Gauss–Newton projection (GAP).** Each RK4 step adds at least one
  evaluation of Ê and N̂ with their decoder VJPs, kept differentiable. That is ≥ 1 extra
  gradient set per 4 stage evaluations, about +25 % on S, S−M and S−Π, roughly +10 h.
- **Reference cost per cycle may be optimistic by an order of magnitude (GAP).** The 25 s
  per cycle comes from Paper 1's breakthrough benchmark. A single timing with a MOF-303-like
  steep isotherm (n = 4) using Paper 1's own scheme and tolerances (rtol 1e-6, atol 1e-10)
  took 199 s at **512** cells for 1.15 t_st (14,693 RHS evaluations, 3,995 LU
  factorisations). The same solve at 1024 cells took 467 s. The
  cost is roughly linear in N_z, so at 2048 cells expect about 1000 s per adsorption step,
  against 25 s budgeted per cycle. That is
  illustrative only (my k_LDF and v), but the risk is real. The reduction order (compute §6)
  covers training only, and data cost is never reducible because sets are "generated once".
  Fix: time the reference with the frozen MOF-303-like isotherm before the freeze, and add a
  pre-declared data-cost contingency (e.g. a coarser verified N_z = 1024 if it passes the
  grid-convergence criterion for this isotherm, or fewer long-test cycles).
- Autoencoders "7 h" against about 4 h by the doc's own proportional rule: conservative,
  fine.

---

## H. Items checked and found correct (summary)
μ_a, f_a, s_a, e_a and the closed-form Π; convexity of f_a; the entropic gradient and the
strict concavity of s(n,a,u); the c_w = 0 reduction of the energy, convection, dispersion,
LDF and wall terms; the face Onsager blocks and their PSD condition; the LDF sign and
equilibrium limit; the upwind tanks-in-series Bregman production with a concave s_g; the
global balances; the continuous-time latent guarantees; the 3×3 active-set QP; the size
estimates of 3.4 (apart from the temperature range); the MDE arithmetic; and every
compute-table product except the pilot line.
