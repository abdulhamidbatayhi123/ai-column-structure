# SDD ledger — plan: docs/superpowers/plans/2026-11-08-p3b-latent-arms.md
Spec: PREREG_P3B_draft.md (branch plan1-reference-data at ccb5b67), docs/formulation_3B.md
Pre-flight: Task 1 physics feeds Tasks 3-7 (E,N,S,ports signatures); Task 4 integrator feeds 6-7; matcher (5) feeds 8. No conflicts found.
Ruling: float64 everywhere — PREREG rank tolerance 1e-8 is below float32 eps — cost if wrong: ~2x slower training.
Task 1: complete (commits ccb5b67..09c72e3, tests: pytest -q tests/test_latent_physics.py → 10 passed)
Task 2: complete (commits 09c72e3..67b2da9, tests: pytest -q tests/test_latent_data.py → 7 passed)
Task 3: Ruling: field nRMSE on the stored channels (A: log c, log x, log T), RMSE/range per channel per trajectory-cycle, averaged (Paper 1 per_variable_nrmse applied as PREREG says "over all stored channels") — cost if wrong: log-space weighting of small concentrations; changeable before the freeze.
Task 3: Ruling: balance gates use the max over all validation snapshots (literal "must meet") — cost if wrong: one outlier snapshot fails the gate.
Task 3: complete (commits 67b2da9..de1c698, tests: pytest -q tests/test_autoencoder.py tests/test_latent_data.py → 17 passed after one test-bound fix)
Task 4: Ruling: violation() (normalised balance residuals) moves to Plan 3 metrics; Task 4 provides the pieces (stage_quadrature, balance_project) — cost if wrong: none, the metric needs the normalisations Plan 3 defines.
Task 4: complete (commits de1c698..556b92d, tests: pytest -q tests/test_integrate.py → 7 passed)
Task 5: Ruling: shared LatentFunctionals helper (p3b/latent/functionals.py) for U+F, P and S — one implementation of the decoded functionals and their latent gradients — cost if wrong: none.
Task 5: Ruling: P penalty terms each divided by a caller-supplied reference scale (the violation-metric normalisations), one weight on their sum (PREREG fixes one weight) — cost if wrong: relative weighting of the three terms differs from an unnormalised sum.
Task 5: complete (commits 556b92d..f045196, tests: pytest -q tests/test_arms_unstructured.py → 7 passed)
Task 6: Ruling: projection algebra exposed as structure_terms() so the degenerate cases (grad S in span(B), parallel columns of B) are tested directly — cost if wrong: none.
Task 6: complete (commits f045196..b315817, tests: pytest -q tests/test_arm_s.py → 8 passed)
Task 7: Ruling: balance_project iterates to 1e-12 relative (max 20) instead of a fixed 3 — 3 left residuals of 1e3 J/m2 after a large step of an untrained U (found by test_projected_step_restores_both_balances) — cost if wrong: more decoder evaluations per step on hard steps.
Task 7: complete (commits b315817..27b0645, tests: pytest -q tests/test_arm_sl_proj.py tests/test_arm_s.py tests/test_integrate.py → 18 passed)
Task 8: Ruling: FNO spectral weights stored as real (...,2) tensors so the S parameter match counts real numbers; log(dt) is an input (snapshot spacing differs by half-cycle and trajectory); residual update x + net(x) — cost if wrong: small architecture difference from a textbook FNO, stated in the paper.
Task 8: complete (commits 27b0645..eafd1c6, tests: pytest -q tests/test_fno.py → 4 passed)
Task 9: Ruling: C reuses the reference simulator and Coarsener (injection to the target cells), so C's files have the reference format; at N_c = target it equals the reference exactly (test) — cost if wrong: none.
Task 9: Ruling: MUSCL-C uses physical (uncompensated) dispersion — a second-order scheme has no leading numerical diffusion; first interior face plain upwind (no second upstream neighbour); inlet face = feed, outlet face = last cell so ports are unchanged; implemented for both systems — cost if wrong: descriptive arm only.
Task 9: Ruling: simulators take column_cls / store_factory and a Jacobian band from the model (default unchanged) — needed for MUSCL's 5-point stencil — cost if wrong: none; Plan 1 tests rerun.
Task 9: complete (commits eafd1c6..63256ff, tests: pytest -q tests/test_solver_arms.py → 9 passed)
Task 10: Ruling: system B multipliers on k_f,eff (per face) and h_v (per cell), reading PREREG's "(system B: the exchange coefficient)" as replacing the LDF coefficient while k_f plays k_z's role — cost if wrong: one extra multiplier in B.
Task 10: Ruling: multipliers start at exactly 1 (last layer weight 0, bias ln(e-1)), so C+L starts as C; local inputs A (log c, q/q_cap, T/300), B (T_f/300, T_s/300, T_f - T_s) — cost if wrong: initialisation only.
Task 10: Ruling: implicit-Euler gradient by one Newton correction with a detached Jacobian at the converged point (exact IFT gradient; finite-difference test to 1e-5) — cost if wrong: none.
Task 10: complete (commits eafd1c6..63256ff, tests: pytest -q tests/test_c_plus_l.py → 9 passed)
PREREG follow-ups (plan1 4819161, cs T5): S-L port term must become S own Pi (F3); C scored after conservative minmod reconstruction onto 128 cells, with injection and own-grid as sensitivity (F2 i). Both to be implemented in the plan 2 fix pass.
Final review: opus reviewer, verdict "With fixes" (1 Critical, 8 Important, 9 Minor).
Final: fixed C1 rank test and projector depend on units — B columns / Jacobian rows scaled to unit norm before svdvals and pinv; test_rank_test_is_unit_free, test_real_column_b_is_well_conditioned_after_scaling, test_balance_project_is_unit_free RED->GREEN
Final: fixed I2 projected step loses or breaks gradients — balance_project iterates without graph, then one differentiable correction with the tangent term at z*; test_projection_gradient_is_exact_for_linear_constraints, test_s_training_gradient_through_projected_steps_matches_finite_differences RED->GREEN
Final: Ruling: the projection gradient is first order in the distance projected (exact for linear constraints; S's step leaves the set by O(h^5), measured 32x per halving, so S's gradient matches FD to 0.5 %); S-Pi, whose field ignores the balances, gets a gradient accurate to first order in that distance (10 % on an untrained U at h 0.5) — cost if wrong: slightly biased S-Pi gradients; exact alternative needs constraint Hessians.
Final: fixed I3 LatentFunctionals detaches by default — create_graph follows torch.is_grad_enabled(); autoencoder frozen inside; test_functionals_stay_attached_when_grad_is_enabled, test_decoder_weights_are_not_trained_through_the_functionals RED->GREEN
Final: fixed I4 C+L gradient test passed trivially — random last layers; test_c_plus_l_gradient_is_nonzero_and_matches_finite_differences RED->GREEN
Final: fixed I5 Newton fails silently — per-block tolerance against max(|y|, |h f|), RuntimeError on non-convergence; test_newton_failure_raises RED->GREEN (the rollout test found the round-off floor of the first per-block scale)
Final: fixed I6 C+L too slow, time error — 9-colour forward-mode Jacobian (exact against dense), K substeps with first-order convergence (PREREG: K set in the pilot); test_colored_jacobian_equals_the_dense_jacobian, test_substeps_converge_to_the_continuous_solution_at_first_order RED->GREEN
Final: Ruling: no vmap batching of C+L windows yet — the coloured Jacobian removes the dominant cost; batching is a Plan 3 performance item — cost if wrong: slower C+L training.
Final: fixed I7 match_width OOM on FNO — candidates built on the meta device; test_match_width_builds_on_the_meta_device RED->GREEN
Final: fixed I8 structured arms do not plug into the integrator — structured_step keeps stage and projection flags ("any stage" counts); test_structured_step_returns_the_stage_flags RED->GREEN
Final: fixed I9 balance_project convergence — physical scale argument, per-sample converged flag; test_balance_project_flags_per_sample_convergence, test_balance_project_uses_the_given_scale RED->GREEN
Final: Ruling (reviewer disagreed with Task 3): field nRMSE now on physical fields (c, q, T; B: T_f, T_s), gate per field (max), channel version reported; PREREG e466875 — test_decoder_gate_uses_physical_fields RED->GREEN
Final: PREREG follow-up F3 done: S-L port term = S's own Pi — test_sl_port_term_is_s_projection RED->GREEN
Final: PREREG follow-up F2(i) done: C mapped by conservative minmod reconstruction (default), injection as sensitivity — test_minmod_reconstruction_is_conservative_and_second_order RED->GREEN (first run: an error in the test's expected values)
Final: minor (deferred): structure_terms NaN if kappa = 0 with port projection; three SVDs where one would do.
Final: minor (deferred): penalty_weight divides by zero if the initial penalty is 0.
Final: minor (deferred): wall_abs is a lower bound on the integral of |wall heat| (docstring).
Final: minor (deferred): torch eta cancellation just above the 1e-14 switch (Taylor series).
Final: minor (deferred): U+F scalar and gradient inputs need a standardiser fitted on training latents (Plan 3).
Final: minor (deferred): TorchStoreRHS has no production() check.
Final: minor (deferred): no second-order test for MUSCL in system B at v < 0.
Final: minor (deferred): kappa_binds counted when delta <= 0 (literal spec; a sentence in the paper).
Final: Ruling (declined-to-judge items): Plan 1 constants (DELTA_AX, D_L velocity), violation metric and training loops (Plan 3), Coarsener internals, FNO activation choices, whether d passes the gate (pilot) — each outside Plan 2 by scope — cost if wrong: as stated per item.
Final: fix pass suite 225/225 (pytest -q tests, 1956 s)
