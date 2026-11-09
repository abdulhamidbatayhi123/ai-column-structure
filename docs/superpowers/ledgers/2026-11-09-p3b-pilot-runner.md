# SDD ledger — plan: docs/superpowers/plans/2026-11-09-p3b-pilot-runner.md
Spec: PREREG Section 10; p3b/pilot/rules.py input schemas.
Task 1: Ruling: a trajectory that does not reach CSS within its stored cycles is recorded as n_cycles + 1 (the rule takes the max, so the long-validation check then runs) — cost if wrong: a conservative check.
Task 1: complete (tests: pytest -q tests/test_pilot_runs.py → 4 passed; RED at import first)
Task 2: Ruling: S provisional kappa for the pilot runs is 1e-3 x the median |P grad S|^2 of encoded pilot snapshots (no trained dynamics needed), breaking the circularity between kappa and the S sweep — cost if wrong: the pilot S runs use a kappa slightly different from the final one (the final value is computed the same way).
Task 2: complete (tests: 3 passed; RED first)
Task 3: Ruling: the near-dry stiffness check is h rho(J) < 2.5 (RK4 real-axis limit 2.785 with a margin) at the encoded validation states with the smallest concentration, rho by power iteration on central-difference JVPs (S calls reverse-mode autograd inside, so forward mode cannot wrap it) — cost if wrong: a slightly different h; the criterion is stated with the pilot result.
Task 3: complete (tests: 2 passed; RED first)
Task 4: Ruling: "reproduces C's eps_W within 1 %" read as max over pilot trajectories and cycles of |eps_W(C+L) - eps_W(C)| <= 0.01 (both are fractions of the reference release), C on the same 32 cells; a failed run of either gives inf, so that K is never chosen — cost if wrong: K one step larger or smaller
Task 4: Ruling: C+L's Newton does not converge on 4 or 8 cells at large snapshot steps (4 cells, 508 s: residual 1.5e4); a damped fallback was tried, did not rescue it (residual 3.8) and was reverted; tests of the K rule use 16 stored cells; the study runs C+L on 32 cells, which converges at T/40 and T/200 — cost if wrong: Newton failures in the pilot K rule show as inf and are visible
Task 5: Ruling: in the pilot H2 and H4 share models (pilot_train holds 32 trajectories, so full pilot data = size 32; the H4 SD is then conservative); H5 has no pilot OOD set, its SD is taken at cycle 20 of pilot_val; SDs from U and S only (the power simulation's contrast), two seeds — cost if wrong: H5 power estimate off
Task 5: Ruling: positive control on the H2 statistic at alpha 0.05, only the word written — cost if wrong: another statistic might have shown a different word
Task 4-5: complete (commit above, tests: pytest tests/test_pilot_runs.py tests/test_pilot_sds.py tests/test_power_sim.py tests/test_pilot_rules.py -> 33 passed; tests/test_cplusl_train.py + moved C+L loss tests -> pass)
Task 6: Ruling: the pilot runs in 5 Kaggle phases (d, K, tau_S -> lr -> iterations -> h -> two-seed runs), with p3b.pilot.collect.stage writing chosen values only between phases (pilot_stage.json in the kit) — the dependencies between the rules force it — cost if wrong: about four extra kit uploads
Task 6: Ruling: A_replication and B pilots (8 x 60) split by design index 0-5 train / 6-7 validation for the decoder gate and S's h check; CSS and reference noise use every trajectory with the key's full cycle count — cost if wrong: thresholds of those keys from 2 validation trajectories
Task 6: Ruling: the lr sweeps, curves and S's h-check training use h_steps 1 (T/200); a 200-snapshot check (scratchpad dbg_s6.py) shows S rolls out 2 cycles at d=8 at h_steps 1, 2, 4, untrained and trained; only d=4 (below the study's grid) needs h_steps >= 2. If S's sweep fails at h_steps 1 in the pilot, stage 1 stops with a visible error — cost if wrong: one phase rerun at h_steps 2
Task 6: finding (dry run): with a 4-d latent and 40 snapshots per cycle, S's port projection gives 16 x the data's latent speed (grad E^, grad N^ nearly parallel, second singular value of B 0.038) and RK4 diverges; recorded honestly as a failed rollout. Fixed two places that would have hidden such a failure: h_check now makes an h with any failed rollout inadmissible (viol_rate 1, "failed" count; test_h_check_makes_an_h_with_a_failed_rollout_inadmissible RED->GREEN), and wall_per_iter refuses a timing made mostly of skipped windows (test RED->GREEN)
Task 6: deviation: pieces.py, collect.py, pilot_chunk.py and pilot_jobs.py were written before their tests; the stage tests and the dry run then passed without a RED step (collect_ae, the kit test and the two fixes above did go RED first). The g2/kappa refactor test was not run RED (4-minute fixture; function absent)
Task 6: complete (tests: 60 passed in the full pilot set, then the dry run and timing test 2 passed after the realistic-data change; 58 min under low RAM)
Final review: opus, whole plan 9e69946..133d3d1: no Critical, 7 Important, 14 Minor; blinding, keys and tau_S source clean.
Final: fixed I1 K compares W(C+L) and W(C) directly (k_discrepancy; failed rows inf) — test_I1_* RED->GREEN
Final: fixed I2 violation rate per snapshot interval at every k — test_I2_violation_rate_does_not_shrink_with_more_steps RED->GREEN
Final: fixed I3 stiffness with each picked state's own interval — test_I3_stiffness_uses_the_picked_snapshots_own_interval RED->GREEN
Final: fixed I4 defects per k, once per interval (rollout any_defect), NaN for failed rollouts, final takes the chosen h — test_I4_* and test_final_takes_the_defects_of_the_chosen_h RED->GREEN
Final: Ruling: a defect step is a snapshot interval with any defect flag (OR over substeps), divided by intervals, matching the study's per-interval flags — cost if wrong: the defect bound differs by at most a factor k
Final: fixed I5 runs with > 5 % skipped iterations: curves and lr losses NaN (not finished), S's h-check piece stops, two-seed runs carry their counts and final stops — test_I5_* RED->GREEN
Final: fixed I6 every piece is {"meta", "value"} with stage sha256 (phase keys only), kit commit, data and autoencoder sha256, q, seeds, torch version; the collector uses only pieces of the current stage; kernel names end in the stage hash — test_I6_* RED->GREEN
Final: fixed I7 load_piece searches the nested download folders — test_I7_* RED->GREEN
Final: fixed (upgraded from minor) M1 positive control at alpha 0.05/3; M2 stage n drops later phases' keys; M4 the /kaggle/input fallback removed and duplicate data names are an error; M5 KEYS built with pilot_key; M13 wall piece times every latent arm and the plan table corrected — tests RED->GREEN; collect_ae refuses two different downloads of one autoencoder
Final: minor (deferred): M3 stage 3 stops only for the primary key; None h for A_Al-fumarate or B reaches pilot_values and fails later in cell_pilot (loudly)
Final: minor (deferred): M6 two_seed_rows records violation counts 0 with inf floors (unused)
Final: minor (deferred): M7 pilot kappa per (key, seed) vs the study's one kappa per key from seed 0
Final: minor (deferred): M8 count capped pairs (both arms failed before cycle 50) beside sds.json
Final: minor (deferred): M9 write the H5 in-distribution and H1-cycle caveats into pilot_values
Final: minor (deferred): M10 wall_per_iter times untrained arms at lr 1e-3, h_steps 1
Final: minor (deferred): M11 system B "near-dry" states by smallest T_f; choose by largest rho or document
Final: minor (deferred): M12 h_steps 1 is T_half/100 per half-cycle, not exactly T_cycle/200; declare in PREREG
Final: minor (deferred): M14 the dry run sets d, K, iterations and h by hand (stage 1 now runs the real rule)
Final: declined to judge (reviewer): Kaggle 12 h limit for curve-S and s-<key> without cross-session resume — Ruling: measure in phase 0/1 timings before phase 2; if curve-S exceeds 10 h, add resume via a checkpoint in /kaggle/working and the kernel's previous output — cost if wrong: one rerun of a phase-2 kernel
Final: fixed (deferred minor taken) M8, M9 pilot_notes.json beside the ten inputs: capped pairs per hypothesis and the H5/H1 caveats — test_M8_M9_notes_count_capped_pairs_and_state_the_caveats RED->GREEN; M12 declared in PREREG Section 10 with a rationale row (number gate and decision list PASS)
Final: fixed (deferred minor taken) M3 stage 3 stops when any key has no admissible h — test_M3_stage3_stops_when_any_key_has_no_admissible_h RED->GREEN
