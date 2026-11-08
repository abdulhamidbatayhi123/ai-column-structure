# Paper 3B Reference Data Pipeline (Plan 1 of 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate, coarsen, store and gate the reference trajectories of both systems
exactly as the pre-registration specifies. Measure the reference cost on the target grids
before anything is frozen.

**Architecture:**
- **One package.** The verified physics in `verify/` moves into a package `p3b/`, and `verify/`
  keeps thin re-export shims, so the existing checks and CI stay valid.
- **New modules on top of it,** one job each: materials and bed constants, design sets
  (Sobol), cycle simulation (system A and system B), coarse storage, reference gates, and a
  timing and grid-convergence harness.
- **Kaggle packaging.** A small kit builder and a resumable queue, after Paper 2's pattern,
  run the generation off the laptop.

**Tech Stack:** Python 3.12, numpy 2.4.6, scipy 1.17.1 (BDF, Sobol QMC), pytest 9.0.3; Paper 1
release v2.0.1 (`gen_parametric_dataset.py` constants, read-only) for the Glueckauf
constants and the bed builder.

**Spec:** `PREREG_P3B_draft.md` (Sections 2, 5.3, 10, 11) at commit be8bec6;
`docs/formulation_3B.md` (Sections 3 to 7); `docs/materials_3B.md`; `docs/compute_3B.md`.

**Plans 2 to 4 (later):**
2. autoencoder and arms;
3. training, evaluation and metrics;
4. pilot rules, power simulation and the Kaggle orchestration of training.

## Global Constraints

**System A:**
- dilute-solute gas model; c_w = 33.6 J mol⁻¹ K⁻¹; hybrid-scheme convection (upwind plus compensated diffusion, Task 7b); exact discrete-gradient
  Fick dispersion; wall and convection as ports;
- N_z = 2048 (system A), 1024 (system B), 128 coarse cells; the coarsening factor must be an
  integer;
- 200 snapshots per cycle, stored float32 (state integrated in float64);
- bed: ε 0.39, ρ_p 1050 kg m⁻³, d_p 3.25 mm, L 0.10 m, D_in 0.05 m, k_z = 0.10 + 0.5 ρ_g C_pg v d_p W m⁻¹ K⁻¹ (Task 7b),
  h_w 10 W m⁻² K⁻¹; LDF coefficient from Paper 1's Glueckauf relation with the material's own
  isotherm;
- materials: from `verify/materials/fit_isotherms.json`; concentration-based −ΔH = 51.3e3
  (MOF-303) and 48.5e3 (Al-fumarate) J mol⁻¹;
- initial state: equilibrium at T_ads with c_0 = 1e-4 · c_in;
- cycle: adsorption for α·t_st at (c_in, T_ads), wall T_ads; then desorption for β·α·t_st with
  the same c_in heated to T_des, wall T_des;
- t_st = (1−ε)ρ_p q*(c_in, T_ads) + ε c_in, times L, divided by v·c_in.

**System B:**
- `ThermalStore` defaults (ε 0.38, L 1.0 m); charge with T_hot from z = 0 for α·t_front, then
  discharge with T_cold from z = L for β·α·t_front;
- t_front = L (C_f + C_s)/(ρ_f c_f v); the bed starts uniform at T_cold.

**Sampling boxes** (scrambled Sobol, seeds fixed in `scripts/design_sets.py`):
- A: RH_ads 0.20 to 0.60, T_ads 293 to 308 K, T_des 333 to 363 K, v 0.03 to 0.30 m s⁻¹, α 0.5
  to 1.2, β 0.5 to 1.0;
- A OOD: RH_ads 0.10 to 0.20, T_des 363 to 383 K, the rest as A;
- B: T_hot 343 to 363 K, T_cold 293 to 303 K, v 0.5e-3 to 3.0e-3 m s⁻¹, α 0.4 to 0.9, β 0.6
  to 1.0.

**Set sizes** (trajectories × cycles):
- A-primary: pilot 32×6 train + 16×60 validate; train 256×6; validation 32×6; long validation
  4×60; long test 48×60; OOD 24×30;
- A-replication and B: pilot 8×60; train 256×6; validation 32×6; long validation 4×60; long
  test 48×60;
- train subsets nested: 16 ⊂ 32 ⊂ 64 ⊂ 128 ⊂ 256 (first k of the pool).

**Gates:**
- closure: rate identities ≤ 1e-12 (float64); integrated closure with cumulative-flux ODE
  states ≤ 10 × rtol; stored float32 data against stored ports ≤ 1e-5 relative;
- entropy diagnostics only on cells with c > 1e-8 c_in and q > 1e-8 q_capacity (system A);
  the share below the floor is reported.

**Solver:** BDF; rtol 1e-6 and atol 1e-10 in the generator (Paper 1's production
tolerances); explicit Jacobian sparsity.

**Process:**
- every output file carries the git commit hash it was produced at;
- no e-mail address in any request;
- local heavy jobs start only with at least 1.5 GB of free RAM;
- stop only processes you started.

## Review Focus

1. **Undershoot at sharp fronts (n ≈ 9 to 19).** A BDF undershoot can give c ≤ 0 or q ≤ 0
   in a few cells. The stored log channels must stay finite: floor at 1e-8 c_in and 1e-8
   q_capacity, count the floored cells. A NaN must never be written. Test in Task 6.
2. **A solver failure mid-trajectory.** The generator must record the failure (set,
   trajectory index, cycle, message) and continue with the next trajectory, never crash the
   batch or write a partial trajectory as complete. Test in Task 4.
3. **Memory.** A fine history (2048 × 3 × 200 × 60 float64, about 590 MB per trajectory)
   must never be held. Coarsen each cycle as soon as it is solved, and keep only the end
   state. Test in Task 6 (the coarsened array's shape; no fine history attribute).
4. **Non-integer coarsening** (for example N_z = 2000 onto 128) must raise ValueError, not
   resample silently. Test in Task 6.
5. **Re-running a set must reproduce it bit for bit:** the same Sobol points, the same
   order, the same nested subsets. Test in Task 3.

---

### Task 1: Package `p3b` and move the verified physics

**Files:**
- Create: `p3b/__init__.py`, `p3b/column.py` (from `verify/consistent_column.py`),
  `p3b/isotherms.py` (from `verify/isotherms.py`), `p3b/thermal_store.py` (from
  `verify/thermal_store.py`), `pyproject.toml` (package `p3b`, no dependencies beyond
  requirements.txt)
- Modify: `verify/consistent_column.py`, `verify/isotherms.py`, `verify/thermal_store.py`
  become re-export shims (`from p3b.column import *`, and so on)
- Modify: `.github/workflows/verify.yml` (add `pip install -e ai-column-structure` before the
  tests)
- Test: the existing `tests/`, unchanged

**Interfaces:**
- Produces:
  - `p3b.column.Column(phys, N, c_w, iso=None)`;
  - `p3b.isotherms.MultiTermIsotherm`, `from_fit(fit, dH, T_ref=303.15)`, `from_paper1(phys)`,
    `p_sat(T)`, `R`;
  - `p3b.thermal_store.ThermalStore(N, ...)`.

  These are the same signatures as today.

- [ ] **Step 1: Run the existing suite and gates, and record that they pass.**
  `python -m pytest -q tests` → 13 passed. `python verify/check_results.py --bindings` → ALL PASS.
- [ ] **Step 2: Move the three modules into `p3b/` with `git mv`, then write the shims.** Inside
  `p3b/column.py`, import `from p3b.isotherms import from_paper1` (no sys.path hacks).
  Paper 1's path insertion stays only where Paper 1 modules are imported.
- [ ] **Step 3: `pip install -e .`, then rerun Step 1.** Expected: 13 passed, ALL PASS.
- [ ] **Step 4: Commit.** `git commit -m "refactor: physics into package p3b; verify/ keeps shims"`

### Task 2: Materials and bed constants

**Files:**
- Create: `p3b/materials.py`
- Test: `tests/test_materials.py`

**Interfaces:**
- Consumes: `p3b.isotherms.from_fit`, `p_sat`, `R`.
- Produces:
  - `BED: dict`;
  - `DH_C: dict[str, float]` = `{"MOF-303": -51.3e3, "Al-fumarate": -48.5e3}`;
  - `load_isotherm(name: str) -> MultiTermIsotherm`;
  - `bed_physics(v: float, T: float)`, a Paper 1 `AdsorptionPhysicsConfig` with the Global
    Constraints' bed values;
  - `k_ldf(iso, phys, c_in: float, T: float) -> float` (Glueckauf, `K = q*/c_in`, Paper 1's
    D_M_WATER, TORTUOSITY and EPS_PELLET);
  - `c_in_from_rh(rh: float, T: float) -> float`;
  - `t_stoich(iso, phys, c_in, T) -> float`.

- [ ] **Step 1: Write the failing test `test_mof303_feed_values`.** At RH 0.40 and 303.15 K,
  v 0.10: `load_isotherm("MOF-303").q_star(c_in, 303.15)` is 22.2003 ± 1e-3,
  `k_ldf(...)` is 4.9702e-4 ± 1e-7 s⁻¹, and `t_stoich(...)/3600` is 5.8745 ± 1e-3 h (values
  from `verify/timing_materials.json`).
- [ ] **Step 2: Write the failing test `test_unknown_material_raises`.**
  `load_isotherm("ZIF-8")` raises KeyError.
- [ ] **Step 3: Run them.** Expected: FAIL (module missing).
- [ ] **Step 4: Implement `p3b/materials.py`.** Read the JSON with a path relative to the
  package root.
- [ ] **Step 5: Run.** Expected: PASS.
- [ ] **Step 6: Commit.** `git commit -m "feat: materials and bed constants"`

### Task 3: Design sets (Sobol, nested, reproducible)

**Files:**
- Create: `p3b/design.py`, `scripts/design_sets.py` (holds the seeds; writes
  `data/design/<system>_<set>.json`)
- Test: `tests/test_design.py`

**Interfaces:**
- Produces:
  - `BOXES: dict[str, dict[str, tuple[float, float]]]`, with keys "A", "A_OOD" and "B";
  - `SET_SIZES: dict[tuple[str, str], tuple[int, int]]`, mapping (system, set) to
    (trajectories, cycles), with systems "A_primary", "A_replication" and "B", and sets
    "pilot_train", "pilot_val", "train", "validation", "long_validation", "long_test", "ood"
    and "scenarios" (A_primary only); A_replication and B use "pilot", "train",
    "validation", "long_validation" and "long_test". "scenarios" is 8 operating points
    (RH_ads, T_ads, v) for H6; their cycle count is set later by the cyclic-steady-state rule;
  - `make_set(system: str, set_name: str, seed: int) -> list[dict]`. Each dict holds the
    sampled quantities plus "index" and "n_cycles";
  - `train_subset(pool: list[dict], k: int) -> list[dict]` (first k).

- [ ] **Step 1: Write the failing tests.**
  - `test_sizes`: every (system, set) has the trajectory and cycle counts of Global
    Constraints.
  - `test_ranges`: every sampled value lies inside its box; OOD uses `A_OOD`.
  - `test_reproducible`: two calls with the same seed are equal element by element.
  - `test_nested`: `train_subset(pool, 32)[:16] == train_subset(pool, 16)`.
  - `test_sets_disjoint`: no two sets of one system share a point (different seeds).
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** Use `scipy.stats.qmc.Sobol(d, scramble=True, seed=seed)`,
  `random_base2(m)` with 2^m ≥ the count, take the first `count` points, and map linearly.
  `scripts/design_sets.py` fixes one seed per (system, set), written as literals in the
  script, and writes the JSONs with the commit hash.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat: reproducible nested Sobol design sets"`

### Task 4: Cycle simulator, system A

**Files:**
- Create: `p3b/simulate_a.py`
- Test: `tests/test_simulate_a.py`

**Interfaces:**
- Consumes: `Column`, `p3b.materials.*`.
- Produces:
  - `simulate_trajectory_a(point: dict, material: str, N_z: int, n_snap: int = 200, on_cycle=None) -> TrajectoryResult`.
    `on_cycle(cycle_index, t, y_fine, ports)` is called once per cycle with that cycle's
    snapshots, so the caller can coarsen and discard (Task 6).
  - `TrajectoryResult`: a dataclass with `ok: bool`, `failed_cycle: int | None`,
    `message: str`, `cpu_s: float`, and `closure: list[float]` (the integrated relative
    closure per half-cycle).
  - The state is augmented with 4 cumulative port states (water in, water out, energy in,
    energy out). `ports` holds their values at the snapshots.

- [ ] **Step 1: Write the failing tests** (N_z = 64 so they run in seconds).
  - `test_prewet_initial_state`: the first snapshot has c = 1e-4 · c_in in every cell and
    q = q*(c_0, T_ads).
  - `test_desorption_feed_same_absolute_humidity`: the desorption inputs use the same c_in
    and T_in = T_des.
  - `test_integrated_closure`: every entry of `closure` ≤ 1e-5 (10 × rtol).
  - `test_failure_is_recorded`: with a point whose v is forced to 0, the function returns
    `ok=False` with `failed_cycle` set, and raises nothing.
  - `test_callback_once_per_cycle`: `on_cycle` is called `n_cycles` times.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** Wrap `Column.rhs` with the 4 extra port-integral states. Integrate
  each half-cycle with `solve_ivp(..., method="BDF", rtol=1e-6, atol=1e-10, jac_sparsity=...)`,
  extending the sparsity pattern with the port rows, and use `t_eval` for the 100 snapshots
  of each half-cycle. Catch the solver's failure and return it as data.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat: system A cycle simulator with port integrals"`

### Task 5: Cycle simulator, system B

**Files:**
- Create: `p3b/simulate_b.py`
- Test: `tests/test_simulate_b.py`

**Interfaces:**
- Consumes: `ThermalStore`.
- Produces: `simulate_trajectory_b(point, N_z, n_snap=200, on_cycle=None) -> TrajectoryResult`,
  with the same dataclass and callback contract as Task 4. Its ports are energy in and energy
  out, and the flow direction is part of `ports`.

- [ ] **Step 1: Write the failing tests** (N_z = 64).
  - `test_flow_reverses`: the charge half has v > 0 and the discharge half v < 0.
  - `test_integrated_closure` ≤ 1e-5.
  - `test_initial_uniform`: T_f = T_s = T_cold.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement**, mirroring Task 4.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat: system B cycle simulator (flow reversal)"`

### Task 6: Coarse storage

**Files:**
- Create: `p3b/store.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Consumes: the `on_cycle` callback contract of Tasks 4 and 5; `Column`; `ThermalStore`.
- Produces:
  - `Coarsener(system: str, N_z: int, N_c: int = 128, model)`, which raises ValueError if
    `N_z % N_c`;
  - `.add_cycle(cycle_index, t, y_fine, ports)` keeps only coarse float32 arrays;
  - `.to_npz(path, meta: dict)` writes:
    - `channels` (n_cycles, 200, N_c, C): system A (log c, log x, log T), C = 3; system B
      (T_f, T_s), C = 2;
    - `outlet` (n_cycles, 200, k): system A (log c_out, log T_out), k = 2; system B (T_out),
      k = 1;
    - `wall_heat_blocks` (n_cycles, 200, N_c);
    - `ports_cum` (n_cycles, 200, n_ports);
    - `conserved_blocks` (n_cycles, 200, N_c, C) in float64 for the gate;
    - `floored_cells` (int);
    - `meta` (incl. the commit hash).

- [ ] **Step 1: Write the failing tests.**
  - `test_non_integer_factor_raises`: `Coarsener("A", 2000, 128)` raises ValueError.
  - `test_totals_exact`: the block averages of (n, a, u) times the coarse Δz sum to the fine
    totals within 1e-12 relative.
  - `test_channels_finite_with_undershoot`: a fine state with c = −1e-9 in one cell gives
    finite channels and `floored_cells == 1`.
  - `test_no_fine_history`: after 3 cycles the object holds no array with N_z in its shape.
  - `test_roundtrip`: decoding the stored log channels through the thermodynamics recovers
    the block (n, a, u) within float32 precision (1e-5 relative).
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** Block-average the conserved variables by reshape-mean, then
  convert to channels through `Column.temperature` and `iso.X`, flooring c and q as in the
  Global Constraints.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat: coarse storage of conserved block averages and fine ports"`

### Task 7: Reference gates on stored trajectories

**Files:**
- Create: `p3b/gates.py`, `scripts/check_reference_gate.py`
- Test: `tests/test_gates.py`

**Interfaces:**
- Consumes: the npz written by `Coarsener.to_npz`; `Column` / `ThermalStore` for entropy.
- Produces:
  - `stored_closure(npz) -> float`: the maximum relative mismatch between the change in the
    totals of `conserved_blocks` and `ports_cum`;
  - `coarse_entropy_margin(npz, model) -> float`: the minimum over snapshots of
    (dS̄/dt − Φ_S)/scale, by finite differences of the stored conserved blocks at the snapshot
    spacing;
  - `floor_share(npz) -> float`;
  - `css_cycle(npz) -> int | None`: the first cycle k ≥ 2 with |W_k − W_(k−1)| / |W_k| < 1e-3,
    where W is the net release from `ports_cum`. The threshold 1e-3 is added to the
    pre-registration's Section 5.3 in this task, with a line in the rationale file;
  - the script prints one PASS/FAIL line per gate for every file in a directory.

- [ ] **Step 1: Write the failing tests** on a trajectory generated in-test (system A,
  N_z = 128, N_c = 32, 2 cycles).
  - `test_stored_closure` ≤ 1e-5.
  - `test_entropy_margin_positive` > 0.
  - `test_floor_share` < 0.01.
  - `test_css_cycle`: on a synthetic W sequence [10, 6, 5.5, 5.505, 5.5051], `css_cycle`
    returns 4.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat: reference gates on stored trajectories"`

### Task 8: Timing and grid-convergence harness (the feasibility measurement)

**Files:**
- Create: `scripts/time_reference.py`
- Test: `tests/test_time_reference.py` (smoke only)

**Interfaces:**
- Consumes: `simulate_trajectory_a`, `simulate_trajectory_b`, `p3b.materials`.
- Produces: `results/timing_reference.json`, a list of records
  `{system, material, N_z, cycle, cpu_s, nfev, ok}`, plus
  `{grid_convergence: max relative change of the outlet curve between N_z and N_z/2}` (Paper
  1's `grid_convergence()` criterion), with the commit hash.
- CLI: `--system A|B --material NAME --nz 512,1024,2048 --cycles 2 --threads 1`.

- [ ] **Step 1: Write the failing smoke test `test_harness_runs_small`.** It runs `--nz 32,64
  --cycles 1` for system A MOF-303 and system B into a temporary directory, and asserts that
  the JSON has 4 records with `cpu_s > 0`, plus the grid-convergence key.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** CPU time comes from `time.process_time`. Set `OMP_NUM_THREADS` and
  `OPENBLAS_NUM_THREADS` to `--threads` before numpy is imported. Write the free RAM at start
  into the record.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat: reference timing and grid-convergence harness"`

### Task 9: Kaggle kit and resumable queue for data generation

**Files:**
- Create: `kaggle/make_kit.py`, `kaggle/generate_chunk.py`, `kaggle/queue.py`,
  `kaggle/README.md`
- Test: `tests/test_kaggle_kit.py`

**Interfaces:**
- Consumes: `p3b.*`, `data/design/*.json` (Task 3), and Paper 2's pattern as a reference
  (`ai-kinetics-discoverability/kaggle_b90/`, read-only).
- Produces:
  - `make_kit(out_dir, commit) -> Path`: a dataset directory containing `p3b/`, the design
    JSONs, the three Paper 1 files at v2.0.1, `kit_commit.txt`, and a SHA-256 manifest;
  - `chunk_indices(n_traj: int, n_chunks: int) -> list[list[int]]`;
  - `generate_chunk.py --system --set --material --chunk K --of M --nz N`, which writes one
    npz per trajectory plus `chunk_K_report.json` (failures included);
  - `queue.py`: pushes the kernels absent on Kaggle, downloads finished outputs into
    `data/reference/<system>/<set>/`, is resumable, and ends with `ALL_OUTPUTS_DOWNLOADED`.

- [ ] **Step 1: Write the failing tests.**
  - `test_chunks_partition`: the union of `chunk_indices(48, 5)` is range(48), with no
    overlaps and chunk sizes differing by at most 1.
  - `test_kit_manifest`: the kit holds `kit_commit.txt` equal to the passed commit, and every
    file in the manifest has a matching SHA-256.
  - `test_generate_chunk_small`: `generate_chunk.py` on 2 trajectories at N_z = 64 writes 2
    npz and a report with `ok=true`.
- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** `queue.py` uses the `kaggle` CLI (installed by the user, as for
  Paper 2). It never stores credentials in the repo, and its README states how to run it.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat: Kaggle kit and resumable queue for reference data"`

### Task 10: The feasibility run and the CI update

**Files:**
- Modify: `docs/compute_3B.md` §1 (replace the PARTIAL row with the measured values),
  `.github/workflows/verify.yml` (the new tests run in CI; the timing harness stays out of CI)
- Create: `results/timing_reference.json` (from the run)

- [ ] **Step 0: Run the Paper 1 limit at N_z = 2048** (`verify/run_checks.py` V1 with
  `N1 = 2048`, on Kaggle CPU). Expected: the same bounds as V1 (below 2e-9 in c, 1e-8 in q,
  2e-7 K in T). Record the values in `verify/results_v1_2048.json`.
- [ ] **Step 1: Run the timing harness on Kaggle CPU.** System A for both materials at
  512, 1024 and 2048 cells, 2 cycles each; system B at 512, 1024 and 2048. Download the
  JSON with `queue.py`.
- [ ] **Step 2: Apply the pre-registered data-cost rule (pre-registration Section 2) to the
  measured cost.** If the cost per cycle exceeds 2 × 30 s at 2048 and the 1024 grid passes
  the grid-convergence criterion (Paper 1's threshold), record N_z = 1024 for system A in
  `docs/compute_3B.md` and in the pre-registration's Section 2. Otherwise keep 2048. Write
  the decision and its numbers into the change table of `docs/prereg_rationale_3B.md`.
- [ ] **Step 3: Run the full local suite.** `python -m pytest -q tests` → all pass;
  `python verify/check_results.py --bindings` → ALL PASS;
  `python verify/check_decision_list.py` → PASS.
- [ ] **Step 4: Commit and push; confirm CI is green** with `gh run list --limit 1` → success.
  `git commit -m "chore: measured reference cost; data-cost decision recorded"`
