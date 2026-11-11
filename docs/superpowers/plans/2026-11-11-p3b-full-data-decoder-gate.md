# Paper 3B full-data decoder gate: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** set d (and the decoder kind) as PREREG Section 4 states it, "the smallest in {16, 24, 32, 48, 64}
that passes the decoder gate (Section 5.1) at full data", where Section 5.1 requires "all seeds of the
full-data autoencoders" to pass on validation (and on long validation at cycles 40 to 60 when the pilot's
CSS is after cycle 6); the conv decoder on the same grid if no MLP d passes; else that system and material
is "not testable".

**Why:** found 2026-11-11. The study read d from `pilot_values.json`, but PREREG 10 does not let the pilot
set d; the pilot gate is a risk report (32 or 6 training trajectories against 256).

**Architecture:** a new autoencoder-gate job per (system, material, decoder kind, d, seed) at size full
(`train_chunk.py --gate-ae NAME`) writes the weights and `decoder_gate` metrics on the study's validation
set (and long validation when required). `scripts/decoder_choice.py` reads them and writes
`results/decoder_choice.json` {key: {"decoder": kind, "d": d} | {"not_testable": true}} with its own and
its inputs' SHA-256. `train_chunk.py` reads d and the decoder from that file (`--decoder-choice`), never
from the pilot values; a missing file or a not-testable key stops with a clear message.

**Spec:** `PREREG_P3B_draft.md` Sections 4, 5.1, 10.

## Global constraints

- The gate jobs train with the pre-registered recipe (60 epochs, batch 256, Adam 1e-3 cosine to 1e-5) and
  the study seeds of the size-full autoencoders; nothing about arms is read.
- Conv jobs run only for keys where no MLP d passed (a second wave), as PREREG 4 orders it.
- The chosen d's gate weights are not reused as the study's autoencoders; the study's AE jobs retrain at the
  chosen d (same seed), so every AE comes from one code path.

## Review focus

1. One seed failing at a d: that d does not pass (all seeds must).
2. CSS after cycle 6: long validation enters the gate; otherwise it does not.
3. A key not testable: its cells are not built; other keys go on.
4. `decoder_choice.json` absent or from other inputs (SHA mismatch): train_chunk refuses.
5. The conv wave starts only after every MLP job of that key has finished.

## Tasks

### Task 1: `full_data_choice(metrics) -> (kind, d) | (None, None)`
In `p3b/latent/autoencoder.py`. `metrics = {kind: {d: [gate dict per seed]}}`; a d passes when every
seed's `passed` is true (and `passed_long` where present). Tests: one failing seed blocks a d; MLP before
conv; none -> (None, None).

### Task 2: gate jobs
`kaggle/train_jobs.py: gate_jobs(keys, kind)` names `p3b-gate-<key>-<kind>-d<d>-s<seed>`; `train_chunk.py
--gate-ae NAME` trains, saves, scores on validation (+ long validation if CSS > 6 from `pilot_values.json`
css) and writes `gate_<name>.json`. Tests: names unique and cover grid x seeds; the job writes the metrics
of `decoder_gate`.

### Task 3: `scripts/decoder_choice.py` and train_chunk reading it
Writes `results/decoder_choice.json`; `train_ae` and `cell_pilot` take d and decoder from it. Tests: the
choice file drives the AE kind and d; absent file raises; not-testable key raises for its cells only.
