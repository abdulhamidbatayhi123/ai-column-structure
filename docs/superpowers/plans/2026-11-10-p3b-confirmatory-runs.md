# Paper 3B confirmatory runs on Kaggle: implementation plan (draft)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** run the frozen study's cells on Kaggle at four one-thread runs per session, continuing runs that
pass the 12-hour session limit, and evaluate each cell from the kernel that holds its final weights.

**Architecture:** `kaggle/train_jobs.py` packs cells into kernels of four commands (the kernel template
already runs each command as its own process). `kaggle/run_queue.py` sees a finished kernel whose output
holds paused cells, pushes a continuation kernel for those cells with the previous output attached, and
records the final kernel of every cell in `kaggle/continuations.json`. `evaluation_jobs` reads that map.

**Tech stack:** Python, Kaggle CLI, the existing queue and chunk runners.

**Spec:** `PREREG_P3B_draft.md` (frozen version at the freeze), `docs/compute_3B.md` (measured costs: four
runs per session give 2.08 × the throughput of one; S at h = 2 and 8,000 iterations takes about 14 h at four
per session), `tests/test_resume_sessions.py` (resume already in the trainer, `train_cell` and
`train_chunk`).

**Status:** draft written 2026-11-10 while waiting for the pilot. Built after the freeze, when the iteration
count, h and the ceiling are known.

## Global constraints

- One thread per run (`threads: 1`); autoencoder jobs keep 4 threads.
- `--hours 11` for every training command: a session ends at 12 h, and a paused run must save its output.
- Every result file keeps the frozen commit; the kit is uploaded and confirmed `ready` before any push.
- Never push, stop or change another project's kernels.

## Review focus

1. A cell paused twice or more: the continuation chain `-r1`, `-r2` keeps the same autoencoder and
   reaches the same weights as one uninterrupted run.
2. A kernel where some cells finish and others pause: only the paused cells continue, and the finished
   cells' outputs are never overwritten.
3. A continuation whose kernel fails (ERROR): it is reported, never silently replaced by a fresh start.
4. The evaluation of a continued cell reads the last continuation's weights, not the first kernel's.
5. A cell pausing without progress (one iteration per session, deadline too short): stop after a set
   number of continuations rather than looping.

## Tasks

### Task 1: one resume file per cell

`train_cell` writes `RESUME_<job>.json` instead of `RESUME.json`, so that four cells sharing one output
directory do not overwrite each other's file. Test: two cells paused in one directory leave two files.

### Task 2: four cells per kernel

`build_jobs(cs, per_kernel=4)`: cells of the same autoencoder (or solver cells) packed four per kernel,
each command `train_chunk.py --cell NAME --out OUT --hours 11`, `expect` = every cell's result file,
`depends_on` = the autoencoder job. Tests: every cell appears once; no kernel mixes autoencoders; the
counts match `cells()`.

### Task 3: continuation kernels in the queue

A pure function `continuation(kernel, paused, n) -> job` (name `<kernel>-r<n>`, the paused cells'
commands with `--resume-from /kaggle/input`, `depends_on` = the previous kernel plus the autoencoder job).
The queue loop calls it when a COMPLETE kernel's output holds `RESUME_*.json` files, and records
`continuations.json` {cell: final kernel}. It stops at `MAX_CONTINUATIONS = 4` with a "needs attention"
line. Tests: the pure function, and the loop with a fake Kaggle CLI (the `run` callable injected).

### Task 4: evaluation from the final kernel

`evaluation_jobs(cs, sets, final=continuations)`: `depends_on` = the cell's final kernel. Test: a
continued cell's evaluation depends on `-r1`, not on the first kernel.

### Task 5: end-to-end dry run

Two quick cells, `--hours` tiny, through the fake queue: pause, continue, finish, evaluate. The weights
equal an uninterrupted run.
