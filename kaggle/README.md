# Reference-data generation on Kaggle

Heavy reference runs (the 2048- and 1024-cell solves) run on Kaggle CPU sessions, as in
Paper 2. Nothing here stores credentials. The `kaggle` CLI reads your own
`~/.kaggle/kaggle.json`.

## 1. Build and upload the kit (once per code commit)

```bash
python kaggle/make_kit.py kaggle/build/kit
kaggle datasets create -p kaggle/build/kit        # first time; afterwards: kaggle datasets version -p ... -m "<commit>"
```

`make_kit.py` refuses to build from an unclean tree. The kit contains:
- `p3b/`, `kaggle/generate_chunk.py`, `scripts/time_reference.py`;
- the isotherm fits and the design sets (`data/design/*.json`, never regenerated);
- Paper 1's three solver files, downloaded read-only from release v2.0.1;
- `kit_commit.txt`, and `MANIFEST.json` with the SHA-256 of every file.

## 2. Queue the chunks

Write a jobs file, for example `kaggle/jobs_pilot.json`:

```json
{"kit_slug": "abdulhamidbatayhi/p3b-reference-kit", "nz": 2048,
 "jobs": [{"system": "A_primary", "set": "pilot_train", "of": 2}]}
```

Then run:

```bash
python kaggle/run_queue.py kaggle/jobs_pilot.json
```

How the queue works:
- Each kernel runs its chunk as four sub-chunks, one per Kaggle CPU core, single-threaded.
- It pushes at most five kernels at a time.
- It downloads every finished kernel into `data/reference/<system>/<set>/<kernel>/` and stops
  with `ALL_OUTPUTS_DOWNLOADED`.
- It is resumable: pushes are logged in `kaggle/registry.txt` and never repeated.
- Only an explicit "not found" counts as absent. Any other poll result is retried later.

## 3. Check what came back

```bash
python scripts/check_reference_gate.py data/reference/A_primary/pilot_train/<kernel>/out
```

Every `chunk_<K>_report.json` lists each trajectory with its status. Failures are reported,
never dropped. `data/reference/` is not tracked in git; the data are archived with the paper.
