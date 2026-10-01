# CODE_MAP

A plain-English tour of the repo. Read the files in the order below. Every file is currently a stub with a one-line docstring, except `src/config.py`.

## Reading order
1. `docs/SPEC.md` and `the project rules` - what to build and how to work.
2. `src/config.py` - every tunable number, each with a why-comment.
3. `src/data.py` -> `src/labels.py` -> `src/features.py` -> `src/dataset.py` -> `src/models.py` -> `src/alarms.py` -> `src/evaluate.py` - the pipeline in the order the data flows.
4. `scripts/` - thin command-line entry points that call `src/`.
5. `tests/` - synthetic-data tests (no network).

## Files
| File | Status | Purpose |
|---|---|---|
| `src/config.py` | done | Constants: URLs, window/horizon/threshold settings, seeds, paths. |
| `src/data.py` | stub | Load shot table and per-shot signals, with a local cache. |
| `src/labels.py` | stub | Note-based and signal-based disruption labels. |
| `src/features.py` | stub | Causal window features. |
| `src/dataset.py` | stub | Shot selection, window table, splits by shot. |
| `src/models.py` | stub | Physics baseline and ML model factories. |
| `src/alarms.py` | stub | Turn scores into a first-alarm time. |
| `src/evaluate.py` | stub | Shot-level metrics, curves, bootstrap CIs, plots. |
| `scripts/*.py` | stub | download, label report, build dataset, run experiment. |
| `tests/test_smoke.py` | done | One trivial test so pytest runs. |
| `tests/synthetic.py`, `tests/test_*.py` | stub | Filled in during later phases. |
