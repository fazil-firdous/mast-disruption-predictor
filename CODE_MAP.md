# CODE_MAP

A plain-English tour of the repo. Read the files in the order below. Files marked stub only have a one-line docstring so far.

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
| `src/data.py` | done (tested on real files, not yet over HTTPS) | Load shot table and per-shot signals, with a local cache. |
| `src/labels.py` | done | Note-based and signal-based disruption labels (see below). |
| `src/features.py` | stub | Causal window features. |
| `src/dataset.py` | stub | Shot selection, window table, splits by shot. |
| `src/models.py` | stub | Physics baseline and ML model factories. |
| `src/alarms.py` | stub | Turn scores into a first-alarm time. |
| `src/evaluate.py` | stub | Shot-level metrics, curves, bootstrap CIs, plots. |
| `scripts/download.py` | done | CLI: `--n 50` random shots (seeded) or `--ids ...`; prints cached/downloaded/failed counts. |
| `scripts/inspect_shots.py` | done | Prints time step, time range, ip sign, missing signals for ~20 shots spread over campaigns. |
| `scripts/label_report.py` | done | Phase 2 report: label tables, timing, sensitivity, plots, review list. Writes `results/label_report/`. |
| `scripts/build_dataset.py`, `run_experiment.py` | stub | Later phases. |
| `notebooks/colab_pipeline.ipynb` | done (not yet run in Colab) | Clone, install, inspect, download, label report. |
| `tests/test_labels.py` | done | Detector, note parser and label switch tests on synthetic shots. |
| `tests/synthetic.py` | done | `make_shot(kind=...)` builds fake shots. |
| `tests/test_data.py` | done | Offline tests of the data layer using fake shots. |
| `tests/test_smoke.py` | done | One trivial test so pytest runs. |
| `tests/test_features.py`, `test_dataset.py`, `test_alarms.py` | stub | Filled in during later phases. |

## src/data.py in detail
- `load_shot_table(use_cache=True)` -> DataFrame, one row per shot. Reads `data/shot_table.parquet` if present, otherwise downloads it from mastapp.site and saves it.
- `load_shot(shot_id, use_cache=True, retries, backoff)` -> DataFrame with columns `time, ip, power_radiated, neutron_rates_total, power_nbi` (float64). Missing signals are NaN columns. Reads `data/raw/{shot_id}.parquet` if present, otherwise streams the zarr `summary` group, retries on errors, and saves the result. Raises if the group has no time or none of the four signals.
- `download_shots(ids, workers, retries, backoff)` -> dict with lists `cached`, `downloaded`, `failed`. Skips shots already on disk, downloads the rest in parallel with a progress bar, and appends failures (shot_id, reason) to `data/skipped.csv`.
- Private helpers: `_open_summary` (the only function that touches the network for signals, so tests replace it), `_summary_to_frame`, `_fetch_shot`, `_cache_path`, `_log_skipped`.

## src/labels.py in detail
- `note_label(comment)` -> bool. True if the operator comment contains "disrupt" and no negation ("no disrupt", "not disrupt", "non-disrupt", "didn't disrupt"). NaN or None gives False.
- `parse_note_time(comment)` -> seconds or NaN. Finds the first "at <number> ms|s" after the word "disrupt". Known to miss many comments.
- `median_filter(x, size=3)` -> array. Running median that keeps the same length.
- `detect_disruption(time, ip)` -> dict with `status` ("disrupted", "clean" or "unknown"), `detected`, `t_disrupt` (s), `quench_rate` (A/s), `peak_ip` (A). Steps: take |ip|, smooth, give up if the peak is under `MIN_PEAK_IP`, find the earliest sample (above `CQ_MIN_FRAC_OF_PEAK` of the peak) that loses `CQ_DROP_FRAC` of its value within `CQ_MAX_MS`, move to the start of that fall, and use a nearby Ip spike as the time if there is one.
- `final_label(note_flag, signal_flag, source=None)` -> bool. Applies `config.LABEL_SOURCE` ("signal", "note" or "both").

## scripts/label_report.py in detail
- `choose_shots` picks a seeded uniform sample plus extra note-disrupted shots. `label_one` labels one shot both ways. `fall_time_ms` measures how fast |Ip| falls from 90% to 10%. `sensitivity` re-runs the detector for several floor values. `plot_cases` draws the disagreement plots. `build_report` writes the markdown from the numbers.
- Run: `python scripts/label_report.py --n 400 --n-extra 150` (downloads what it needs), or `--ids-file results/label_report/sample_ids.json` to reuse the exact shots behind the committed report.

## Where to look for results
`results/label_report/report.md`, `per_shot.csv`, `review_list.csv`, three `disagreements_*.png`, `sample_ids.json`.
