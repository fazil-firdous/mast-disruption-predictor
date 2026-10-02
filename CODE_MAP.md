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
| `src/features.py` | done | Causal window features (see below). |
| `src/dataset.py` | stub | Shot selection, window table, splits by shot. |
| `src/models.py` | stub | Physics baseline and ML model factories. |
| `src/alarms.py` | stub | Turn scores into a first-alarm time. |
| `src/evaluate.py` | stub | Shot-level metrics, curves, bootstrap CIs, plots. |
| `scripts/download.py` | done | CLI: `--n 50` random shots (seeded) or `--ids ...`. Checks the connection first, then prints cached/downloaded/failed counts. |
| `scripts/inspect_shots.py` | done | Prints time step, time range, ip sign, missing signals for ~20 shots spread over campaigns. |
| `scripts/feature_check.py` | done | Phase 3: features on real shots, real-data causality check, plot of shot 11860. Writes `results/features/`. |
| `scripts/check_live.py` | done | Loads the shot table and shot 11860 from the real servers and checks them against SPEC 3.1. Run by CI. |
| `.github/workflows/ci.yml` | done | On every push: `pytest` on Python 3.12, plus `check_live.py` on a fresh machine with normal internet. |
| `scripts/label_report.py` | done | Phase 2 report: label tables, timing, sensitivity, plots, review list. Writes `results/label_report/`. |
| `scripts/build_dataset.py`, `run_experiment.py` | stub | Later phases. |
| `notebooks/colab_pipeline.ipynb` | done (not yet run in Colab) | Clone, install, inspect, download, label report. |
| `tests/test_labels.py` | done | Detector, note parser and label switch tests on synthetic shots. |
| `tests/synthetic.py` | done | `make_shot(kind=...)` builds fake shots. |
| `tests/test_data.py` | done | Offline tests of the data layer using fake shots. |
| `tests/test_smoke.py` | done | One trivial test so pytest runs. |
| `tests/__init__.py` | done | Empty. Makes `tests` a package so pytest puts the repo root on the path and `from src import ...` works. |
| `tests/test_features.py` | done | Causality, window bounds, plasma-off rule, missing signals, ratios. |
| `tests/test_dataset.py`, `test_alarms.py` | stub | Filled in during later phases. |

## src/data.py in detail
- `load_shot_table(use_cache=True)` -> DataFrame, one row per shot. Reads `data/shot_table.parquet` if present, otherwise downloads it from mastapp.site and saves it.
- `load_shot(shot_id, use_cache=True, retries, backoff)` -> DataFrame with columns `time, ip, power_radiated, neutron_rates_total, power_nbi` (float64). Missing signals are NaN columns. Reads `data/raw/{shot_id}.parquet` if present, otherwise streams the zarr `summary` group, retries on errors (but not when the shot does not exist), and saves the result. Raises if the group is empty, has no time, or has none of the four signals. An empty group is retried, because zarr 3 returns one when the connection fails.
- `check_connection(shot_id=REFERENCE_SHOT)` -> (ok, message). Reads one shot straight from the server with no cache and no retries, so a blocked network fails fast.
- `download_shots(ids, workers, retries, backoff)` -> dict with lists `cached`, `downloaded`, `failed`. Skips shots already on disk, downloads the rest in parallel with a progress bar, and appends failures (shot_id, reason) to `data/skipped.csv`.
- Private helpers: `_open_summary` (the only function that touches the network for signals, so tests replace it), `_summary_to_frame`, `_fetch_shot`, `_cache_path`, `_log_skipped`.

## src/labels.py in detail
- `note_label(comment)` -> bool. True if the operator comment contains "disrupt" and no negation ("no disrupt", "not disrupt", "non-disrupt", "didn't disrupt"). NaN or None gives False.
- `parse_note_time(comment)` -> seconds or NaN. Finds the first "at <number> [ms|s]" after the word "disrupt". A number without a unit counts as seconds below 1.5 and ms from 10 up.
- `median_filter(x, size=3)` -> array. Running median that keeps the same length.
- `detect_disruption(time, ip)` -> dict with `status` ("disrupted", "clean" or "unknown"), `detected`, `t_disrupt` (s), `quench_rate` (A/s), `peak_ip` (A). Steps: take |ip|, smooth, give up if the peak is under `MIN_PEAK_IP`, find the earliest sample (above `CQ_MIN_FRAC_OF_PEAK` of the peak) that loses `CQ_DROP_FRAC` of its value within `CQ_MAX_MS`, move to the start of that fall, and use a nearby Ip spike as the time if there is one.
- `final_label(note_flag, signal_flag, source=None)` -> bool. Applies `config.LABEL_SOURCE` ("signal", "note" or "both"; currently "both").
- `shot_label(time, ip, comment, source=None)` -> dict with `status`, `disrupted`, `t_disrupt`, `note`, `signal_status`. The one function later phases should call to label a shot. `unknown` shots are to be skipped.

## scripts/label_report.py in detail
- `choose_shots(table, n, seed, pool_ids)` -> sorted list of ids: every note-disrupted shot in the pool (at most n/2), filled up to n with random note-clean shots.
- `fall_time_ms(time, ip)` -> ms for |Ip| to go from 90% to 10% of peak on the final fall.
- `label_one(shot_id, table_row)` -> one row with both labels and trace measurements.
- `plot_cases(cases, path, title, show_fall)` draws |Ip| with the signal time (red) and note time (blue).
- `sensitivity(df)` re-runs the detector for several `CQ_MIN_FRAC_OF_PEAK` values. `final_counts` counts disrupted shots under each `LABEL_SOURCE`. `md_table` formats tables. `build_report` writes the markdown from the numbers.
- Run: `python scripts/label_report.py --n 400` (whole table as pool), `--pool-file ids.json` to sample from a list, or `--ids-file results/label_report/sample_ids.json` to reuse the exact shots behind the committed report.

## Where to look for results
`results/features/summary.md` and `shot_11860.png` (Phase 3). `results/label_report/report.md`, `per_shot.csv`, `review_list.csv`, three `disagreements_*.png`, `rampdowns.png`, `sample_ids.json`.

## src/features.py in detail
Read it top to bottom; each function uses only the ones above it.
- `SIGNALS` maps feature prefixes to shot columns: `ip_abs` (|ip|), `prad`, `neutrons`, `pnbi`. `STATS` = mean, std, last, slope, delta.
- `prepare_signals(shot_df)` -> DataFrame with `time` and the four signals (ip made absolute).
- `find_t_start(sig)` -> first time |Ip| >= IP_ON, or NaN (never, or already on at the first sample).
- `find_t_plasma_off(sig, t_start)` -> first time the plasma has ended (|Ip| < IP_ON, from t_start + WINDOW_MS on, after reaching MIN_PEAK_IP), or +inf.
- `window_features(sig, t_end, t_start)` -> dict of 30 features from samples in (t_end - WINDOW_MS, t_end] plus causal running values. This is the function the causality tests target.
- `window_ends(sig, t_start, t_stop)` -> array of window end times (real sample times, every STRIDE_MS, before t_stop and before the plasma ends).
- `shot_windows(shot_df, t_stop=inf)` -> one row per window: `t_end`, `t_start`, then the features. Pass `t_stop = t_disrupt` for disrupted shots.
- `feature_names()` -> the 30 model input columns in a fixed order.
