# CODE_MAP

A plain-English tour of the repo. Read the files in the order below. 

## Reading order
1. `docs/SPEC.md` and `the project rules` - what to build and how to work.
2. `src/config.py` - every tunable number, each with a why-comment.
3. `src/data.py` -> `src/labels.py` -> `src/features.py` -> `src/dataset.py` -> `src/alarms.py` -> `src/models.py` -> `src/evaluate.py` -> `src/experiment.py` - the pipeline in the order the data flows. `src/notify.py` is separate (Telegram).
4. `scripts/` - thin command-line entry points that call `src/`.
5. `tests/` - synthetic-data tests (no network).

## Files
| File | Status | Purpose |
|---|---|---|
| `src/config.py` | done | Constants: URLs, window/horizon/threshold settings, seeds, paths, Phase 4 window policy, model settings. |
| `src/data.py` | done (checked against the real servers by CI) | Load shot table and per-shot signals, with a local cache. |
| `src/labels.py` | done | Note-based and signal-based disruption labels. |
| `src/features.py` | done | Causal window features. |
| `src/dataset.py` | done (synthetic tests) | Shot pool, labels for every source, shot selection, window table, splits by shot. |
| `src/alarms.py` | done | Turn a score sequence into a first-alarm time. |
| `src/models.py` | done | Physics baseline B0 and the three ML models with their small grids. |
| `src/evaluate.py` | done | Shot outcomes, metrics, trade-off curve, threshold choice, bootstrap CIs, plots. |
| `src/experiment.py` | done | Tune on validation, test once, write `results/<exp>/`. Shared by `run_experiment.py` and `run_all.py`. |
| `src/notify.py` | done | Telegram messages and files. Prints only if no bot is configured. |
| `scripts/download.py` | done | CLI: `--n 50` random shots (seeded) or `--ids ...`. Checks the connection first. |
| `scripts/inspect_shots.py` | done | Time step, time range, ip sign, missing signals for ~20 shots spread over campaigns. |
| `scripts/label_report.py` | done | Phase 2 report. Writes `results/label_report/`. |
| `scripts/feature_check.py` | done | Phase 3 features on real shots, real-data causality check. Writes `results/features/`. |
| `scripts/check_live.py` | done | Loads the shot table and shot 11860 from the real servers. Run by CI. |
| `scripts/build_dataset.py` | done | `--scale small|full`: download, label, select, window, split. Writes `data/processed/windows.parquet` and `shots.parquet`. |
| `scripts/run_experiment.py` | done | `--exp E0..E7 [--split temporal] [--model hgb]`. |
| `scripts/run_all.py` | done | The whole pipeline with Telegram messages, `results/SUMMARY.md` and a results zip. |
| `deploy/setup_vps.sh`, `deploy/run_pipeline.sh` | done | One-time VPS setup (venv, `.env`, nightly cron) and the run wrapper (pull, test, run, send log). |
| `.github/workflows/ci.yml` | done | On every push: `pytest`, plus `check_live.py` on a fresh machine. |
| `notebooks/colab_pipeline.ipynb` | done (not yet run in Colab) | Clone, install, every phase in order. |
| `tests/synthetic.py` | done | `make_shot(kind=...)` builds fake shots. |
| `tests/test_data.py`, `test_labels.py`, `test_features.py` | done | Data layer, labels, causality and window rules. |
| `tests/test_dataset.py` | done | Labels and selection, no window at or after t_disrupt, horizon labels, near-quench flag, splits (no shot in two splits, stratified, temporal). |
| `tests/test_alarms.py` | done | K-in-a-row rule on hand-made sequences, fast version agrees with the loop. |
| `tests/test_evaluate.py` | done | Outcomes and rates, threshold choice, bootstrap, baseline rule, two tiny end-to-end experiment runs, Telegram without credentials. |
| `tests/__init__.py` | done | Empty. Makes `tests` a package so `from src import ...` works. |

## src/data.py in detail
- `load_shot_table(use_cache=True)` -> DataFrame, one row per shot. Reads `data/shot_table.parquet` if present, otherwise downloads it from mastapp.site and saves it.
- `load_shot(shot_id, use_cache=True, retries, backoff)` -> DataFrame with columns `time, ip, power_radiated, neutron_rates_total, power_nbi` (float64). Missing signals are NaN columns. Reads `data/raw/{shot_id}.parquet` if present, otherwise streams the zarr `summary` group, retries on errors (but not when the shot does not exist), and saves the result. Raises if the group is empty, has no time, or has none of the four signals. An empty group is retried, because zarr 3 returns one when the connection fails.
- `check_connection(shot_id=REFERENCE_SHOT)` -> (ok, message). Reads one shot straight from the server with no cache and no retries, so a blocked network fails fast.
- `download_shots(ids, workers, retries, backoff)` -> dict with lists `cached`, `downloaded`, `failed`. Skips shots already on disk, downloads the rest in parallel with a progress bar, and appends failures (shot_id, reason) to `data/skipped.csv`.
- `log_skipped(rows)` appends (shot_id, reason) rows to `data/skipped.csv`; `dataset.py` uses it too.
- Private helpers: `_open_summary` (the only function that touches the network for signals, so tests replace it), `_summary_to_frame`, `_fetch_shot`, `_cache_path`.

## src/labels.py in detail
- `note_label(comment)` -> bool. True if the operator comment contains "disrupt" and no negation ("no disrupt", "not disrupt", "non-disrupt", "didn't disrupt"). NaN or None gives False.
- `parse_note_time(comment)` -> seconds or NaN. Finds the first "at <number> [ms|s]" after the word "disrupt". A number without a unit counts as seconds below 1.5 and ms from 10 up.
- `median_filter(x, size=3)` -> array. Running median that keeps the same length.
- `detect_disruption(time, ip, min_frac_of_peak=None)` -> dict with `status` ("disrupted", "clean" or "unknown"), `detected`, `t_disrupt` (s), `quench_rate` (A/s), `peak_ip` (A). Steps: take |ip|, smooth, give up if the peak is under `MIN_PEAK_IP`, find the earliest sample (above `CQ_MIN_FRAC_OF_PEAK` of the peak) that loses `CQ_DROP_FRAC` of its value within `CQ_MAX_MS`, move to the start of that fall, and use a nearby Ip spike as the time if there is one.
- `final_label(note_flag, signal_flag, source=None)` -> bool. Applies `config.LABEL_SOURCE` ("signal", "note" or "both"; currently "both").
- `shot_label(time, ip, comment, source=None)` -> dict with `status`, `disrupted`, `t_disrupt`, `note`, `signal_status`, `t_quench` (detector time whatever the label). The one function later phases should call to label a shot. `unknown` shots are to be skipped.

## scripts/label_report.py in detail
- `choose_shots(table, n, seed, pool_ids)` -> sorted list of ids: every note-disrupted shot in the pool (at most n/2), filled up to n with random note-clean shots.
- `fall_time_ms(time, ip)` -> ms for |Ip| to go from 90% to 10% of peak on the final fall.
- `label_one(shot_id, table_row)` -> one row with both labels and trace measurements.
- `plot_cases(cases, path, title, show_fall)` draws |Ip| with the signal time (red) and note time (blue).
- `sensitivity(df)` re-runs the detector for several `CQ_MIN_FRAC_OF_PEAK` values. `final_counts` counts disrupted shots under each `LABEL_SOURCE`. `md_table` formats tables. `build_report` writes the markdown from the numbers.
- Run: `python scripts/label_report.py --n 400` (whole table as pool), `--pool-file ids.json` to sample from a list, or `--ids-file results/label_report/sample_ids.json` to reuse the exact shots behind the committed report.

## Where to look for results
`results/SUMMARY.md` and `results/E0` ... `results/E7` once `run_all.py` has run (each has `metrics.json`, `config.json`, plots). `results/features/summary.md` and `shot_11860.png` (Phase 3). `results/label_report/report.md`, `per_shot.csv`, `review_list.csv`, three `disagreements_*.png`, `rampdowns.png`, `sample_ids.json`.

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
- `DERIVED_INPUTS` and `features_without(signal)` -> feature list minus everything built from one signal (E5 ablation).

## src/dataset.py in detail
- `candidate_ids(table, n_clean)` -> every note-disrupted shot plus a random pool of note-clean shots, worth downloading.
- `label_shot(shot_id, comment, campaign)` -> one row with `status_/t_disrupt_` for each of "both", "signal", "note", plus `t_quench`, and `status`, `disrupted`, `t_disrupt` for the configured source.
- `label_shots(ids, table)` -> those rows for many shots; `unknown` shots are logged to `skipped.csv` and left out.
- `select_shots(shots, n_disrupted, n_clean)` -> seeded sample (None = take all).
- `stop_time(row)` -> when that shot's windows end. With `STOP_AT_QUENCH`, at the detector's quench for every shot.
- `build_windows(shots)` -> all windows of all shots; window-less shots are logged.
- `window_labels(t_end, t_disrupt, horizon_ms)` -> 1 inside the horizon, else 0 (the SPEC 8.3 rule for one shot).
- `labelled_windows(windows, shots, source, horizon_ms)` -> windows with `label`, `disrupted`, `t_disrupt`, `near_quench` for one label source and horizon; windows at or after `t_disrupt` dropped.
- `random_split(shots)` / `temporal_split(shots)` -> shot_id -> split name. Whole shots only, stratified by label.

## src/alarms.py in detail
- `first_alarm(scores, times, threshold, k)` -> time of the first window that is the k-th in a row at or above the threshold, or NaN. The plain version to read first.
- `alarm_level(scores, k)` -> smallest of the last k scores at each window. The rule fires exactly when this reaches the threshold.
- `alarm_times(scores, times, thresholds, k)` -> first-alarm time for many thresholds at once (used for curves); tested against `first_alarm`.

## src/models.py in detail
- `PhysicsBaseline(slope_quantile, ratio_quantile)` -> B0. `fit` sets the two thresholds from training clean windows; `predict_proba` gives 1 when |Ip| falls faster than the slope threshold or Prad/Pnbi is above the ratio threshold.
- `make_model(name, params)` -> unfitted "baseline", "logreg" (imputer + scaler + LR), "forest" (imputer + RF) or "hgb".
- `GRIDS`, `grid(name)` -> at most 12 parameter combinations per model. `scores(model, X)` -> P(disruption coming) per window.

## src/evaluate.py in detail
- `shot_alarm_table(windows, scores)` -> one row per shot and a shots x thresholds array of alarm times.
- `outcomes(shots, t_alarm)` -> caught / tardy / missed / false_alarm / quiet per shot, and warning times in ms.
- `shot_metrics(result, warning_ms)` -> rates and warning-time median and quartiles.
- `tradeoff_curve(shots, alarm_matrix)` -> metrics at every threshold. `choose_threshold(curve)` -> lowest threshold with <= 5% false alarms. `meets_limit(rate)`.
- `bootstrap_ci(result, warning_ms)` -> 95% intervals, resampling disrupted and clean shots separately.
- `window_metrics(y, scores, threshold)` -> precision, recall, F1, PR-AUC.
- Plots: `plot_tradeoff`, `plot_warning_hist`, `plot_importance`, `plot_example`.

## src/experiment.py in detail
- `tune(name, train, val, columns)` -> best grid combination by validation catch rate at the chosen threshold.
- `test_once(best, test, columns, out_dir)` -> test metrics with CIs, writes the trade-off, warning-time and example plots.
- `run_model(...)` -> tune, optionally test, permutation importance, `metrics.json`, `config.json`.
- `run(exp, split, model)` -> one experiment id. E0-E3 one model each; E4 horizon sweep, E5 signal ablation, E7 label source (all three on validation only, via `sweep`); E6 baseline and best model on the temporal split.

## src/notify.py in detail
- `send(text)` and `send_file(path, caption)` post to Telegram using `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` from the environment or `.env`. They never raise, so a network hiccup cannot kill a training run.
