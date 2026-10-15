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
| `src/conformal.py` | built | Conformal alarm threshold with a provable false-alarm limit (E8, ROADMAP B1). |
| `src/pu.py` | built | Positive-unlabelled learning for unnoted quenches (E9, ROADMAP A1). |
| `docs/ROADMAP.md` | done | The agreed long-term research plan. Read it after SPEC. |
| `src/notify.py` | done | Telegram messages and files. Prints only if no bot is configured. |
| `scripts/download.py` | done | CLI: `--n 50` random shots (seeded) or `--ids ...`. Checks the connection first. |
| `scripts/inspect_shots.py` | done | Time step, time range, ip sign, missing signals for ~20 shots spread over campaigns. |
| `scripts/label_report.py` | done | Phase 2 report. Writes `results/label_report/`. |
| `scripts/feature_check.py` | done | Phase 3 features on real shots, real-data causality check. Writes `results/features/`. |
| `scripts/check_live.py` | done | Loads the shot table and shot 11860 from the real servers. Run by CI. |
| `scripts/build_dataset.py` | done | `--scale small|full`: download, label, select, window, split. Writes `data/processed/windows.parquet` and `shots.parquet`. |
| `scripts/run_experiment.py` | done | `--exp E0..E7 [--split temporal] [--model hgb]`. |
| `scripts/run_all.py` | done | The whole pipeline with Telegram messages, `results/SUMMARY.md` and a results zip. |
| `deploy/setup_vps.sh` | done | One-time VPS setup: venv, private `.env` (token, chat id, `MAST_HOME`), tests, network check, systemd service for the research server. |
| `scripts/research_server.py` | done | Runs forever: waits out outages, full run per new commit, E10 seeds when idle, daily heartbeat, Telegram commands from the owner only. |
| `src/server.py` | done | The research server's decisions and texts (state file, command parsing, what to run next), kept separate so they are testable. |
| `src/registry.py` | done | Save / load trained models with their feature list and threshold, in `MODELS_DIR` (private). |
| `src/predict.py`, `scripts/predict_shot.py` | done | Run a saved model over a whole shot as if live; plot score, alarm and the detector's quench. |
| `src/relay.py` | done | GitHub data relay, server side: start the fetch job, find and follow its run, download and safely unpack the artifact into `DATA_DIR`. |
| `scripts/relay_fetch.py`, `.github/workflows/fetch_data.yml` | done | GitHub data relay, GitHub side: download the shots a run needs (live shot table, or the 400 report shots if mastapp.site is down) and write `manifest.json`. |
| `src/extras.py` | done (real data checked) | Magnetics and control signals reduced causally to 1 ms: saddle-coil n=1 (locked mode), Mirnov RMS, programmed current, solenoid current. Cached per shot. |
| `src/sources.py` | done | Data-source watcher: shot-table status, FAIR-MAST folders and shot-store count, DEFUSE status, and what changed. |
| `src/literature.py` | done | Weekly arXiv search for new disruption-prediction papers; each reported once on Telegram. |
| `src/gold.py` | done | Blind hand-labelling on Telegram (/gold, /d /n /u) and agreement report (/goldreport). |
| `docs/literature_review.md`, `docs/paper/draft.md` | started | Checked references and what they mean for our claims; paper skeleton (numbers TBD from real runs). |
| `scripts/diagnose_network.py` | done | DNS, TCP per address, HTTPS and zarr checks, with a plain-language verdict; sent to Telegram. |
| `scripts/inspect_groups.py` | done | Catalogue of every signal stored for a shot (all groups) from the store metadata. Writes `results/data_catalogue/`. |
| `scripts/termination_study.py` | done (real data) | Programmed current and solenoid current at the quench, noted vs unnoted. Writes `results/termination_study/`. |
| `.github/workflows/ci.yml` | done | On every push: `pytest`, plus `check_live.py` on a fresh machine. |
| `notebooks/colab_pipeline.ipynb` | done (not yet run in Colab) | Clone, install, every phase in order. |
| `tests/synthetic.py` | done | `make_shot(kind=...)` builds fake shots. |
| `tests/test_data.py`, `test_labels.py`, `test_features.py` | done | Data layer, labels, causality and window rules. |
| `tests/test_dataset.py` | done | Labels and selection, no window at or after t_disrupt, horizon labels, near-quench flag, splits (no shot in two splits, stratified, temporal). |
| `tests/test_alarms.py` | done | K-in-a-row rule on hand-made sequences, fast version agrees with the loop. |
| `tests/test_evaluate.py` | done | Outcomes and rates, threshold choice, bootstrap, baseline rule, two tiny end-to-end experiment runs, Telegram without credentials. |
| `tests/test_research.py` | done | Conformal rank and guarantee (simulated), PU weights, premature rate, E8/E9 end to end. |
| `tests/test_server.py` | done | Server decisions, command parsing, Telegram parsing, `.env` loading, diagnosis verdicts, connection retries, catalogue. |
| `tests/__init__.py` | done | Empty. Makes `tests` a package so `from src import ...` works. |

## Where things are written
`src/config.py` reads the private `.env` (git-ignored) at import. `MAST_HOME` there decides where data, results, models and logs go (`DATA_DIR`, `RESULTS_DIR`, `MODELS_DIR`, `LOGS_DIR`). Unset, it is the current folder, as before.

## src/data.py in detail
- `load_shot_table(use_cache=True)` -> DataFrame, one row per shot. Reads `data/shot_table.parquet` if present, otherwise downloads it from mastapp.site and saves it.
- `load_shot(shot_id, use_cache=True, retries, backoff)` -> DataFrame with columns `time, ip, power_radiated, neutron_rates_total, power_nbi` (float64). Missing signals are NaN columns. Reads `data/raw/{shot_id}.parquet` if present, otherwise streams the zarr `summary` group, retries on errors (but not when the shot does not exist), and saves the result. Raises if the group is empty, has no time, or has none of the four signals. An empty group is retried, because zarr 3 returns one when the connection fails.
- `check_connection(shot_id=REFERENCE_SHOT, attempts=1)` -> (ok, message). Reads one shot straight from the server with no cache; big runs pass more attempts so one timeout does not kill them.
- `check_shot_table()` -> (ok, message). The shot table is cached or can be downloaded now.
- `open_group(shot_id, group)` -> xarray Dataset of any group (pulse_schedule, pf_active, magnetics, ...), not cached.
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

## src/conformal.py in detail
- `shot_alarm_levels(windows, scores)` -> per shot, the highest threshold at which it would still alarm.
- `conformal_threshold(clean_levels, alpha)` -> threshold so a new clean shot alarms with probability <= alpha (if shots are exchangeable). Uses the ceil((n+1)(1-alpha))-th smallest calibration level; +inf if there are too few shots.
- `min_calibration_shots(alpha)` -> how many clean calibration shots that needs (19 for 5%, 99 for 1%).

## src/pu.py in detail
- `fit_pu(params, train, holdout, columns)` -> (model, c, weights). Elkan & Noto: g learns "labelled positive vs rest", c = mean g on held-out positives, each unlabelled window is used as positive with weight w = (1-c)/c * g/(1-g) and as negative with weight 1-w. g for the weights is out-of-fold (grouped by shot).
- `label_frequency`, `unlabelled_weights` are the two formulas on their own, so they can be tested.

## E8 and E9 in src/experiment.py
- `run_conformal` (E8): tune on half the validation shots, calibrate on the other half's clean shots, report test false-alarm and catch rate for each alpha, next to the usual tuned threshold.
- `run_pu` (E9): standard vs PU model with identical settings, on all test shots and on the unambiguous ones (`unambiguous()`: disrupted shots plus shots with no quench at all). Also reports c and the mean weight of unnoted-quench windows, i.e. how disruption-like the model thinks they are.
- `premature_rate` (in `evaluate.shot_metrics`): share of disrupted shots caught more than `PREMATURE_WARNING_MS` early.

## The research server in detail
- `src/server.py`: `new_state`, `read_state`, `write_state` (the state file in `LOGS_DIR`), `parse_command('/predict 11860')`, `next_job(state, head, now, processed_ready)` -> 'pipeline' for every new commit (a failed one is retried after `PIPELINE_RETRY_HOURS`), else the next E10 seed, else nothing. `status_text`, `HELP`.
- `scripts/research_server.py`: `main` loops forever: pull code every `UPDATE_CHECK_MINUTES` (and restart itself on new code), heartbeat, pick the next job, wait out network outages (`wait_for_network`), run jobs as child processes (`run_job`) while still answering commands (`poll_commands` -> `handle_command`). A lock file stops a second copy.
- Commands come only from `TELEGRAM_CHAT_ID`, and each maps to a fixed action; nothing typed in the chat is run as a shell command.
- Data access (`server.access_mode`): 'direct' (STFC and shot table answer), 'direct-report' (STFC answers, mastapp.site down), 'relay' (STFC blocks this server and `GITHUB_TOKEN` is set), or wait. `server.pipeline_args` turns that into `run_all.py` options (`--offline`, `--from-report`). `relay_needed` fetches again only for a new scale or after `/fetch`.

## The GitHub data relay in detail
- `scripts/relay_fetch.py` runs on GitHub's machine: `choose_shots(scale)` -> (ids, 'live' or 'report'); downloads them; writes `data/manifest.json` (scale, table source, counts, commit, time).
- `src/relay.py` runs on the server: `trigger(scale)` -> request time; `find_run(requested_at)` / `pick_run(runs, requested_at)` -> the run that request started (GitHub gives no id back); `run_status(run_id)`; `download(run_id)` -> manifest; `unpack(zip, data_dir)` refuses entries that would land outside `data_dir`. The token is read from the environment at call time and is never part of `config` (so never in results).
- `build_dataset.build(..., offline=True)` and `run_all.py --offline` use only shots already on disk (`data.is_cached`).

## Models you can reuse
- `src/registry.py`: `save(model, tag, ...)`, `load(tag)` -> (model, meta), `saved_tags()`, `best_tag()`. E0-E3 and E6 save their tuned model (tag = experiment folder, e.g. `E3`, `E6_hgb`).
- `src/predict.py`: `predict(shot_df, model, meta)` -> (windows with score, t_alarm). `plot_prediction(...)`.
- E10 (`experiment.run_seed`, `seed_summary`): one more random split per seed; the summary gives mean and spread over all seeds so far.

## src/extras.py in detail
- `bin_index`, `bin_mean`, `bin_rms`: fast samples into 1 ms bins ending at each grid time (only past samples).
- `hold_last`: a slow signal held at its last value (programmed current, solenoid current).
- `early_offset`, `n1_amplitude`, `saddle_n1`: per-coil offset from its first 5 ms of data, then a cosine fit around the 12 coils at each time; NaN until the offset is known.
- `mirnov_rms`, `slow_signal`, `compute_extras`, `load_extras`, `download_extras`, `with_extras` (merges cached extras into a shot, NaN if none).
- In `features.py`: `EXTRA_SIGNALS` (lm, mirnov, sol) and `ip_ref_error`; `feature_names(extended=True)` adds 19 features; the 30 v1 features are unchanged.

## Watchers and the gold tool in detail
- `sources.probe()` -> snapshot dict; `sources.changes(old, new)` -> sentences for Telegram. On a blocked server the probe runs on GitHub (`relay_fetch.py --task probe`).
- `literature.search()`, `parse_feed(xml)`, `check()` (new papers only), `digest(papers)`.
- `gold.build_queue(shots)` (seeded, over-samples unnoted quenches), `next_item`, `record`, `plot_blind`, `evaluate`, `report_text`.
- Experiments E11 (without t_since_start), E12 (UKAEA-style), E13 (feature sets) are knobs of `experiment.sweep` via `sweep_settings`.
