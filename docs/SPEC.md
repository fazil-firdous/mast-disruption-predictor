# SPEC — MAST Plasma Disruption Predictor

Version 1. Lives at `docs/SPEC.md`. If code and spec disagree, the spec wins. If the spec is wrong about the data, fix the spec and log it in `NOTES.md`.

---

## 1. Goal

Build a machine-learning system that looks at a MAST tokamak plasma discharge ("shot") as it unfolds and raises an **alarm before a disruption happens**, using only information available up to that moment. Evaluate it honestly: how many disruptions it catches, how early, and how often it raises false alarms on shots that never disrupted.

This is a research project, not production software. Success means: a correct, leak-free pipeline, honest results (good or bad), clear plots, and code that can be explained line by line.

## 2. Physics background (for context and for the README)

A **tokamak** confines plasma (hydrogen isotopes heated to tens of millions of °C) in a doughnut-shaped magnetic cage. MAST is a *spherical* tokamak run by UKAEA at Culham (2000–2013). Its successor is MAST-U. The public FAIR-MAST dataset covers MAST campaigns M5–M9.

A **shot** is one plasma pulse, lasting a few hundred milliseconds on MAST. Typical phases:
- **Breakdown and ramp-up:** plasma current `Ip` rises from zero.
- **Flat-top:** `Ip` held roughly constant. This is where the physics experiments run.
- **Ramp-down:** a controlled, relatively slow decrease of `Ip` to zero. That's a normal ending.

A **disruption** is an uncontrolled loss of confinement. It usually has a **thermal quench** (the plasma loses its heat in well under a millisecond), often seen as a brief **spike in `Ip`**, followed by a **current quench** (`Ip` crashes to zero in a few milliseconds). In large machines like ITER, disruptions can seriously damage the machine, so predicting them early enough to act (mitigate, or steer the plasma back to safety) is an important open problem.

Common precursors, which is why these signals matter:
- **Radiated power rising** relative to heating power (impurities cooling the plasma, radiative collapse).
- **Neutron rate falling** (the plasma core is losing performance).
- **Current evolution:** sudden dips, spikes, or loss of control.
- **Heating changes**, like neutral beams (NBI) switching off or on.
- Magnetic mode activity (locked modes), which is *not* in the verified signal set. See Phase 10.

## 3. Data sources

### 3.1 Verified (checked by the owner in Colab, Sept 2026)
- **Shot table** (one row per shot, 11,573 rows, ~189 columns):
  ```python
  pd.read_parquet("https://mastapp.site/parquet/level2/shots")
  ```
  Useful columns: `shot_id`, `campaign` (values like `"M5"`), `shot_postshot_comment`, `plasma_max_current`, `plasma_end_time`, `plasma_flat_top_start_time`, `plasma_flat_top_end_time`, `shot_flat_top_duration`, `shot_abort`, `shot_useful`.
- **Per-shot signals**, streamed anonymously:
  ```python
  xr.open_zarr(f"https://s3.echo.stfc.ac.uk/mast/level2/shots/{shot_id}.zarr", group="summary")
  ```
  Requires **`zarr>=3`**: the stores were re-ingested as Zarr v3 (`zarr.json` files, ingested 2026-09-22), which `zarr<3` cannot read (`KeyError: '.zmetadata'`). Corrected in Phase 2, see `NOTES.md`.
  Variables in the `summary` group (shot 11860):
  | name | meaning | units |
  |---|---|---|
  | `time` | time coordinate | s |
  | `ip` | plasma current (toroidal) | A |
  | `power_radiated` | radiated power | W |
  | `neutron_rates_total` | total neutron rate | Hz |
  | `power_nbi` | NBI power coupled to plasma | W |
  Shot 11860 has 385 samples from −0.100 s to 0.284 s: a uniform **1 ms** time base. Other shots may differ, so check.
  The `ip` attribute says positive = anticlockwise viewed from above. Sign can differ between shots, so **always work with `abs(ip)`**.
- Not every shot has every variable. Code must handle missing variables and missing shots.
- Licence: CC BY-SA 4.0. Credit "FAIR-MAST, UKAEA" in the README.
- Reference case: shot **11860**, note "DISRUPTION AT 220MS". `|Ip|` peaks near 700 kA, spikes to ~850 kA at ~0.210 s, then collapses to ~0 by ~0.213 s.

### 3.2 To verify before relying on them
- ~~Time base and sampling rate across campaigns.~~ Checked in Phase 1 (50 shots, M5–M9): uniform 1 ms everywhere, always starting at −0.100 s; record length varies (0.18–0.84 s end).
- ~~Units of `neutron_rates_total` and `power_nbi`~~ Checked in Phase 1: plausible SI values (Hz, W), no scaled shots seen. `power_nbi` has small negative sensor offsets; `neutron_rates_total` and `power_nbi` can have leading NaNs.
- ~~`shot_abort` / `shot_useful`~~ Checked: `shot_useful` is 1 or NaN (never 0), `shot_abort` is 1 for 39 shots. Decision: do not filter on either; shot selection uses the signal checks in 8.1.
- Campaign values present: M5, M6, M7, M8, M9 and `Unknown` (292 shots).
- Other groups (e.g. `amc`, `thomson_scattering`, magnetics) for Phase 10. Catalogue: https://mastapp.site.

## 4. Definitions

| Term | Definition |
|---|---|
| shot | one plasma discharge, identified by `shot_id` |
| `t_start` | first time `|Ip| >= IP_ON` (default 50 kA), i.e. plasma exists |
| `t_disrupt` | disruption time from the signal-based detector (Section 5.2), for shots labelled disrupted under `LABEL_SOURCE` |
| current quench (CQ) | the fast fall of `|Ip|` during a disruption |
| window | a short slice of a shot ending at time `t_end`. Features use samples in `(t_end − W, t_end]` plus causal running quantities |
| horizon `H` | windows with `t_disrupt − H <= t_end < t_disrupt` are labelled positive |
| score | model output in [0, 1] for one window |
| alarm | fires at the first `t_end` where score ≥ threshold for `K` consecutive windows |
| warning time | `t_disrupt − t_alarm` (ms) |
| caught | disrupted shot with alarm and warning time ≥ `MIN_WARNING` |
| tardy | disrupted shot with alarm but warning time < `MIN_WARNING` |
| missed | disrupted shot with no alarm before `t_disrupt` |
| false alarm | clean shot with any alarm at any time |

Default parameters (all in `src/config.py`):
```
IP_ON          = 50e3    # A
WINDOW_MS      = 20      # feature window length
STRIDE_MS      = 5       # step between window ends
HORIZON_MS     = 30      # positive-label horizon
K_CONSEC       = 2       # consecutive windows over threshold to alarm
MIN_WARNING_MS = 5       # minimum useful warning time
CQ_DROP_FRAC   = 0.8     # current must fall by this fraction of pre-quench value...
CQ_MAX_MS      = 10      # ...within this many ms to count as a quench
MIN_PEAK_IP    = 100e3   # ignore shots whose |Ip| never exceeds this
RANDOM_SEED    = 42
LABEL_SOURCE   = "both"  # decided after Phase 2, see 5.3
```
Detector parameters added in Phase 2 (see 5.2 and `NOTES.md`):
```
CQ_MIN_FRAC_OF_PEAK = 0.5   # quench must start above half of peak |Ip|
CQ_ONSET_FRAC       = 0.9   # onset = last sample within 90% of the level at the top of the fall
SPIKE_MIN_RISE      = 0.03  # spike must be 3% above the level just before it
SPIKE_LOOKBACK_MS   = 3
```

## 5. Labels

### 5.1 Note-based label (already working)
`disrupted_note = contains("disrupt") AND NOT matches(no disrupt | not disrupt | non-disrupt | didn.?t disrupt)`, case-insensitive, NaN → False. Result on the full table: 855 disrupted / 11,573 (7.4%).

Also parse a reported time where present (patterns like `AT 220MS`, `AT 0.22S`) into `t_note` in seconds. Leave it NaN otherwise.

### 5.2 Signal-based detector
Operator notes are incomplete and inconsistent, so the primary label comes from the current trace:

1. `x = |Ip|`, smoothed with a 3-sample median filter.
2. Skip the shot (label `unknown`) if `max(x) < MIN_PEAK_IP`.
3. For each candidate time `t` after the peak region, find where `x` falls by at least `CQ_DROP_FRAC` of its value at `t` within `CQ_MAX_MS`. Take the **earliest** such `t` as the current-quench start.
4. Set `t_disrupt` to the CQ start. If there's a local maximum (the `Ip` spike) within 3 ms before it, use that time instead, since the spike marks the thermal quench.
5. A controlled ramp-down is slow (tens of ms), so it won't meet the `CQ_MAX_MS` criterion. Confirm this on real ramp-downs and log it.
6. Output per shot: `detected` (bool), `t_disrupt` (s or NaN), `quench_rate` (A/s), `peak_ip` (A).

Phase 2 refinements (needed to pass the tests and found on real data): candidates must start above `CQ_MIN_FRAC_OF_PEAK` of the peak (otherwise the tail of every ramp-down qualifies); the CQ start is moved from the earliest candidate to the last sample within `CQ_ONSET_FRAC` of the top of the fall (the earliest candidate can be up to `CQ_MAX_MS` early); a spike must rise `SPIKE_MIN_RISE` above the preceding level.

### 5.3 Reconciliation report (Phase 2 deliverable)
Compare note labels and signal labels over the processed shots:
- 2×2 table: note yes/no vs signal yes/no.
- For shots with both `t_note` and `t_disrupt`: distribution of `t_disrupt − t_note` (ms).
- 6–10 plots of disagreement cases, both kinds.
- A short plain-language summary of which label is more trustworthy and why.

**Final label policy (decided by the owner after the Phase 2 report): `LABEL_SOURCE = "both"`.** A shot is disrupted only if the note says so **and** the detector finds a current quench; `t_disrupt` is the detector's time. Every other shot with a known signal status is clean, and shots with status `unknown` are skipped. Reason: the detector alone flags about 74% of note-clean shots, because most MAST pulses end in a fast quench; the notes alone have no reliable time. Implemented as `labels.shot_label()`. The switch still accepts `"signal"` and `"note"` (for E7). The 4 note-only cases are kept in `results/label_report/review_list.csv`.

Consequence for later phases: many clean shots also end in a fast current quench, so the last part of a clean shot looks like a disruption. Phase 3 must deal with this explicitly when building windows (see `NOTES.md`).

## 6. Leakage rules (read carefully)

Every feature for a window ending at `t_end` must be computable at `t_end` in real time:
- Only samples with `time <= t_end`.
- Running quantities (e.g. running max of `|Ip|`) are computed cumulatively up to `t_end`, never over the whole shot.
- Never use shot-table summary columns like `plasma_end_time`, `shot_flat_top_duration`, or `plasma_max_current` as features. They're computed after the shot ends and some directly encode the disruption. They may be used for *selecting* and *checking* shots, never as model inputs.
- Feature scaling (if any) is fitted on the training split only.
- Windows with `t_end >= t_disrupt` are **dropped**. After the disruption there's nothing to predict.

## 7. Features (per window)

Signals: `ip_abs`, `prad`, `neutrons`, `pnbi`. For each signal present, computed over the window samples:
- `mean`, `std`, `last` (value at `t_end`)
- `slope`: least-squares slope over the window, in units/s
- `delta`: `last − first` in the window

Cross-signal and running features:
- `ip_frac_of_running_max = ip_last / max(ip_abs[time <= t_end])`
- `prad_over_pnbi = prad_mean / (pnbi_mean + 1e5)`. The offset avoids blowups when beams are off. State this in the comments.
- `prad_over_ip = prad_mean / (ip_mean + 1e3)`
- `neutron_slope_rel = neutrons_slope / (neutrons_mean + 1)`
- `t_since_start = t_end − t_start` (s)
- `pnbi_on = pnbi_last > 1e5`

Missingness: if a signal is absent for the whole shot, its features are NaN and `has_<signal> = 0`. Models must handle NaN. HistGradientBoosting does natively. For RF and logistic regression, impute with training-set medians and keep the `has_` flags.

Only compute windows where `t_end >= t_start + WINDOW_MS`.

Phase 3 additions (all causal, see `NOTES.md`):
- Windows also stop once the plasma has ended: the first time from `t_start + WINDOW_MS` on that `|Ip| < IP_ON`, after `|Ip|` has once reached `MIN_PEAK_IP`. Both conditions are needed because of breakdown transients.
- `t_start` is undefined (no windows) if `|Ip| >= IP_ON` already at the first sample, because no breakdown is visible.
- `has_<signal>` = 1 if the signal has any finite sample at or before `t_end`. For a signal absent from the whole shot this is the same as the definition above, and it never looks ahead.
- `pnbi_on` is NaN when `pnbi_last` is NaN.
- Window sample set: `t_end - WINDOW_MS < time <= t_end`, so 20 samples on the 1 ms grid. `t_end` is always a real sample time.

## 8. Dataset construction

### 8.1 Shot selection
- Start with a balanced subset: `N_DISRUPTED = 300`, `N_CLEAN = 300`, sampled randomly (seeded) from shots that pass basic checks: `summary` group opens, `ip` present, `max|Ip| >= MIN_PEAK_IP`.
- Scale to all detected disruptions plus ~3000 clean shots once the pipeline is proven. Size is one config value.
- Log every skipped shot and the reason to `data/skipped.csv`.

### 8.2 Caching
- Download each shot's `summary` signals once. Save to `data/raw/{shot_id}.parquet` with columns `time, ip, power_radiated, neutron_rates_total, power_nbi` (NaN column if absent).
- Parallel download with `concurrent.futures.ThreadPoolExecutor` (default 8 workers), 3 retries with backoff, progress bar.
- Window table saved to `data/processed/windows.parquet`. Columns: `shot_id`, `t_end`, all features, `label`, `split`.

### 8.3 Window labels
- Clean shot: every window `label = 0`.
- Disrupted shot: `label = 1` if `t_disrupt − HORIZON <= t_end < t_disrupt`, `label = 0` if earlier, dropped if `t_end >= t_disrupt`.

### 8.4 Splits
- **Random-by-shot:** 60/20/20 train/val/test, stratified by shot label, seeded.
- **Temporal (Phase 7):** train on campaigns M5–M7, test on M8–M9. Shots with campaign `Unknown` are left out of the temporal split (they cannot be placed in time) but are used in the random split. This is a small version of the real open problem of generalising to new conditions.
- The test split is touched **once**, for final numbers. All tuning happens on validation.

## 9. Models

In order of complexity. Each gets the same interface: `fit(X_train, y_train)`, `predict_proba(X) -> scores`.

- **B0 — physics baseline:** alarm when `ip_slope` < a negative threshold, or `prad_over_pnbi` > a threshold. Tune thresholds on validation. Every ML model must beat this, or the README says it doesn't.
- **M1 — logistic regression** with class weights and standardisation (fit on train).
- **M2 — random forest** (`class_weight="balanced_subsample"`, ~300 trees, `min_samples_leaf` ≥ 5).
- **M3 — HistGradientBoostingClassifier** (`class_weight="balanced"`, early stopping on a validation fold).

Hyperparameters: a small grid (≤ 12 combinations per model) scored on **validation shot-level metrics**, not window accuracy.

## 10. Alarm logic and threshold choice

- For each shot, run the model over windows in time order and apply the `K_CONSEC` rule to get `t_alarm` (or none).
- Pick the threshold on the validation set so the **false-alarm rate on clean shots ≤ 5%**, then report catch rate at that operating point. Also report the full trade-off curve.

## 11. Evaluation

Never report accuracy alone. With ~93% clean windows, it's meaningless.

Window level (secondary): precision, recall, F1, PR-AUC.

Shot level (primary):
- **Catch rate** = caught / disrupted shots
- **Tardy rate**, **miss rate**
- **False-alarm rate** = clean shots with an alarm / clean shots
- **Warning time**: median and interquartile range over caught shots, plus a histogram
- **Trade-off curve**: catch rate vs false-alarm rate as the threshold sweeps from 0 to 1
- **95% confidence intervals** by bootstrapping over shots (1000 resamples)

Interpretation: permutation importance on validation, and 6 example shots (3 caught, 1 missed, 1 tardy, 1 false alarm) plotted with `|Ip|`, `prad`, the model score, threshold, `t_alarm`, and `t_disrupt` on shared time axes.

Outputs per experiment: `results/<exp_id>/metrics.json`, `config.json`, and PNGs.

## 12. Tests (pytest, synthetic data, no network)

`tests/synthetic.py` makes fake shots: ramp-up → flat-top → either a sharp quench at a known time (with an optional spike) or a slow ramp-down, plus noise.

Required tests:
- Detector finds a synthetic quench within ±2 ms of the true time.
- Detector does **not** flag a slow ramp-down.
- Detector returns `unknown` for a tiny-current shot.
- **Causality:** changing samples after `t_end` doesn't change that window's features (exact equality).
- No window from a disrupted shot has `t_end >= t_disrupt`.
- Labels: windows inside the horizon are 1, earlier ones 0.
- Splits: no `shot_id` appears in two splits.
- Alarm logic: `K_CONSEC` behaves correctly on a hand-made score sequence.
- Note parser: the negation cases and time parsing (`"DISRUPTION AT 220MS"` → 0.22).

## 13. Repository layout

```
mast-disruption/
  CONTRIBUTING.md
  README.md
  CODE_MAP.md
  NOTES.md
  requirements.txt
  .gitignore                 # data/, __pycache__/, .ipynb_checkpoints/
  docs/SPEC.md
  src/
    __init__.py
    config.py                # every parameter, each with a why-comment
    data.py                  # load_shot_table(), load_shot(shot_id), download_shots(ids)
    labels.py                # note_label(), parse_note_time(), detect_disruption(time, ip)
    features.py              # window_features(shot_df, t_end, t_start), shot_windows(shot_df, ...)
    dataset.py               # select_shots(), build_windows(), make_splits()
    models.py                # baseline + model factories
    alarms.py                # first_alarm(scores, times, thr, k)
    evaluate.py              # shot_metrics(), tradeoff_curve(), bootstrap_ci(), plots
  scripts/
    download.py              # cache raw shots
    label_report.py          # Phase 2 reconciliation report
    build_dataset.py
    run_experiment.py        # --exp E2 --split random|temporal
  notebooks/
    colab_pipeline.ipynb
  tests/
    synthetic.py
    test_labels.py
    test_features.py
    test_dataset.py
    test_alarms.py
  results/
```

## 14. Experiments

| ID | What | Why |
|---|---|---|
| E0 | Baseline B0 | the bar to beat |
| E1 | Logistic regression | simplest ML |
| E2 | Random forest | main model |
| E3 | HistGradientBoosting | usually strongest on tabular data |
| E4 | Horizon sweep H ∈ {10, 20, 30, 50} ms (best model) | trade-off between early warning and ambiguity |
| E5 | Signal ablation: drop one signal at a time | which physics signals matter |
| E6 | Temporal split M5–M7 → M8–M9 | does it generalise to later campaigns? |
| E7 | Note labels vs signal labels | how much label quality changes the result |
| E8 | Conformal threshold (ROADMAP B1) | a provable false-alarm limit, and where it breaks |
| E9 | Positive-unlabelled learning (ROADMAP A1) | treat unnoted quenches as unlabelled, not clean |
| E10 | Seed sweep: the same model on many random splits | how much of a result is luck of the split |
| E11 | Shortcut test: without `t_since_start` | is the model learning shot length instead of physics? |
| E12 | UKAEA-style protocol: current-only labels, 10/30/60 ms | like-for-like with the UKAEA FAIR-MAST benchmark |
| E13 | Add magnetics (locked-mode n=1, Mirnov), solenoid and current-control error | do physics precursors beat the shortcut? |

Research extensions beyond v1 are planned in `docs/ROADMAP.md`.

## 15. Out of scope for v1
Deep learning (LSTM, transformers), real-time deployment, MAST-U data, cross-machine transfer, profile/diagnostic groups beyond `summary`. These can be Phase 10+ once v1 is solid and understood.

## 16. Honest limitations to state in the README
Labels are algorithmic and imperfect. Only 4 global signals are used, with no magnetics or profiles. MAST (2000–2013), not MAST-U. Offline evaluation, not real-time control. Results come from a subset of shots unless stated otherwise.
