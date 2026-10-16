# NOTES

Dated log of data surprises, decisions, and anything I was not sure about.

## 2026-10-02 - Phase 0: scaffold and network check

Environment
- The cloud sandbox ships Python 3.14, where `numcodecs` fails to build. I used a Python 3.11 virtualenv (`uv venv -p 3.11 .venv`, ignored by git).

Network check from the sandbox (original result, before the diagnosis below)
- DNS works for `mastapp.site` and `s3.echo.stfc.ac.uk`, and `https://example.com` returns 200.
- Direct HTTPS from the sandbox to both data hosts fails: `curl` gets "connection reset" during the TLS handshake (or times out) and `pd.read_parquet(<shot table url>)` gives `SSL: UNEXPECTED_EOF`. Retried over several minutes with the same result. This is a block on the sandbox's outbound connections to those two hosts. It is not fixed.
- So the sandbox still cannot run `load_shot_table()` or open shot 11860 over HTTPS by itself.

## 2026-10-02 - Phase 1/2 correction: the stores are Zarr v3, so `zarr<3` is wrong

- The first sandbox error `KeyError: '.zmetadata'` was NOT only the network. A cloud browser can reach both hosts. Listing the bucket shows the shot stores now use Zarr v3 (`zarr.json` in every group and array, no `.zgroup`, `.zarray` or `.zmetadata`). `summary/zarr.json` says `ingested_at: 2026-09-22T15:33:23Z`.
- `zarr<3` (2.18.7) cannot read these stores and fails with exactly that `KeyError`. `zarr>=3` (3.1.6 tested) reads them correctly.
- SPEC 3.1 said to pin `zarr<3` because zarr 3 opened the (then v2) stores as empty groups. The data has been re-ingested since that check. I changed `requirements.txt`, `the project rules` (dependency list and comment example) and `docs/SPEC.md` 3.1 to `zarr>=3`. This overrides the Phase 0 instruction to pin `zarr<3`. Please confirm.
- Not verified: opening the stores over HTTPS with zarr 3 from this sandbox (blocked). The code is tested on real Zarr v3 files read from a local path. First thing to check in Colab: `python scripts/inspect_shots.py --n 20`. If a Colab image has an old xarray or zarr, upgrade it.

How the real data got into the sandbox (be clear about this)
- The cloud browser fetched, byte for byte, each shot's `summary` group (the `zarr.json` files and the single chunk of `time`, `ip`, `power_radiated`, `neutron_rates_total`, `power_nbi`) from the real server and the 12.5 MB shot table from `mastapp.site`. These were written to `data/mirror/{shot_id}.zarr` (801 shots) and `data/shot_table.parquet`, both git-ignored. I added a tiny root `zarr.json` to each local copy.
- `src/data.py` then ran unchanged against the local copies by setting the env var `MAST_ZARR_URL="<path>/data/mirror/{shot_id}.zarr"`. Everything below that says "real data" means real FAIR-MAST bytes read through `load_shot`/`download_shots`, not an HTTPS download from the sandbox.
- Shot table: 11,573 rows, 189 columns, matching SPEC 3.1. `shot_id` runs 11766 to 30471, unique. Campaign counts: M5 1960, M6 2697, M7 3514, M8 2009, M9 1101, `Unknown` 292 (not in the SPEC; the temporal split in Phase 7 must decide what to do with `Unknown`). `plasma_max_current` appears to be in kA (shot 11860: 704.7), not A.

## 2026-10-02 - Phase 1: data layer and inspection

Runs (real data via the local copies)
- `download_shots` on 50 random shots (10 per campaign M5 to M9, seeded): 50 downloaded, 0 failed. A second run: 50 cached, 0 downloaded. Later all 801 shots loaded with 0 failures, so `data/skipped.csv` was never created by a real run. (Earlier, a real network failure from the sandbox was logged there correctly.)
- Offline tests cover cache hits, NaN columns, retries, failure logging, the empty-group error, and reading a real Zarr v3 store.

Inspection of 20 shots (4 per campaign M5 to M9, `scripts/inspect_shots.py --n 20 --cached-only`) plus checks over the 50 Phase 1 shots
| item | finding |
|---|---|
| time step | 1.000 ms, uniform and increasing, in all 50 shots of all five campaigns |
| time range | always starts at -0.100 s. Ends between 0.18 s and 0.84 s. Samples per shot: median 523 (M5), 575 (M6), 516 (M7), 683 (M8), 860 (M9). Later campaigns have longer records |
| sign of `ip` | positive at the peak in 49 of 50 shots. Shot 13611 (M5) is negative, so `abs(ip)` is needed (confirmed). Small negative `ip` values (a few percent of samples) occur after the current falls |
| pre-shot `ip` | median |ip| before -0.05 s is 6.8 kA, max 29 kA. Well below `IP_ON` = 50 kA |
| variables missing for the whole shot | `power_radiated` 5 of 50, `neutron_rates_total` 3, `power_nbi` 4, `ip` 0. Extra shots seen later: 25424 has only `ip` and `time`, 25809 has only `ip` and `power_radiated` |
| leading NaNs | `neutron_rates_total` is NaN for the first 3 samples (-0.100 to -0.098 s) in 38 shots. `power_nbi` is NaN for the first 91 samples (-0.100 to -0.010 s) in 21 shots. No NaNs in the middle or at the end. `ip` has no NaNs |
| units / scale | `power_radiated` median max 2.2 MW, `power_nbi` median max 1.8 MW (up to 3.9 MW, M8 highest), `neutron_rates_total` median max 4.5e13 Hz. All plausible in SI units, no sign of scaled values. 5th percentile of max neutron rate is 1.9e10 and some shots have max 0 |
| odd | `power_nbi` has small negative values (min -7.6 kW) in 42 of 50 shots, i.e. sensor offset around zero. `pnbi_on = pnbi_last > 1e5` in SPEC 7 is unaffected. M9 peak currents are lower (median 464 kA vs about 800 kA elsewhere) |
| shot 11860 | 385 samples from -0.100 to 0.284 s, ip +853 kA at 0.210 s, 679 kA at 0.205 s and 46 kA at 0.213 s, -18 kA at 0.214 s. Matches SPEC 3.1 (spike near 850 kA, ~0 by ~0.213 s) |

Decisions
- A summary group with no `time` or none of the four signals is a load failure, not NaN columns. This catches wrong zarr versions loudly.
- A variable whose length differs from `time` is treated as missing (not seen in the 801 shots).
- `xr.open_zarr(..., consolidated=False)`: avoids the consolidated-metadata fallback warning and works on local copies.
- Retries wait 2, 4, 8 s (`RETRY_BACKOFF_S`). Failures are appended to `data/skipped.csv` from the main thread only.
- `shot_useful` is 1.0 for 6,589 shots and NaN for 4,984, never 0. So "exclude `shot_useful == False`" would drop nothing and "keep only useful" would drop 43%. `shot_abort` is 1.0 for 39 shots, NaN otherwise. Decision for now: do not filter on either; the signal checks (`max|Ip| >= MIN_PEAK_IP`) already select shots. Revisit if aborted shots turn out to be a problem.

## 2026-10-02 - Phase 2: labels and reconciliation report (STOP FOR REVIEW)

Implemented `src/labels.py` (`note_label`, `parse_note_time`, `median_filter`, `detect_disruption`, `final_label`) and `scripts/label_report.py`. 42 tests pass, including all SPEC 12 label tests (quench found within +-2 ms with and without spike and for both ip signs, slow ramp-down not flagged, tiny current gives `unknown`, note negations, `"DISRUPTION AT 220MS"` -> 0.22).

Checks against the SPEC
- Note label on the full table: 855 disrupted of 11,573 (7.4%), exactly as SPEC 5.1 states.

Detector judgement calls (SPEC 5.2 left room)
- SPEC step 3 ("earliest t where |Ip| falls 80% within 10 ms") fires up to `CQ_MAX_MS` too early, because every sample in the 10 ms before a quench also qualifies. I take the earliest candidate, find where the fall ends, then move to the last sample still within 90% of the level at the top of the fall (`CQ_ONSET_FRAC`). The spike rule (step 4) is applied after that, and only if the spike is 3% above the level just before it (`SPIKE_MIN_RISE`). Without this the synthetic test missed by more than 2 ms.
- A candidate must start above `CQ_MIN_FRAC_OF_PEAK` = 0.5 of peak current. Reason: the last stretch of any linear ramp-down also falls 80% in 10 ms once |Ip| is small (my first synthetic ramp-down test failed at 0.2). Cost: a disruption that starts below half of peak current is not found (see note-only cases). On real data the floor is also what keeps some ramp-downs unflagged: shot 28104 ramps down slowly and then drops from about 350 kA (44% of peak) to 0 in a few ms.
- NaN samples in `ip` are dropped before detection.
- `LABEL_SOURCE = "both"` means disrupted only when the note and the detector agree. The SPEC did not define it.
- `parse_note_time`: first "at <number> [ms|s]" after the word "disrupt". Unit-less numbers are read as seconds below 1.5 and ms from 10 up, otherwise NaN. I added this after the timing plots showed "disrupts at 0.302 broken pellet at 0.263s" being read as 0.263. Times found in note-disrupted comments: 379 before this change, 467 after (the 2026-10-02 fix-up below removed some false matches). It can still pick the wrong number when the comment gives a cause time (e.g. 25659 "Disrupted from FA trip at 0.18s", quench at 0.259 s).

Report run (all numbers from `results/label_report/report.md`)
- Sample: 400 shots = all 175 note-disrupted shots in the pool + 225 random note-clean shots (seed 42). The pool is the 801 shots copied so far (`data/pool_ids.json`, built from random and note-disrupted draws). In Colab the default pool is the whole shot table. The exact ids are in `results/label_report/sample_ids.json`.
- 2x2: note yes and signal yes 171, note yes and signal no 4, note no and signal yes 167, note no and signal no 58. Agreement overall 57.2% (inflated toward note-yes by the sampling).
- Notes are confirmed by the signal: 97.7% of note-disrupted shots show a current quench.
- The detector flags 74.2% of note-clean shots. Flagged shots look the same with and without a note: median quench rate 2.0e8 vs 1.9e8 A/s, peak 781 vs 764 kA, 90%->10% fall 4 vs 5 ms. Unflagged shots fall in 124 ms (median). DATA SURPRISE: most MAST shots, including ones the operator called fine ("OK", "Very similar to previous shot"), end with a fast current quench rather than a slow ramp-down to zero. Earlier (uniform sample of 400) the median flagged quench was 5.4 ms before the shot table's `plasma_end_time` with or without a note.
- Timing (91 shots with both times): `t_disrupt - t_note` median +1.0 ms, quartiles -2.0 to +5.0 ms, 78% within 10 ms, 6 differ by more than 20 ms (mostly the note gives a cause time, e.g. "FA trip at 0.18s", or a time unrelated to the quench).
- Ramp-downs: 3 random note-clean shots with a slow end (21275, 26320, 26613; 101 to 157 ms) are not flagged (`rampdowns.png`). 2 of 62 unflagged shots fall faster than 50 ms; borderline.
- 4 note-disrupted shots are not flagged (`review_list.csv`): 15555, 19388, 21673, 25108. All disrupt at low current or during the ramp-down (below the 0.5 floor).
- Sensitivity: share of note-clean shots flagged is 88% / 84% / 74% / 64% for floor 0.2 / 0.35 / 0.5 / 0.65. Note-disrupted: 99% / 98% / 98% / 96%. No floor value separates the two groups.

Decisions waiting for the owner
1. ~~`LABEL_SOURCE`~~ Decided 2026-10-02 by the owner: `"both"`. A shot is disrupted only when the note says so AND the detector finds a quench; `t_disrupt` comes from the detector. On the report sample that is 171 shots. Shots where only one source says disrupted (167 detector-only, 4 note-only) count as not disrupted. Consequence to remember for Phase 3: many "clean" shots still end in a fast quench, so the end of every shot must be handled carefully when building windows.
2. ~~`Unknown` campaign~~ Decided (2026-10-02, owner asked me to decide): left out of the temporal split, used in the random split. They cannot be placed before or after M7, so putting them on either side would blur the "later campaigns" test.
3. ~~`zarr>=3`~~ Decided: keep `zarr>=3,<4` (it is the only version that can read the stores), plus `xarray>=2024.10` (first versions with zarr 3 support).

Not done / not verified
- `notebooks/colab_pipeline.ipynb` has not been run in Colab.
- No HTTPS download of shot data from the sandbox has worked (see the network sections).

## 2026-10-02 - Fix-up before Phase 3

The owner asked me to fix the open items myself before Phase 3. What changed and why:
- **Note time parser.** "at 0.169delay" was read as 0 s because the regex backtracked to "at 0" (shot 22760). The number now has to be followed by a space, a unit or punctuation, and times of 0 or below are NaN. 2 new tests.
- **Retries.** A shot that does not exist (`FileNotFoundError`, which includes zarr's `GroupNotFoundError`) is no longer retried with 2 + 4 + 8 s waits.
- **DATA/NETWORK SURPRISE.** With the network blocked, zarr 3 does not raise an error: `xr.open_zarr` waits about 11 s and returns an EMPTY dataset (checked from the sandbox on shot 11860). So an empty group stays retryable and is reported as "empty or has no time (network blocked, or zarr<3 installed?)". Treating it as permanent would silently lose shots on a flaky connection.
- **`check_connection()`** in `src/data.py` reads the reference shot once. `scripts/download.py` calls it first and stops with a clear message if it fails (`--skip-check` to bypass). The notebook calls it too. Checked from the sandbox: it stops after one attempt with that message.
- **HTTP path checked.** I served the copied stores and the shot table from a local web server and ran the real HTTP code (`fsspec` + `aiohttp` + zarr 3): shot table 11,573 x 189 rows read over HTTP, 40 shots downloaded and identical to the cached files, a missing shot logged to `skipped.csv`. Still not checked: the real HTTPS servers from this sandbox (blocked).
- **`labels.shot_label(time, ip, comment)`** applies `LABEL_SOURCE` to one shot and returns status, `disrupted` and `t_disrupt`. With `"note"` it falls back to the note time when the detector finds nothing. A disrupted shot with no usable time becomes `unknown` (cannot be windowed). 3 new tests.
- **`LABEL_SOURCE = "both"`** written into SPEC 4 and 5.3, with the reason. The detector parameters added in Phase 2 are now listed in SPEC 4 and 5.2. SPEC 3.2's "to verify" items are marked done with what was found. SPEC 8.4 says what happens to `Unknown`.
- **Report.** It now shows the disrupted count under each `LABEL_SOURCE` (both 171, signal 338, note 175 of 400). The |Ip| trace is grey so the red detector line is visible. Re-run after the parser fix; only the timing line changed (91 shots, 6 over 20 ms).
- **Small fixes.** `inspect_shots.py --n 20` now really gives 20 shots when the campaign count does not divide 20. Removed an empty root `conftest.py` (`tests/__init__.py` already puts the repo root on the path). Test writes zarr without consolidated metadata, so `pytest -W error` runs with no warnings. 49 tests pass.

Still open, cannot be fixed from here
- The notebook has not been run in Colab, and the real HTTPS servers cannot be reached from the sandbox. The first Colab run (connection-check cell) is the real test.

## 2026-10-02 - Live check of shot 11860, and a loader bug it found

- I could not run Colab myself (no logged-in browser is available). The closest equivalent is a fresh cloud machine with normal internet. So I added a GitHub Actions workflow (`.github/workflows/ci.yml`) that does what the notebook does: clean checkout, `pip install -r requirements.txt` on Python 3.12, then `scripts/check_live.py` against the real servers. It also runs `pytest`.
- First run: tests passed, live check FAILED. The `summary` group opened empty after 1.7 s.
- Cause (my bug from Phase 1): `xr.open_zarr(..., consolidated=False)`. The real stores list their arrays only in the root `zarr.json` (`consolidated_metadata`, 352 entries for 11860, including `summary/ip`, `summary/time`). S3 over plain HTTPS cannot list folders, so without consolidated metadata zarr finds no arrays. My local HTTP test had passed only because Python's `http.server` serves folder listings, which S3 does not. A debug run on GitHub Actions tried each option against the live server: `consolidated=False` gives 0 variables; `consolidated=True` or `None` gives all 4.
- Fix: `consolidated=True` in `src/data.py`. New regression test `test_open_summary_uses_root_consolidated_metadata` (it fails with the old setting, checked). The local copies in `data/mirror` got the same consolidated root metadata (`zarr.consolidate_metadata`); all 801 cached shots re-read identical.
- Second CI run (zarr 3.4.0, xarray 2026.9.0): shot table 11573 x 189; shot 11860 has 385 samples from -0.100 to 0.284 s at a 1 ms step; max |Ip| 853 kA at 0.210 s; 46 kA at 0.213 s; all four signals present. All SPEC 3.1 checks PASS. **Shot 11860 loads from the real servers on a fresh machine.** Colab itself is still untested; it is the same code path.

## 2026-10-02 - Phase 3: causal window features

Implemented `src/features.py` (`prepare_signals`, `find_t_start`, `find_t_plasma_off`, `window_features`, `window_ends`, `shot_windows`, `feature_names`) per SPEC 6 and 7, and `scripts/feature_check.py`. 30 features per window. 69 tests pass.

Tests (SPEC 12 causality and more)
- Causality: random huge values written after `t_end` leave that window's features exactly equal (4 window ends tested). A whole-shot version: tripling |Ip| after 0.15 s leaves all earlier rows identical.
- Planted leaks are caught: a 1 ms look-ahead in the window mask fails 5 tests; a whole-shot running max fails 6.
- Also tested: exactly 20 samples per window, slope in units/s, ip sign has no effect, windows start at `t_start + WINDOW_MS` and step by `STRIDE_MS`, nothing at or after `t_stop`, windows stop when the plasma ends, missing signal gives NaN features and `has_` = 0, running max ignores a later spike, ratio offsets, a tiny shot gives no windows, breakdown transients.

Judgement calls
- **Plasma-off rule.** SPEC 7 does not say when windows end for a clean shot. With no plasma there is nothing to predict, so windows stop the first time `|Ip| < IP_ON`. Two guards were needed, both found on real data:
  (1) 14195: |Ip| touches 58 kA at breakdown, then hovers at 30-50 kA for tens of ms before rising to 743 kA. So the plasma only counts as "ended" after |Ip| has once reached `MIN_PEAK_IP`.
  (2) 11826, 16568, 16593, 20749, 21014: within ~5 ms of `t_start`, |Ip| spikes past 100 kA and dips to about 10 kA for one or two samples, then ramps up. So the search starts at the first possible window end, `t_start + WINDOW_MS`.
  With both, windows end within a few ms of the shot table's `plasma_end_time` (median 3 ms before for clean shots; 8 ms for disrupted shots, which stop at `t_disrupt`). The largest gap is 26842 (69 ms): there the current really collapses to ~30 kA at 0.12 s ("No P4 - plasma expands out"), so stopping is right.
- **Current already on at the first sample.** 25424, 25470 (and 25523 among the cached shots), all M8: |Ip| is 260-365 kA at -0.100 s and stays almost flat for over 2 s ("OK.", "TF and P4 OK."). These are not normal MAST pulses (probably engineering or test shots; I can't confirm). With no breakdown there is no `t_start`, so they get no windows. Phase 4 should log them in `skipped.csv`.
- **`has_<signal>`** uses samples up to `t_end` only (strictly causal; same as the SPEC for wholly missing signals).
- Window = `t_end - 20 ms < time <= t_end`. `t_end` is snapped to a real sample. Time comparisons use a 1e-9 s slack because the stored times are floats like 0.21000000000000002.

Run on the 400 report shots (`results/features/summary.md`)
- 26,360 windows in about 17 s. 0 infinite values. 2 shots without windows (25424, 25470). Median windows per shot: 74 clean, 55 disrupted.
- Real-data causality: each shot cut at a random window end and recomputed from the cut data only. 12,986 rows compared, all identical.
- Missing values (share of windows): prad 19.7%, pnbi 9.1%, neutrons 4.2%, prad/pnbi 26.9%. These come from whole-shot absence: 80, 36 and 20 of 399 shots have no prad, pnbi or neutrons.
- DATA NOTE: `neutron_rates_total` is exactly 0 for the whole shot in 59 of the 760 cached shots that have it (11860 included). In all 59, the beams never went above 100 kW (11860 peaks at 16 W). Ohmic plasmas make few neutrons, so I keep the zeros as real values, not as missing.

Input for Phase 4 (not decided here; `shot_windows` takes `t_stop` so either option is possible)
- Under `LABEL_SOURCE = "both"`, 167 of the 229 clean shots still end in a fast current quench that nobody noted. Their windows give 1,002 clean windows in the 30 ms before that quench. That is about as many as the 1,022 positive windows from the 171 disrupted shots, and the two kinds look alike on these signals.
- The same clean shots also keep 189 windows at or after their quench, while disrupted shots stop at `t_disrupt`. A model could learn "the quench is under way, so it's clean", which is the wrong lesson.
- Suggestion: stop every shot's windows at the detector's quench time (`t_stop` = detector `t_disrupt` whenever a quench is found), so both classes end the same way. The 1,002 near-quench clean windows would remain; whether to keep them (honest, hard), drop them, or weight them is the main Phase 4 decision.

## 2026-10-15 - Phases 4 to 7 in one go, plus the VPS runner (owner's instruction)

The owner asked me to finish the whole pipeline without stopping between phases, then run it on a VPS with Telegram updates. So the "one phase at a time / STOP FOR REVIEW" rule was waived for this session at the owner's request.

Network: this sandbox still cannot reach `mastapp.site`, `s3.echo.stfc.ac.uk` or `api.telegram.org` (connection reset), and the 801-shot local copy from earlier sessions is gone (fresh container, data is never committed). **So every piece of code below is tested only on synthetic shots. No real-data model result exists yet.** The first real numbers will come from `scripts/run_all.py` on the VPS.

Cleanups from the code review
- `detect_disruption` takes an optional `min_frac_of_peak`, so `label_report.sensitivity()` no longer changes `config` while it runs.
- The unexplained `+ 10` ms in the spike rule is now `config.SPIKE_BASELINE_MS`.
- `feature_check.py` passes `t_stop` as an argument instead of hiding it in `DataFrame.attrs`; `lambda` assignment and unclosed `json.load(open(...))` removed; `tests/test_smoke.py` removed (real tests exist).
- `data._log_skipped` is now `data.log_skipped`, because `dataset.py` logs unknown and window-less shots there too.
- Known small inconsistency, left as is on purpose: the detector finds the plasma start on the median-filtered |Ip|, `features.find_t_start` on the raw |Ip|. They can differ by one sample. Changing the detector would change the Phase 2 labels, and the detector's `t_start` is only used to skip the breakdown when searching for a quench.

Phase 4 decision (owner said "leave everything", so I took my recommendation from the Phase 3 notes)
- `STOP_AT_QUENCH = True`: every shot's windows end at the detector's quench, disrupted or clean. Before, clean shots kept windows during their own quench while disrupted shots stopped, which would teach "quench under way = clean".
- `NEAR_QUENCH_CLEAN = "keep"`: clean windows in the 30 ms before an unnoted quench stay in training as label 0. Honest and hard. Setting it to `"drop"` removes them from training only; evaluation always keeps every window, otherwise we would hide exactly the cases that cause false alarms.
- Shot pool: every note-disrupted shot (under "both" a disruption needs a note) plus 1.5x the needed clean shots at random, then label, then sample `N_DISRUPTED` / `N_CLEAN`. `--scale full` takes every disruption and `FULL_N_CLEAN = 3000` clean shots (SPEC 8.1).
- The window table stores labels for all three `LABEL_SOURCE`s, so E7 relabels without rebuilding.

Phase 5-7 judgement calls
- Alarm time = end of the K-th window over threshold. For threshold sweeps, `alarms.alarm_times` uses "smallest of the last K scores >= threshold", which is the same rule; a test checks it against the plain loop on 50 random sequences.
- Caught = alarm at least `MIN_WARNING_MS` before `t_disrupt`, however early (SPEC 4). Very early alarms are not penalised; worth revisiting.
- Threshold: lowest one with validation false-alarm rate <= 5%. If no threshold reaches that, the one with the fewest false alarms is used and `meets_false_alarm_limit` is false in `metrics.json` (found in the synthetic rehearsal: B0's 0/1 scores still fire at threshold 1.0).
- Grid choice: the combination that meets the limit with the highest validation catch rate, PR-AUC breaks ties.
- B0 thresholds are quantiles of the training clean windows (e.g. a slope lower than 99.9% of normal windows), so no hand-picked physical numbers that might not fit MAST.
- HistGradientBoosting's early stopping holds out 10% of training windows at random. Windows from one shot can then sit on both sides, which only affects when it stops, never val or test.
- E4, E5 and E7 compare settings on **validation** only; the test split is used once per main model (E0-E3) and in E6 (SPEC 8.4).
- Added `src/experiment.py` (not in SPEC 13) so `run_experiment.py` and `run_all.py` share one implementation.

Synthetic rehearsal (300 fake shots: noted quench with a radiation rise, unnoted quench, slow ramp-down): `run_all.py` ran E0-E7 with 0 failures in under 5 minutes on this sandbox. The numbers are meaningless for physics and are not reported anywhere.

VPS and Telegram
- `src/notify.py` sends messages and files through the Telegram Bot API with `requests` (no new dependency). Token and chat id come from `.env` (git-ignored) or the environment. A failed message never stops a run.
- `deploy/setup_vps.sh` installs everything, writes `.env`, sends a test message, runs pytest and adds a nightly cron job. `deploy/run_pipeline.sh` pulls, runs pytest, runs `run_all.py --scale full`, sends the log, and skips commits that already ran successfully.

## 2026-10-15 - Research track started (docs/ROADMAP.md)

The owner wants this treated as long-term, research-grade work aimed at a career in fusion, not a school project. `the project rules` now says so and points to `docs/ROADMAP.md`, the agreed plan. Readability rules stay: the owner studies the code daily and explains it back.

Literature check (two quick web searches, not a review): a UKAEA group already has a FAIR-MAST disruption benchmark (stacked BiLSTM, 417 shots, labels auto-detected from the plasma current; IAEA FEC poster). No tokamak disruption work using PU learning or conformal false-alarm guarantees turned up. Claims of novelty must wait for a proper review (ROADMAP G1).

Built (tested on synthetic shots only; no real-data numbers yet):
- **C1 premature alarms.** `premature_rate` = share of disrupted shots caught more than `PREMATURE_WARNING_MS` = 200 ms early. They still count as caught (SPEC 4). 200 ms is a judgement call: MAST flat-tops last ~200-300 ms.
- **B1 conformal threshold (E8).** Validation shots are split in half by shot: one half tunes the model, the other half's clean shots calibrate the threshold. Using the same shots for both would break the exchangeability argument. The guarantee is marginal (on average over calibration sets), so one test set can exceed alpha by chance; `guarantee_holds` in `conformal.csv` is an empirical check, not the theorem. Needs at least 19 clean calibration shots for 5% and 99 for 1%; otherwise the threshold is +inf (never alarm). Expected to break on the temporal split, which is the point of running it there.
- **A1 PU learning (E9).** Elkan & Noto (2008). Assumes noted disruptions are a random sample of all disruptions (SCAR); operators probably noted the obvious ones more often, so c may be biased high. Weights use 5-fold out-of-fold scores grouped by shot, because in-sample scores of a boosted model are over-confident. Fixed HGB settings (`PU_PARAMS`) for both standard and PU models, so the only difference is the labels. Evaluated on all test shots and on unambiguous ones (disrupted, or no quench at all).
- Synthetic rehearsal: E8 and E9 run. At alpha = 0.01 the 19 calibration clean shots were too few and the method correctly refused to alarm. In the fake data the unnoted quenches have no radiation precursor, and PU gave them a mean weight of 0.007, which is the expected answer for that data.

## 2026-10-15 - VPS failure, real-data access, termination study, research server

VPS run of commit 4f703f1 (owner's report): 99 tests passed, then `build dataset` failed: `ConnectionTimeoutError` to `s3.echo.stfc.ac.uk` while opening shot 11860. GitHub Actions and (today) this sandbox reach that host, so it looks specific to the VPS's network path. Not diagnosed yet: `scripts/diagnose_network.py` (new) tests DNS, TCP to each of the host's 4 IPv4 addresses (no IPv6 records), HTTPS and zarr, and says what to do. The build now retries the connection 4 times (`CONNECTION_CHECK_ATTEMPTS`) instead of once.

Data servers today (2026-10-15, from this sandbox)
- `s3.echo.stfc.ac.uk`: reachable for the first time from this sandbox; shot 11860 reads (385 samples).
- `mastapp.site`: down. TLS resets from here, and HTTP 503 Service Unavailable through an independent fetcher. So the shot table (notes, campaigns) cannot be downloaded by anyone right now. `build_dataset.py --from-report` builds from the 400 Phase 2 shots instead. The report's comments are cut at 90 characters, which changes the parsed note label of 8 shots (13644, 14317, 15007, 18583, 23168, 23319, 29631, 30284) and the parsed note time of 8 (13644, 14317, 15210, 16369, 18583, 20259, 20824, 29631), so the stand-in comment is rebuilt from the stored `note` and `t_note` columns; checked: 0 label and 0 time differences.
- `level2/defuse/` on the same bucket lists 3,067 HDF5 files (`MASTno<shot>.h5`, shots 12139-30471, 2024-09-13) but every download returns 403 AccessDenied. Not public; not pursued further.

Data catalogue (`scripts/inspect_groups.py`, metadata only)
- Shot 11860: 12 groups (bolometer, gas_injection, interferometer, magnetics, pf_active, pf_passive, pulse_schedule, soft_x_rays, spectrometer_visible, summary, tf, wall). Shot 27000: 15 groups, adding controllers, equilibrium (q95, li, beta_N, elongation, line-average n_e) and thomson_scattering. So equilibrium features exist only for some campaigns: survey before use.
- `pulse_schedule` (11860): `i_plasma` (A, UDA XDC_IP_T_IPREF), `n_e_line`, `z_ref` on a 0.05 ms grid from -0.100 to 0.284 s, ending with the summary record. Assumption, not verified: this is the programmed reference. It may include real-time changes by the control system, so as a feature only values with time <= t would be used.
- `pf_active/coil_current` channel `SOL` is the central solenoid current (channel list: SOL, P2IL FEED, ... P6U; units A).

Termination study (`scripts/termination_study.py`, real data, `results/termination_study/report.md`)
- Question: are the quenches nobody noted (167 of the 400 report shots) planned endings? Answer: no. At the detector's quench time the programmed current is still at its maximum (median fraction 1.000 in both groups; >= 0.9 in 95.3% of noted and 92.8% of unnoted). Both kinds of quench happen during the programmed flat-top.
- Solenoid current at the quench is more negative for unnoted quenches (median -37.0 vs -33.6 kA), i.e. closer to the end of the swing, consistent with flux exhaustion as part of the story. Weak effect: AUC 0.596, permutation p = 0.0022 (10,000 shuffles); per campaign 0.53 (M5) to 0.69 (M9). The most negative solenoid current seen is -66.5 kA, so the hardware limit is not simply where the unnoted quenches sit.
- Caveats: the 400-shot sample over-represents noted disruptions; quench times come from our detector; the solenoid limit per campaign is unknown.

Infrastructure (owner's request: public code, private runs, run forever)
- `MAST_HOME` (from the git-ignored `.env`) decides where data, results, models and logs go. On the VPS it is `~/mast-private`; unset, nothing changes.
- `scripts/research_server.py` replaces the nightly cron job and `deploy/run_pipeline.sh` (removed). Judgement calls: pull at most every 10 min; a failed commit is retried every 6 h; seeds are idle work and do not need the network; a lock file prevents two copies (systemd and screen); Telegram commands only from `TELEGRAM_CHAT_ID`, fixed actions only, `/predict` accepts digits only.
- Trained models are saved with pickle (`src/registry.py`). Pickle can run code from a file, so only the owner's own models should be loaded.

First full real-data run (2026-10-15, this sandbox, private workspace)
- `build_dataset.py --from-report` + `run_all.py --skip-build` on the 400 report shots: 400 downloaded from S3, 398 with windows (25424, 25470 have none, as before), 171 disrupted / 227 clean, 26,169 windows. E0-E9 ran with 0 failed steps. Following the owner's wish that training results stay private, the model numbers are not committed; they were reported to the owner directly. What they mean for the plan:
- Under `LABEL_SOURCE = "both"` no model catches more than a few percent of disruptions at <= 5% false alarms. Expected from the label problem: most clean shots in this sample end in an unnoted quench that looks like a disruption.
- With "signal" labels (any quench is a disruption) the catch rate at <= 5% false alarms is also near zero (validation, E7), although window PR-AUC is higher. So with only the 4 global signals, the 30 ms before most MAST quenches are hard to tell from normal flat-top. This moves ROADMAP D1 (saddle-coil locked-mode and Mirnov signals) to the top.
- The most important feature in gradient boosting is `t_since_start`. Possible shortcut (typical shot length rather than physics); an ablation without it is needed.
- Temporal split: the threshold tuned on M5-M7 gives a clearly higher false-alarm rate on M8-M9 (drift). The conformal threshold kept the test false-alarm rate under alpha at 5% and 10%; at 1% it needs >= 99 calibration clean shots and refused to alarm with 23.
- PU (E9) estimates c around 0.4, and unnoted-quench windows got a higher mean weight than other unlabelled windows. Small sample, SCAR assumption: indicative only.
- `scripts/check_live.py` now checks the shot table last and names it in the failure, so a mastapp.site outage (CI red on PR #2) no longer looks like a data-layer bug.

## 2026-10-15 - VPS blocked by STFC; GitHub data relay

- Owner's curl tests from the VPS (IP 2.58.56.69): every request to `s3.echo.stfc.ac.uk` and `mastapp.site` failed with "Failed to connect ... port 443 after 20000 ms" (no TCP answer). A TCP traceroute to 130.246.179.110 passes the provider, Telia/Lumen and JANET (146.97.x.x, up to 146.97.41.34) and then goes silent, so packets are dropped at STFC's edge. `https://www.ox.ac.uk` answered 403 from the same IP, while ukaea.org, pypi.org and api.telegram.org worked. Conclusion: STFC (and at least one other UK university) refuse this IP range; it is not a routing fault on the provider's side and not our code. We do not try to get around the firewall (no proxies, no fingerprint tricks).
- Fix chosen by the owner: a GitHub data relay. GitHub's runners can reach STFC (CI read shot 11860 today). `fetch_data.yml` runs `scripts/relay_fetch.py` and uploads `data/` as a one-day artifact; `src/relay.py` on the server starts the job, waits, downloads and unpacks it; `run_all.py --offline` then trains without touching the network. The data is public (CC BY-SA), nothing is committed, and trained models and results stay on the server. Long-term, a server on an IP range STFC accepts is still better for the multi-GB magnetics data (ROADMAP D1), though the relay could also reduce such signals before sending them.
- Rehearsed here: `relay_fetch.py --scale full` with mastapp.site down fell back to the 400 report shots and downloaded all 400 (7.9 MB, so ~20 KB per shot and roughly 80 MB for a full run). Unpacked into a fresh workspace and built with `--offline --from-report` while the data URL pointed to an unreachable host: 398 shots, 26,169 windows, identical to the direct build.
- Private settings moved to `~/mast-private/.env` (config reads it as well as a repo `.env`), so deleting the code folder loses nothing. `setup_vps.sh` copies an old repo `.env` there (keeping a backup) and asks for the optional GitHub token. The token is read from the environment only when used and is filtered out of `config_snapshot`, so it never lands in result files or the results zip sent to Telegram.
- The owner pasted a GitHub token and the Telegram bot token into the chat. Neither was used, stored or committed by me; advised to revoke both and paste the new ones only into the VPS setup.

## 2026-10-15 - Magnetics (D1), watchers, gold tool, literature check

VPS status (owner's report): the research server started on commit 41b9565 in /root/mast-private; the network diagnosis confirms every TCP connection to both STFC hosts times out from the VPS (IPv4) and IPv6 has no route there at all. So the VPS works only through the GitHub relay. `data.host_reachable` (10 s TCP check) now makes a blocked host fail in seconds; before, the zarr and pandas clients could take minutes per check.

Magnetics data facts (inspected on shots 11860 (M5), 20259 (M7) and 27000 (M8); nothing below is assumed without a check unless marked)
- `magnetics/b_field_tor_probe_saddle_field`: 12 channels named M01..M012 (voltage channels M01..M12), units T, UDA `ASM/SAD/OUT`, time base `time_saddle` with a 20 us step (50 kHz).
- Geometry: three rows of 12 saddle coils (lower, middle, upper; `sad_out_l01`.. etc.), all at R = 1.999 m; the middle row spans z = -0.4..0.4 m, each coil covers 20 degrees, centres at 15, 45, ..., 345 degrees. ASSUMPTION (name match only, not confirmed by documentation): measured channel M0k is middle-row coil k. Ask the FAIR-MAST team to confirm.
- The saddle data is stored only while the plasma exists: NaN before about +2 ms (20259: from 0.0026 s; 27000: from 0.0022 s) and after the plasma ends. So there is no pre-plasma baseline; each coil's offset is its mean over its first 5 ms (`EXTRAS_BASELINE_MS`), and lm_n1 stays NaN until then.
- Channel M09 is entirely NaN in 20259 and 27000 (dead coil). The n=1 fit skips NaN coils and needs at least 6 (`MIN_SADDLE_CHANNELS`).
- Centre-column Mirnov coils: 5 poloidal (201, 210, 220, 230, 240) and 3 toroidal (201, 206, 212), units T, 2 us step (500 kHz). In 20259 the first poloidal coil is all NaN; the code takes the first coil with data. Fluctuation RMS is of order 0.02-0.3 uT in flat-top.
- Shot 20259 (note: "Looks like a locked mode at..."): lm_n1 is ~1.7-2.1 G through the flat-top, falls to 1.2 G at 0.290 s and jumps to 3.5 G at the thermal quench (0.295 s); Mirnov RMS shows bursts at 0.250 s (0.19 uT) and 0.285 s (0.075 uT) against ~0.025 uT normally. The steady ~2 G is probably vacuum pickup from the PF coils plus error fields, not plasma: lm_n1 is a crude locked-mode proxy until vacuum pickup is subtracted (ROADMAP D1 next step).
- `pulse_schedule/i_plasma` and `pf_active` SOL are held at their last value (causal), as in the termination study.

Watchers
- The S3 listing returns about 700 folders per page and marks the rest as truncated, so pages are followed by continuation token (a count-based stop first gave 701 instead of 11,573). Full count: 11,573 shot stores, exactly the number of rows in the shot table. `level2/` holds defuse/, netcdf/ and shots/. DEFUSE sample file: 403. Shot table: no answer from this sandbox.
- arXiv API (export.arxiv.org) works; the first check found 40 matching papers; a second check found 0 new (ids without version suffix, so new versions are not new papers).

Real-data results on the 400-shot report sample (private runs; numbers reported to the owner, not committed)
- E11 (validation): removing `t_since_start` cuts the HGB catch rate and PR-AUC clearly. Part of the small v1 skill comes from time since start (shot length), not physics. Every model claim must now be checked with and without it.
- E12 (validation, current-only labels as in the UKAEA benchmark): window PR-AUC rises with horizon, partly because the positive share rises; compare against each horizon's base rate. Shot-level catch at <= 5% false alarms stays near zero.

Literature check (docs/literature_review.md) corrected two of our earlier "possibly new" ideas: conformal prediction has been applied to disruptions (JET, Springer chapter) and survival / time-to-disruption models exist (Keith et al. 2024; arXiv:2609.24443). ROADMAP updated. PU learning for unrecorded disruptions and a label audit of MAST notes were not found.

Gold tool design: blind plots (measured and programmed current, radiated power, solenoid current; no note, no detector time). Seeded queue: 35 unnoted quenches, 30 noted quenches, 30 without a quench, 5 noted without a quench. Answers d / n / u stored in MAST_HOME/gold/labels.csv; a later answer for the same shot replaces the earlier one in the evaluation.

E13 on the 400-shot sample (2026-10-15, private run; numbers given to the owner)
- Extras computed for all 398 shots on real data in ~16 min (0 failures). Coverage over windows: lm_n1 92%, Mirnov 94%, solenoid 100%, ip_ref_error 99.6%.
- Validation result: adding the magnetics did not raise the catch rate with time-since-start present; without time-since-start, the magnetics set did better than v1 without it (E11). With 34 disrupted validation shots, one shot is ~3 percentage points, so every difference here is within a few shots: not significant.
- Reading: (1) lm_n1 still contains vacuum pickup from the PF coils, so it is mostly noise for now; (2) 19 more features on 238 training shots invites overfitting; (3) the "both" labels still call most quenches clean, which caps every model. Next: vacuum-pickup subtraction for the saddle coils, normalised physics features (e.g. lm_n1 relative to its flat-top level, log Mirnov RMS), the seed sweep with extended features for error bars, and the full dataset once mastapp.site is back.

## 2026-10-16 - Relay refused by GitHub (403), not a data-server problem

VPS report: the first data-source check failed with `403 Forbidden` on `POST /repos/fazil-mastu/mastu/actions/workflows/fetch_data.yml/dispatches`. That request goes to GitHub, not to STFC or mastapp.site, so the mastapp.site outage plays no part. A 403 (not 401) means GitHub accepted the token as valid but refused this action: the fine-grained token needs repository access to `mastu` and the repository permission "Actions: Read and write". The account is a personal one, so no organisation token-approval policy is involved. The same refusal blocks the relay fetch, so the server cannot run the pipeline until the token is fixed; it retries the data-source check after 24 h, or at once on `/sources`.

`relay.check(response)` now replaces `raise_for_status()` for GitHub API calls: the Telegram message carries GitHub's own message, the permission GitHub names in `X-Accepted-GitHub-Permissions`, and what to change.

The relay workflow had never run before; a first `probe` run was started from this session (run 38858508318) to check the GitHub side independently of the VPS token.
