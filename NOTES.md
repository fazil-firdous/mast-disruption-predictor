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

## 2026-10-02 - Phase 2: labels and reconciliation report

Implemented `src/labels.py` (`note_label`, `parse_note_time`, `median_filter`, `detect_disruption`, `final_label`) and `scripts/label_report.py`. Tests: 38 pass, all from SPEC 12 for labels (quench within +-2 ms, slow ramp-down not flagged, tiny current gives unknown, note parser negations and `"DISRUPTION AT 220MS"` -> 0.22).

Checks against the SPEC
- Note label on the full table: 855 disrupted of 11,573 (7.4%), exactly as SPEC 5.1 states.

Detector judgement calls (SPEC 5.2 left room)
- SPEC step 3 ("earliest t where |Ip| falls 80% within 10 ms") fires up to `CQ_MAX_MS` too early, because every sample in the 10 ms before a quench also qualifies. I take the earliest candidate, find where the fall ends, then move to the last sample still within 90% of the level at the top of the fall (`CQ_ONSET_FRAC`). The spike rule (SPEC step 4) is applied after that, and only if the spike is 3% above the level before it (`SPIKE_MIN_RISE`). Without this the synthetic test missed by more than 2 ms.
- A candidate must start above `CQ_MIN_FRAC_OF_PEAK` of peak current. Reason: the last stretch of any linear ramp-down also falls 80% in 10 ms once |Ip| is small, which my first synthetic test showed. I first used 0.2, which flagged the tail of ramp-downs. 0.5 passes the tests, but it also means a disruption that happens below half of peak current is not found (see the note-only cases below).
- NaN samples in `ip` are dropped before detection.
- `LABEL_SOURCE = "both"` means disrupted only when the note and the detector agree. The SPEC did not define it.

Results (uniform random sample of 400 shots, seeded; plus 150 extra note-disrupted shots; full text in `results/label_report/report.md`)
- 2x2 on the uniform sample: note no / signal no 93, note no / signal yes 285, note yes / signal no 0, note yes / signal yes 22. Notes mark 5.5%, the detector 76.8%.
- DATA SURPRISE: most MAST shots end with an |Ip| spike and a fast fall. In 453 flagged shots the median time for |Ip| to go from 90% to 10% of peak is 5 ms, in the 97 unflagged shots 124 ms. Flagged shots with and without a note look the same: median quench rate 2.0e8 vs 1.9e8 A/s, peak 782 vs 766 kA, fall 4 vs 5 ms. The median flagged quench is 5.4 ms before the shot table's `plasma_end_time` in both groups. So on `ip` alone the detector cannot tell a noted disruption from an ordinary end of pulse. The SPEC's balanced "300 disrupted, 300 clean" would be 300 mostly ordinary end-of-pulse quenches against clean shots that end in a slow ramp-down. A model could learn "this shot ends in a ramp-down" instead of "this shot disrupts". This is not fixed, and I did not tune the detector to match the notes (that would make the comparison circular).
- The SPEC's step 5 is confirmed in one direction: slow ramp-downs (median 124 ms) are not flagged.
- Timing where both exist (65 shots): `t_disrupt - t_note` median +1.0 ms, quartiles -2.0 to +6.0 ms, 78% within 10 ms. The detector's time is consistent with the notes.
- Sensitivity of the flagged share to `CQ_MIN_FRAC_OF_PEAK` on the uniform sample: 0.2 -> 90.7%, 0.35 -> 84.8%, 0.5 -> 76.7%, 0.65 -> 67.5%. The headline does not go away.
- 4 note-disrupted shots have no detected quench (`results/label_report/review_list.csv`): 15555 ("rolled off slowly but disrupted at 600ms"), 19388 ("disrupts near end of rampdown"), 21673 (arcing, ramp down), 25108 (IRE then MARFE). The plots show a slow ramp-down with a small spike at low current. These are disruptions at low current that the 0.5 floor misses.
- `t_note` parser: first `at <number> ms|s` after the word "disrupt". It fails on comments like "disrupts at 0.302 broken pellet ... at 0.263s" (takes 0.263; shot 13300) and returns NaN for many comments without a unit. Only 379 of the 855 note-disrupted shots in the table get a time at all.

Decisions waiting for the owner
1. Label policy. Options: keep `signal` (SPEC default, but about 77% of shots become "disrupted"), switch to `note` (precise but only 7% of shots and misses the un-noted ones), or define disruption differently (e.g. quench that is not the planned end of the pulse, using more than `ip`). I have not changed `LABEL_SOURCE`; it is still `"signal"`.
2. Whether `Unknown` campaign shots (292) go into the temporal split.
3. Whether the `zarr>=3` change above is accepted.

Not done / not verified
- The notebook `notebooks/colab_pipeline.ipynb` has not been run in Colab.
- No HTTPS download of shot data from the sandbox has worked. See the network sections above.
