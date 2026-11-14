# Does the public preprocessed dataset of a released benchmark model carry the observation schedule? (protocol)

Written 2026-11-14, before the file below was downloaded or opened. The git commit that adds this file is its
timestamp. Changes after the first result go at the bottom with their date. No model is fitted.

## What is known (from reading public code only)

The public repository of the CCNN model of DisruptionBench (github.com/Aphoh/fusion-dp, MIT licence, last commit
22fb64b, 2024-05-22) has two data modules:
- `LucasDataModule` (the default, `cfg/config.yaml`: `dataset.name: 'Lucas'`): downloads a preprocessed file,
  `lucas_data_f32.pickle.gzip`, from a public URL written in the code, and uses each shot's array as it is, with the
  last 8 time steps removed (`end_cutoff_timesteps: 8`).
- `DhruvaDataModule`: reads a local file of the same kind and first applies `resample_array`, which keeps the first
  row of every block of 20 rows (EAST) or 5 rows (DIII-D) and repeats it over the block.
Each shot in the file holds `data` (an array, time by channels), `label` and `machine`.

## Question

In the public preprocessed file used by the default data module, does the pattern of value changes over time,
without the values themselves, tell disrupted from non-disrupted shots? And does `resample_array`, as written in the
repository, remove that?

## Opening the file safely

The file is a pickle from the internet. It is opened only with a restricted unpickler that allows numpy arrays,
dtypes, scalars and plain Python containers and refuses everything else (listing what it refused). No model or code
from the repository is run.

## Statistic

Per shot, on its array with the last 8 rows removed (as the default module does). A channel that increases by the same
step at every row (a time or step counter) is excluded; every other channel is a signal.
- **P1 (primary): change occupancy:** over the last 20 rows before the cut, the share of row-to-row steps at which at
  least one signal changes value (an adjacent pair where both values are NaN is not a change).
- **P2: whole-record change occupancy:** the same over all rows up to the cut.
- **P3: after `resample_array`:** P1 recomputed after applying the repository's `resample_array` (EAST factor 20,
  DIII-D factor 5, C-Mod none), exactly as written there.
Shot-level AUC, disrupted (label 1) against non-disrupted (label 0), per machine, 95% range from 2,000 bootstrap
resamples of shots (seed 42). Score direction fixed now: more changes = more disruptive.

## Reading (fixed now)

- **P1:** AUC at least 0.60 with its range above 0.5: the default public pipeline's input data carry the observation
  schedule after its own end cut: a sequence model trained through it can learn the outcome from when values change.
  Otherwise not, by this statistic.
- **P3:** the same rule: if not met, `resample_array` removes this change-pattern signal (it does not establish that
  it removes freshness differences); if met, it does not.
- |AUC - 0.5| above 0.10 in the reversed direction is reported as a reversed association, not as no association.
- Not claimed: that the reported DisruptionBench or CCNN results used the default module, or that any reported
  performance is affected.

## Secondary

Shots and class counts per machine; rows per shot; the excluded counter channel, if any; the median P1 by class.

## Changes after the first result

- **2026-11-14, the file is not available.** The URL in the repository's code (`datamodules/lucas.py`) returned 404 Not
  Found on 2026-11-14, so the primary analysis could not be run; nothing was opened. The released default data path
  therefore cannot be reproduced from public sources today; the Dataverse release is the public source.
- **Same day, a check of the repository's correction instead (exploratory):** `resample_array` with factor 20, applied
  to a 5 ms forward-fill from the first slice, gives exactly the values of the harmonisation study's R2 grid (100 ms
  from the first slice, latest slice at or before each point) without R2's maximum-age rule (unit test,
  `tests/test_public_pipeline_audit.py`). That rule emptied 0 to 1% of points in EAST and DIII-D. So the
  harmonisation study's R2 results stand, to that approximation, for the released correction applied to the public
  files: residual timing AUC 0.938 in EAST, raw observation age 0.981 in DIII-D (see NOTES).
