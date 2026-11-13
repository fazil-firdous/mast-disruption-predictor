# Faster signals, step 1: soft X-rays (protocol)

Written 2026-11-13, after the soft X-ray quality audit and before any soft X-ray feature was computed or any model
with soft X-rays was trained. The git commit that adds this file is its timestamp. Changes after the first result
go at the bottom with their date.

## What is already known (and what was looked at)

- Step 0 (`docs/preregistration_fast_signals.md`): magnetic summaries over 5 ms instead of 20 ms added no measurable
  discrimination at 50 ms. A new diagnostic is the next question.
- The quality audit (`scripts/sxr_audit.py`, NOTES.md 2026-11-13) read 600 twin pairs (1,154 shots, 1,141 with
  soft X-rays). At SNR >= 5, 437 pairs (73%) are usable in both shots. The two horizontal cameras are in all 1,141
  shots (median SNR 20 and 18, 1% of chords saturated); the tangential camera in 928.
- The audit's missingness check: lost usable and twin not, 62 pairs; the reverse, 39. The audit's usability needs a
  chord to cover 95% of the shot's whole plasma, and the cameras stop at a fixed time, so a twin that runs longer is
  more often 'not covered'. This is likely why, but it is not proven. So this study drops no shot or pair by
  quality, uses no 'has soft X-rays' input, and reports the primary result also on the pairs where both shots have
  soft X-ray inputs at the comparison moment.
- Looked at: the chord-against-time pictures and slow-motion videos of three development pairs (M5/M6: 11880/11876,
  16296/16295, 18921/18916). In 18921 the whole profile drops at once several times in the last ~100 ms (crash-like
  steps) and the twin shows none; in 11880/11876 the emission moves across the chords in both shots, at different
  times. Three pairs are anecdotes. They shaped the choice of the 'drop' and 'position' features below, and that is
  why they are listed here.

## Question

Do soft X-ray summaries, added to the existing full model, tell a lost shot from its twin better at a fixed time
before the loss?

## Signals (computed on GitHub, `scripts/sxr_series.py`, from the raw cameras)

The two horizontal cameras (`horizontal_cam_lower`, `horizontal_cam_upper`), 18 chords each, 500 kHz. The tangential
camera is not used: it is missing in a fifth of the shots, and a camera that some shots lack invites a model to
learn the missingness.

Per chord, causally: the offset is the mean of its samples before t = 0 (before the plasma); a chord with fewer than
50 such samples, or with no variation in them, is left out for that shot. In each 1 ms bin (k-1, k] ms, a chord's
value is its mean minus the offset; a bin where the chord does not change at all (stuck or saturated) is empty for
that chord. A bin needs at least 12 chords, otherwise all its values are empty. Per camera and bin, from the chord
values b_i (i = 0..17, position y_i = i/17, b+ = max(b, 0)):

- **level**: mean of b over the chords present.
- **position**: sum(y b+) / sum(b+), where the emission sits across the camera's view.
- **width**: the spread of y weighted by b+.
- **peak**: max(b+) / sum(b+), how concentrated the emission is.
- **fluct**: the standard deviation within the bin of the camera's summed signal (all chords present, offset removed),
  divided by the number of chords: fluctuations above ~1 kHz, which the 1 ms means cannot show.

The value of bin k is known at time k ms. Bins are kept from 25 ms before the plasma starts (|Ip| > 50 kA) to 5 ms
after it ends, or from 0 to the end of recording when the plasma window is unknown.

## Features (per window, on the server, `scripts/soft_xrays.py`)

For a window ending at t_end, only bins with k <= t_end (in ms) are used; the window holds the last 20 bins, as the
present features hold the last 20 ms.

- For each camera and each of the five signals: mean, standard deviation, slope and first-to-last change over the
  window, by the same function as the present features (`src/features.py`, `_stats`). 40 features.
- For level and fluct of each camera: the relative change (log10 of the window mean over the median of the shot's
  own bins from the plasma start to the window), by `features.relative_change`. 4 features.
- For each camera: **drop**, the largest fall of level from one bin to the next within the window, as a share of the
  earlier bin's level (0 if level never falls; empty where level is not positive). 2 features.

46 features. Empty where the data are not there; no flag for missing data.

## Models

Trained exactly like the v3 baseline (as in step 0): candidates left out of training, tuned and judged on clear
endings, same model family, tuning grid, label horizon of 30 ms, window ends and stride. Only the inputs change.

- **A, full (existing):** the saved v3 baseline runs.
- **S, full + soft X-rays (primary):** the full inputs plus the 46 soft X-ray features.

Runs: S on random splits 1 to 20 (the v3 splits) and on the chronological split (train and tune on M5-M7, test on
M8-M9). 21 runs.

## Primary quantity

C(50 ms) of S minus C(50 ms) of A: both shots of a pair scored at the same moment, 50 ms before the lost shot's loss
(the fixed-lead test of `scripts/how_early.py`, unchanged), on the common cohort of all twins, paired over splits 1
to 20. 50 ms as in step 0: the first lead beyond the 30 ms the models are trained for.

**Smallest gain worth having:** +0.03.

## Reading

- Resolved over splits (range excludes zero), positive, and at least +0.03: soft X-ray summaries add early
  discrimination at 50 ms.
- Not resolved: "these soft X-ray summaries add no measurable discrimination at 50 ms beyond the full model". Not
  that the soft X-rays hold no earlier information.
- **Required checks**, whose sign must agree for the result to be called consistent: (1) the same difference on the
  pairs where both shots have soft X-ray inputs at the 50 ms moment (any camera's level mean present); (2) on plan
  twins (`plan_twins/pairs.csv`).
- The pooled range over groups of pairs sharing a twin is reported beside it and does not decide.

## Secondary, all reported whatever they show

- The same difference at 10, 30 and 100 ms (descriptive).
- On field-and-shape twins.
- The placebo (two good repeats at their lost shot's landmark) for A and S.
- The chronological split, pooled ranges only. No M8 or M9 shot was looked at in pictures or videos.
- The usual metrics (pAUC, caught, false alarms), paired differences over the 20 splits.
- Missingness at 50 ms: the share of lost shots and of twins with soft X-ray inputs at that moment.

## Stop rule

A null here ends the soft X-ray part of this study with these summaries; it is reported as a null, with its range.
Other soft X-ray features would be a new protocol, written before they are computed.

## Changes after the first result

(none yet)
