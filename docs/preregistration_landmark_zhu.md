# Landmark module on the public Zhu et al. disruptive records: do the signals add next-100 ms risk beyond the clock
# and the recording schedule? (protocol)

Written 2026-11-23, before any statistic of this study was computed. The git commit that adds this file is its
timestamp. It applies the landmark primary validated on planted shortcuts (docs/preregistration_landmark_v3.md,
PASSED) to real records for the first time. Changes after the first result go at the bottom.

## What this is, and what it is not

Analysis **conditional on an eventual disruption**: only the disrupted records of the public files (Harvard Dataverse
doi:10.7910/DVN/XIOHW1, version 2, fig4 `*_for_nn.mat`; EAST split from the release's own lists). Within those
records, at each decision time, does knowing the 12 stored signals improve the predicted probability that the
disruption comes within the next 100 ms, beyond what elapsed time and the recording schedule already give? It is not
a deployable all-shot estimate (non-disrupted shots are excluded by design), says nothing about the published models,
and is file-relative: the files' first slice is not a physical origin.

## Records and decision grid (fixed now)

- Disrupted record: any finite `time_until_disrupt`. Its disruption time t_d = median of (time + time_until_disrupt)
  over its slices. Reported: the median gap between t_d and the record's last slice.
- Decisions L = t0 + 300 ms + k x 20 ms for every L before t_d (t0 the first slice); window: slices in (L - 300 ms, L].
- Positive if t_d is in (L + 10 ms, L + 100 ms]; negative if t_d > L + 100 ms; decisions with t_d in (L, L + 10 ms]
  are too late and left out (counted).
- Split by shot: EAST the release's lists; DIII-D and C-Mod the audit's seeded split (seed 42, 60/20/20) over all shots,
  then only disrupted ones kept. Training decisions from training shots, results on test shots only.
- Timing features: slices in the window, age of the last slice, largest gap, quantised to 1 us (C-Mod's stored times
  carry offsets up to 0.124 us; membership is decided with a 1e-9 s tolerance first).

## Models (same learner: gradient boosting, max_iter 200, learning rate 0.1, seed 42, early stopping on 10% of the
## training decisions, 10 rounds)

- C: elapsed time.
- CS: elapsed time + timing features (clock and schedule).
- CSV: elapsed time + timing features + the 36 value features (last, mean, last minus first of the 12 signals).
- CV: elapsed time + value features.

## Primary

EAST: paired Brier gain of CSV over CS on test decisions (equal weight per decision), 95% range from 1,000 resamples of
test shots (seed 42). Reading: range above 0: the stored signals add next-100 ms risk information beyond the clock and
the recording schedule in these records. Otherwise: no improvement shown (not shown absent).

## Secondary and reported with it

- The same for DIII-D and C-Mod (C-Mod with the timing caveat).
- Gain of CS over C (how much the recording schedule adds to the clock) and of CV over C (signals over the clock
  alone), with ranges.
- Decision AUCs of C, CS, CSV, CV; calibration in the large; counts (shots, decisions, positives, too-late).
- Power reference from the synthetic study, printed with the reading.

## Changes after the first result

(none yet)
