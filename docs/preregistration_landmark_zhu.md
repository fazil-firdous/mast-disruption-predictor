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

### Result (2026-11-24)

Primary, EAST (468 test disrupted shots, 76,345 decisions, 2,092 positive): Brier gain of the signals over clock +
schedule **-0.00046 (-0.00063 to -0.00032): no improvement shown**. Read with the decision AUCs: clock 0.636, clock +
schedule **1.000**, + signals 0.999, clock + signals 0.954. The recording schedule alone separates the last 100 ms
before a disruption from earlier decisions almost perfectly, so there is nothing left for the signals to add: the
null is a ceiling, not evidence that the signals carry no information (they add +0.0095, +0.0082 to +0.0109, over the
clock alone). DIII-D the same (clock 0.655, clock + schedule 0.995; signals over clock + schedule -0.00004, -0.00016 to
+0.00007). C-Mod differs: the schedule adds little (0.790 -> 0.828; gain +0.0070, +0.0013 to +0.0119); signals over
clock + schedule +0.016 (-0.014 to +0.043), not resolved with 152 test shots and 1,490 decisions. Every disrupted
record ends at its disruption (median gap 0.0 ms).

### Diagnostic 1 (2026-11-25, after the result; descriptive, specified before it is computed)

The published account of these files describes a sampling increase shortly before disruptions, yet the timing
features separated decisions 10-100 ms ahead from earlier ones almost perfectly. To reconcile the two from the
original time stamps, without any model: (1) for disrupted and non-disrupted records of each machine, the gap between
consecutive slices as a function of time before the record's disruption time (disrupted) or last slice (non-disrupted),
in lead bins 0-10, 10-20, 20-40, 40-60, 60-100, 100-150, 150-200, 200-300, 300-500, 500-1000, 1000-2000 and
2000-5000 ms: slices, median gap and 10th/90th percentiles, and the share of records with a slice in the bin; (2) per
disrupted record, the median gap in 0-100, 100-300 and 300-1000 ms before t_d; (3) the cohort flow of the landmark
analysis (test disrupted records; with at least one decision; with decisions of both kinds) against the endpoint
module's test cases; (4) one worked EAST test record: its time stamps in the last 400 ms, and for each decision in that
span the slices in its window, the age of the last slice, the largest gap and the label. Also recorded: the early
stopping in the landmark models drew its 10% validation decisions at random from training decisions, so validation
decisions share shots with training decisions; this affects when training stops, not the test shots.
