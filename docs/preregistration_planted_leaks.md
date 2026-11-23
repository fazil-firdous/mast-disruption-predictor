# Audit toolkit, planted-leak validation: do the checks fire on the leak they target, and only on it? (protocol)

Written 2026-11-23, before any check was run on any synthetic dataset. The git commit that adds this file is its
timestamp. Changes after the first result go at the bottom.

## What this is

A software validation of three generic checks of the audit toolkit, on synthetic records where the truth is known.
It is not evidence about any real machine. It answers: on data with one planted shortcut, does each check flag that
shortcut, and does it stay quiet on the others and on clean data?

## The checks (generic; input: shot, time, disrupted, signals; split by shot, seed 42, 60/20/20; test shots only)

Every shot's landmark is its last slice minus 40 ms; the window is the 300 ms before it (slices at or before it only).
- **T, observation-process ceiling:** a logistic model (standardised) on timing features only (number of slices in the
  window, age of the last slice, largest gap in the window), trained on the training shots; flag if its test AUC is at
  least 0.80 (the timing study's line).
- **D, clock ceiling:** the same on one feature, the elapsed time from the first slice to the landmark; flag if test
  AUC is at least 0.80.
- **E, endpoint dependence:** endpoint module 2 exactly as registered (`scripts/endpoint_zhu.py`: frozen values-only
  boosting on 36 features; control at its own end vs at the case's elapsed time; matched concordance; 2,000 bootstrap
  resamples); flag if |Delta| is at least 0.03 and the range excludes 0.

## Synthetic records (fixed now)

600 shots, every third disrupted. Time from 0 to the shot's duration; one slice every 10 ms. 12 signals, each an AR(1)
series (coefficient 0.95, innovation s.d. 0.1, started from its stationary distribution) plus independent measurement
noise (s.d. 0.05). Genuine precursor in every family: in disrupted shots, signal 0 gains s times a linear ramp from 0 to
1 over the last 300 ms. Durations uniform on 1.0 to 3.0 s for both classes unless stated.

Families (one planted shortcut each):
- **L0, clean:** as above.
- **L_T, sampling shortcut:** non-disrupted shots stored at one slice every 100 ms (disrupted stay at 10 ms).
- **L_D, duration shortcut:** disrupted durations uniform on 0.6 to 1.4 s, non-disrupted on 1.6 to 3.0 s.
- **L_E, end-of-record shortcut:** non-disrupted shots get the same kind of ramp (s times 0 to 1 over the last 300 ms)
  on signal 1: an end-of-record signature that exists only near their own end.

Expected flags: L0 none; L_T only T; L_D only D; L_E only E.

**Strength s.** Chosen once before any check runs: the smallest of {0.25, 0.5, 1, 2} for which the values-only model
(the E check's frozen model, ordinary AUC with controls at E) on family L0, generator seed 0, lies in 0.75 to 0.90.
Seed 0 is not a validation seed. If none qualifies, the value closest to 0.825 is used and this is reported.

## Validation (fixed now)

Generator seeds 1 to 20 for each family (80 datasets). Success: every expected flag fires in at least 18 of 20
replicates, and every unexpected flag fires in at most 2 of 20 (including all three checks on L0). The full 4 x 3 table
of fire counts is reported with the median statistic per cell, whatever the outcome. No generator setting, threshold or
check is changed after the first table; if a cell fails, it is reported as failed and its reason investigated in a
dated note, not tuned away.

## Changes after the first result

### Result and investigation 1 (2026-11-23, after the first table)

The first table (NOTES.md 2026-11-23) **failed the registered rule in one cell**: E fired on the duration family L_D in
3 of 20 replicates (limit 2). All other 11 cells passed (T 20/20 on L_T and 0/20 elsewhere; D 20/20 on L_D and 0/20
elsewhere; E 20/20 on L_E, 1/20 on L0 and L_T). The strength rule found no s with AUC in 0.75 to 0.90 (0.744 at
s = 0.5, 0.945 at s = 1); the closest, 0.5, was used, as specified. The validation stands as FAILED.

Investigation, specified now, before it is run; descriptive, it changes nothing above: E's false-alarm rate on clean
data, family L0, generator seeds 101 to 200 (100 new datasets), E only, same settings. Reported: the share that fires
(Wilson 95% interval), the share with the bootstrap range excluding 0 regardless of size, the SD of Delta across
replicates against the median bootstrap half-width, and test-set sizes. Question: is 3/20 consistent with a modest
false-alarm rate at this small test size (about 40 cases and 80 controls, against 503 and 1,203 on EAST), or is the
bootstrap range too narrow?
