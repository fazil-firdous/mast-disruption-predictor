# Audit toolkit, all-disruptive mode: planted-leak validation (protocol)

Written 2026-11-23, before the all-disruptive mode was run on any of the datasets below. The git commit that adds this
file is its timestamp. The mode itself (`scripts/audit.py`, 5aa7a8d) was written and run only on its demo data
(NOTES.md 2026-11-23). Changes after the first result go at the bottom.

## What this is

A software validation, on synthetic records, of the two decisions the all-disruptive mode makes about a model and its
data: the sampling flag and the clock-matched flag. Not evidence about any machine. The clock-only AUC is a property of
how the data are windowed, not a decision about a model; it is reported for every family but not validated as a flag.

## Settings (the toolkit's, fixed)

`scripts/audit.py` all-disruptive mode exactly as committed: windows of 20 ms every 5 ms; positive if the window ends
within 30 ms of the disruption and at least 2 ms before it; seeded 60/20/20 split by shot; values-only boosting;
sampling flag if the sampling-only AUC is at least 0.80; clock-matched flag if |C_all - C_clock| is at least 0.03 and
its bootstrap range (1,000 resamples of shots) excludes 0; negatives matched within 10 ms of elapsed time.

## Synthetic records (fixed now)

200 shots per dataset, every shot ends in a disruption at its last slice. One slice per ms. Four signals: AR(1)
(coefficient 0.95, innovation s.d. 0.1, stationary start) plus measurement noise s.d. 0.05. Genuine precursor in every
family: signal 0 gains 0.5 times a ramp from 0 to 1 over the last 30 ms.
- **A0, clean:** durations uniform on 0.10 to 0.50 s.
- **A_C, clock shortcut:** durations uniform on 0.18 to 0.24 s, and signal 1 gains 10 x (time in s), a slow drift
  through which a values model can read the clock.
- **A_S, sampling shortcut:** as A0, but the last 30 ms are stored at one slice every 0.25 ms.
- **A_W, harmless drift:** as A0 (wide durations) with the same drift as A_C: the clock is readable but carries little
  about the label.

Expected: sampling flag only on A_S; clock-matched flag only on A_C.

## Validation (fixed now)

Generator seeds 1 to 20 per family. Success: each expected flag fires in at least 18 of 20; each unexpected flag in at
most 2 of 20. Reported for every cell: fire count, median statistic; and per family the median clock-only AUC and the
median ordinary window AUC. No setting changed after the first table; a failed cell is reported as failed.

## Changes after the first result

(none yet)
