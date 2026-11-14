# What a values-based predictor keeps when the observation schedule is harmonised (protocol)

Written 2026-11-14, after the sampling audit (parts 1 and 2) and the timing-only predictor (NOTES.md 2026-11-14), and
before any model of this study was fitted. The git commit that adds this file is its timestamp. A sensitivity
analysis, as an outside reviewer framed it: a performance drop is not automatically "the shortcut", and what remains
is not automatically "physics". Changes after the first result go at the bottom.

## Task (the same as the timing studies)

Shot-level: is the shot disrupted (any finite `time_until_disrupt`)? The time used ends 40 ms before the shot's last
slice (`t_cut`), and features come from the 300 ms window (t_cut - 300 ms, t_cut]. Endpoint-aligned and
retrospective. Files and splits as in `docs/preregistration_timing_model.md`: EAST on its official split (primary),
DIII-D and C-Mod on our seeded split.

## Three representations of the same records

- **R0, published slices:** the slices in the window as they are.
- **R1, forward-fill to 5 ms (DisruptionBench-like):** a grid every 5 ms from the shot's first slice to t_cut; each
  grid point takes the latest slice at or before it (no maximum age).
- **R2, harmonised:** a grid every Delta from the shot's first slice to t_cut, where Delta is the machine's median
  slice spacing in non-disrupted shots, fixed now from the audit: EAST 100 ms, DIII-D 25 ms, C-Mod 5 ms. Each grid
  point takes the latest slice at or before it; if that slice is older than Delta, the point is empty. The rule is
  the same for both classes, and its origin is the first slice. (On the EAST grid the window holds 3 points, on
  DIII-D 12 and on C-Mod 60.)

## Features (plasma values; the same for every representation)

For each of the file's 12 plasma signals, over the window's points (slices for R0, grid points for R1 and R2): the
last value, the mean, and the last minus the first value. 36 features. No standard deviation (its estimate depends
on how many points there are). No time stamp, count or age is an input.

## Model

Histogram gradient boosting (scikit-learn, `max_iter=200`, `learning_rate=0.1`, everything else default, seed 42),
fitted on training shots only, evaluated **once** on test shots. No tuning (so validation shots are not used). Same
model and settings for every representation and machine.

## Residual timing check on R2 (before reading any R2 result)

The age of the observation behind each R2 grid point (grid time minus the slice time used), averaged over the window,
as the only input of a timing-only logistic regression (as in the timing study) on the same split: its test AUC is the
**residual timing AUC**. The share of empty R2 points by class is reported.

## Primary

On EAST's official test shots: **Drop = AUC(R1) - AUC(R2)**, with a 95% range from 2,000 paired bootstrap resamples of
test shots (seed 42). Beside it: AUC(R0), AUC(R1), AUC(R2) and the residual timing AUC.

## Reading (fixed now)

- **Residual timing AUC above 0.60 (range above 0.5):** harmonisation did not remove the timing difference; the R2
  result is not read as "after harmonisation", only reported.
- Otherwise, **Drop resolved and at least 0.05:** a sizeable part of the R1 performance depends on the observation
  schedule that harmonisation removes, or on short-timescale information that it also removes; the two cannot be
  separated here.
- Otherwise, **Drop under 0.05 or not resolved:** the R1 performance largely survives harmonisation.
- In every case: what R2 keeps is not called pure physics; R2 also discards genuine fast measurements.

## Secondary

- The same for DIII-D and C-Mod (C-Mod's Delta equals R1's grid, so its R1 and R2 differ only by the maximum age).
- The timing-only model of the timing study on the same test shots, beside the values models.
- Median observation age at R1 and R2 points by class.

## Changes after the first result

(none yet)
