# Endpoint module 2: does the evaluation of the public Zhu et al. files depend on where the control is evaluated? (protocol)

Written 2026-11-21, before any statistic of this study was computed. The git commit that adds this file is its
timestamp. It follows the MAST endpoint module (docs/preregistration_endpoint.md) and an outside review of its result.
Changes after the first result go at the bottom.

## What this is, and what it is not

A file-relative stress test. In the public files (Harvard Dataverse doi:10.7910/DVN/XIOHW1, version 2) a disrupted
record ends at the disruption and a non-disrupted record ends wherever its record ends. We compare a frozen values-only
predictor's matched concordance when a non-disrupted control is evaluated (E) 40 ms before its own last slice, or (M)
at the same elapsed time since its first retained slice as the case's decision point. Elapsed time since the first
retained slice is a record-relative comparator, **not a phase proxy**: the first slice need not mark breakdown or the
flat-top, and the files carry no phase markers. Nothing here reproduces the published evaluation of Zhu et al. or
diagnoses any published model. Closest prior design: Do et al. (2026, sepsis), control anchors at the end of the record
against onset-matched pseudo-onsets.

## Data, split, model (fixed, trained once)

Per machine (C-Mod, DIII-D, EAST): the fig4 `*_for_nn.mat` file; the label of the audit (disrupted if any
`time_until_disrupt` is finite). Split by shot: EAST the release's own lists (`EAST_data_partition.mat`); DIII-D and
C-Mod the audit's seeded split (seed 42, 60/20/20). Model: the harmonisation study's values-only boosting
(HistGradientBoostingClassifier, max_iter 200, learning rate 0.1, seed 42) on its 36 R0 features (last, mean, last
minus first of the 12 signals over the slices in the 300 ms before a landmark, only slices at or before the landmark),
trained once on the training shots with every shot's landmark at its last slice minus 40 ms, as in that study. The
model and preprocessing are then frozen. The test shots only are used below.

## The two arms

For a test case c (disrupted): t0_c its first slice, its decision point d_c = t_last,c - 40 ms, elapsed time
e_c = d_c - t0_c, rounded down to 10 ms. The case is scored at d_c in both arms.
For a test control k (non-disrupted) and a case c, the pair (c, k) is eligible if k's record has at least one slice
in the 300 ms before t0_k + e_c and its last slice is at least 40 ms after t0_k + e_c (so that, as at E, at least 40 ms
of record follows the evaluation point). No minimum separation between the two control points is required; their
separation is reported. Then:
- **M:** k scored at L_M = t0_k + e_c;
- **E:** k scored at L_E = t_last,k - 40 ms.
The same eligible (c, k) combinations enter both arms. Matched concordance per arm: for each case, the share of its
eligible controls that it outranks (ties one half), averaged over cases. Delta = C_E - C_M.

## Primary

EAST (the release's own test list): Delta, with a 95% range from 2,000 bootstrap resamples of test shots (cases and
controls resampled independently, with replacement, seed 42; each resample recomputes the eligible combinations'
weights).

## Reading (fixed now, two-sided)

- |Delta| at least 0.03 and its range excluding 0: in these files, the measured discrimination of this frozen values-only
  predictor depends materially on where the control is evaluated (file-relative; not a physical phase claim).
- Otherwise: no material dependence is shown for this predictor and these records (not shown equivalent).

## Secondary and reported with it

- The same for DIII-D and C-Mod (C-Mod's sub-microsecond time offsets make its window edges uncertain by one slice; its
  result is reported with that caveat).
- The usual shot-level AUC of the frozen model (controls at E), for reference.
- Eligibility losses: cases with no eligible control, controls never eligible, by machine.
- The separation between L_M and L_E (median, quartiles), and the share of combinations with L_M within 40 ms of L_E.
- Observation-process audit at both control points: median number of slices in the 300 ms window and age of the last
  slice, for controls at M and at E, and a timing-only concordance (score = number of slices in the window) in both
  arms.
- Paired negative-score change (control score at E minus at M), mean and median.

## Changes after the first result

(none yet)
