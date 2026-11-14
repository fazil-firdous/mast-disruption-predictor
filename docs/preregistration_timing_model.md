# A timing-only predictor on the public files' own held-out split (protocol)

Written 2026-11-14, after the sampling audit parts 1 and 2 (NOTES.md 2026-11-14) and before any model of this study
was fitted. The git commit that adds this file is its timestamp. Changes after the first result go at the bottom.

## Question

Can a predictor that sees only observation times, trained without test labels, tell disrupted from non-disrupted
shots on the held-out test shots that the public files themselves designate?

The fixed-statistic AUCs of the audit already show retrospective separability. This study asks the narrower,
model-based question an outside reviewer suggested, on an official split, evaluated once.

## Data and split

- EAST: `EAST_for_nn.mat` and `EAST_data_partition.mat` from the fig6 data of Zhu et al. (Harvard Dataverse
  doi:10.7910/DVN/XIOHW1, v2). The split is the file's `trainingShotlist`, `validationShotlist` and `testShotlist`,
  used as given. Shots of the lists that are not in `EAST_for_nn.mat` are counted and left out.
- DIII-D and C-Mod (secondary): the files carry no split, so a seeded split by shot (seed 42; 60/20/20, stratified by
  class), clearly labelled as ours.
- Label: disrupted if any `time_until_disrupt` is finite.

## Features (timestamps only; no plasma value is read)

One row per shot. The time used ends 40 ms before the shot's last slice (as in the audit and DisruptionBench).
- **occ:** the share of 5 ms grid steps in the 100 ms before the cut that hold at least one slice time
  (`scripts/sampling_audit_2.flag_share`).
- **gap:** the median slice spacing in that 100 ms (`scripts/sampling_audit.spacing`).
- **maxgap:** the largest slice spacing in that 100 ms, including the step from the last slice before the window
  into it; NaN with fewer than 2 slices up to the cut.
Missing values are replaced by the training set's median (fitted on training shots only).

## Model

Logistic regression on the three features, standardised with the training set's mean and spread. The inverse
regularisation strength C is chosen from {0.01, 0.1, 1, 10} by AUC on the validation shots. The chosen model is
refitted on training shots only and evaluated **once** on the test shots.

## Primary

Test AUC on EAST's official test shots, with a 95% range from 2,000 bootstrap resamples of test shots (seed 42).
Beside it, on the same test shots, the fixed-statistic AUCs of occ and of minus gap (no fitting).

## Reading (fixed now)

- Test AUC range above 0.5 and AUC at least 0.80: a predictor trained only on observation times discriminates
  disrupted from non-disrupted shots on the files' own held-out test shots.
- Range above 0.5, AUC under 0.80: reported as above chance but weaker than that.
- Range including 0.5: not shown by this predictor.

Endpoint-aligned and retrospective: the cut uses where the record ends, so this is not an online warning system,
and nothing here says that any published model used these features.

## Secondary

- The same model and statistics for DIII-D and C-Mod on our seeded split.
- Class counts in each split and the number of list shots missing from the data file.
- The chosen C and the validation AUC.

## Changes after the first result

(none yet)
