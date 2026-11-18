# How much does an outcome-associated observation schedule alone inflate a values-only predictor? (protocol)

Written 2026-11-18, before any data of this study were read or any model of it trained. The git commit that adds this
file is its timestamp. Changes after the first result go at the bottom.

## Why

In the public DIII-D and EAST files of Zhu et al. (2021) the observation schedule separates disrupted from
non-disrupted shots (timing-only AUC 0.998 to 1.000; NOTES.md 2026-11-14). Values-only models also score 0.98 to 0.99
there, on every representation we tried, and the harmonisation study could not remove the schedule difference, so
it could not say how much of a values-only score the schedule explains (technical note, "not established"). In those
files the question cannot be answered: the dense measurements exist only in disrupted shots.

C-Mod in the same release is sampled uniformly (5 ms in both classes; timing-only AUC 0.53 to 0.61). That allows a
controlled experiment: impose an EAST-like schedule on C-Mod's own measurements, so that the measurements are
identical and only the schedule differs, and measure the change in a values-only predictor. This is a different
question from the harmonisation study (which tried to remove a schedule; its stop rule concerned further attempts at
that), and no representation is tuned here.

## Data

`C_Mod_for_nn.mat` from Harvard Dataverse doi:10.7910/DVN/XIOHW1, version 2 (fig4 zip, MD5
b605a694bc3ff5038dc3b27c5e91b23b); label as in the audit (disrupted if any `time_until_disrupt` is finite); the 12
plasma signals of the file. Shots with fewer than 2 slices in either condition are left out of both.

## The two conditions (the measurements kept are the same values in both, only which slices are kept differs)

Per shot, slices sorted by time, indexed i = 0..n-1 (n-1 the last slice):
- **U (uniform, control):** keep slice i if (n-1-i) is a multiple of 10 (every tenth slice, counted back from the last,
  so about every 50 ms in both classes and the last slice always kept).
- **I (injected):** as U for non-disrupted shots; for disrupted shots, as U plus every slice within 300 ms of the last
  slice. This mimics the EAST pattern (dense segment of 100 to 300 ms before the end, about ten times the sparse
  cadence) on C-Mod's own measurements.

## Representations, model, split (as in the harmonisation study, unchanged)

The cut is 40 ms before each shot's last slice; the window is the 300 ms before the cut. R0: features from the slices
in the window; R1: 5 ms forward-fill grid from the first slice; R2: grid every 50 ms from the first slice with a
maximum age of 50 ms (the class-independent sparse grid; our approximation of the released correction). Features:
last, mean and last minus first of each of the 12 signals (36). Model: HistGradientBoostingClassifier(max_iter=200,
learning_rate=0.1, random_state=42). Split: by shot, seeded (42) and stratified, 60/20/20; train on train, test once on
test, in each condition separately (a benchmark whose train and test share the schedule). The same test shots in both
conditions.

## Checks before reading (fixed now)

- **Injection check:** the timing-only predictor of the timing study (occupancy, spacing, largest gap; logistic
  regression, C chosen on validation) must reach a test AUC of at least 0.80 with its range above 0.5 under I.
  Otherwise the injection did not create a schedule shortcut and nothing below is read.
- **Control check:** under U the same predictor's test AUC range must include 0.5. Otherwise U is not schedule-free;
  the primary is reported but read only as "relative to a control that itself carries some timing information".

## Primary

On the test shots, R0: **Inflation = AUC(values-only, I) - AUC(values-only, U)**, with a 95% range from 2,000 paired
bootstrap resamples of test shots (seed 42).

## Reading (fixed now)

- Inflation resolved (range above 0) and at least 0.02: a label-associated schedule of this shape, on identical
  measurements, raises a values-only predictor's test AUC by that much.
- Resolved and below 0.02: a measurable but small inflation.
- Range includes 0: an inflation is not shown for a values-only predictor of this kind (it may still be large for
  predictors given time stamps).
- In every case: this is C-Mod with an imposed schedule, not EAST or DIII-D; it measures what this kind of schedule
  can do, not what it did in any published model.

## Secondary

- The same inflation on R1 and R2 (does forward-fill keep it; does the class-independent grid remove it?).
- Timing-only AUC under U and I; values-only AUC under each condition and representation.
- A time-aware variant: the 36 value features plus the three timing features, I minus U on R0.
- The median number of slices in the window by class and condition.

## Changes after the first result

(none yet)
