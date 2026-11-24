# Endpoint module 2: does the evaluation of the public Zhu et al. files depend on where the control is evaluated? (protocol)

Written 2026-11-21, before any statistic of this study was computed. The git commit that adds this file is its
timestamp. It follows the MAST endpoint module (docs/preregistration_endpoint.md).
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

### Amendment 1 (2026-11-21, written AFTER the registered result had been seen; post-result sensitivities)

The registered run finished and its full output was read before this amendment (NOTES.md 2026-11-21: EAST Delta
+0.003, +0.000 to +0.007, not material; DIII-D +0.009; C-Mod +0.036, material by the rule). A design check made after the run had started suggested verification items and at most two sensitivities. Everything
below is a post-result sensitivity, not part of the confirmatory test; the registered output is kept unchanged and
remains the result.

**Verification (before any interpretation).** EAST: number of IDs in each of the release's lists; duplicates within a
list; IDs in more than one list; listed IDs not in the fig4 file and file shots in no list; whether the lists hold shot
numbers (all matched IDs are present as `shot` values). All machines: shots with repeated or unsorted timestamps. The
features use only slices at or before the landmark (`at`: slices in (L - 300 ms, L]); no fill, interpolation or
whole-record transform. Bootstrap clarification: within a resample a case enters with its multiplicity, its eligible
controls with theirs; a sampled case with no sampled eligible control is left out of that resample's average (as the
code already did); the share of such case draws is now reported. The model uses no validation data. The AUC line is
"ordinary AUC, one score per shot, controls at E"; no M AUC is defined.

**S1, common recorded history (no refit).** From the registered eligible pairs, keep those where the case, E and M
each have at least 300 ms of record before their landmark (landmark minus the shot's first slice >= 300 ms); same frozen
scores, same Delta and bootstrap. Report cases, unique controls and combinations before and after. No other cutoff.

**S2, mixed-landmark training (one additional fit per machine).** Same training shots, features, algorithm and
settings. Training cases stay at last slice - 40 ms. Each training control, with probability 1/2 (numpy
default_rng(20261121), drawn in shot order), is moved to an M-like landmark t0_k + e, with e drawn uniformly from the
training cases' rounded elapsed times that are eligible for that control under the registered rule (a slice in the
300 ms before, at least 40 ms of record after); with no eligible e it stays at E (counted). One row per training shot.
The refit is frozen and evaluated on the exact registered test pairs and landmarks: C_E, C_M, Delta with the same
bootstrap, beside the registered model's. Report how many controls moved and how far. No other seed, proportion or
tuning.

Scope: EAST stays primary; DIII-D and C-Mod are run with the same rules and reported as descriptive (C-Mod is where the
registered rule read "material", so leaving it out would hide the most relevant check). Reading: if the Deltas of the
two models differ, that is dependence on the training landmark choice, not evidence about which model is better.
