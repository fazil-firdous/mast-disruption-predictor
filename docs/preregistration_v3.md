# v3 pre-registration

Written 2026-11-09, before any v3 model was trained or scored. The git commit that adds this file is its timestamp.
Any change after the first v3 result is listed at the bottom with its date and reason, and the paper reports it.

## Data and splits (unchanged from v2)

- The 11,471 labelled FAIR-MAST level-2 shots, the v2 features, gradient boosting (`SWEEP_MODEL = "hgb"`).
- Shot-level splits 60/20/20, seeds 1-50. **Splits 1-20 are for development** (choosing settings, fixing bugs).
  **Splits 21-50 confirm**, and only the tests below are run on them, once.
- Baseline: the confirmed v2 recipe with candidates left out of training (`clear-drop-candidates`, E15; +9.3 points
  on splits 21-50, NOTES 2026-11-06).

## Twin pairs (fixed before training)

`events/programme_twins.csv` from `scripts/twin_check.py` at commit 61737cb: for each shot lost in the flat-top
(recorded or candidate), the nearest earlier shot within 30, same campaign, same programmed current (RMS difference
under 10% of the peak, peaks within 10%), that was not lost. 3,902 pairs. A pair is used in a split only when both
shots are in its test set (expected about 4% of pairs, about 150 per split).

## Primary test: H1, the matched-pair test

For each pair, both shots are scored over the same programme time: from the start of the flat-top to
`t_loss - MIN_WARNING_MS`, where `t_loss` is the lost shot's quench time. Each shot's score is its highest window
score in that range. The pair counts 1 if the lost shot scores higher, 0.5 for a tie, 0 otherwise.
**C** = the mean over the pairs of a split.

- **H1a:** C of the baseline model is above 0.5 (corrected resampled t-test on C - 0.5 over splits 21-50, one-sided).
- **H1b:** C of the baseline model is above C of the machine-only model (`machine` recipe, which sees only the
  programme and the solenoid), paired by split, Nadeau-Bengio corrected.
- **Expected if the shortcut story is right:** the machine-only model's C is near 0.5, because twins share the
  programme by construction.

## Secondary tests (Holm-Bonferroni over H2 and H3 together)

- **H2, evidence labels:** recipe `evidence` = the baseline, plus candidates whose post-shot comment names a loss of
  the plasma within 20 ms of the quench (`words_vote = 2`) labelled as disruptions at that time. All other
  candidates stay out. Compared with the baseline on the same clear-ending test shots, metric pAUC (caught vs false
  alarms up to 10% false alarms). Hypothesis: pAUC is higher.
- **H3, alarm with hysteresis:** the alarm fires when the score passes an upper threshold and stays above a lower one
  for a hold time. All three are tuned on validation shots, like the present threshold. Compared with the present
  rule (two windows in a row over one threshold), same model and shots, metric pAUC. Hypothesis: pAUC is higher.

## Reported, not tested (no p-values claimed)

- C on the subset of twins whose programmes match exactly (RMS difference under 1%).
- Catch rate and warning time per operator-named cause, and the alarm time against the operator-written event time.
- Catch rate on Thornton's full-performance disruptions; the temporal split (train on earlier campaigns, test on
  M9).
- The label-model P (`scripts/mast_label_model.py`) as soft labels. Comparison only: it encodes our solenoid rule,
  which the model can see, so it cannot be a fair winner.

## Decision rule

- The v3 default recipe changes from the baseline only if its hypothesis (H2 or H3) holds after Holm on splits 21-50.
  Development results on splits 1-20 choose nothing beyond the tuned settings listed above.
- H1 is reported whatever the result. A C near 0.5 for every model would mean no model reads the plasma well
  enough to tell twins apart, and the paper would say so.

## Cut from v3 for time

The shortcut penalty in training, the time-to-disruption output and cause-specific windows. H1 tests the shortcut
directly; the other two move to later work.

## Changes after the first v3 result

(none yet)
