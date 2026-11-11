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

## Implementation details (2026-11-09, `scripts/v3_runs.py`, before any v3 run)

These settle details the text above left open. None was chosen after seeing a v3 number.

- **Machine-only model:** trained like the baseline (same labels, candidates left out, judged on clear endings);
  only the inputs differ, so H1b compares inputs alone.
- **Evidence positives** are labelled like recorded disruptions: the HORIZON_MS (30 ms) before the detector's quench
  time, in training only. Validation and test are unchanged.
- **Hysteresis grid:** lower threshold = ratio x upper with ratio in {1, 0.8, 0.6}, hold in {1, 2, 3, 4} windows
  (5 ms apart). The rule fires when the score hold-1 windows earlier was over the upper threshold and every score
  since is over the lower one. Ratio 1 with hold 2 is the present rule. The rule with the best validation pAUC is
  chosen per split (ties to the present rule). Upper thresholds are swept like the present threshold.
- **H1 time range:** the flat-top starts when the lost shot's programmed current first reaches 90% of its peak (the
  first window if the programme is missing). A twin pair counts only when both shots have windows in the range.
- **One-sided p-values:** halved two-sided corrected p when the mean change is in the predicted direction.
- **Disk:** test window scores are saved as float32 parquet per run (about 2 MB, 150 runs).

## Changes after the first v3 result

Nothing in the registered tests above was changed. The analyses below were added after the registered results were
known; they are exploratory, and the paper labels them so.

### 2026-11-10 and 11: exploratory analyses (results known when this was written)

The ladder of inputs (programme-only and solenoid-only models), C on exact and loose programme matches, by beam
heating, by early height, by how long the twin went on, by lost-shot group and by lead time; the model-free twin
differences; the same-time statistic (both shots cut to the time they share) and C pooled over pairs with a range
from resampling groups of pairs that share a twin. Code: `scripts/v3_explore.py`, `scripts/twin_differences.py`.

### 2026-11-11: the chronological split (written before any of its output was seen)

Code as committed at 36cd1a6, started on the server before this note; no output had been read when this was written.

- **Runs:** programme-only, machine-side (registered name: machine-only) and baseline, trained like the v3
  baseline. Training and every tuned setting (model parameters, alarm threshold) use campaigns M5-M7 only, with a
  validation slice from those campaigns; campaigns M8-M9 are the test set and are used once.
- **Pairs:** the programme twins fixed before v3 (commit 61737cb) whose two shots are both in the test set.
- **Reported, whatever the result:** for each model the usual metrics (pAUC, caught, false alarms) and C with the
  registered statistic and with the same-time statistic, on all test pairs and on same-heating pairs; C for
  recorded and for candidate lost shots; and the paired difference baseline minus machine-side on identical pairs.
- **Ranges:** one split, so ranges come from resampling groups of pairs that share a twin. They condition on the
  fitted models and do not cover what a different training set would give.
- **Reading:** this is one test, not a replication. C above 0.5 with a range clear of it supports the within-pair
  result on campaigns the model never trained on. A range that includes 0.5 is uncertainty, not evidence of no
  skill. Either way the result is reported next to the random-split result, and no setting is changed after it.

### 2026-11-11, later: four final checks (written before any of them was run)

All four use the saved scores of the random-split runs, are exploratory, and are reported whatever they show. No
further analysis is added to this study after them.

- **Placebo: two good repeats.** For each lost shot with two same-programme kept shots K1 (nearer, later) and K2
  (earlier), the twin test is run on K1 against K2 as if K1 were the lost shot: both scored from the lost shot's
  flat-top start to 5 ms before the lost shot's loss, same-time statistic, pairs counted when K1 and K2 are both in
  a split's test set. Ranges: over splits, and pooled with groups of shots connected through any shared shot
  resampled together (a shot can be K1 in one triplet and K2 in another). *Reading:* a value above 0.5 means the
  later of two good repeats tends to score higher under this protocol, an asymmetry that could also be present in
  the main test; a value near 0.5 is reassuring and does not prove the main test unbiased. It is reported beside
  the main result and is not subtracted from it.
- **Early beam heating.** Pairs are also split by beam heating compared only from the flat-top start to 100 ms
  before the loss (the boundary already used for the early height), so that a beam that changed in response to the
  event near the loss does not decide the match. A different design, not a correction for confounding.
- **One other summary of a shot's scores.** Besides the highest score (registered, and its same-time form), the
  score at each shot's last shared window. It asks whether the lost shot is ranked higher at one fixed time before
  the loss. All three summaries are reported.
- **Recorded minus candidate.** The difference between the pooled C of recorded and of candidate lost shots, with
  groups of pairs sharing a twin resampled jointly and the difference recomputed in each resample.

### What each range covers

- *Over splits* (corrected resampled t-test): the model changing with the training set and the test shots drawn;
  not one kept shot serving as the twin of several lost shots within a split.
- *Over groups of pairs sharing a twin* (cluster bootstrap on saved scores): shared twins; conditions on the fitted
  models.
- The corrected resampled t-test is kept as the pre-specified approximate analysis. Its coverage has not been checked
  for matched pairs with shared twins. Ranges for the chronological split are conditional on the fitted models and
  on that one change of campaigns.
