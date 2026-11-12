# How early? Protocol

Written 2026-11-12, before any number of this study was computed. The git commit that adds this file is its
timestamp. Changes after the first result go at the bottom with their date and reason.

## What is already known, and what that means

This study reuses the shots, the twin pairs and the models of the twin test (`docs/preregistration_v3.md`). Every
campaign has been looked at there, so nothing here is a fresh confirmation; a protocol written now constrains the
choices still to be made and nothing more. Seen already, with a different summary (each shot's highest score up to
a cut-off): the full model's twin score falls from 0.68 near the loss to about 0.60 at 50 ms and 0.55 at 100 ms.
A clean confirmation would need a second machine or new data.

## Question

How long before a loss does a model that sees the plasma tell the lost shot from its same-programme twin better
than a model that sees only the machine side, when both shots are compared at one fixed moment?

## Definitions

- **Pairs.** The 3,902 programme twins of the twin test, unchanged.
- **Landmark.** For a pair and a lead h, the time T = (the lost shot's quench time) minus h. The quench time only
  places the landmark for this look back; no model sees it.
- **Score at the landmark.** For each shot, the score of its latest window that ends at or before T and no more than
  one window stride (5 ms) earlier. A shot without such a window has no score there.
- **Eligible pair at lead h.** T is in the lost shot's flat-top (not before its flat-top start, where that is
  known) and both shots have a score at the landmark. A twin that has already ended is therefore left out at that
  lead and is not carried forward.
- **C(h).** The share of eligible pairs in which the lost shot's score at the landmark is the higher one; ties
  count half.
- **Leads.** 10, 30, 50, 100 and 200 ms.
- **Common cohort.** Pairs eligible at every one of 10, 30, 50 and 100 ms. Trends over these four leads are read on
  this cohort, so that a change of pairs is not mistaken for a change with time. 200 ms is reported on its own
  eligible pairs and on the pairs eligible at all five leads.
- **Models.** Full (the v3 baseline), machine-side (programmed current and solenoid), programme-only, solenoid-only;
  all compared on identical pairs.

## Part A: the models that exist (saved scores, no training)

These models were trained to warn 30 ms ahead.

- **Primary quantity:** C(50 ms) of the full model minus C(50 ms) of the machine-side model, on the common cohort,
  over the 50 random splits (corrected range over splits) and pooled over pairs (range over groups of pairs
  sharing a twin). 50 ms is the first lead beyond the 30 ms the models were trained for.
- **Secondary, all reported:** C(h) of each model at every lead on the common cohort and on each lead's own
  eligible pairs, with the number of pairs; the full-minus-machine-side difference at every lead; the same on the
  chronological split (trained on M5-M7, tested on M8-M9; pooled range only); the same on pairs with the same
  early beam heating.
- **Placebo:** K1 against K2 with the landmark taken from their lost shot, at every lead, eligible when both have a
  score there. Reported beside the main numbers, not subtracted.

## Part B: a longer training horizon (new training, run only after Part A is written up)

One factor is changed: the positive windows of training reach 100 ms before a recorded disruption instead of 30 ms.
Features, model family, tuning grid, splits and everything else stay as in the v3 baseline.

- **Runs:** the full and the machine-side model with the 100 ms horizon, on random splits 1 to 20 and on the
  chronological split.
- **Primary quantity:** C(100 ms) of the 100 ms-horizon full model minus C(100 ms) of the 30 ms-horizon full model,
  on identical pairs of the common cohort, paired over splits 1 to 20.
- **Secondary:** the same difference at 10, 30 and 50 ms (a longer horizon may cost late discrimination); the
  usual metrics (pAUC, caught, false alarms); the full-minus-machine-side difference with the 100 ms horizon.
- A 200 ms horizon and finer time resolution are not part of this study.

## Ranges and reading

- Two ranges, side by side, as in the twin test: over splits (the model changing with the training set; does not
  allow for shared twins) and pooled over pairs with groups sharing a twin resampled together (allows for shared
  twins; conditions on the fitted models). The chronological split has the pooled range only.
- A difference whose range over splits includes zero is called not resolved over training sets, whatever the
  pooled range says.
- If the full model's lead over the machine-side model at 50 ms is not resolved, the conclusion is that these
  models and features do not show an earlier advantage, not that the plasma holds no earlier information.
- Everything listed is reported whatever it shows. Anything added later is listed below as an addition.

## Changes after the first result

- **2026-11-12, after Part B's output had been read. Addition: a range for the usual metrics.** The protocol lists
  the usual metrics of Part B as a secondary but gives them no range. Added: the difference (100 ms horizon minus
  30 ms horizon) in pAUC, share caught and share of false alarms, paired over splits 1 to 20, with the corrected
  range over splits. Seen before this was written: the means (full model pAUC 0.341 and 0.363, caught 36.2% and
  39.7%; machine-side pAUC 0.182 and 0.237, caught 19.0% and 26.5%) and the twin-test results of Part B. Not seen:
  the ranges. Reading, fixed now: if a model's pAUC difference is resolved over splits while its twin-test
  difference at 100 ms is not, the longer horizon raised the usual metric without a measurable gain in telling a
  lost shot from its twin earlier. If the pAUC difference is not resolved either, the longer horizon changed
  nothing that this study can measure. No model, pair or lead is changed.
- **Same date, same state of knowledge. Addition: the machine-side model's own change with the horizon.** The
  protocol lists C(h) with the 100 ms horizon minus the 30 ms horizon for the full model only. Added: the same
  difference for the machine-side model at 10, 30, 50 and 100 ms, paired over splits 1 to 20 on the common
  cohort, because the machine-side model's usual metrics moved more than the full model's. Seen before this was
  written: the machine-side model's C with the 100 ms horizon on splits 1 to 20 (0.626, 0.591, 0.588, 0.563) and
  with the 30 ms horizon on all 50 splits (0.616, 0.608, 0.569, 0.530). Not seen: the paired difference. It is
  read with the rule above.
