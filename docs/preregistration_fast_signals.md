# Faster signals: protocol, step 0 (magnetic summaries over 5 ms)

Written 2026-11-13, before any model of this study was trained and before any 5 ms feature was computed. The git
commit that adds this file is its timestamp. Changes after the first result go at the bottom with their date.

## What is already known

Every campaign and every twin pair has been looked at in the twin test (`docs/preregistration_v3.md`) and in
`docs/preregistration_how_early.md`, so this is a frozen new analysis of examined data, not a fresh confirmation.
Known: with the present features (20 ms windows) the full model's lead over a machine-side model at fixed times
before the loss is clear only in the last ~10 ms, and a 100 ms training horizon gave no resolved earlier gain.

## Question

Do the same magnetic signals, summarised over 5 ms instead of 20 ms, add discrimination between a lost shot and its
twin at a fixed time before the loss, beyond the existing full model?

The narrower hypothesis: a 20 ms window keeps a fluctuation's energy but blurs when it starts and how fast it
grows. A 5 ms window may show the onset or the growth earlier. It does not test a new diagnostic (that is step 1
and later, soft X-rays, with their own protocol).

## Signals and features

The six magnetic signals of the present model: the n=1 locked-mode amplitude (`lm`), the Mirnov RMS (`mirnov`) and
the Mirnov RMS in four bands (`mir_lf` 1-5 kHz, `mir_tm` 5-20 kHz, `mir_mf` 20-50 kHz, `mir_hf` 50-250 kHz). Each
is already on the 1 ms grid, computed from trailing 1 ms bins (`src/extras.py`, `bin_index`; tested causal).

- **5 ms summaries:** for each of the six signals, the same statistics as now (mean, standard deviation, slope,
  first-to-last change) over the samples in (t_end - 5 ms, t_end], and the same relative change (log10 of the
  window mean over the median of the shot's own history before the 20 ms window). The last value is the same
  sample in both windows and is not duplicated. Computed by the same functions as the present features
  (`src/features.py`), only the window length differs. 30 features.
- **Growth (separate contrast):** for each of the six signals, log10 of the 5 ms mean over the 20 ms mean of the same
  window (NaN where either is not positive). 6 features.
- The 1 ms band spectra are kept as they are. A 1 ms bin holds 1 to 5 cycles at 1-5 kHz, so the lowest band is
  noisy in both window lengths; this is a limitation of the present features too and is stated in the paper.
- Check before training: the same code with a 20 ms window must reproduce the stored 20 ms features; the script
  reports the largest difference.

## Models

All trained exactly like the v3 baseline (candidates left out of training, tuned and judged on clear endings, same
model family, tuning grid, label horizon of 30 ms, window ends and stride). Only the inputs change.

- **A, full (existing):** the v3 baseline runs, saved.
- **C, 20 + 5 ms (primary):** the full inputs plus the 30 five-millisecond features.
- **B, 5 ms instead of 20 ms:** the full inputs with the corresponding 20 ms features (mean, std, slope, delta and
  relative change of the six signals) replaced by the 5 ms ones.
- **G, growth:** the full inputs plus the 6 growth features.

Runs: B, C and G on random splits 1 to 20 (the splits of the v3 runs) and on the chronological split (train and
tune on M5-M7, test on M8-M9). 63 runs.

## Primary quantity

C(50 ms) of model C minus C(50 ms) of model A, with both shots of a pair scored at the same moment, 50 ms before the
lost shot's loss (the fixed-lead test of `scripts/how_early.py`, unchanged), on the common cohort of all twins,
paired over splits 1 to 20.

- 50 ms is chosen because it is the first lead beyond the 30 ms the models are trained for, as in the how-early
  study, not from any result.
- All twins rather than plan twins for the primary: the two models share every machine-side input, so the plan
  enters both alike, and all twins give about 124 pairs per split against fewer on plan twins. The same difference
  on plan twins is a required check: its sign must agree for the result to be called consistent.
- **Smallest gain worth having:** +0.03.

## Reading

- Resolved over splits (range excludes zero), positive, and at least +0.03: the 5 ms summaries add early
  discrimination at 50 ms.
- Not resolved: "summarising these magnetic signals over 5 ms instead of 20 ms adds no measurable discrimination at
  50 ms". Not that the plasma holds no earlier information.
- The pooled range over groups of pairs sharing a twin is reported beside it and does not decide.

## Secondary, all reported whatever they show

- The same difference at 10, 30 and 100 ms (descriptive; no test is claimed for them).
- B minus A and G minus A at every lead.
- On plan twins and field-and-shape twins (`plan_twins/pairs.csv`).
- The placebo (two good repeats at their lost shot's landmark) for A and C: whether the new features raise the
  ordering between good repeats as well.
- The chronological split, pooled ranges only.
- The usual metrics (pAUC, caught, false alarms), paired differences over the 20 splits.

## Stop rule

A null result here does not stop the soft X-ray audit: soft X-rays are a different observation of the plasma. The
soft X-ray study gets its own protocol after a quality audit that looks at no outcome.

## Changes after the first result

(none; the result is in NOTES.md, 2026-11-13)
