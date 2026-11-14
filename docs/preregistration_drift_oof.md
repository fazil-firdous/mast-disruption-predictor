# Drift study, part 2: every pair scored in every repetition (protocol)

Written 2026-11-14, after the first drift study (`docs/preregistration_drift.md`, result in NOTES.md 2026-11-14)
and before any model of this part was trained. The git commit that adds this file is its timestamp. Changes after
the first result go at the bottom with their date.

## What is already known (and why this part)

Part 1 found Delta = P_A - P_B = +0.087 for the full model, not resolved over the 50 v3 splits (-0.029 to +0.202);
pooled over pairs +0.072 (+0.037 to +0.108), conditional on the fitted models. A pair counted in a split only when
both its shots were in that split's test set (about 1 pair in 50), which is why the range over splits is wide. The
raw signals suggested that before a loss the later good repeat more often had more beam power and a larger
locked-mode amplitude. So the direction is known; this part is a frozen re-test on examined data with far more
pairs per repetition, not an independent confirmation. Its purpose is precision.

## What changes, and what stays

- **Stays:** populations A and B, their rules, the shared-time mean-score statistic, MIN_WINDOWS = 5, the 30 ms end
  margin, the three models and their inputs, the training recipe of the v3 baseline (as in
  `scripts/fast_magnetics.fit`: candidates left out of training, tuned and judged on clear endings, the same model
  family and grid). All from `scripts/drift.py`, unchanged.
- **Changes: out-of-fold scores.** Shots are put into 5 folds so that every shot of a pair, and the lost shot of an
  A triplet, sits in the same fold: shots linked through any pair of A or B form one group, and whole groups are
  assigned to folds at random (seeded), balancing the number of shots. Shots in no pair are assigned at random.
  For each fold, a model is trained with that fold as the test set and the rest split into training and
  validation at random (validation = a quarter of the rest, so 60/20/20 overall, as the v3 splits). Every shot is
  then scored once per repetition by a model that never saw it, and every pair counts in every repetition.
- **Repetitions:** 10, each with its own fold assignment (seeds 1 to 10). Runs: 5 folds x 10 repetitions x 3 models
  = 150. The full model's 50 runs come first.

## Primary

**Delta = P_A - P_B for the full model.** Point estimate: the mean over the 10 repetitions. Range (decides): a
two-level bootstrap, 2,000 draws; each draw picks one repetition at random and resamples groups of linked pairs (A
and B together) with replacement, then computes Delta. So the range includes both which pairs were observed and
which training sets were drawn. Smallest difference worth reading: 0.03.

## Reading (fixed now)

- Range excludes zero, Delta positive and at least 0.03: before a loss, the later good repeat scores higher more
  often than in repeats where no loss follows; losses come at the end of a cross-shot trend in these scores.
- Range excludes zero, Delta positive, under 0.03: a trend too small to matter for evaluation; reported as such.
- Range includes zero and excludes +0.03: no worthwhile difference; the part 1 point estimate was not borne out.
- Range includes both zero and +0.03: still unresolved at this sample size; reported as unresolved.
- Range excludes zero, Delta negative: reported as such; no reading fixed.

## Key secondary: is it the operators' beam scan?

The same Delta on **heating-matched pairs**: both shots' mean beam power over the shared time below 50 kW, or within
10% of the larger of the two. Same two-level bootstrap. Fixed reading, only if the primary is resolved positive:

- Resolved and positive on heating-matched pairs too: the trend is not only more beam power shot by shot.
- Not resolved there while the point estimate falls to under half of the primary: consistent with an operator scan
  in beam power (not proof: the matched set is smaller and different).
- Otherwise: undecided between the two.

## Other secondary (reported whatever they show)

- P_A and P_B (two-level bootstrap) for each model; the programme-only model should be near 0.5 in both.
- Delta by shot gap (1, 2-3, 4+) and by campaign, full model, two-level bootstrap.
- The range over repetitions alone (training variation only) and the pair bootstrap of repetition-averaged wins
  (pair sampling only), beside the deciding range.
- Within A: the share of triplets where K2 to K1 and K1 to L move the same way, two-level bootstrap.
- The share of heating-matched pairs in A and in B.

## Stop rule

One analysis. If the primary is unresolved, the drift question is reported as unresolved with this range; no
further re-test of the same question on these data.

## Changes after the first result

- **2026-11-14, reference for the step agreement (a secondary).** The summary printed "0.5 = no link" for the share
  of triplets where K2 to K1 and K1 to L move the same way. That is wrong: for three exchangeable shots the two steps
  share the middle shot and agree in one third of triplets by chance (simulated: 0.333, and up to about 0.36 when the
  lost shot is scored higher on average, as it is). The reference is now about 1/3. No number changes; the wording of
  the summary line does. Found after the result had been seen, and recorded here for that reason.
