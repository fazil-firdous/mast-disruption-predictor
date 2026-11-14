# Do MAST losses come at the end of a cross-shot trend? (protocol)

Written 2026-11-14, before any number of this study was computed. The git commit that adds this file is its
timestamp. Changes after the first result go at the bottom with their date.

## What is already known

The twin test's placebo (`docs/preregistration_v3.md`; `scripts/v3_explore.py`): for a lost shot L with two earlier
same-programme kept shots, K1 (the nearest) and K2 (the next), K1 scores above K2 in about 54% of pairs, for the
full model and the machine-side model alike. Every model score used here has been looked at in earlier studies
(the twin test, how early, plan twins, faster signals), so this is a frozen new analysis of examined data.

## Question

Is the later-scores-higher ordering among good repeats specific to sequences that end in a loss (an approach to
the loss across shots), or is it ordinary drift between repeats whatever follows?

## Populations

Shared rules (as `scripts/twin_check.py` and `scripts/twin_differences.py`): same campaign; looking back at most
REACH = 30 shots; 'kept' = flux exhausted, programmed ramp-down or smooth end; 'lost' = recorded or candidate;
'same programme' = programmed current within 10% of its peak (RMS) and peaks within 10% (`twin_check.compare`).

- **A, before a loss:** the triplets of `scripts/twin_differences.py` (`twin_differences/triplets.csv`): the pair
  (E, F) = (K2, K1).
- **B, no loss follows:** for every kept shot F with a programme, E is the nearest earlier kept shot with the same
  programme as F (lost shots in between are skipped, as for A). The pair (E, F) is in B only if the next shot after
  F within REACH with the same programme as F is not lost (kept, or there is none). Pairs of A are not in B by
  construction (their F is followed by the lost L).

## Statistic (identical for A and B; no landmark is borrowed)

For one model and one saved run, a pair counts if both shots are in that run's test set. Each shot's score is the
mean of its saved test-window scores whose window ends lie in the shared time: from the later of the two shots'
flat-top starts (`v3_runs.flat_top_start`) to 30 ms before the earlier of the two shots' last windows (30 ms = the
label horizon, so neither shot's own ending enters). At least 5 windows of each shot are needed. win = 1 if F's
mean is higher, 0.5 if equal, 0 otherwise. P = mean win over the pairs that count.

Models: the full model (`v3/baseline`, primary), the machine-side model (`v3/machine`) and the programme-only model
(`v3_explore/programme`). Runs: every saved random split of these models (splits 1 to 50 of the v3 runs).

## Primary

**Delta = P_A - P_B for the full model**, paired split by split, with the Nadeau-Bengio corrected range over splits
(`src/stats.corrected_interval`). Beside it, not deciding: the pooled range over pairs (each pair's win averaged
over the splits that tested it), with groups of pairs linked through shared shots resampled together (2,000
resamples, A and B together). Smallest difference worth reading: 0.03.

## Reading (fixed now)

- Delta resolved (range over splits excludes zero), positive and at least 0.03: the shots before a loss rise in
  score across repeats more than ordinary repeats do; losses come at the end of a cross-shot trend in these scores.
  An observation about the scores, not a mechanism.
- Delta not resolved, and P_B resolved above 0.5: ordinary drift between repeats; for the twin test the fair
  reference is P_B, not 0.5.
- Delta not resolved and P_B not resolved above 0.5: no drift is shown by this statistic; the earlier 0.54 belongs
  to that statistic and the time it scored.
- Delta resolved and negative: reported as such; no reading fixed in advance.

## Secondary (reported whatever they show)

- P_A and P_B over splits for each model. The programme-only model should be near 0.5 in both (a check that the
  programme matching works).
- P_A and P_B by shot gap F - E (1, 2-3, 4 or more) and by campaign, full model, pooled.
- Without any model: for line-average density, radiated power, beam power, locked-mode amplitude and the
  Greenwald fraction, the share of pairs where F's mean over the same shared time is higher, A against B, pooled.
- Within A: the share of triplets where the step K2 to K1 and the step K1 to L have the same sign, full model,
  with L's mean taken over the shared time of K1 and L ending 30 ms before L's loss.
- The number of pairs in A and B, their shot gaps and campaigns side by side.

## Stop rule

One analysis, no retraining, saved scores only. If B's gaps or campaigns differ strongly from A's, the gap-matched
comparison (secondary) is reported next to the primary and the wording says so; the primary is not replaced.

## Changes after the first result

(none yet)
