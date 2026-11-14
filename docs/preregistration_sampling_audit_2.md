# Sampling audit, part 2: checks an outside reviewer asked for (protocol)

Written 2026-11-14, after part 1 (`docs/preregistration_sampling_audit.md`, result in NOTES.md 2026-11-14) and the
MAST self-audit, and before any statistic below was computed. The git commit that adds this file is its timestamp.
Changes after the first result go at the bottom with their date. No model is fitted.

## What is already known

Part 1, with the time used ending 40 ms before each shot's last slice: the slice spacing alone gives a shot-level
AUC of 1.000 (DIII-D) and 0.999 (EAST), 0.526 (C-Mod); after forward-fill to 5 ms, the share of steps where any value
changes gives 0.998, 0.999 and 0.522. EAST's whole-record spacing gives 0.981. S1 at a 300 ms cut gave 0.302 in EAST.
The MAST data have fixed time bases (T = 0.500).

## Why these checks

1. Part 1's S2 looks at values (a change needs a changed value), so it is not a pure timing statistic.
2. "EAST is separable from the start" was not tested: S3 ends at the endpoint-relative cut.
3. EAST's 0.302 at a 300 ms cut is unexplained.
4. "40 ms before the last slice" need not be "40 ms before the disruption".
5. A database-merge artefact (classes from different shot ranges) cannot be told from a solver setting unless class
   and shot range overlap.

## Checks (all per machine, same files, same label: disrupted if any `time_until_disrupt` is finite)

- **C1, observation flags (pure timing):** on the 5 ms grid from the first slice to the 40 ms cut, a grid step is
  flagged if at least one slice time falls in it (`(g - 5 ms, g]`); no value is read. Statistic: the share of
  flagged steps in the 100 ms before the cut. Shot-level AUC (score = share), 2,000 bootstrap resamples of shots.
- **C2, fixed early window:** the median spacing of slices with `t_first + 100 ms < time <= t_first + 300 ms`, on the
  cohort of shots whose record (t_last - t_first) is at least 600 ms and which have at least 2 slices in the window.
  AUC of minus the spacing; the cohort size per class is reported.
- **C3, the cut diagnostic:** on the cohort of shots with a record of at least 1.0 s (the same cohort at every cut),
  for cuts of 0, 20, 40, 100 and 300 ms before the last slice: per class, the number of shots, the median number of
  slices in the 100 ms before the cut, and the AUC of minus S1 (part 1's spacing). Plus the median record length per
  class on the full set.
- **C4, last slice against the disruption time:** for disrupted shots, t_last - t_disrupt, with t_disrupt the median of
  `time + time_until_disrupt` over the shot's slices: median, 5th and 95th percentile, per machine.
- **C5, shot range:** shots split into ten equal-count bins by shot number; per bin, the number of each class and
  the AUC of minus S1 where both classes have at least 20 shots; the mean of those within-bin AUCs, and the number
  of bins that have both classes.

## Readings (fixed now)

- **C1:** AUC range above 0.5 and AUC at least 0.60: observation times alone, with no value read, carry outcome
  information after the 40 ms cut and on a 5 ms grid. Otherwise not, by this statistic.
- **C2:** the same rule: if met, the sampling differs by class early in the record, away from its end; if not, part
  1's separation is an end-of-record property and "from the start" is not supported.
- **C5:** within-bin AUCs averaging at least 0.90 over bins that have both classes: the separation is not a
  difference between shot ranges. If fewer than three bins have both classes, the two cannot be told apart from
  these files, and the report says so.
- **C3, C4:** descriptive, reported whatever they show; no reading fixed.

## Not claimed

That any published model used this, or that any reported performance is inflated.

## Changes after the first result

(none yet)
