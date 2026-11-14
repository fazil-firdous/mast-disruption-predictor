# Does a public multi-machine disruption dataset carry its outcome in its sampling? (protocol)

Written 2026-11-14, before any statistic below was computed. The git commit that adds this file is its timestamp.
Changes after the first result go at the bottom with their date.

## Data

The public files of Zhu et al., "Hybrid deep learning architecture for general disruption prediction across
tokamaks" (Nucl. Fusion 61, 026007, 2021), Harvard Dataverse doi:10.7910/DVN/XIOHW1, version 2:
`20ja072_fig4_data.zip` (md5 b605a694bc3ff5038dc3b27c5e91b23b) with `C_Mod_for_nn.mat`, `d3d_for_nn.mat` and
`EAST_for_nn.mat`. One row per time slice: `shot`, `time`, `time_until_disrupt` (finite only in disrupted
shots) and 12 plasma signals. These data underlie later multi-machine work, including the DisruptionBench
benchmark (Spangher et al., J. Fusion Energy 44, 26, 2025). Use with citation was confirmed by the
DisruptionBench corresponding author (2026-11-14). The data are not redistributed.

## What is already known (looked at before this protocol)

A count, with no model: the median spacing between consecutive time slices, by class, in the last 50 ms of
each shot's record and 100-500 ms before its end.
- C-Mod: 5 ms in both classes, both periods.
- DIII-D: disrupted shots 2 ms in the last 50 ms and 25 ms earlier; non-disrupted 25 ms throughout.
- EAST: disrupted shots 10 ms throughout; non-disrupted 100 ms.

Known from the literature: Zhu et al. and DisruptionBench describe a sampling increase before disruptions caused
by an equilibrium-solver setting, which both say they corrected in code that is not public; DisruptionBench
removes the last 40 ms of each shot and forward-fills every machine to a 5 ms grid.

## Question

Does the time structure of the public files, without any plasma value, tell disrupted from non-disrupted
shots, (a) in the records as published and (b) after the two preprocessing steps of DisruptionBench (last 40 ms
removed, forward-fill to 5 ms)?

## Statistics (no model, no training)

For each shot, the time used ends 40 ms before its last slice (`t_cut = t_last - 0.040 s`), as in DisruptionBench.
Label: disrupted if any `time_until_disrupt` is finite.

- **S1, spacing (primary):** the median gap between consecutive slices with `t_cut - 0.100 s < time <= t_cut`
  (the 100 ms before the cut). Shots with fewer than 2 slices there get the gap from the last two slices at or
  before `t_cut`; shots with fewer than 2 slices up to `t_cut` are left out and counted.
- **S2, change rate after forward-fill (secondary):** forward-fill all 12 signals of the shot onto a 5 ms grid
  (`t_first + k * 5 ms`, up to `t_cut`; each grid point takes the latest slice at or before it); S2 is the share of
  grid steps in the 100 ms before `t_cut` at which at least one signal changes value. This is what a model sees
  after DisruptionBench's preprocessing, without the values themselves.
- **S3, whole-record spacing (secondary):** the median gap over the whole record up to `t_cut`.

Score direction fixed now: denser sampling and more changes count as 'more disruptive' (score = -S1, S2, -S3).

## Primary quantity

Per machine, the shot-level AUC of -S1 for disrupted against non-disrupted shots, with a 95% range from 2,000
bootstrap resamples of shots (seed 42). Ties count one half.

## Reading (fixed now)

- AUC range above 0.5 and AUC at least 0.60: in that machine's public files, the sampling itself carries outcome
  information after the last 40 ms are removed; a model given these records can learn it without any plasma value.
- AUC range including 0.5: no such signal in that machine's sampling by this statistic.
- C-Mod is expected near 0.5 (uniform 5 ms sampling); it is the comparison, not a test.
- For S2: the same reading for the forward-filled records.
- Not claimed in any case: that any published model used this shortcut or that its reported performance is
  inflated. That would need that model's own pipeline, which is not public.

## Secondary (reported whatever they show)

- S2 and S3 AUCs per machine, same bootstrap.
- Median S1 and S2 by class, per machine.
- The share of shots left out per machine and class.
- S1 recomputed with cuts at 0, 20, 100 and 300 ms before the last slice instead of 40 ms (how close to the end
  the difference sits).

## Stop rule and conduct

One analysis. Before any public post, the result is sent to the dataset's and benchmark's authors.

## Changes after the first result

(none yet)
