# MAST: do EFIT values repeat differently before losses? (exploratory protocol)

Written 2026-11-14, before any statistic below was computed. The git commit that adds this file is its timestamp.
Exploratory: it cannot recover native EFIT reconstruction times, only value-repetition signatures on the published
grid. Changes after the first result go at the bottom with their date.

## Why

The MAST timing self-audit (`docs/preregistration_mast_timing_audit.md`) found every twin pair tied, because FAIR-MAST
level 2 publishes equilibrium on a fixed 5 ms grid by zero-order hold (each point takes the latest source value). If
the native reconstructions were sparser in some shots, the published series would hold the same value over several
5 ms points. So adjacent equal values are a trace, although not a proof, of a sparser source (a new reconstruction
can also return the same value, and a stationary plasma can repeat values).

## Data

- Published EFIT grid points: the slice times collected for the self-audit (`efit_times/efit_times.parquet`).
- Values: the six EFIT variables of the twin test's predictor (`config.EFIT_VARIABLES`: minor radius, q95, li,
  beta_N, elongation, magnetic-axis height), read from the cached extras (`data/extras/`), where each published value
  is held from its slice time plus EFIT_DELAY_MS. The value of slice i is the extras value at the first grid time at
  or after slice time i + delay.
- Pairs: the twin pairs (`events/programme_twins.csv`), as in the self-audit; the same programme-time landmarks.

## Statistic

For a shot, a lead h, t = t_loss(L) - h, and the slices whose values are available in (t - 100 ms, t]:
- **R (primary measure):** among adjacent pairs of those slices, the share where all six variables are exactly
  equal (NaN compared as missing: an adjacent pair with any NaN is left out, never counted as a change).
- **R_any (secondary):** the share where at least one variable is exactly equal.
- **RUN (secondary):** the longest run of slices with all six exactly equal, in slices.

Rounding tolerance: none (exact equality), fixed now.

Paired comparison as in the self-audit: for each pair and lead (both shots still running at t, L's flat-top
started, both with at least 3 such slices), win = 1 if L repeats more (larger R), 0.5 if equal, 0 if less. T = mean win;
range from resampling groups of pairs that share a shot (2,000, seed 42). Leads 10, 30, 50, 100 ms; primary R at 30 ms.

## Reading (fixed now)

- T(R, 30 ms) within 0.47 to 0.53 with a range including 0.5: no difference in value repetition between lost shots and
  twins at the moment compared.
- T resolved outside 0.47 to 0.53: value repetition differs. Not read as a shortcut or a source-schedule difference
  (plasma dynamics can change it too); reported, and the FAIR-MAST maintainers' answer on native time stamps decides
  what follows.
- Otherwise: reported as found.

## Secondary

All three measures at all leads, T and the mean paired difference; the share of adjacent pairs with all six equal over
the flat-top, lost against kept shots (medians).

## Changes after the first result

(none yet)
