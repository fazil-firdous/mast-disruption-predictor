# MAST at source level: do the native EFIT reconstruction times differ between lost shots and their twins? (protocol)

Written 2026-11-14, before any level-1 reconstruction time of a twin pair was read. Looked at before writing: the
level-1 `efm` group of three shots chosen without regard to outcome (30421, the documentation's example, and 27000 and
22000, used as examples since October): `all_times` and `cnvrgd_times` exist, are identical in those shots, and have
irregular gaps (5, 10, 15, 20, 25, 30, 35 and 50 ms; median 5 ms). No outcome of those shots was looked at. The git
commit that adds this file is its timestamp. Changes after the first result go at the bottom with their date.

## Why

The level-2 self-audit (`docs/preregistration_mast_timing_audit.md`) measured publication-grid coordinates (equilibrium
held onto a fixed 5 ms grid by zero-order hold) and found every pair tied, which says nothing about the native schedule.
FAIR-MAST level 1 documents `efm/all_times` (attempted reconstructions) and `efm/cnvrgd_times` (converged ones). The
gaps above show that the native schedule is irregular, so the question of whether it differs before losses is real.

## Data

- For every shot of the window table, the level-1 `efm` group's `all_times` and `cnvrgd_times`, read on GitHub
  (`scripts/efit_times_fetch.py --level1`), stored as one row per attempted time with a converged flag.
- A converged reconstruction is taken as available at its time + EFIT_DELAY_MS (5 ms, the pipeline's assumption; not
  a measured latency, and not wall-clock completion).
- Pairs, landmarks and eligibility exactly as in the level-2 self-audit: twin pairs, t = t_loss(L) - h in both shots,
  h = 10, 30, 50, 100 ms, both shots still running at t (summary grid reaches t), L's flat-top started.

## Measures (per shot at t)

- **SE1 (primary measure):** converged reconstructions available in the 20 ms window (t - 20 ms, t].
- **SE2:** the age at t of the latest available converged reconstruction.
- **SE3:** the median gap between converged reconstructions available in (t - 100 ms, t].
- **SA1:** attempted reconstructions in the 20 ms window that did not converge.

## Statistic

Timing concordance T: the share of pairs where the lost shot is observed more densely (more SE1, smaller SE2 and SE3,
more SA1 counts as "more"), ties one half; groups of pairs sharing a shot resampled together (2,000, seed 42).
**Primary: SE1 at 30 ms.** Key: SE2 at 30 ms.

## Reading (fixed now; two-sided)

- T within 0.47 to 0.53 and its range including 0.5: no difference in native converged-reconstruction density at the
  moment compared.
- Range entirely above 0.53 or entirely below 0.47: the native schedule differs between lost shots and twins. Then a
  further protocol, written before it runs, tests whether that difference reaches the predictor's inputs (the level-2
  held values) and changes prediction; the twin paper's journal is told if a central claim changes.
- Otherwise: reported as found, no reading fixed.
- Not claimed in any case: online availability, real-time latency, or that the twin test's predictor used it.

## Secondary

All measures at all leads, T and mean paired differences; unpaired, lost against kept shots, SE1 in the 20 ms window
ending 40 ms before each shot's last grid sample (AUC); the share of shots whose level-1 `efm` group is missing or
whose `all_times` and `cnvrgd_times` differ.

## Changes after the first result

(none yet)
