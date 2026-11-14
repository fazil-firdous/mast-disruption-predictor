# Does the twin test's late lead need the locked-mode inputs? (protocol)

Written 2026-11-14, after reading the FAIR-MAST release history (NOTES.md 2026-11-14: in the September 2026 files that
our data come from, saddle coils M02, M08 and M012 have their field sign reversed from shot 24782, so our n=1
locked-mode amplitude `lm_n1` is fitted with three inverted coils in those shots) and before any model of this study was
trained. The git commit that adds this file is its timestamp. Changes after the first result go at the bottom.

## What is already known

The no-EFIT check (`docs/preregistration_no_efit.md`) found the full model's lead over the machine-side model resolved
at 10 ms on splits 1 to 20 (A minus M +0.119, +0.063 to +0.174) and not at 30, 50 or 100 ms, with or without EFIT. That
is why 10 ms is the primary lead here, chosen with that knowledge.

## Model

**L, full without the locked-mode inputs:** the v3 baseline's inputs minus every feature that uses `lm`
(`features.uses_signal`). Trained exactly like the baseline on splits 1 to 20 and the chronological split. A and M are
the saved v3 runs.

## Primary

C(10 ms) of L minus C(10 ms) of M, fixed-lead twin test (`scripts/how_early.py`), common cohort, over splits 1 to 20.

## Reading (fixed now)

- Resolved and positive: the 10 ms lead over the machine-side model does not need the locked-mode inputs, so the
  saddle-coil sign fault cannot be what it rests on.
- Not resolved: without the locked-mode inputs the 10 ms lead is not shown; the locked-mode inputs, faulty in part of
  the data, carry part of it, and the twin paper's revision must say so (and the result should be recomputed with
  corrected saddle signs).

## Secondary

A minus L at every lead (what the locked-mode inputs add); the same on shots before 24782 only (pairs whose lost shot is
below 24782) and from 24782 on; the chronological split (its test campaigns are mostly from 24782 on), pooled; the
placebo; the usual metrics.

## Changes after the first result

(none yet)
