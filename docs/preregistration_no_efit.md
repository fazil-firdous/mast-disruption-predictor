# Does the twin test's plasma-signal lead need the EFIT inputs? (protocol)

Written 2026-11-14, after the MAST source-level audit (lost shots had slightly more and fresher native EFIT
reconstructions near the loss: T = 0.529 at 30 ms; NOTES.md) and before any model of this study was trained. The git
commit that adds this file is its timestamp. Changes after the first result go at the bottom.

## Question

The EFIT inputs (q95, li, beta_N, elongation, magnetic-axis height, and the Greenwald fraction, which uses EFIT's minor
radius) are the only inputs whose native schedule could differ between lost shots and twins. Does the full model's lead
over the machine-side model, the twin test's central comparison, survive without them?

## Model

**N, full without EFIT:** the inputs of the v3 baseline minus every feature that uses q95, li, betan, kappa, zmag or fgw
(`features.uses_signal`). Trained exactly like the v3 baseline (`scripts/fast_magnetics.fit`), on random splits 1 to 20
and on the chronological split. The full (A) and machine-side (M) models are the saved v3 runs.

## Primary

C(30 ms) of N minus C(30 ms) of M, fixed-lead twin test (`scripts/how_early.py`, unchanged), common cohort of all twins,
over splits 1 to 20 (Nadeau-Bengio corrected), pooled range beside it.

## Reading (fixed now)

- Resolved and positive: the plasma-signal lead over the machine-side model does not need the EFIT inputs, so a
  difference in EFIT's schedule cannot be what it rests on.
- Not resolved: without EFIT the lead is not shown at 30 ms; then the EFIT inputs, through their values or their
  schedule (not separable here), carry part of it.
- In either case also reported: A minus N at 30 ms (what EFIT adds), with its range.

## Secondary

The same differences at 10, 50 and 100 ms; on plan twins; the chronological split (pooled); the placebo for N; the usual
metrics of N against A.

## Changes after the first result

- **2026-11-14, the second clause of the "not resolved" reading does not apply.** That clause ("then the EFIT inputs
  carry part of it") presupposed that the full model's own lead over the machine-side model is resolved at 30 ms on
  splits 1 to 20. Computed after the result (exploratory context, not registered): A minus M at 30 ms is +0.052 (-0.021
  to +0.126), not resolved either. So the registered statement stands only as written in its first clause: without
  EFIT the lead is not shown at 30 ms, and neither is it with EFIT on these 20 splits. Recorded here because the
  comparison was computed after the result was seen.
