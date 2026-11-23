# Audit toolkit, all-disruptive landmark primary v3: validation with early stopping on fresh seeds (protocol)

Written 2026-11-23, before this validation was run. The git commit that adds this file is its timestamp. It follows
the failed v2 validation, investigation 1 and the power study (docs/preregistration_landmark_mode.md; NOTES.md
2026-11-23). Changes after the first result go at the bottom.

## What changes from v2, and why (all decided after v2's results; stated as such)

- Both learners use early stopping (validation fraction 0.1 of the training decisions, 10 rounds without improvement,
  seed 42): v2's negative gain on null data came partly from overfitting (investigation 1).
- Precursor strength 1.0 instead of 0.5: the power study showed the primary cannot resolve strength 0.5 at 200 shots
  (1/20) and resolves strength 1 every time (20/20). Choosing a strength the primary can see is a design choice made
  from the power study, not from these seeds; the validation tests specificity and that the decision rule works where
  power is adequate, not power itself.
- The clock-matched contrast is reported only, with no pass/fail: v2 showed it moves under a rising hazard with no
  clock proxy (V 14/20), so it is a sensitivity measure, not a detector.

Everything else as in v2: 200 shots, 1 ms ticks, four AR(1) signals, 20 ms burn-in, hazards (constant 0.005 per tick;
rising 0.002 (1 + t / 0.05 s)), drift 10 x t on signal 1 added after all random draws, W 20 ms, s 5 ms, H 30 ms, lead
2 ms, Brier gain with 1,000 shot-cluster resamples, flag if its range is above 0.

## Families and rule (fixed now)

N, N+D, V, V+D (no precursor) and P, P+D (constant hazard, precursor 1.0 x a 0-to-1 ramp over the last 30 ms on
signal 0). Generator seeds 41 to 60 (unused so far). Expected: the primary fires in P and P+D (at least 18 of 20) and
not in N, N+D, V, V+D (at most 2 of 20 each). Full table reported whatever the outcome; nothing tuned after it.
Seeds 41 to 60 are then spent.

## Changes after the first result

(none yet)
