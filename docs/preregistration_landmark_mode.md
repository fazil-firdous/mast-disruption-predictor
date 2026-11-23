# Audit toolkit, all-disruptive mode v2: fixed-horizon landmark comparison, and its planted-leak validation (protocol)

Written 2026-11-23, before any of it was coded or run. The git commit that adds this file is its timestamp. It
follows an outside review (NOTES.md 2026-11-23): the primary for machines where every shot disrupts should ask whether
past diagnostic values improve the prediction of an event in the next H ms beyond a flexible elapsed-time baseline;
the clock-matched contrast becomes secondary. The earlier all-disruptive validation (A_W failure) stays as recorded.
Changes after the first result go at the bottom.

## The analysis (what the toolkit will compute)

Input: shot, time, signals; per shot an event time (t_disrupt, default the last slice); optional programme columns
named `prog_*` (constant within a shot, known before the shot).

- **Decision grid, forward from the origin:** the origin is the shot's first slice. Decisions at L = t0 + W + k s
  (W window, s stride), for every L before the event. At each L only slices in (L - W, L] are used.
- **Labels:** positive if the event falls in (L + lead, L + H]; negative if the shot is event-free through L + H;
  decisions with the event in (L, L + lead] are too late to warn, are left out of both classes and are counted. A shot
  stops contributing after its event.
- **Two models, same learner and settings** (HistGradientBoosting, max_iter 200, learning rate 0.1, seed 42), trained
  on the training shots' decisions (seeded 60/20/20 split by shot):
  - baseline: elapsed time L - t0, plus the `prog_*` columns if present;
  - augmented: the same plus the value features of the window (last, mean, last minus first of each signal).
- **Primary:** paired Brier gain on the test shots' decisions, equal weight per decision: mean over decisions of
  (baseline squared error - augmented squared error); positive means the values help. 95% range from 1,000 bootstrap
  resamples of test shots as clusters (seed 42).
- **Reading:** range above 0: past diagnostic values improve H-ms risk prediction beyond the baseline actually given.
  Otherwise: no improvement shown (not shown absent). Always with: which baseline variables were available, the counts
  (decisions, positives, too-late decisions, shots), and calibration in the large (mean predicted against observed
  positive share) for both models.
- **Secondary:** the clock-matched contrast C_all - C_clock of the augmented model as before (a measure of
  sensitivity to the comparison rule, not of leakage), and the clock-only and sampling-only AUCs.

Default settings for the validation: W 20 ms, s 5 ms, H 30 ms, lead 2 ms.

## Validation on synthetic records (fixed now)

200 shots per dataset, one slice per ms, four signals (AR(1) coefficient 0.95, innovation s.d. 0.1, stationary start;
measurement noise s.d. 0.05). A burn-in of 20 ms with no event; after it, on every 1 ms tick each running shot
disrupts with probability p(t); the shot's last slice is its event. Every shot eventually disrupts.

- Constant hazard: p = 0.005 per tick (next-30 ms risk 1 - 0.995^30, about 0.14, the same at every elapsed time).
- Time-varying hazard: p(t) = 0.002 (1 + t / 0.05 s), rising with elapsed time.
- Precursor (where present): signal 0 gains 0.5 times a ramp from 0 to 1 over the 30 ms before the event.
- Drift (where present): signal 1 gains 10 x (time in s), from which elapsed time can be read.

Families, each with and without drift built from the same random draws (same event times, noise and split), so the
drift is the only difference within a pair:
- **N, N+D:** constant hazard, no precursor.
- **V, V+D:** time-varying hazard, no precursor.
- **P, P+D:** constant hazard with the precursor.

Expected primary (Brier gain range above 0): P and P+D only; not in N, N+D, V, V+D (in V+D the drift carries only
the clock, which the baseline already has).
Expected secondary clock-matched flag (|C_all - C_clock| at least 0.03, range excluding 0): not in N+D (readable but
uninformative clock); fires in V+D (readable and informative clock). Reported, not required, for the other four.

Generator checks reported for every family: the next-30 ms positive share against the analytic 0.14 under constant
hazard, and the correlation of the drift signal with elapsed time.

Seeds 1 to 20 per family. Success: expected flags in at least 18 of 20, unexpected in at most 2 of 20. The full table is
reported whatever the outcome; nothing is tuned after it.

## Changes after the first result

(none yet)
