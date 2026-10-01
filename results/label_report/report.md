# Label reconciliation report (SPEC 5.3)

Shots processed: 550 (400 uniform random shots for the headline numbers, the rest are extra note-disrupted shots used for the timing comparison and example plots). Shots with `unknown` signal label (|Ip| never above 100000 A): 0.

## 2x2 table, uniform random sample

|  | signal no | signal yes |
|---|---|---|
| note no | 93 | 285 |
| note yes | 0 | 22 |

Operator notes mark 5.5% of this sample as disrupted. The signal detector marks 76.8%.

## Timing where both exist

65 shots have a note time and a detected quench. t_disrupt minus t_note (ms): median 1.0, quartiles -2.0 to 6.0, 78% within 10 ms.

## Does a slow ramp-down look different from a quench?

Time for |Ip| to fall from 90% to 10% of peak, median over shots: 5.0 ms for the 453 shots flagged disrupted, 124.0 ms for the 97 shots not flagged. Shots with no 10% point before the data ends: 0 of the not-flagged ones.

## Are noted and un-noted quenches different?

Medians over shots flagged by the detector:

|  | quench_rate | peak_ip | fall_ms |
|---|---|---|---|
| note no | 1.92e+08 | 7.66e+05 | 5 |
| note yes | 2.01e+08 | 7.82e+05 | 4 |

## Sensitivity to the starting-current floor

| CQ_MIN_FRAC_OF_PEAK | flagged | share |
|---|---|---|
| 0.2 | 363 | 0.907 |
| 0.35 | 339 | 0.848 |
| 0.5 | 307 | 0.767 |
| 0.65 | 270 | 0.675 |

## Summary

- 98% of note-disrupted shots (168 of 172) show a quench in |Ip|, so a note is rarely contradicted by the signal.
- The detector finds a quench in 77% of the uniform sample, far more than the notes. Flagged shots with and without a note have similar quench rates and fall times (table above), so the current trace alone does not separate a noted disruption from an ordinary end of pulse.
- Where both exist the two agree on timing (above), so the detector's time is usable when it is right about the event.
- Neither label has independent ground truth here. The final choice of `LABEL_SOURCE` is the owner's call (SPEC 5.3).

Plots: `disagreements_note_only.png`, `disagreements_signal_only.png`, `disagreements_timing.png`. Per-shot table: `per_shot.csv`. Note-only list for manual review: `review_list.csv`.
