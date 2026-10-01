# Label reconciliation report (SPEC 5.3)

Sample: 400 shots, made of every note-disrupted shot in the pool (175) plus 225 random note-clean shots (seed 42). Because note-disrupted shots are over-represented, the overall agreement below is not the agreement on a typical shot; read the rates per row. Shots labelled `unknown` (|Ip| never above 100000 A): 0.

## 2x2 table

|  | signal no | signal yes |
|---|---|---|
| note no | 58 | 167 |
| note yes | 4 | 171 |

- Overall agreement: 57.2% of shots.
- Note-disrupted shots that the detector also flags: 97.7% (171 of 175).
- Note-clean shots that the detector flags anyway: 74.2% (167 of 225).

## Timing where both exist

99 shots have a note time and a detected quench. t_disrupt minus t_note (ms): median 1.0, quartiles -2.0 to 5.0, 78% within 10 ms, 8 differ by more than 20 ms.

## What does a flagged shot look like?

Time for |Ip| to fall from 90% to 10% of peak (median): 4.0 ms for the 338 flagged shots, 124.0 ms for the 62 not flagged.

Medians over flagged shots, split by note:

|  | quench_rate | peak_ip | fall_ms |
|---|---|---|---|
| note no | 1.85e+08 | 7.64e+05 | 5 |
| note yes | 2.02e+08 | 7.81e+05 | 4 |

## Normal ramp-downs (SPEC 5.2 step 5)

Three random note-clean shots with a clear slow end, 90%->10% fall of at least 50 ms (`rampdowns.png`): shots 24559, 28104, 29989, fall times 127, 155, 115 ms, detector result clean, clean, clean. Of the 62 unflagged shots, 2 fall faster than that (shortest 24 ms). They are borderline cases, often a spike and a partial drop followed by a final fall that starts below the CQ_MIN_FRAC_OF_PEAK floor.

## Sensitivity to the starting-current floor

| CQ_MIN_FRAC_OF_PEAK | flagged, note clean | flagged, note disrupted |
|---|---|---|
| 0.2 | 0.884 | 0.989 |
| 0.35 | 0.844 | 0.983 |
| 0.5 | 0.742 | 0.977 |
| 0.65 | 0.64 | 0.96 |

Plots: `disagreements_note_only.png`, `disagreements_signal_only.png`, `disagreements_timing.png`, `rampdowns.png`. Per-shot table: `per_shot.csv`. Note-only list for manual review: `review_list.csv`. Exact shot list: `sample_ids.json`.
