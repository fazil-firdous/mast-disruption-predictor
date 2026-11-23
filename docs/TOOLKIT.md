# Disruption-predictor audit toolkit

One command that asks: **is a disruption predictor reading the plasma, or the clock and the way the data were
recorded?** It runs on your own computer. Nothing is uploaded; it prints summary numbers and writes them to
`audit_report.txt` next to your table.

```
python scripts/audit.py your_table.csv                     # machines where some shots disrupt and some do not
python scripts/audit.py your_table.csv --all-disrupt \
       --window-ms 20 --step-ms 5 --horizon-ms 30 --lead-ms 2  # machines where every shot disrupts
python scripts/audit.py --demo                             # synthetic example, no data needed
```

Requirements: Python 3.10+, numpy, pandas, scipy, scikit-learn (pyarrow for parquet). Typical run: seconds to a few
minutes for a few hundred shots.

## Input: one row per shot and time slice

| Column | Meaning |
|---|---|
| `shot` | shot number (any id) |
| `time` | seconds |
| any other numeric column | a signal (Ip, density, Mirnov amplitude, ...); empty values allowed |
| `disrupted` | mixed mode only: 1 if the shot disrupts, 0 if not; one value per shot |
| `t_disrupt` | optional, seconds: the disruption time; default the shot's last slice |
| `prog_*` | optional, all-disruptive mode: values known before the shot (requested current, pulse length); used in the baseline |

Examples: `examples/example_mixed.csv`, `examples/example_all_disruptive.csv`. The tool stops with a plain message
for repeated (shot, time) rows, a `disrupted` value that changes within a shot, a disruption time before the first
slice, or a single class in mixed mode, and warns for very few shots, times that look like ms, and mostly empty
signals. Use `--time-resolution` to give the clock precision documented for your data.

## Mixed mode (some shots disrupt)

Every shot is evaluated 40 ms before its last slice, with the 300 ms before that point.
- **T, timing alone:** how well the recording schedule (slices, age of the last slice, largest gap) separates the
  classes. Flag at AUC 0.80. If flagged, any score can partly come from how the data were recorded.
- **D, clock alone:** the same for elapsed time since the first slice.
- **E, endpoint:** a values-only model's matched concordance when each non-disrupted shot is evaluated at its own end
  versus at the disrupted shot's elapsed time. Flag if |Delta| is at least 0.03 and its bootstrap range excludes 0.
  The range's half-widths are printed: with a small test set they are wide and a 0.03 change cannot be seen.

## All-disruptive mode (every shot disrupts, e.g. small tokamaks without a controlled ramp-down)

There, the "positive" windows always sit at the end of a shot, so elapsed time is informative by construction, and
any signal that drifts with time lets a model read the clock.
- **Primary, fixed-horizon landmark comparison:** decisions every `step` from `window` after the first slice; a
  decision is positive if the disruption comes 2 to 30 ms later (too-late decisions are left out and counted). Two
  models with the same learner (gradient boosting with early stopping): elapsed time (+ `prog_*`) alone, and the same
  plus the diagnostic values. The number is the **paired Brier gain** on held-out shots, with a range from resampling
  shots. Above 0: the diagnostics improve the next-30 ms risk beyond the baseline you gave it.
- **Secondary:** decision AUCs, elapsed-time-alone and sampling-alone AUCs, and a clock-matched concordance (each
  positive against negatives of other shots at the same elapsed time). The clock-matched number shows sensitivity to
  the comparison rule; it moves under a rising hazard even without a clock proxy, so it is not a leakage detector.

**Power.** On synthetic shots with about 200 shots, the primary resolved a precursor lifting the decision AUC to about
0.78 in 20 of 20 runs, and one lifting it to about 0.63 in 1 of 20 (13 of 20 with 500 shots). "No improvement shown"
on a small machine is not evidence that no precursor exists.

## How it was tested (synthetic records with planted shortcuts; every protocol committed before its run)

| What | Protocol | Result |
|---|---|---|
| Mixed mode T, D, E | `docs/preregistration_planted_leaks.md` | 11 of 12 cells pass; **FAILED** one (E 3/20 on the duration family, limit 2). Follow-up: E fired 2/100 on new clean data. |
| All-disruptive screen (sampling, clock-matched) | `docs/preregistration_planted_leaks_all.md` | 7 of 8 pass; **FAILED** on a "harmless drift" family whose clock was in fact informative. |
| Landmark primary v2 | `docs/preregistration_landmark_mode.md` | **FAILED**: no power at the planted precursor, negative gain from overfitting. Early stopping and a power study followed. |
| Landmark primary v3 | `docs/preregistration_landmark_v3.md` | see NOTES.md |

Failures are kept as failures; explanations found afterwards are marked as such. The tests are software validations
on synthetic data, not evidence about any machine.

## What it does not do

It does not tell you that a model is physically right, does not replace an alarm-level evaluation (warning time,
false alarms per shot), and does not fix labels: check how your disruption times were defined first.
