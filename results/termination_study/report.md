# Termination study

Quench shots from `results/label_report/per_shot.csv`: 337 measured, 1 failed to load. Times are the detector's quench times.

## Programmed current at the quench

| group | shots | median ref fraction | share >= 0.9 |
|---|---|---|---|
| noted | 170 | 1.000 | 0.953 |
| unnoted | 167 | 1.000 | 0.928 |

## Central solenoid current at the quench

| group | median (kA) | 25% (kA) | 75% (kA) |
|---|---|---|---|
| noted | -33.6 | -40.4 | -23.3 |
| unnoted | -37.0 | -42.9 | -29.6 |

AUC for 'more negative solenoid current means unnoted': 0.596 (permutation p = 0.0022, 10000 shuffles). Most negative solenoid current seen: -66.5 kA.

Per campaign:

| campaign | shots | AUC |
|---|---|---|
| M5 | 51 | 0.53 |
| M6 | 83 | 0.52 |
| M7 | 111 | 0.62 |
| M8 | 45 | 0.63 |
| M9 | 39 | 0.69 |
| Unknown | 8 | 0.75 |

Plot: `solenoid_at_quench.png`. Per-shot values: `per_shot.csv`.
