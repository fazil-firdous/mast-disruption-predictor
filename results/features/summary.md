# Phase 3 feature check

Shots: 400. Windows: 26360. Features per window: 30. Infinite values: 0. Shots with no windows: 2 (25424, 25470).

## Windows per shot

| status | shots | median | min | max |
|---|---|---|---|---|
| clean | 229.0 | 74 | 0.0 | 136.0 |
| disrupted | 171.0 | 55 | 18.0 | 113.0 |

## Causality on real shots

Each shot was cut at a random window end and its windows recomputed from the cut data only. 13248 rows compared, all identical (exact equality).

## Missing values (share of windows)

| feature | NaN share |
|---|---|
| prad_mean | 0.199 |
| prad_std | 0.199 |
| prad_last | 0.199 |
| prad_slope | 0.199 |
| prad_delta | 0.199 |
| neutrons_mean | 0.042 |
| neutrons_std | 0.042 |
| neutrons_last | 0.042 |
| neutrons_slope | 0.042 |
| neutrons_delta | 0.042 |
| pnbi_mean | 0.090 |
| pnbi_std | 0.090 |
| pnbi_last | 0.090 |
| pnbi_slope | 0.090 |
| pnbi_delta | 0.090 |
| prad_over_pnbi | 0.270 |
| prad_over_ip | 0.199 |
| neutron_slope_rel | 0.042 |
| pnbi_on | 0.090 |

Missing values come from signals absent for the whole shot (`has_<signal>` = 0); see NOTES.md.

Plot: `shot_11860.png`.
