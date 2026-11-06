# How every MAST shot ended (automatic termination labels)

`per_shot.csv`: one row per labelled FAIR-MAST level-2 shot (11,471 of 11,573; the rest never reached 100 kA), made by `scripts/auto_labels.py` (rules in `src/termination.py`) on the v2 dataset, 2026-11-01. Statistics: `python scripts/label_statistics.py` (with `MAST_HOME` unset it reads this file).

| Column | Meaning |
|---|---|
| `shot_id` | MAST shot number |
| `campaign` | M5-M9, or Unknown |
| `note` | the operator's post-shot comment mentions a disruption and does not negate it (`src/labels.py`) |
| `t_quench` | time (s) of the fast current quench (>= 80% fall within 10 ms) found by the detector; empty if none |
| `ref_frac` | programmed plasma current at the quench as a fraction of its maximum |
| `sol_ratio` | central-solenoid sweep rate over the 20 ms before the quench, as a fraction of its flat-top rate |
| `termination` | `flat_top_drop` (lost during the programmed flat-top with flux left), `flux_exhausted` (solenoid ratio < 0.25), `programmed_rampdown` (ref_frac < 0.8), `smooth_end` (no fast quench), `unknown` (signals missing) |

The classes say when the plasma was lost, not whether it disrupted. They are algorithmic and not checked by a MAST expert.

Derived from FAIR-MAST (UKAEA), licence CC BY-SA 4.0; this table is shared under the same licence. Cite S. Jackson et al., SoftwareX 27, 101869 (2024).
