# Landmark comparison on MAST lost discharges: do the diagnostics add next-100 ms risk beyond what the machine knows? (protocol)

Written 2026-11-28, before any statistic of this study was computed. The git commit that adds this file is its
timestamp. It applies the landmark primary validated on planted shortcuts (docs/preregistration_landmark_v3.md,
PASSED) to MAST. Changes after the first result go at the bottom.

## What this is

Conditional on a discharge being lost (a recorded disruption, or a candidate: a detected current quench in the
flat-top with flux available and no operator note), at each 20 ms window ending every 5 ms, is the current quench
10-100 ms ahead? MAST level-2 data are on a common time grid for every discharge, so the recording-schedule shortcut of
the public multi-machine files does not apply; the clock and the programme still can. The question: what do the
diagnostics add over a model that knows only elapsed time, the programmed current and the solenoid current? It
complements the twin test (a different estimand on the same archive); it is retrospective and says nothing about
deployment.

## Data and labels (fixed now)

The v3 window table with the corrected saddle-array features (`scripts/saddle_correction.load_tables`, all gates
passed), labelled with `src/dataset.labelled_windows`. Lost discharges: `disrupted` or `candidate`, with a finite
detected current quench T = `t_quench`. For each window ending at t: positive if 10 ms < T - t <= 100 ms; negative if
T - t > 100 ms; left out if T - t <= 10 ms (too late) or t >= T. Non-lost discharges are not used.

## Models (same learner for all: histogram gradient boosting, 200 iterations, learning rate 0.1, early stopping on
## 10% of the training discharges held out by discharge, keeping the iteration with the lowest held-out log loss)

- C: elapsed time since plasma start (`t_since_start`).
- P: programmed-current features and elapsed time (the twin paper's programme-and-time inputs).
- M: machine-side inputs (programmed current, solenoid current, elapsed time).
- F: every input of the corrected full model (133 columns).

## Splits

The twin paper's random splits by discharge, seeds 1 to 20 (`src/dataset.random_split`; training and test parts;
validation discharges unused), and its chronological split (M5-M7 train, M8-M9 test) as a secondary.

## Primary

Brier gain of F over M, G = mean over held-out decisions of (p_M - Y)^2 - (p_F - Y)^2 (equal weight per decision),
averaged over the 20 random splits with the corrected range over splits (Nadeau-Bengio, `src/stats.corrected_interval`).
Reading: range above 0: the diagnostics improve next-100 ms risk among lost discharges beyond the machine-side inputs.
Otherwise: no improvement shown (not shown absent).

## Secondary and reported with it

Gains of M over P, P over C and F over C (same construction); pooled decision AUC and absolute Brier loss of each
model and of a constant at the training positive share; counts (discharges, decisions, positives, too-late); the
chronological split's gains with a 95% interval from 1,000 resamples of test discharges; the primary separately for
recorded disruptions and for candidates.
Power reference: the synthetic study (NOTES.md 2026-11-23).

## Changes after the first result

(none yet)
