# Draft: Label quality and physics signals for disruption prediction on the MAST spherical tokamak

Working title. Skeleton started 2026-10-15. **Every number below is TBD until it comes from the full run on the owner's server**; the section notes say which script produces it. Do not fill anything by hand.

## Abstract
TBD — written last.

## 1. Introduction
- Disruptions and why prediction matters (Vega et al. 2022; de Vries et al. 2011).
- Most ML predictors are trained on conventional tokamaks (DIII-D, JET, C-Mod, EAST); MAST is a spherical tokamak, the design of the UK STEP programme.
- FAIR-MAST (Jackson et al. 2024) makes MAST data open; UKAEA baselines exist (IAEA FEC) with current-only labels.
- Our questions: (1) how good are the disruption labels? (2) what can global and magnetic signals predict, honestly evaluated?

## 2. Data
- FAIR-MAST level-2 summary signals (1 ms) and the extra signals of `src/extras.py`; campaigns M5-M9. Table of shot counts — TBD (`scripts/build_dataset.py --scale full`).
- Causal features, window and horizon definitions (SPEC 4, 6, 7).

## 3. Labels
- Operator notes vs current-quench detector (Phase 2 report, `results/label_report/`): the detector confirms 97.7% of note-disrupted shots and also flags 74.2% of note-clean shots (400-shot sample).
- Termination study (`results/termination_study/`): both kinds of quench happen during the programmed flat-top; unnoted ones sit slightly closer to the end of the solenoid swing (AUC 0.60, p = 0.002).
- Gold labels by hand (`src/gold.py`): agreement of each automatic label with ~100 blind human labels — TBD.
- PU learning (E9): estimate of how many unnoted quenches behave like disruptions — TBD on the full dataset.

## 4. Methods
- Models: physics baseline, logistic regression, random forest, HistGradientBoosting (SPEC 9).
- Alarm rule and shot-level metrics (SPEC 10-11); validation-chosen threshold; test split used once.
- Conformal threshold (E8) and its behaviour under campaign drift (E6).
- Robustness: seed sweep (E10), shortcut test without time-since-start (E11).

## 5. Results
- Main table E0-E3 — TBD.
- Signals: ablation (E5) and magnetics (E13) — TBD.
- Labels: E7 and gold-label agreement — TBD.
- UKAEA-style protocol (E12) — TBD; compare with their published numbers if they become available.
- Temporal generalisation (E6) and conformal guarantee (E8) — TBD.

## 6. Discussion and limitations
- What global signals can and cannot do on MAST; what magnetics add.
- Label policy as a first-order effect on reported performance.
- Limits: saddle-coil n=1 without vacuum-pickup subtraction; equilibrium data only for later campaigns; offline evaluation.

## 7. Reproducibility
- Code: this repository (public); data: FAIR-MAST (CC BY-SA 4.0); seeds and configs saved with every result (`config.json`).

## References
See `docs/literature_review.md` (every entry checked; full-text checks still needed where marked).
