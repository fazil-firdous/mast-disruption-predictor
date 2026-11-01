# Draft: How MAST plasmas end, and what disruption labels reward

Working title. Skeleton started 2026-10-15, rebuilt around the label audit on 2026-10-31. **Every model number is TBD until it comes from the full v2 run on the private research server**; each section names the script that produces it. The label facts quoted are the ones already reproduced in `results/` and `NOTES.md` (2026-10-25). Nothing is filled in by hand.

## Abstract
TBD, written last. Planned shape: open MAST data are being used for machine-learning disruption prediction with labels taken from the plasma current alone. Classifying how each of the 11,573 FAIR-MAST shots ended, from the programmed current and the central-solenoid sweep at the current quench, shows that (i) a large share of current-quench "disruptions" are flux-exhausted or programmed endings, and (ii) a large share of shots without an operator disruption note were lost during flat-top with flux left. We test what each label set rewards with a predictor that never sees the plasma (E14), and whether the doubtful labels hide predictive skill (E15), with error bars corrected for overlapping splits.

## 1. Introduction
- Disruptions and why prediction matters (Vega et al. 2022; de Vries et al. 2011). Mature predictors on conventional tokamaks: DIII-D, C-Mod, EAST (Rea, Granetz et al.), JET; deep learning across machines (Kates-Harbeck et al. 2019); a cross-machine benchmark (DisruptionBench 2025).
- MAST is a spherical tokamak, the line of the UK STEP programme. FAIR-MAST (Jackson et al. 2024, 2025) made its data open; UKAEA event-prediction benchmarks (IAEA FEC) and TokaMark (Rousseau, Jackson et al. 2026) build on it. The FAIR-MAST disruption benchmark labels disruptions from the plasma current alone.
- Prior MAST disruption work: DECAF on the MAST database (Berkery et al.), Thornton's thesis (2011). Nothing we found audits MAST's operator notes against the termination physics, or separates MAST endings by solenoid flux (literature review, 2026-10-25). We say "could not find", not "has not been done".
- Questions: (1) How did each MAST shot end, and how do the operator notes and current-only labels relate to that? (2) What does a predictor trained on each label set actually learn? (3) How much skill do global, magnetic, density and equilibrium signals give, honestly evaluated?

## 2. Data
- FAIR-MAST level 2, every shot of the official shot table (11,573; campaigns M5-M9 and 'Unknown'). Shot counts after quality cuts: TBD (`scripts/build_dataset.py --scale full`, dataset message in `results/REPORT.md`).
- Signals at 1 ms: plasma current, radiated power, neutron rate, beam power; saddle-coil n=1 amplitude; Mirnov RMS and four band RMS values (1-5, 5-20, 20-50, 50-250 kHz); line-average density and a causally built Greenwald fraction; EFIT q95, li, beta_N, elongation, magnetic-axis height (each slice used 5 ms after its time); programmed and solenoid current (`src/extras.py`; data facts in NOTES 2026-10-30).
- Causal windows: 20 ms every 5 ms, plus each fluctuation's change against the shot's own earlier median and 100 ms trends; no future samples, enforced by tests (`tests/test_features.py`).

## 3. How MAST shots ended (the label audit)
- Three label sources: operator note, current-quench detector (>= 80% fall within 10 ms), and both. On the 400-shot sample the detector confirms 97.7% of noted disruptions and fires on 74.2% of shots with no note (`results/label_report/report.md`).
- Termination classes (`src/termination.py`): at the detector's quench, programmed current below 80% of its maximum = programmed ramp-down; solenoid sweeping at under a quarter of its flat-top rate = flux exhausted; otherwise lost during flat-top with flux left; no quench = smooth end. The thresholds are physical choices fixed on the 400-shot sample before the full run.
- Facts on 3,804 shots (`scripts/label_statistics.py`, NOTES 2026-10-25): 623 of 847 noted disruptions (73.6%, Wilson 95% 70.5-76.4%) and 1,132 of 2,957 un-noted shots (38.3%, 36.5-40.0%) were lost during flat-top with flux left; 1,223 of the 3,089 detector quenches (40%) were flux-exhausted or ramp-down endings. Replication on the 3,562 shots outside the sample: 72.7% and 38.4%; solenoid-ratio AUC 0.66 (0.64-0.68). Full 11,573-shot table: TBD (`scripts/auto_labels.py` on the v2 dataset).
- Independent signals (`scripts/termination_evidence.py`): at the quench the candidates resemble recorded disruptions (current fall within 10 ms in 91% vs 95% vs 44% of flux-exhausted endings; similar pre-quench current spike); in end-of-shot neutron rate they sit in between. Probably mostly, not all, disruptions.
- Before the quench (new): the share of each ending group a model alarms on in time, for a model that never saw the candidates as disruptions (`evaluate.alarms_by_ending`, E15 'candidates left out'): TBD.
- Expert check: TBD (none has been done yet).

## 4. Methods
- Models: physics rule, logistic regression, random forest, gradient boosting; tuned on validation shots; alarm = two windows in a row over a threshold chosen so at most 5% of validation normal shots alarm; test split used once (SPEC 9-11).
- E14, what current-only labels reward: a machine-only predictor (solenoid current, programmed current, time since start; never the plasma) against all signals, on current-only labels and on ours.
- E15, whether the doubtful labels hide skill: the candidates as normal, left out of training, or as disruptions, all tuned and judged on the same clear-ending shots (noted disruptions the detector sees; shots with no quench).
- Error bars: repeated random splits (E10, 50 seeds) and paired comparisons (20 splits each, each against its own baseline), judged with the corrected resampled t-test (Nadeau & Bengio 2003), because the splits share shots. With 12 comparisons x 2 measures, about one 'clear' verdict can come from chance; we also report Bonferroni-type corrected verdicts.
- Also: campaign drift (E6, train M5-M7, test M8-M9), conformal false-alarm threshold (E8), positive-unlabelled learning with a calibrated g-model (E9), the UKAEA-style protocol (E12).

## 5. Results
- 5.1 Termination classes on all shots: TBD.
- 5.2 E14: machine-only vs all signals, on current-only labels and on ours (catch rate at 5% false alarms, PR-AUC against its random baseline, corrected ranges): TBD.
- 5.3 E15: the three treatments of the candidates on clear endings, and alarm rates by ending group: TBD.
- 5.4 Signals: v1 vs v1+magnetics vs v2; leave one group out (density, EFIT, Mirnov bands); time-since-start shortcut: TBD. (On 3,804 shots and 50 splits, the v1 magnetics raised the catch rate; that interim finding is in NOTES 2026-10-30.)
- 5.5 Main models E0-E3 and the seed sweep: TBD.
- 5.6 Drift (E6), conformal (E8), PU (E9), UKAEA protocol (E12): TBD.

## 6. Discussion and limitations
- What a label set rewards decides what a benchmark measures; on MAST, current-only labels include endings caused by flux running out or by the programme.
- The termination classes say when the plasma was lost, not whether it disrupted: a ramp-down or flux-exhausted quench can still be a disruption (Berkery et al. find MAST disruptive above the Greenwald limit in the ramp-down).
- No expert has checked the classes; the solenoid limit differs between shots (handled by each shot's own flat-top rate); EFIT is assumed to be reconstructed slice by slice; offline evaluation, not real-time control.
- Next: a second machine. The TCABR dataset (2,189 discharges, 435 labelled disruptive, 1 us Mirnov coils; Zenodo 2026) allows a cross-machine test of features without units; it has no solenoid current, so the termination classes do not transfer directly.

## 7. Reproducibility
- Code: this repository (public). Data: FAIR-MAST (CC BY-SA 4.0). Seeds and settings saved with every result (`config.json`); every result reused only under an identical code and data fingerprint (`src/resume.py`).

## References
See `docs/literature_review.md` (every entry checked; full-text checks still needed where marked). Statistics: C. Nadeau, Y. Bengio, "Inference for the generalization error", *Machine Learning* 52, 239-281 (2003). TCABR dataset: Zenodo record 21843354 (2026), to be added to the review once used.
