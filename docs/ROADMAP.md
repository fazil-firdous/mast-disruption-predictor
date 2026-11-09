# ROADMAP — from v1 pipeline to research-grade disruption prediction

Long-term plan, started 2026-10-15. The goal is research that could stand up in a fusion / ML venue and go beyond published disruption predictors. Every item keeps the hard rules in `CONTRIBUTING.md`: no leakage, split by shot, every number from code that ran, no invented data details.

Status markers: **done** (code + tests), **built** (code + tests, waiting for a real-data run), **next**, **later**.

## Where we stand relative to the field
- A UKAEA group has a disruption benchmark on FAIR-MAST: a stacked BiLSTM on 417 shots, with ground truth auto-detected from the plasma current (IAEA FEC poster, see README credits). So "first disruption predictor on MAST data" is not a claim we can make.
- Our own Phase 2 report found that a current-quench detector also flags 74% of shots whose operator note calls them fine, because most MAST shots end in a fast quench. Labels from the current alone therefore mix normal terminations with disruptions. **Label quality is our main scientific angle.**
- Literature check of 2026-10-15 (`docs/literature_review.md`): conformal prediction **has** been applied to disruption prediction (JET, Springer chapter) and survival / time-to-disruption models **exist** (Keith et al. 2024; arXiv:2609.24443). So B1 and C2 are not novel in themselves. Not found so far: PU learning for disruptions nobody recorded, and a measured audit of MAST's operator-note labels. Say "we could not find it", never "nobody has done it".

## v3 plan (started 2026-11-09)
- **A. MAST-Events, the first public per-shot event catalogue for MAST:**
  1. `scripts/mast_events.py` (built): events, causes and times from the comments, and repeat-shot pairs.
  2. Evidence fusion into P(disrupted), cause and time with a confidence (weak supervision).
  3. Validation against Thornton's full-performance disruptions, a larger hand check, and an expert check in TokTagger if offered.
  4. Public release.
- **B. Predictor v3 trained on it:** soft labels, a penalty when a machine-only model explains the alarm, hysteresis alarm, time-to-disruption output, cause-specific risks.
- **Pre-registration:** `docs/preregistration_v3.md` (2026-11-09), written before any v3 run.
- **C. Evaluation:**
  - shot-level skill and window AUROC on the same runs, plus a current-only 'metric audit' (GenToka, arXiv:2611.08153, reports AUROC 0.87 from the plasma current alone);
  - repeat-shot twins, later campaigns and Thornton's disruptions as test sets;
  - hypotheses written down before the runs, as for splits 21-50.
- **D. Later:** a deep model on a GPU; MAST to TCABR.

## Track A — Labels (core contribution)
- **A1 Positive-unlabelled (PU) learning — built (E9).** Note-confirmed disruptions are labelled positives; the pre-quench windows of unnoted quenches are unlabelled, not negative. Elkan & Noto (2008) weighting. Also gives an estimate of how many unnoted quenches look like real disruptions, which is a physics result in itself.
- **A2 Gold-standard subset — built, waiting for an expert (`src/gold.py`, Telegram /gold; could be offered as a TokTagger project, UKAEA's own annotation tool, NOTES 2026-11-06).** Blind labelling needs someone trained to read MAST shots (NOTES 2026-10-25). A MAST physicist labelling ~100 shots would let `/goldreport` measure every automatic label.
- **A3 Current-only labels vs ours — built (E7, E12).** E12 copies the public part of the UKAEA FAIR-MAST protocol (current-only labels, 10/30/60 ms).
- **A4 Termination physics — built and in training (v2, NOTES 2026-10-30).** `src/termination.py` classes every quench as flat-top drop with flux left, flux exhausted (solenoid stopped), programmed ramp-down, or smooth end; the classes feed the 'physics' label source (E7) and E15 (candidates as normal, left out, or as disruptions, judged on the same clear endings), and every model reports its alarm rate per ending group (precursor evidence for the candidates). Earlier (`results/termination_study/`): unnoted quenches happen during the programmed flat-top like noted disruptions. Next: loop voltage and vertical position at the quench, to split flux exhaustion from MHD-driven endings; an expert check of a blind sample.
- **A5 DEFUSE labels — dropped as public data; DEFUSE was run on 15,553 MAST shots (samueljackson92/defuse-data), so its flat-top windows and quench times could be asked for.** `s3://mast/level2/defuse/` listed 3,067 per-shot HDF5 files but returned 403; in ukaea/fair-mast #230 the team removed the folder (Oct 2026) without answering the access or licence questions. Not a public event database for now; ask only if an expert contact offers it.

## Track B — Guarantees
- **B1 Conformal false-alarm control — built (E8); not novel in itself (see literature review), our angle is the shot-level guarantee under campaign drift.** Threshold from a held-out set of clean shots so that P(false alarm on a new clean shot) <= alpha, provably, if shots are exchangeable. Then test where the guarantee breaks (temporal split = campaign drift).
- **B2 Calibrated probabilities — later.** Isotonic / Platt calibration so a score of 0.8 means ~80%.

## Track C — Time
- **C1 Premature alarms — built.** Alarms more than `PREMATURE_WARNING_MS` before the disruption are reported separately, because an alarm 300 ms early is probably not about this disruption.
- **C2 Time-to-disruption hazard model — next, not novel (Keith et al. 2024): implement to compare, cite them.** Discrete-time survival model: P(disruption within t ms) for every t, instead of one fixed horizon. Their code is public (MIT-PSFC/disruption-survival-analysis: auton-survival Cox, deep Cox, DSM and DCM models, a random forest, and a Kaplan-Meier style predictor after Tinguely et al. 2019 that extrapolates the risk with a linear fit over the last 50-200 ms). A discrete-time hazard with our HGB needs no new dependency: one row per (window, horizon), label 'disrupts within horizon'.
- **C3 Alarm with hysteresis — next (science change, full rerun).** The MIT code alarms when the score passes an upper threshold and stays above a lower one for a hold time (three tuned numbers, after the alarm logic used at JET). Ours is two windows in a row over one threshold. Tune the three on validation shots like the threshold now; test with the repeat runs.
- **C4 Threshold-free shot metric — done (`scripts/paper_numbers.py` pAUC).** Area under caught vs false alarms up to 10% false alarms, from saved curves: less noisy than the catch rate at one threshold (after the MIT code's area under the warning-time curve up to 5% false alarms).

## Track D — Physics signals (biggest expected gain)
- **D1 Magnetic precursors — v2 built (`src/extras.py`, `src/features.py`).** Saddle-coil n=1 amplitude, Mirnov RMS and four band RMS values (1-5, 5-20, 20-50, 50-250 kHz), solenoid current, current-control error; each fluctuation also as change against the shot's own earlier median and as a 100 ms trend. The v1 magnetics clearly helped over 50 repeat runs (NOTES 2026-10-30). A fixed linear PF-coil pickup subtraction made the saddle signal worse (NOTES 2026-10-16), hence change rather than level. Open: how `ASM/SAD/OUT` is made; channel-to-coil mapping (asked in ukaea/fair-mast #237). Data caveats from ukaea/fair-mast #211: level-2 centre-column Mirnov data are quantised (about 43.7 nT steps in shot 29980, which UKAEA attributes to the native digitiser), so the high bands of quiet shots may be quantisation noise (uniform-noise RMS ~13 nT); the OMAHA arrays (M9, shots > 28390) have more bits. Some saddle coils have no record in some shots (e.g. M10), hence MIN_SADDLE_CHANNELS. Next: measure each shot's quantisation step and flag band values near it.
- **D3 Density and the Greenwald limit — built (v2).** Line-average density from the summary group and the Greenwald fraction built causally from it, the current and the EFIT minor radius (NOTES 2026-10-30); leave-one-group-out comparison 'no_density'. DECAF on MAST (Berkery et al.) found disruptivity rising above the Greenwald limit, especially in the ramp-down.
- **D4 More shots — level 2 done (v2: every one of the 11,573 shots).** Level 1 dropped: UKAEA plans to deprecate it and distribute only level 2 as the FAIR-MAST dataset (ukaea/fair-mast #224, Oct 2026). About 100 M8/M9 shots have no NBI record at source (#232), so their beam power is missing, not zero.
- **D2 Equilibrium and limits — built (v2).** EFIT q95, li, beta_N, elongation and magnetic-axis height in 92-100% of shots per campaign (5 ms grid), each used one slice after its time; leave-one-group-out comparison 'no_efit'. ASSUMPTION to confirm with the FAIR-MAST team: slice-by-slice reconstructions. Next: features as distance to operational limits.

## Track E — Models
- **E1 Sequence models — later.** 1D CNN / LSTM / transformer on the raw 1 ms signals (needs PyTorch: ask before adding). Direct comparison with the UKAEA BiLSTM.
- **E2 Transfer — later.** MAST to MAST-U when data is public; few-shot adaptation.

## Track H — Infrastructure (done)
- Data-source watcher (shot table in the FAIR-MAST bucket, new FAIR-MAST shots or folders, DEFUSE public) and weekly arXiv literature watcher, both on Telegram. Since 2026-10-24 the shot table is read from the S3 bucket when mastapp.site is down; UKAEA confirmed on 2026-10-27 that the bucket table is the official one (mastapp.site retired). At the next science change: read only the bucket table, and move the URLs into `NOT_SCIENCE_SETTINGS` so a changed address never retrains anything.
- Shortcut test (E11): a large part of the v1 model's small skill on 400 real shots came from time since start.
- GitHub data relay for servers STFC blocks (`src/relay.py`, `.github/workflows/fetch_data.yml`); private settings in the permanent `~/mast-private/.env`.
- Public code, private runs (`MAST_HOME`), saved models (`src/registry.py`), live prediction (`scripts/predict_shot.py`), network diagnosis, always-on research server with Telegram commands, E10 seed sweep for spreads, then repeat-run comparisons (E5, E11, E13 on the same 50 splits, Nadeau-Bengio corrected), with jobs side by side on every CPU core.

## Track F — Explanation and demo
- **F1 Why did it alarm — next.** Per-alarm attribution of the score to features.
- **F2 Live replay — later.** A shot played millisecond by millisecond with the score and alarm.

## Track G — Scientific output
- **G1 Literature review — next.** Proper review of disruption prediction (JET, DIII-D, C-Mod, EAST, J-TEXT, MAST) before any novelty claim.
- **G2 Paper-style write-up — later.** Reproducible splits, all experiments, honest limitations; arXiv-ready.

## Working method
Each item lands as its own pull request with tests, a NOTES.md entry and a CODE_MAP.md update.
