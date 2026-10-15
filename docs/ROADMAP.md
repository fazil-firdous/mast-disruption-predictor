# ROADMAP — from v1 pipeline to research-grade disruption prediction

Agreed with the owner on 2026-10-15. The goal is not a portfolio piece but research that could stand up in a fusion / ML venue and go beyond published disruption predictors. Every item keeps the hard rules in `the project rules`: no leakage, split by shot, every number from code that ran, no invented data details.

Status markers: **done** (code + tests), **built** (code + tests, waiting for a real-data run), **next**, **later**.

## Where we stand relative to the field
- A UKAEA group has a disruption benchmark on FAIR-MAST: a stacked BiLSTM on 417 shots, with ground truth auto-detected from the plasma current (IAEA FEC poster, see README credits). So "first disruption predictor on MAST data" is not a claim we can make.
- Our own Phase 2 report found that a current-quench detector also flags 74% of shots whose operator note calls them fine, because most MAST shots end in a fast quench. Labels from the current alone therefore mix normal terminations with disruptions. **Label quality is our main scientific angle.**
- In two quick literature searches (2026-10-15) we found no tokamak disruption work using positive-unlabelled learning or conformal false-alarm guarantees. Say "we could not find it", never "nobody has done it", until a proper literature review is done (item G1).

## Track A — Labels (core contribution)
- **A1 Positive-unlabelled (PU) learning — built (E9).** Note-confirmed disruptions are labelled positives; the pre-quench windows of unnoted quenches are unlabelled, not negative. Elkan & Noto (2008) weighting. Also gives an estimate of how many unnoted quenches look like real disruptions, which is a physics result in itself.
- **A2 Gold-standard subset — next.** The owner hand-labels ~100 disagreement shots with a small review tool; this measures the label noise rate directly.
- **A3 Current-only labels vs ours — next.** Train and test with UKAEA-style current-only labels and with ours; show what changes.
- **A4 Termination physics — started (real data, `results/termination_study/`).** Unnoted quenches happen during the programmed flat-top like noted disruptions (programmed current at the quench: median 1.00 of its maximum in both groups), and slightly closer to the end of the central-solenoid swing (AUC 0.60, permutation p = 0.002, stronger in later campaigns). Next: loop voltage, vertical position and locked-mode amplitude at the quench, to split "flux exhaustion" from MHD-driven terminations.
- **A5 DEFUSE labels — blocked.** `s3://mast/level2/defuse/` lists 3,067 per-shot HDF5 files (shots 12139-30471), but downloading them returns 403 Access Denied. Ask the FAIR-MAST team whether they will be released; they could be an independent event database.

## Track B — Guarantees
- **B1 Conformal false-alarm control — built (E8).** Threshold from a held-out set of clean shots so that P(false alarm on a new clean shot) <= alpha, provably, if shots are exchangeable. Then test where the guarantee breaks (temporal split = campaign drift).
- **B2 Calibrated probabilities — later.** Isotonic / Platt calibration so a score of 0.8 means ~80%.

## Track C — Time
- **C1 Premature alarms — built.** Alarms more than `PREMATURE_WARNING_MS` before the disruption are reported separately, because an alarm 300 ms early is probably not about this disruption.
- **C2 Time-to-disruption hazard model — next.** Discrete-time survival model: P(disruption within t ms) for every t, instead of one fixed horizon.

## Track D — Physics signals (biggest expected gain)
- **D1 Magnetic precursors — next, data located.** `results/data_catalogue/` (from `scripts/inspect_groups.py`): saddle coils `magnetics/b_field_tor_probe_saddle_field` (12 channels, UDA `ASM/SAD/OUT`, the usual locked-mode detector), centre-column Mirnov coils (array lengths suggest ~50 kHz and ~500 kHz over the 0.38 s record of 11860; not yet checked against their time arrays), flux loops, outer Br/Bv arrays. Locked modes are the strongest known precursor.
- **D2 Equilibrium and limits — later, data located for later shots.** Shot 27000 has an `equilibrium` group (q95, li, beta_N, elongation, line-averaged n_e) and `thomson_scattering`; shot 11860 (M5) has neither, so availability must be surveyed per campaign. Must check whether EFIT used any later time slices before using it as a causal feature. Features as distance to operational limits.

## Track E — Models
- **E1 Sequence models — later.** 1D CNN / LSTM / transformer on the raw 1 ms signals (needs PyTorch: ask before adding). Direct comparison with the UKAEA BiLSTM.
- **E2 Transfer — later.** MAST to MAST-U when data is public; few-shot adaptation.

## Track H — Infrastructure (done)
- GitHub data relay for servers STFC blocks (`src/relay.py`, `.github/workflows/fetch_data.yml`); private settings in the permanent `~/mast-private/.env`.
- Public code, private runs (`MAST_HOME`), saved models (`src/registry.py`), live prediction (`scripts/predict_shot.py`), network diagnosis, always-on research server with Telegram commands, E10 seed sweep for spreads.

## Track F — Explanation and demo
- **F1 Why did it alarm — next.** Per-alarm attribution of the score to features.
- **F2 Live replay — later.** A shot played millisecond by millisecond with the score and alarm.

## Track G — Scientific output
- **G1 Literature review — next.** Proper review of disruption prediction (JET, DIII-D, C-Mod, EAST, J-TEXT, MAST) before any novelty claim.
- **G2 Paper-style write-up — later.** Reproducible splits, all experiments, honest limitations; arXiv-ready.

## Working method
The owner studies the new code daily and explains it back before the next item starts. Each item lands as its own PR with tests, a NOTES.md entry and a CODE_MAP.md update.
