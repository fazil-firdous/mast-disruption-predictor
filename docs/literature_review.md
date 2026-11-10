# Literature review (ROADMAP G1) — working draft

Started 2026-10-15. Every entry was checked against at least one bibliographic source (journal page, DOI record, repository or arXiv) by web search; details marked *(abstract only)* were read from abstracts or summaries, not the full paper, and must be checked before being quoted in a paper. The weekly arXiv watcher (`src/literature.py`) adds new papers to the private reading list in `MAST_HOME/literature/papers.csv`; promote the relevant ones to this file.

## 1. Reviews and the big picture
| Reference | What it is | Why it matters here |
|---|---|---|
| J. Vega, A. Murari, S. Dormido-Canto, G. A. Rattá, M. Gelfusa and JET Contributors, "Disruption prediction with artificial intelligence techniques in tokamak plasmas", *Nature Physics* 18, 741–750 (2022), doi:10.1038/s41567-022-01602-2 | Review of AI-based disruption predictors | Standard entry point; frames warning time, false alarms and the need to predict on new machines |
| P. C. de Vries et al., "Survey of disruption causes at JET", *Nuclear Fusion* 51, 053018 (2011) | Root causes of 2,309 JET disruptions | Locked (neoclassical tearing) modes are the largest root cause, human error second; a floor of fast, precursor-less disruptions *(abstract only)*. Motivates the saddle-coil locked-mode signal (ROADMAP D1) and explains why some disruptions cannot be predicted |

## 2. Machine-learning predictors
| Reference | What it is | Why it matters here |
|---|---|---|
| J. Kates-Harbeck, A. Svyatkovskiy, W. Tang, "Predicting disruptive instabilities in controlled fusion plasmas through deep learning", *Nature* 568, 526–531 (2019), doi:10.1038/s41586-019-1116-4 | FRNN: recurrent + convolutional network on DIII-D and JET, cross-machine prediction | The best-known deep-learning predictor; our sequence-model plan (ROADMAP E1) would be compared to this family |
| C. Rea, R. S. Granetz and co-workers (MIT/GA), random-forest disruption predictors on DIII-D (DPRF, running in the plasma control system), Alcator C-Mod and EAST; e.g. Rea & Granetz, *Fusion Science and Technology* (2018), and K. J. Montes et al., "Machine learning for disruption warnings on Alcator C-Mod, DIII-D, and EAST", *Nuclear Fusion* (2019) | Random forests on mostly dimensionless signals; >100 ms warnings on DIII-D; very different useful signals per machine *(abstract only)* | Closest in method to our v1 models (random forest, HistGradientBoosting); their finding that C-Mod disruptions are hard to predict on 1–2 ms timescales parallels our MAST result with global signals |
| DisruptionBench (MIT), *Journal of Fusion Energy* (2025), doi:10.1007/s10894-025-00495-2 | Common benchmark over C-Mod, DIII-D and EAST with zero/few/many-shot regimes; random forest, Hybrid Deep Learner, GPT-2-style transformer, CCNN | The model for a fair benchmark; does not include MAST |
| UKAEA FAIR-MAST team, event-prediction benchmarks on FAIR-MAST, IAEA FEC poster and paper (conferences.iaea.org event 392, contribution 36328; author list to be checked) | Baselines on FAIR-MAST incl. disruption prediction (stacked BiLSTM, 417 shots, labels detected from the plasma current, lead times 10/30/60 ms) | The direct comparison for this project (our E12 copies their protocol as far as it is public); their labels are current-only, which our Phase 2 report shows also flags most normal MAST terminations |

## 3. Labels, time-to-disruption and uncertainty
| Reference | What it is | Why it matters here |
|---|---|---|
| L. Keith, A. Nagpal, C. Rea, R. A. Tinguely, "Risk-aware framework development for disruption prediction: Alcator C-Mod and DIII-D survival analysis", *Journal of Fusion Energy* (2024), doi:10.1007/s10894-024-00413-y | Cox, Deep Cox and Deep Survival Machines vs random forest; longer warnings at similar ROC *(abstract only)* | Time-to-disruption (our ROADMAP C2) is **not new**; cite and build on it |
| "Horizon-Aware Early Event Prediction for Tokamak Disruption Alarms", arXiv:2609.24443 (2026) | Horizon-probability methods vs survival methods on DIII-D, C-Mod, EAST *(abstract only)* | Very recent; same question as our horizon sweep (E4) and fixed-horizon labels |
| Survival analysis via random forests, arXiv:1907.04291 | Early survival framing of disruption prediction *(title and abstract only)* | Same lineage |
| "Conformal Prediction of Disruptions from Scratch: Application to an ITER Scenario", Springer book chapter, doi:10.1007/978-3-319-33395-3_5 (JET ITER-like-wall data) | Conformal predictors for disruption prediction *(abstract only)* | Conformal prediction for disruptions is **not new**. Our E8 is narrower: a shot-level false-alarm guarantee and where it fails under campaign drift; check the full text before claiming even that |
| Ai et al., "Disruption precursor onset time study based on semi-supervised anomaly detection", arXiv:2303.14965; *Nuclear Engineering and Technology* (2024) | Fixed-lead-time labels mislabel windows; anomaly detection infers precursor onset and relabels | A label-noise paper, but about the **timing** of precursors in known disruptions, not about disruptions nobody recorded |
| C. Elkan, K. Noto, "Learning classifiers from only positive and unlabeled data", KDD (2008) | The PU method used in E9 | Method reference |

## 4. Spherical tokamaks and MAST
- DisruptionPy (G. L. Trevisan et al., MIT PSFC Disruptions Group; github.com/MIT-PSFC/disruption-py, doi:10.5281/zenodo.13935223, MIT licence) has a MAST module (density, radiated-power and profile peaking, Thomson, D-alpha, soft X-ray, vertical position). Its MAST disruption times come from a private database. Its test configuration names 30113, 30014 and 30421 as disruptive and 29659, 28864 and 29666 as not: in our table the three non-disruptive ones end smoothly, 30014 is a candidate (no note, lost in flat-top with flux left), 30113 is flux-exhausted, and 30421 has no fast quench for our detector (to check on the server).
- FAIR-MAST ingestion code (github.com/ukaea/fair-mast-ingestion, read at commit c692231): level 2 is resampled with xarray `interp(method='zero')`, a zero-order hold that takes the last sample at or before each grid time, for 31 of 32 groups (summary, magnetics, pf_active, pulse_schedule, equilibrium, interferometer among them); only Thomson scattering uses linear interpolation. So the level-2 resampling does not look ahead.

| Reference | What it is | Why it matters here |
|---|---|---|
| S. Jackson et al., "FAIR-MAST: A fusion device data management system", *SoftwareX* 27, 101869 (2024), doi:10.1016/j.softx.2024.101869 | The data service this project uses | Must be cited for the data |
| S. Jackson, S. Khan, N. Cummings et al., "An Open Data Service for Supporting Research in Machine Learning on Tokamak Data", *IEEE Transactions on Plasma Science* (2025), doi:10.1109/TPS.2025.3583419 | The FAIR-MAST service paper for ML users | The FAIR-MAST documentation asks for both papers to be cited |
| S. P. Gerhardt et al., disruption detection in the high-beta spherical torus NSTX (PPPL report, 2013) | No single signal and threshold works well on NSTX *(abstract only)* | Spherical-tokamak precedent: matches what we see on MAST |
| DECAF (Disruption Event Characterization and Forecasting) with ML-assisted stability (Piccione, Berkery, Sabbagh et al.; NSTX models tested on MAST) | Physics-plus-ML event chains *(conference material only)* | The physics-event approach for spherical tokamaks; DECAF-style event labels would be an alternative to our gold set |
| J. W. Berkery et al., DECAF on the MAST database: automatic disruption detection on 15,363 discharges of M5-M9 (NSTX-U meeting slides 2018; APS DPP 2018 poster "Disruptivity and density limits in MAST and other tokamaks"); and "Density limits as disruption forecasters for spherical tokamaks", *Plasma Physics and Controlled Fusion* (UKAEA preprint) | Disruptivity rises above the Greenwald limit in NSTX and MAST, especially in the current ramp-down *(slides, poster and preprint abstract only)* | The closest MAST disruption database. It counts ramp-down disruptions as disruptions, so our 'programmed ramp-down' and 'flux exhausted' classes say when the plasma was lost, not that it ended harmlessly. Comparing our classes with the DECAF MAST list is the obvious check |
| A. J. Thornton, "The impact of transient mitigation schemes on the MAST edge plasma", PhD thesis, University of York (January 2011); thesis, IDL code and saved database archived at github.com/spamela/disruption_detector (Apache-2.0, S. Pamela 2021) | Read (section 4.2.1 and the code). His 'full performance disruptions': in the current flat-top, NBI ceasing after the disruption, and 'the central solenoid flux ... should not have reached its operational limit or started to ramp in the opposite direction'. Applied to shots 3500-22066: 9,864 discharges, 8,871 'disrupt at any point', 1,122 full-performance disruptions | **Prior art for our termination rule**: the same physical criteria, used by him to select a clean disruption sample. Our contribution is applying such a rule to every shot to audit the operator notes and current-only labels, and what predictors learn from each label set. On 6,501 shared shots our classes reproduce his flags (`scripts/thornton_compare.py`, NOTES 2026-11-06) |
| ITPA International Disruption Database (EFDA/EUROfusion preprint EFDP14025) | Multi-machine database of disruptions and other fast current terminations (VDEs, massive gas injection), with MAST and NSTX among the machines *(abstract only)* | Its fast-termination criterion is the same kind as our quench detector, and like it would not separate the endings our classes separate |
| C. Rousseau, S. Jackson et al., "TokaMark: A Comprehensive Benchmark for MAST Tokamak Plasma Models", arXiv:2602.10132 (2026) | 14 forecasting tasks on FAIR-MAST (11,573 shots), including forecasting the plasma current 100 ms ahead (task 4-4) | Read in full (v3) and its code (github.com/UKAEA-IBM-STFC-Fusion-FMs/tokamark): 14 forecasting tasks (150 ms of input, 100 ms ahead), no disruption labels, so a forecasting benchmark, not a disruption-label one. Its curated dataset is in `s3://mast/tokamark/v1/` (groups equilibrium, gas_injection, interferometer, magnetics, pf_active, pulse_schedule, spectrometer_visible, summary; UKAEA confirmed on 2026-10-27 that this is the TokaMark dataset). Companion: TokaMind, a multi-modal transformer foundation model, arXiv:2602.15084 |

## 5. What this means for our claims (as of 2026-10-15)
- **Not new, cite instead:** time-to-disruption / survival models; conformal prediction for disruptions; random-forest and deep-learning predictors; benchmarks on C-Mod/DIII-D/EAST.
- **Not found so far (still needs a proper search before any claim):** positive-unlabelled learning for disruptions nobody recorded; a measured audit of operator-note labels against current-quench labels and the programmed current (our Phase 2 report and termination study); machine-learning disruption prediction on MAST data beyond the UKAEA FAIR-MAST baselines.
- **Correction 2026-11-06:** Thornton (2011, section 4.2.1) used the solenoid-flux and flat-top criteria to select MAST disruptions. The earlier 'nothing found that uses the central-solenoid flux state' is wrong for the rule itself; what remains new is its use to audit operator notes and current-only labels on every shot.
- **Targeted search 2026-10-25** (web search, not a systematic review), for missed operator labels, current-quench false positives and solenoid-flux-based separation of endings: nothing found that uses the central-solenoid flux state to classify MAST terminations or audits MAST operator notes against current-quench labels. Closest: DECAF on MAST (automatic detection; ramp-down disruptions count as disruptions), the UKAEA benchmark (current-derived ground truth, 417 shots; its poster notes that ambiguous cases without a sharp current drop remain unresolved), the ITPA database (fast terminations), Ai et al. (label noise in precursor timing). "Not found" is not "does not exist".
- **Our own real-data findings so far (private runs, 400 shots):** most MAST terminations look like disruptions in the current but are not noted; with the 4 global signals, almost no disruption is caught at <= 5% false alarms; part of what the model learns is time since start (E11). These shape the paper's story: label quality and physics signals first, model complexity second.

## 6. The twin test: has anyone matched disrupted shots with same-programme repeats? (searched 2026-11-10)

Eight web searches (matched or paired discharges; repeat shots with the same pulse schedule; shot-to-shot
variability of disruption occurrence; shortcut learning or spurious correlation in disruption predictors;
counterfactual, natural-experiment or case-control evaluation; pairwise or concordance evaluation). Not a systematic
review, and several publisher sites could not be opened from the working environment, so the entries below rest on
abstracts and search summaries unless a full text is cited elsewhere in this file.

- **Not found:** an evaluation of a disruption predictor on disrupted shots paired with same-programme shots that
  were not lost; a study of how often nominally identical discharges disrupt; a model given only the programme as a
  control for what the usual metrics reward.
- **Nearest work:**
  - Counterfactual *explanations* for a resistive-wall-mode stability classifier on NSTX (Piccione, Sabbagh,
    Andreopoulos; Princeton data record doi:10.11578/1888266): hypothetical changes of inputs that would flip a
    prediction. Model interpretation, not matched real shots.
  - Spurious correlations in cross-machine transfer: "Balancing physics priors and data for disruption prediction in
    future tokamaks" (Nuclear Fusion, doi:10.1088/1741-4326/ae55b7) argues that statistical models can learn
    unphysical correspondences between machines and adds physics priors. A different shortcut, a different remedy.
  - Murari et al., "A control oriented strategy of disruption prediction to avoid the configuration collapse of
    tokamak reactors" (Nature Communications 2024, doi:10.1038/s41467-024-46242-7), and the JET literature around
    it: anomalies of the kind used as precursors also occur in discharges that end safely. This is the reason a
    matched comparison is informative; it is not one.
  - Robustness by augmentation: "Time Series Viewmakers for Robust Disruption Prediction" (arXiv:2410.11065).
  - Shot-to-shot scatter is studied for runaway-electron beams after disruptions (DIII-D, TEXTOR), not for whether a
    disruption happens.
- **What the paper may say:** "we found no earlier matched repeat-shot evaluation" with the search described, not
  "the first". Reference comparisons with a good repeat shot are routine by hand in the control room; the claim is
  about a systematic test on thousands of pairs.
- **To do before the paper is public:** open the full texts of the three nearest papers (authors, year, exact
  claims) and search the JET, DIII-D and ASDEX Upgrade disruption-database papers for "repeat" and "reference pulse".
