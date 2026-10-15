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
| Reference | What it is | Why it matters here |
|---|---|---|
| S. Jackson et al., "FAIR-MAST: A fusion device data management system", *SoftwareX* 27, 101869 (2024), doi:10.1016/j.softx.2024.101869 | The data service this project uses | Must be cited for the data |
| S. P. Gerhardt et al., disruption detection in the high-beta spherical torus NSTX (PPPL report, 2013) | No single signal and threshold works well on NSTX *(abstract only)* | Spherical-tokamak precedent: matches what we see on MAST |
| DECAF (Disruption Event Characterization and Forecasting) with ML-assisted stability (Piccione, Berkery, Sabbagh et al.; NSTX models tested on MAST) | Physics-plus-ML event chains *(conference material only)* | The physics-event approach for spherical tokamaks; DECAF-style event labels would be an alternative to our gold set |

## 5. What this means for our claims (as of 2026-10-15)
- **Not new, cite instead:** time-to-disruption / survival models; conformal prediction for disruptions; random-forest and deep-learning predictors; benchmarks on C-Mod/DIII-D/EAST.
- **Not found so far (still needs a proper search before any claim):** positive-unlabelled learning for disruptions nobody recorded; a measured audit of operator-note labels against current-quench labels and the programmed current (our Phase 2 report and termination study); machine-learning disruption prediction on MAST data beyond the UKAEA FAIR-MAST baselines.
- **Our own real-data findings so far (private runs, 400 shots):** most MAST terminations look like disruptions in the current but are not noted; with the 4 global signals, almost no disruption is caught at <= 5% false alarms; part of what the model learns is time since start (E11). These shape the paper's story: label quality and physics signals first, model complexity second.
