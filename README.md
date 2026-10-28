# MAST Plasma Disruption Predictor

Machine learning that watches a MAST tokamak plasma pulse as it unfolds and tries to raise an alarm before a disruption, using only information available up to that moment, and a close look at how trustworthy the disruption labels themselves are.

*Built by a high-school student who is passionate about fusion energy and wants to go further in plasma physics and machine learning.*

## How it works
1. **Data**: public MAST data from FAIR-MAST (UKAEA): plasma current, radiated power, neutron rate and beam power at 1 ms, plus magnetic and control signals.
2. **Labels**: a shot counts as disrupted only if the operator's note says so and a fast current quench is found; the quench gives the disruption time.
3. **Windows**: every 5 ms, features from the last 20 ms only. No future samples, enforced by tests.
4. **Models**: a physics rule, logistic regression, random forest and gradient boosting.
5. **Alarm**: two windows in a row above a threshold, chosen on validation shots so at most 5% of normal shots raise an alarm.
6. **Evaluation per shot**: caught, late, missed, false alarm, warning time, with bootstrap 95% intervals, on test shots used once. Also train-on-early-campaigns, test-on-late (M5–M7 vs M8–M9), and 50 repeat runs on random splits for error bars.

## What it has found so far (real MAST data)
- **The labels disagree.** A current-quench detector confirms 97.7% of the shots the operators called disruptions, but also fires on 74.2% of the shots they called normal (`results/label_report/report.md`, 400 shots; 98.2% and 76.3% on all 3,804 shots used). Labels taken from the current alone, as in published FAIR-MAST benchmarks, mix normal endings with disruptions.
- **Automatic termination labels** (`scripts/auto_labels.py`) sort each ending by physics: lost during flat-top with flux left, after the central solenoid ran out of flux, during a programmed ramp-down, or smooth.
  - On all 3,804 shots used, 623 of the 847 operator-noted disruptions were lost during flat-top with flux left (74%; 77% on the 400-shot sample).
  - So were 1,132 of the 2,957 shots nobody called a disruption (38%): candidates for disruptions that were never recorded.
  - The detector fires on 3,089 shots, and 1,223 of those (40%) ended after the solenoid ran out of flux or during a programmed ramp-down, endings that labels from the current alone count as disruptions.
  - The rules were set on the 400-shot sample; on the 3,562 shots outside it the shares are the same (72.7% and 38.4%), and the solenoid ratio separates noted from un-noted quench endings with AUC 0.66 (95% 0.64-0.68): a real but modest difference (`scripts/label_statistics.py`).
  - On signals the rules never use (`scripts/termination_evidence.py`, all 2,721 shots of the three groups), the candidates look like recorded disruptions where the quench itself is concerned: the current falls within 10 ms in 91% of them, 95% of recorded disruptions and 44% of flux-exhausted endings, and they show the same small current spike before the quench. In heating at the end they sit in between, so they are probably mostly, not all, disruptions. No expert has checked them yet, and a quench during ramp-down or after flux exhaustion can still be a disruption: the classes describe when the plasma was lost, not whether it disrupted.
- **Four global signals are not enough.** On 3,804 shots, over 43 repeat runs, the predictor does not clearly catch more disruptions than an alarm that fires at random as often, once the repeat runs' shared shots are allowed for. Over the repeat runs so far, plasma current is the only one of the four whose removal clearly costs caught disruptions; the single-split hint that radiated power matters most is not confirmed.
- **Magnetic signals help (interim).** Adding the saddle-coil locked-mode signal, Mirnov fluctuations, solenoid current and the current-control error clearly raises the share of disruptions caught in time at the same false-alarm rate, and by about as much without time since start. Interim, from 29 of 50 paired repeat runs, and one of seven comparisons: a strong hint until all 50 are done.
- **Shortcut check.** On one split, part of what the model learned was how long a shot had lasted (E11); over the repeat runs so far, removing that feature makes no clear difference, so the shortcut is not confirmed. Every claim is still checked with and without that feature.
- **Error bars for every claim.** The single-split findings (E5, E11, E13) are being re-tested on 50 random splits each, paired with the repeat runs and judged with a test that allows for the splits sharing shots (`scripts/repeat_runs.py`).
- **A negative result.** A fixed correction for coil pickup in the saddle-coil locked-mode signal made it worse on held-out shots, so it is not used (`NOTES.md`, 2026-10-16).

Trained models and their exact numbers stay in a private workspace, not in this repository.

## Research track
The plan is in `docs/ROADMAP.md`; the checked references are in `docs/literature_review.md`.

| Experiment | Question |
|---|---|
| E0–E3 | How good is each model? |
| E4 | How far ahead should it look? |
| E5 | Which signal matters? |
| E6 | Does it hold on later campaigns? |
| E7 | Which labels to trust? |
| E8 | Can a false-alarm rate be guaranteed? (conformal prediction) |
| E9 | Unrecorded disruptions as "unknown": positive-unlabelled learning |
| E10 | Repeat runs for error bars |
| E11 | Shortcut test (without time since start) |
| E12 | The rules of the UKAEA FAIR-MAST benchmark |
| E13 | Magnetic signals (saddle-coil locked modes, Mirnov coils, solenoid current) |
| Comparisons | Do the E5, E11 and E13 findings hold on 50 different splits? |

## How to run
Everything runs from `notebooks/colab_pipeline.ipynb` or a terminal:

```
pip install -r requirements.txt
pytest
python scripts/build_dataset.py --scale small      # 300 disrupted + 300 normal shots -> data/processed/
python scripts/run_experiment.py --exp E3          # one experiment -> results/E3/
python scripts/run_all.py --scale small            # everything, E0-E13 (unchanged results are reused)
python scripts/auto_labels.py                      # how each shot ended -> results/auto_labels/
python scripts/label_statistics.py                 # their shares, intervals, tests and the replication
python scripts/termination_evidence.py             # the candidates on signals the rules never use
python scripts/repeat_runs.py --comparison magnetics --seed 1   # one paired repeat run with one change
python scripts/repeat_runs.py --summary            # every comparison so far, with corrected 95% ranges
python scripts/predict_shot.py --shot 11860        # the best trained model on one shot, as if live
```

Outputs (data, models, results, logs) go to the folder named by `MAST_HOME`, or the current folder if unset (git-ignored). `python scripts/diagnose_network.py` explains any connection problem. To run continuously on a server, see `docs/SERVER.md`.

## Where to start reading
`docs/GUIDE.md` tells the story in simple words; `CODE_MAP.md` describes every file and function; `NOTES.md` is the log of data facts and decisions; `CONTRIBUTING.md` has the rules for changing the code.

## Limitations
Labels are algorithmic and not yet checked by a MAST expert. The main models use four global signals; the magnetics are a first version, and there are no profiles. MAST (2000–2013), not MAST-U. Offline evaluation, not real-time control.

## Credits
Data: FAIR-MAST, UKAEA, licence CC BY-SA 4.0 (documentation: https://ukaea.github.io/fair-mast). Please cite S. Jackson et al., "FAIR-MAST: A fusion device data management system", *SoftwareX* 27, 101869 (2024), and S. Jackson et al., "An Open Data Service for Supporting Research in Machine Learning on Tokamak Data", *IEEE Transactions on Plasma Science* (2025), doi:10.1109/TPS.2025.3583419.
Related work: UKAEA FAIR-MAST event-prediction benchmarks (IAEA FEC poster, https://conferences.iaea.org/event/392/contributions/36328/); positive-unlabelled learning: Elkan & Noto, KDD 2008.
