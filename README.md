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
- **The labels disagree.** A current-quench detector confirms 97.7% of the shots the operators called disruptions, but also fires on 74.2% of the shots they called normal (`results/label_report/report.md`, 400 shots). Labels taken from the current alone, as in published FAIR-MAST benchmarks, mix normal endings with disruptions.
- **Automatic termination labels** (`scripts/auto_labels.py`) sort each ending by physics: lost during flat-top with flux left, after the central solenoid ran out of flux, during a programmed ramp-down, or smooth.
  - On the 400-shot sample, 134 of 175 operator-noted disruptions were lost during flat-top with flux left.
  - So were 83 of 223 shots nobody called a disruption: candidates for disruptions that were never recorded.
- **Four global signals are not enough.** On 3,804 shots, averaged over repeat runs, the predictor catches only slightly more disruptions than an alarm that fires at random as often. Radiated power is the most useful signal.
- **Shortcut check.** Part of what the model learned was how long a shot had lasted, not plasma physics (E11). Every claim is now checked with and without that feature.
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

## How to run
Everything runs from `notebooks/colab_pipeline.ipynb` or a terminal:

```
pip install -r requirements.txt
pytest
python scripts/build_dataset.py --scale small      # 300 disrupted + 300 normal shots -> data/processed/
python scripts/run_experiment.py --exp E3          # one experiment -> results/E3/
python scripts/run_all.py --scale small            # everything, E0-E13 (unchanged results are reused)
python scripts/auto_labels.py                      # how each shot ended -> results/auto_labels/
python scripts/predict_shot.py --shot 11860        # the best trained model on one shot, as if live
```

Outputs (data, models, results, logs) go to the folder named by `MAST_HOME`, or the current folder if unset (git-ignored). `python scripts/diagnose_network.py` explains any connection problem. To run continuously on a server, see `docs/SERVER.md`.

## Where to start reading
`docs/GUIDE.md` tells the story in simple words; `CODE_MAP.md` describes every file and function; `NOTES.md` is the log of data facts and decisions; `CONTRIBUTING.md` has the rules for changing the code.

## Limitations
Labels are algorithmic and not yet checked by a MAST expert. The main models use four global signals; the magnetics are a first version, and there are no profiles. MAST (2000–2013), not MAST-U. Offline evaluation, not real-time control.

## Credits
Data: FAIR-MAST, UKAEA, licence CC BY-SA 4.0. Please cite S. Jackson et al., "FAIR-MAST: A fusion device data management system", *SoftwareX* 27, 101869 (2024).
Related work: UKAEA FAIR-MAST event-prediction benchmarks (IAEA FEC poster, https://conferences.iaea.org/event/392/contributions/36328/); positive-unlabelled learning: Elkan & Noto, KDD 2008.
