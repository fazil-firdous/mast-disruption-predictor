# MAST Plasma Disruption Predictor

A machine-learning project that watches a MAST tokamak plasma pulse as it unfolds and tries to raise an alarm before a disruption, using only information available up to that moment. Work is staged by phase (see `docs/SPEC.md` and `the project rules`).

**Status: the whole v1 pipeline is written and tested on synthetic shots (Phases 0 to 7: data layer, labels, causal features, dataset and splits, alarm logic, four models, shot-level evaluation, experiments E0-E7). It has not yet been run on real data, so there are no model results yet.**

## How it works
1. **Data**: four global signals per shot from FAIR-MAST (`|Ip|`, radiated power, neutron rate, NBI power), 1 ms sampling.
2. **Labels**: a shot counts as disrupted only if the operator note says so and a fast current quench is found in `Ip` (`LABEL_SOURCE = "both"`); the quench gives the disruption time.
3. **Windows**: every 5 ms, 30 features from the last 20 ms only (no future samples, tested). Windows within 30 ms before the disruption are positive.
4. **Models**: a physics rule (B0), logistic regression, random forest, HistGradientBoosting.
5. **Alarm**: two windows in a row above a threshold. The threshold is chosen on validation shots so that at most 5% of clean shots raise an alarm.
6. **Evaluation** per shot: catch rate, tardy rate, miss rate, false-alarm rate, warning time, with bootstrap 95% intervals, on a test split touched once. Also a temporal split (train on M5-M7, test on M8-M9).

## Research track
Beyond v1, the long-term plan is in `docs/ROADMAP.md`. Built so far (synthetic tests only, no real-data numbers yet):
- **E8 conformal alarm threshold**: a provable limit on the false-alarm rate for new clean shots, and a test of where it breaks (later campaigns).
- **E9 positive-unlabelled learning**: shots that end in a fast quench without an operator note are treated as unlabelled instead of clean, and the model estimates how disruption-like they are.
- **Premature alarms**: catches more than 200 ms early are reported separately.

## How to run
Everything can run from `notebooks/colab_pipeline.ipynb` (not yet run in Colab). From a terminal:

```
pip install -r requirements.txt
pytest
python scripts/build_dataset.py --scale small      # 300 disrupted + 300 clean shots -> data/processed/
python scripts/run_experiment.py --exp E3          # one experiment -> results/E3/
python scripts/run_all.py --scale small            # everything, E0-E7, with Telegram messages if configured
```

Earlier phase checks: `scripts/inspect_shots.py`, `scripts/download.py`, `scripts/label_report.py`, `scripts/feature_check.py`.

### Running unattended on a VPS with Telegram updates
1. Make a Telegram bot with @BotFather and note the token and your chat id.
2. On a fresh Ubuntu VPS: `git clone https://github.com/fazil-mastu/mastu.git && cd mastu && bash deploy/setup_vps.sh`. It installs everything, asks for the token and chat id (stored in `.env`, never committed), sends a test message, runs the tests and adds a nightly job.
3. Start the first full run: `tmux new -s mast 'deploy/run_pipeline.sh --force'`. Detach with Ctrl-b d. Telegram gets a message when each step starts and finishes, the headline numbers and trade-off plot for each model, any error with its traceback, and finally a zip of `results/` and the full log.
4. After that, the nightly job re-runs the pipeline only when new code has been pushed.

## Results
TBD - not yet run on real data. Only the label comparison exists so far (`results/label_report/report.md`, from code that ran on 400 real shots): the current-quench detector confirms 97.7% of note-disrupted shots but also flags 74.2% of note-clean shots, because most MAST shots end with a fast current quench whether or not anyone called it a disruption. That is why `LABEL_SOURCE = "both"`.

## Limitations
Labels are algorithmic and imperfect. Only 4 global signals are used, with no magnetics or profiles. MAST (2000 to 2013), not MAST-U. Offline evaluation, not real-time control. Results come from a subset of shots unless stated otherwise. The model code has only been tested on synthetic shots so far.

## Credits
Data: FAIR-MAST, UKAEA, licence CC BY-SA 4.0 (https://mastapp.site).
Related work: UKAEA FAIR-MAST event-prediction benchmarks (IAEA FEC poster, https://conferences.iaea.org/event/392/contributions/36328/). PU learning: Elkan & Noto, KDD 2008.
