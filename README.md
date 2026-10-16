# MAST Plasma Disruption Predictor

A machine-learning project that watches a MAST tokamak plasma pulse as it unfolds and tries to raise an alarm before a disruption, using only information available up to that moment. Work is staged by phase (see `docs/SPEC.md` and `the project rules`).

**Status: the v1 pipeline (Phases 0 to 7) and the first research extensions (E8-E10) are written, tested (127 tests) and have run end to end on real MAST shots. Trained models and their results live on the owner's private server, not in this repository (see "Public code, private runs").**

## How it works
1. **Data**: four global signals per shot from FAIR-MAST (`|Ip|`, radiated power, neutron rate, NBI power), 1 ms sampling.
2. **Labels**: a shot counts as disrupted only if the operator note says so and a fast current quench is found in `Ip` (`LABEL_SOURCE = "both"`); the quench gives the disruption time.
3. **Windows**: every 5 ms, 30 features from the last 20 ms only (no future samples, tested). Windows within 30 ms before the disruption are positive.
4. **Models**: a physics rule (B0), logistic regression, random forest, HistGradientBoosting.
5. **Alarm**: two windows in a row above a threshold. The threshold is chosen on validation shots so that at most 5% of clean shots raise an alarm.
6. **Evaluation** per shot: catch rate, tardy rate, miss rate, false-alarm rate, warning time, with bootstrap 95% intervals, on a test split touched once. Also a temporal split (train on M5-M7, test on M8-M9).

## Research track
Beyond v1, the long-term plan is in `docs/ROADMAP.md`. Built so far:
- **E8 conformal alarm threshold**: a provable limit on the false-alarm rate for new clean shots, and a test of where it breaks (later campaigns).
- **E9 positive-unlabelled learning**: shots that end in a fast quench without an operator note are treated as unlabelled instead of clean, and the model estimates how disruption-like they are.
- **Premature alarms**: catches more than 200 ms early are reported separately.
- **E10 seed sweep**: the same experiment on many random splits, so every number comes with its spread.
- **Termination study** (`scripts/termination_study.py`, real data, `results/termination_study/report.md`): quenches the operators did not note happen during the programmed flat-top just like noted disruptions, and slightly closer to the end of the central-solenoid swing.
- **Data catalogue** (`scripts/inspect_groups.py`, `results/data_catalogue/`): every signal stored per shot.
- **Magnetics and control signals** (`src/extras.py`, E13): saddle-coil n=1 amplitude (locked modes), Mirnov fluctuations, solenoid current and the gap between measured and programmed current.
- **Shortcut test** (E11) and **UKAEA-style protocol** (E12).
- **Gold labels by hand** on Telegram (`/gold`), blind to every automatic label.
- **Watchers**: daily data-source check (shot table back? new shots? DEFUSE public?) and a weekly arXiv digest of new disruption-prediction papers.
- **Literature review** with checked references (`docs/literature_review.md`) and a paper skeleton (`docs/paper/draft.md`).

## How to run (anyone)
Everything can run from `notebooks/colab_pipeline.ipynb` or a terminal:

```
pip install -r requirements.txt
pytest
python scripts/build_dataset.py --scale small      # 300 disrupted + 300 clean shots -> data/processed/
python scripts/run_experiment.py --exp E3          # one experiment -> results/E3/, model -> models/E3/
python scripts/run_all.py --scale small            # everything, E0-E13 (unchanged results are reused)
python scripts/predict_shot.py --shot 11860        # run the best trained model on one shot, as if live
```

If `mastapp.site` (the shot table) is down, `python scripts/build_dataset.py --from-report results/label_report/per_shot.csv` builds a dataset from the 400 shots of the Phase 2 report instead. `python scripts/diagnose_network.py` explains any connection problem.

## Public code, private runs
The repository holds only code, docs and small reference results. Everything a training run produces (downloaded data, trained models, results, logs) goes to the folder named by `MAST_HOME`; unset, that is the current folder (git-ignored). On a personal server the setup puts everything, including the private settings file with the tokens (`~/mast-private/.env`), in `~/mast-private`. That folder is outside the code, so deleting or re-cloning the code never touches it, and your own runs never end up on GitHub. Anyone who clones the repo gets the same pipeline and trains their own models.

## Running forever on a personal VPS
1. Make a Telegram bot with @BotFather and note the token and your chat id.
2. `git clone https://github.com/fazil-mastu/mastu.git && cd mastu && bash deploy/setup_vps.sh`. It installs everything, asks for the bot token, chat id and (optionally) a GitHub token, stores them in `~/mast-private/.env`, runs the tests and the network check, and installs the research server as a systemd service (it restarts after crashes and reboots).
   - **If STFC blocks your server** (its IP range; check with `scripts/diagnose_network.py`), give the setup a GitHub fine-grained token: Settings > Developer settings > Fine-grained tokens, repository `mastu` only, permission **Actions: read and write**. The server then uses the **GitHub data relay**: a GitHub Actions job (`.github/workflows/fetch_data.yml`), which can reach STFC, downloads the shots and hands them over as a one-day artifact; your server unpacks them into `~/mast-private/data` and trains there. Only public MAST data passes through GitHub; nothing is committed.
3. The research server (`scripts/research_server.py`) then: waits out network outages, runs the full pipeline for every new commit pushed to GitHub, spends idle time on the E10 seed sweep, sends a daily heartbeat, and answers commands from your chat only: `/status` (the picture), `/report` (every experiment in words, plus all result files), `/details`, `/log`, `/predict 11860`, `/diagnose`, `/run`, `/fetch`, `/sources`, `/papers`, `/gold` (then `/d`, `/n`, `/u`), `/goldreport`, `/update`, `/pause`, `/resume`, `/stop`, `/help`.
4. Prefer `screen`? Stop the service (`sudo systemctl disable --now mast-research`) and run `screen -S mast .venv/bin/python scripts/research_server.py`. Only one copy can run at a time.
5. Restarts and updates never retrain for nothing. Every result is stored with a fingerprint of the science code and data that made it (`src/resume.py`); after a restart, or a commit that only touches docs, the server or the messages, the run reuses those results and says so. Only work whose inputs changed runs again, and a job left running by a killed server is stopped so two runs never clash.
6. Telegram stays quiet: you get one status picture (`src/status_card.py`) at the end of a full run, every 10 repeat runs and once a day: how far training has got, what the model can do compared with a random guess, which signals it relies on, and what is needed next. Failures and things that need you still come as short messages. Everything else is kept in `logs/messages.log`, and every experiment is described in plain words (`src/explain.py`) in `results/REPORT.md` (`/report`).

## Results
Model results: TBD in this repository; they are produced on the owner's private server. Real-data analyses committed here: the label comparison (`results/label_report/report.md`, from code that ran on 400 real shots): the current-quench detector confirms 97.7% of note-disrupted shots but also flags 74.2% of note-clean shots, because most MAST shots end with a fast current quench whether or not anyone called it a disruption. That is why `LABEL_SOURCE = "both"`. The termination study adds that both kinds of quench happen while the programmed current is still at its flat-top value (median 1.00 of its maximum in both groups).

## Limitations
Labels are algorithmic and imperfect. The main models use 4 global signals, and averaged over 43 random splits they are not yet better than an alarm that fires at random as often. The magnetics (E13) are a first version: the saddle-coil signal may contain pickup from the PF coils, and a fixed correction made it worse (NOTES.md, 2026-10-16). There are no profiles. MAST (2000 to 2013), not MAST-U. Offline evaluation, not real-time control. Results come from a subset of shots unless stated otherwise.

## Credits
Data: FAIR-MAST, UKAEA, licence CC BY-SA 4.0 (https://mastapp.site).
Related work: UKAEA FAIR-MAST event-prediction benchmarks (IAEA FEC poster, https://conferences.iaea.org/event/392/contributions/36328/). PU learning: Elkan & Noto, KDD 2008.
