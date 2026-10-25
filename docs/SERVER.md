# Running the pipeline continuously on a server

Optional. Everything also runs from a terminal or the Colab notebook (see the README). This page describes the always-on research server used for the long runs.

## What it does
`scripts/research_server.py` runs forever:
- it waits out network outages and runs the full pipeline for every new commit pushed to GitHub;
- results whose science code and data are unchanged are reused, not retrained (`src/resume.py`);
- idle time goes to the E10 seed sweep (the same experiment on 50 random splits, for error bars);
- it labels every shot's ending automatically after each new dataset (`scripts/auto_labels.py`);
- jobs run at low CPU priority, so other programs on the server stay responsive;
- in its terminal it shows a small tokamak and progress bars (`src/banner.py`).

Only one copy can run at a time (lock file). A job left running by a killed server is stopped when the next one starts.

## Setup
`git clone https://github.com/fazil-mastu/mastu.git && cd mastu && bash deploy/setup_vps.sh`

The setup installs everything into a virtual environment, keeps all private settings and outputs in `~/mast-private` (outside the code, so re-cloning never touches them), runs the tests and a network check, and installs a systemd service. To use `screen` instead, disable the service (`sudo systemctl disable --now mast-research`) and run `.venv/bin/python scripts/research_server.py --scale full` inside a screen session.

## If the server cannot reach the data servers
`python scripts/diagnose_network.py` tells you. In that case give the setup a GitHub fine-grained token (repository `mastu` only, permission **Actions: read and write**). The server then uses the **GitHub data relay**: a GitHub Actions job (`.github/workflows/fetch_data.yml`) downloads the shots and hands them over as a one-day artifact, which the server unpacks into `~/mast-private/data`. Only public MAST data passes through GitHub, and nothing is committed.

The shot table is read from mastapp.site, or from its copy in the FAIR-MAST S3 bucket (`fairmast-meta/l2_shots.parquet`) when mastapp.site does not answer.

## Optional notifications (Telegram)
If a Telegram bot token and chat id are given at setup, the server sends one status picture (`src/status_card.py`) at the end of a full run, every 10 repeat runs and once a day, plus a short message when something fails. Everything else stays in `logs/messages.log` and `results/REPORT.md`. Commands are accepted from that one chat only, and each maps to a fixed action: `/status`, `/report`, `/details`, `/log`, `/predict 11860`, `/diagnose`, `/run`, `/fetch`, `/sources`, `/papers`, `/gold` (optional expert labelling), `/goldreport`, `/update`, `/pause`, `/resume`, `/stop`, `/help`. Without a bot, messages are simply printed.
