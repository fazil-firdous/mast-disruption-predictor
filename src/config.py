"""Every tunable parameter in one place, so each choice can be found and explained."""
import os

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_env_file(path):
    """Copy KEY=VALUE lines from a private .env file into os.environ. Variables already set in the shell win."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"'))


# Private settings (bot token, chat id, GitHub token, MAST_HOME) live only on the owner's server. The main copy is
# in the permanent private folder, so deleting or re-cloning the code loses nothing; a repo .env still works too.
PRIVATE_ENV = os.path.expanduser(os.path.join("~", "mast-private", ".env"))
_load_env_file(os.path.join(REPO_DIR, ".env"))
_load_env_file(PRIVATE_ENV)

# Data sources (SPEC 3.1)
SHOT_TABLE_URL = "https://mastapp.site/parquet/level2/shots"
# env override lets tests and offline work point at a local copy of the stores
SHOT_ZARR_URL = os.environ.get("MAST_ZARR_URL", "https://s3.echo.stfc.ac.uk/mast/level2/shots/{shot_id}.zarr")
SUMMARY_GROUP = "summary"
REFERENCE_SHOT = 11860  # operator note says "DISRUPTION AT 220MS"; used as a sanity check

# Definitions (SPEC Section 4)
IP_ON = 50e3  # A; below this there is effectively no plasma, so t_start is the first |Ip| >= 50 kA
WINDOW_MS = 20  # 20 samples at 1 ms: long enough for a stable slope, short enough to react
STRIDE_MS = 5  # new window every 5 ms; finer steps mostly add near-duplicate windows
HORIZON_MS = 30  # warn up to 30 ms ahead; much earlier and precursors are hard to tell from normal flat-top
K_CONSEC = 2  # need two windows in a row over threshold, so one noisy window can't fire an alarm
MIN_WARNING_MS = 5  # less than ~5 ms leaves no time to act, so such alarms count as tardy
CQ_DROP_FRAC = 0.8  # a real current quench loses most of the current, not a small dip
CQ_MAX_MS = 10  # MAST quenches take a few ms; controlled ramp-downs take tens of ms
MIN_PEAK_IP = 100e3  # A; shots that never reach 100 kA are too weak to say anything about disruptions
RANDOM_SEED = 42  # one seed for splits, sampling and models, so every run is repeatable

# Label policy (SPEC 5.3)
LABEL_SOURCE = "both"  # "signal" | "note" | "both"; owner chose "both" after the Phase 2 report: the detector alone flags ~74% of note-clean shots
SPIKE_LOOKBACK_MS = 3  # Ip spike within 3 ms before the CQ marks the thermal quench (SPEC 5.2 step 4)
MEDIAN_FILTER_SAMPLES = 3  # smallest median filter that removes single-sample glitches
CQ_MIN_FRAC_OF_PEAK = 0.5  # quench must start above half of peak |Ip|; the last stretch of any linear ramp-down also falls 80% in 10 ms once |Ip| is small
CQ_ONSET_FRAC = 0.9  # quench onset = last sample still within 90% of the pre-fall level; the 80%-drop search alone can fire up to CQ_MAX_MS too early
SPIKE_BASELINE_MS = 10  # the "level before the spike" is the median of the 10 ms before the lookback window
SPIKE_MIN_RISE = 0.03  # a spike counts only if it is 3% above the level just before it, so flat-top noise is not mistaken for one

# Feature offsets (SPEC Section 7)
PNBI_OFFSET = 1e5  # W; stops prad/pnbi blowing up when the beams are off
IP_OFFSET = 1e3  # A; stops prad/ip blowing up near zero current
NEUTRON_OFFSET = 1.0  # Hz; stops the relative neutron slope dividing by zero
PNBI_ON_THRESHOLD = 1e5  # W; above 100 kW the beams count as on

# Dataset (SPEC Section 8)
N_DISRUPTED = 300  # balanced starting subset: big enough to learn, small enough to download quickly
N_CLEAN = 300
DOWNLOAD_WORKERS = 8  # parallel downloads; more risks being throttled by the server
DOWNLOAD_RETRIES = 3  # transient HTTP errors are common when streaming many shots
FULL_N_CLEAN = 3000  # SPEC 8.1 "scale up": all detected disruptions plus ~3000 clean shots (run_all.py --scale full)
CLEAN_POOL_FACTOR = 1.5  # download 1.5x the clean shots needed, since some turn out unknown or disrupted once labelled
SPLIT_FRACTIONS = (0.6, 0.2, 0.2)  # train / val / test, split by shot
TEMPORAL_VAL_FRACTION = 0.2  # in the temporal split, 20% of the M5-M7 shots are held back for tuning
TEMPORAL_TRAIN_CAMPAIGNS = ("M5", "M6", "M7")  # the table also has "Unknown" (292 shots); it cannot be placed in time, so it is left out
TEMPORAL_TEST_CAMPAIGNS = ("M8", "M9")

# Phase 4 window policy (NOTES.md 2026-10-15)
STOP_AT_QUENCH = True  # end every shot's windows at the detector's quench, so clean and disrupted shots end the same way
NEAR_QUENCH_CLEAN = "keep"  # "keep" or "drop" clean windows in the HORIZON before an unnoted quench, in training only

# Models (SPEC Section 9)
RF_TREES = 300  # SPEC 9; more trees barely change the scores but cost time
HGB_VALIDATION_FRACTION = 0.1  # share of training windows HistGradientBoosting holds back to decide when to stop
N_THRESHOLDS = 201  # alarm thresholds tried from 0 to 1 in steps of 0.005
PERMUTATION_REPEATS = 5  # shuffles per feature for permutation importance; enough to see which features matter

# Evaluation (SPEC Sections 10-11)
MAX_FALSE_ALARM_RATE = 0.05  # pick the threshold so at most 5% of clean shots raise an alarm
N_BOOTSTRAP = 1000  # resamples over shots for the 95% confidence intervals
PREMATURE_WARNING_MS = 200  # MAST flat-tops last ~200-300 ms, so an alarm earlier than this is unlikely to be about this disruption

# Extra physics signals (src/extras.py, ROADMAP D1)
USE_EXTRAS = True  # also fetch magnetics, programmed current and solenoid current (~3-4 MB read per shot)
EXTRAS_BASELINE_MS = 5  # saddle-coil offsets: mean of each coil's first 5 ms of data (stored only while plasma exists)

# Gold labels by hand (src/gold.py, ROADMAP A2): over-sample the unnoted quenches, the case that matters most
GOLD_PLAN = {"unnoted_quench": 35, "noted_quench": 30, "no_quench": 30, "noted_no_quench": 5}

# Research extensions (docs/ROADMAP.md)
CONFORMAL_ALPHAS = (0.01, 0.05, 0.10)  # false-alarm levels to guarantee in E8
PU_PARAMS = {"learning_rate": 0.1, "max_leaf_nodes": 31}  # fixed HGB settings for E9, so standard and PU models differ only in the labels

# Signals cached per shot (SPEC 8.2): parquet column -> variable name in the summary group
SIGNAL_VARIABLES = {
    "ip": "ip",
    "power_radiated": "power_radiated",
    "neutron_rates_total": "neutron_rates_total",
    "power_nbi": "power_nbi",
}
RETRY_BACKOFF_S = 2.0  # wait 2 s, then 4 s, then 8 s between retries, so a busy server can recover
TCP_CHECK_TIMEOUT_S = 10  # a server that has not answered a plain connection in 10 s is blocked or down
CONNECTION_CHECK_ATTEMPTS = 4  # before a big run; one timeout on the VPS (2026-10-15) killed a whole run

# Paths. Public code, private runs: on a personal server MAST_HOME points outside the repo, so downloaded data,
# trained models, results and logs never end up in git. Unset, everything goes under the current folder as before.
HOME_DIR = os.environ.get("MAST_HOME", ".")
DATA_DIR = os.path.join(HOME_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(HOME_DIR, "results")
MODELS_DIR = os.path.join(HOME_DIR, "models")
LOGS_DIR = os.path.join(HOME_DIR, "logs")
SHOT_TABLE_CACHE = os.path.join(DATA_DIR, "shot_table.parquet")  # the table is ~11.5k rows, so one copy avoids re-downloading it
SKIPPED_CSV = os.path.join(DATA_DIR, "skipped.csv")  # every shot we could not load, with the reason (SPEC 8.1)
REPORT_CSV = os.path.join(REPO_DIR, "results", "label_report", "per_shot.csv")  # 400 labelled shots; stand-in when mastapp.site is down

# GitHub data relay (ROADMAP H): for servers whose IP range STFC blocks, a GitHub Actions machine downloads the shots
GITHUB_REPO = os.environ.get("GITHUB_REPO", "fazil-mastu/mastu")
RELAY_WORKFLOW = "fetch_data.yml"
RELAY_BRANCH = "main"  # GitHub only runs dispatched workflows that exist on this branch
RELAY_TIMEOUT_HOURS = 6  # GitHub's own limit for one job

# Research server (scripts/research_server.py)
NETWORK_RETRY_MINUTES = 30  # when the data server is unreachable, check again this often instead of giving up
PIPELINE_RETRY_HOURS = 6  # a failed full run of the same commit is retried at most this often
HEARTBEAT_HOURS = 24  # one "still alive" Telegram message a day
SEED_SWEEP_MAX = 50  # idle work: re-split and re-test with seeds 1..50 for honest mean +- spread (E10)
TELEGRAM_POLL_S = 25  # long-poll length for bot commands; also the loop's natural tick
WATCH_SOURCES_HOURS = 24  # check once a day whether mastapp.site is back, FAIR-MAST grew, or DEFUSE went public
LITERATURE_DAYS = 7  # weekly arXiv check; disruption papers appear a few times a month
LITERATURE_MAX_RESULTS = 40  # newest matches per check; more than a week ever brings
CARD_EVERY_SEEDS = 10  # send the status picture after every 10 repeat runs, not after each one
NEXT_STEP = "magnetic signals as change, not level"  # the research step in progress (docs/ROADMAP.md), shown on the status picture
UPDATE_CHECK_MINUTES = 10  # look for new commits on GitHub this often; more often is pointless load on GitHub
