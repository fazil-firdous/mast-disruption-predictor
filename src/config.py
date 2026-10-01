"""Every tunable parameter in one place, so each choice can be found and explained."""
import os

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
SPLIT_FRACTIONS = (0.6, 0.2, 0.2)  # train / val / test, split by shot
TEMPORAL_TRAIN_CAMPAIGNS = ("M5", "M6", "M7")  # the table also has "Unknown" (292 shots); it cannot be placed in time, so it is left out
TEMPORAL_TEST_CAMPAIGNS = ("M8", "M9")

# Evaluation (SPEC Sections 10-11)
MAX_FALSE_ALARM_RATE = 0.05  # pick the threshold so at most 5% of clean shots raise an alarm
N_BOOTSTRAP = 1000  # resamples over shots for the 95% confidence intervals

# Signals cached per shot (SPEC 8.2): parquet column -> variable name in the summary group
SIGNAL_VARIABLES = {
    "ip": "ip",
    "power_radiated": "power_radiated",
    "neutron_rates_total": "neutron_rates_total",
    "power_nbi": "power_nbi",
}
RETRY_BACKOFF_S = 2.0  # wait 2 s, then 4 s, then 8 s between retries, so a busy server can recover

# Paths
DATA_DIR = "data"
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
RESULTS_DIR = "results"
SHOT_TABLE_CACHE = "data/shot_table.parquet"  # the table is ~11.5k rows, so one copy avoids re-downloading it
SKIPPED_CSV = "data/skipped.csv"  # every shot we could not load, with the reason (SPEC 8.1)
