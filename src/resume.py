"""Skip work that is already done. A saved result is reused only when the science code and the data that made it
are exactly the same as now, so a server restart or a docs-only commit never retrains, and no number is stale.

A stamp is a short fingerprint (sha256) of every science source file, the science settings in config.py and the
input data files. Any change to them, even a comment in a science file, gives a new stamp and the work is redone once.
"""
import hashlib
import json
import os

import numpy as np

from src import config

# these files run the server, talk to Telegram or GitHub, or word the messages: they cannot change a number.
# config.py is left out too; its science values are stamped one by one (science_settings), so comments do not count
NOT_SCIENCE = {"__init__.py", "config.py", "server.py", "notify.py", "relay.py", "sources.py", "literature.py",
               "gold.py", "explain.py", "resume.py", "status_card.py", "banner.py", "stats.py"}
# config settings that only steer the server, downloads, Telegram or where files go: changing them changes no number
NOT_SCIENCE_SETTINGS = {
    "REPO_DIR", "PRIVATE_ENV", "HOME_DIR", "DATA_DIR", "RAW_DIR", "PROCESSED_DIR", "RESULTS_DIR", "MODELS_DIR",
    "LOGS_DIR", "SHOT_TABLE_CACHE", "SKIPPED_CSV", "REPORT_CSV", "DOWNLOAD_WORKERS", "DOWNLOAD_RETRIES",
    "RETRY_BACKOFF_S", "TCP_CHECK_TIMEOUT_S", "CONNECTION_CHECK_ATTEMPTS", "GITHUB_REPO", "RELAY_WORKFLOW",
    "RELAY_BRANCH", "RELAY_TIMEOUT_HOURS", "NETWORK_RETRY_MINUTES", "PIPELINE_RETRY_HOURS", "HEARTBEAT_HOURS",
    "SEED_SWEEP_MAX", "TELEGRAM_POLL_S", "WATCH_SOURCES_HOURS", "LITERATURE_DAYS", "LITERATURE_MAX_RESULTS",
    "CARD_EVERY_SEEDS", "NEXT_STEP", "UPDATE_CHECK_MINUTES", "JOB_NICE", "MEMORY_SPARE",
    # stamped on its own by scripts/repeat_runs.py, so adding a comparison never reruns E0-E15
    "COMPARISONS", "COMPARISON_BASELINES", "COMPARISON_SEEDS",
    # where the data comes from: the data itself is stamped by content, so a moved address retrains nothing
    "SHOT_TABLE_URL", "SHOT_ZARR_URL", "SUMMARY_GROUP", "RELAY_PARTS", "RELAY_WORK_HOURS"}
SCIENCE_SCRIPTS = ["build_dataset.py"]  # scripts with science logic in them (shot selection, splits)
STAMP_FILE = "stamp.json"


def science_files():
    """Every science source file (src/*.py minus NOT_SCIENCE, plus SCIENCE_SCRIPTS), sorted."""
    src = os.path.join(config.REPO_DIR, "src")
    files = [os.path.join(src, name) for name in os.listdir(src) if name.endswith(".py") and name not in NOT_SCIENCE]
    files += [os.path.join(config.REPO_DIR, "scripts", name) for name in SCIENCE_SCRIPTS]
    return sorted(files)


def science_settings():
    """Every UPPERCASE value in config except NOT_SCIENCE_SETTINGS (and anything secret), as sorted JSON text."""
    values = {k: v for k, v in vars(config).items()
              if k.isupper() and k not in NOT_SCIENCE_SETTINGS and "TOKEN" not in k and "SECRET" not in k}
    return json.dumps(values, sort_keys=True, default=str)


def processed_files():
    """The dataset every experiment reads (written by scripts/build_dataset.py)."""
    return [os.path.join(config.PROCESSED_DIR, name) for name in ("shots.parquet", "windows.parquet")]


def file_digest(paths):
    """sha256 (hex) over the names and contents of the given files, in the given order."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(os.path.basename(path).encode())
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def code_digest():
    """Digest of the science source files as they are on disk now."""
    return file_digest(science_files())


# taken when the program starts, i.e. from the code it actually loaded; a file edited while it runs must not
# lend its fingerprint to results made by the old code still in memory
CODE_AT_START = code_digest()


def stamp(data_files, extra="", code=None):
    """Fingerprint of the science code (as loaded at start, unless code is given), the science settings, the given
    data files (name and content) and any extra text."""
    digest = hashlib.sha256((code or CODE_AT_START).encode())
    digest.update(science_settings().encode())
    digest.update(file_digest(sorted(data_files)).encode())
    digest.update(extra.encode())
    return digest.hexdigest()[:16]


def load(folder, current):
    """The outcome saved in folder if it was made under the current stamp, else None."""
    path = os.path.join(folder, STAMP_FILE)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        saved = json.load(f)
    return saved["outcome"] if saved.get("stamp") == current else None


def save(folder, current, outcome):
    """Record that outcome was made under the current stamp. Written last and atomically, so a job killed
    half-way never leaves a stamp for unfinished work."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, STAMP_FILE)
    with open(path + ".tmp", "w") as f:
        json.dump({"stamp": current, "outcome": clean_json(outcome)}, f, indent=2)
    os.replace(path + ".tmp", path)


def clean_json(value):
    """value with numpy numbers turned into plain ones and NaN / inf into None, so it is valid JSON."""
    if isinstance(value, dict):
        return {str(k): clean_json(x) for k, x in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(x) for x in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
