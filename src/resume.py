"""Skip work that is already done. A saved result is reused only when the science code and the data that made it
are exactly the same as now, so a server restart or a docs-only commit never retrains, and no number is stale.

A stamp is a short fingerprint (sha256) of every science source file plus the input data files. Any change to
them, even a comment in a science file, gives a new stamp and the work is redone once.
"""
import hashlib
import json
import os

import numpy as np

from src import config

# these files run the server, talk to Telegram or GitHub, or word the messages: they cannot change a number
NOT_SCIENCE = {"__init__.py", "server.py", "notify.py", "relay.py", "sources.py", "literature.py", "gold.py",
               "explain.py", "resume.py"}
SCIENCE_SCRIPTS = ["build_dataset.py"]  # scripts with science logic in them (shot selection, splits)
STAMP_FILE = "stamp.json"


def science_files():
    """Every science source file (src/*.py minus NOT_SCIENCE, config.py included, plus SCIENCE_SCRIPTS), sorted."""
    src = os.path.join(config.REPO_DIR, "src")
    files = [os.path.join(src, name) for name in os.listdir(src) if name.endswith(".py") and name not in NOT_SCIENCE]
    files += [os.path.join(config.REPO_DIR, "scripts", name) for name in SCIENCE_SCRIPTS]
    return sorted(files)


def processed_files():
    """The dataset every experiment reads (written by scripts/build_dataset.py)."""
    return [os.path.join(config.PROCESSED_DIR, name) for name in ("shots.parquet", "windows.parquet")]


def stamp(data_files, extra=""):
    """Fingerprint of the science code, the given data files (path and content) and any extra text."""
    digest = hashlib.sha256()
    for path in science_files() + sorted(data_files):
        digest.update(os.path.basename(path).encode())
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
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
