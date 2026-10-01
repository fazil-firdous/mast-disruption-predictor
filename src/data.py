"""Load the shot table and per-shot summary signals, with a local parquet cache."""
import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import xarray as xr
from tqdm import tqdm

from src import config

COLUMNS = ["time"] + list(config.SIGNAL_VARIABLES)


def load_shot_table(use_cache=True):
    """Return the shot table (one row per shot) as a DataFrame, cached locally after the first download."""
    if use_cache and os.path.exists(config.SHOT_TABLE_CACHE):
        return pd.read_parquet(config.SHOT_TABLE_CACHE)
    table = pd.read_parquet(config.SHOT_TABLE_URL)
    os.makedirs(os.path.dirname(config.SHOT_TABLE_CACHE), exist_ok=True)
    table.to_parquet(config.SHOT_TABLE_CACHE)
    return table


def _open_summary(shot_id):
    # the stores were re-ingested as zarr v3 on 2026-09-22, which zarr<3 cannot read (KeyError '.zmetadata')
    url = config.SHOT_ZARR_URL.format(shot_id=shot_id)
    return xr.open_zarr(url, group=config.SUMMARY_GROUP, consolidated=False)


def _summary_to_frame(ds):
    """Turn an opened summary group into a DataFrame with the fixed column set."""
    if "time" not in ds.coords and "time" not in ds.variables:
        raise ValueError("summary group is empty or has no time (network blocked, or zarr<3 installed?)")
    times = ds["time"].values
    frame = pd.DataFrame({"time": times.astype("float64")})
    found = 0
    for column, variable in config.SIGNAL_VARIABLES.items():
        if variable in ds.variables and ds[variable].shape == times.shape:
            frame[column] = ds[variable].values.astype("float64")
            found += 1
        else:
            frame[column] = float("nan")
    if found == 0:
        raise ValueError("summary group has none of the expected signals")
    return frame[COLUMNS]


def _cache_path(shot_id):
    return os.path.join(config.RAW_DIR, f"{shot_id}.parquet")


def _fetch_shot(shot_id, retries, backoff):
    """Download one shot, retrying with growing waits. Raises the last error if all attempts fail."""
    last_error = None
    for attempt in range(retries + 1):
        try:
            return _summary_to_frame(_open_summary(shot_id))
        except Exception as error:
            last_error = error
            # a missing shot will not appear on a retry. An empty group is retried: zarr 3 returns one
            # when the connection fails, so it may be a network problem rather than missing data
            if isinstance(error, FileNotFoundError):
                break
            if attempt < retries:
                time.sleep(backoff * 2**attempt)
    raise last_error


def load_shot(shot_id, use_cache=True, retries=config.DOWNLOAD_RETRIES, backoff=config.RETRY_BACKOFF_S):
    """Return columns time, ip, power_radiated, neutron_rates_total, power_nbi for one shot.

    Signals the shot lacks come back as NaN columns. Reads data/raw/{shot_id}.parquet if it exists.
    """
    path = _cache_path(shot_id)
    if use_cache and os.path.exists(path):
        return pd.read_parquet(path)
    frame = _fetch_shot(shot_id, retries, backoff)
    os.makedirs(config.RAW_DIR, exist_ok=True)
    frame.to_parquet(path)
    return frame


def check_connection(shot_id=config.REFERENCE_SHOT):
    """Try to read one known shot straight from the server (no cache, no retries).

    Returns (ok, message). Use it before a big download so a blocked network fails fast with a clear reason.
    """
    try:
        frame = _summary_to_frame(_open_summary(shot_id))
    except Exception as error:
        return False, f"could not read shot {shot_id}: {type(error).__name__}: {error}"[:300]
    return True, f"read shot {shot_id}: {len(frame)} samples"


def _log_skipped(rows):
    """Append (shot_id, reason) rows to data/skipped.csv, writing the header if the file is new."""
    if not rows:
        return
    os.makedirs(os.path.dirname(config.SKIPPED_CSV), exist_ok=True)
    is_new = not os.path.exists(config.SKIPPED_CSV)
    with open(config.SKIPPED_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["shot_id", "reason"])
        writer.writerows(rows)


def download_shots(ids, workers=config.DOWNLOAD_WORKERS, retries=config.DOWNLOAD_RETRIES,
                   backoff=config.RETRY_BACKOFF_S):
    """Cache the summary signals of every shot in ids, in parallel, and log failures.

    Returns a dict with the lists 'cached' (already on disk), 'downloaded' and 'failed' (shot ids).
    """
    result = {"cached": [], "downloaded": [], "failed": []}
    to_fetch = []
    for shot_id in ids:
        if os.path.exists(_cache_path(shot_id)):
            result["cached"].append(shot_id)
        else:
            to_fetch.append(shot_id)

    skipped_rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(load_shot, s, True, retries, backoff): s for s in to_fetch}
        for future in tqdm(as_completed(futures), total=len(futures), desc="downloading"):
            shot_id = futures[future]
            try:
                future.result()
                result["downloaded"].append(shot_id)
            except Exception as error:
                result["failed"].append(shot_id)
                skipped_rows.append([shot_id, f"{type(error).__name__}: {error}"[:200]])
    _log_skipped(skipped_rows)
    return result
