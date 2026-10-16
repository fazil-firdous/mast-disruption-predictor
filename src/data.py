"""Load the shot table and per-shot summary signals, with a local parquet cache."""
import csv
import os
import socket
import time
import urllib.parse
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


def host_reachable(url, timeout=config.TCP_CHECK_TIMEOUT_S):
    """True if a plain TCP connection to the URL's host opens within timeout. Local paths count as reachable.

    A blocked server never answers, and the HTTP libraries can take minutes to give up; this takes seconds.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return True
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((parsed.hostname, port), timeout=timeout):
            return True
    except OSError:
        return False


def check_shot_table():
    """(ok, message): the shot table is cached, or can be downloaded now. mastapp.site has had outages (503)."""
    if os.path.exists(config.SHOT_TABLE_CACHE):
        return True, "shot table cached"
    if not host_reachable(config.SHOT_TABLE_URL):
        return False, "shot table server does not answer (blocked from here, or down)"
    try:
        table = load_shot_table()
    except Exception as error:
        return False, f"shot table server unavailable: {type(error).__name__}: {error}"[:300]
    return True, f"shot table downloaded: {len(table)} shots"


def _open_summary(shot_id):
    """Open one shot's summary group over HTTPS (the only network call for signals, so tests replace it)."""
    # zarr v3 stores since 2026-09-22 (zarr<3 cannot read them). The array list lives only in the root's
    # consolidated metadata and S3 cannot list folders, so consolidated=False sees an empty group.
    url = config.SHOT_ZARR_URL.format(shot_id=shot_id)
    return xr.open_zarr(url, group=config.SUMMARY_GROUP, consolidated=True)


def open_group(shot_id, group):
    """Open any group of one shot's store (e.g. 'pulse_schedule', 'pf_active', 'magnetics') as an xarray Dataset.

    Not cached: used for studies and for checking a group before it becomes a feature.
    """
    url = config.SHOT_ZARR_URL.format(shot_id=shot_id)
    return xr.open_zarr(url, group=group, consolidated=True)


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


def cache_path(shot_id):
    """Where one shot's cached signals live: data/raw/{shot_id}.parquet."""
    return os.path.join(config.RAW_DIR, f"{shot_id}.parquet")


def is_cached(shot_id):
    """True if the shot's signals are already on disk (so loading it needs no network)."""
    return os.path.exists(cache_path(shot_id))


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
    path = cache_path(shot_id)
    if use_cache and os.path.exists(path):
        return pd.read_parquet(path)
    frame = _fetch_shot(shot_id, retries, backoff)
    os.makedirs(config.RAW_DIR, exist_ok=True)
    frame.to_parquet(path)
    return frame


def check_connection(shot_id=config.REFERENCE_SHOT, attempts=1, backoff=config.RETRY_BACKOFF_S):
    """Try to read one known shot straight from the server (no cache), up to attempts times.

    Returns (ok, message). The default single attempt fails fast; big runs pass more attempts so one
    timeout does not kill them. If it keeps failing, run scripts/diagnose_network.py.
    """
    url = config.SHOT_ZARR_URL.format(shot_id=shot_id)
    if not host_reachable(url):
        return False, f"{urllib.parse.urlparse(url).hostname} does not answer (blocked from here, or down)"
    message = ""
    for attempt in range(attempts):
        try:
            frame = _summary_to_frame(_open_summary(shot_id))
            return True, f"read shot {shot_id}: {len(frame)} samples"
        except Exception as error:
            message = f"could not read shot {shot_id} ({attempt + 1} attempts): {type(error).__name__}: {error}"[:300]
        if attempt < attempts - 1:
            time.sleep(backoff * 2**attempt)
    return False, message


def log_skipped(rows):
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
        if os.path.exists(cache_path(shot_id)):
            result["cached"].append(shot_id)
        else:
            to_fetch.append(shot_id)

    skipped_rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(load_shot, s, use_cache=True, retries=retries, backoff=backoff): s for s in to_fetch}
        for future in tqdm(as_completed(futures), total=len(futures), desc="downloading"):
            shot_id = futures[future]
            try:
                future.result()
                result["downloaded"].append(shot_id)
            except Exception as error:
                result["failed"].append(shot_id)
                skipped_rows.append([shot_id, f"{type(error).__name__}: {error}"[:200]])
    log_skipped(skipped_rows)
    return result
