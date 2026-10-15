"""Extra physics signals beyond the summary group (ROADMAP D1), reduced to the shot's 1 ms time grid, causally.

- lm_n1: n=1 amplitude of the outboard midplane saddle coils (12 coils, 30 degrees apart): the usual
  locked-mode signal. Each channel is averaged over each 1 ms bin, its offset (mean of its first
  EXTRAS_BASELINE_MS of data) is removed, and B(phi) = a0 + a1 cos(phi) + b1 sin(phi) is fitted;
  lm_n1 = sqrt(a1^2 + b1^2) in tesla. It stays NaN until that offset is known, so it never looks ahead.
- mirnov_rms: RMS of one centre-column Mirnov coil about its 1 ms mean (fluctuations above ~1 kHz), tesla.
- ip_ref: programmed plasma current (pulse_schedule), A; sol_current: central solenoid current (pf_active), A.

Every value at grid time t uses only fast samples with t_prev < time <= t, so it is known at t.
Cached per shot in data/extras/{shot_id}.parquet.
"""
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from src import config
from src.data import load_shot, open_group

COLUMNS = ["time", "lm_n1", "mirnov_rms", "ip_ref", "sol_current"]
SADDLE_PHI_DEG = np.arange(15.0, 360.0, 30.0)  # centres of the middle-row saddle coils M01..M12 (see NOTES.md)
MIN_SADDLE_CHANNELS = 6  # fewer working coils cannot pin down an n=1 cosine plus offset reliably


def bin_index(t_fast, t_grid):
    """For each fast sample, the grid bin it belongs to: bin i holds t_grid[i-1] < t <= t_grid[i].

    Samples after the last grid time get index len(t_grid) and are ignored by the callers.
    """
    return np.searchsorted(t_grid, t_fast, side="left")


def bin_mean(t_fast, x, t_grid):
    """Mean of x within each grid bin (NaN where a bin has no finite sample)."""
    idx = bin_index(t_fast, t_grid)
    ok = np.isfinite(x) & (idx < len(t_grid))
    sums = np.bincount(idx[ok], weights=x[ok], minlength=len(t_grid))[:len(t_grid)]
    counts = np.bincount(idx[ok], minlength=len(t_grid))[:len(t_grid)]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / counts, np.nan)


def bin_rms(t_fast, x, t_grid):
    """RMS of x about its own mean within each grid bin (NaN where a bin has fewer than 2 samples)."""
    mean = bin_mean(t_fast, x, t_grid)
    idx = bin_index(t_fast, t_grid)
    ok = np.isfinite(x) & (idx < len(t_grid))
    deviation = x[ok] - mean[idx[ok]]
    sq = np.bincount(idx[ok], weights=deviation ** 2, minlength=len(t_grid))[:len(t_grid)]
    counts = np.bincount(idx[ok], minlength=len(t_grid))[:len(t_grid)]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 1, np.sqrt(sq / counts), np.nan)


def hold_last(t_slow, x, t_grid):
    """Value of x at the last slow sample at or before each grid time (NaN before the first sample)."""
    ok = np.isfinite(x)
    t_slow, x = t_slow[ok], x[ok]
    idx = np.searchsorted(t_slow, t_grid, side="right") - 1
    out = np.full(len(t_grid), np.nan)
    out[idx >= 0] = x[idx[idx >= 0]]
    return out


def n1_amplitude(binned, phi_deg):
    """n=1 amplitude for each time from binned channel values (channels x times), skipping NaN channels."""
    amplitude = np.full(binned.shape[1], np.nan)
    phi = np.radians(phi_deg)
    for j in range(binned.shape[1]):
        ok = np.isfinite(binned[:, j])
        if ok.sum() < MIN_SADDLE_CHANNELS:
            continue
        design = np.column_stack([np.ones(ok.sum()), np.cos(phi[ok]), np.sin(phi[ok])])
        coef = np.linalg.lstsq(design, binned[ok, j], rcond=None)[0]
        amplitude[j] = np.hypot(coef[1], coef[2])
    return amplitude


def early_offset(t_fast, x):
    """(mean of x over its first EXTRAS_BASELINE_MS of finite data, time that mean is known); (NaN, inf) if none.

    The saddle data is stored only while the plasma exists (from ~+2 ms), so there is no pre-plasma baseline.
    """
    ok = np.isfinite(x)
    if not ok.any():
        return np.nan, np.inf
    known_at = t_fast[ok][0] + config.EXTRAS_BASELINE_MS / 1000
    first = ok & (t_fast <= known_at)
    return float(np.mean(x[first])), known_at


def saddle_n1(shot_id, t_grid):
    """lm_n1 on t_grid for one shot (all NaN if the saddle data is missing)."""
    mag = open_group(shot_id, "magnetics")
    if "b_field_tor_probe_saddle_field" not in mag:
        return np.full(len(t_grid), np.nan)
    t_fast = mag["time_saddle"].values
    field = mag["b_field_tor_probe_saddle_field"].values
    binned = np.vstack([bin_mean(t_fast, channel, t_grid) for channel in field])
    offsets, known = zip(*[early_offset(t_fast, channel) for channel in field])
    amplitude = n1_amplitude(binned - np.array(offsets)[:, None], SADDLE_PHI_DEG[:len(field)])
    finite_known = [k for k in known if np.isfinite(k)]
    if finite_known:
        amplitude[t_grid < max(finite_known)] = np.nan
    return amplitude


def mirnov_rms(shot_id, t_grid):
    """mirnov_rms on t_grid from the first centre-column poloidal Mirnov coil that has data (else all NaN)."""
    mag = open_group(shot_id, "magnetics")
    name = "b_field_pol_probe_cc_field"
    if name not in mag:
        return np.full(len(t_grid), np.nan)
    t_fast = mag["time_mirnov"].values
    for k in range(mag[name].shape[0]):
        x = mag[name][k].values
        if np.isfinite(x).any():
            return bin_rms(t_fast, x, t_grid)
    return np.full(len(t_grid), np.nan)


def slow_signal(shot_id, group, variable, t_grid, channel=None):
    """A slow signal held at its last value up to each grid time; NaN if the group, variable or channel is missing."""
    try:
        ds = open_group(shot_id, group)
    except Exception:
        return np.full(len(t_grid), np.nan)
    if variable not in ds:
        return np.full(len(t_grid), np.nan)
    values = ds[variable].values
    if channel is not None:
        names = [str(c) for c in ds["current_channel"].values]
        if channel not in names:
            return np.full(len(t_grid), np.nan)
        values = values[names.index(channel)]
    return hold_last(ds["time"].values, np.asarray(values, dtype=float), t_grid)


def compute_extras(shot_id):
    """All extra signals of one shot on its summary time grid, as a DataFrame with COLUMNS."""
    t_grid = load_shot(shot_id)["time"].to_numpy()
    out = pd.DataFrame({"time": t_grid})
    for column, func in [("lm_n1", saddle_n1), ("mirnov_rms", mirnov_rms)]:
        try:
            out[column] = func(shot_id, t_grid)
        except Exception:
            out[column] = np.nan
    out["ip_ref"] = np.abs(slow_signal(shot_id, "pulse_schedule", "i_plasma", t_grid))
    out["sol_current"] = slow_signal(shot_id, "pf_active", "coil_current", t_grid, channel="SOL")
    return out[COLUMNS]


def _cache_path(shot_id):
    return os.path.join(config.DATA_DIR, "extras", f"{shot_id}.parquet")


def load_extras(shot_id, allow_download=True):
    """Extras for one shot from data/extras/, computed and cached on first use. None if not cached and
    allow_download is False (offline runs)."""
    path = _cache_path(shot_id)
    if os.path.exists(path):
        return pd.read_parquet(path)
    if not allow_download:
        return None
    frame = compute_extras(shot_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frame.to_parquet(path)
    return frame


def download_extras(ids, workers=config.DOWNLOAD_WORKERS):
    """Compute and cache extras for every shot in ids (skipping cached ones). Returns {'ok': n, 'failed': [ids]}.

    Each shot reads ~3-4 MB of fast magnetics from STFC, so this runs in parallel like download_shots.
    """
    todo = [i for i in ids if not os.path.exists(_cache_path(i))]
    failed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(load_extras, i): i for i in todo}
        for future in tqdm(as_completed(futures), total=len(futures), desc="extras"):
            try:
                future.result()
            except Exception:
                failed.append(futures[future])
    return {"ok": len(ids) - len(failed), "failed": failed}


def with_extras(shot_df, shot_id):
    """shot_df plus the cached extra columns (NaN columns if this shot has none cached)."""
    extra = load_extras(shot_id, allow_download=False)
    out = shot_df.copy()
    for column in COLUMNS[1:]:
        if extra is not None and len(extra) == len(out) and np.allclose(extra["time"], out["time"]):
            out[column] = extra[column].to_numpy()
        else:
            out[column] = np.nan
    return out
