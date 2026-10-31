"""Extra physics signals beyond the summary group (ROADMAP D1), reduced to the shot's 1 ms time grid, causally.

- lm_n1: n=1 amplitude of the outboard midplane saddle coils (12 coils, 30 degrees apart): the usual
  locked-mode signal. Each channel is averaged over each 1 ms bin, its offset (mean of its first
  EXTRAS_BASELINE_MS of data) is removed, and B(phi) = a0 + a1 cos(phi) + b1 sin(phi) is fitted;
  lm_n1 = sqrt(a1^2 + b1^2) in tesla. It stays NaN until that offset is known, so it never looks ahead.
- mirnov_rms: RMS of one centre-column Mirnov coil about its 1 ms mean (fluctuations above ~1 kHz), tesla.
- ip_ref: programmed plasma current (pulse_schedule), A; sol_current: central solenoid current (pf_active), A.
- mirnov_lf / _tm / _mf / _hf: RMS of the same Mirnov coil in the frequency bands of config.MIRNOV_BANDS_KHZ, from
  the spectrum of each 1 ms bin (~500 samples at 500 kHz), tesla. Which band carries a mode differs between shots.
- ne_line_avg: line-average electron density from the summary group (m^-3), on the summary grid itself.
- efit_*: EFIT minor radius, q95, li, beta_N, elongation and magnetic-axis height (config.EFIT_VARIABLES), each
  held from EFIT_DELAY_MS after its slice time, so an EFIT value is used only once it is surely known.

Every value at grid time t uses only fast samples with t_prev < time <= t, so it is known at t.
Cached per shot in data/extras/{shot_id}.parquet; a cached file from before a column was added is computed again.
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from src import config
from src.data import load_shot, open_group

COLUMNS = (["time", "lm_n1", "mirnov_rms", "ip_ref", "sol_current"] + list(config.MIRNOV_BANDS_KHZ)
           + ["ne_line_avg"] + list(config.EFIT_VARIABLES))
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


def band_rms(t_fast, x, t_grid, dt):
    """RMS of x in each band of config.MIRNOV_BANDS_KHZ within each grid bin, from the bin's own spectrum.

    Returns {band name: array over t_grid}; NaN where a bin has fewer than 16 finite samples or a gap.
    By Parseval, summing the squared spectrum over a band gives that band's share of the variance.
    """
    idx = bin_index(t_fast, t_grid)
    out = {name: np.full(len(t_grid), np.nan) for name in config.MIRNOV_BANDS_KHZ}
    bounds = np.searchsorted(idx, np.arange(len(t_grid) + 1), side="left")
    for i in range(len(t_grid)):
        block = x[bounds[i]:bounds[i + 1]]
        if len(block) < 16 or not np.isfinite(block).all():
            continue
        spectrum = np.abs(np.fft.rfft(block - block.mean())) ** 2
        freq = np.fft.rfftfreq(len(block), dt)
        # one-sided spectrum: interior bins stand for both signs of frequency
        scale = 2.0 / len(block) ** 2
        for name, (low, high) in config.MIRNOV_BANDS_KHZ.items():
            band = (freq >= low * 1e3) & (freq < high * 1e3)
            out[name][i] = np.sqrt(scale * spectrum[band].sum())
    return out


def mirnov_signals(shot_id, t_grid):
    """mirnov_rms and the band RMS columns on t_grid, from the first centre-column poloidal Mirnov coil that has
    data (all NaN if none)."""
    empty = {name: np.full(len(t_grid), np.nan) for name in ["mirnov_rms"] + list(config.MIRNOV_BANDS_KHZ)}
    mag = open_group(shot_id, "magnetics")
    name = "b_field_pol_probe_cc_field"
    if name not in mag:
        return empty
    t_fast = mag["time_mirnov"].values
    for k in range(mag[name].shape[0]):
        x = mag[name][k].values
        if np.isfinite(x).any():
            dt = float(np.median(np.diff(t_fast)))
            return {"mirnov_rms": bin_rms(t_fast, x, t_grid), **band_rms(t_fast, x, t_grid, dt)}
    return empty


def efit_signals(shot_id, t_grid):
    """The config.EFIT_VARIABLES columns on t_grid, each EFIT slice held from EFIT_DELAY_MS after its time
    (all NaN if the shot has no equilibrium group)."""
    out = {name: np.full(len(t_grid), np.nan) for name in config.EFIT_VARIABLES}
    try:
        efit = open_group(shot_id, "equilibrium")
    except Exception:
        return out
    known_at = efit["time"].values + config.EFIT_DELAY_MS / 1000
    for name, variable in config.EFIT_VARIABLES.items():
        if variable in efit and efit[variable].ndim == 1:
            out[name] = hold_last(known_at, np.asarray(efit[variable].values, dtype=float), t_grid)
    return out


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
    try:
        out["lm_n1"] = saddle_n1(shot_id, t_grid)
    except Exception:
        out["lm_n1"] = np.nan
    for func in (mirnov_signals, efit_signals):
        try:
            for column, values in func(shot_id, t_grid).items():
                out[column] = values
        except Exception:
            pass
    out["ip_ref"] = np.abs(slow_signal(shot_id, "pulse_schedule", "i_plasma", t_grid))
    out["sol_current"] = slow_signal(shot_id, "pf_active", "coil_current", t_grid, channel="SOL")
    out["ne_line_avg"] = slow_signal(shot_id, "summary", "line_average_n_e", t_grid)
    return out.reindex(columns=COLUMNS)


def cache_path(shot_id):
    """Where one shot's extra signals are cached: data/extras/{shot_id}.parquet."""
    return os.path.join(config.DATA_DIR, "extras", f"{shot_id}.parquet")


def is_current(path):
    """True if the cached extras file at path exists and has every column in COLUMNS."""
    if not os.path.exists(path):
        return False
    return set(COLUMNS) <= set(pq.read_schema(path).names)


def load_extras(shot_id, allow_download=True):
    """Extras for one shot from data/extras/, computed and cached on first use (or again if the cached file lacks
    a column). Offline (allow_download False): the cached file as it is, or None if there is none."""
    path = cache_path(shot_id)
    if is_current(path) or (os.path.exists(path) and not allow_download):
        return pd.read_parquet(path)
    if not allow_download:
        return None
    frame = compute_extras(shot_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frame.to_parquet(path)
    return frame


def download_extras(ids, workers=config.DOWNLOAD_WORKERS, deadline=None):
    """Compute and cache extras for every shot in ids (skipping current cached ones). Returns {'ok': n, 'failed': [ids]}.

    Each shot reads ~3-4 MB of fast magnetics and several groups from STFC, so this runs in parallel like
    download_shots. deadline (time.time() seconds): shots not started by then are left out and counted as failed,
    so a time-limited job (the GitHub relay) still hands over what it has.
    """
    todo = [i for i in ids if not is_current(cache_path(i))]
    failed = []
    pool = ThreadPoolExecutor(max_workers=workers)
    futures = {pool.submit(load_extras, i): i for i in todo}
    try:
        for future in tqdm(as_completed(futures), total=len(futures), desc="extras"):
            if future.cancelled():
                failed.append(futures[future])
                continue
            try:
                future.result()
            except Exception:
                failed.append(futures[future])
            if deadline is not None and time.time() > deadline:
                for pending in futures:
                    pending.cancel()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return {"ok": len(ids) - len(failed), "failed": failed}


def with_extras(shot_df, shot_id):
    """shot_df plus the cached extra columns (NaN columns if this shot has none cached)."""
    extra = load_extras(shot_id, allow_download=False)
    out = shot_df.copy()
    matches = extra is not None and len(extra) == len(out) and np.allclose(extra["time"], out["time"])
    for column in COLUMNS[1:]:
        out[column] = extra[column].to_numpy() if matches and column in extra else np.nan
    return out
