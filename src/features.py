"""Causal per-window features computed only from samples up to the window end."""
import numpy as np
import pandas as pd

from src import config

# feature-name prefix -> column in the cached shot table
SIGNALS = {
    "ip_abs": "ip",
    "prad": "power_radiated",
    "neutrons": "neutron_rates_total",
    "pnbi": "power_nbi",
}
STATS = ["mean", "std", "last", "slope", "delta"]
EPS = 1e-9  # s; time values are floats like 0.21000000000000002, so compare with a little slack


def prepare_signals(shot_df):
    """Return a DataFrame with time and the four feature signals (ip_abs, prad, neutrons, pnbi)."""
    out = pd.DataFrame({"time": shot_df["time"].to_numpy(dtype=float)})
    for name, column in SIGNALS.items():
        values = shot_df[column].to_numpy(dtype=float)
        # abs() because MAST's ip sign depends on field direction
        out[name] = np.abs(values) if name == "ip_abs" else values
    return out


def find_t_start(sig):
    """First time |Ip| >= IP_ON (plasma exists), or NaN if it never gets there.

    Also NaN if |Ip| is already above IP_ON at the first sample: no breakdown is seen, so there is no start
    to measure from (seen on a few long M8 shots such as 25424, see NOTES.md).
    """
    ip = sig["ip_abs"].to_numpy()
    above = np.where(ip >= config.IP_ON)[0]
    if len(above) == 0 or above[0] == 0:
        return np.nan
    return float(sig["time"].iloc[above[0]])


def find_t_plasma_off(sig, t_start):
    """First time the plasma has ended: |Ip| below IP_ON, from the first possible window end
    (t_start + WINDOW_MS) on, and after |Ip| has once reached MIN_PEAK_IP.

    Returns +inf if that never happens. Windows stop here: with no plasma there is nothing to predict.
    It is causal, because a window at t_end is dropped only if the drop already happened by t_end.
    Both conditions guard against breakdown: within a few ms of t_start |Ip| can spike past 100 kA and dip to
    ~10 kA for a sample (e.g. 16568), and some shots hover around IP_ON for tens of ms (e.g. 14195).
    """
    time = sig["time"].to_numpy()
    ip = np.nan_to_num(sig["ip_abs"].to_numpy(), nan=0.0)
    after = time >= t_start - EPS
    reached = np.maximum.accumulate(np.where(after, ip, 0.0)) >= config.MIN_PEAK_IP
    searchable = time >= t_start + config.WINDOW_MS / 1000 - EPS
    ended = np.where(searchable & reached & (ip < config.IP_ON))[0]
    return float(time[ended[0]]) if len(ended) else np.inf


def _stats(t, x):
    """mean, std, last, slope (units/s) and delta (last - first) of one signal inside one window."""
    ok = np.isfinite(x)
    out = dict.fromkeys(STATS, np.nan)
    if not ok.any():
        return out
    tv, xv = t[ok], x[ok]
    out["mean"] = float(np.mean(xv))
    out["std"] = float(np.std(xv))
    out["last"] = float(x[-1])  # NaN if the sample at t_end itself is missing
    if len(xv) >= 2:
        out["slope"] = float(np.polyfit(tv - tv[-1], xv, 1)[0])
        out["delta"] = float(xv[-1] - xv[0])
    return out


def window_features(sig, t_end, t_start):
    """Features for the window ending at t_end, from samples with t_end - WINDOW_MS < time <= t_end only.

    sig comes from prepare_signals(). Returns a dict of feature name -> float (NaN where not computable).
    """
    time = sig["time"].to_numpy()
    past = time <= t_end + EPS
    in_window = past & (time > t_end - config.WINDOW_MS / 1000 + EPS)
    t = time[in_window]

    feats = {}
    for name in SIGNALS:
        x = sig[name].to_numpy()
        feats[f"has_{name}"] = float(np.isfinite(x[past]).any())
        for stat, value in _stats(t, x[in_window]).items():
            feats[f"{name}_{stat}"] = value

    ip_so_far = sig["ip_abs"].to_numpy()[past]
    ip_so_far = ip_so_far[np.isfinite(ip_so_far)]
    running_max = ip_so_far.max() if len(ip_so_far) else np.nan
    feats["ip_frac_of_running_max"] = feats["ip_abs_last"] / running_max if running_max > 0 else np.nan
    # offsets keep the ratios finite when beams are off or current is near zero (SPEC 7)
    feats["prad_over_pnbi"] = feats["prad_mean"] / (feats["pnbi_mean"] + config.PNBI_OFFSET)
    feats["prad_over_ip"] = feats["prad_mean"] / (feats["ip_abs_mean"] + config.IP_OFFSET)
    feats["neutron_slope_rel"] = feats["neutrons_slope"] / (feats["neutrons_mean"] + config.NEUTRON_OFFSET)
    feats["t_since_start"] = float(t_end - t_start)
    last_pnbi = feats["pnbi_last"]
    feats["pnbi_on"] = float(last_pnbi > config.PNBI_ON_THRESHOLD) if np.isfinite(last_pnbi) else np.nan
    return feats


def window_ends(sig, t_start, t_stop=np.inf):
    """Sample times to end windows at: from t_start + WINDOW_MS, every STRIDE_MS, while t_end < t_stop
    and before the plasma switches off."""
    time = sig["time"].to_numpy()
    if not np.isfinite(t_start):
        return np.array([])
    last = min(time[-1], t_stop - EPS, find_t_plasma_off(sig, t_start) - EPS)
    first = t_start + config.WINDOW_MS / 1000
    targets = np.arange(first, last + EPS, config.STRIDE_MS / 1000)
    # snap each target to the latest real sample at or before it
    idx = np.searchsorted(time, targets + EPS, side="right") - 1
    idx = np.unique(idx[idx >= 0])
    ends = time[idx]
    return ends[(ends >= first - EPS) & (ends < t_stop - EPS)]


def shot_windows(shot_df, t_stop=np.inf):
    """All windows of one shot as a DataFrame (one row per t_end, plus t_end and t_start columns).

    t_stop: drop windows with t_end >= t_stop. For a disrupted shot pass t_disrupt (SPEC 6).
    """
    sig = prepare_signals(shot_df)
    t_start = find_t_start(sig)
    rows = []
    for t_end in window_ends(sig, t_start, t_stop):
        row = {"t_end": float(t_end), "t_start": t_start}
        row.update(window_features(sig, t_end, t_start))
        rows.append(row)
    return pd.DataFrame(rows)


def feature_names():
    """Names of the model input columns produced by window_features, in a fixed order."""
    names = []
    for name in SIGNALS:
        names += [f"{name}_{stat}" for stat in STATS]
    names += ["ip_frac_of_running_max", "prad_over_pnbi", "prad_over_ip", "neutron_slope_rel",
              "t_since_start", "pnbi_on"]
    names += [f"has_{name}" for name in SIGNALS]
    return names
