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
# extra physics signals from src/extras.py (ROADMAP D1); NaN when a shot has no extras
EXTRA_SIGNALS = {
    "lm": "lm_n1",
    "mirnov": "mirnov_rms",
    "sol": "sol_current",
}
# v2 signals (NOTES 2026-10-30): Mirnov band RMS, density and the Greenwald fraction, EFIT, and the programmed current.
# fgw is not stored anywhere: prepare_signals builds it from density, current and the EFIT minor radius
V2_SIGNALS = {
    "mir_lf": "mirnov_lf",
    "mir_tm": "mirnov_tm",
    "mir_mf": "mirnov_mf",
    "mir_hf": "mirnov_hf",
    "ne": "ne_line_avg",
    "fgw": None,
    "q95": "efit_q95",
    "li": "efit_li",
    "betan": "efit_betan",
    "kappa": "efit_kappa",
    "zmag": "efit_zmag",
    "ref": "ip_ref",
}
# signal groups, so a comparison can leave one out (config.COMPARISONS 'drop_group')
GROUPS = {"density": ["ne", "fgw"], "efit": ["q95", "li", "betan", "kappa", "zmag"],
          "mirnov_bands": ["mir_lf", "mir_tm", "mir_mf", "mir_hf"]}
# fluctuation amplitudes: their level differs from shot to shot (coil pickup, calibration), so the change against the
# shot's own earlier values says more than the level (NOTES 2026-10-16: a fixed pickup correction failed)
REL_SIGNALS = ["lm", "mirnov", "mir_lf", "mir_tm", "mir_mf", "mir_hf"]
# slow trends: a locked mode growing, density creeping towards the limit, the current profile or the plasma moving
LONG_SIGNALS = ["lm", "mirnov", "fgw", "li", "q95", "zmag"]
STATS = ["mean", "std", "last", "slope", "delta"]
EPS = 1e-9  # s; time values are floats like 0.21000000000000002, so compare with a little slack


def prepare_signals(shot_df):
    """Return a DataFrame with time, the four summary signals (ip_abs, prad, neutrons, pnbi), the extra signals
    (lm, mirnov, sol) and ip_ref. Extra columns the shot does not have are NaN."""
    out = pd.DataFrame({"time": shot_df["time"].to_numpy(dtype=float)})
    for name, column in SIGNALS.items():
        values = shot_df[column].to_numpy(dtype=float)
        # abs() because MAST's ip sign depends on field direction
        out[name] = np.abs(values) if name == "ip_abs" else values
    for name, column in list(EXTRA_SIGNALS.items()) + list(V2_SIGNALS.items()) + [("ip_ref", "ip_ref")]:
        if column is not None:
            out[name] = shot_df[column].to_numpy(dtype=float) if column in shot_df else np.nan
    out["fgw"] = greenwald_fraction(out["ne"].to_numpy(), out["ip_abs"].to_numpy(),
                                    shot_df["efit_a"].to_numpy(dtype=float) if "efit_a" in shot_df else np.nan)
    return out


def greenwald_fraction(ne, ip_abs, minor_radius):
    """Line-average density over the Greenwald limit n_G = Ip[MA] / (pi a^2) [1e20 m^-3], sample by sample.

    NaN without plasma current (below IP_ON) or a minor radius. All three inputs are known at each sample time,
    so this is causal; the summary group's own greenwald_density may be interpolated between EFIT slices.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        limit = np.where(ip_abs >= config.IP_ON, ip_abs / 1e6 / (np.pi * minor_radius ** 2) * 1e20, np.nan)
        return ne / limit


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
        out["slope"] = slope(tv, xv)
        out["delta"] = float(xv[-1] - xv[0])
    return out


def slope(t, x):
    """Least-squares slope of x against t (units/s), the same as np.polyfit(t, x, 1)[0] but much faster."""
    dt = t - t.mean()
    denominator = float(np.dot(dt, dt))
    return float(np.dot(dt, x - x.mean()) / denominator) if denominator > 0 else np.nan


def relative_change(x_window_last, history):
    """log10 of the window's last value over the median of the shot's own earlier values (NaN if either is
    missing or not positive, or the history has fewer than REL_MIN_SAMPLES samples)."""
    history = history[np.isfinite(history)]
    if len(history) < config.REL_MIN_SAMPLES or not np.isfinite(x_window_last):
        return np.nan
    baseline = float(np.median(history))
    if baseline <= 0 or x_window_last <= 0:
        return np.nan
    return float(np.log10(x_window_last / baseline))


def window_features(sig, t_end, t_start):
    """Features for the window ending at t_end, from samples with t_end - WINDOW_MS < time <= t_end only.

    sig comes from prepare_signals(). Returns a dict of feature name -> float (NaN where not computable).
    """
    time = sig["time"].to_numpy()
    past = time <= t_end + EPS
    in_window = past & (time > t_end - config.WINDOW_MS / 1000 + EPS)
    t = time[in_window]

    feats = {}
    for name in list(SIGNALS) + list(EXTRA_SIGNALS) + list(V2_SIGNALS):
        x = sig[name].to_numpy()
        feats[f"has_{name}"] = float(np.isfinite(x[past]).any())
        for stat, value in _stats(t, x[in_window]).items():
            feats[f"{name}_{stat}"] = value
    # the shot's own history before this window, and the last LONG_WINDOW_MS, for the change features
    history = past & ~in_window & (time >= t_start - EPS)
    long_window = past & (time > t_end - config.LONG_WINDOW_MS / 1000 + EPS)
    for name in REL_SIGNALS:
        x = sig[name].to_numpy()
        feats[f"{name}_rel"] = relative_change(feats[f"{name}_mean"], x[history])
    for name in LONG_SIGNALS:
        x, tl = sig[name].to_numpy()[long_window], time[long_window]
        ok = np.isfinite(x)
        feats[f"{name}_slope_long"] = slope(tl[ok], x[ok]) if ok.sum() >= config.REL_MIN_SAMPLES else np.nan

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
    # how far the current lags the programmed current: losing control of Ip is a classic disruption precursor
    ip_ref_last = sig["ip_ref"].to_numpy()[past][-1] if past.any() else np.nan
    feats["ip_ref_error"] = (ip_ref_last - feats["ip_abs_last"]) / (ip_ref_last + config.IP_OFFSET)
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


def feature_names(extended=False):
    """Names of the model input columns, in a fixed order: the 30 v1 features, plus the extra-signal features
    (ROADMAP D1) when extended is True."""
    names = []
    for name in SIGNALS:
        names += [f"{name}_{stat}" for stat in STATS]
    names += ["ip_frac_of_running_max", "prad_over_pnbi", "prad_over_ip", "neutron_slope_rel",
              "t_since_start", "pnbi_on"]
    names += [f"has_{name}" for name in SIGNALS]
    if extended:
        for name in EXTRA_SIGNALS:
            names += [f"{name}_{stat}" for stat in STATS]
        names += ["ip_ref_error"] + [f"has_{name}" for name in EXTRA_SIGNALS]
    return names


def v2_names():
    """v1 + magnetics, plus every v2 signal's window statistics, the change and slow-trend features."""
    names = feature_names(extended=True)
    for name in V2_SIGNALS:
        names += [f"{name}_{stat}" for stat in STATS] + [f"has_{name}"]
    names += [f"{name}_rel" for name in REL_SIGNALS] + [f"{name}_slope_long" for name in LONG_SIGNALS]
    return names


def machine_names():
    """A predictor that never looks at the plasma: the solenoid current, the programmed current and the time since
    start (E14). If it predicts the 'disruptions' of some labels well, those labels mostly mark the machine's
    programme or its flux running out, not plasma instabilities."""
    return [f"{name}_{stat}" for name in ("sol", "ref") for stat in STATS] + ["has_sol", "has_ref", "t_since_start"]


FEATURE_SETS = {"v1": feature_names, "v1+magnetics": lambda: feature_names(extended=True), "v2": v2_names,
                "machine": machine_names}


def model_features(feature_set=None):
    """The input columns of a feature set (default config.FEATURE_SET)."""
    return FEATURE_SETS[feature_set or config.FEATURE_SET]()


def uses_signal(feature, signal):
    """True if a feature is computed from the given signal (its own statistics, flag, change features, or a
    cross-signal feature built from it)."""
    if feature.startswith(f"{signal}_") or feature == f"has_{signal}":
        return True
    if signal in DERIVED_INPUTS.get(feature, []):
        return True
    # fgw is built from ne and ip_abs, and the current-control error from ip_ref and ip_abs
    return (signal in ("ne", "ip_abs") and feature.startswith("fgw_")) or \
        (signal in ("ref", "ip_abs") and feature == "ip_ref_error")


# signals each cross-signal feature is built from, for the signal ablation (E5)
DERIVED_INPUTS = {
    "ip_frac_of_running_max": ["ip_abs"],
    "prad_over_pnbi": ["prad", "pnbi"],
    "prad_over_ip": ["prad", "ip_abs"],
    "neutron_slope_rel": ["neutrons"],
    "t_since_start": [],
    "pnbi_on": ["pnbi"],
}


def features_without(signal, columns=None):
    """The model's features (default model_features()) minus every feature that uses the given signal (e.g. 'prad')."""
    return [name for name in (columns or model_features()) if not uses_signal(name, signal)]


def features_without_group(group, columns=None):
    """The model's features minus every feature of one signal group of GROUPS (e.g. 'efit')."""
    kept = columns or model_features()
    for signal in GROUPS[group]:
        kept = features_without(signal, kept)
    return kept
