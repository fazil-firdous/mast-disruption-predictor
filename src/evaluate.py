"""Shot-level metrics, trade-off curves, bootstrap confidence intervals, and plots."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from src import alarms, config

THRESHOLDS = np.linspace(0, 1, config.N_THRESHOLDS)


def shot_alarm_table(windows, window_scores, thresholds=THRESHOLDS):
    """One row per shot: shot_id, disrupted, t_disrupt, and a (shots x thresholds) array of first-alarm times.

    windows must hold shot_id, t_end, disrupted and t_disrupt (see dataset.labelled_windows).
    """
    df = windows[["shot_id", "t_end", "disrupted", "t_disrupt"]].copy()
    df["score"] = window_scores
    df = df.sort_values(["shot_id", "t_end"])
    shots, times = [], []
    for shot_id, group in df.groupby("shot_id", sort=True):
        shots.append({"shot_id": shot_id, "disrupted": bool(group["disrupted"].iloc[0]),
                      "t_disrupt": float(group["t_disrupt"].iloc[0])})
        times.append(alarms.alarm_times(group["score"].to_numpy(), group["t_end"].to_numpy(), thresholds))
    return pd.DataFrame(shots), np.array(times)


def outcomes(shots, t_alarm):
    """Outcome of each shot for one threshold: 'caught', 'tardy', 'missed', 'false_alarm' or 'quiet'.

    Also returns warning times in ms (NaN unless the disrupted shot raised an alarm).
    """
    disrupted = shots["disrupted"].to_numpy()
    alarmed = np.isfinite(t_alarm)
    warning_ms = np.where(disrupted & alarmed, 1000 * (shots["t_disrupt"].to_numpy() - t_alarm), np.nan)
    result = np.full(len(shots), "quiet", dtype=object)
    result[~disrupted & alarmed] = "false_alarm"
    result[disrupted & ~alarmed] = "missed"
    result[disrupted & alarmed & (warning_ms >= config.MIN_WARNING_MS)] = "caught"
    result[disrupted & alarmed & (warning_ms < config.MIN_WARNING_MS)] = "tardy"
    return result, warning_ms


def shot_metrics(result, warning_ms):
    """Catch, tardy, miss and false-alarm rates plus warning-time median and quartiles over caught shots."""
    n_disrupted = int(np.isin(result, ["caught", "tardy", "missed"]).sum())
    n_clean = int(np.isin(result, ["false_alarm", "quiet"]).sum())
    caught_warning = warning_ms[result == "caught"]
    has_caught = len(caught_warning) > 0
    # premature catches still count as caught (SPEC 4), but are reported so very early alarms are not hidden
    premature = int((caught_warning > config.PREMATURE_WARNING_MS).sum())
    return {
        "n_disrupted": n_disrupted, "n_clean": n_clean,
        "catch_rate": float((result == "caught").sum() / max(n_disrupted, 1)),
        "tardy_rate": float((result == "tardy").sum() / max(n_disrupted, 1)),
        "miss_rate": float((result == "missed").sum() / max(n_disrupted, 1)),
        "false_alarm_rate": float((result == "false_alarm").sum() / max(n_clean, 1)),
        "premature_rate": float(premature / max(n_disrupted, 1)),
        "warning_median_ms": float(np.median(caught_warning)) if has_caught else np.nan,
        "warning_q25_ms": float(np.percentile(caught_warning, 25)) if has_caught else np.nan,
        "warning_q75_ms": float(np.percentile(caught_warning, 75)) if has_caught else np.nan,
    }


def tradeoff_curve(shots, alarm_matrix, thresholds=THRESHOLDS):
    """Catch rate and false-alarm rate (and the rest of shot_metrics) at every threshold."""
    rows = []
    for j, threshold in enumerate(thresholds):
        result, warning_ms = outcomes(shots, alarm_matrix[:, j])
        rows.append({"threshold": float(threshold), **shot_metrics(result, warning_ms)})
    return pd.DataFrame(rows)


def choose_threshold(curve, max_false_alarm=config.MAX_FALSE_ALARM_RATE):
    """Lowest threshold whose false-alarm rate is within the limit (lower thresholds catch more). SPEC 10.

    If no threshold meets the limit, returns the one with the fewest false alarms; check meets_limit() after.
    """
    ok = curve[curve["false_alarm_rate"] <= max_false_alarm]
    if len(ok) == 0:
        fewest = curve[curve["false_alarm_rate"] == curve["false_alarm_rate"].min()]
        return float(fewest["threshold"].min())
    return float(ok["threshold"].min())


def meets_limit(false_alarm_rate, max_false_alarm=config.MAX_FALSE_ALARM_RATE):
    """True if a false-alarm rate is within the SPEC 10 limit."""
    return bool(false_alarm_rate <= max_false_alarm + 1e-12)


def bootstrap_ci(result, warning_ms, n=config.N_BOOTSTRAP, seed=config.RANDOM_SEED):
    """95% intervals for shot_metrics by resampling shots with replacement.

    Disrupted and clean shots are resampled separately, so every resample keeps the same class sizes.
    """
    rng = np.random.default_rng(seed)
    disrupted_idx = np.where(np.isin(result, ["caught", "tardy", "missed"]))[0]
    clean_idx = np.where(np.isin(result, ["false_alarm", "quiet"]))[0]
    samples = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(disrupted_idx, len(disrupted_idx)), rng.choice(clean_idx, len(clean_idx))])
        samples.append(shot_metrics(result[idx], warning_ms[idx]))
    samples = pd.DataFrame(samples)
    keys = ["catch_rate", "tardy_rate", "miss_rate", "false_alarm_rate", "premature_rate", "warning_median_ms"]
    return {k: [float(samples[k].quantile(0.025)), float(samples[k].quantile(0.975))] for k in keys}


def window_metrics(y, window_scores, threshold):
    """Window-level precision, recall, F1 at the threshold, and PR-AUC (secondary metrics, SPEC 11)."""
    predicted = (np.asarray(window_scores) >= threshold).astype(int)
    return {
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "pr_auc": float(average_precision_score(y, window_scores)) if np.any(y) else np.nan,
    }


def plot_tradeoff(curves, path, operating_points=None):
    """Catch rate against false-alarm rate. curves maps a name to a tradeoff_curve DataFrame."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, curve in curves.items():
        ax.plot(curve["false_alarm_rate"], curve["catch_rate"], lw=1.5, label=name)
    for name, (fa, catch) in (operating_points or {}).items():
        ax.plot(fa, catch, "o", color="black", ms=5)
        ax.annotate(name, (fa, catch), textcoords="offset points", xytext=(5, -10), fontsize=8)
    ax.axvline(config.MAX_FALSE_ALARM_RATE, color="0.6", ls="--", lw=1)
    ax.set_xlabel("false-alarm rate (clean shots with an alarm)")
    ax.set_ylabel(f"catch rate (warning >= {config.MIN_WARNING_MS} ms)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def plot_warning_hist(warning_ms, result, path):
    """Histogram of warning times over caught and tardy shots."""
    values = warning_ms[np.isin(result, ["caught", "tardy"])]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.hist(values, bins=np.arange(0, max(60, np.nanmax(values, initial=0) + 5), 5), color="0.4")
    ax.axvline(config.MIN_WARNING_MS, color="red", lw=1, label=f"minimum useful ({config.MIN_WARNING_MS} ms)")
    ax.set_xlabel("warning time (ms)")
    ax.set_ylabel("shots")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def plot_importance(importance, path, top=15):
    """Horizontal bar chart of the top permutation importances (mean drop in PR-AUC)."""
    top_rows = importance.sort_values("mean").tail(top)
    fig, ax = plt.subplots(figsize=(6, 0.3 * len(top_rows) + 1))
    ax.barh(top_rows["feature"], top_rows["mean"], xerr=top_rows["std"], color="0.4")
    ax.set_xlabel("drop in validation PR-AUC when the feature is shuffled")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def plot_example(shot_df, win, threshold, t_alarm, t_disrupt, title, ax_pair):
    """Two stacked axes for one shot: |Ip| and Prad on top, model score with threshold below."""
    top, bottom = ax_pair
    top.plot(shot_df["time"], np.abs(shot_df["ip"]) / 1e3, color="0.3", lw=1, label="|Ip| (kA)")
    if shot_df["power_radiated"].notna().any():
        top.plot(shot_df["time"], shot_df["power_radiated"] / 1e3, color="tab:orange", lw=1, label="Prad (kW)")
    bottom.plot(win["t_end"], win["score"], ".-", ms=3, lw=1)
    bottom.axhline(threshold, color="0.5", ls="--", lw=1)
    bottom.set_ylim(-0.05, 1.05)
    for ax in ax_pair:
        if np.isfinite(t_disrupt):
            ax.axvline(t_disrupt, color="red", lw=1.2)
        if np.isfinite(t_alarm):
            ax.axvline(t_alarm, color="tab:blue", lw=1.2, ls=":")
        ax.tick_params(labelsize=7)
    top.set_title(title, fontsize=8)
    top.legend(fontsize=6, loc="upper left")
    bottom.set_ylabel("score", fontsize=7)
    bottom.set_xlim(top.get_xlim())
