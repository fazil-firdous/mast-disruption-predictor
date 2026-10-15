"""Run a saved model over a whole shot as if it were live, and plot the result."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import alarms, features, models


def predict(shot_df, model, meta):
    """Score every window of a shot and apply the alarm rule. Returns (windows with a score column, t_alarm or NaN).

    No stop time is passed: live, nobody knows how or when the shot will end.
    """
    win = features.shot_windows(shot_df)
    if len(win) == 0:
        return win, np.nan
    win["score"] = models.scores(model, win[meta["columns"]])
    t_alarm = alarms.first_alarm(win["score"].to_numpy(), win["t_end"].to_numpy(), meta["threshold"], meta["k_consec"])
    return win, t_alarm


def plot_prediction(shot_df, win, t_alarm, threshold, t_quench, title, path):
    """|Ip| on top, score and threshold below; blue dotted = alarm, red = detector quench (known only afterwards)."""
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    top.plot(shot_df["time"], np.abs(shot_df["ip"]) / 1e3, color="0.3", lw=1)
    top.set_ylabel("|Ip| (kA)")
    if len(win):
        bottom.plot(win["t_end"], win["score"], ".-", ms=3, lw=1)
    bottom.axhline(threshold, color="0.5", ls="--", lw=1)
    bottom.set_ylim(-0.05, 1.05)
    bottom.set_ylabel("score")
    bottom.set_xlabel("time (s)")
    for ax in (top, bottom):
        if np.isfinite(t_alarm):
            ax.axvline(t_alarm, color="tab:blue", ls=":", lw=1.5)
        if np.isfinite(t_quench):
            ax.axvline(t_quench, color="red", lw=1.2)
    top.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)
