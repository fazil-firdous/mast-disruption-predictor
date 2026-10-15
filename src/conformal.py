"""Conformal alarm threshold: a provable false-alarm limit on new clean shots (ROADMAP B1).

The guarantee holds if the new clean shots are exchangeable with the calibration shots (drawn the same way).
Campaign drift breaks that, which is exactly what the temporal split tests.
"""
import math

import numpy as np
import pandas as pd

from src import alarms, config


def shot_alarm_levels(windows, window_scores, k=config.K_CONSEC):
    """One row per shot: shot_id, disrupted, and level = the highest threshold at which the shot would alarm.

    A shot alarms at threshold thr exactly when level >= thr (level is the best 'smallest of the last k scores').
    Shots with fewer than k windows can never alarm and get level -inf.
    """
    df = windows[["shot_id", "t_end", "disrupted"]].assign(score=window_scores).sort_values(["shot_id", "t_end"])
    rows = []
    for shot_id, group in df.groupby("shot_id", sort=True):
        level = alarms.alarm_level(group["score"].to_numpy(), k)
        best = np.nanmax(level) if np.isfinite(level).any() else -np.inf
        rows.append({"shot_id": shot_id, "disrupted": bool(group["disrupted"].iloc[0]), "level": float(best)})
    return pd.DataFrame(rows)


def conformal_threshold(clean_levels, alpha):
    """Smallest threshold with P(a new clean shot alarms) <= alpha, from n calibration clean shots.

    Sort the n levels; take the r-th smallest with r = ceil((n + 1)(1 - alpha)), and set the threshold just
    above it. A new exchangeable clean shot exceeds the r-th of n+1 values with probability at most
    (n + 1 - r) / (n + 1) <= alpha. If r > n there are too few calibration shots and no threshold can
    promise alpha, so +inf (never alarm) is returned.
    """
    levels = np.sort(np.asarray(clean_levels, dtype=float))
    n = len(levels)
    rank = math.ceil((n + 1) * (1 - alpha))
    if n == 0 or rank > n:
        return np.inf
    return float(np.nextafter(levels[rank - 1], np.inf))


def min_calibration_shots(alpha):
    """Fewest clean calibration shots for which a finite conformal threshold exists at this alpha."""
    return math.ceil((1 - alpha) / alpha)
