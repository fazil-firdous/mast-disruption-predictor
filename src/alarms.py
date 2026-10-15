"""Turn a time-ordered score sequence into a first-alarm time using the K-consecutive rule."""
import numpy as np

from src import config


def first_alarm(scores, times, threshold, k=config.K_CONSEC):
    """Time of the first window where the score has been >= threshold for k windows in a row, or NaN.

    scores and times must be in time order for one shot. The alarm time is the end of the k-th window.
    """
    in_a_row = 0
    for score, t in zip(scores, times):
        in_a_row = in_a_row + 1 if score >= threshold else 0
        if in_a_row >= k:
            return float(t)
    return np.nan


def alarm_level(scores, k=config.K_CONSEC):
    """For each window, the smallest of the last k scores (NaN for the first k-1 windows).

    The k-in-a-row rule fires at a window exactly when this value is >= the threshold, which lets
    alarm_times() handle many thresholds at once instead of looping over each one.
    """
    scores = np.asarray(scores, dtype=float)
    level = np.full(len(scores), np.nan)
    for i in range(k - 1, len(scores)):
        level[i] = scores[i - k + 1:i + 1].min()
    return level


def alarm_times(scores, times, thresholds, k=config.K_CONSEC):
    """First-alarm time of one shot for every threshold in thresholds (NaN where no alarm). Same rule as first_alarm.

    best_so_far never decreases, so the first window where it reaches a threshold is found with searchsorted.
    """
    times = np.asarray(times, dtype=float)
    level = np.nan_to_num(alarm_level(scores, k), nan=-np.inf)
    if len(level) == 0:
        return np.full(len(thresholds), np.nan)
    best_so_far = np.maximum.accumulate(level)
    idx = np.searchsorted(best_so_far, thresholds, side="left")
    out = np.full(len(thresholds), np.nan)
    fired = idx < len(times)
    out[fired] = times[idx[fired]]
    return out
