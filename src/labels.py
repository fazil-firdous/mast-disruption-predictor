"""Disruption labels: operator-note parsing and the signal-based current-quench detector."""
import re

import numpy as np

from src import config

NEGATION = re.compile(r"no disrupt|not disrupt|non-disrupt|didn.?t disrupt", re.IGNORECASE)
# "AT 220MS", "at ~310 ms", "AT 0.22S", and unit-less "at 0.302" or "at 190"
# the lookaheads stop "at 0.169delay" being read as "at 0" by backtracking
NOTE_TIME = re.compile(r"\bat\s*~?\s*(\d+(?:\.\d+)?)(?![\d.])\s*(ms|s)?(?![a-z])", re.IGNORECASE)


def note_label(comment):
    """True if the operator comment mentions a disruption and does not negate it. NaN or None gives False."""
    if not isinstance(comment, str):
        return False
    text = comment.lower()
    return "disrupt" in text and NEGATION.search(text) is None


def parse_note_time(comment):
    """Disruption time in seconds taken from a comment like 'DISRUPTION AT 220MS', or NaN if none is given.

    Only a time written after the word 'disrupt' is used, so unrelated times earlier in the comment are skipped.
    """
    if not isinstance(comment, str):
        return float("nan")
    start = comment.lower().find("disrupt")
    if start < 0:
        return float("nan")
    match = NOTE_TIME.search(comment, start)
    if match is None:
        return float("nan")
    value = float(match.group(1))
    if value <= 0:
        return float("nan")
    unit = (match.group(2) or "").lower()
    if unit == "ms":
        return value / 1000
    if unit == "s":
        return value
    # no unit: MAST shots last under ~1 s, so a small number is seconds and a large one is ms
    if value < 1.5:
        return value
    if value >= 10:
        return value / 1000
    return float("nan")


def median_filter(x, size=config.MEDIAN_FILTER_SAMPLES):
    """Running median with edge values repeated, so the output has the same length as x."""
    half = size // 2
    padded = np.concatenate([np.full(half, x[0]), x, np.full(half, x[-1])])
    shifted = [padded[i:i + len(x)] for i in range(size)]
    return np.median(np.stack(shifted), axis=0)


def _result(status, t_disrupt=np.nan, quench_rate=np.nan, peak_ip=np.nan):
    return {"status": status, "detected": status == "disrupted", "t_disrupt": t_disrupt,
            "quench_rate": quench_rate, "peak_ip": peak_ip}


def _first_quench_candidate(t, x, floor, t_start):
    """Index of the earliest sample from which |Ip| falls by CQ_DROP_FRAC within CQ_MAX_MS, or None."""
    for i in range(len(x)):
        if t[i] < t_start or x[i] < floor:
            continue
        end = np.searchsorted(t, t[i] + config.CQ_MAX_MS / 1000, side="right")
        if np.any(x[i + 1:end] <= (1 - config.CQ_DROP_FRAC) * x[i]):
            return i
    return None


def detect_disruption(time, ip):
    """Find the disruption time of one shot from its plasma current.

    Returns a dict with status ('disrupted', 'clean' or 'unknown'), detected (bool), t_disrupt (s),
    quench_rate (A/s) and peak_ip (A). 'unknown' means |Ip| never reached MIN_PEAK_IP.
    """
    time = np.asarray(time, dtype=float)
    ip = np.asarray(ip, dtype=float)
    keep = np.isfinite(time) & np.isfinite(ip)
    t = time[keep]
    if len(t) < 5:
        return _result("unknown")
    # abs() because MAST's ip sign depends on field direction
    x = median_filter(np.abs(ip[keep]))
    peak = float(x.max())
    if peak < config.MIN_PEAK_IP:
        return _result("unknown", peak_ip=peak)

    above = np.where(x >= config.IP_ON)[0]
    t_start = t[above[0]]
    floor = max(config.IP_ON, config.CQ_MIN_FRAC_OF_PEAK * peak)
    first = _first_quench_candidate(t, x, floor, t_start)
    if first is None:
        return _result("clean", peak_ip=peak)

    # end of the fall: first sample at or below 20% of the starting level
    low = (1 - config.CQ_DROP_FRAC) * x[first]
    end = first + 1 + int(np.argmax(x[first + 1:] <= low))
    # the search above can fire early, so move to the last sample still near the top of the fall
    level = x[first:end + 1].max()
    onset = first + int(np.max(np.where(x[first:end + 1] >= config.CQ_ONSET_FRAC * level)[0]))
    t_disrupt = t[onset]

    # an Ip spike just before the fall marks the thermal quench, so use its time instead
    spike_window = (t >= t[onset] - config.SPIKE_LOOKBACK_MS / 1000) & (t <= t[onset])
    before = (t >= t[onset] - (config.SPIKE_LOOKBACK_MS + 10) / 1000) & (t < t[onset] - config.SPIKE_LOOKBACK_MS / 1000)
    if before.any():
        spike_idx = np.where(spike_window)[0][np.argmax(x[spike_window])]
        if x[spike_idx] >= (1 + config.SPIKE_MIN_RISE) * np.median(x[before]):
            t_disrupt = t[spike_idx]

    rate = (x[onset] - x[end]) / (t[end] - t[onset]) if t[end] > t[onset] else np.nan
    return _result("disrupted", float(t_disrupt), float(rate), peak)


def final_label(note_flag, signal_flag, source=None):
    """Combine the two labels according to config.LABEL_SOURCE ('signal', 'note' or 'both').

    'both' means the shot counts as disrupted only when the note and the detector agree.
    """
    source = source or config.LABEL_SOURCE
    if source == "signal":
        return bool(signal_flag)
    if source == "note":
        return bool(note_flag)
    if source == "both":
        return bool(note_flag and signal_flag)
    raise ValueError(f"LABEL_SOURCE must be 'signal', 'note' or 'both', got {source!r}")


def shot_label(time, ip, comment, source=None):
    """Final label for one shot under config.LABEL_SOURCE.

    Returns a dict with status ('disrupted', 'clean' or 'unknown'), disrupted (bool), t_disrupt (s or NaN)
    and the two inputs note (bool) and signal_status. t_disrupt comes from the detector; with source='note'
    it falls back to the note time when the detector found nothing. 'unknown' shots should be skipped.
    """
    source = source or config.LABEL_SOURCE
    found = detect_disruption(time, ip)
    note = note_label(comment)
    result = {"note": note, "signal_status": found["status"], "status": "unknown",
              "disrupted": False, "t_disrupt": np.nan}
    if found["status"] == "unknown":
        return result
    disrupted = final_label(note, found["detected"], source)
    t_disrupt = found["t_disrupt"]
    if disrupted and np.isnan(t_disrupt):
        t_disrupt = parse_note_time(comment)
    if disrupted and np.isnan(t_disrupt):
        # a disruption with no usable time cannot be windowed, so leave it out rather than guess
        return result
    result.update(status="disrupted" if disrupted else "clean", disrupted=disrupted,
                  t_disrupt=t_disrupt if disrupted else np.nan)
    return result
