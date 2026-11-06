"""Statistics over repeat runs (E10 and the repeat-run comparisons). Computed fresh from saved results every time and
never cached, so this file is outside every stamp (resume.NOT_SCIENCE): changing it can make no saved number stale.
"""
import numpy as np
from scipy import stats

from src import config

MIN_SEEDS = 5  # no range or verdict before this many splits: with 2-3, a few shots that happen to agree look certain


def corrected_interval(changes, level=0.95):
    """(mean, low, high) of paired changes over repeated random splits. The variance is scaled by
    (1/n + n_test/n_train) instead of 1/n (Nadeau & Bengio, Machine Learning 52, 2003), because the splits share
    shots and a plain t-test would be far too sure of itself. No range below MIN_SEEDS splits."""
    n = len(changes)
    mean = float(np.mean(changes)) if n else None
    if n < MIN_SEEDS:
        return mean, None, None
    train, _, test = config.SPLIT_FRACTIONS
    half = stats.t.ppf(0.5 + level / 2, n - 1) * np.std(changes, ddof=1) * np.sqrt(1 / n + test / train)
    return mean, float(mean - half), float(mean + half)


def corrected_p(changes):
    """Two-sided p-value of the corrected resampled t-test that corrected_interval inverts (mean change 0), or None
    below MIN_SEEDS splits. A change with zero spread gives 0 (if non-zero) or 1."""
    changes = np.asarray(changes, dtype=float)
    n = len(changes)
    if n < MIN_SEEDS:
        return None
    train, _, test = config.SPLIT_FRACTIONS
    se = np.std(changes, ddof=1) * np.sqrt(1 / n + test / train)
    if se == 0:
        return 0.0 if np.mean(changes) != 0 else 1.0
    return float(2 * stats.t.sf(abs(np.mean(changes)) / se, n - 1))


def p_from_interval(mean, low, high, n, level=0.95):
    """The corrected test's two-sided p-value worked back from a saved mean change and its corrected range over n
    splits (as in comparisons/summary.csv), or None if the range is missing or zero wide."""
    if low is None or high is None or not (np.isfinite(low) and np.isfinite(high)) or n < MIN_SEEDS or high <= low:
        return None
    se = (high - low) / 2 / stats.t.ppf(0.5 + level / 2, n - 1)
    return float(2 * stats.t.sf(abs(mean) / se, n - 1))


def holm(p_values):
    """Holm-Bonferroni adjusted p-values (same order as given; None stays None): a family of tests is judged together,
    so a 'clear' verdict among many is not luck. Holds whatever the dependence between the tests."""
    known = [(p, i) for i, p in enumerate(p_values) if p is not None]
    out = [None] * len(p_values)
    running = 0.0
    for rank, (p, i) in enumerate(sorted(known)):
        running = max(running, min(1.0, p * (len(known) - rank)))
        out[i] = running
    return out
