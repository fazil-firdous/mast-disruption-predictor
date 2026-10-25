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
