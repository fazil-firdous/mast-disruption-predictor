"""Tests for scripts/timing_model.py on made-up records: the max gap, the features read no values, the seeded split,
and fit-and-test separating dense from sparse records while staying at chance on identical schedules. No network."""
import numpy as np
import pandas as pd

from scripts import timing_model as tm


def test_max_gap():
    t = np.array([0.0, 0.1, 0.2, 0.205, 0.21, 0.3])
    assert abs(tm.max_gap(t, 0.3, span=0.1) - 0.09) < 1e-9  # the step into the window from 0.21 to 0.3 counts
    assert np.isnan(tm.max_gap(np.array([0.1]), 0.2))


def records(dense, n=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for shot in range(n):
        disrupted = shot % 3 == 0
        step = 0.010 if (disrupted and dense) else 0.100
        t = np.concatenate([np.arange(0.0, 1.0, 0.1), np.arange(1.0, 1.5, step)])
        rows.append(pd.DataFrame({"shot": shot, "time": np.round(t, 4), "q95": rng.normal(3, 0.1, len(t)),
                                  "time_until_disrupt": (1.5 - t) if disrupted else np.full(len(t), np.nan)}))
    return pd.concat(rows)


def test_fit_and_test():
    table = tm.features(records(dense=True))
    assert set(table.columns) == {"shot", "disrupted", "occ", "gap", "maxgap"}
    split = tm.our_split(table)
    assert {v for v in split.values()} == {"train", "val", "test"}
    r, fixed, c, val_auc, counts = tm.fit_and_test(table, split)
    assert r[0] > 0.99 and fixed["occ"] > 0.99 and counts["test"][0] > 0
    same = tm.features(records(dense=False))
    r, fixed, *_ = tm.fit_and_test(same, tm.our_split(same))
    assert abs(r[0] - 0.5) < 1e-9 and tm.reading(r) == "not shown by this predictor"
