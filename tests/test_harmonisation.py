"""Tests for scripts/harmonisation.py on made-up records: grid points take the latest slice and respect the maximum
age, features read values only, and the whole analysis runs and reads as fixed on a schedule-only difference. No
network."""
import numpy as np
import pandas as pd

from scripts import harmonisation as hm, timing_model as tm


def test_grid_points_and_age():
    t = np.array([0.0, 0.1, 0.2, 0.21, 0.22, 0.5])
    v = np.arange(len(t), dtype=float)[:, None]
    pts, age = hm.grid_points(t, v, 0.5, 0.1, max_age=0.1, window=0.5)
    got = pts[:, 0]  # grid 0.1, 0.2, 0.3, 0.4, 0.5; at 0.4 the latest slice (0.22) is 180 ms old: empty
    assert list(got[[0, 1, 2, 4]]) == [1.0, 2.0, 4.0, 5.0] and np.isnan(got[3]) and abs(age[2] - 0.08) < 1e-9
    pts2, _ = hm.grid_points(t, v, 0.5, 0.05, max_age=0.05, window=0.3)
    assert np.isnan(pts2[2, 0]) and pts2[0, 0] == 4.0  # 0.35: latest slice 0.22 is older than 50 ms: empty


def test_feature_row():
    v = np.array([[1.0, np.nan], [2.0, np.nan], [4.0, np.nan]])
    assert hm.feature_row(v)[:3] == [4.0, 7 / 3, 3.0] and all(np.isnan(hm.feature_row(v)[3:]))


def records(n=240, seed=0):
    """Disrupted shots are sampled densely near the end; values are pure noise, so only the schedule differs."""
    rng = np.random.default_rng(seed)
    rows = []
    for shot in range(n):
        disrupted = shot % 3 == 0
        t = np.concatenate([np.arange(0.0, 1.0, 0.1), np.arange(1.0, 1.5, 0.01 if disrupted else 0.1)])
        rows.append(pd.DataFrame({"shot": shot, "time": np.round(t, 4), "q95": rng.normal(3, 0.3, len(t)),
                                  "li": rng.normal(1, 0.1, len(t)),
                                  "time_until_disrupt": (1.5 - t) if disrupted else np.full(len(t), np.nan)}))
    return pd.concat(rows)


def test_analysis_runs_and_reads():
    rows = hm.shot_rows(records(), ["q95", "li"], delta=0.1)
    assert len(rows) == 240 and len(rows["r1"].iloc[0]) == 6
    lines = hm.analyse(rows, tm.our_split(rows), "EAST")
    text = "\n".join(lines)
    assert "residual timing AUC on R2" in text and "PRIMARY Drop = AUC(R1) - AUC(R2)" in text
    # values are noise: every representation should be near chance except through the schedule (R1 sees step structure
    # only through 'last minus first', which noise makes uninformative), so no resolved large drop is required here;
    # the reading line must exist and be one of the three fixed readings
    assert any(k in text for k in ("did NOT remove", "sizeable part", "largely survives"))
