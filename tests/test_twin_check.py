"""Tests for scripts/twin_check.py on made-up programmes. No network."""
import numpy as np
import pandas as pd

from scripts import twin_check


def frame(peak, ramp_end=0.1, end=0.4):
    t = np.arange(0, end, 0.001)
    ref = np.where(t < ramp_end, peak * t / ramp_end, peak)
    return pd.DataFrame({"time": t, "ip_ref": ref})


def test_same_and_different_programmes():
    same = twin_check.compare(twin_check.programme(frame(800e3)), twin_check.programme(frame(805e3)))
    assert same["rms"] < 0.02 and same["peak_ratio"] > 0.99
    other = twin_check.compare(twin_check.programme(frame(800e3)), twin_check.programme(frame(500e3)))
    assert other["rms"] > 0.1
    assert twin_check.compare(None, twin_check.programme(frame(1))) is None


def test_check_marks_twins_with_the_same_programme():
    pairs = pd.DataFrame({"shot_id": [2, 4], "other": [1, 3], "relation": ["repeat"] * 2, "source": ["x"] * 2,
                          "group": ["candidate", "candidate"], "other_group": ["smooth_end", "smooth_end"],
                          "twin": [True, True]})
    labels = pd.DataFrame({"shot_id": [1, 2, 3, 4], "t_quench": [np.nan, 0.3, np.nan, 0.3]})
    shapes = {1: frame(800e3), 2: frame(802e3), 3: frame(800e3), 4: frame(400e3)}
    out = twin_check.check(pairs, labels, load=lambda s: shapes[s])
    assert out["same_programme"].tolist() == [True, False]
    assert "with the same programme: 1" in twin_check.text(out)


def labels_for(rows):
    """per_shot-like labels from (shot, note, termination, t_quench)."""
    return pd.DataFrame(rows, columns=["shot_id", "note", "termination", "t_quench"]).assign(campaign="M9")


def test_control_compares_unnamed_neighbours():
    pairs = pd.DataFrame({"shot_id": [20], "other": [19], "relation": ["repeat"], "source": ["x"],
                          "group": ["candidate"], "other_group": ["smooth_end"], "twin": [True]})
    labels = labels_for([(10, False, "smooth_end", np.nan), (19, False, "smooth_end", np.nan),
                         (20, False, "flat_top_drop", 0.3)])
    shapes = {10: frame(400e3), 19: frame(800e3), 20: frame(800e3)}
    out = twin_check.control(pairs, labels, load=lambda s: shapes.get(s))
    assert out[["shot_id", "other"]].values.tolist() == [[20, 10]]
    assert not out["same_programme"].iloc[0]


def test_programme_twins_find_the_nearest_kept_shot_with_the_same_programme():
    labels = labels_for([(1, False, "smooth_end", np.nan), (2, False, "smooth_end", np.nan),
                         (3, True, "flat_top_drop", 0.3), (4, False, "flat_top_drop", 0.3),
                         (5, False, "flat_top_drop", 0.3)])
    shapes = {1: frame(800e3), 2: frame(500e3), 3: frame(800e3), 4: frame(800e3), 5: frame(300e3)}
    out = twin_check.programme_twins(labels, load=lambda s: shapes.get(s))
    # 3 and 4 skip 2 (other programme) and the lost shot 3, and pair with 1; 5 has no same-programme kept shot
    assert out[["shot_id", "other", "group"]].values.tolist() == [[3, 1, "recorded"], [4, 1, "candidate"]]
    assert "Programme twins" in twin_check.text(twin_check.check(pd.DataFrame(
        {"shot_id": [3], "other": [1], "twin": [True]}), labels, load=lambda s: shapes.get(s)), None, out, labels)
